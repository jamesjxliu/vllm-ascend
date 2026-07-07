import gc
import logging
import os
import sys

import torch
import torch.distributed as dist

import vllm_ascend.envs as envs_ascend

logger = logging.getLogger(__name__)

_gva_is_inited: bool = False
_original_npu_mem_get_info = None


def is_zbal_enabled() -> bool:
    return envs_ascend.VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE > 0


def is_gva_inited() -> bool:
    """Return whether the zbal GVA heap has been bootstrapped.

    In standard mode, ``init_zbal`` sets this flag immediately.
    In mix-alloc mode, the flag is set later by ``lazy_init_zbal_gva_mem``
    so callers can decide whether to defer model execution.
    """
    return _gva_is_inited


def get_dist_backend() -> str:
    return "zbal" if is_zbal_enabled() else "hccl"


def init_zbal(
    world_size: int,
    gpu_id: int,
    world_rank: int,
    do_check: bool = True,
) -> int:
    """Initialize ZBAL early, before any NPU allocation.

    Mix-alloc: switches the allocator, defers zbal_bootstrap to
    :func:`lazy_init_zbal_gva_mem`.
    Standard: full zbal_init immediately.
    """
    zbal_mem_size = envs_ascend.VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE
    if not zbal_mem_size > 0:
        return 1

    # Force mix-alloc mode: weights & KV cache use DMA VMM, activations use
    # SMA/GVA. This avoids memory fragmentation and matches the validated
    # sglang path (see ZBAL_NPU_ALLOC_CONF in zbal docs). Without this,
    # ALL allocations go through the SMA allocator, which can return
    # non-standard memory layouts that trigger
    # "VEC instruction error: ub address out of bounds" in aclnn* kernels.
    # Must be set before importing zbal (zbal reads it at C++ init time).
    os.environ.setdefault("ZBAL_NPU_ALLOC_CONF", "use_vmm_for_static_memory:True")
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

    global _gva_is_inited, _original_npu_mem_get_info
    from zbal import is_mix_alloc, switch_to_allocator, zbal_init

    if is_mix_alloc():
        switch_to_allocator()
        _patch_memory_stats_for_mix_alloc()
        return 1

    bootstrap_url = envs_ascend.VLLM_ASCEND_ZBAL_BOOTSTRAP_URL
    if bootstrap_url:
        ret = zbal_init(
            world_size, gpu_id, world_rank,
            zbal_mem_size * (1024**2), ip_port=bootstrap_url,
        )
    else:
        ret = zbal_init(
            world_size, gpu_id, world_rank,
            zbal_mem_size * (1024**2),
        )

    _gva_is_inited = True

    if do_check and not ret:
        logger.error("[ZBAL] zbal init failed!")
        sys.exit(-1)

    _patch_npu_mem_get_info()
    return ret


def lazy_init_zbal_gva_mem(
    device: torch.device | str,
    gpu_id: int,
    world_rank: int,
    world_size: int,
    cpu_group: dist.ProcessGroup | None = None,
    do_check: bool = True,
) -> int:
    """Bootstrap zbal with GVA sized from remaining HBM (mix-alloc only).

    Must be called after KV cache allocation so GVA = pool − used.
    """
    from zbal import is_mix_alloc, zbal_init

    if not is_mix_alloc():
        logger.info("lazy init only for mix-alloc mode, skipping")
        return 1

    global _gva_is_inited

    total_memory_gb = 61.2
    free_gpu_memory_gb = _get_available_gpu_memory_gb(
        device, gpu_id,
        distributed=world_size > 1,
        cpu_group=cpu_group,
        empty_cache=True,
    )
    used_memory_gb = total_memory_gb - free_gpu_memory_gb
    gva_in_mb = envs_ascend.VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE - int(used_memory_gb * 1024)
    gva_in_mb = gva_in_mb - gva_in_mb % 128
    print(f"[ZBAL] rank {world_rank} GVA: {gva_in_mb} MB")

    assert not _gva_is_inited, "zbal gva already initialized"

    bootstrap_url = envs_ascend.VLLM_ASCEND_ZBAL_BOOTSTRAP_URL
    if bootstrap_url:
        res = zbal_init(
            world_size, gpu_id, world_rank,
            gva_in_mb * (1024**2), ip_port=bootstrap_url,
        )
    else:
        res = zbal_init(
            world_size, gpu_id, world_rank,
            gva_in_mb * (1024**2),
        )
    _gva_is_inited = True

    if do_check and not res:
        logger.error("[ZBAL] zbal lazy init failed!")
        sys.exit(-1)
    return res


def _get_available_gpu_memory_gb(
    device: torch.device | str,
    gpu_id: int,
    distributed: bool = False,
    cpu_group: dist.ProcessGroup | None = None,
    empty_cache: bool = True,
) -> float:
    """Return free NPU memory in GiB, optionally synced across ranks."""
    device = torch.device(device) if isinstance(device, str) else device
    if empty_cache:
        torch.npu.empty_cache()

    if is_zbal_enabled():
        from zbal import is_mix_alloc
        if not is_mix_alloc():
            free_bytes, _ = _zbal_mem_get_info()
        else:
            free_bytes, _ = torch.npu.mem_get_info()
    else:
        free_bytes, _ = torch.npu.mem_get_info()

    free_gb = free_bytes / (1024**3)

    if distributed and cpu_group is not None:
        free_mem_tensor = torch.tensor([free_gb], dtype=torch.float32)
        dist.all_reduce(free_mem_tensor, op=dist.ReduceOp.MIN, group=cpu_group)
        free_gb = free_mem_tensor.item()

    return free_gb


def _zbal_mem_get_info(device=None):
    import zbal
    return zbal.mem_get_info()


def _patch_npu_mem_get_info():
    """Replace torch.npu.mem_get_info with zbal's version (standard mode)."""
    global _original_npu_mem_get_info
    if _original_npu_mem_get_info is not None:
        return
    _original_npu_mem_get_info = torch.npu.mem_get_info
    torch.npu.mem_get_info = _zbal_mem_get_info
    logger.info("[ZBAL] patched mem_get_info -> zbal")


def _patch_memory_stats_for_mix_alloc():
    """Wrap memory_stats paths for mix-alloc.

    zbal mix allocator does not support ``get_device_stats``, so any call to
    ``torch_npu._C._npu_memoryStats`` raises RuntimeError. Patch all Python
    entry points on ``torch_npu.npu``, ``torch_npu.npu.memory``, and
    ``torch.accelerator`` (the entry vLLM's MemorySnapshot calls), plus the
    C-level ``_npu_memoryStats`` when assignable, so all wrappers return ``{}``
    on that error.
    """
    try:
        import torch_npu
    except ImportError:
        return

    import torch

    ERR_MARKER = "do not support get_device_stats"

    def _safe_call(fn, device):
        try:
            return fn(device)
        except RuntimeError as e:
            if ERR_MARKER in str(e):
                return {}
            raise

    def _make_patched(orig):
        if getattr(orig, "_zbal_patched", False):
            return None
        def _patched(device=None):
            return _safe_call(orig, device)
        _patched._zbal_patched = True
        return _patched

    patched = []
    targets = [
        ("torch_npu.npu", torch_npu.npu),
        ("torch_npu.npu.memory", torch_npu.npu.memory),
        ("torch.accelerator", torch.accelerator),
    ]
    for label, module in targets:
        for attr in ("memory_stats", "memory_stats_as_nested_dict"):
            if not hasattr(module, attr):
                continue
            new_fn = _make_patched(getattr(module, attr))
            if new_fn is None:
                continue
            try:
                setattr(module, attr, new_fn)
                patched.append(f"{label}.{attr}")
            except (AttributeError, TypeError):
                pass

    # Try to patch the C-level function too (may be read-only on some builds).
    try:
        c_orig = torch_npu._C._npu_memoryStats
        new_c = _make_patched(c_orig)
        if new_c is not None:
            torch_npu._C._npu_memoryStats = new_c
            patched.append("_C._npu_memoryStats")
    except (AttributeError, TypeError):
        pass

    if patched:
        logger.info("[ZBAL] patched memory_stats paths for mix-alloc: %s", ", ".join(patched))


def get_comm_name_from_group(
    device_group: dist.ProcessGroup,
    rank: int | None = None,
) -> str:
    """Return comm group name string for a device group (HCCL or ZBAL).

    *rank* is passed to HCCL's get_hccl_comm_name; ignored in ZBAL mode.
    """
    backend = device_group._get_backend(torch.device("npu"))
    if is_zbal_enabled():
        return backend.get_zbal_comm_name()
    if rank is None:
        rank = torch.distributed.get_rank(group=device_group)
    return backend.get_hccl_comm_name(rank)
