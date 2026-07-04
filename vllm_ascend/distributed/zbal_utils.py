import logging
import sys

import torch
import torch.distributed as dist

import vllm_ascend.envs as envs_ascend

logger = logging.getLogger(__name__)

_gva_is_inited: bool = False
_original_npu_mem_get_info = None


def is_zbal_enabled() -> bool:
    return envs_ascend.VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE > 0


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

    global _gva_is_inited, _original_npu_mem_get_info
    from zbal import is_mix_alloc, zbal_init

    if is_mix_alloc():
        # Keep native allocator — zbal SMA returns misaligned memory
        # for AICORE kernels on this CANN/torch_npu version.
        # Only bootstrap fabric in lazy_init_zbal_gva_mem.
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
    from zbal import is_mix_alloc

    if not is_mix_alloc():
        return 1

    global _gva_is_inited
    assert not _gva_is_inited, "zbal gva already initialized"

    free_bytes, total_bytes = torch.npu.mem_get_info()
    pool_bytes = envs_ascend.VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE * 1024**2
    gva_bytes = max(pool_bytes - (total_bytes - free_bytes), 128 * 1024**2)
    gva_bytes = gva_bytes - (gva_bytes % 0x200000)
    logger.info("[ZBAL] rank %s GVA: %d MiB", world_rank, gva_bytes // (1024**2))

    import os
    from zbal.zbal import ZBALBootstrapOption, ZBALBootstrapType, zbal_bootstrap

    if "MEMFABRIC_HYBRID_LIBRARY_PATH" not in os.environ:
        import memfabric_hybrid as mf
        lib_path = mf.get_lib_path()
        if lib_path:
            os.environ["MEMFABRIC_HYBRID_LIBRARY_PATH"] = lib_path

    opt = ZBALBootstrapOption()
    opt.btType = ZBALBootstrapType.BOOT_BY_MEMFABRIC
    opt.worldSize = world_size
    opt.rankId = world_rank
    opt.deviceId = gpu_id
    opt.deviceMemorySize = gva_bytes
    opt.commMetaSpaceSize = 1024
    opt.commGroupCap = 64
    opt.ipPort = envs_ascend.VLLM_ASCEND_ZBAL_BOOTSTRAP_URL or "tcp://127.0.0.1:6789"

    ret = zbal_bootstrap(opt)
    _gva_is_inited = True
    if do_check and ret != 0:
        logger.error("[ZBAL] zbal bootstrap failed!")
        sys.exit(-1)
    return 0 if ret != 0 else 1


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
    """Patch _npu_memoryStats at C level for mix-alloc
    (get_device_stats not supported).  All Python memory APIs
    (memory_stats, memory_reserved, max_memory_allocated, etc.)
    eventually call this single C function."""
    try:
        import torch_npu._C
    except ImportError:
        return

    _orig = torch_npu._C._npu_memoryStats

    def _safe(device=None):
        try:
            return _orig(device)
        except RuntimeError as e:
            if "do not support get_device_stats" in str(e):
                return {}
            raise

    torch_npu._C._npu_memoryStats = _safe
    logger.info("[ZBAL] patched _npu_memoryStats for mix-alloc")


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
