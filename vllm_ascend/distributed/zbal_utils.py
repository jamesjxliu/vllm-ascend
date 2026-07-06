"""ZBAL integration helpers.

ZBAL (Zero Buffer Accelerate Library) is an NPU-specific communication
acceleration library. It is enabled by setting ``VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE``
to a positive value (in MiB). When unset or 0, vllm-ascend falls back to the
standard HCCL backend and this module is a no-op.

Design contract with the caller
-------------------------------
* ``init_zbal`` runs early (before any NPU allocation) and performs either:
    - mix-alloc mode: only switches the allocator; ``zbal_init`` is deferred
      to :func:`lazy_init_zbal_gva_mem` so that GVA size can be sized from the
      HBM remaining after weights + KV cache allocation.
    - standard mode: calls ``zbal_init`` immediately.
* ``lazy_init_zbal_gva_mem`` is only called in mix-alloc mode, after KV cache
  allocation. It synchronises the GVA size across ranks via ``cpu_group``
  (all-reduce MIN), so every rank initialises the same GVA size - required
  by the underlying communicator.
"""

import logging
import sys
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

import vllm_ascend.envs as envs_ascend

if TYPE_CHECKING:
    import torch_npu  # noqa: F401

logger = logging.getLogger(__name__)

# Module-level mutable state. vllm-ascend assumes one worker per process,
# so these are safe to keep as module globals. They are guarded against
# double-initialisation.
_gva_is_inited: bool = False
_original_npu_mem_get_info = None
_patched_memory_stats: bool = False


def _import_zbal():
    """Import zbal with a friendly error message.

    Raises ``ImportError`` with a clear hint when zbal is requested but not
    installed, instead of letting the bare import error propagate.
    """
    try:
        import zbal  # noqa: F401
        return zbal
    except ImportError as e:
        raise ImportError(
            "VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE is set but the `zbal` package is "
            "not installed. Install zbal or unset the env var to use HCCL."
        ) from e


def is_zbal_enabled() -> bool:
    """Return True iff zbal is enabled via env config."""
    return envs_ascend.VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE > 0


def get_dist_backend() -> str:
    """Return the torch.distributed backend string to use.

    Returns ``"zbal"`` when zbal is enabled, ``"hccl"`` otherwise. HCCL
    callers remain completely unaffected when zbal is disabled.
    """
    return "zbal" if is_zbal_enabled() else "hccl"


def init_zbal(
    world_size: int,
    gpu_id: int,
    world_rank: int,
    do_check: bool = True,
) -> int:
    """Initialize ZBAL early, before any NPU allocation.

    The ``world_size`` / ``world_rank`` here refer to the **zbal communicator
    scope**. Callers should pass TP size / TP rank for standard mode (matching
    the validated sglang path), because zbal builds a single communicator that
    TP allreduce operates on. PP > 1 is only supported in mix-alloc mode.

    Mix-alloc: switches the allocator, defers ``zbal_bootstrap`` to
    :func:`lazy_init_zbal_gva_mem`.
    Standard: full ``zbal_init`` immediately.
    """
    zbal_mem_size = envs_ascend.VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE
    if not zbal_mem_size > 0:
        return 1

    global _gva_is_inited
    zbal = _import_zbal()
    from zbal import is_mix_alloc, switch_to_allocator, zbal_init

    if is_mix_alloc():
        # Only switch allocator; zbal_init is deferred to lazy_init_zbal_gva_mem
        # so GVA can be sized from remaining HBM after KV cache allocation.
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

    # NOTE: we intentionally do NOT globally patch torch.npu.mem_get_info.
    # zbal's mem_get_info has different semantics (returns pool stats), and
    # vllm core code (MemorySnapshot, sleep/wake accounting) expects the
    # native NPU view. Callers that need zbal's view should use
    # `_zbal_mem_get_info` explicitly.
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

    Must be called after KV cache allocation so GVA = pool - used.

    Memory source: in mix-alloc mode the zbal heap is NOT yet initialised
    when this function runs (zbal_init is what bootstraps it), so we MUST
    read the native ``torch.npu.mem_get_info()`` rather than
    ``zbal.mem_get_info()``. The latter would return ``(free>0, total=0)``
    for an un-initialised heap, producing a negative ``used`` and an
    absurdly large GVA that fails SHM allocation. (Matches sglang's
    ``get_available_gpu_memory`` NPU branch: "mix mode fall back into npu
    mem info since gva may not be inited yet".)

    Cross-rank sync: zbal requires every rank to initialise the **same** GVA
    size. When ``cpu_group`` is provided and ``world_size > 1``, we
    all-reduce(MIN) free across ranks so uneven per-rank usage (e.g.
    embedding/lm_head only on rank 0) does not produce divergent GVA sizes.
    """
    from zbal import is_mix_alloc, zbal_init

    if not is_mix_alloc():
        logger.debug(
            "lazy_init_zbal_gva_mem is a no-op outside mix-alloc mode; skipping."
        )
        return 1

    global _gva_is_inited
    assert not _gva_is_inited, "zbal gva already initialized"

    # CRITICAL: use native NPU stats, NOT zbal.mem_get_info(). In mix-alloc
    # mode the zbal heap is not initialised yet, so zbal.mem_get_info() would
    # return inconsistent values (free>0, total=0) and break the GVA math.
    free_bytes, total_bytes = torch.npu.mem_get_info()

    # Sync free across ranks: GVA must be identical on every rank.
    if cpu_group is not None and world_size > 1:
        stats = torch.tensor([free_bytes, total_bytes], dtype=torch.int64)
        dist.all_reduce(stats, op=dist.ReduceOp.MIN, group=cpu_group)
        free_bytes, total_bytes = int(stats[0].item()), int(stats[1].item())

    pool_mb = envs_ascend.VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE
    free_mb = free_bytes // (1024**2)
    total_mb = total_bytes // (1024**2)
    used_mb = max(total_mb - free_mb, 0)
    gva_mb = pool_mb - used_mb
    # Align to 128 MiB (required by zbal, matches sglang).
    gva_mb = gva_mb - (gva_mb % 128)

    if gva_mb <= 0:
        logger.error(
            "[ZBAL] GVA size non-positive (%d MiB). pool=%d MiB, used=%d MiB, "
            "free=%d MiB, total=%d MiB. Reduce VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE "
            "or lower gpu_memory_utilization.",
            gva_mb, pool_mb, used_mb, free_mb, total_mb,
        )
        if do_check:
            sys.exit(-1)
        return 0

    logger.info(
        "[ZBAL] rank %s GVA: %d MiB (pool=%d MiB, used=%d MiB, free=%d MiB, "
        "total=%d MiB)",
        world_rank, gva_mb, pool_mb, used_mb, free_mb, total_mb,
    )

    gva_bytes = gva_mb * (1024**2)
    bootstrap_url = envs_ascend.VLLM_ASCEND_ZBAL_BOOTSTRAP_URL
    if bootstrap_url:
        res = zbal_init(world_size, gpu_id, world_rank, gva_bytes, ip_port=bootstrap_url)
    else:
        res = zbal_init(world_size, gpu_id, world_rank, gva_bytes)
    _gva_is_inited = True

    if do_check and not res:
        logger.error("[ZBAL] zbal lazy init failed!")
        sys.exit(-1)
    return res


def _zbal_mem_get_info(device=None):
    """Wrapper around zbal.mem_get_info().

    zbal does not support per-device queries (it operates on the current
    device's pool). We assert the caller is not asking for a foreign device,
    to avoid silently returning wrong-device stats.
    """
    zbal = _import_zbal()
    if device is not None:
        d = torch.device(device) if isinstance(device, str) else device
        if d.type == "npu" and d.index is not None:
            cur = torch.npu.current_device()
            assert d.index == cur, (
                f"zbal.mem_get_info only operates on the current NPU device "
                f"(index {cur}), got request for index {d.index}."
            )
    return zbal.mem_get_info()


def _patch_npu_mem_get_info():
    """Deprecated: no-op. Kept only to avoid breaking external callers.

    We no longer globally replace torch.npu.mem_get_info because zbal's view
    (pool stats) is incompatible with vllm's core memory accounting
    (MemorySnapshot, sleep/wake). Use :func:`_zbal_mem_get_info` directly
    when zbal's view is needed.
    """
    logger.debug(
        "[ZBAL] _patch_npu_mem_get_info is a no-op; callers should use "
        "_zbal_mem_get_info explicitly when zbal's view is needed."
    )


def _patch_memory_stats_for_mix_alloc():
    """Patch _npu_memoryStats at C level for mix-alloc.

    In mix-alloc mode the underlying get_device_stats call raises
    "do not support get_device_stats". All Python memory APIs
    (memory_stats, memory_reserved, max_memory_allocated, etc.) eventually
    call this single C function. We wrap it to return an empty dict on that
    specific error so vllm's memory_stats consumers do not crash.

    Idempotent: safe to call multiple times.
    """
    global _patched_memory_stats
    if _patched_memory_stats:
        return
    try:
        import torch_npu
        import torch_npu._C  # noqa: F401
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
    _patched_memory_stats = True
    logger.info("[ZBAL] patched _npu_memoryStats for mix-alloc")


def get_comm_name_from_group(
    device_group: dist.ProcessGroup,
    rank: int | None = None,
) -> str:
    """Return comm group name string for a device group (HCCL or ZBAL).

    *rank* is passed to HCCL's get_hccl_comm_name; ignored in ZBAL mode.

    When zbal is enabled but the backend does not expose ``get_zbal_comm_name``
    (e.g. a stub for testing), we fall back to raising a clear error rather
    than crashing with ``AttributeError`` on the hot path.
    """
    backend = device_group._get_backend(torch.device("npu"))
    if is_zbal_enabled():
        getter = getattr(backend, "get_zbal_comm_name", None)
        if getter is None:
            raise RuntimeError(
                "zbal is enabled but the registered ProcessGroup backend does "
                "not expose get_zbal_comm_name(); ensure zbal registers a "
                "Backend with this method."
            )
        return getter()
    if rank is None:
        rank = torch.distributed.get_rank(group=device_group)
    return backend.get_hccl_comm_name(rank)
