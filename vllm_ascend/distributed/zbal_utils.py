#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""ZBAL (Zero Buffer Accelerate Library) integration helpers.

This module wraps the ``zbal`` Python module to provide a unified set of
entry points for vllm-ascend. The integration mirrors the sglang reference
implementation (``sglang/srt/hardware_backend/npu/utils.py``) and supports
two modes:

* **standard mode**: ``zbal_init`` is called eagerly with the full pool size
  (``VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE`` MiB) as the GVA. The default NPU
  allocator is replaced by the zbal allocator during ``zbal_init``.
* **mix-alloc mode**: detected via ``zbal.is_mix_alloc()``. The allocator is
  switched early (before any NPU allocation, e.g. weight loading) via
  ``switch_to_allocator()``; the actual ``zbal_init`` (which bootstraps the
  GVA heap and communicator) is deferred to :func:`lazy_init_zbal_gva_mem`
  so that GVA can be sized from the remaining HBM after weights + KV cache.

Environment variables (defined in ``vllm_ascend/envs.py``):

* ``VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE`` (MiB, default 0): total zbal pool size.
  0 or unset disables zbal and the default HCCL backend is used.
* ``VLLM_ASCEND_ZBAL_BOOTSTRAP_URL`` (optional ``ip:port``): explicit
  rendezvous endpoint for multi-node zbal init.
"""

from __future__ import annotations

import gc
import sys
from typing import Any

import torch
import torch.distributed as dist

from vllm.logger import logger
import vllm_ascend.envs as envs_ascend


# Module-level state. Mirrors sglang's ``gva_is_inited``.
_gva_is_inited: bool = False
_patched_memory_stats: bool = False


def is_zbal_enabled() -> bool:
    """Return True if zbal is enabled via env var (pool size > 0)."""
    return envs_ascend.VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE > 0


def get_dist_backend() -> str:
    """Return the distributed backend string for torch.distributed.

    Returns ``"zbal"`` when zbal is enabled, otherwise ``"hccl"``.
    Mirrors sglang's ``_DEVICE_TO_DISTRIBUTED_BACKEND["npu"]`` logic.
    """
    return "zbal" if is_zbal_enabled() else "hccl"


def _import_zbal() -> Any:
    """Import the ``zbal`` module with a friendly error message."""
    try:
        import zbal  # type: ignore[import-not-found]
        return zbal
    except ImportError as e:
        raise ImportError(
            "VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE is set but the `zbal` package is "
            "not installed. Please install zbal or unset the env var to use "
            "the default HCCL backend."
        ) from e


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
        logger.info("[ZBAL] mix-alloc mode: switching allocator; zbal_init deferred.")
        switch_to_allocator()
        _patch_memory_stats_for_mix_alloc()
        return 1

    logger.info(
        "[ZBAL] standard mode: zbal_init immediately (world_size=%s, gpu_id=%s, "
        "rank=%s, pool=%d MiB)",
        world_size, gpu_id, world_rank, zbal_mem_size,
    )
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

    # Release cached blocks so the GVA math sees the real free memory.
    gc.collect()
    torch.npu.empty_cache()

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
    # torch.npu.mem_get_info() already returns total minus system reserved
    # (e.g. ~61.3 GB for a 64 GB device), so no additional subtraction needed.
    # sglang hardcodes 61.2 GB which achieves the same effect.
    used_mb = max(total_mb - free_mb, 0)
    gva_mb = pool_mb - used_mb
    # Align to 128 MiB (required by zbal, matches sglang).
    gva_mb = gva_mb - (gva_mb % 128)

    if gva_mb <= 0:
        logger.error(
            "[ZBAL] GVA size non-positive (%d MiB). pool=%d MiB, used=%d MiB, "
            "free=%d MiB, total=%d MiB.\n"
            "GVA = pool - used, so pool must be larger than used.\n"
            "Current vLLM usage (weights + KV cache + activations) = %d MiB.\n"
            "To fix this, either:\n"
            "  1. Increase VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE to > %d MiB (current "
            "pool is too small), OR\n"
            "  2. Lower --gpu-memory-utilization so vLLM uses less memory (e.g. "
            "reduce KV cache size), freeing up room for the zbal GVA heap.\n"
            "Example: if NPU total = %d MiB and you want %d MiB for zbal GVA, "
            "set VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE=%d and lower "
            "gpu_memory_utilization so vLLM uses at most %d MiB.",
            gva_mb, pool_mb, used_mb, free_mb, total_mb,
            used_mb,
            used_mb,
            total_mb,
            max(used_mb + 128, pool_mb),
            pool_mb - 128,
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


def is_gva_inited() -> bool:
    """Return whether zbal GVA has been bootstrapped (standard or mix-alloc)."""
    return _gva_is_inited


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

    vllm's NPUCommunicator calls ``get_hccl_comm_name``; for a zbal backend
    the underlying c10d ProcessGroup exposes ``get_zbal_comm_name`` instead.
    This helper picks the right accessor based on the backend string.
    """
    backend = device_group.options.backend.lower() if hasattr(device_group, "options") else "hccl"
    if backend == "zbal":
        # zbal ProcessGroup provides get_zbal_comm_name
        if rank is None:
            return device_group.get_zbal_comm_name()
        return device_group.get_zbal_comm_name(rank)
    # default hccl path
    if rank is None:
        return device_group.get_hccl_comm_name()
    return device_group.get_hccl_comm_name(rank)
