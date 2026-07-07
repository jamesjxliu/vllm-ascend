#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

"""ZBAL (Zero Buffer Accelerate Library) integration utilities.

ZBAL is an NPU-specific communication acceleration library based on
memfabric for unified HBM memory pooling and AIV-driven MTE operator
acceleration. It replaces the standard HCCL backend for collective
communication and provides accelerated DeepEP dispatch/combine
operators.

Usage::

    from vllm_ascend.distributed.zbal_utils import (
        is_zbal_enabled,
        init_zbal,
        lazy_init_zbal_gva_mem,
    )

    # Phase 1: Early init before any torch allocation
    if is_zbal_enabled():
        init_zbal(world_size, gpu_id, rank)

    # Phase 2: Lazy GVA memory registration after model loading
    if is_zbal_enabled():
        lazy_init_zbal_gva_mem(device, gpu_id, rank, world_size, cpu_group)
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import torch

import vllm_ascend.envs as envs_ascend

if TYPE_CHECKING:
    import torch.distributed as dist

logger = logging.getLogger(__name__)

# Track whether GVA has been initialized (global per-process)
_gva_is_inited: bool = False
# Save original torch.npu.mem_get_info for restoration
_original_npu_mem_get_info = None


def is_zbal_enabled() -> bool:
    """Return True if ZBAL should be used for the current run."""
    return envs_ascend.VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE > 0


def get_zbal_backend() -> str:
    """Return the correct distributed backend string for NPU.

    Returns "zbal" when ZBAL is enabled, otherwise "hccl".
    """
    return "zbal" if is_zbal_enabled() else "hccl"


def init_zbal(
    world_size: int,
    gpu_id: int,
    world_rank: int,
    do_check: bool = True,
) -> int:
    """Initialize ZBAL early (before any torch allocation).

    In mix-alloc mode this only switches the allocator; the actual
    ``zbal_init`` call is deferred to :func:`lazy_init_zbal_gva_mem`.
    In standard mode this performs full initialization immediately.

    Args:
        world_size: Total number of ranks.
        gpu_id: Local NPU device index.
        world_rank: Global rank of the current process.
        do_check: If True, exit the process on initialization failure.

    Returns:
        ``1`` on success, or raises/returns non-zero on failure.
    """
    zbal_mem_size = envs_ascend.VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE
    if not zbal_mem_size > 0:
        return 1

    global _gva_is_inited, _original_npu_mem_get_info
    from zbal import is_mix_alloc, switch_to_allocator, zbal_init

    if is_mix_alloc():
        switch_to_allocator()
        # use lazy init for mix alloc
        return 1

    # Standard (non-mix) mode: full zbal initialization
    bootstrap_url = envs_ascend.VLLM_ASCEND_ZBAL_BOOTSTRAP_URL
    if bootstrap_url:
        ret = zbal_init(
            world_size,
            gpu_id,
            world_rank,
            zbal_mem_size * (1024**2),
            ip_port=bootstrap_url,
        )
    else:
        ret = zbal_init(
            world_size,
            gpu_id,
            world_rank,
            zbal_mem_size * (1024**2),
        )

    _gva_is_inited = True

    if do_check and not ret:
        logger.error("[ZBAL] zbal init failed!")
        sys.exit(-1)

    # In non-mix mode, zbal takes full control of HBM. Patch
    # torch.npu.mem_get_info so memory profiling (MemorySnapshot,
    # sleep mode, etc.) reports zbal-managed memory instead of raw
    # NPU driver values.
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
    """Lazy-initialize ZBAL GVA memory for mix-alloc mode.

    This should be called **after** model weights are loaded but **before**
    CUDA graph capture, so that GVA memory is carved from the remaining
    free HBM space. Only meaningful when ``zbal.is_mix_alloc()`` is True.

    Args:
        device: The torch device (e.g. ``torch.device("npu")``).
        gpu_id: Local NPU device index.
        world_rank: Global rank of the current process.
        world_size: Total number of ranks.
        cpu_group: CPU-side process group for ``all_gather`` memory stats
                   (used to detect unbalanced OS memory across ranks).
        do_check: If True, exit the process on initialization failure.

    Returns:
        ``1`` on success, or raises/returns non-zero on failure.
    """
    from zbal import is_mix_alloc, zbal_init

    if not is_mix_alloc():
        logger.info(
            "lazy init is supported only in mix alloc mode, "
            "this action will be passed"
        )
        return 1

    global _gva_is_inited

    # Compute free GPU memory and subtract used portion from the total
    # zbal allocation budget so GVA only consumes remaining free HBM.
    total_memory_gb = 61.2  # reserve ~2.5 GB for workspace & OS outside torch
    free_gpu_memory_gb = _get_available_gpu_memory_gb(
        device,
        gpu_id,
        distributed=world_size > 1,
        cpu_group=cpu_group,
        empty_cache=True,
    )

    used_memory_gb = total_memory_gb - free_gpu_memory_gb
    used_memory_in_mb = int(used_memory_gb * 1024)
    gva_in_mb = envs_ascend.VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE - used_memory_in_mb
    # Align to 128 MB boundary
    gva_in_mb = gva_in_mb - gva_in_mb % 128
    logger.info(
        "[ZBAL] rank %s allocated %s MB gva space.",
        world_rank,
        gva_in_mb,
    )

    assert not _gva_is_inited, "zbal gva should be inited only once"

    bootstrap_url = envs_ascend.VLLM_ASCEND_ZBAL_BOOTSTRAP_URL
    if bootstrap_url:
        res = zbal_init(
            world_size,
            gpu_id,
            world_rank,
            gva_in_mb * (1024**2),
            ip_port=bootstrap_url,
        )
    else:
        res = zbal_init(
            world_size,
            gpu_id,
            world_rank,
            gva_in_mb * (1024**2),
        )

    _gva_is_inited = True

    # In mix mode, zbal_init has now been called so zbal controls HBM.
    # Patch mem_get_info so downstream memory profiling sees zbal's view.
    _patch_npu_mem_get_info()

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
    """Return free NPU memory in GiB, optionally synchronised across ranks.

    When ZBAL is active (non-mix mode), the query goes through zbal's
    heap stats rather than ``torch.npu.mem_get_info``.

    Args:
        device: The torch device.
        gpu_id: Local NPU device index.
        distributed: If True, synchronise free memory across ``cpu_group``.
        cpu_group: Process group for cross-rank sync.
        empty_cache: If True, call ``torch.npu.empty_cache()`` before query.

    Returns:
        Free memory in GiB.
    """
    device = torch.device(device) if isinstance(device, str) else device

    if empty_cache:
        torch.npu.empty_cache()

    if is_zbal_enabled():
        from zbal import is_mix_alloc

        if not is_mix_alloc():
            # Standard mode: use zbal's heap stats
            # mem_get_info returns (free_bytes, total_bytes)
            free_bytes, _ = _zbal_mem_get_info()
        else:
            # Mix mode: GVA may not be inited yet, fall back to npu query
            free_bytes, _ = torch.npu.mem_get_info()
    else:
        free_bytes, _ = torch.npu.mem_get_info()

    free_gb = free_bytes / (1024**3)

    if distributed and cpu_group is not None:
        import torch.distributed as dist

        free_mem_tensor = torch.tensor([free_gb], dtype=torch.float32)
        dist.all_reduce(free_mem_tensor, op=dist.ReduceOp.MIN, group=cpu_group)
        free_gb = free_mem_tensor.item()

    return free_gb


def _zbal_mem_get_info():
    """Proxy for zbal's memory query.

    Returns ``(free_bytes, total_bytes)``.
    """
    import zbal

    return zbal.mem_get_info()


def _patch_npu_mem_get_info():
    """Replace ``torch.npu.mem_get_info`` with zbal's version.

    After zbal takes control of HBM (via ``zbal_init`` in non-mix mode or
    ``lazy_init_zbal_gva_mem`` in mix mode), the native NPU driver query
    may report incorrect free/total values.  This patch ensures that
    ``MemorySnapshot``, sleep-mode memory tracking, and any other
    downstream code see zbal-managed memory.
    """
    global _original_npu_mem_get_info
    if _original_npu_mem_get_info is not None:
        return  # already patched

    _original_npu_mem_get_info = torch.npu.mem_get_info
    torch.npu.mem_get_info = _zbal_mem_get_info
    logger.info("[ZBAL] Patched torch.npu.mem_get_info -> zbal.mem_get_info")


def _unpatch_npu_mem_get_info():
    """Restore the original ``torch.npu.mem_get_info``."""
    global _original_npu_mem_get_info
    if _original_npu_mem_get_info is not None:
        torch.npu.mem_get_info = _original_npu_mem_get_info
        _original_npu_mem_get_info = None
        logger.info("[ZBAL] Restored torch.npu.mem_get_info")


def get_comm_name_from_group(
    device_group: dist.ProcessGroup,
    rank: int | None = None,
) -> str:
    """Return the communication group name string for a device group.

    This abstracts the difference between HCCL and ZBAL backends:

    - HCCL: ``backend.get_hccl_comm_name(rank)`` where *rank* defaults to
      the calling process's rank within *device_group*.
    - ZBAL: ``backend.get_zbal_comm_name()`` (no rank argument needed).

    The optional *rank* parameter allows callers to pass an explicit
    ``global_rank`` or ``tp_rank`` when the HCCL backend requires it
    (e.g. ``MatmulAllreduceRowParallelOp``).  It is **ignored** in ZBAL
    mode.

    Args:
        device_group: A ``torch.distributed.ProcessGroup`` backed by either
                      ``"hccl"`` or ``"zbal"``.
        rank: Optional rank hint for ``get_hccl_comm_name``.
              If ``None``, the local rank within the group is used.

    Returns:
        The communication group name string that can be passed to NPU
        custom operators (e.g. dispatch/combine kernels, all-reduce fusion).
    """
    backend = device_group._get_backend(torch.device("npu"))
    if is_zbal_enabled():
        return backend.get_zbal_comm_name()

    if rank is None:
        rank = torch.distributed.get_rank(group=device_group)
    return backend.get_hccl_comm_name(rank)
