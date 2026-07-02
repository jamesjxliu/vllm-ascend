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
import os
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

    Both mix-alloc and standard modes call ``zbal_init`` immediately.
    This is required because ``init_process_group("zbal")`` (which runs
    later in ``_init_worker_distributed_environment``) depends on the
    SMA allocator and GVA heap being fully initialized. Deferring
    ``zbal_init`` to after ``init_process_group`` leaves the zbal
    communicator in an inconsistent state and causes "ub address out of
    bounds" errors (aclnn* error 507015) during inference.

    In mix-alloc mode, the GVA heap size is computed dynamically as
    ``GVA = pool - used``, where ``pool = VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE``
    and ``used = total_device_memory - free_device_memory``. This ensures
    the GVA heap only consumes remaining free HBM, leaving room for
    weights and KV cache (which use DMA VMM).

    In standard (non-mix) mode, ``VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE`` is
    used directly as the zbal-managed memory size.

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

    # Enable mix-alloc mode: weights & KV cache use DMA VMM, activations
    # use SMA/GVA. Must be set before importing zbal (zbal reads it at
    # C++ init time). Both ZBAL_NPU_ALLOC_CONF and PYTORCH_NPU_ALLOC_CONF
    # are required for mix-alloc (matching v0.18.0_zbal and sglang paths).
    os.environ.setdefault("ZBAL_NPU_ALLOC_CONF", "use_vmm_for_static_memory:True")
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

    global _gva_is_inited, _original_npu_mem_get_info
    from zbal import is_mix_alloc, switch_to_allocator, zbal_init

    if is_mix_alloc():
        switch_to_allocator()
        _patch_memory_stats_for_mix_alloc()
        # In mix-alloc mode, defer zbal_init to lazy_init_zbal_gva_mem
        # (after weights are loaded). This ensures weights use DMA VMM
        # (gGVASpaceInited=false → dma_malloc, low address 0x12c...),
        # while activations use GVA (gGVASpaceInited=true → sma_malloc,
        # high address 0x280...). This separation is the mix-alloc design
        # intent and is required because some operators (e.g.
        # npu_quant_matmul) do not support GVA addresses for weights.
        # ProcessGroupZBAL supports delayed initCommunicator: if
        # zbal_bootstrap has not run, PrepareCommunicator skips
        # initCommunicator and returns Z_OK; collective operations
        # lazily call initCommunicator after zbal_bootstrap completes.
        return 1


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
    # Use print (not logger) so the GVA size is always visible even if the
    # logger has not been configured yet at this point in startup.
    print(
        f"[ZBAL] rank {world_rank} GVA: {gva_in_mb} MB "
        f"({gva_in_mb / 1024:.2f} GB)",
        flush=True,
    )
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

    # Print GVA heap address range for debugging "ub address out of bounds".
    # GVA addresses live in a high virtual address range (~40 TB). If a
    # tensor's data_ptr() falls outside this range, aclnn* kernels may
    # trigger VEC instruction errors when accessing UB.
    try:
        import zbal as _zbal_mod
        if hasattr(_zbal_mod, "mem_get_info"):
            _gva_free, _gva_total = _zbal_mod.mem_get_info()
            print(
                f"[ZBAL] rank {world_rank} GVA heap: "
                f"free={_gva_free / (1024**3):.2f} GB "
                f"total={_gva_total / (1024**3):.2f} GB",
                flush=True,
            )
    except Exception as _e:
        print(f"[ZBAL] could not query GVA heap info: {_e}", flush=True)

    _gva_is_inited = True

    # NOTE: Do NOT call _patch_npu_mem_get_info() here in mix-alloc mode.
    # In mix-alloc, zbal's mem_get_info reports the GVA heap view, not the
    # full device memory. Patching torch.npu.mem_get_info here would cause
    # downstream code (memory profiling, sleep mode, etc.) to see incorrect
    # values, which can lead to memory corruption and "ub address out of
    # bounds" errors in aclnn* kernels. The validated v0.18.0_zbal path
    # does not patch mem_get_info in mix-alloc mode.
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


def _patch_memory_stats_for_mix_alloc():
    """Patch upper-layer memory profiling callers for mix-alloc.

    In mix-alloc mode, zbal's allocator does not support
    ``get_device_stats``. vLLM v0.20.2rc2 has multiple callers that
    eventually reach ``torch_npu._C._npu_memoryStats`` (which raises
    RuntimeError in mix-alloc):

    1. ``MemorySnapshot.measure()`` (called in ``_init_device`` and
       ``memory_profiling``) calls ``torch.accelerator.memory_reserved``
       → ``torch_npu.npu.memory.memory_reserved`` → ``memory_stats``
       → ``_npu_memoryStats``.
    2. ``DeviceMemoryProfiler.current_memory_usage()`` (called in
       ``load_model``) calls ``NPUPlatform.get_current_memory_usage``
       → ``torch.npu.max_memory_allocated`` → ``memory_stats``
       → ``_npu_memoryStats``.

    The intermediate Python functions hold direct references to each
    other at definition time, so patching module attributes cannot
    intercept the chain. Patching the C-level ``_npu_memoryStats`` was
    tried (both unconditional ``{}`` and lazy fallback), but both broke
    the SMA allocator's internal state tracking and caused
    "VEC instruction error: ub address out of bounds" in aclnn* kernels
    during inference — the SMA allocator likely calls
    ``_npu_memoryStats`` internally.

    Solution: patch the upper-layer callers directly so they never reach
    ``_npu_memoryStats`` in mix-alloc mode:
    - ``MemorySnapshot.measure``: use ``torch.npu.mem_get_info`` for
      free/total memory; skip torch memory queries.
    - ``NPUPlatform.get_current_memory_usage``: return 0.0 directly.
    - ``torch_npu.npu.memory.memory_stats_as_nested_dict``: return {}
      (matches v0.18.0_zbal, for any other caller).

    This keeps the C-level ``_npu_memoryStats`` untouched, so the SMA
    allocator's internal state tracking is not affected.
    """
    # Patch 1: MemorySnapshot.measure — skip torch memory queries.
    try:
        from vllm.utils.mem_utils import MemorySnapshot

        def _measure_patched(self):
            free_bytes, total_bytes = torch.npu.mem_get_info()
            self.free_memory = free_bytes
            self.total_memory = total_bytes
            # torch_memory / torch_peak are unavailable in mix-alloc mode.
            self.torch_memory = 0
            self.torch_peak = 0
            # Other optional fields default to 0 if not already set.
            for attr in ("weights_memory", "non_torch_memory"):
                if not hasattr(self, attr):
                    setattr(self, attr, 0)

        MemorySnapshot.measure = _measure_patched
        logger.info("[ZBAL] patched MemorySnapshot.measure for mix-alloc")
    except ImportError:
        pass

    # Patch 2: NPUPlatform.get_current_memory_usage — return 0.0.
    try:
        from vllm_ascend.platform import NPUPlatform

        @classmethod
        def _get_current_memory_usage_patched(cls, device=None):
            return 0.0

        NPUPlatform.get_current_memory_usage = _get_current_memory_usage_patched
        logger.info("[ZBAL] patched NPUPlatform.get_current_memory_usage for mix-alloc")
    except ImportError:
        pass

    # Patch 3: memory_stats_as_nested_dict — return {} (matches v0.18.0_zbal).
    try:
        import torch_npu

        def _stats_patched(device=None):
            return {}

        torch_npu.npu.memory.memory_stats_as_nested_dict = _stats_patched
        logger.info("[ZBAL] patched memory_stats_as_nested_dict for mix-alloc")
    except ImportError:
        pass


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
