#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from __future__ import annotations

import logging
import os
import sys

import torch
import torch.distributed as dist

from vllm.logger import logger
import vllm_ascend.envs as envs_ascend


# Track whether GVA has been initialized (global per-process)
_gva_is_inited: bool = False


def is_zbal_enabled() -> bool:
    """Return True if ZBAL should be used for the current run."""
    return envs_ascend.VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE > 0


def get_zbal_backend() -> str:
    """Return the correct distributed backend string for NPU.

    Returns "zbal" when ZBAL is enabled, otherwise "hccl".
    """
    return "zbal" if is_zbal_enabled() else "hccl"


def init_zbal() -> int:
    """Phase 1: switch to SMA allocator and patch memory profiling.
    Returns 1 on success. Raises if not in mix-alloc mode.
    """
    if not is_zbal_enabled():
        return 0

    global _gva_is_inited
    from zbal import is_mix_alloc, switch_to_allocator

    if is_mix_alloc():
        switch_to_allocator()
        return 0

    raise RuntimeError(
        "ZBAL non-mix-alloc mode is not supported. Please set "
        "ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True to "
        "enable mix-alloc mode."
    )


def lazy_init_zbal_gva_mem(
    device: torch.device | str,
    gpu_id: int,
    world_rank: int,
    world_size: int,
    cpu_group: dist.ProcessGroup | None = None,
    do_check: bool = True,
) -> int:
    """Phase 2: call ``zbal_init`` to carve GVA heap from free HBM.

    Must be called AFTER weights and KV cache are loaded (they use DMA
    VMM). After this, activations use GVA via ``sma_malloc``.

    GVA size = VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE - used (weights + kv_cache),
    aligned to 128 MB.
    """
    from zbal import is_mix_alloc, zbal_init

    if not is_mix_alloc():
        logger.info("lazy init is supported only in mix alloc mode, skip.")
        return 0

    global _gva_is_inited
    if _gva_is_inited:
        logger.info("[ZBAL] skip lazy init because it was already called once")
        return 0
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
    gva_in_mb = gva_in_mb - gva_in_mb % 128  # align to 128 MB

    logger.info(
        "[ZBAL] rank %s GVA: %s MB (%.2f GB)",
        world_rank,
        gva_in_mb,
        gva_in_mb / 1024,
    )

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

    if do_check and not res:
        logger.error("[ZBAL] zbal lazy init failed!")
        return -1

    return res


def _get_available_gpu_memory_gb(
    device: torch.device | str,
    gpu_id: int,
    distributed: bool = False,
    cpu_group: dist.ProcessGroup | None = None,
    empty_cache: bool = True,
) -> float:
    """Return free NPU memory in GiB, optionally synced across ranks.

    Uses ``torch.npu.mem_get_info`` (device global view, not GVA heap)
    since GVA may not be initialized yet when this is called.
    """
    device = torch.device(device) if isinstance(device, str) else device

    if empty_cache:
        torch.npu.empty_cache()

    free_bytes, _ = torch.npu.mem_get_info()
    free_gb = free_bytes / (1024**3)

    if distributed and cpu_group is not None:
        import torch.distributed as dist

        free_mem_tensor = torch.tensor([free_gb], dtype=torch.float32)
        dist.all_reduce(free_mem_tensor, op=dist.ReduceOp.MIN, group=cpu_group)
        free_gb = free_mem_tensor.item()

    return free_gb


def get_comm_name_from_group(
    device_group: dist.ProcessGroup,
    rank: int | None = None,
) -> str:
    """Return the comm group name for HCCL or ZBAL backend.

    - HCCL: ``backend.get_hccl_comm_name(rank)`` (rank defaults to local
      rank within the group).
    - ZBAL: ``backend.get_zbal_comm_name()`` (rank argument ignored).
    """
    backend = device_group._get_backend(torch.device("npu"))
    if is_zbal_enabled():
        return backend.get_zbal_comm_name()

    if rank is None:
        rank = torch.distributed.get_rank(group=device_group)
    return backend.get_hccl_comm_name(rank)
