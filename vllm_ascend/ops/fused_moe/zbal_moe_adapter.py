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

"""ZBAL MoE adapter.

This module provides an adapter that wraps ZBAL Buffer's dispatch and combine
interfaces, offering DeepEP-like functionality for high-throughput intranode
all-to-all communication on Ascend NPUs.

All ``zbal`` imports are deferred to method-level (lazy imports) to avoid
import failures in non-NPU environments, consistent with the existing
``zbal_utils.py`` pattern.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist

import vllm_ascend.envs as envs_ascend
from vllm_ascend.distributed.zbal_utils import is_zbal_enabled

logger = logging.getLogger(__name__)

# GVA base address threshold (0 means unknown — set after zbal_init).
# Tensors with data_ptr below this are in DMA VMM; above are in GVA.
_GVA_BASE_ADDR: int = 0


def _set_gva_base_addr():
    """Try to query GVA base address from zbal runtime."""
    global _GVA_BASE_ADDR
    if _GVA_BASE_ADDR != 0:
        return
    try:
        from zbal import get_gva_base_addr
        _GVA_BASE_ADDR = get_gva_base_addr()
        logger.info("[ZBAL] GVA base addr: 0x%x", _GVA_BASE_ADDR)
    except Exception:
        # zbal might not expose this API; threshold comparison will be skipped.
        pass


def _log_tensor_addrs(tag: str, *tensors):
    """Log data_ptr of each tensor to diagnose GVA vs DMA VMM placement."""
    _set_gva_base_addr()
    parts = []
    for i, t in enumerate(tensors):
        if t is None:
            continue
        ptr = t.data_ptr()
        region = "?"
        if _GVA_BASE_ADDR != 0:
            region = "GVA" if ptr >= _GVA_BASE_ADDR else "DMA_VMM"
        parts.append(
            f"t{i}: ptr=0x{ptr:x} region={region} "
            f"shape={list(t.shape)} dtype={t.dtype} "
            f"nbytes={t.numel() * t.element_size()}"
        )
    if parts:
        logger.warning(
            "[ZBALAddrCheck] %s rank=%s %s",
            tag,
            dist.get_rank() if dist.is_initialized() else -1,
            " | ".join(parts),
        )


class ZBALMoEAdapter:
    """Adapter for ZBAL Buffer's dispatch/combine interfaces.

    Supports standard dispatch/combine, low-latency dispatch/combine,
    and fused deep MoE operations.
    """

    def __init__(
        self,
        group: "dist.ProcessGroup",
        num_experts: int,
        hidden_size: int,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
    ):
        """Initialize the ZBAL MoE adapter.

        Args:
            group: Distributed communication group.
            num_experts: Total number of experts.
            hidden_size: Hidden dimension size.
            num_nvl_bytes: Buffer size for intranode HCCS communication.
            num_rdma_bytes: Buffer size for internode RDMA communication.
            low_latency_mode: Whether to enable low-latency mode.
        """
        if not is_zbal_enabled():
            raise RuntimeError(
                "ZBAL is not enabled. Please set VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE > 0"
            )

        self.group = group
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.low_latency_mode = low_latency_mode

        self.rank = group.rank()
        self.group_size = group.size()

        logger.info(
            "[ZBALMoEAdapter] Initializing adapter with rank=%s, "
            "group_size=%s, num_experts=%s, hidden_size=%s, low_latency_mode=%s",
            self.rank, self.group_size, num_experts, hidden_size, low_latency_mode,
        )

        # Lazy import: zbal is only available on NPU environments.
        from zbal.zbal_buffer import Buffer

        self.buffer = Buffer(
            group=group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=low_latency_mode,
        )

        self.dispatch_config = Buffer.get_dispatch_config(self.group_size)
        self.combine_config = Buffer.get_combine_config(self.group_size)

        logger.info("[ZBALMoEAdapter] Adapter initialized successfully")

    # ------------------------------------------------------------------
    # Standard dispatch / combine
    # ------------------------------------------------------------------

    def dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config: Optional[Any] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Dispatch tokens to different ranks using ZBAL Buffer.

        Args:
            x: Input tokens, shape ``[num_tokens, hidden]``, dtype ``bfloat16``.
            topk_idx: Expert indices, shape ``[num_tokens, num_topk]``, dtype ``int64``.
            topk_weights: Expert weights, shape ``[num_tokens, num_topk]``, dtype ``float``.
            expert_alignment: Align received tokens per expert to this value.
            num_worst_tokens: Worst-case token count (for NPU-graph compatibility).
            config: Custom performance tuning config. Uses default if ``None``.
            async_finish: If ``True``, current stream will not wait for completion.
            allocate_on_comm_stream: If ``True``, allocate tensors on comm stream.

        Returns:
            recv_x: Received tokens, shape ``[received_token_count, hidden]``.
            recv_topk_idx: Received expert indices.
            handle_dict: Communication handle and related info for combine.
        """
        logger.debug(
            "[ZBALMoEAdapter] Dispatch started: x.shape=%s, topk_idx.shape=%s",
            x.shape, topk_idx.shape,
        )

        config = config or self.dispatch_config

        # Step 1: Calculate dispatch layout.
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            layout_event,
        ) = self.buffer.get_dispatch_layout(
            topk_idx=topk_idx,
            num_experts=self.num_experts,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Print tensor addresses before dispatch to diagnose GVA/DMA memory issues.
        _log_tensor_addrs("dispatch_input", x, topk_idx, topk_weights)

        # Step 2: Execute dispatch (pass topk_weights so zbal can forward them).
        # ZBAL buffer.dispatch returns 6 values:
        # (recv_x, recv_topk_idx, recv_topk_weights,
        #  num_recv_tokens_per_expert_list, handle, event)
        # In non-quant mode recv_x is a tensor; in quant mode
        # (DEEP_NORMAL_MODE_USE_INT8_QUANT=1) recv_x is a tuple
        # (recv_x, recv_x_scales).
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            dispatch_event,
        ) = self.buffer.dispatch(
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            expert_alignment=expert_alignment,
            num_worst_tokens=num_worst_tokens,
            config=config,
            previous_event=None,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Unpack recv_x in quant mode (tuple of tensor + scales).
        recv_x_scales = None
        if isinstance(recv_x, tuple):
            recv_x, recv_x_scales = recv_x

        handle_dict = {
            "handle": handle,
            "event": dispatch_event,
            "num_recv_tokens_per_expert_list": num_recv_tokens_per_expert_list,
            "num_tokens_per_expert": num_tokens_per_expert,
            "num_tokens_per_rank": num_tokens_per_rank,
            "is_token_in_rank": is_token_in_rank,
        }

        logger.debug(
            "[ZBALMoEAdapter] Dispatch completed: recv_x.shape=%s", recv_x.shape
        )

        return recv_x, recv_topk_idx, handle_dict, recv_x_scales

    def combine(
        self,
        x: torch.Tensor,
        handle_dict: Dict[str, Any],
        topk_weights: Optional[torch.Tensor] = None,
        config: Optional[Any] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        """Combine (reduce) tokens from different ranks using ZBAL Buffer.

        Args:
            x: Tokens to send, shape ``[num_tokens, hidden]``, dtype ``bfloat16``.
            handle_dict: Communication handle returned by :meth:`dispatch`.
            topk_weights: Expert weights for weighted reduction.
            config: Custom performance tuning config. Uses default if ``None``.
            async_finish: If ``True``, current stream will not wait for completion.
            allocate_on_comm_stream: If ``True``, allocate tensors on comm stream.

        Returns:
            Reduced token tensor, shape ``[num_tokens, hidden]``.
        """
        logger.debug("[ZBALMoEAdapter] Combine started: x.shape=%s", x.shape)

        config = config or self.combine_config
        handle = handle_dict["handle"]

        # Print tensor addresses before combine to diagnose GVA/DMA memory issues.
        _log_tensor_addrs("combine_input", x, None, topk_weights)

        recv_x, _recv_topk_weights, event = self.buffer.combine(
            x=x,
            handle=handle,
            topk_weights=topk_weights,
            config=config,
            previous_event=None,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        logger.debug(
            "[ZBALMoEAdapter] Combine completed: recv_x.shape=%s", recv_x.shape
        )

        return recv_x

    # ------------------------------------------------------------------
    # Low-latency dispatch / combine
    # ------------------------------------------------------------------

    def low_latency_dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_tokens_per_rank: int,
        use_fp8: bool = True,
        round_scale: bool = False,
        use_ue8m0: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ) -> Tuple[Any, torch.Tensor, Dict[str, Any], Any]:
        """Low-latency dispatch using ZBAL Buffer.

        Args:
            x: Input tokens, shape ``[num_tokens, hidden]``, dtype ``bfloat16``.
            topk_idx: Expert indices, shape ``[num_tokens, num_topk]``, dtype ``int64``.
            num_max_tokens_per_rank: Max tokens to dispatch per rank.
            use_fp8: Whether to enable FP8 casting.
            round_scale: Whether to round scales to power of 2.
            use_ue8m0: Whether to use UE8M0 format (requires ``round_scale=True``).
            async_finish: If ``True``, current stream will not wait for completion.
            return_recv_hook: If ``True``, return a receiving hook.

        Returns:
            recv_x: Received tokens (tensor or tuple for FP8).
            recv_count: Token count per local expert, shape ``[num_local_experts]``.
            handle_dict: Communication handle and related info.
            event: Event object.
        """
        logger.debug(
            "[ZBALMoEAdapter] Low-latency dispatch started: x.shape=%s, "
            "topk_idx.shape=%s", x.shape, topk_idx.shape,
        )

        (
            recv_x,
            recv_count,
            handle,
            event,
            hook,
        ) = self.buffer.low_latency_dispatch(
            x=x,
            topk_idx=topk_idx,
            num_max_dispatch_tokens_per_rank=num_max_tokens_per_rank,
            num_experts=self.num_experts,
            use_fp8=use_fp8,
            round_scale=round_scale,
            use_ue8m0=use_ue8m0,
            async_finish=async_finish,
            return_recv_hook=return_recv_hook,
        )

        handle_dict = {
            "handle": handle,
            "num_max_tokens_per_rank": num_max_tokens_per_rank,
            "hidden_size": self.hidden_size,
            "num_experts": self.num_experts,
            "use_fp8": use_fp8,
        }

        logger.debug(
            "[ZBALMoEAdapter] Low-latency dispatch completed: recv_x type=%s, "
            "recv_count.shape=%s", type(recv_x).__name__, recv_count.shape,
        )

        return recv_x, recv_count, handle_dict, event

    def low_latency_combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle_dict: Dict[str, Any],
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Any, Any]:
        """Low-latency combine using ZBAL Buffer.

        Args:
            x: Tokens to send, shape ``[num_local_experts,
                num_max_tokens_per_rank * num_ranks, hidden]``.
            topk_idx: Expert indices, shape ``[num_combined_tokens, num_topk]``.
            topk_weights: Expert weights, shape ``[num_combined_tokens, num_topk]``.
            handle_dict: Communication handle returned by
                :meth:`low_latency_dispatch`.
            zero_copy: Whether the tensor is already in the RDMA buffer.
            async_finish: If ``True``, current stream will not wait for completion.
            return_recv_hook: If ``True``, return a receiving hook.
            out: Optional in-place output tensor.

        Returns:
            combined_x: Reduced token tensor.
            event: Event object.
            hook: Receiving hook (if requested).
        """
        logger.debug(
            "[ZBALMoEAdapter] Low-latency combine started: x.shape=%s, "
            "topk_idx.shape=%s", x.shape, topk_idx.shape,
        )

        handle = handle_dict["handle"]

        combined_x, event, hook = self.buffer.low_latency_combine(
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            handle=handle,
            zero_copy=zero_copy,
            async_finish=async_finish,
            return_recv_hook=return_recv_hook,
            out=out,
        )

        logger.debug(
            "[ZBALMoEAdapter] Low-latency combine completed: "
            "combined_x.shape=%s", combined_x.shape,
        )

        return combined_x, event, hook

    # ------------------------------------------------------------------
    # Fused deep MoE
    # ------------------------------------------------------------------

    def fused_deep_moe(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        gmm1_permuted_weight: torch.Tensor,
        gmm1_permuted_weight_scale: torch.Tensor,
        gmm2_weight: torch.Tensor,
        gmm2_weight_scale: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        quant_mode: int = 1,
        num_max_dispatch_tokens_per_rank: int = 0,
        fuse_mode: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused dispatch + GMM1 + SwiGLU + GMM2 + combine MoE operation.

        Args:
            x: Input tokens, shape ``[bs, h]``.
            topk_idx: Top-K expert IDs, shape ``[bs, topK]``.
            gmm1_permuted_weight: GMM1 weight (INT8).
            gmm1_permuted_weight_scale: GMM1 weight scale (float).
            gmm2_weight: GMM2 weight (INT8).
            gmm2_weight_scale: GMM2 weight scale (float).
            topk_weights: Per-token expert weights, shape ``[bs, topK]``.
            num_experts: Total number of MoE experts.
            quant_mode: Quantization mode (default 1 = INT8).
            num_max_dispatch_tokens_per_rank: Global batch size across all
                EP ranks (0 = current bs).
            fuse_mode: Fusion mode (reserved).

        Returns:
            output: Combined output tokens, shape ``[bs, h]``.
            expert_token_nums: Token count per local expert.
        """
        logger.debug(
            "[ZBALMoEAdapter] Fused deep MoE started: x.shape=%s, "
            "topk_idx.shape=%s", x.shape, topk_idx.shape,
        )

        output, expert_token_nums = self.buffer.fused_deep_moe(
            x=x,
            topk_idx=topk_idx,
            gmm1_permuted_weight=gmm1_permuted_weight,
            gmm1_permuted_weight_scale=gmm1_permuted_weight_scale,
            gmm2_weight=gmm2_weight,
            gmm2_weight_scale=gmm2_weight_scale,
            topk_weights=topk_weights,
            num_experts=num_experts,
            quant_mode=quant_mode,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            fuse_mode=fuse_mode,
        )

        logger.debug(
            "[ZBALMoEAdapter] Fused deep MoE completed: output.shape=%s, "
            "expert_token_nums.shape=%s", output.shape, expert_token_nums.shape,
        )

        return output, expert_token_nums

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clean_low_latency_buffer(
        self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int
    ):
        """Clean the low-latency buffer if it is dirty.

        Must be called before executing any low-latency kernel after running
        the normal dispatch/combine.

        Args:
            num_max_dispatch_tokens_per_rank: Max tokens to dispatch per rank.
            hidden: Hidden dimension size.
            num_experts: Total number of experts.
        """
        self.buffer.clean_low_latency_buffer(
            num_max_dispatch_tokens_per_rank, hidden, num_experts
        )
        logger.debug("[ZBALMoEAdapter] Low-latency buffer cleaned")

    def capture_event(self) -> Any:
        """Capture a NPU event on the current stream.

        Returns:
            The captured event.
        """
        return self.buffer.capture()

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Any:
        """Get recommended dispatch config for the given rank count."""
        from zbal.zbal_buffer import Buffer

        return Buffer.get_dispatch_config(num_ranks)

    @staticmethod
    def get_combine_config(num_ranks: int) -> Any:
        """Get recommended combine config for the given rank count."""
        from zbal.zbal_buffer import Buffer

        return Buffer.get_combine_config(num_ranks)

    @staticmethod
    def get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_ranks: int,
        num_experts: int,
    ) -> int:
        """Get RDMA size hint for low-latency mode."""
        from zbal.zbal_buffer import Buffer

        return Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts
        )
