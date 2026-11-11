# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.models.deepseek_v41.cache_config import group_cache_specs, make_cache_groups


def test_v41_multiple_backings_use_total_block_bytes():
    from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
    from vllm.v1.core.kv_cache_planning import _pool_bytes_per_block_from_builder

    runtime = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41")),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, prefix_cache_retention_interval=None),
        attention_config=SimpleNamespace(hisparse_config=None),
        kv_transfer_config=None,
        speculative_config=None,
    )
    specs = {
        f"model.layers.{layer}.self_attn.swa_cache": AscendSlidingWindowMLASpec(
            block_size=128,
            num_kv_heads=1,
            head_size=head_size,
            dtype=torch.bfloat16,
            sliding_window=1024,
            model_version="deepseek_v41",
        )
        for layer, head_size in enumerate((512, 576))
    }
    groups = make_cache_groups(group_cache_specs(specs))
    builder = get_kv_cache_config_builder(runtime)
    block_bytes = _pool_bytes_per_block_from_builder(builder, runtime, groups)
    assert block_bytes == sum(spec.unpadded_page_size_bytes for spec in specs.values())
    placement = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=7)
    assert placement.num_blocks == 7
    assert len({tensor.size for tensor in placement.kv_cache_tensors}) == 2
    assert sum(tensor.size for tensor in placement.kv_cache_tensors) == 7 * block_bytes
    assert builder._get_pool_bytes_per_block(runtime, []) is None
