# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.ascend_config import EngramOffloadConfig
from vllm_ascend.models.deepseek_v41.engram.embedding import AscendParallelEngramEmbedding
from vllm_ascend.models.deepseek_v41.engram.offload import EngramOffloadPool, make_offload_tables


def test_global_layout_independent_of_tp_and_zero_capacity_readers():
    tables, stride = make_offload_tables([19, 23], 2)
    assert [table.offset for table in tables] == [1, 11]
    assert stride == 23
    for table in tables:
        covered = [row for owner in range(2) for row in range(*table.owner_range(owner))]
        assert covered == list(range(table.rows))
    pool = object.__new__(EngramOffloadPool)
    pool.tables = tables
    pool.owner_index = None
    pool.storage = None
    pool.row_bytes, pool.weight_bytes = 72, 64
    pool.dtype, pool.scale_dtype = torch.int8, torch.float32
    assert pool.owner_range(0) == (0, 0)
    weight, scale = pool.weights(0)
    assert weight.shape == (0, 64) and scale.shape == (0, 2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"local_world_size": 1},
        {"world_size": 3},
        {"owner_ranks": []},
        {"owner_ranks": [2, 2]},
        {"owner_ranks": [2, 4]},
        {"rank_offset": 4},
        {"store_url": "tcp://127.0.0.1:65535"},
        {"store_url": "etcd://127.0.0.1:8500"},
        {"unknown": True},
    ],
)
def test_pool_config_rejects_invalid_topology(overrides):
    values = dict(store_url="tcp://127.0.0.1:8500", world_size=4, local_world_size=2, owner_ranks=[2, 3])
    with pytest.raises(ValueError):
        EngramOffloadConfig(**(values | overrides))


@pytest.mark.parametrize("owner", [None, 0, 1])
def test_checkpoint_ownership_is_independent_of_head_partition(tmp_path, owner):
    key = "layers.1.engram.embed.weight"
    scale_key = key.removesuffix(".weight") + ".scale"
    codes = torch.arange(19 * 64).reshape(19, 64).remainder(127).to(torch.int8)
    scales = torch.full((19, 2), 0.125).to(torch.float8_e8m0fnu)
    save_file({key: codes, scale_key: scales}, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "weights.safetensors", scale_key: "weights.safetensors"}})
    )
    pool = object.__new__(EngramOffloadPool)
    pool.tables, rows = make_offload_tables([19], 2)
    pool.row_bytes, pool.weight_bytes = 72, 64
    pool.dtype, pool.scale_dtype = torch.int8, torch.float32
    pool.owner_index = owner
    pool.storage = torch.zeros((rows, 72), dtype=torch.uint8) if owner is not None else None
    pool.finish_load = Mock()
    table = object.__new__(AscendParallelEngramEmbedding)
    torch.nn.Module.__init__(table)
    table._offload_pool, table.layer_hash_index = pool, 0
    table.vocab_start_idx, table.vocab_end_idx = 3, 6
    table.storage_start_idx, table.storage_end_idx = pool.owner_range(0)
    table.block_size = 32
    weight, scale = pool.weights(0)
    table.weight = torch.nn.Parameter(weight, requires_grad=False)
    table.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
    table.load_checkpoint(tmp_path, key, chunk_rows=3)
    pool.finish_load.assert_called_once_with(0, None)
    start, end = pool.owner_range(0)
    torch.testing.assert_close(table.weight, codes[start:end])
    torch.testing.assert_close(table.weight_scale_inv, scales[start:end].float())
    if owner is not None:
        assert not pool.storage[0].any()


def test_load_readiness_propagates_failure_and_uses_new_epoch():
    pool = object.__new__(EngramOffloadPool)
    pool.config, pool.rank = SimpleNamespace(world_size=2), 0
    pool.load_epochs = [0]
    pool.store = Mock()
    pool.store.get.side_effect = [b"", b"checkpoint read failed", b"", b""]
    with pytest.raises(RuntimeError, match="rank 1: checkpoint read failed"):
        pool.finish_load(0, None)
    pool.finish_load(0, None)
    assert [call.args[0] for call in pool.store.set.call_args_list] == ["loaded/0/0/0", "loaded/0/1/0"]
