# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import importlib
import importlib.util
import inspect
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp


def _load_balanced_moe_module():
    repo_root = Path(__file__).resolve().parents[3]
    python_root = repo_root / "python"
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
    try:
        return importlib.import_module("mori.ops.balanced_moe")
    except Exception:
        mori_pkg = sys.modules.setdefault("mori", types.ModuleType("mori"))
        mori_pkg.__path__ = [str(python_root / "mori")]
        ops_pkg = sys.modules.setdefault("mori.ops", types.ModuleType("mori.ops"))
        ops_pkg.__path__ = [str(python_root / "mori" / "ops")]

    module_path = (
        repo_root
        / "python"
        / "mori"
        / "ops"
        / "balanced_moe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mori.ops.balanced_moe",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_balanced_moe = _load_balanced_moe_module()
BalancedMoeNormalTopKDispatchState = _balanced_moe.BalancedMoeNormalTopKDispatchState
BalancedMoeRuntimeLayout = _balanced_moe.BalancedMoeRuntimeLayout
build_balanced_moe_runtime_layout = _balanced_moe.build_balanced_moe_runtime_layout
build_balanced_moe_plan_from_global_counts = (
    _balanced_moe.build_balanced_moe_plan_from_global_counts
)
build_balanced_moe_plan_from_topk_ids = (
    _balanced_moe.build_balanced_moe_plan_from_topk_ids
)
build_owner_compact_exchange_plan = _balanced_moe.build_owner_compact_exchange_plan
build_owner_compact_exchange_runtime_plan = (
    _balanced_moe.build_owner_compact_exchange_runtime_plan
)
build_owner_compact_exchange_runtime_plan_from_plan = (
    _balanced_moe.build_owner_compact_exchange_runtime_plan_from_plan
)
build_owner_compact_exchange_plan_from_plan = (
    _balanced_moe.build_owner_compact_exchange_plan_from_plan
)
build_owner_compact_need_masks = _balanced_moe.build_owner_compact_need_masks
build_normal_topk_dispatch_tensors = _balanced_moe.build_normal_topk_dispatch_tensors
build_source_partition = _balanced_moe.build_source_partition
build_source_partition_from_offsets = _balanced_moe.build_source_partition_from_offsets
count_local_routes_by_owner_expert = _balanced_moe.count_local_routes_by_owner_expert
combine_balanced_moe_compact = _balanced_moe.combine_balanced_moe_compact
combine_normal_topk_tokens = _balanced_moe.combine_normal_topk_tokens
dispatch_balanced_moe_compact = _balanced_moe.dispatch_balanced_moe_compact
dispatch_permute_normal_topk_tokens = _balanced_moe.dispatch_permute_normal_topk_tokens
dispatch_normal_topk_tokens = _balanced_moe.dispatch_normal_topk_tokens
exchange_owner_compact_needed_rows = _balanced_moe.exchange_owner_compact_needed_rows
gather_local_counts_to_global = _balanced_moe.gather_local_counts_to_global
unpermute_combine_normal_topk_tokens = (
    _balanced_moe.unpermute_combine_normal_topk_tokens
)
balanced_moe_backend_capabilities = _balanced_moe.balanced_moe_backend_capabilities
prepare_owner_compact_needed_rows_runtime_plan = (
    _balanced_moe.prepare_owner_compact_needed_rows_runtime_plan
)


def _counts_owner_source_local():
    # Shape: [owner_rank, source_rank, local_expert].
    return torch.tensor(
        [
            [[2, 8], [1, 1], [1, 1]],
            [[1, 4], [0, 5], [1, 20]],
            [[2, 1], [12, 1], [2, 1]],
        ],
        dtype=torch.int64,
    )


def test_balanced_moe_source_public_abi_matches_backend_contract():
    assert _balanced_moe.__all__ == [
        "BALANCED_MOE_BACKEND_ABI_VERSION",
        "BALANCED_MOE_BACKEND_CAPABILITIES",
        "BalancedMoeCompactCombineOutput",
        "BalancedMoeCompactDispatchOutput",
        "BalancedMoeHotExpert",
        "BalancedMoeNormalTopKDispatchState",
        "BalancedMoeOwnerCompactExchangePlan",
        "BalancedMoePlan",
        "BalancedMoeRuntimeLayout",
        "BalancedMoeSourcePartition",
        "balanced_moe_backend_capabilities",
        "build_balanced_moe_plan",
        "build_balanced_moe_plan_from_global_counts",
        "build_balanced_moe_plan_from_topk_ids",
        "build_balanced_moe_runtime_layout",
        "build_owner_compact_exchange_plan",
        "build_owner_compact_exchange_plan_from_plan",
        "build_owner_compact_exchange_runtime_plan",
        "build_owner_compact_exchange_runtime_plan_from_plan",
        "build_owner_compact_need_masks",
        "build_normal_topk_dispatch_tensors",
        "build_source_partition",
        "build_source_partition_from_offsets",
        "combine_balanced_moe_compact",
        "combine_balanced_moe_compact_rows",
        "combine_normal_topk_tokens",
        "count_local_routes_by_owner_expert",
        "dispatch_permute_normal_topk_tokens",
        "dispatch_balanced_moe_compact",
        "dispatch_balanced_moe_compact_rows",
        "dispatch_normal_topk_tokens",
        "exchange_owner_compact_needed_rows",
        "gather_local_counts_to_global",
        "prepare_owner_compact_needed_rows_runtime_plan",
        "unpermute_combine_normal_topk_tokens",
    ]

    signature = inspect.signature(gather_local_counts_to_global)
    assert signature.parameters["group"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    topk_signature = inspect.signature(build_balanced_moe_plan_from_topk_ids)
    assert topk_signature.parameters["num_local_experts"].kind is inspect.Parameter.KEYWORD_ONLY


def test_balanced_moe_backend_capabilities_are_explicit():
    capabilities = balanced_moe_backend_capabilities()

    assert capabilities["abi_version"] == 1
    assert capabilities["backend"] == "mori"
    assert capabilities["hot_expert_planner"] is True
    assert capabilities["source_partition"] is True
    assert capabilities["owner_compact_exchange_plan"] is True
    assert capabilities["owner_compact_runtime_layout"] is True
    assert capabilities["owner_compact_exchange_autograd"] is True
    assert capabilities["normal_topk_dispatch_tensors"] is True
    assert capabilities["normal_topk_ep_dispatch"] is False
    assert capabilities["normal_topk_ep_dispatch_backend"] == "not_implemented"
    assert capabilities["normal_topk_ep_dispatch_permute"] is False
    assert capabilities["normal_topk_ep_dispatch_permute_backend"] == "not_implemented"
    assert capabilities["balanced_moe_compact_dispatch"] is True
    assert (
        capabilities["balanced_moe_compact_dispatch_backend"]
        == "mori_ep_dispatch_combine_op"
    )
    assert capabilities["balanced_moe_compact_combine"] is True
    assert (
        capabilities["balanced_moe_compact_combine_backend"]
        == "mori_ep_dispatch_combine_op"
    )
    assert capabilities["balanced_moe_compact_rows_dispatch"] is True
    assert (
        capabilities["balanced_moe_compact_rows_dispatch_backend"]
        == "mori_ep_dispatch_combine_op"
    )
    assert capabilities["balanced_moe_compact_rows_combine"] is True
    assert (
        capabilities["balanced_moe_compact_rows_combine_backend"]
        == "mori_ep_dispatch_combine_op"
    )
    assert capabilities["balanced_moe_compact_rows_native"] is True
    assert (
        capabilities["balanced_moe_compact_rows_native_status"]
        == "mori_ep_dispatch_combine_op"
    )
    assert (
        capabilities["owner_compact_exchange_transport"]
        == "torch_distributed_all_to_all_single"
    )
    assert capabilities["owner_compact_exchange_transports"] == (
        "torch_distributed_all_to_all_single",
        "mori_sdma_padded_all2all",
    )
    assert capabilities["native_hot_helper_transport"] is True
    assert capabilities["native_hot_helper_transport_status"] == (
        "opt_in_padded_sdma_all2all"
    )
    assert (
        capabilities["native_owner_compact_exchange_transport"]
        == "mori_sdma_padded_all2all"
    )


def test_normal_topk_dispatch_tensors_are_shared_but_raw_dispatch_is_rejected():
    selected = torch.tensor([[3, 0], [4, 3], [1, 4]], dtype=torch.int64)
    weights = torch.tensor(
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
        dtype=torch.float32,
    )
    partition = build_source_partition(
        selected,
        build_balanced_moe_plan_from_global_counts(
            _counts_owner_source_local(),
            hot_expert_num=2,
        ),
        ep_rank=0,
        presort_by="owner_compact",
    )

    normal_ids, normal_weights = build_normal_topk_dispatch_tensors(
        selected,
        weights,
        partition=partition,
    )
    torch.testing.assert_close(
        normal_ids,
        torch.tensor([[-1, 0], [-1, -1], [1, -1]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        normal_weights,
        torch.tensor([[0.0, 0.1], [0.0, 0.0], [0.7, 0.0]], dtype=torch.float32),
    )

    with pytest.raises(NotImplementedError, match="raw top-k"):
        dispatch_normal_topk_tokens(
            torch.zeros(3, 4),
            normal_ids,
            normal_weights,
            num_experts=6,
            group=None,
        )
    with pytest.raises(NotImplementedError, match="raw top-k"):
        combine_normal_topk_tokens(
            torch.zeros(3, 4),
            BalancedMoeNormalTopKDispatchState(
                recv_topk_ids=None,
                recv_topk_weights=None,
                tokens_per_expert=torch.zeros(6, dtype=torch.int64),
                handle=None,
                num_experts=6,
            ),
            group=None,
        )
    with pytest.raises(NotImplementedError, match="raw top-k"):
        dispatch_permute_normal_topk_tokens(
            torch.zeros(3, 4),
            normal_ids,
            normal_weights,
            num_experts=6,
            num_local_experts=2,
            group=None,
        )
    with pytest.raises(NotImplementedError, match="raw top-k"):
        unpermute_combine_normal_topk_tokens(
            torch.zeros(3, 4),
            BalancedMoeNormalTopKDispatchState(
                recv_topk_ids=None,
                recv_topk_weights=None,
                tokens_per_expert=torch.zeros(6, dtype=torch.int64),
                handle=None,
                num_experts=6,
            ),
            group=None,
        )


def test_count_local_routes_by_owner_expert_from_raw_topk_ids():
    selected = torch.tensor(
        [
            [3, 0, 1],
            [4, 3, 2],
            [5, 1, 4],
            [0, 3, -1],
        ],
        dtype=torch.int64,
    )

    counts = count_local_routes_by_owner_expert(
        selected,
        num_local_experts=2,
        world_size=3,
    )

    torch.testing.assert_close(
        counts,
        torch.tensor(
            [
                [2, 2],
                [1, 3],
                [2, 1],
            ],
            dtype=torch.int64,
        ),
    )
    with pytest.raises(ValueError, match="negative"):
        count_local_routes_by_owner_expert(
            selected,
            num_local_experts=2,
            world_size=3,
            ignore_negative=False,
        )


def test_balanced_moe_public_ops_exports_runtime_plan_builder():
    try:
        mori_ops = importlib.import_module("mori.ops")
        balanced_moe_pkg = importlib.import_module("mori.ops.balanced_moe")
    except ImportError:
        pytest.skip("requires a built MORI package with balanced_moe exports")

    assert (
        mori_ops.build_balanced_moe_runtime_layout
        is balanced_moe_pkg.build_balanced_moe_runtime_layout
    )
    assert (
        mori_ops.build_owner_compact_exchange_runtime_plan
        is balanced_moe_pkg.build_owner_compact_exchange_runtime_plan
    )
    assert (
        mori_ops.build_owner_compact_exchange_runtime_plan_from_plan
        is balanced_moe_pkg.build_owner_compact_exchange_runtime_plan_from_plan
    )
    assert (
        mori_ops.build_balanced_moe_plan_from_topk_ids
        is balanced_moe_pkg.build_balanced_moe_plan_from_topk_ids
    )
    assert (
        mori_ops.count_local_routes_by_owner_expert
        is balanced_moe_pkg.count_local_routes_by_owner_expert
    )
    assert (
        mori_ops.build_normal_topk_dispatch_tensors
        is balanced_moe_pkg.build_normal_topk_dispatch_tensors
    )
    assert (
        mori_ops.dispatch_balanced_moe_compact
        is balanced_moe_pkg.dispatch_balanced_moe_compact
    )
    assert (
        mori_ops.combine_balanced_moe_compact
        is balanced_moe_pkg.combine_balanced_moe_compact
    )
    assert (
        mori_ops.dispatch_permute_normal_topk_tokens
        is balanced_moe_pkg.dispatch_permute_normal_topk_tokens
    )
    assert (
        mori_ops.unpermute_combine_normal_topk_tokens
        is balanced_moe_pkg.unpermute_combine_normal_topk_tokens
    )


def test_balanced_moe_plan_matches_mindspeed_greedy_policy():
    plan = build_balanced_moe_plan_from_global_counts(
        _counts_owner_source_local(),
        hot_expert_num=2,
    )

    assert plan.selected_global_experts == (3, 4)
    assert plan.selected_owner_ranks == (1, 2)
    assert plan.selected_local_experts == (1, 0)
    assert plan.owner_counts == (0, 1, 1)
    assert plan.max_owned_per_rank == 1
    assert plan.owner_shard_offsets == (1, 2)
    assert plan.owner_compact_offsets == (0, 1)
    assert plan.owner_shard_active_offsets == (1, 2)
    assert plan.owner_compact_owner_ranks == (1, 2)
    assert plan.owner_compact_local_experts == (1, 0)
    assert plan.owner_compact_need_masks == (
        (True, True),
        (False, True),
        (True, False),
    )
    assert plan.needed_owner_compact_offsets(0) == (0, 1)
    assert plan.needed_owner_compact_offsets(1) == (1,)
    assert plan.needed_owner_compact_offsets(2) == (0,)
    assert plan.owner_load_before == (14, 31, 19)
    assert plan.exec_load_after == (20, 19, 25)
    assert plan.selected_rows_total == 45
    assert plan.remote_rows_total == 38
    assert plan.modeled_max_load_reduction_pct == pytest.approx(
        (31 - 25) / 31 * 100.0
    )


def test_balanced_moe_plan_threshold_can_disable_execution():
    plan = build_balanced_moe_plan_from_global_counts(
        _counts_owner_source_local(),
        hot_expert_num=2,
        min_reduction_pct=25.0,
    )

    assert plan.hot_experts == ()
    assert plan.selected_global_experts == ()
    assert plan.owner_counts == (0, 0, 0)
    assert plan.exec_load_after == plan.owner_load_before
    assert plan.selected_rows_total == 0
    assert plan.remote_rows_total == 0
    assert plan.owner_compact_need_masks == ((), (), ())
    assert plan.modeled_max_load_reduction_pct == 0.0


def test_owner_compact_need_masks_from_serialized_offsets():
    plan = build_balanced_moe_plan_from_global_counts(
        _counts_owner_source_local(),
        hot_expert_num=2,
    )

    need_masks = build_owner_compact_need_masks(
        selected_source_rank_counts=[
            item.source_rank_counts for item in plan.hot_experts
        ],
        selected_owner_ranks=plan.selected_owner_ranks,
        selected_owner_compact_offsets=plan.owner_compact_offsets,
        world_size=plan.world_size,
        active_owner_compact_count=len(plan.owner_shard_active_offsets),
    )

    assert need_masks == plan.owner_compact_need_masks
    assert tuple(
        tuple(idx for idx, needed in enumerate(row) if needed) for row in need_masks
    ) == ((0, 1), (1,), (0,))


def test_owner_compact_exchange_plan_is_rank_local_and_symmetric():
    plan = build_balanced_moe_plan_from_global_counts(
        _counts_owner_source_local(),
        hot_expert_num=2,
    )

    rank0 = build_owner_compact_exchange_plan_from_plan(plan, rank=0)
    assert rank0.needed_owner_compact_offsets == (0, 1)
    assert rank0.compact_to_needed_index == (0, 1)
    assert rank0.send_owner_compact_offsets_by_rank == ((), (), ())
    assert rank0.recv_owner_compact_offsets_by_rank == ((), (0,), (1,))
    assert rank0.input_splits == (0, 0, 0)
    assert rank0.output_splits == (0, 1, 1)

    rank1 = build_owner_compact_exchange_plan_from_plan(plan, rank=1)
    assert rank1.needed_owner_compact_offsets == (1,)
    assert rank1.compact_to_needed_index == (-1, 0)
    assert rank1.send_owner_compact_offsets_by_rank == ((0,), (), (0,))
    assert rank1.recv_owner_compact_offsets_by_rank == ((), (), (1,))
    assert rank1.input_splits == (1, 0, 1)
    assert rank1.output_splits == (0, 0, 1)
    assert rank1.max_needed_count == 2
    assert rank1.needed_density == pytest.approx(1.0)

    rank2 = build_owner_compact_exchange_plan_from_plan(plan, rank=2)
    assert rank2.needed_owner_compact_offsets == (0,)
    assert rank2.compact_to_needed_index == (0, -1)
    assert rank2.send_owner_compact_offsets_by_rank == ((1,), (1,), ())
    assert rank2.recv_owner_compact_offsets_by_rank == ((), (0,), ())
    assert rank2.input_splits == (1, 1, 0)
    assert rank2.output_splits == (0, 1, 0)


def test_owner_compact_exchange_plan_exposes_flat_forward_backward_abi():
    plan = build_balanced_moe_plan_from_global_counts(
        _counts_owner_source_local(),
        hot_expert_num=2,
    )

    rank0 = build_owner_compact_exchange_plan_from_plan(plan, rank=0)
    assert rank0.send_owner_compact_offsets == ()
    assert rank0.recv_owner_compact_offsets == (0, 1)
    assert rank0.recv_to_needed_index == (0, 1)
    assert rank0.grad_send_needed_indices_by_rank == ((), (0,), (1,))
    assert rank0.grad_send_needed_indices == (0, 1)
    assert rank0.grad_recv_owner_compact_offsets == ()

    rank1 = build_owner_compact_exchange_plan_from_plan(plan, rank=1)
    assert rank1.send_owner_compact_offsets == (0, 0)
    assert rank1.recv_owner_compact_offsets == (1,)
    assert rank1.recv_to_needed_index == (0,)
    assert rank1.grad_send_needed_indices_by_rank == ((), (), (0,))
    assert rank1.grad_send_needed_indices == (0,)
    assert rank1.grad_recv_owner_compact_offsets == (0, 0)
    assert rank1.input_split_offsets == (0, 1, 1, 2)
    assert rank1.output_split_offsets == (0, 0, 0, 1)
    assert rank1.grad_input_split_offsets == (0, 0, 0, 1)
    assert rank1.grad_output_split_offsets == (0, 1, 1, 2)

    rank2 = build_owner_compact_exchange_plan_from_plan(plan, rank=2)
    assert rank2.send_owner_compact_offsets == (1, 1)
    assert rank2.recv_owner_compact_offsets == (0,)
    assert rank2.recv_to_needed_index == (0,)
    assert rank2.grad_send_needed_indices_by_rank == ((), (0,), ())
    assert rank2.grad_send_needed_indices == (0,)
    assert rank2.grad_recv_owner_compact_offsets == (1, 1)

    as_dict = rank1.as_dict()
    assert as_dict["send_owner_compact_offsets"] == [0, 0]
    assert as_dict["recv_owner_compact_offsets"] == [1]
    assert as_dict["recv_to_needed_index"] == [0]
    assert as_dict["grad_send_needed_indices_by_rank"] == [[], [], [0]]
    assert as_dict["grad_send_needed_indices"] == [0]
    assert as_dict["grad_recv_owner_compact_offsets"] == [0, 0]
    assert as_dict["input_split_offsets"] == [0, 1, 1, 2]
    assert as_dict["output_split_offsets"] == [0, 0, 0, 1]
    assert as_dict["grad_input_split_offsets"] == [0, 0, 0, 1]
    assert as_dict["grad_output_split_offsets"] == [0, 1, 1, 2]


def test_owner_compact_exchange_plan_tensor_abi_matches_flat_fields():
    plan = build_balanced_moe_plan_from_global_counts(
        _counts_owner_source_local(),
        hot_expert_num=2,
    )
    rank1 = build_owner_compact_exchange_plan_from_plan(plan, rank=1)

    tensors = rank1.as_tensors(dtype=torch.int32, split_dtype=torch.int16)

    assert tensors["needed_owner_compact_offsets"].dtype == torch.int32
    assert tensors["input_splits"].dtype == torch.int16
    assert tensors["needed_owner_compact_offsets"].tolist() == [1]
    assert tensors["compact_to_needed_index"].tolist() == [-1, 0]
    assert tensors["send_owner_compact_offsets"].tolist() == [0, 0]
    assert tensors["recv_owner_compact_offsets"].tolist() == [1]
    assert tensors["recv_to_needed_index"].tolist() == [0]
    assert tensors["grad_send_needed_indices"].tolist() == [0]
    assert tensors["grad_recv_owner_compact_offsets"].tolist() == [0, 0]
    assert tensors["input_splits"].tolist() == [1, 0, 1]
    assert tensors["output_splits"].tolist() == [0, 0, 1]
    assert tensors["input_split_offsets"].tolist() == [0, 1, 1, 2]
    assert tensors["output_split_offsets"].tolist() == [0, 0, 0, 1]
    assert tensors["grad_input_split_offsets"].tolist() == [0, 0, 0, 1]
    assert tensors["grad_output_split_offsets"].tolist() == [0, 1, 1, 2]
    assert tensors["input_split_offsets"].dtype == torch.int16


def test_owner_compact_runtime_plan_precomputes_local_indices_once():
    plan = build_balanced_moe_plan_from_global_counts(
        _counts_owner_source_local(),
        hot_expert_num=2,
    )
    rank1 = build_owner_compact_exchange_plan_from_plan(plan, rank=1)
    compact_local_indices = torch.tensor([7, 9], dtype=torch.int64)

    runtime_plan = prepare_owner_compact_needed_rows_runtime_plan(
        compact_local_indices=compact_local_indices,
        plan=rank1,
    )

    assert runtime_plan["needed_row_count"] == 1
    assert runtime_plan["input_splits_tuple"] == (1, 0, 1)
    assert runtime_plan["output_splits_tuple"] == (0, 0, 1)
    assert runtime_plan["input_split_offsets_tuple"] == (0, 1, 1, 2)
    assert runtime_plan["output_split_offsets_tuple"] == (0, 0, 0, 1)
    assert runtime_plan["grad_input_split_offsets_tuple"] == (0, 0, 0, 1)
    assert runtime_plan["grad_output_split_offsets_tuple"] == (0, 1, 1, 2)
    assert runtime_plan["send_owner_compact_offsets"].tolist() == [0, 0]
    assert runtime_plan["send_local_indices"].tolist() == [7, 7]
    assert runtime_plan["grad_recv_owner_compact_offsets"].tolist() == [0, 0]
    assert runtime_plan["grad_recv_local_indices"].tolist() == [7, 7]
    assert runtime_plan["transport"] == "torch_distributed_all_to_all_single"


def test_owner_compact_runtime_plan_builder_keeps_translation_in_mori():
    plan = build_balanced_moe_plan_from_global_counts(
        _counts_owner_source_local(),
        hot_expert_num=2,
    )
    exchange_plan, runtime_plan = build_owner_compact_exchange_runtime_plan(
        owner_compact_need_masks=plan.owner_compact_need_masks,
        owner_compact_owner_ranks=plan.owner_compact_owner_ranks,
        rank=2,
        compact_local_indices=[3, 5],
    )

    assert exchange_plan.needed_owner_compact_offsets == (0,)
    assert runtime_plan["needed_row_count"] == 1
    assert runtime_plan["send_owner_compact_offsets"].tolist() == [1, 1]
    assert runtime_plan["send_local_indices"].tolist() == [5, 5]
    assert runtime_plan["grad_recv_owner_compact_offsets"].tolist() == [1, 1]
    assert runtime_plan["grad_recv_local_indices"].tolist() == [5, 5]
    assert runtime_plan["transport"] == "torch_distributed_all_to_all_single"


def test_owner_compact_runtime_plan_carries_mori_transport_choice():
    plan = build_balanced_moe_plan_from_global_counts(
        _counts_owner_source_local(),
        hot_expert_num=2,
    )
    _exchange_plan, runtime_plan = build_owner_compact_exchange_runtime_plan_from_plan(
        plan,
        rank=1,
        compact_local_indices=torch.tensor([7, 9], dtype=torch.int64),
        transport="mori_sdma",
    )

    assert runtime_plan["transport"] == "mori_sdma_padded_all2all"


def test_owner_compact_runtime_plan_accepts_native_transport_alias():
    plan = build_balanced_moe_plan_from_global_counts(
        _counts_owner_source_local(),
        hot_expert_num=2,
    )
    _exchange_plan, runtime_plan = build_owner_compact_exchange_runtime_plan_from_plan(
        plan,
        rank=1,
        compact_local_indices=torch.tensor([7, 9], dtype=torch.int64),
        transport="native",
    )

    assert runtime_plan["transport"] == "mori_sdma_padded_all2all"


def test_owner_compact_runtime_plan_from_plan_keeps_translation_in_mori():
    plan = build_balanced_moe_plan_from_global_counts(
        _counts_owner_source_local(),
        hot_expert_num=2,
    )
    exchange_plan, runtime_plan = build_owner_compact_exchange_runtime_plan_from_plan(
        plan,
        rank=1,
        compact_local_indices=torch.tensor([7, 9], dtype=torch.int64),
    )

    assert exchange_plan.needed_owner_compact_offsets == (1,)
    assert runtime_plan["needed_row_count"] == 1
    assert runtime_plan["send_owner_compact_offsets"].tolist() == [0, 0]
    assert runtime_plan["send_local_indices"].tolist() == [7, 7]
    assert runtime_plan["grad_recv_owner_compact_offsets"].tolist() == [0, 0]
    assert runtime_plan["grad_recv_local_indices"].tolist() == [7, 7]
    assert runtime_plan["transport"] == "torch_distributed_all_to_all_single"


def test_owner_compact_runtime_plan_exchange_matches_fallback_and_backward():
    local_rows = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    compact_local_indices = torch.tensor([2, 0], dtype=torch.int64)
    tensor_plan = {
        "needed_owner_compact_offsets": torch.tensor([0, 1], dtype=torch.int64),
        "send_owner_compact_offsets": torch.tensor([0, 1], dtype=torch.int64),
        "recv_to_needed_index": torch.tensor([0, 1], dtype=torch.int64),
        "grad_send_needed_indices": torch.tensor([0, 1], dtype=torch.int64),
        "grad_recv_owner_compact_offsets": torch.tensor([0, 1], dtype=torch.int64),
        "input_splits": torch.tensor([2], dtype=torch.int64),
        "output_splits": torch.tensor([2], dtype=torch.int64),
    }
    runtime_plan = prepare_owner_compact_needed_rows_runtime_plan(
        compact_local_indices=compact_local_indices,
        plan=tensor_plan,
    )

    fallback_rows = local_rows.detach().clone().requires_grad_(True)
    runtime_rows = local_rows.detach().clone().requires_grad_(True)
    fallback_out = exchange_owner_compact_needed_rows(
        fallback_rows,
        compact_local_indices=compact_local_indices,
        plan=tensor_plan,
    )
    runtime_out = exchange_owner_compact_needed_rows(
        runtime_rows,
        compact_local_indices=None,
        plan=runtime_plan,
    )

    expected = torch.stack((local_rows[2], local_rows[0]), dim=0)
    assert torch.equal(fallback_out, expected)
    assert torch.equal(runtime_out, expected)

    grad_out = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
    fallback_out.backward(grad_out)
    runtime_out.backward(grad_out)

    expected_grad = torch.zeros_like(local_rows)
    expected_grad[2] = grad_out[0]
    expected_grad[0] = grad_out[1]
    assert torch.equal(fallback_rows.grad, expected_grad)
    assert torch.equal(runtime_rows.grad, expected_grad)


def test_owner_compact_exchange_accepts_explicit_split_offsets():
    local_rows = torch.arange(6, dtype=torch.float32).reshape(3, 2).requires_grad_()
    plan = {
        "needed_owner_compact_offsets": torch.tensor([0, 1], dtype=torch.int64),
        "send_owner_compact_offsets": torch.tensor([0, 1], dtype=torch.int64),
        "recv_to_needed_index": torch.tensor([0, 1], dtype=torch.int64),
        "grad_send_needed_indices": torch.tensor([0, 1], dtype=torch.int64),
        "grad_recv_owner_compact_offsets": torch.tensor([0, 1], dtype=torch.int64),
        "input_splits": torch.tensor([2], dtype=torch.int64),
        "output_splits": torch.tensor([2], dtype=torch.int64),
        "input_split_offsets": torch.tensor([0, 2], dtype=torch.int64),
        "output_split_offsets": torch.tensor([0, 2], dtype=torch.int64),
        "grad_input_split_offsets": torch.tensor([0, 2], dtype=torch.int64),
        "grad_output_split_offsets": torch.tensor([0, 2], dtype=torch.int64),
    }

    out = exchange_owner_compact_needed_rows(
        local_rows,
        compact_local_indices=torch.tensor([2, 0], dtype=torch.int64),
        plan=plan,
    )

    expected = torch.stack((local_rows.detach()[2], local_rows.detach()[0]), dim=0)
    torch.testing.assert_close(out, expected)
    out.sum().backward()
    torch.testing.assert_close(
        local_rows.grad,
        torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]),
    )


def test_owner_compact_exchange_rejects_bad_explicit_split_offsets():
    local_rows = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    plan = {
        "needed_owner_compact_offsets": torch.tensor([0, 1], dtype=torch.int64),
        "send_owner_compact_offsets": torch.tensor([0, 1], dtype=torch.int64),
        "recv_to_needed_index": torch.tensor([0, 1], dtype=torch.int64),
        "grad_send_needed_indices": torch.tensor([0, 1], dtype=torch.int64),
        "grad_recv_owner_compact_offsets": torch.tensor([0, 1], dtype=torch.int64),
        "input_splits": torch.tensor([2], dtype=torch.int64),
        "output_splits": torch.tensor([2], dtype=torch.int64),
        "input_split_offsets": torch.tensor([0, 1], dtype=torch.int64),
        "output_split_offsets": torch.tensor([0, 2], dtype=torch.int64),
        "grad_input_split_offsets": torch.tensor([0, 2], dtype=torch.int64),
        "grad_output_split_offsets": torch.tensor([0, 2], dtype=torch.int64),
    }

    with pytest.raises(ValueError, match="input_split_offsets span"):
        exchange_owner_compact_needed_rows(
            local_rows,
            compact_local_indices=torch.tensor([2, 0], dtype=torch.int64),
            plan=plan,
        )


def test_owner_compact_exchange_sdma_transport_requires_rocm_tensor():
    local_rows = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    compact_local_indices = torch.tensor([2, 0], dtype=torch.int64)
    tensor_plan = {
        "needed_owner_compact_offsets": torch.tensor([0, 1], dtype=torch.int64),
        "send_owner_compact_offsets": torch.tensor([0, 1], dtype=torch.int64),
        "recv_to_needed_index": torch.tensor([0, 1], dtype=torch.int64),
        "grad_send_needed_indices": torch.tensor([0, 1], dtype=torch.int64),
        "grad_recv_owner_compact_offsets": torch.tensor([0, 1], dtype=torch.int64),
        "input_splits": torch.tensor([2], dtype=torch.int64),
        "output_splits": torch.tensor([2], dtype=torch.int64),
    }
    runtime_plan = prepare_owner_compact_needed_rows_runtime_plan(
        compact_local_indices=compact_local_indices,
        plan=tensor_plan,
    )
    runtime_plan["transport"] = "mori_sdma_padded_all2all"

    with pytest.raises(ValueError, match="CUDA/ROCm"):
        exchange_owner_compact_needed_rows(
            local_rows,
            compact_local_indices=None,
            plan=runtime_plan,
        )


def test_owner_compact_exchange_requires_local_indices_without_runtime_plan():
    local_rows = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    tensor_plan = {
        "needed_owner_compact_offsets": torch.tensor([0], dtype=torch.int64),
        "send_owner_compact_offsets": torch.tensor([0], dtype=torch.int64),
        "recv_to_needed_index": torch.tensor([0], dtype=torch.int64),
        "grad_send_needed_indices": torch.tensor([0], dtype=torch.int64),
        "grad_recv_owner_compact_offsets": torch.tensor([0], dtype=torch.int64),
        "input_splits": torch.tensor([1], dtype=torch.int64),
        "output_splits": torch.tensor([1], dtype=torch.int64),
    }

    with pytest.raises(ValueError, match="compact_local_indices is required"):
        exchange_owner_compact_needed_rows(
            local_rows,
            compact_local_indices=None,
            plan=tensor_plan,
        )


def test_owner_compact_exchange_single_rank_empty_smoke():
    local_rows = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    plan = {
        "needed_owner_compact_offsets": [],
        "send_owner_compact_offsets": [],
        "recv_to_needed_index": [],
        "grad_send_needed_indices": [],
        "grad_recv_owner_compact_offsets": [],
        "input_splits": [0],
        "output_splits": [0],
    }

    out = exchange_owner_compact_needed_rows(
        local_rows,
        compact_local_indices=[0, 1],
        plan=plan,
    )

    assert out.shape == (0, 2)


def _distributed_owner_compact_exchange_worker(
    rank: int,
    world_size: int,
    init_file: str,
):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        plan = build_owner_compact_exchange_plan(
            owner_compact_need_masks=((False, True), (True, False)),
            owner_compact_owner_ranks=(0, 1),
            rank=rank,
        )
        if rank == 0:
            compact_local_indices = torch.tensor([1, 0], dtype=torch.int64)
            local_rows = torch.tensor(
                [[0.0, 0.0], [10.0, 11.0]],
                requires_grad=True,
            )
            expected_needed = torch.tensor([[20.0, 21.0]])
            grad_needed = torch.tensor([[1.0, 2.0]])
            expected_grad = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        else:
            compact_local_indices = torch.tensor([0, 1], dtype=torch.int64)
            local_rows = torch.tensor(
                [[0.0, 0.0], [20.0, 21.0]],
                requires_grad=True,
            )
            expected_needed = torch.tensor([[10.0, 11.0]])
            grad_needed = torch.tensor([[3.0, 4.0]])
            expected_grad = torch.tensor([[0.0, 0.0], [1.0, 2.0]])

        runtime_plan = prepare_owner_compact_needed_rows_runtime_plan(
            compact_local_indices=compact_local_indices,
            plan=plan,
        )
        needed_rows = exchange_owner_compact_needed_rows(
            local_rows,
            plan=runtime_plan,
        )

        torch.testing.assert_close(needed_rows, expected_needed)
        needed_rows.backward(grad_needed)
        torch.testing.assert_close(local_rows.grad, expected_grad)
    finally:
        torch.distributed.destroy_process_group()


def _distributed_topk_plan_worker(
    rank: int,
    world_size: int,
    init_file: str,
):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        topk_ids = torch.ones((10, 1), dtype=torch.int64)
        local_counts = count_local_routes_by_owner_expert(
            topk_ids,
            num_local_experts=2,
            world_size=world_size,
        )
        global_counts = gather_local_counts_to_global(local_counts)
        torch.testing.assert_close(
            global_counts,
            torch.tensor(
                [
                    [[0, 10], [0, 10]],
                    [[0, 0], [0, 0]],
                ],
                dtype=torch.int64,
            ),
        )

        plan = build_balanced_moe_plan_from_topk_ids(
            topk_ids,
            hot_expert_num=1,
            num_local_experts=2,
            min_reduction_pct=1.0,
        )
        assert plan.selected_global_experts == (1,)
        assert plan.selected_owner_ranks == (0,)
        assert plan.selected_local_experts == (1,)
        assert plan.owner_load_before == (20, 0)
        assert plan.exec_load_after == (10, 10)
        assert plan.selected_rows_total == 20
        assert plan.remote_rows_total == 10
        assert plan.owner_compact_need_masks == ((False,), (True,))
        assert pytest.approx(plan.modeled_max_load_reduction_pct) == 50.0
    finally:
        torch.distributed.destroy_process_group()


def test_build_balanced_moe_plan_from_topk_ids_two_rank_gloo():
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "balanced_moe_topk_plan_init")
        mp.start_processes(
            _distributed_topk_plan_worker,
            args=(world_size, init_file),
            nprocs=world_size,
            join=True,
            start_method="fork",
        )


def test_owner_compact_exchange_two_rank_gloo_autograd():
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "balanced_moe_exchange_init")
        mp.start_processes(
            _distributed_owner_compact_exchange_worker,
            args=(world_size, init_file),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )


def test_source_partition_marks_only_remote_hot_routes():
    plan = build_balanced_moe_plan_from_global_counts(
        _counts_owner_source_local(),
        hot_expert_num=2,
    )
    selected = torch.tensor(
        [
            [3, 0, 1],
            [4, 3, 2],
            [5, 1, 4],
            [0, 3, 4],
        ],
        dtype=torch.int64,
    )

    rank0 = build_source_partition(selected, plan, ep_rank=0)
    assert rank0.remote_flat_positions.tolist() == [0, 3, 4, 8, 10, 11]
    assert rank0.remote_hot_offsets.tolist() == [0, 1, 0, 1, 0, 1]
    assert rank0.remote_owner_shard_offsets.tolist() == [1, 2, 1, 2, 1, 2]
    assert rank0.remote_owner_compact_offsets.tolist() == [0, 1, 0, 1, 0, 1]
    assert rank0.remote_token_indices.tolist() == [0, 1, 1, 2, 3, 3]
    assert rank0.normal_route_mask.tolist() == [
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 0],
    ]

    rank1 = build_source_partition(selected, plan, ep_rank=1)
    assert rank1.remote_flat_positions.tolist() == [3, 8, 11]
    assert rank1.remote_hot_offsets.tolist() == [1, 1, 1]
    assert rank1.remote_owner_shard_offsets.tolist() == [2, 2, 2]
    assert rank1.normal_route_mask.tolist() == [
        [1, 1, 1],
        [0, 1, 1],
        [1, 1, 0],
        [1, 1, 0],
    ]


def test_source_partition_from_offsets_matches_plan_partition():
    plan = build_balanced_moe_plan_from_global_counts(
        _counts_owner_source_local(),
        hot_expert_num=2,
    )
    selected = torch.tensor(
        [
            [3, 0, 1],
            [4, 3, 2],
            [5, 1, 4],
            [0, 3, 4],
        ],
        dtype=torch.int64,
    )

    from_plan = build_source_partition(selected, plan, ep_rank=0)
    from_offsets = build_source_partition_from_offsets(
        selected,
        selected_global_experts=plan.selected_global_experts,
        selected_owner_ranks=plan.selected_owner_ranks,
        selected_owner_shard_offsets=plan.owner_shard_offsets,
        selected_owner_compact_offsets=plan.owner_compact_offsets,
        ep_rank=0,
    )

    assert torch.equal(from_offsets.keep_flat_mask, from_plan.keep_flat_mask)
    assert torch.equal(from_offsets.remote_hot_mask, from_plan.remote_hot_mask)
    assert torch.equal(from_offsets.normal_route_mask, from_plan.normal_route_mask)
    assert torch.equal(from_offsets.remote_flat_positions, from_plan.remote_flat_positions)
    assert torch.equal(from_offsets.remote_hot_offsets, from_plan.remote_hot_offsets)
    assert torch.equal(
        from_offsets.remote_owner_shard_offsets,
        from_plan.remote_owner_shard_offsets,
    )
    assert torch.equal(
        from_offsets.remote_owner_compact_offsets,
        from_plan.remote_owner_compact_offsets,
    )
    assert torch.equal(from_offsets.remote_token_indices, from_plan.remote_token_indices)


def test_source_partition_can_presort_by_owner_compact():
    plan = build_balanced_moe_plan_from_global_counts(
        _counts_owner_source_local(),
        hot_expert_num=2,
    )
    selected = torch.tensor(
        [
            [4, 3, 1],
            [3, 4, 2],
            [0, 4, 3],
            [3, 0, 4],
        ],
        dtype=torch.int64,
    )

    partition = build_source_partition_from_offsets(
        selected,
        selected_global_experts=plan.selected_global_experts,
        selected_owner_ranks=plan.selected_owner_ranks,
        selected_owner_shard_offsets=plan.owner_shard_offsets,
        selected_owner_compact_offsets=plan.owner_compact_offsets,
        ep_rank=0,
        presort_by="owner_compact",
    )

    assert partition.remote_offsets_presorted_by == "owner_compact"
    assert partition.remote_owner_compact_offsets.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert partition.remote_hot_offsets.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert partition.remote_flat_positions.tolist() == [1, 3, 8, 9, 0, 4, 7, 11]
    assert partition.remote_group_ends is not None
    assert partition.remote_group_ends.tolist() == [4, 8]


def test_runtime_layout_builder_groups_partition_and_exchange_plan():
    plan = build_balanced_moe_plan_from_global_counts(
        _counts_owner_source_local(),
        hot_expert_num=2,
    )
    selected = torch.tensor(
        [
            [4, 3, 1],
            [3, 4, 2],
            [0, 4, 3],
            [3, 0, 4],
        ],
        dtype=torch.int64,
    )

    layout = build_balanced_moe_runtime_layout(
        selected,
        plan,
        ep_rank=0,
        compact_local_indices=[11, 13],
    )

    assert isinstance(layout, BalancedMoeRuntimeLayout)
    assert layout.source_partition.remote_offsets_presorted_by == "owner_compact"
    assert layout.source_partition.remote_owner_compact_offsets.tolist() == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ]
    assert layout.source_partition.remote_group_ends is not None
    assert layout.source_partition.remote_group_ends.tolist() == [4, 8]
    assert layout.owner_compact_exchange_plan.needed_owner_compact_offsets == (
        0,
        1,
    )
    runtime_plan = layout.owner_compact_exchange_runtime_plan
    assert runtime_plan["needed_row_count"] == 2
    assert runtime_plan["send_local_indices"].tolist() == []
    assert runtime_plan["grad_recv_local_indices"].tolist() == []

    layout_dict = layout.as_dict()
    source_dict = layout_dict["source_partition"]
    assert source_dict["remote_offsets_presorted_by"] == "owner_compact"
    assert source_dict["remote_owner_compact_offsets"].tolist() == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ]
    exchange_dict = layout_dict["owner_compact_exchange_plan"]
    assert exchange_dict["needed_owner_compact_offsets"] == [0, 1]
    runtime_dict = layout_dict["owner_compact_exchange_runtime_plan"]
    assert runtime_dict["needed_row_count"] == 2


def test_compact_dispatch_output_groups_normal_and_hot_abi():
    try:
        mori_ops = importlib.import_module("mori.ops")
    except ImportError:
        pytest.skip("requires a built MORI package with dispatch/combine exports")
    if not hasattr(mori_ops, "BalancedMoeCompactDispatchOutput"):
        pytest.skip("requires MORI dispatch/combine compact output export")
    BalancedMoeCompactDispatchOutput = mori_ops.BalancedMoeCompactDispatchOutput
    normal_x = torch.empty((2, 3, 4))
    normal_count = torch.tensor([1, 1], dtype=torch.int32)
    normal_src = torch.empty((2, 3), dtype=torch.int32)
    normal_range = torch.empty((0,), dtype=torch.int64)
    hot_x = torch.empty((5, 4))
    hot_scores = torch.empty((5,), dtype=torch.float32)
    hot_src = torch.arange(5, dtype=torch.int64)
    hot_group_ends = torch.tensor([2, 5], dtype=torch.int32)

    output = BalancedMoeCompactDispatchOutput(
        normal_packed_recv_x=normal_x,
        normal_packed_recv_count=normal_count,
        normal_packed_recv_src_info=normal_src,
        normal_packed_recv_layout_range=normal_range,
        hot_packed_x=hot_x,
        hot_packed_scores=hot_scores,
        hot_packed_src_info=hot_src,
        hot_packed_count=None,
        hot_group_ends=hot_group_ends,
        hot_offsets_presorted_by="owner_compact",
    )

    assert output.normal_outputs == (normal_x, normal_count, normal_src, normal_range)
    assert output.hot_outputs == (hot_x, hot_scores, hot_src)
    assert output.hot_group_ends is hot_group_ends
    assert output.hot_offsets_presorted_by == "owner_compact"


def test_compact_combine_output_groups_source_and_token_abi():
    try:
        mori_ops = importlib.import_module("mori.ops")
    except ImportError:
        pytest.skip("requires a built MORI package with dispatch/combine exports")
    if not hasattr(mori_ops, "BalancedMoeCompactCombineOutput"):
        pytest.skip("requires MORI compact combine output export")
    BalancedMoeCompactCombineOutput = mori_ops.BalancedMoeCompactCombineOutput
    rows = torch.empty((3, 4))
    flat_positions = torch.tensor([0, 3, 5], dtype=torch.int64)
    token_output = torch.empty((2, 4))

    output = BalancedMoeCompactCombineOutput(
        source_rank_rows=rows,
        source_rank_flat_positions=flat_positions,
        token_output=token_output,
    )

    assert output.source_outputs == (rows, flat_positions)
    assert output.token_output is token_output
    assert output.has_token_output is True
    assert hasattr(mori_ops.EpDispatchCombineOp, "combine_balanced_moe_compact")

    row_only_output = BalancedMoeCompactCombineOutput(
        source_rank_rows=rows,
        source_rank_flat_positions=None,
    )
    assert row_only_output.source_outputs == (rows, None)
    assert row_only_output.token_output is None
    assert row_only_output.has_token_output is False


def test_balanced_moe_compact_module_api_delegates_to_native_op():
    class FakeCompactOp:
        def __init__(self):
            self.dispatch_call = None
            self.combine_call = None

        def dispatch_balanced_moe_compact(self, *args, **kwargs):
            self.dispatch_call = (args, kwargs)
            return "dispatch-output"

        def combine_balanced_moe_compact(self, *args, **kwargs):
            self.combine_call = (args, kwargs)
            return "combine-output"

    op = FakeCompactOp()
    x = torch.empty((3, 4))
    weights = torch.empty((3, 2), dtype=torch.float32)
    indices = torch.empty((3, 2), dtype=torch.int64)
    partition = build_source_partition(
        indices.fill_(0),
        build_balanced_moe_plan_from_global_counts(
            _counts_owner_source_local(),
            hot_expert_num=1,
        ),
        ep_rank=0,
        presort_by="owner_compact",
    )

    dispatch_result = dispatch_balanced_moe_compact(
        op,
        x,
        weights,
        None,
        indices,
        partition,
        2,
        hot_slot_kind="owner_compact",
        block_num=17,
        rdma_block_num=3,
        warp_per_block=5,
        compute_hot_counts=True,
    )
    assert dispatch_result == "dispatch-output"
    dispatch_args, dispatch_kwargs = op.dispatch_call
    assert dispatch_args[0] is x
    assert dispatch_args[1] is weights
    assert dispatch_args[2] is None
    assert dispatch_args[3] is indices
    assert dispatch_args[4] is partition
    assert dispatch_args[5] == 2
    assert dispatch_kwargs == {
        "hot_slot_kind": "owner_compact",
        "block_num": 17,
        "rdma_block_num": 3,
        "warp_per_block": 5,
        "compute_hot_counts": True,
    }

    rows = torch.empty((4, 4))
    flat_positions = torch.arange(4, dtype=torch.int64)
    splits = torch.tensor([2, 2], dtype=torch.int64)
    combine_result = combine_balanced_moe_compact(
        op,
        rows,
        flat_positions,
        None,
        None,
        splits,
        splits,
        4,
        block_num=11,
        warp_per_block=7,
        return_flat_positions=False,
        top_scores_flat=weights.reshape(-1),
        top_k=2,
        flat_position_offset=6,
        token_output_rows=3,
        return_token_output=True,
    )
    assert combine_result == "combine-output"
    combine_args, combine_kwargs = op.combine_call
    assert combine_args[0] is rows
    assert combine_args[1] is flat_positions
    assert combine_args[2] is None
    assert combine_args[3] is None
    assert combine_args[4] is splits
    assert combine_args[5] is splits
    assert combine_args[6] == 4
    assert combine_kwargs["block_num"] == 11
    assert combine_kwargs["warp_per_block"] == 7
    assert combine_kwargs["return_flat_positions"] is False
    torch.testing.assert_close(combine_kwargs["top_scores_flat"], weights.reshape(-1))
    assert combine_kwargs["top_k"] == 2
    assert combine_kwargs["flat_position_offset"] == 6
    assert combine_kwargs["token_output_rows"] == 3
    assert combine_kwargs["return_token_output"] is True
