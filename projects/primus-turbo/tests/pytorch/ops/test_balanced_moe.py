###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp


def _load_balanced_moe_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "primus_turbo"
        / "pytorch"
        / "ops"
        / "moe"
        / "balanced_moe.py"
    )
    spec = importlib.util.spec_from_file_location("_balanced_moe_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_balanced_moe = _load_balanced_moe_module()
BalancedMoeOwnerCompactExchangePlan = (
    _balanced_moe.BalancedMoeOwnerCompactExchangePlan
)
BalancedMoeCompactCombineOutput = _balanced_moe.BalancedMoeCompactCombineOutput
BalancedMoeCompactDispatchOutput = _balanced_moe.BalancedMoeCompactDispatchOutput
BalancedMoeNormalTopKDispatchState = _balanced_moe.BalancedMoeNormalTopKDispatchState
build_balanced_moe_plan_from_global_counts = (
    _balanced_moe.build_balanced_moe_plan_from_global_counts
)
build_balanced_moe_plan_from_topk_ids = (
    _balanced_moe.build_balanced_moe_plan_from_topk_ids
)
build_balanced_moe_runtime_layout = _balanced_moe.build_balanced_moe_runtime_layout
build_normal_topk_dispatch_tensors = _balanced_moe.build_normal_topk_dispatch_tensors
build_owner_compact_exchange_plan_from_plan = (
    _balanced_moe.build_owner_compact_exchange_plan_from_plan
)
build_owner_compact_exchange_plan = _balanced_moe.build_owner_compact_exchange_plan
build_owner_compact_exchange_runtime_plan_from_plan = (
    _balanced_moe.build_owner_compact_exchange_runtime_plan_from_plan
)
build_source_partition = _balanced_moe.build_source_partition
count_local_routes_by_owner_expert = _balanced_moe.count_local_routes_by_owner_expert
combine_balanced_moe_compact = _balanced_moe.combine_balanced_moe_compact
combine_balanced_moe_compact_rows = _balanced_moe.combine_balanced_moe_compact_rows
combine_normal_topk_tokens = _balanced_moe.combine_normal_topk_tokens
dispatch_balanced_moe_compact = _balanced_moe.dispatch_balanced_moe_compact
dispatch_balanced_moe_compact_rows = _balanced_moe.dispatch_balanced_moe_compact_rows
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


def _counts() -> torch.Tensor:
    # Shape: [owner_rank, source_rank, local_expert].
    return torch.tensor(
        [
            [[2, 8], [1, 1], [1, 1]],
            [[1, 4], [0, 5], [1, 20]],
            [[2, 1], [12, 1], [2, 1]],
        ],
        dtype=torch.int64,
    )


def _plan():
    return build_balanced_moe_plan_from_global_counts(_counts(), hot_expert_num=2)


def test_balanced_moe_backend_capabilities_are_explicit():
    capabilities = balanced_moe_backend_capabilities()

    assert capabilities["abi_version"] == 1
    assert capabilities["backend"] == "primus_turbo"
    assert capabilities["hot_expert_planner"] is True
    assert capabilities["source_partition"] is True
    assert capabilities["owner_compact_exchange_plan"] is True
    assert capabilities["owner_compact_runtime_layout"] is True
    assert capabilities["owner_compact_exchange_autograd"] is True
    assert capabilities["normal_topk_dispatch_tensors"] is True
    assert capabilities["normal_topk_ep_dispatch"] is True
    assert (
        capabilities["normal_topk_ep_dispatch_backend"]
        == "primus_turbo_moe_dispatch"
    )
    assert capabilities["normal_topk_ep_dispatch_permute"] is True
    assert (
        capabilities["normal_topk_ep_dispatch_permute_backend"]
        == "primus_turbo_moe_dispatch_permute"
    )
    assert capabilities["balanced_moe_compact_dispatch"] is True
    assert (
        capabilities["balanced_moe_compact_dispatch_backend"]
        == "primus_turbo_raw_topk_ep_plus_source_compact_hot_rows"
    )
    assert capabilities["balanced_moe_compact_combine"] is True
    assert (
        capabilities["balanced_moe_compact_combine_backend"]
        == "primus_turbo_raw_topk_ep_plus_torch_index_add"
    )
    assert capabilities["balanced_moe_compact_rows_api"] is True
    assert capabilities["balanced_moe_compact_rows_reference_dispatch"] is True
    assert capabilities["balanced_moe_compact_rows_reference_dispatch_backend"] == (
        "torch_distributed_all_to_all_single_reference"
    )
    assert capabilities["balanced_moe_compact_rows_reference_combine"] is True
    assert capabilities["balanced_moe_compact_rows_reference_combine_backend"] == (
        "torch_distributed_all_to_all_single_reference"
    )
    assert capabilities["balanced_moe_compact_rows_dispatch"] is False
    assert capabilities["balanced_moe_compact_rows_dispatch_backend"] == (
        "not_implemented"
    )
    assert capabilities["balanced_moe_compact_rows_combine"] is False
    assert capabilities["balanced_moe_compact_rows_combine_backend"] == (
        "not_implemented"
    )
    assert capabilities["balanced_moe_compact_rows_native"] is False
    assert capabilities["balanced_moe_compact_rows_native_status"] == (
        "compact-row dispatch/combine still use the torch reference path; "
        "owner-compact hot-weight exchange can use MORI SDMA"
    )
    assert (
        capabilities["owner_compact_exchange_transport"]
        == "torch_distributed_all_to_all_single"
    )
    assert capabilities["owner_compact_exchange_transports"] == (
        "torch_distributed_all_to_all_single",
        "primus_turbo_moe_dispatch",
        "mori_sdma_padded_all2all",
    )
    assert capabilities["native_hot_helper_transport"] is True
    assert capabilities["native_hot_helper_transport_status"] == (
        "owner_compact_exchange_can_delegate_to_mori_sdma_padded_all2all"
    )
    assert (
        capabilities["native_owner_compact_exchange_transport"]
        == "mori_sdma_padded_all2all"
    )


def test_normal_topk_dispatch_tensors_mask_remote_hot_routes():
    selected = torch.tensor([[3, 0], [4, 3], [1, 4]], dtype=torch.int64)
    weights = torch.tensor(
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
        dtype=torch.float32,
    )
    partition = build_source_partition(
        selected,
        _plan(),
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

    ids_only, missing_weights = build_normal_topk_dispatch_tensors(
        selected,
        partition=partition,
    )
    torch.testing.assert_close(ids_only, normal_ids)
    assert missing_weights is None


def test_normal_topk_dispatch_wrapper_calls_primus_turbo_runtime(monkeypatch):
    calls: dict[str, object] = {}

    def fake_moe_dispatch(
        x,
        *,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
        num_worst_tokens=0,
    ):
        calls["dispatch"] = {
            "x": x,
            "token_indices": token_indices,
            "token_probs": token_probs,
            "num_experts": num_experts,
            "group": group,
            "async_finish": async_finish,
            "allocate_on_comm_stream": allocate_on_comm_stream,
            "num_worst_tokens": num_worst_tokens,
        }
        return (
            x + 1,
            token_indices + 10,
            token_probs + 2,
            torch.tensor([2, 1, 0], dtype=torch.int64),
            ("fake-handle",),
        )

    def fake_moe_combine(
        expert_output,
        group,
        handle,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        calls["combine"] = {
            "expert_output": expert_output,
            "group": group,
            "handle": handle,
            "async_finish": async_finish,
            "allocate_on_comm_stream": allocate_on_comm_stream,
        }
        return expert_output - 1

    fake_module = types.ModuleType(
        "primus_turbo.pytorch.ops.moe.moe_dispatch_combine"
    )
    fake_module.moe_dispatch = fake_moe_dispatch
    fake_module.moe_combine = fake_moe_combine
    monkeypatch.setitem(sys.modules, "primus_turbo", types.ModuleType("primus_turbo"))
    monkeypatch.setitem(
        sys.modules, "primus_turbo.pytorch", types.ModuleType("primus_turbo.pytorch")
    )
    monkeypatch.setitem(
        sys.modules,
        "primus_turbo.pytorch.ops",
        types.ModuleType("primus_turbo.pytorch.ops"),
    )
    monkeypatch.setitem(
        sys.modules,
        "primus_turbo.pytorch.ops.moe",
        types.ModuleType("primus_turbo.pytorch.ops.moe"),
    )
    monkeypatch.setitem(
        sys.modules,
        "primus_turbo.pytorch.ops.moe.moe_dispatch_combine",
        fake_module,
    )

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    ids = torch.tensor([[1, -1], [2, 0]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 0.0], [0.25, 0.25]], dtype=torch.bfloat16)
    recv_x, state = dispatch_normal_topk_tokens(
        x,
        ids,
        weights,
        num_experts=3,
        group="fake-group",
        async_finish=True,
        allocate_on_comm_stream=True,
        num_worst_tokens=7,
    )

    assert isinstance(state, BalancedMoeNormalTopKDispatchState)
    torch.testing.assert_close(recv_x, x + 1)
    assert state.handle == ("fake-handle",)
    assert state.num_experts == 3
    assert state.async_finish is True
    assert state.allocate_on_comm_stream is True
    dispatch_call = calls["dispatch"]
    assert dispatch_call["group"] == "fake-group"
    assert dispatch_call["num_experts"] == 3
    assert dispatch_call["num_worst_tokens"] == 7
    assert dispatch_call["token_indices"].dtype == torch.int64
    assert dispatch_call["token_probs"].dtype == torch.float32

    combined = combine_normal_topk_tokens(recv_x, state, group="fake-group")
    torch.testing.assert_close(combined, x)
    combine_call = calls["combine"]
    assert combine_call["group"] == "fake-group"
    assert combine_call["handle"] == ("fake-handle",)
    assert combine_call["async_finish"] is True
    assert combine_call["allocate_on_comm_stream"] is True


def test_normal_topk_dispatch_permute_wrapper_calls_primus_turbo_runtime(monkeypatch):
    calls: dict[str, object] = {}

    def fake_moe_dispatch(
        x,
        *,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
        num_worst_tokens=0,
    ):
        calls["dispatch"] = {
            "token_indices": token_indices,
            "token_probs": token_probs,
            "num_experts": num_experts,
            "group": group,
            "async_finish": async_finish,
            "allocate_on_comm_stream": allocate_on_comm_stream,
            "num_worst_tokens": num_worst_tokens,
        }
        return (
            x + 1,
            token_indices + 10,
            token_probs + 2,
            torch.tensor([2, 1, 0], dtype=torch.int64),
            ("fake-handle",),
        )

    def fake_moe_permute(
        hidden_states,
        indices,
        *,
        num_local_experts,
        num_topk,
        pad_multiple,
        num_permuted_tokens,
        probs,
        probs_layout,
    ):
        calls["permute"] = {
            "hidden_states": hidden_states,
            "indices": indices,
            "num_local_experts": num_local_experts,
            "num_topk": num_topk,
            "pad_multiple": pad_multiple,
            "num_permuted_tokens": num_permuted_tokens,
            "probs": probs,
            "probs_layout": probs_layout,
        }
        row_id_map = torch.tensor([1, 0], dtype=torch.int64)
        tokens_per_expert = torch.tensor([1, 1], dtype=torch.int64)
        num_dispatched = torch.tensor([2], dtype=torch.int64)
        return (
            hidden_states + 3,
            row_id_map,
            tokens_per_expert,
            None,
            num_dispatched,
            None,
            probs + 4,
        )

    def fake_moe_unpermute(
        hidden_states,
        row_id_map,
        num_dispatched_token_tensor,
        *,
        restore_shape,
        num_local_experts,
        pad_multiple,
    ):
        calls["unpermute"] = {
            "hidden_states": hidden_states,
            "row_id_map": row_id_map,
            "num_dispatched_token_tensor": num_dispatched_token_tensor,
            "restore_shape": restore_shape,
            "num_local_experts": num_local_experts,
            "pad_multiple": pad_multiple,
        }
        return hidden_states - 3, None

    def fake_moe_combine(
        expert_output,
        group,
        handle,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        calls["combine"] = {
            "expert_output": expert_output,
            "group": group,
            "handle": handle,
            "async_finish": async_finish,
            "allocate_on_comm_stream": allocate_on_comm_stream,
        }
        return expert_output - 1

    fake_dispatch_module = types.ModuleType(
        "primus_turbo.pytorch.ops.moe.moe_dispatch_combine"
    )
    fake_dispatch_module.moe_dispatch = fake_moe_dispatch
    fake_dispatch_module.moe_combine = fake_moe_combine
    fake_pytorch = types.ModuleType("primus_turbo.pytorch")
    fake_pytorch.ops = types.SimpleNamespace(
        moe_permute=fake_moe_permute,
        moe_unpermute=fake_moe_unpermute,
    )
    monkeypatch.setitem(sys.modules, "primus_turbo", types.ModuleType("primus_turbo"))
    monkeypatch.setitem(sys.modules, "primus_turbo.pytorch", fake_pytorch)
    monkeypatch.setitem(
        sys.modules,
        "primus_turbo.pytorch.ops",
        types.ModuleType("primus_turbo.pytorch.ops"),
    )
    monkeypatch.setitem(
        sys.modules,
        "primus_turbo.pytorch.ops.moe",
        types.ModuleType("primus_turbo.pytorch.ops.moe"),
    )
    monkeypatch.setitem(
        sys.modules,
        "primus_turbo.pytorch.ops.moe.moe_dispatch_combine",
        fake_dispatch_module,
    )

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    ids = torch.tensor([[1, -1], [2, 0]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 0.0], [0.25, 0.25]], dtype=torch.bfloat16)
    permuted, tokens_per_expert, permuted_weights, state = (
        dispatch_permute_normal_topk_tokens(
            x,
            ids,
            weights,
            num_experts=3,
            num_local_experts=2,
            group="fake-group",
            pad_multiple=16,
            num_permuted_tokens=128,
            async_finish=True,
            allocate_on_comm_stream=True,
            num_worst_tokens=7,
            use_cuda_num_tokens_per_expert=True,
        )
    )

    torch.testing.assert_close(permuted, x + 4)
    torch.testing.assert_close(tokens_per_expert, torch.tensor([1, 1]))
    torch.testing.assert_close(permuted_weights, weights.float() + 6)
    assert state.row_id_map is not None
    assert state.num_dispatched_token_tensor is not None
    assert state.hidden_shape_before_permute == tuple((x + 1).shape)
    assert state.num_local_experts == 2
    assert state.num_topk == 2
    assert state.pad_multiple == 16

    permute_call = calls["permute"]
    assert permute_call["indices"].dtype == torch.int64
    assert permute_call["probs"].dtype == torch.float32
    assert permute_call["num_local_experts"] == 2
    assert permute_call["num_topk"] == 2
    assert permute_call["pad_multiple"] == 16
    assert permute_call["num_permuted_tokens"] == 128
    assert permute_call["probs_layout"] == "topk"

    combined = unpermute_combine_normal_topk_tokens(
        permuted,
        state,
        group="fake-group",
    )
    torch.testing.assert_close(combined, x)
    unpermute_call = calls["unpermute"]
    assert unpermute_call["restore_shape"] == tuple((x + 1).shape)
    assert unpermute_call["num_local_experts"] == 2
    assert unpermute_call["pad_multiple"] == 16
    combine_call = calls["combine"]
    assert combine_call["group"] == "fake-group"
    assert combine_call["handle"] == ("fake-handle",)
    assert combine_call["async_finish"] is True
    assert combine_call["allocate_on_comm_stream"] is True


def test_balanced_moe_compact_dispatch_combine_contract(monkeypatch):
    calls: dict[str, object] = {}

    def fake_moe_dispatch(
        x,
        *,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
        num_worst_tokens=0,
    ):
        calls["dispatch"] = {
            "token_indices": token_indices,
            "token_probs": token_probs,
            "num_experts": num_experts,
            "group": group,
            "async_finish": async_finish,
            "allocate_on_comm_stream": allocate_on_comm_stream,
            "num_worst_tokens": num_worst_tokens,
        }
        return (
            x + 1,
            token_indices,
            token_probs,
            torch.tensor([2, 1, 0], dtype=torch.int64),
            ("fake-handle",),
        )

    def fake_moe_permute(
        hidden_states,
        indices,
        *,
        num_local_experts,
        num_topk,
        pad_multiple,
        num_permuted_tokens,
        probs,
        probs_layout,
    ):
        calls["permute"] = {
            "hidden_states": hidden_states,
            "indices": indices,
            "num_local_experts": num_local_experts,
            "num_topk": num_topk,
            "pad_multiple": pad_multiple,
            "num_permuted_tokens": num_permuted_tokens,
            "probs": probs,
            "probs_layout": probs_layout,
        }
        return (
            hidden_states + 3,
            torch.arange(hidden_states.size(0), dtype=torch.int64),
            torch.tensor([1, 1], dtype=torch.int64),
            None,
            torch.tensor([hidden_states.size(0)], dtype=torch.int64),
            None,
            probs + 4,
        )

    def fake_moe_unpermute(
        hidden_states,
        row_id_map,
        num_dispatched_token_tensor,
        *,
        restore_shape,
        num_local_experts,
        pad_multiple,
    ):
        calls["unpermute"] = {
            "hidden_states": hidden_states,
            "row_id_map": row_id_map,
            "num_dispatched_token_tensor": num_dispatched_token_tensor,
            "restore_shape": restore_shape,
            "num_local_experts": num_local_experts,
            "pad_multiple": pad_multiple,
        }
        return hidden_states - 3, None

    def fake_moe_combine(
        expert_output,
        group,
        handle,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        calls["combine"] = {
            "expert_output": expert_output,
            "group": group,
            "handle": handle,
            "async_finish": async_finish,
            "allocate_on_comm_stream": allocate_on_comm_stream,
        }
        return expert_output - 1

    fake_dispatch_module = types.ModuleType(
        "primus_turbo.pytorch.ops.moe.moe_dispatch_combine"
    )
    fake_dispatch_module.moe_dispatch = fake_moe_dispatch
    fake_dispatch_module.moe_combine = fake_moe_combine
    fake_pytorch = types.ModuleType("primus_turbo.pytorch")
    fake_pytorch.ops = types.SimpleNamespace(
        moe_permute=fake_moe_permute,
        moe_unpermute=fake_moe_unpermute,
    )
    monkeypatch.setitem(sys.modules, "primus_turbo", types.ModuleType("primus_turbo"))
    monkeypatch.setitem(sys.modules, "primus_turbo.pytorch", fake_pytorch)
    monkeypatch.setitem(
        sys.modules,
        "primus_turbo.pytorch.ops",
        types.ModuleType("primus_turbo.pytorch.ops"),
    )
    monkeypatch.setitem(
        sys.modules,
        "primus_turbo.pytorch.ops.moe",
        types.ModuleType("primus_turbo.pytorch.ops.moe"),
    )
    monkeypatch.setitem(
        sys.modules,
        "primus_turbo.pytorch.ops.moe.moe_dispatch_combine",
        fake_dispatch_module,
    )

    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float32,
    )
    selected = torch.tensor([[3, 0], [4, 3], [1, 4]], dtype=torch.int64)
    weights = torch.tensor(
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
        dtype=torch.float32,
    )
    partition = build_source_partition(
        selected,
        _plan(),
        ep_rank=0,
        presort_by="owner_compact",
    )

    dispatch_output = dispatch_balanced_moe_compact(
        x,
        selected,
        weights,
        num_experts=6,
        num_local_experts=2,
        group="fake-group",
        partition=partition,
        pad_multiple=16,
        num_permuted_tokens=128,
        async_finish=True,
        allocate_on_comm_stream=True,
        num_worst_tokens=7,
        use_cuda_num_tokens_per_expert=True,
    )

    assert isinstance(dispatch_output, BalancedMoeCompactDispatchOutput)
    torch.testing.assert_close(dispatch_output.normal_expert_input, x + 4)
    torch.testing.assert_close(
        dispatch_output.hot_expert_input,
        x.index_select(0, dispatch_output.hot_token_indices),
    )
    torch.testing.assert_close(
        dispatch_output.hot_expert_weights,
        weights.reshape(-1).index_select(0, partition.remote_flat_positions),
    )
    assert dispatch_output.hot_offsets_presorted_by == "owner_compact"
    assert dispatch_output.hot_group_ends is not None

    dispatch_call = calls["dispatch"]
    torch.testing.assert_close(
        dispatch_call["token_indices"],
        torch.tensor([[-1, 0], [-1, -1], [1, -1]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        dispatch_call["token_probs"],
        torch.tensor([[0.0, 0.1], [0.0, 0.0], [0.7, 0.0]], dtype=torch.float32),
    )

    hot_output = torch.ones_like(dispatch_output.hot_expert_input)
    combine_output = combine_balanced_moe_compact(
        dispatch_output.normal_expert_input,
        dispatch_output,
        group="fake-group",
        hot_expert_output=hot_output,
    )

    assert isinstance(combine_output, BalancedMoeCompactCombineOutput)
    expected = x.clone()
    expected.index_add_(
        0,
        dispatch_output.hot_token_indices,
        hot_output * dispatch_output.hot_expert_weights.reshape(-1, 1),
    )
    torch.testing.assert_close(combine_output.normal_combined, x)
    torch.testing.assert_close(combine_output.combined, expected)
    assert combine_output.hot_offsets_presorted_by == "owner_compact"


def test_compact_rows_dispatch_wrapper_delegates_to_backend_method():
    calls: dict[str, object] = {}

    class FakeCompactRowsOp:
        def dispatch_balanced_moe_compact_rows(
            self,
            local_rows,
            local_flat_positions,
            local_num_tokens_per_expert,
            recv_counts_rank_major,
            input_splits,
            output_splits,
            num_output_rows,
            flat_position_rank_stride,
            *,
            block_num=-1,
            warp_per_block=-1,
            return_flat_positions=True,
        ):
            calls["dispatch"] = {
                "local_rows": local_rows,
                "local_flat_positions": local_flat_positions,
                "local_num_tokens_per_expert": local_num_tokens_per_expert,
                "recv_counts_rank_major": recv_counts_rank_major,
                "input_splits": input_splits,
                "output_splits": output_splits,
                "num_output_rows": num_output_rows,
                "flat_position_rank_stride": flat_position_rank_stride,
                "block_num": block_num,
                "warp_per_block": warp_per_block,
                "return_flat_positions": return_flat_positions,
            }
            return "dispatch-output"

    local_rows = torch.randn(3, 2)
    local_flat_positions = torch.tensor([4, 5, 6], dtype=torch.int64)
    local_num_tokens_per_expert = torch.tensor([1, 2], dtype=torch.int32)
    recv_counts_rank_major = torch.tensor([2, 1], dtype=torch.int32)
    input_splits = torch.tensor([1, 2], dtype=torch.int32)
    output_splits = torch.tensor([2, 1], dtype=torch.int32)

    output = dispatch_balanced_moe_compact_rows(
        FakeCompactRowsOp(),
        local_rows,
        local_flat_positions,
        local_num_tokens_per_expert,
        recv_counts_rank_major,
        input_splits,
        output_splits,
        3,
        1024,
        block_num=16,
        warp_per_block=4,
        return_flat_positions=False,
    )

    assert output == "dispatch-output"
    assert calls["dispatch"]["local_rows"] is local_rows
    assert calls["dispatch"]["local_flat_positions"] is local_flat_positions
    assert calls["dispatch"]["local_num_tokens_per_expert"] is (
        local_num_tokens_per_expert
    )
    assert calls["dispatch"]["recv_counts_rank_major"] is recv_counts_rank_major
    assert calls["dispatch"]["input_splits"] is input_splits
    assert calls["dispatch"]["output_splits"] is output_splits
    assert calls["dispatch"]["num_output_rows"] == 3
    assert calls["dispatch"]["flat_position_rank_stride"] == 1024
    assert calls["dispatch"]["block_num"] == 16
    assert calls["dispatch"]["warp_per_block"] == 4
    assert calls["dispatch"]["return_flat_positions"] is False


def test_compact_rows_combine_wrapper_delegates_to_backend_method():
    calls: dict[str, object] = {}

    class FakeCompactRowsOp:
        def combine_standard_ep_compact_native(
            self,
            expert_major_rows,
            expert_major_flat_positions,
            expert_major_to_rank_major_indices,
            recv_counts_rank_major,
            input_splits,
            output_splits,
            num_output_rows,
            *,
            block_num=-1,
            warp_per_block=-1,
            return_flat_positions=True,
            top_scores_flat=None,
            top_k=0,
            flat_position_offset=0,
            token_output_rows=0,
            return_token_output=False,
        ):
            calls["combine"] = {
                "expert_major_rows": expert_major_rows,
                "expert_major_flat_positions": expert_major_flat_positions,
                "expert_major_to_rank_major_indices": (
                    expert_major_to_rank_major_indices
                ),
                "recv_counts_rank_major": recv_counts_rank_major,
                "input_splits": input_splits,
                "output_splits": output_splits,
                "num_output_rows": num_output_rows,
                "block_num": block_num,
                "warp_per_block": warp_per_block,
                "return_flat_positions": return_flat_positions,
                "top_scores_flat": top_scores_flat,
                "top_k": top_k,
                "flat_position_offset": flat_position_offset,
                "token_output_rows": token_output_rows,
                "return_token_output": return_token_output,
            }
            return "combine-output"

    expert_major_rows = torch.randn(3, 2)
    expert_major_flat_positions = torch.tensor([4, 5, 6], dtype=torch.int64)
    expert_major_to_rank_major_indices = torch.tensor([2, 0, 1], dtype=torch.int64)
    recv_counts_rank_major = torch.tensor([2, 1], dtype=torch.int32)
    input_splits = torch.tensor([1, 2], dtype=torch.int32)
    output_splits = torch.tensor([2, 1], dtype=torch.int32)
    top_scores_flat = torch.tensor([0.5, 0.25, 0.125], dtype=torch.float32)

    output = combine_balanced_moe_compact_rows(
        FakeCompactRowsOp(),
        expert_major_rows,
        expert_major_flat_positions,
        expert_major_to_rank_major_indices,
        recv_counts_rank_major,
        input_splits,
        output_splits,
        3,
        block_num=32,
        warp_per_block=8,
        return_flat_positions=False,
        top_scores_flat=top_scores_flat,
        top_k=6,
        flat_position_offset=128,
        token_output_rows=64,
        return_token_output=True,
    )

    assert output == "combine-output"
    assert calls["combine"]["expert_major_rows"] is expert_major_rows
    assert calls["combine"]["expert_major_flat_positions"] is (
        expert_major_flat_positions
    )
    assert calls["combine"]["expert_major_to_rank_major_indices"] is (
        expert_major_to_rank_major_indices
    )
    assert calls["combine"]["recv_counts_rank_major"] is recv_counts_rank_major
    assert calls["combine"]["input_splits"] is input_splits
    assert calls["combine"]["output_splits"] is output_splits
    assert calls["combine"]["num_output_rows"] == 3
    assert calls["combine"]["block_num"] == 32
    assert calls["combine"]["warp_per_block"] == 8
    assert calls["combine"]["return_flat_positions"] is False
    assert calls["combine"]["top_scores_flat"] is top_scores_flat
    assert calls["combine"]["top_k"] == 6
    assert calls["combine"]["flat_position_offset"] == 128
    assert calls["combine"]["token_output_rows"] == 64
    assert calls["combine"]["return_token_output"] is True


def test_compact_rows_torch_reference_dispatch_combine_roundtrip():
    local_rows = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    local_flat_positions = torch.tensor([0, 1, 3], dtype=torch.int64)
    local_num_tokens_per_expert = torch.tensor([2, 1], dtype=torch.int64)
    recv_counts_rank_major = torch.tensor([2, 1], dtype=torch.int64)
    splits = torch.tensor([3], dtype=torch.int64)

    expert_rows, expert_positions = dispatch_balanced_moe_compact_rows(
        None,
        local_rows,
        local_flat_positions,
        local_num_tokens_per_expert,
        recv_counts_rank_major,
        splits,
        splits,
        3,
        6,
    )

    torch.testing.assert_close(expert_rows, local_rows)
    torch.testing.assert_close(expert_positions, local_flat_positions)

    source_rows, source_positions, token_output = combine_balanced_moe_compact_rows(
        None,
        expert_rows,
        expert_positions,
        None,
        recv_counts_rank_major,
        splits,
        splits,
        3,
        top_scores_flat=torch.tensor(
            [0.5, 0.25, 0.0, 0.125],
            dtype=torch.float32,
        ),
        top_k=2,
        token_output_rows=2,
        return_token_output=True,
    )

    torch.testing.assert_close(source_rows, local_rows)
    torch.testing.assert_close(source_positions, local_flat_positions)
    torch.testing.assert_close(
        token_output,
        torch.tensor(
            [
                [0.5, 0.25],
                [0.25, 0.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_compact_rows_wrappers_reject_missing_backend_method():
    rows = torch.randn(1, 2)
    counts = torch.ones(1, dtype=torch.int32)

    with pytest.raises(TypeError, match="compact-row API requires"):
        dispatch_balanced_moe_compact_rows(
            object(),
            rows,
            None,
            counts,
            counts,
            counts,
            counts,
            1,
            1,
        )

    with pytest.raises(TypeError, match="compact-row API requires"):
        combine_balanced_moe_compact_rows(
            object(),
            rows,
            None,
            None,
            counts,
            counts,
            counts,
            1,
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


def test_greedy_hot_expert_plan_uses_owner_compact_layout():
    plan = _plan()

    assert plan.world_size == 3
    assert plan.num_local_experts == 2
    assert plan.selected_global_experts == (3, 4)
    assert plan.selected_owner_ranks == (1, 2)
    assert plan.selected_local_experts == (1, 0)
    assert plan.owner_counts == (0, 1, 1)
    assert plan.max_owned_per_rank == 1
    assert plan.owner_shard_offsets == (1, 2)
    assert plan.owner_shard_active_offsets == (1, 2)
    assert plan.owner_compact_offsets == (0, 1)
    assert plan.owner_compact_owner_ranks == (1, 2)
    assert plan.owner_compact_local_experts == (1, 0)
    assert plan.owner_load_before == (14, 31, 19)
    assert plan.exec_load_after == (20, 19, 25)
    assert plan.selected_rows_total == 45
    assert plan.remote_rows_total == 38
    assert plan.owner_compact_need_masks == (
        (True, True),
        (False, True),
        (True, False),
    )
    assert pytest.approx(plan.modeled_max_load_reduction_pct) == (6 / 31 * 100)


def test_min_reduction_threshold_rejects_plan():
    plan = build_balanced_moe_plan_from_global_counts(
        _counts(),
        hot_expert_num=2,
        min_reduction_pct=99.0,
    )

    assert plan.hot_experts == ()
    assert plan.selected_rows_total == 0
    assert plan.remote_rows_total == 0
    assert plan.exec_load_after == plan.owner_load_before


def test_source_partition_presorts_remote_hot_rows_by_owner_compact():
    selected = torch.tensor(
        [
            [3, 0],
            [4, 3],
            [1, 4],
            [3, 4],
        ],
        dtype=torch.int64,
    )
    partition = build_source_partition(
        selected,
        _plan(),
        ep_rank=0,
        presort_by="owner_compact",
    )

    torch.testing.assert_close(
        partition.normal_route_mask,
        torch.tensor(
            [
                [0, 1],
                [0, 0],
                [1, 0],
                [0, 0],
            ],
            dtype=torch.uint8,
        ),
    )
    torch.testing.assert_close(
        partition.remote_flat_positions,
        torch.tensor([0, 3, 6, 2, 5, 7], dtype=torch.int64),
    )
    torch.testing.assert_close(
        partition.remote_hot_offsets,
        torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        partition.remote_owner_shard_offsets,
        torch.tensor([1, 1, 1, 2, 2, 2], dtype=torch.int64),
    )
    torch.testing.assert_close(
        partition.remote_owner_compact_offsets,
        torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        partition.remote_token_indices,
        torch.tensor([0, 1, 3, 1, 2, 3], dtype=torch.int64),
    )
    torch.testing.assert_close(
        partition.remote_group_ends,
        torch.tensor([3, 6], dtype=torch.int32),
    )
    assert partition.remote_offsets_presorted_by == "owner_compact"


def test_owner_compact_exchange_plan_matches_forward_and_backward_edges():
    plan = _plan()

    rank0 = build_owner_compact_exchange_plan_from_plan(plan, rank=0)
    assert rank0.needed_owner_compact_offsets == (0, 1)
    assert rank0.compact_to_needed_index == (0, 1)
    assert rank0.input_splits == (0, 0, 0)
    assert rank0.output_splits == (0, 1, 1)
    assert rank0.recv_owner_compact_offsets == (0, 1)
    assert rank0.grad_send_needed_indices == (0, 1)
    assert rank0.as_dict()["recv_to_needed_index"] == [0, 1]

    rank1 = build_owner_compact_exchange_plan_from_plan(plan, rank=1)
    assert rank1.needed_owner_compact_offsets == (1,)
    assert rank1.compact_to_needed_index == (-1, 0)
    assert rank1.input_splits == (1, 0, 1)
    assert rank1.output_splits == (0, 0, 1)
    assert rank1.send_owner_compact_offsets == (0, 0)
    assert rank1.recv_owner_compact_offsets == (1,)
    assert rank1.grad_send_needed_indices == (0,)
    assert rank1.grad_recv_owner_compact_offsets == (0, 0)
    assert rank1.input_split_offsets == (0, 1, 1, 2)
    assert rank1.output_split_offsets == (0, 0, 0, 1)
    assert rank1.grad_input_split_offsets == (0, 0, 0, 1)
    assert rank1.grad_output_split_offsets == (0, 1, 1, 2)
    assert rank1.as_tensors(dtype=torch.int32)["input_splits"].tolist() == [1, 0, 1]
    assert rank1.as_tensors(dtype=torch.int32)["input_split_offsets"].tolist() == [
        0,
        1,
        1,
        2,
    ]


def test_owner_compact_runtime_plan_precomputes_local_indices_once():
    plan = _plan()
    exchange_plan, runtime_plan = build_owner_compact_exchange_runtime_plan_from_plan(
        plan,
        rank=1,
        compact_local_indices=torch.tensor([7, 9], dtype=torch.int64),
        dtype=torch.int32,
        split_dtype=torch.int16,
    )

    assert exchange_plan.needed_owner_compact_offsets == (1,)
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
    assert runtime_plan["input_splits"].dtype == torch.int16
    assert runtime_plan["transport"] == "torch_distributed_all_to_all_single"


def test_owner_compact_runtime_plan_accepts_primus_turbo_dispatch_transport():
    _exchange_plan, runtime_plan = build_owner_compact_exchange_runtime_plan_from_plan(
        _plan(),
        rank=1,
        compact_local_indices=torch.tensor([7, 9], dtype=torch.int64),
        transport="primus_turbo_moe_dispatch",
    )

    assert runtime_plan["transport"] == "primus_turbo_moe_dispatch"


@pytest.mark.parametrize(
    "transport",
    [
        "mori_sdma_padded_all2all",
        "backend",
        "primus",
        "primus_turbo",
        "deepep",
        "turboep",
    ],
)
def test_owner_compact_runtime_plan_rejects_non_native_transport_aliases(transport):
    with pytest.raises(ValueError, match="Primus-Turbo"):
        build_owner_compact_exchange_runtime_plan_from_plan(
            _plan(),
            rank=1,
            compact_local_indices=torch.tensor([7, 9], dtype=torch.int64),
            transport=transport,
        )


def test_runtime_layout_combines_source_partition_and_exchange_plan():
    selected = torch.tensor([[3, 0], [4, 3]], dtype=torch.int64)
    layout = build_balanced_moe_runtime_layout(
        selected,
        _plan(),
        ep_rank=0,
        compact_local_indices=torch.tensor([11, 13], dtype=torch.int64),
        presort_by="owner_compact",
    )

    assert layout.owner_compact_exchange_plan.needed_owner_compact_offsets == (0, 1)
    assert layout.owner_compact_exchange_runtime_plan["send_local_indices"].tolist() == []
    torch.testing.assert_close(
        layout.source_partition.remote_owner_compact_offsets,
        torch.tensor([0, 0, 1], dtype=torch.int64),
    )

    layout_dict = layout.as_dict()
    assert layout_dict["source_partition"]["remote_offsets_presorted_by"] == (
        "owner_compact"
    )
    torch.testing.assert_close(
        layout_dict["source_partition"]["remote_owner_compact_offsets"],
        torch.tensor([0, 0, 1], dtype=torch.int64),
    )
    assert layout_dict["owner_compact_exchange_plan"][
        "needed_owner_compact_offsets"
    ] == [0, 1]
    assert layout_dict["owner_compact_exchange_runtime_plan"][
        "needed_row_count"
    ] == 2
    assert layout_dict["owner_compact_exchange_runtime_plan"]["transport"] == (
        "torch_distributed_all_to_all_single"
    )


def _single_rank_exchange_plan():
    return BalancedMoeOwnerCompactExchangePlan(
        rank=0,
        world_size=1,
        active_owner_compact_count=2,
        needed_owner_compact_offsets=(0, 1),
        compact_to_needed_index=(0, 1),
        send_owner_compact_offsets_by_rank=((0, 1),),
        recv_owner_compact_offsets_by_rank=((0, 1),),
        input_splits=(2,),
        output_splits=(2,),
        max_needed_count=2,
        needed_density=1.0,
    )


def test_exchange_owner_compact_needed_rows_single_rank_autograd():
    local_rows = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
        requires_grad=True,
    )
    plan = _single_rank_exchange_plan()

    needed_rows = exchange_owner_compact_needed_rows(
        local_rows,
        compact_local_indices=torch.tensor([1, 0], dtype=torch.int64),
        plan=plan,
    )

    torch.testing.assert_close(
        needed_rows,
        torch.tensor([[20.0, 21.0], [10.0, 11.0]]),
    )
    needed_rows.backward(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torch.testing.assert_close(
        local_rows.grad,
        torch.tensor([[3.0, 4.0], [1.0, 2.0], [0.0, 0.0]]),
    )


def test_exchange_owner_compact_needed_rows_uses_runtime_indices():
    local_rows = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
        requires_grad=True,
    )
    runtime_plan = prepare_owner_compact_needed_rows_runtime_plan(
        compact_local_indices=torch.tensor([1, 0], dtype=torch.int64),
        plan=_single_rank_exchange_plan(),
    )

    needed_rows = exchange_owner_compact_needed_rows(local_rows, plan=runtime_plan)

    torch.testing.assert_close(
        needed_rows,
        torch.tensor([[20.0, 21.0], [10.0, 11.0]]),
    )
    needed_rows.sum().backward()
    torch.testing.assert_close(
        local_rows.grad,
        torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
    )


def test_exchange_owner_compact_needed_rows_primus_turbo_dispatch_single_rank_autograd():
    local_rows = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
        requires_grad=True,
    )
    runtime_plan = prepare_owner_compact_needed_rows_runtime_plan(
        compact_local_indices=torch.tensor([1, 0], dtype=torch.int64),
        plan=_single_rank_exchange_plan(),
        transport="primus_turbo_moe_dispatch",
    )

    needed_rows = exchange_owner_compact_needed_rows(local_rows, plan=runtime_plan)

    torch.testing.assert_close(
        needed_rows,
        torch.tensor([[20.0, 21.0], [10.0, 11.0]]),
    )
    needed_rows.backward(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torch.testing.assert_close(
        local_rows.grad,
        torch.tensor([[3.0, 4.0], [1.0, 2.0], [0.0, 0.0]]),
    )


def test_exchange_owner_compact_needed_rows_accepts_explicit_offsets():
    local_rows = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
        requires_grad=True,
    )
    plan = _single_rank_exchange_plan().as_dict()
    plan.update(
        {
            "input_split_offsets": [0, 2],
            "output_split_offsets": [0, 2],
            "grad_input_split_offsets": [0, 2],
            "grad_output_split_offsets": [0, 2],
        }
    )

    needed_rows = exchange_owner_compact_needed_rows(
        local_rows,
        compact_local_indices=torch.tensor([1, 0], dtype=torch.int64),
        plan=plan,
    )

    torch.testing.assert_close(
        needed_rows,
        torch.tensor([[20.0, 21.0], [10.0, 11.0]]),
    )
    needed_rows.sum().backward()
    torch.testing.assert_close(
        local_rows.grad,
        torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
    )


def test_exchange_owner_compact_needed_rows_rejects_bad_explicit_offsets():
    local_rows = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
        requires_grad=True,
    )
    plan = _single_rank_exchange_plan().as_dict()
    plan.update(
        {
            "input_split_offsets": [0, 1],
            "output_split_offsets": [0, 2],
            "grad_input_split_offsets": [0, 2],
            "grad_output_split_offsets": [0, 2],
        }
    )

    with pytest.raises(ValueError, match="input_split_offsets span"):
        exchange_owner_compact_needed_rows(
            local_rows,
            compact_local_indices=torch.tensor([1, 0], dtype=torch.int64),
            plan=plan,
        )


@pytest.mark.parametrize(
    "transport",
    [
        "mori_sdma_padded_all2all",
        "backend",
        "primus",
        "primus_turbo",
        "deepep",
        "turboep",
    ],
)
def test_exchange_owner_compact_needed_rows_rejects_unsupported_transport(transport):
    local_rows = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
        requires_grad=True,
    )
    runtime_plan = prepare_owner_compact_needed_rows_runtime_plan(
        compact_local_indices=torch.tensor([1, 0], dtype=torch.int64),
        plan=_single_rank_exchange_plan(),
    )
    runtime_plan["transport"] = transport

    with pytest.raises(ValueError, match="Primus-Turbo"):
        exchange_owner_compact_needed_rows(local_rows, plan=runtime_plan)


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


def _distributed_exchange_worker(rank: int, world_size: int, init_file: str):
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


def test_exchange_owner_compact_needed_rows_two_rank_gloo_autograd():
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "balanced_moe_exchange_init")
        mp.start_processes(
            _distributed_exchange_worker,
            args=(world_size, init_file),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )
