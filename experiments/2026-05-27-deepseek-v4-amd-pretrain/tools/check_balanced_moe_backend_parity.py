#!/usr/bin/env python3
"""Check MORI and Primus-Turbo balanced-MoE planner/runtime ABI parity.

This is intentionally CPU-only and source-file based.  It catches drift in the
hot/cold/helper owner-compact contract before remote MI350 throughput gates.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib.util
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

torch: Any | None = None


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKENDS = {
    "mori": EXPERIMENT_ROOT
    / "sources"
    / "references"
    / "mori"
    / "python"
    / "mori"
    / "ops"
    / "balanced_moe.py",
    "primus_turbo": EXPERIMENT_ROOT
    / "sources"
    / "wip"
    / "primus-turbo"
    / "primus_turbo"
    / "pytorch"
    / "ops"
    / "moe"
        / "balanced_moe.py",
}
DEFAULT_TORCHTITAN_MOE = (
    EXPERIMENT_ROOT
    / "sources"
    / "wip"
    / "torchtitan"
    / "torchtitan"
    / "models"
    / "common"
    / "moe.py"
)
DEFAULT_TORCHTITAN_TOKEN_DISPATCHER = (
    EXPERIMENT_ROOT
    / "sources"
    / "wip"
    / "torchtitan"
    / "torchtitan"
    / "models"
    / "common"
    / "token_dispatcher.py"
)

REQUIRED_CAPABILITIES = {
    "abi_version",
    "backend",
    "hot_expert_planner",
    "source_partition",
    "owner_compact_exchange_plan",
    "owner_compact_runtime_layout",
    "owner_compact_exchange_autograd",
    "normal_topk_dispatch_tensors",
    "normal_topk_ep_dispatch",
    "normal_topk_ep_dispatch_backend",
    "normal_topk_ep_dispatch_permute",
    "normal_topk_ep_dispatch_permute_backend",
    "balanced_moe_compact_dispatch",
    "balanced_moe_compact_dispatch_backend",
    "balanced_moe_compact_combine",
    "balanced_moe_compact_combine_backend",
    "balanced_moe_compact_rows_dispatch",
    "balanced_moe_compact_rows_dispatch_backend",
    "balanced_moe_compact_rows_combine",
    "balanced_moe_compact_rows_combine_backend",
    "owner_compact_exchange_transport",
    "owner_compact_exchange_transports",
    "native_hot_helper_transport",
    "native_hot_helper_transport_status",
    "native_owner_compact_exchange_transport",
}

SHARED_CAPABILITY_KEYS = {
    "abi_version",
    "hot_expert_planner",
    "source_partition",
    "owner_compact_exchange_plan",
    "owner_compact_runtime_layout",
    "owner_compact_exchange_autograd",
    "normal_topk_dispatch_tensors",
}

REQUIRED_SHARED_EXPORTS = {
    "BALANCED_MOE_BACKEND_ABI_VERSION",
    "BALANCED_MOE_BACKEND_CAPABILITIES",
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
    "count_local_routes_by_owner_expert",
    "exchange_owner_compact_needed_rows",
    "gather_local_counts_to_global",
    "prepare_owner_compact_needed_rows_runtime_plan",
}

REQUIRED_NORMAL_TOPK_EXPORTS = {
    "BalancedMoeNormalTopKDispatchState",
    "combine_normal_topk_tokens",
    "dispatch_normal_topk_tokens",
    "dispatch_permute_normal_topk_tokens",
    "unpermute_combine_normal_topk_tokens",
}

REQUIRED_COMPACT_EXPORTS = {
    "BalancedMoeCompactCombineOutput",
    "BalancedMoeCompactDispatchOutput",
    "combine_balanced_moe_compact",
    "dispatch_balanced_moe_compact",
}

REQUIRED_NATIVE_COMPACT_ROW_EXPORTS = {
    "combine_balanced_moe_compact_rows",
    "dispatch_balanced_moe_compact_rows",
}


def _load_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    for package_root in (
        path.parents[2] if path.parts[-3:] == ("mori", "ops", "balanced_moe.py") else None,
        path.parents[4]
        if path.parts[-5:]
        == ("primus_turbo", "pytorch", "ops", "moe", "balanced_moe.py")
        else None,
    ):
        if package_root is not None:
            package_root_text = str(package_root)
            if package_root_text not in sys.path:
                sys.path.insert(0, package_root_text)
    spec = importlib.util.spec_from_file_location(f"_balanced_moe_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_torch():
    global torch
    if torch is None:
        try:
            import torch as torch_module
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "check_balanced_moe_backend_parity.py requires PyTorch for "
                "tensor-layout parity checks. Install torch or run this tool "
                "inside the MI350 training/runtime image."
            ) from exc
        torch = torch_module
    return torch


def _counts_owner_source_local() -> torch.Tensor:
    return torch.tensor(
        [
            [[2, 8], [1, 1], [1, 1]],
            [[1, 4], [0, 5], [1, 20]],
            [[2, 1], [12, 1], [2, 1]],
        ],
        dtype=torch.int64,
    )


def _topk_ids() -> torch.Tensor:
    return torch.tensor(
        [
            [3, 0, 1],
            [4, 3, 2],
            [5, 1, 4],
            [0, 3, -1],
        ],
        dtype=torch.int64,
    )


def _selected_routes() -> torch.Tensor:
    return torch.tensor(
        [
            [3, 0],
            [4, 3],
            [1, 4],
            [3, 4],
        ],
        dtype=torch.int64,
    )


def _assert_close(label: str, left: Any, right: Any, path: str = "") -> None:
    torch_module = _require_torch()
    here = f"{label}{path}"
    if isinstance(left, torch_module.Tensor) or isinstance(right, torch_module.Tensor):
        if not isinstance(left, torch_module.Tensor) or not isinstance(right, torch_module.Tensor):
            raise AssertionError(f"{here}: tensor/non-tensor mismatch")
        try:
            torch_module.testing.assert_close(
                left.detach().cpu(),
                right.detach().cpu(),
                check_dtype=True,
                rtol=0,
                atol=0,
            )
        except AssertionError as exc:
            raise AssertionError(f"{here}: {exc}") from exc
        return

    if dataclasses.is_dataclass(left) or dataclasses.is_dataclass(right):
        if not dataclasses.is_dataclass(left) or not dataclasses.is_dataclass(right):
            raise AssertionError(f"{here}: dataclass/non-dataclass mismatch")
        left_fields = {field.name: getattr(left, field.name) for field in dataclasses.fields(left)}
        right_fields = {field.name: getattr(right, field.name) for field in dataclasses.fields(right)}
        _assert_close(label, left_fields, right_fields, path)
        return

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise AssertionError(f"{here}: mapping/non-mapping mismatch")
        if set(left.keys()) != set(right.keys()):
            raise AssertionError(
                f"{here}: key mismatch {sorted(left.keys())} != {sorted(right.keys())}"
            )
        for key in sorted(left.keys(), key=str):
            _assert_close(label, left[key], right[key], f"{path}.{key}")
        return

    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
            raise AssertionError(f"{here}: sequence/non-sequence mismatch")
        if len(left) != len(right):
            raise AssertionError(f"{here}: length mismatch {len(left)} != {len(right)}")
        for idx, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_close(label, left_item, right_item, f"{path}[{idx}]")
        return

    if isinstance(left, float) or isinstance(right, float):
        if abs(float(left) - float(right)) > 1e-9:
            raise AssertionError(f"{here}: float mismatch {left} != {right}")
        return

    if left != right:
        raise AssertionError(f"{here}: value mismatch {left!r} != {right!r}")


def _plan_as_dict(plan: Any) -> dict[str, Any]:
    if not hasattr(plan, "as_dict"):
        raise TypeError("balanced-MoE plan must expose as_dict()")
    return plan.as_dict()


def _capabilities(module: Any) -> dict[str, Any]:
    if not hasattr(module, "balanced_moe_backend_capabilities"):
        raise AssertionError("backend must expose balanced_moe_backend_capabilities()")
    capabilities = module.balanced_moe_backend_capabilities()
    if not isinstance(capabilities, Mapping):
        raise AssertionError("balanced_moe_backend_capabilities() must return a mapping")
    missing = REQUIRED_CAPABILITIES - set(capabilities.keys())
    if missing:
        raise AssertionError(f"backend capabilities missing keys: {sorted(missing)}")
    return dict(capabilities)


def _check_module_exports(
    name: str,
    module: Any,
    capabilities: Mapping[str, Any],
) -> None:
    exported = set(getattr(module, "__all__", ()))
    missing_shared = sorted(REQUIRED_SHARED_EXPORTS - exported)
    if missing_shared:
        raise AssertionError(f"{name}: __all__ missing shared ABI exports {missing_shared}")
    missing_shared_attrs = sorted(
        symbol for symbol in REQUIRED_SHARED_EXPORTS if not hasattr(module, symbol)
    )
    if missing_shared_attrs:
        raise AssertionError(f"{name}: missing shared ABI attributes {missing_shared_attrs}")

    missing_normal_topk_attrs = sorted(
        symbol for symbol in REQUIRED_NORMAL_TOPK_EXPORTS if not hasattr(module, symbol)
    )
    if missing_normal_topk_attrs:
        raise AssertionError(
            f"{name}: missing normal top-k dispatch attributes "
            f"{missing_normal_topk_attrs}"
        )

    if bool(capabilities.get("balanced_moe_compact_rows_native", False)):
        required_compact_exports = (
            REQUIRED_COMPACT_EXPORTS | REQUIRED_NATIVE_COMPACT_ROW_EXPORTS
        )
        missing_compact_exports = sorted(required_compact_exports - exported)
        if missing_compact_exports:
            raise AssertionError(
                f"{name}: balanced_moe_compact_rows_native=True but __all__ misses "
                f"compact balanced-MoE exports {missing_compact_exports}"
            )
        missing_compact_attrs = sorted(
            symbol for symbol in required_compact_exports if not hasattr(module, symbol)
        )
        if missing_compact_attrs:
            raise AssertionError(
                f"{name}: balanced_moe_compact_rows_native=True but module misses "
                f"compact balanced-MoE attributes {missing_compact_attrs}"
            )


def _check_native_compact_delegate_smoke(name: str, module: Any) -> None:
    class FakeOp:
        def dispatch_balanced_moe_compact(
            self,
            input,
            weights,
            scales,
            indices,
            source_partition,
            num_hot_slots=None,
            **kwargs,
        ):
            return module.BalancedMoeCompactDispatchOutput(
                normal_packed_recv_x=input,
                normal_packed_recv_count=torch.tensor([1], dtype=torch.int32),
                normal_packed_recv_src_info=torch.tensor([0], dtype=torch.int32),
                normal_packed_recv_layout_range=torch.tensor([0, 1], dtype=torch.int32),
                hot_packed_x=input,
                hot_packed_scores=weights.reshape(-1),
                hot_packed_src_info=torch.tensor([0], dtype=torch.int32),
                hot_packed_count=None,
                hot_group_ends=None,
                hot_offsets_presorted_by=str(kwargs.get("hot_slot_kind", "")),
            )

        def combine_balanced_moe_compact(
            self,
            expert_major_rows,
            expert_major_flat_positions,
            expert_major_to_rank_major_indices,
            recv_counts_rank_major,
            input_splits,
            output_splits,
            num_output_rows,
            **kwargs,
        ):
            return module.BalancedMoeCompactCombineOutput(
                source_rank_rows=expert_major_rows,
                source_rank_flat_positions=expert_major_flat_positions,
                token_output=None,
            )

        def dispatch_standard_ep_compact_native(
            self,
            local_rows,
            local_flat_positions,
            local_num_tokens_per_expert,
            recv_counts_rank_major,
            input_splits,
            output_splits,
            num_output_rows,
            flat_position_rank_stride,
            **kwargs,
        ):
            if kwargs.get("return_flat_positions", True):
                return local_rows[:num_output_rows], local_flat_positions
            return local_rows[:num_output_rows], None

        def combine_standard_ep_compact_native(
            self,
            expert_major_rows,
            expert_major_flat_positions,
            expert_major_to_rank_major_indices,
            recv_counts_rank_major,
            input_splits,
            output_splits,
            num_output_rows,
            **kwargs,
        ):
            source_rank_rows = expert_major_rows[:num_output_rows]
            source_rank_flat_positions = (
                expert_major_flat_positions[:num_output_rows]
                if expert_major_flat_positions is not None
                else None
            )
            if kwargs.get("return_token_output", False):
                return source_rank_rows, source_rank_flat_positions, expert_major_rows
            return source_rank_rows, source_rank_flat_positions

    x = torch.zeros(1, 2)
    weights = torch.ones(1, 1)
    dispatch_output = module.dispatch_balanced_moe_compact(
        FakeOp(),
        x,
        weights,
        torch.ones(1, 1),
        torch.zeros(1, 1, dtype=torch.int64),
        source_partition=object(),
    )
    if dispatch_output.hot_offsets_presorted_by != "owner_compact":
        raise AssertionError(f"{name}: compact dispatch delegate did not preserve kwargs")

    combine_output = module.combine_balanced_moe_compact(
        FakeOp(),
        x,
        torch.tensor([0], dtype=torch.int64),
        None,
        None,
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
        1,
    )
    _assert_close(name, combine_output.source_rank_rows, x, ".compact_delegate.rows")

    row_dispatch, row_positions = module.dispatch_balanced_moe_compact_rows(
        FakeOp(),
        x,
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
        1,
        1,
    )
    _assert_close(name, row_dispatch, x, ".compact_rows.dispatch")
    _assert_close(
        name,
        row_positions,
        torch.tensor([0], dtype=torch.int64),
        ".compact_rows.dispatch_positions",
    )

    row_combine, row_combine_positions = module.combine_balanced_moe_compact_rows(
        FakeOp(),
        x,
        torch.tensor([0], dtype=torch.int64),
        None,
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
        1,
    )
    _assert_close(name, row_combine, x, ".compact_rows.combine")
    _assert_close(
        name,
        row_combine_positions,
        torch.tensor([0], dtype=torch.int64),
        ".compact_rows.combine_positions",
    )

    try:
        module.dispatch_balanced_moe_compact(
            object(),
            x,
            weights,
            torch.ones(1, 1),
            torch.zeros(1, 1, dtype=torch.int64),
            source_partition=object(),
        )
    except TypeError as exc:
        if "dispatch_balanced_moe_compact" not in str(exc):
            raise AssertionError(
                f"{name}: missing-op dispatch error was unclear: {exc}"
            ) from exc
    else:
        raise AssertionError(f"{name}: compact dispatch accepted object without op method")


def _load_torchtitan_selector(torchtitan_moe_path: Path):
    source = torchtitan_moe_path.read_text()
    tree = ast.parse(source, filename=str(torchtitan_moe_path))
    selector_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_standard_ep_balanced_moe_module_matches_impl"
    ]
    if len(selector_nodes) != 1:
        raise AssertionError(
            "TorchTitan moe.py must define exactly one "
            "_standard_ep_balanced_moe_module_matches_impl() helper"
        )
    module_ast = ast.Module(body=selector_nodes, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    namespace: dict[str, Any] = {}
    exec(compile(module_ast, str(torchtitan_moe_path), "exec"), namespace)
    return namespace["_standard_ep_balanced_moe_module_matches_impl"]


def _load_torchtitan_needed_weight_transport_parser(torchtitan_moe_path: Path):
    source = torchtitan_moe_path.read_text()
    tree = ast.parse(source, filename=str(torchtitan_moe_path))
    parser_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_standard_ep_needed_weight_exchange_transport"
    ]
    if len(parser_nodes) != 1:
        raise AssertionError(
            "TorchTitan moe.py must define exactly one "
            "_standard_ep_needed_weight_exchange_transport() helper"
        )
    module_ast = ast.Module(body=parser_nodes, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    namespace: dict[str, Any] = {"os": os}
    exec(compile(module_ast, str(torchtitan_moe_path), "exec"), namespace)
    return namespace["_standard_ep_needed_weight_exchange_transport"]


def _load_torchtitan_capability_selector(torchtitan_token_dispatcher_path: Path):
    source = torchtitan_token_dispatcher_path.read_text()
    tree = ast.parse(source, filename=str(torchtitan_token_dispatcher_path))
    selector_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_standard_ep_balanced_moe_module_supports_capability"
    ]
    if len(selector_nodes) != 1:
        raise AssertionError(
            "TorchTitan token_dispatcher.py must define exactly one "
            "_standard_ep_balanced_moe_module_supports_capability() helper"
        )
    module_ast = ast.Module(body=selector_nodes, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    namespace: dict[str, Any] = {}
    exec(compile(module_ast, str(torchtitan_token_dispatcher_path), "exec"), namespace)
    return namespace["_standard_ep_balanced_moe_module_supports_capability"]


def _check_torchtitan_token_dispatcher_compact_import_boundary(
    torchtitan_token_dispatcher_path: Path,
) -> None:
    source = torchtitan_token_dispatcher_path.read_text()
    required = (
        "_standard_ep_mori_balanced_moe_module",
        "dispatch_balanced_moe_compact_rows",
        "combine_balanced_moe_compact_rows",
    )
    missing = [needle for needle in required if needle not in source]
    if missing:
        raise AssertionError(
            "TorchTitan native compact path must consume mori.ops.balanced_moe "
            f"compact-row wrappers; missing source markers {missing}"
        )


def _check_torchtitan_backend_selector(
    torchtitan_moe_path: Path,
    modules: Mapping[str, Any],
) -> None:
    selector = _load_torchtitan_selector(torchtitan_moe_path)
    transport_parser = _load_torchtitan_needed_weight_transport_parser(
        torchtitan_moe_path
    )
    mori_module = modules["mori"]
    primus_module = modules["primus_turbo"]

    for exchange_impl in ("auto", "backend"):
        if not selector("mori.ops.balanced_moe", mori_module, exchange_impl):
            raise AssertionError(f"selector rejected MORI for {exchange_impl}")
        if not selector(
            "primus_turbo.pytorch.ops.moe.balanced_moe",
            primus_module,
            exchange_impl,
        ):
            raise AssertionError(f"selector rejected Primus-Turbo for {exchange_impl}")

    if not selector("mori.ops.balanced_moe", mori_module, "native"):
        raise AssertionError("selector rejected MORI for native")
    if not selector(
        "primus_turbo.pytorch.ops.moe.balanced_moe",
        primus_module,
        "native",
    ):
        raise AssertionError(
            "selector rejected Primus-Turbo for native hot-helper transport "
            "even though the backend advertises native_hot_helper_transport=True"
        )

    if not selector("mori.ops.balanced_moe", mori_module, "mori"):
        raise AssertionError("selector rejected MORI for explicit mori")
    if selector(
        "primus_turbo.pytorch.ops.moe.balanced_moe",
        primus_module,
        "mori",
    ):
        raise AssertionError("selector accepted Primus-Turbo for explicit mori")

    for exchange_impl in ("primus", "primus_turbo"):
        if selector("mori.ops.balanced_moe", mori_module, exchange_impl):
            raise AssertionError(f"selector accepted MORI for explicit {exchange_impl}")
        if not selector(
            "primus_turbo.pytorch.ops.moe.balanced_moe",
            primus_module,
            exchange_impl,
        ):
            raise AssertionError(
                f"selector rejected Primus-Turbo for explicit {exchange_impl}"
            )

    env_name = "TORCHTITAN_STANDARD_EP_NEEDED_WEIGHT_EXCHANGE_TRANSPORT"
    canary_name = "CANARY_STANDARD_EP_NEEDED_WEIGHT_EXCHANGE_TRANSPORT"
    old_env = os.environ.get(env_name)
    old_canary = os.environ.get(canary_name)
    try:
        os.environ.pop(canary_name, None)
        expected_transports = {
            "torch": "torch_distributed_all_to_all_single",
            "all_to_all_single": "torch_distributed_all_to_all_single",
            "native": "native",
            "primus_turbo_moe_dispatch": "primus_turbo_moe_dispatch",
            "mori_sdma": "mori_sdma_padded_all2all",
        }
        for alias, expected in expected_transports.items():
            os.environ[env_name] = alias
            parsed = transport_parser()
            if parsed != expected:
                raise AssertionError(
                    "TorchTitan needed-weight transport parser normalized "
                    f"{alias!r} to {parsed!r}, expected {expected!r}"
                )
    finally:
        if old_env is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = old_env
        if old_canary is None:
            os.environ.pop(canary_name, None)
        else:
            os.environ[canary_name] = old_canary


def _check_torchtitan_backend_capability_selector(
    torchtitan_token_dispatcher_path: Path,
    modules: Mapping[str, Any],
) -> None:
    selector = _load_torchtitan_capability_selector(torchtitan_token_dispatcher_path)
    mori_module = modules["mori"]
    primus_module = modules["primus_turbo"]

    shared_capabilities = (
        "hot_expert_planner",
        "source_partition",
        "owner_compact_exchange_plan",
        "owner_compact_runtime_layout",
        "owner_compact_exchange_autograd",
        "normal_topk_dispatch_tensors",
    )
    for capability in shared_capabilities:
        if not selector("mori.ops.balanced_moe", mori_module, capability):
            raise AssertionError(f"selector rejected MORI {capability}")
        if not selector(
            "primus_turbo.pytorch.ops.moe.balanced_moe",
            primus_module,
            capability,
        ):
            raise AssertionError(f"selector rejected Primus-Turbo {capability}")

    if selector(
        "mori.ops.balanced_moe",
        mori_module,
        "normal_topk_ep_dispatch_permute",
    ):
        raise AssertionError(
            "selector accepted MORI for Primus-Turbo raw top-k dispatch+permute"
        )
    if not selector(
        "primus_turbo.pytorch.ops.moe.balanced_moe",
        primus_module,
        "normal_topk_ep_dispatch_permute",
    ):
        raise AssertionError(
            "selector rejected Primus-Turbo for raw top-k dispatch+permute"
        )


def _check_torchtitan_backend_dispatch_hooks(
    torchtitan_token_dispatcher_path: Path,
) -> None:
    source = torchtitan_token_dispatcher_path.read_text()
    tree = ast.parse(source, filename=str(torchtitan_token_dispatcher_path))

    top_level_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    if "_standard_ep_backend_normal_topk_dispatch_permute" not in top_level_functions:
        raise AssertionError(
            "TorchTitan token_dispatcher.py must expose the opt-in "
            "_standard_ep_backend_normal_topk_dispatch_permute() gate"
        )

    metadata_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "AllToAllDispatchMetadata"
    ]
    if len(metadata_nodes) != 1:
        raise AssertionError(
            "TorchTitan token_dispatcher.py must define exactly one "
            "AllToAllDispatchMetadata class"
        )
    metadata_fields = {
        node.target.id
        for node in metadata_nodes[0].body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    }
    if "normal_topk_backend_state" not in metadata_fields:
        raise AssertionError(
            "AllToAllDispatchMetadata must carry normal_topk_backend_state so "
            "backend dispatch returns through the matching backend combine"
        )

    dispatcher_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "AllToAllTokenDispatcher"
    ]
    if len(dispatcher_nodes) != 1:
        raise AssertionError(
            "TorchTitan token_dispatcher.py must define exactly one "
            "AllToAllTokenDispatcher class"
        )
    dispatcher_methods = {
        node.name
        for node in dispatcher_nodes[0].body
        if isinstance(node, ast.FunctionDef)
    }
    if "_backend_normal_topk_unpermute_combine" not in dispatcher_methods:
        raise AssertionError(
            "AllToAllTokenDispatcher must route backend normal-topk output "
            "through a backend-owned unpermute/combine helper"
        )

    required_source_markers = (
        "TORCHTITAN_STANDARD_EP_BACKEND_NORMAL_TOPK_DISPATCH_PERMUTE",
        "normal_topk_ep_dispatch_permute",
        "dispatch_permute_normal_topk_tokens",
        "unpermute_combine_normal_topk_tokens",
        "backend_normal_topk_dispatch_permute_active",
    )
    for marker in required_source_markers:
        if marker not in source:
            raise AssertionError(
                f"TorchTitan token_dispatcher.py is missing backend dispatch "
                f"marker {marker!r}"
            )


def _check_single_rank_exchange(label: str, module) -> tuple[torch.Tensor, torch.Tensor]:
    plan = module.build_owner_compact_exchange_plan(
        owner_compact_need_masks=((True, True),),
        owner_compact_owner_ranks=(0, 0),
        rank=0,
    )
    local_rows = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
        requires_grad=True,
    )
    runtime_plan = module.prepare_owner_compact_needed_rows_runtime_plan(
        compact_local_indices=torch.tensor([1, 0], dtype=torch.int64),
        plan=plan,
    )
    needed_rows = module.exchange_owner_compact_needed_rows(
        local_rows,
        plan=runtime_plan,
    )
    needed_rows.backward(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    if local_rows.grad is None:
        raise AssertionError(f"{label}: missing local_rows grad")
    return needed_rows.detach(), local_rows.grad.detach()


def _check_explicit_offset_exchange(
    label: str,
    module,
) -> tuple[torch.Tensor, torch.Tensor]:
    exchange_plan = module.build_owner_compact_exchange_plan(
        owner_compact_need_masks=((True, True),),
        owner_compact_owner_ranks=(0, 0),
        rank=0,
    )
    local_rows = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
        requires_grad=True,
    )
    needed_rows = module.exchange_owner_compact_needed_rows(
        local_rows,
        compact_local_indices=torch.tensor([1, 0], dtype=torch.int64),
        needed_owner_compact_offsets=exchange_plan.needed_owner_compact_offsets,
        send_owner_compact_offsets=exchange_plan.send_owner_compact_offsets,
        recv_to_needed_index=exchange_plan.recv_to_needed_index,
        grad_send_needed_indices=exchange_plan.grad_send_needed_indices,
        grad_recv_owner_compact_offsets=exchange_plan.grad_recv_owner_compact_offsets,
        input_splits=exchange_plan.input_splits,
        output_splits=exchange_plan.output_splits,
        input_split_offsets=exchange_plan.input_split_offsets,
        output_split_offsets=exchange_plan.output_split_offsets,
        grad_input_split_offsets=exchange_plan.grad_input_split_offsets,
        grad_output_split_offsets=exchange_plan.grad_output_split_offsets,
    )
    needed_rows.backward(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    if local_rows.grad is None:
        raise AssertionError(f"{label}: missing local_rows grad")
    return needed_rows.detach(), local_rows.grad.detach()


def _check_bad_explicit_offset_rejection(label: str, module) -> None:
    exchange_plan = module.build_owner_compact_exchange_plan(
        owner_compact_need_masks=((True, True),),
        owner_compact_owner_ranks=(0, 0),
        rank=0,
    )
    local_rows = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    try:
        module.exchange_owner_compact_needed_rows(
            local_rows,
            compact_local_indices=torch.tensor([1, 0], dtype=torch.int64),
            needed_owner_compact_offsets=exchange_plan.needed_owner_compact_offsets,
            send_owner_compact_offsets=exchange_plan.send_owner_compact_offsets,
            recv_to_needed_index=exchange_plan.recv_to_needed_index,
            grad_send_needed_indices=exchange_plan.grad_send_needed_indices,
            grad_recv_owner_compact_offsets=exchange_plan.grad_recv_owner_compact_offsets,
            input_splits=exchange_plan.input_splits,
            output_splits=exchange_plan.output_splits,
            input_split_offsets=(0, 1),
            output_split_offsets=exchange_plan.output_split_offsets,
            grad_input_split_offsets=exchange_plan.grad_input_split_offsets,
            grad_output_split_offsets=exchange_plan.grad_output_split_offsets,
        )
    except ValueError as exc:
        if "input_split_offsets" not in str(exc):
            raise AssertionError(
                f"{label}: bad explicit offset rejected with unexpected message: {exc}"
            ) from exc
        return
    raise AssertionError(f"{label}: bad explicit input_split_offsets were accepted")


def _check_backend_transport_contracts(
    modules: Mapping[str, Any],
    capabilities_by_name: Mapping[str, Mapping[str, Any]],
) -> None:
    """Keep backend-specific transport claims explicit.

    The planner/runtime layout should stay byte-for-byte comparable across
    MORI and Primus-Turbo.  Primus owns the hot-helper ABI while MORI owns the
    SDMA transport primitive, so both backends must now normalize the SDMA
    aliases and avoid falling back to torch when native transport is requested.
    """

    torch_aliases = (
        None,
        "",
        "auto",
        "torch",
        "all_to_all_single",
        "torch_distributed_all_to_all_single",
    )
    mori_sdma_aliases = (
        "mori_sdma",
        "mori_sdma_padded",
        "mori_sdma_padded_all2all",
    )
    native_transport_aliases_by_backend = {
        "mori": (
            "native",
            "native_hot_helper",
            "mori_native",
        ),
        "primus_turbo": (
            "native",
            "native_hot_helper",
            "mori_native",
        ),
    }
    owner_compact_not_transport_aliases = (
        "backend",
        "primus",
        "primus_turbo",
        "deepep",
        "turboep",
    )
    expected_extra_transports = {
        "mori": ("mori_sdma_padded_all2all",),
        "primus_turbo": (
            "primus_turbo_moe_dispatch",
            "mori_sdma_padded_all2all",
        ),
    }
    expected_native_transport = {
        "mori": True,
        "primus_turbo": True,
    }

    for name, module in modules.items():
        capabilities = capabilities_by_name[name]
        normalizer = getattr(
            module,
            "_normalize_owner_compact_exchange_transport",
            None,
        )
        if normalizer is None:
            raise AssertionError(
                f"{name}: backend must expose _normalize_owner_compact_exchange_transport"
            )

        advertised = tuple(capabilities["owner_compact_exchange_transports"])
        expected = ("torch_distributed_all_to_all_single",) + expected_extra_transports[name]
        if advertised != expected:
            raise AssertionError(
                f"{name}: transport advertisement drifted: {advertised!r} != {expected!r}"
            )
        if bool(capabilities["native_hot_helper_transport"]) is not expected_native_transport[name]:
            raise AssertionError(
                f"{name}: native_hot_helper_transport must be "
                f"{expected_native_transport[name]!r}"
            )

        for alias in torch_aliases:
            normalized = normalizer(alias)
            if normalized != "torch_distributed_all_to_all_single":
                raise AssertionError(
                    f"{name}: torch alias {alias!r} normalized to {normalized!r}"
                )

        for alias in mori_sdma_aliases:
            if name in {"mori", "primus_turbo"}:
                normalized = normalizer(alias)
                if normalized != "mori_sdma_padded_all2all":
                    raise AssertionError(
                        f"{name}: MORI SDMA alias {alias!r} normalized to {normalized!r}"
                    )
            else:
                try:
                    normalizer(alias)
                except ValueError as exc:
                    if alias not in str(exc) or "unsupported" not in str(exc):
                        raise AssertionError(
                            f"{name}: unsupported transport {alias!r} had unclear "
                            f"error: {exc}"
                        ) from exc
                else:
                    raise AssertionError(
                        f"{name}: unsupported MORI SDMA transport {alias!r} was accepted"
                    )

        for alias in native_transport_aliases_by_backend[name]:
            native_transport = capabilities["native_owner_compact_exchange_transport"]
            if native_transport is not None:
                normalized = normalizer(alias)
                if normalized != native_transport:
                    raise AssertionError(
                        f"{name}: native alias {alias!r} normalized to "
                        f"{normalized!r}, expected {native_transport!r}"
                    )
            else:
                try:
                    normalizer(alias)
                except ValueError as exc:
                    if alias not in str(exc) or "unsupported" not in str(exc):
                        raise AssertionError(
                            f"{name}: unsupported native transport {alias!r} "
                            f"had unclear error: {exc}"
                        ) from exc
                else:
                    raise AssertionError(
                        f"{name}: native transport alias {alias!r} was accepted "
                        "without native_owner_compact_exchange_transport"
                    )

        if name not in {"mori", "primus_turbo"}:
            try:
                normalizer("mori_native")
            except ValueError as exc:
                if "mori_native" not in str(exc) or "unsupported" not in str(exc):
                    raise AssertionError(
                        f"{name}: MORI-specific alias had unclear error: {exc}"
                    ) from exc
            else:
                raise AssertionError(
                    f"{name}: MORI-specific alias was accepted as native transport"
                )

        for alias in owner_compact_not_transport_aliases:
            try:
                normalizer(alias)
            except ValueError as exc:
                message = str(exc)
                if alias not in message or "unsupported" not in message:
                    raise AssertionError(
                        f"{name}: non-transport alias {alias!r} had unclear "
                        f"error: {exc}"
                    ) from exc
            else:
                raise AssertionError(
                    f"{name}: non-transport alias {alias!r} was accepted as an "
                    "owner-compact hot-row transport"
                )


def _check_normal_topk_ep_dispatch_contract(
    modules: Mapping[str, Any],
    capabilities_by_name: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_dispatch = {
        "mori": False,
        "primus_turbo": True,
    }
    expected_dispatch_permute = {
        "mori": False,
        "primus_turbo": True,
    }
    expected_backend = {
        "mori": "not_implemented",
        "primus_turbo": "primus_turbo_moe_dispatch",
    }
    expected_permute_backend = {
        "mori": "not_implemented",
        "primus_turbo": "primus_turbo_moe_dispatch_permute",
    }
    expected_compact_rows = {
        "mori": True,
        "primus_turbo": False,
    }
    expected_compact_rows_backend = {
        "mori": "mori_ep_dispatch_combine_op",
        "primus_turbo": "not_implemented",
    }
    required_symbols = (
        "BalancedMoeNormalTopKDispatchState",
        "dispatch_normal_topk_tokens",
        "dispatch_permute_normal_topk_tokens",
        "combine_normal_topk_tokens",
        "unpermute_combine_normal_topk_tokens",
    )

    for name, module in modules.items():
        capabilities = capabilities_by_name[name]
        for symbol in required_symbols:
            if not hasattr(module, symbol):
                raise AssertionError(f"{name}: missing {symbol}")
        if bool(capabilities["normal_topk_ep_dispatch"]) is not expected_dispatch[name]:
            raise AssertionError(
                f"{name}: normal_topk_ep_dispatch must be "
                f"{expected_dispatch[name]!r}"
            )
        if capabilities["normal_topk_ep_dispatch_backend"] != expected_backend[name]:
            raise AssertionError(
                f"{name}: normal_topk_ep_dispatch_backend drifted: "
                f"{capabilities['normal_topk_ep_dispatch_backend']!r}"
            )
        if bool(capabilities["normal_topk_ep_dispatch_permute"]) is not expected_dispatch_permute[name]:
            raise AssertionError(
                f"{name}: normal_topk_ep_dispatch_permute must be "
                f"{expected_dispatch_permute[name]!r}"
            )
        if (
            capabilities["normal_topk_ep_dispatch_permute_backend"]
            != expected_permute_backend[name]
        ):
            raise AssertionError(
                f"{name}: normal_topk_ep_dispatch_permute_backend drifted: "
                f"{capabilities['normal_topk_ep_dispatch_permute_backend']!r}"
            )
        if bool(capabilities["balanced_moe_compact_rows_dispatch"]) is not expected_compact_rows[name]:
            raise AssertionError(
                f"{name}: balanced_moe_compact_rows_dispatch must be "
                f"{expected_compact_rows[name]!r}"
            )
        if bool(capabilities["balanced_moe_compact_rows_combine"]) is not expected_compact_rows[name]:
            raise AssertionError(
                f"{name}: balanced_moe_compact_rows_combine must be "
                f"{expected_compact_rows[name]!r}"
            )
        if (
            capabilities["balanced_moe_compact_rows_dispatch_backend"]
            != expected_compact_rows_backend[name]
        ):
            raise AssertionError(
                f"{name}: balanced_moe_compact_rows_dispatch_backend drifted: "
                f"{capabilities['balanced_moe_compact_rows_dispatch_backend']!r}"
            )
        if (
            capabilities["balanced_moe_compact_rows_combine_backend"]
            != expected_compact_rows_backend[name]
        ):
            raise AssertionError(
                f"{name}: balanced_moe_compact_rows_combine_backend drifted: "
                f"{capabilities['balanced_moe_compact_rows_combine_backend']!r}"
            )

    mori_module = modules["mori"]
    try:
        mori_module.dispatch_normal_topk_tokens(
            torch.zeros(1, 2),
            torch.zeros(1, 1, dtype=torch.int64),
            torch.zeros(1, 1, dtype=torch.float32),
            num_experts=1,
            group=None,
        )
    except NotImplementedError as exc:
        if "raw top-k" not in str(exc):
            raise AssertionError(
                "MORI raw top-k dispatch rejection had unclear error: "
                f"{exc}"
            ) from exc
    else:
        raise AssertionError("MORI unexpectedly accepted raw top-k dispatch")

    try:
        mori_module.dispatch_permute_normal_topk_tokens(
            torch.zeros(1, 2),
            torch.zeros(1, 1, dtype=torch.int64),
            torch.zeros(1, 1, dtype=torch.float32),
            num_experts=1,
            num_local_experts=1,
            group=None,
        )
    except NotImplementedError as exc:
        if "raw top-k" not in str(exc):
            raise AssertionError(
                "MORI raw top-k dispatch+permute rejection had unclear error: "
                f"{exc}"
            ) from exc
    else:
        raise AssertionError(
            "MORI unexpectedly accepted raw top-k dispatch+permute"
        )


def _check_normal_topk_dispatch_tensors(
    label: str,
    module,
    selected: torch.Tensor,
    partition,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.arange(
        selected.numel(),
        dtype=torch.float32,
    ).reshape_as(selected)
    normal_ids, normal_weights = module.build_normal_topk_dispatch_tensors(
        selected,
        weights,
        partition=partition,
    )
    expected_mask = partition.normal_route_mask.to(torch.bool)
    expected_ids = torch.where(
        expected_mask,
        selected,
        selected.new_full(selected.shape, -1),
    )
    expected_weights = torch.where(
        expected_mask,
        weights,
        weights.new_zeros(weights.shape),
    )
    _assert_close(label, normal_ids, expected_ids, ".normal_topk.ids")
    _assert_close(label, normal_weights, expected_weights, ".normal_topk.weights")

    ids_without_weights, missing_weights = module.build_normal_topk_dispatch_tensors(
        selected,
        partition=partition,
    )
    if missing_weights is not None:
        raise AssertionError(f"{label}.normal_topk: missing weights must stay None")
    _assert_close(label, ids_without_weights, expected_ids, ".normal_topk.ids_only")

    try:
        module.build_normal_topk_dispatch_tensors(
            selected,
            normal_route_mask=torch.ones(selected.numel(), dtype=torch.uint8),
        )
    except ValueError as exc:
        if "same shape" not in str(exc):
            raise AssertionError(
                f"{label}.normal_topk: bad mask shape had unclear error: {exc}"
            ) from exc
    else:
        raise AssertionError(f"{label}.normal_topk: bad mask shape was accepted")

    return normal_ids, normal_weights


def check_parity(
    backends: Mapping[str, Path],
    *,
    torchtitan_moe_path: Path,
    torchtitan_token_dispatcher_path: Path,
) -> None:
    _require_torch()
    modules = {name: _load_module(name, path) for name, path in backends.items()}
    if len(modules) != 2:
        raise ValueError("this parity tool expects exactly two backends")
    (left_name, left_module), (right_name, right_module) = tuple(modules.items())
    label = f"{left_name} vs {right_name}"

    left_capabilities = _capabilities(left_module)
    right_capabilities = _capabilities(right_module)
    capabilities_by_name = {
        left_name: left_capabilities,
        right_name: right_capabilities,
    }
    _check_module_exports(left_name, left_module, left_capabilities)
    _check_module_exports(right_name, right_module, right_capabilities)
    _check_torchtitan_token_dispatcher_compact_import_boundary(
        torchtitan_token_dispatcher_path
    )
    for name, module in modules.items():
        if bool(capabilities_by_name[name].get("balanced_moe_compact_rows_native", False)):
            _check_native_compact_delegate_smoke(name, module)
    if left_capabilities["backend"] == right_capabilities["backend"]:
        raise AssertionError(
            f"{label}.capabilities.backend: expected different backend names"
        )
    left_common = {key: left_capabilities[key] for key in SHARED_CAPABILITY_KEYS}
    right_common = {key: right_capabilities[key] for key in SHARED_CAPABILITY_KEYS}
    _assert_close(label, left_common, right_common, ".capabilities")
    for name, capabilities in (
        (left_name, left_capabilities),
        (right_name, right_capabilities),
    ):
        transports = tuple(capabilities["owner_compact_exchange_transports"])
        default_transport = capabilities["owner_compact_exchange_transport"]
        if default_transport not in transports:
            raise AssertionError(
                f"{name}.capabilities.owner_compact_exchange_transport "
                f"{default_transport!r} is not in transports {transports!r}"
            )
    _check_backend_transport_contracts(modules, capabilities_by_name)
    _check_normal_topk_ep_dispatch_contract(modules, capabilities_by_name)

    counts = _counts_owner_source_local()
    left_plan = left_module.build_balanced_moe_plan_from_global_counts(
        counts,
        hot_expert_num=2,
    )
    right_plan = right_module.build_balanced_moe_plan_from_global_counts(
        counts,
        hot_expert_num=2,
    )
    _assert_close(label, _plan_as_dict(left_plan), _plan_as_dict(right_plan), ".plan")

    left_reject = left_module.build_balanced_moe_plan_from_global_counts(
        counts,
        hot_expert_num=2,
        min_reduction_pct=99.0,
    )
    right_reject = right_module.build_balanced_moe_plan_from_global_counts(
        counts,
        hot_expert_num=2,
        min_reduction_pct=99.0,
    )
    _assert_close(
        label,
        _plan_as_dict(left_reject),
        _plan_as_dict(right_reject),
        ".rejected_plan",
    )

    left_counts = left_module.count_local_routes_by_owner_expert(
        _topk_ids(),
        num_local_experts=2,
        world_size=3,
    )
    right_counts = right_module.count_local_routes_by_owner_expert(
        _topk_ids(),
        num_local_experts=2,
        world_size=3,
    )
    _assert_close(label, left_counts, right_counts, ".local_counts")

    selected = _selected_routes()
    for rank in range(3):
        for presort_by in ("off", "selected", "owner_compact"):
            left_partition = left_module.build_source_partition(
                selected,
                left_plan,
                ep_rank=rank,
                presort_by=presort_by,
            )
            right_partition = right_module.build_source_partition(
                selected,
                right_plan,
                ep_rank=rank,
                presort_by=presort_by,
            )
            _assert_close(
                label,
                left_partition,
                right_partition,
                f".source_partition.rank{rank}.{presort_by}",
            )
            _assert_close(
                label,
                left_partition.as_dict(),
                right_partition.as_dict(),
                f".source_partition_dict.rank{rank}.{presort_by}",
            )
            if presort_by == "owner_compact":
                left_normal_ids, left_normal_weights = (
                    _check_normal_topk_dispatch_tensors(
                        left_name,
                        left_module,
                        selected,
                        left_partition,
                    )
                )
                right_normal_ids, right_normal_weights = (
                    _check_normal_topk_dispatch_tensors(
                        right_name,
                        right_module,
                        selected,
                        right_partition,
                    )
                )
                _assert_close(
                    label,
                    left_normal_ids,
                    right_normal_ids,
                    f".normal_topk.rank{rank}.ids",
                )
                _assert_close(
                    label,
                    left_normal_weights,
                    right_normal_weights,
                    f".normal_topk.rank{rank}.weights",
                )

    compact_local_indices = torch.tensor([11, 13], dtype=torch.int64)
    for rank in range(3):
        left_exchange, left_runtime = (
            left_module.build_owner_compact_exchange_runtime_plan_from_plan(
                left_plan,
                rank=rank,
                compact_local_indices=compact_local_indices,
                dtype=torch.int32,
                split_dtype=torch.int16,
            )
        )
        right_exchange, right_runtime = (
            right_module.build_owner_compact_exchange_runtime_plan_from_plan(
                right_plan,
                rank=rank,
                compact_local_indices=compact_local_indices,
                dtype=torch.int32,
                split_dtype=torch.int16,
            )
        )
        _assert_close(label, left_exchange, right_exchange, f".exchange.rank{rank}")
        _assert_close(label, left_runtime, right_runtime, f".runtime.rank{rank}")

        left_layout = left_module.build_balanced_moe_runtime_layout(
            selected,
            left_plan,
            ep_rank=rank,
            compact_local_indices=compact_local_indices,
            presort_by="owner_compact",
            dtype=torch.int32,
            split_dtype=torch.int16,
        )
        right_layout = right_module.build_balanced_moe_runtime_layout(
            selected,
            right_plan,
            ep_rank=rank,
            compact_local_indices=compact_local_indices,
            presort_by="owner_compact",
            dtype=torch.int32,
            split_dtype=torch.int16,
        )
        _assert_close(label, left_layout, right_layout, f".runtime_layout.rank{rank}")
        _assert_close(
            label,
            left_layout.as_dict(),
            right_layout.as_dict(),
            f".runtime_layout_dict.rank{rank}",
        )

    left_needed, left_grad = _check_single_rank_exchange(left_name, left_module)
    right_needed, right_grad = _check_single_rank_exchange(right_name, right_module)
    _assert_close(label, left_needed, right_needed, ".single_rank_exchange.output")
    _assert_close(label, left_grad, right_grad, ".single_rank_exchange.grad")

    left_needed, left_grad = _check_explicit_offset_exchange(left_name, left_module)
    right_needed, right_grad = _check_explicit_offset_exchange(
        right_name,
        right_module,
    )
    _assert_close(label, left_needed, right_needed, ".explicit_offset_exchange.output")
    _assert_close(label, left_grad, right_grad, ".explicit_offset_exchange.grad")
    _check_bad_explicit_offset_rejection(left_name, left_module)
    _check_bad_explicit_offset_rejection(right_name, right_module)
    _check_torchtitan_backend_selector(torchtitan_moe_path, modules)
    _check_torchtitan_backend_capability_selector(
        torchtitan_token_dispatcher_path,
        modules,
    )
    _check_torchtitan_backend_dispatch_hooks(torchtitan_token_dispatcher_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mori",
        type=Path,
        default=DEFAULT_BACKENDS["mori"],
        help="Path to MORI balanced_moe.py",
    )
    parser.add_argument(
        "--primus-turbo",
        type=Path,
        default=DEFAULT_BACKENDS["primus_turbo"],
        help="Path to Primus-Turbo balanced_moe.py",
    )
    parser.add_argument(
        "--torchtitan-moe",
        type=Path,
        default=DEFAULT_TORCHTITAN_MOE,
        help="Path to TorchTitan common/moe.py for policy selector checks",
    )
    parser.add_argument(
        "--torchtitan-token-dispatcher",
        type=Path,
        default=DEFAULT_TORCHTITAN_TOKEN_DISPATCHER,
        help=(
            "Path to TorchTitan common/token_dispatcher.py for backend "
            "capability selector checks"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backends = {
        "mori": args.mori.resolve(),
        "primus_turbo": args.primus_turbo.resolve(),
    }
    check_parity(
        backends,
        torchtitan_moe_path=args.torchtitan_moe.resolve(),
        torchtitan_token_dispatcher_path=(
            args.torchtitan_token_dispatcher.resolve()
        ),
    )
    print("balanced-MoE backend parity OK")
    for name, path in backends.items():
        print(f"  {name}: {path}")
    print(f"  torchtitan_moe: {args.torchtitan_moe.resolve()}")
    print(f"  torchtitan_token_dispatcher: {args.torchtitan_token_dispatcher.resolve()}")


if __name__ == "__main__":
    main()
