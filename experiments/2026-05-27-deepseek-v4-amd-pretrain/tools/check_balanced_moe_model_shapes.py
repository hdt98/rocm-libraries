#!/usr/bin/env python3
"""Check balanced-MoE backend ABI coverage for the required model shapes.

This is not a throughput benchmark.  It builds deterministic imbalanced routed
top-k tensors for the model families named in the goal and verifies that MORI
and Primus-Turbo produce the same native balanced-MoE plan, partition, exchange
metadata, and runtime layout.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import torch

from check_balanced_moe_backend_parity import (
    DEFAULT_BACKENDS,
    _assert_close,
    _load_module,
    _plan_as_dict,
)


@dataclasses.dataclass(frozen=True)
class ModelShape:
    name: str
    reduced_layers: int
    total_experts: int
    routed_top_k: int
    shared_experts: int
    hidden_size: int
    moe_intermediate_size: int
    attention: str
    notes: str


MODEL_SHAPES = (
    ModelShape(
        name="deepseek_v4_flash_12layer_ceiling_probe",
        reduced_layers=12,
        total_experts=256,
        routed_top_k=6,
        shared_experts=1,
        hidden_size=4096,
        moe_intermediate_size=2048,
        attention="DSv4 Flash: 2 dense + 5 C4A + 5 C128A",
        notes="First 12 full-dimension DeepSeek-V4 Flash layers.",
    ),
    ModelShape(
        name="qwen3_5_397b_a17b_8layer",
        reduced_layers=8,
        total_experts=512,
        routed_top_k=10,
        shared_experts=1,
        hidden_size=4096,
        moe_intermediate_size=1024,
        attention="GQA 32Q/2KV head_dim=256",
        notes="Reduced 60 -> 8 layers; routed top-10 plus one shared expert.",
    ),
    ModelShape(
        name="kimi_k2_6_6layer",
        reduced_layers=6,
        total_experts=384,
        routed_top_k=8,
        shared_experts=1,
        hidden_size=4096,
        moe_intermediate_size=2048,
        attention="MLA",
        notes=(
            "Reduced 61 -> 6 layers; native hidden 7168 may OOM optimizer state, "
            "so 4096 is the fallback target."
        ),
    ),
)


def _append_unique(values: list[int], value: int) -> None:
    if value not in values:
        values.append(value)


def _make_topk_ids(
    shape: ModelShape,
    *,
    source_rank: int,
    world_size: int,
    tokens_per_source: int,
) -> torch.Tensor:
    if shape.total_experts % world_size != 0:
        raise ValueError(
            f"{shape.name}: total_experts={shape.total_experts} is not divisible "
            f"by world_size={world_size}"
        )
    num_local_experts = shape.total_experts // world_size
    rows: list[list[int]] = []
    primary_hot_owner = 0
    secondary_hot_owner = 1 % world_size
    tertiary_hot_owner = 3 % world_size
    hot_local_span = max(1, min(num_local_experts, shape.routed_top_k))

    for token in range(tokens_per_source):
        route: list[int] = []

        # Keep the first few routed slots intentionally hot so the greedy
        # planner has a meaningful owner-load imbalance on every model family.
        for local_expert in range(min(3, shape.routed_top_k, hot_local_span)):
            _append_unique(route, primary_hot_owner * num_local_experts + local_expert)
        if len(route) < shape.routed_top_k:
            _append_unique(
                route,
                secondary_hot_owner * num_local_experts + (token % hot_local_span),
            )
        if len(route) < shape.routed_top_k:
            _append_unique(
                route,
                tertiary_hot_owner * num_local_experts + ((token // 2) % hot_local_span),
            )

        candidate_seed = source_rank * 131 + token * 17
        candidate_idx = 0
        while len(route) < shape.routed_top_k:
            candidate = (
                candidate_seed
                + candidate_idx * 29
                + len(route) * 7
            ) % shape.total_experts
            _append_unique(route, int(candidate))
            candidate_idx += 1

        rows.append(route[: shape.routed_top_k])

    return torch.tensor(rows, dtype=torch.int64)


def _shape_counts(
    module: Any,
    shape: ModelShape,
    *,
    world_size: int,
    tokens_per_source: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    num_local_experts = shape.total_experts // world_size
    topk_by_source = [
        _make_topk_ids(
            shape,
            source_rank=source_rank,
            world_size=world_size,
            tokens_per_source=tokens_per_source,
        )
        for source_rank in range(world_size)
    ]
    local_counts = [
        module.count_local_routes_by_owner_expert(
            topk_ids,
            num_local_experts=num_local_experts,
            world_size=world_size,
        )
        for topk_ids in topk_by_source
    ]
    return torch.stack(local_counts, dim=1).contiguous(), topk_by_source


def _check_shape(
    shape: ModelShape,
    *,
    left_name: str,
    left_module: Any,
    right_name: str,
    right_module: Any,
    world_size: int,
    hot_experts: int,
    tokens_per_source: int,
) -> dict[str, Any]:
    label = f"{shape.name}: {left_name} vs {right_name}"
    left_counts, left_topk_by_source = _shape_counts(
        left_module,
        shape,
        world_size=world_size,
        tokens_per_source=tokens_per_source,
    )
    right_counts, right_topk_by_source = _shape_counts(
        right_module,
        shape,
        world_size=world_size,
        tokens_per_source=tokens_per_source,
    )
    _assert_close(label, left_counts, right_counts, ".counts")
    _assert_close(label, left_topk_by_source, right_topk_by_source, ".topk_ids")

    left_plan = left_module.build_balanced_moe_plan_from_global_counts(
        left_counts,
        hot_expert_num=hot_experts,
    )
    right_plan = right_module.build_balanced_moe_plan_from_global_counts(
        right_counts,
        hot_expert_num=hot_experts,
    )
    _assert_close(label, _plan_as_dict(left_plan), _plan_as_dict(right_plan), ".plan")

    if not left_plan.hot_experts:
        raise AssertionError(f"{shape.name}: planner selected no hot experts")
    if left_plan.remote_rows_total <= 0:
        raise AssertionError(f"{shape.name}: planner selected no remote helper rows")

    compact_local_indices = torch.arange(
        len(left_plan.hot_experts),
        dtype=torch.int64,
    )
    for rank in range(world_size):
        selected = left_topk_by_source[rank]
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
                left_partition.as_dict(),
                right_partition.as_dict(),
                f".partition.rank{rank}.{presort_by}",
            )
            if presort_by == "owner_compact":
                weights = torch.arange(
                    selected.numel(),
                    dtype=torch.float32,
                ).reshape_as(selected)
                left_normal_ids, left_normal_weights = (
                    left_module.build_normal_topk_dispatch_tensors(
                        selected,
                        weights,
                        partition=left_partition,
                    )
                )
                right_normal_ids, right_normal_weights = (
                    right_module.build_normal_topk_dispatch_tensors(
                        selected,
                        weights,
                        partition=right_partition,
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
        _assert_close(
            label,
            left_layout.as_dict(),
            right_layout.as_dict(),
            f".runtime_layout.rank{rank}",
        )

    return {
        "name": shape.name,
        "reduced_layers": shape.reduced_layers,
        "total_experts": shape.total_experts,
        "routed_top_k": shape.routed_top_k,
        "shared_experts": shape.shared_experts,
        "hidden_size": shape.hidden_size,
        "moe_intermediate_size": shape.moe_intermediate_size,
        "attention": shape.attention,
        "notes": shape.notes,
        "world_size": world_size,
        "tokens_per_source": tokens_per_source,
        "selected_hot_experts": len(left_plan.hot_experts),
        "remote_rows_total": left_plan.remote_rows_total,
        "selected_rows_total": left_plan.selected_rows_total,
        "modeled_max_load_reduction_pct": left_plan.modeled_max_load_reduction_pct,
        "selected_global_experts": list(left_plan.selected_global_experts),
        "selected_owner_ranks": list(left_plan.selected_owner_ranks),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mori", type=Path, default=DEFAULT_BACKENDS["mori"])
    parser.add_argument(
        "--primus-turbo",
        type=Path,
        default=DEFAULT_BACKENDS["primus_turbo"],
    )
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--hot-experts", type=int, default=8)
    parser.add_argument("--tokens-per-source", type=int, default=256)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.world_size <= 0:
        raise ValueError("--world-size must be positive")
    if args.hot_experts <= 0:
        raise ValueError("--hot-experts must be positive")
    if args.tokens_per_source <= 0:
        raise ValueError("--tokens-per-source must be positive")

    backends = {
        "mori": args.mori.resolve(),
        "primus_turbo": args.primus_turbo.resolve(),
    }
    modules = {name: _load_module(name, path) for name, path in backends.items()}
    (left_name, left_module), (right_name, right_module) = tuple(modules.items())
    capabilities = {
        name: module.balanced_moe_backend_capabilities()
        for name, module in modules.items()
    }

    results = [
        _check_shape(
            shape,
            left_name=left_name,
            left_module=left_module,
            right_name=right_name,
            right_module=right_module,
            world_size=args.world_size,
            hot_experts=args.hot_experts,
            tokens_per_source=args.tokens_per_source,
        )
        for shape in MODEL_SHAPES
    ]
    report = {
        "backends": {name: str(path) for name, path in backends.items()},
        "backend_capabilities": capabilities,
        "world_size": args.world_size,
        "hot_experts": args.hot_experts,
        "tokens_per_source": args.tokens_per_source,
        "models": results,
    }

    if args.write_json is not None:
        args.write_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("balanced-MoE model shape coverage OK")
        for result in results:
            print(
                "  {name}: experts={total_experts}, topk={routed_top_k}, "
                "hot={selected_hot_experts}, remote_rows={remote_rows_total}, "
                "modeled_reduction={modeled_max_load_reduction_pct:.2f}%".format(
                    **result
                )
            )


if __name__ == "__main__":
    main()
