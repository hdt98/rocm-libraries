#!/usr/bin/env python3
"""Gloo smoke for MORI balanced-MoE owner-compact exchange.

This loads the source file directly so a container with an installed compiled
MORI package can still validate the patched Python utility without rebuilding.
Run with:

    torchrun --standalone --nproc_per_node=3 smoke_mori_owner_compact_exchange_gloo.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch
import torch.distributed as dist


def _load_balanced_moe():
    source = os.environ.get("MORI_BALANCED_MOE_SOURCE")
    if not source:
        raise RuntimeError("MORI_BALANCED_MOE_SOURCE must point at balanced_moe.py")
    spec = importlib.util.spec_from_file_location("mori_balanced_moe_smoke", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _counts_owner_source_local() -> torch.Tensor:
    return torch.tensor(
        [
            [[2, 8], [1, 1], [1, 1]],
            [[1, 4], [0, 5], [1, 20]],
            [[2, 1], [12, 1], [2, 1]],
        ],
        dtype=torch.int64,
    )


def main() -> None:
    dist.init_process_group("gloo")
    rank = int(dist.get_rank())
    module = _load_balanced_moe()

    plan = module.build_balanced_moe_plan_from_global_counts(
        _counts_owner_source_local(),
        hot_expert_num=2,
    )
    exchange_plan = module.build_owner_compact_exchange_plan_from_plan(
        plan,
        rank=rank,
    )

    local_rows = (
        torch.arange(4, dtype=torch.float32).reshape(2, 2) + float(rank * 100)
    )
    local_rows.requires_grad_(True)
    needed = module.exchange_owner_compact_needed_rows(
        local_rows,
        compact_local_indices=plan.owner_compact_local_experts,
        plan=exchange_plan.as_dict(),
        group=dist.group.WORLD,
    )

    expected_needed = {
        0: torch.tensor([[102.0, 103.0], [200.0, 201.0]]),
        1: torch.tensor([[200.0, 201.0]]),
        2: torch.tensor([[102.0, 103.0]]),
    }[rank]
    torch.testing.assert_close(needed, expected_needed)

    needed.sum().backward()
    expected_grad = torch.zeros_like(local_rows)
    if rank == 1:
        expected_grad[1].fill_(2.0)
    elif rank == 2:
        expected_grad[0].fill_(2.0)
    torch.testing.assert_close(local_rows.grad, expected_grad)
    if rank == 0:
        print("mori owner-compact exchange gloo smoke passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
