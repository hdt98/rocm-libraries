#!/usr/bin/env python3
"""Smoke-test MORI hot-helper packed-row kernel.

This is a bring-up probe for the experimental native hot/cold/helper MORI
dispatcher path. It validates only the helper-side row pack primitive:

  (token, topk slot) hot route -> owner-hot slot -> compact helper rows

Rows within each owner slot are allocated with atomics, so the parity check
sorts by the saved flat source position before comparing payloads.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import torch


def _sorted_rows(
    x: torch.Tensor,
    scores: torch.Tensor,
    src_info: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if count == 0:
        return x[:0], scores[:0], src_info[:0]
    order = torch.argsort(src_info[:count])
    return x[:count][order], scores[:count][order], src_info[:count][order]


def run_smoke(hidden_dim: int, tokens: int, topk: int) -> dict[str, Any]:
    from mori import cpp as mori_cpp
    from mori import shmem

    assert torch.cuda.is_available(), "ROCm/PyTorch CUDA device is required"
    torch.manual_seed(7)
    device = torch.device("cuda")
    uid = shmem.shmem_get_unique_id()
    shmem_status = shmem.shmem_init_attr(
        shmem.MORI_SHMEM_INIT_WITH_UNIQUEID,
        0,
        1,
        uid,
    )

    x = (torch.arange(tokens * hidden_dim, device=device, dtype=torch.float32) / 17.0).reshape(
        tokens, hidden_dim
    )
    x = x.to(torch.bfloat16).contiguous()
    weights = torch.randn(tokens, topk, device=device, dtype=torch.float32).contiguous()

    # Include collisions within owner slots so the atomic row allocator is exercised.
    hot_flat_positions = torch.tensor(
        [0, 3, 5, 7, 8, 10, 11],
        device=device,
        dtype=torch.int64,
    )
    hot_flat_positions = hot_flat_positions[hot_flat_positions < tokens * topk].contiguous()
    hot_owner_slots = torch.tensor(
        [0, 1, 0, 1, 2, 2, 0],
        device=device,
        dtype=torch.int64,
    )[: hot_flat_positions.numel()].contiguous()

    num_hot_slots = int(hot_owner_slots.max().item()) + 1
    max_rows = int(hot_flat_positions.numel())
    out_x = torch.empty(
        (num_hot_slots, max_rows, hidden_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    out_scores = torch.empty((num_hot_slots, max_rows), device=device, dtype=torch.float32)
    out_src = torch.empty((num_hot_slots, max_rows), device=device, dtype=torch.int64)
    out_count = torch.empty((num_hot_slots,), device=device, dtype=torch.int32)

    cfg = mori_cpp.EpDispatchCombineConfig(
        rank=0,
        world_size=1,
        hidden_dim=hidden_dim,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=x.element_size(),
        max_num_inp_token_per_rank=tokens,
        num_experts_per_rank=1,
        num_experts_per_token=topk,
        max_total_recv_tokens=0,
        warp_num_per_block=4,
        block_num=4,
        use_external_inp_buf=True,
        kernel_type=mori_cpp.EpDispatchCombineKernelType.IntraNode,
        gpu_per_node=1,
        rdma_block_num=0,
        num_qp_per_pe=1,
        quant_type=mori_cpp.EpDispatchCombineQuantType.None_,
    )

    mori_cpp.launch_hot_helper_pack(
        cfg,
        1,  # HIP_R_16BF
        x.data_ptr(),
        weights.data_ptr(),
        hot_flat_positions.data_ptr(),
        hot_owner_slots.data_ptr(),
        hot_flat_positions.numel(),
        out_x.data_ptr(),
        out_scores.data_ptr(),
        out_src.data_ptr(),
        out_count.data_ptr(),
        num_hot_slots,
        max_rows,
        4,
        4,
        torch.cuda.current_stream().cuda_stream,
        hidden_dim,
    )
    torch.cuda.synchronize()

    counts = out_count.cpu().tolist()
    expected_counts = [
        int((hot_owner_slots == slot).sum().item()) for slot in range(num_hot_slots)
    ]
    passed = counts == expected_counts
    max_x_diff = 0.0
    max_score_diff = 0.0
    mismatches: list[dict[str, Any]] = []

    for slot in range(num_hot_slots):
        count = counts[slot]
        got_x, got_scores, got_src = _sorted_rows(out_x[slot], out_scores[slot], out_src[slot], count)
        expected_flat = torch.sort(hot_flat_positions[hot_owner_slots == slot]).values
        expected_tokens = torch.div(expected_flat, topk, rounding_mode="floor")
        expected_topk = expected_flat % topk
        expected_x = x[expected_tokens]
        expected_scores = weights[expected_tokens, expected_topk]

        src_ok = torch.equal(got_src.cpu(), expected_flat.cpu())
        x_diff = (
            (got_x.float() - expected_x.float()).abs().max().item()
            if count
            else 0.0
        )
        score_diff = (
            (got_scores - expected_scores).abs().max().item()
            if count
            else 0.0
        )
        max_x_diff = max(max_x_diff, float(x_diff))
        max_score_diff = max(max_score_diff, float(score_diff))
        slot_passed = src_ok and x_diff == 0.0 and score_diff == 0.0
        passed = passed and slot_passed
        if not slot_passed:
            mismatches.append(
                {
                    "slot": slot,
                    "count": count,
                    "src_ok": bool(src_ok),
                    "x_diff": float(x_diff),
                    "score_diff": float(score_diff),
                }
            )

    result = {
        "passed": bool(passed),
        "tokens": tokens,
        "topk": topk,
        "hidden_dim": hidden_dim,
        "num_hot_routes": int(hot_flat_positions.numel()),
        "num_hot_slots": num_hot_slots,
        "counts": counts,
        "expected_counts": expected_counts,
        "max_x_diff": max_x_diff,
        "max_score_diff": max_score_diff,
        "mismatches": mismatches,
        "mori_kernel_dir": os.environ.get("MORI_KERNEL_DIR", ""),
        "shmem_init_status": int(shmem_status),
    }
    result["shmem_finalize_status"] = int(shmem.shmem_finalize())
    return result


def run_wrapper_smoke(hidden_dim: int, tokens: int, topk: int) -> dict[str, Any]:
    from mori import shmem
    from mori.ops.dispatch_combine import (
        EpDispatchCombineConfig,
        EpDispatchCombineKernelType,
        EpDispatchCombineOp,
    )

    assert torch.cuda.is_available(), "ROCm/PyTorch CUDA device is required"
    torch.manual_seed(11)
    device = torch.device("cuda")
    uid = shmem.shmem_get_unique_id()
    shmem_status = shmem.shmem_init_attr(
        shmem.MORI_SHMEM_INIT_WITH_UNIQUEID,
        0,
        1,
        uid,
    )

    x = (torch.arange(tokens * hidden_dim, device=device, dtype=torch.float32) / 23.0).reshape(
        tokens, hidden_dim
    )
    x = x.to(torch.bfloat16).contiguous()
    weights = torch.randn(tokens, topk, device=device, dtype=torch.float32).contiguous()
    indices = torch.arange(topk, device=device, dtype=torch.int32).repeat(tokens, 1).contiguous()
    hot_flat_positions = torch.tensor(
        [0, 3, 5, 7, 8, 10, 11],
        device=device,
        dtype=torch.int64,
    )
    hot_flat_positions = hot_flat_positions[hot_flat_positions < tokens * topk].contiguous()
    hot_owner_slots = torch.tensor(
        [0, 1, 0, 1, 2, 2, 0],
        device=device,
        dtype=torch.int64,
    )[: hot_flat_positions.numel()].contiguous()
    num_hot_slots = int(hot_owner_slots.max().item()) + 1
    max_rows = int(hot_flat_positions.numel())
    normal_route_mask = torch.ones((tokens, topk), device=device, dtype=torch.uint8)
    normal_route_mask.view(-1)[hot_flat_positions] = 0

    cfg = EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=0,
        world_size=1,
        hidden_dim=hidden_dim,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=x.element_size(),
        max_num_inp_token_per_rank=tokens,
        num_experts_per_rank=topk,
        num_experts_per_token=topk,
        max_total_recv_tokens=0,
        warp_num_per_block=4,
        block_num=4,
        use_external_inp_buf=True,
        kernel_type=EpDispatchCombineKernelType.IntraNode,
        gpu_per_node=1,
        rdma_block_num=0,
        num_qp_per_pe=1,
        quant_type="none",
    )
    op = EpDispatchCombineOp(cfg)
    outputs = op.dispatch_hotcold_standard_moe(
        x,
        weights,
        None,
        indices,
        normal_route_mask,
        hot_flat_positions,
        hot_owner_slots,
        num_hot_slots,
        max_rows,
        block_num=4,
        warp_per_block=4,
    )
    torch.cuda.synchronize()

    (
        normal_x,
        normal_count,
        normal_src_info,
        _normal_layout_range,
        hot_x,
        hot_scores,
        hot_src_info,
        hot_count,
    ) = outputs

    counts = hot_count.cpu().tolist()
    expected_counts = [
        int((hot_owner_slots == slot).sum().item()) for slot in range(num_hot_slots)
    ]
    passed = counts == expected_counts
    max_x_diff = 0.0
    max_score_diff = 0.0
    mismatches: list[dict[str, Any]] = []

    for slot in range(num_hot_slots):
        count = counts[slot]
        got_x, got_scores, got_src = _sorted_rows(hot_x[slot], hot_scores[slot], hot_src_info[slot], count)
        expected_flat = torch.sort(hot_flat_positions[hot_owner_slots == slot]).values
        expected_tokens = torch.div(expected_flat, topk, rounding_mode="floor")
        expected_topk = expected_flat % topk
        expected_x = x[expected_tokens]
        expected_scores = weights[expected_tokens, expected_topk]

        src_ok = torch.equal(got_src.cpu(), expected_flat.cpu())
        x_diff = (
            (got_x.float() - expected_x.float()).abs().max().item()
            if count
            else 0.0
        )
        score_diff = (
            (got_scores - expected_scores).abs().max().item()
            if count
            else 0.0
        )
        max_x_diff = max(max_x_diff, float(x_diff))
        max_score_diff = max(max_score_diff, float(score_diff))
        slot_passed = src_ok and x_diff == 0.0 and score_diff == 0.0
        passed = passed and slot_passed
        if not slot_passed:
            mismatches.append(
                {
                    "slot": slot,
                    "count": count,
                    "src_ok": bool(src_ok),
                    "x_diff": float(x_diff),
                    "score_diff": float(score_diff),
                }
            )

    normal_counts = normal_count.cpu().tolist()
    expected_normal_counts = [
        int((((indices == expert) & normal_route_mask.bool()).sum()).item())
        for expert in range(topk)
    ]
    passed = passed and normal_counts == expected_normal_counts

    result = {
        "passed": bool(passed),
        "mode": "wrapper_hotcold_standard_moe",
        "tokens": tokens,
        "topk": topk,
        "hidden_dim": hidden_dim,
        "hot_counts": counts,
        "expected_hot_counts": expected_counts,
        "normal_counts": normal_counts,
        "expected_normal_counts": expected_normal_counts,
        "normal_checksum": float(normal_x.float().sum().item()),
        "normal_src_checksum": int(normal_src_info.long().sum().item()),
        "max_x_diff": max_x_diff,
        "max_score_diff": max_score_diff,
        "mismatches": mismatches,
        "mori_kernel_dir": os.environ.get("MORI_KERNEL_DIR", ""),
        "shmem_init_status": int(shmem_status),
    }
    result["shmem_finalize_status"] = int(shmem.shmem_finalize())
    return result


def run_wrapper_ddp_smoke(hidden_dim: int, tokens: int, topk: int) -> dict[str, Any]:
    import torch.distributed as dist
    from mori import shmem
    from mori.ops.dispatch_combine import (
        EpDispatchCombineConfig,
        EpDispatchCombineKernelType,
        EpDispatchCombineOp,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError("wrapper_ddp smoke expects at least two ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    uid_box: list[bytes | None]
    if rank == 0:
        uid_box = [shmem.shmem_get_unique_id()]
    else:
        uid_box = [None]
    dist.broadcast_object_list(uid_box, src=0)
    uid = uid_box[0]
    assert uid is not None
    shmem_status = shmem.shmem_init_attr(
        shmem.MORI_SHMEM_INIT_WITH_UNIQUEID,
        rank,
        world_size,
        uid,
    )

    torch.manual_seed(101 + rank)
    num_experts_per_rank = topk
    total_experts = world_size * num_experts_per_rank
    x = (
        torch.arange(tokens * hidden_dim, device=device, dtype=torch.float32).reshape(
            tokens, hidden_dim
        )
        + rank * 1000
    )
    x = (x / 29.0).to(torch.bfloat16).contiguous()
    weights = torch.randn(tokens, topk, device=device, dtype=torch.float32).contiguous()

    local_expert = rank * num_experts_per_rank
    remote_rank = (rank + 1) % world_size
    remote_expert = remote_rank * num_experts_per_rank
    indices = torch.empty((tokens, topk), device=device, dtype=torch.int32)
    indices[:, 0] = local_expert
    indices[:, 1] = remote_expert

    # Mark a few remote-helper slots hot and remove them from normal dispatch.
    hot_flat_positions = torch.tensor(
        [1, 3, 5, 7],
        device=device,
        dtype=torch.int64,
    )
    hot_flat_positions = hot_flat_positions[hot_flat_positions < tokens * topk].contiguous()
    hot_owner_slots = torch.full_like(hot_flat_positions, rank)
    num_hot_slots = world_size
    max_rows = max(1, tokens)
    normal_route_mask = torch.ones((tokens, topk), device=device, dtype=torch.uint8)
    normal_route_mask.view(-1)[hot_flat_positions] = 0

    cfg = EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=x.element_size(),
        max_num_inp_token_per_rank=tokens,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=topk,
        max_total_recv_tokens=0,
        warp_num_per_block=4,
        block_num=8,
        use_external_inp_buf=True,
        kernel_type=EpDispatchCombineKernelType.IntraNode,
        gpu_per_node=world_size,
        rdma_block_num=0,
        num_qp_per_pe=1,
        quant_type="none",
    )
    op = EpDispatchCombineOp(cfg)
    outputs = op.dispatch_hotcold_standard_moe(
        x,
        weights,
        None,
        indices,
        normal_route_mask,
        hot_flat_positions,
        hot_owner_slots,
        num_hot_slots,
        max_rows,
        block_num=8,
        warp_per_block=4,
    )
    torch.cuda.synchronize()

    (
        _normal_x,
        normal_count,
        _normal_src_info,
        _normal_layout_range,
        hot_x,
        hot_scores,
        hot_src_info,
        hot_count,
    ) = outputs

    all_indices = [torch.empty_like(indices) for _ in range(world_size)]
    all_masks = [torch.empty_like(normal_route_mask) for _ in range(world_size)]
    dist.all_gather(all_indices, indices)
    dist.all_gather(all_masks, normal_route_mask)
    expected_normal_counts = []
    for local in range(num_experts_per_rank):
        expert = rank * num_experts_per_rank + local
        total = 0
        for idx, mask in zip(all_indices, all_masks):
            total += int((((idx == expert) & mask.bool()).sum()).item())
        expected_normal_counts.append(total)

    counts = hot_count.cpu().tolist()
    expected_hot_counts = [
        int((hot_owner_slots == slot).sum().item()) for slot in range(num_hot_slots)
    ]
    hot_passed = counts == expected_hot_counts
    max_x_diff = 0.0
    max_score_diff = 0.0
    for slot in range(num_hot_slots):
        count = counts[slot]
        got_x, got_scores, got_src = _sorted_rows(hot_x[slot], hot_scores[slot], hot_src_info[slot], count)
        expected_flat = torch.sort(hot_flat_positions[hot_owner_slots == slot]).values
        expected_tokens = torch.div(expected_flat, topk, rounding_mode="floor")
        expected_topk = expected_flat % topk
        expected_x = x[expected_tokens]
        expected_scores = weights[expected_tokens, expected_topk]
        src_ok = torch.equal(got_src.cpu(), expected_flat.cpu())
        x_diff = (
            (got_x.float() - expected_x.float()).abs().max().item()
            if count
            else 0.0
        )
        score_diff = (
            (got_scores - expected_scores).abs().max().item()
            if count
            else 0.0
        )
        max_x_diff = max(max_x_diff, float(x_diff))
        max_score_diff = max(max_score_diff, float(score_diff))
        hot_passed = hot_passed and src_ok and x_diff == 0.0 and score_diff == 0.0

    normal_counts = normal_count.cpu().tolist()
    normal_passed = normal_counts == expected_normal_counts
    rank_result = {
        "rank": rank,
        "hot_counts": counts,
        "expected_hot_counts": expected_hot_counts,
        "normal_counts": normal_counts,
        "expected_normal_counts": expected_normal_counts,
        "hot_passed": bool(hot_passed),
        "normal_passed": bool(normal_passed),
        "max_x_diff": max_x_diff,
        "max_score_diff": max_score_diff,
        "shmem_init_status": int(shmem_status),
    }
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, rank_result)
    finalize_status = int(shmem.shmem_finalize())
    dist.destroy_process_group()

    if rank == 0:
        passed = all(r["hot_passed"] and r["normal_passed"] for r in gathered)
        return {
            "passed": bool(passed),
            "mode": "wrapper_hotcold_standard_moe_ddp",
            "world_size": world_size,
            "tokens": tokens,
            "topk": topk,
            "hidden_dim": hidden_dim,
            "ranks": gathered,
            "rank0_shmem_finalize_status": finalize_status,
            "mori_kernel_dir": os.environ.get("MORI_KERNEL_DIR", ""),
        }
    return {"passed": True, "rank": rank}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])


def _summarize_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median_ms": 0.0, "p10_ms": 0.0, "p90_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(float(v) for v in values)
    return {
        "median_ms": _percentile(ordered, 0.5),
        "p10_ms": _percentile(ordered, 0.1),
        "p90_ms": _percentile(ordered, 0.9),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def _time_cuda(fn, *, warmup: int, iters: int) -> tuple[dict[str, float], float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    checksum = 0.0
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        checksum = float(fn())
        stop.record()
        stop.synchronize()
        times.append(float(start.elapsed_time(stop)))
    summary = _summarize_ms(times)
    summary["iters"] = float(iters)
    return summary, checksum


def _parse_int_list(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def run_timing(
    *,
    hidden_dim: int,
    tokens: int,
    topk: int,
    route_rows_list: list[int],
    num_hot_slots: int,
    slot_skew: float,
    warmup: int,
    iters: int,
    score_before_experts: bool,
    pack_block_num: int,
    pack_warp_per_block: int,
) -> dict[str, Any]:
    from mori import cpp as mori_cpp
    from mori import shmem

    assert torch.cuda.is_available(), "ROCm/PyTorch CUDA device is required"
    torch.manual_seed(20260609)
    device = torch.device("cuda")
    uid = shmem.shmem_get_unique_id()
    shmem_status = shmem.shmem_init_attr(
        shmem.MORI_SHMEM_INIT_WITH_UNIQUEID,
        0,
        1,
        uid,
    )

    max_route_rows = max(route_rows_list)
    min_tokens = int(math.ceil(max_route_rows / max(1, topk)))
    tokens = max(int(tokens), min_tokens)
    total_routes = tokens * topk
    x = torch.randn(tokens, hidden_dim, device=device, dtype=torch.bfloat16).contiguous()
    weights = torch.randn(tokens, topk, device=device, dtype=torch.float32).contiguous()
    flat_scores = weights.reshape(-1)

    slot_ids = torch.arange(num_hot_slots, device=device, dtype=torch.float32)
    if slot_skew > 0:
        probs = 1.0 / torch.pow(slot_ids + 1.0, float(slot_skew))
    else:
        probs = torch.ones_like(slot_ids)
    probs = probs / probs.sum()

    cfg = mori_cpp.EpDispatchCombineConfig(
        rank=0,
        world_size=1,
        hidden_dim=hidden_dim,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=x.element_size(),
        max_num_inp_token_per_rank=tokens,
        num_experts_per_rank=1,
        num_experts_per_token=topk,
        max_total_recv_tokens=0,
        warp_num_per_block=4,
        block_num=8,
        use_external_inp_buf=True,
        kernel_type=mori_cpp.EpDispatchCombineKernelType.IntraNode,
        gpu_per_node=1,
        rdma_block_num=0,
        num_qp_per_pe=1,
        quant_type=mori_cpp.EpDispatchCombineQuantType.None_,
    )

    cases: list[dict[str, Any]] = []
    for route_rows in route_rows_list:
        route_rows = int(route_rows)
        if route_rows > total_routes:
            raise ValueError(
                f"route_rows={route_rows} exceeds tokens*topk={total_routes}"
            )
        hot_flat_positions = torch.randperm(
            total_routes, device=device, dtype=torch.int64
        )[:route_rows].contiguous()
        hot_owner_slots = torch.multinomial(
            probs, route_rows, replacement=True
        ).to(torch.int64).contiguous()
        slot_counts = torch.bincount(
            hot_owner_slots, minlength=num_hot_slots
        ).to(torch.int64)
        max_rows_per_slot = max(1, int(slot_counts.max().item()))
        out_x = torch.empty(
            (num_hot_slots, max_rows_per_slot, hidden_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        out_scores = torch.empty(
            (num_hot_slots, max_rows_per_slot),
            device=device,
            dtype=torch.float32,
        )
        out_src = torch.empty(
            (num_hot_slots, max_rows_per_slot),
            device=device,
            dtype=torch.int64,
        )
        out_count = torch.empty((num_hot_slots,), device=device, dtype=torch.int32)
        compact_x = torch.empty(
            (route_rows, hidden_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        compact_scores = torch.empty((route_rows,), device=device, dtype=torch.float32)
        compact_src = torch.empty((route_rows,), device=device, dtype=torch.int64)
        compact_count = torch.empty((num_hot_slots,), device=device, dtype=torch.int32)

        def torch_sort_materialize() -> float:
            order = torch.argsort(hot_owner_slots, stable=True)
            positions = hot_flat_positions.index_select(0, order)
            slots = hot_owner_slots.index_select(0, order)
            token_idx = torch.div(positions, topk, rounding_mode="floor")
            scores = flat_scores.index_select(0, positions)
            rows = x.index_select(0, token_idx)
            if score_before_experts:
                rows = (rows.float() * scores.reshape(-1, 1)).to(x.dtype)
            group_ends = torch.cumsum(
                torch.bincount(slots, minlength=num_hot_slots).to(torch.int64),
                dim=0,
                dtype=torch.int32,
            )
            return rows[:1, :1].float().sum().item() + scores[:1].sum().item() + group_ends[-1].float().item()

        order_once = torch.argsort(hot_owner_slots, stable=True)
        positions_once = hot_flat_positions.index_select(0, order_once).contiguous()
        slots_once = hot_owner_slots.index_select(0, order_once).contiguous()
        token_idx_once = torch.div(positions_once, topk, rounding_mode="floor").contiguous()
        group_ends_once = torch.cumsum(
            torch.bincount(slots_once, minlength=num_hot_slots).to(torch.int64),
            dim=0,
            dtype=torch.int32,
        )

        def torch_presorted_materialize() -> float:
            scores = flat_scores.index_select(0, positions_once)
            rows = x.index_select(0, token_idx_once)
            if score_before_experts:
                rows = (rows.float() * scores.reshape(-1, 1)).to(x.dtype)
            return rows[:1, :1].float().sum().item() + scores[:1].sum().item() + group_ends_once[-1].float().item()

        def mori_pack_prealloc() -> float:
            mori_cpp.launch_hot_helper_pack(
                cfg,
                1,  # HIP_R_16BF
                x.data_ptr(),
                weights.data_ptr(),
                hot_flat_positions.data_ptr(),
                hot_owner_slots.data_ptr(),
                route_rows,
                out_x.data_ptr(),
                out_scores.data_ptr(),
                out_src.data_ptr(),
                out_count.data_ptr(),
                num_hot_slots,
                max_rows_per_slot,
                pack_block_num,
                pack_warp_per_block,
                torch.cuda.current_stream().cuda_stream,
                hidden_dim,
            )
            return out_x[:1, :1, :1].float().sum().item() + out_scores[:1, :1].sum().item() + out_count.float().sum().item()

        def mori_pack_alloc() -> float:
            local_x = torch.empty_like(out_x)
            local_scores = torch.empty_like(out_scores)
            local_src = torch.empty_like(out_src)
            local_count = torch.empty_like(out_count)
            mori_cpp.launch_hot_helper_pack(
                cfg,
                1,  # HIP_R_16BF
                x.data_ptr(),
                weights.data_ptr(),
                hot_flat_positions.data_ptr(),
                hot_owner_slots.data_ptr(),
                route_rows,
                local_x.data_ptr(),
                local_scores.data_ptr(),
                local_src.data_ptr(),
                local_count.data_ptr(),
                num_hot_slots,
                max_rows_per_slot,
                pack_block_num,
                pack_warp_per_block,
                torch.cuda.current_stream().cuda_stream,
                hidden_dim,
            )
            return local_x[:1, :1, :1].float().sum().item() + local_scores[:1, :1].sum().item() + local_count.float().sum().item()

        def mori_compact_pack_prealloc() -> float:
            mori_cpp.launch_hot_helper_compact_pack(
                cfg,
                1,  # HIP_R_16BF
                x.data_ptr(),
                weights.data_ptr(),
                hot_flat_positions.data_ptr(),
                hot_owner_slots.data_ptr(),
                route_rows,
                compact_x.data_ptr(),
                compact_scores.data_ptr(),
                compact_src.data_ptr(),
                compact_count.data_ptr(),
                num_hot_slots,
                pack_block_num,
                pack_warp_per_block,
                torch.cuda.current_stream().cuda_stream,
                hidden_dim,
            )
            return compact_x[:1, :1].float().sum().item() + compact_scores[:1].sum().item() + compact_count.float().sum().item()

        def mori_compact_pack_nocount() -> float:
            mori_cpp.launch_hot_helper_compact_pack(
                cfg,
                1,  # HIP_R_16BF
                x.data_ptr(),
                weights.data_ptr(),
                hot_flat_positions.data_ptr(),
                hot_owner_slots.data_ptr(),
                route_rows,
                compact_x.data_ptr(),
                compact_scores.data_ptr(),
                compact_src.data_ptr(),
                0,
                num_hot_slots,
                pack_block_num,
                pack_warp_per_block,
                torch.cuda.current_stream().cuda_stream,
                hidden_dim,
            )
            return compact_x[:1, :1].float().sum().item() + compact_scores[:1].sum().item()

        timings: dict[str, dict[str, float]] = {}
        checksums: dict[str, float] = {}
        for label, fn in (
            ("torch_sort_materialize", torch_sort_materialize),
            ("torch_presorted_materialize", torch_presorted_materialize),
            ("mori_pack_prealloc", mori_pack_prealloc),
            ("mori_pack_alloc", mori_pack_alloc),
            ("mori_compact_pack_prealloc", mori_compact_pack_prealloc),
            ("mori_compact_pack_nocount", mori_compact_pack_nocount),
        ):
            timing, checksum = _time_cuda(fn, warmup=warmup, iters=iters)
            timings[label] = timing
            checksums[label] = float(checksum)

        torch.cuda.synchronize()
        observed_counts = out_count.cpu().tolist()
        observed_compact_counts = compact_count.cpu().tolist()
        expected_counts = slot_counts.cpu().tolist()
        cases.append(
            {
                "route_rows": route_rows,
                "hidden_dim": hidden_dim,
                "tokens": tokens,
                "topk": topk,
                "num_hot_slots": num_hot_slots,
                "max_rows_per_slot": max_rows_per_slot,
                "slot_skew": float(slot_skew),
                "score_before_experts": bool(score_before_experts),
                "pack_block_num": int(pack_block_num),
                "pack_warp_per_block": int(pack_warp_per_block),
                "expected_counts": [int(v) for v in expected_counts],
                "observed_mori_counts": [int(v) for v in observed_counts],
                "observed_mori_compact_counts": [
                    int(v) for v in observed_compact_counts
                ],
                "counts_match": (
                    observed_counts == expected_counts
                    and observed_compact_counts == expected_counts
                ),
                "timings": timings,
                "checksums": checksums,
            }
        )

    result = {
        "passed": all(c["counts_match"] for c in cases),
        "mode": "timing",
        "cases": cases,
        "mori_kernel_dir": os.environ.get("MORI_KERNEL_DIR", ""),
        "mori_jit_cache_dir": os.environ.get("MORI_JIT_CACHE_DIR", ""),
        "enable_standard_moe_adapt": os.environ.get("ENABLE_STANDARD_MOE_ADAPT", ""),
        "shmem_init_status": int(shmem_status),
    }
    result["shmem_finalize_status"] = int(shmem.shmem_finalize())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("helper_pack", "wrapper", "wrapper_ddp", "timing"),
        default="helper_pack",
    )
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--route-rows-list", type=str, default="69635,128284,147159")
    parser.add_argument("--num-hot-slots", type=int, default=8)
    parser.add_argument("--slot-skew", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--score-before-experts", action="store_true")
    parser.add_argument("--pack-block-num", type=int, default=8)
    parser.add_argument("--pack-warp-per-block", type=int, default=4)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    if args.mode == "wrapper_ddp":
        result = run_wrapper_ddp_smoke(args.hidden_dim, args.tokens, args.topk)
    elif args.mode == "wrapper":
        result = run_wrapper_smoke(args.hidden_dim, args.tokens, args.topk)
    elif args.mode == "timing":
        result = run_timing(
            hidden_dim=args.hidden_dim,
            tokens=args.tokens,
            topk=args.topk,
            route_rows_list=_parse_int_list(args.route_rows_list),
            num_hot_slots=args.num_hot_slots,
            slot_skew=args.slot_skew,
            warmup=args.warmup,
            iters=args.iters,
            score_before_experts=args.score_before_experts,
            pack_block_num=args.pack_block_num,
            pack_warp_per_block=args.pack_warp_per_block,
        )
    else:
        result = run_smoke(args.hidden_dim, args.tokens, args.topk)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    should_write = args.output and int(os.environ.get("RANK", "0")) == 0
    if should_write:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
