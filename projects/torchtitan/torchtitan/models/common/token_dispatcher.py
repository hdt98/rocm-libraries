# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import atexit
import gc
import importlib
import json
import os

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.tensor import DeviceMesh

from torchtitan.config import Configurable
from torchtitan.models.common.dsv4_profile_timing import (
    dsv4_profile_timing_enabled,
    dsv4_timed_stage,
    flush_dsv4_profile_timing,
)
from torchtitan.ops.scatter_add import (
    _weighted_scatter_flat_backward,
    deterministic_scatter_add,
    weighted_scatter_add,
    weighted_scatter_add_from_flat_positions,
)


_STANDARD_EP_HOT_EXPERT_CALL_ID = 0
_STANDARD_EP_MORI_COMPACT_PACK_CONFIG_CACHE = {}
_STANDARD_EP_MORI_COMPACT_PACK_RUNTIME_CACHE = {}
_STANDARD_EP_MORI_NATIVE_COMPACT_RUNTIME_CACHE = None
_STANDARD_EP_COMPACT_ROWS_BALANCED_MOE_MODULE_CACHE = {}
_STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE = {}
_STANDARD_EP_MORI_NATIVE_COMPACT_OP_REFS = {}
_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_COUNTS = {}
_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_PENDING = []
_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_ATEXIT_REGISTERED = False


def _standard_ep_hot_expert_timing_path() -> str | None:
    path = os.environ.get("TORCHTITAN_MOE_HOT_EXPERT_TIMING_PATH")
    if path is None or path.strip() == "":
        path = os.environ.get("TORCHTITAN_MORI_AITER_TIMING_PATH")
    if path is None or path.strip() == "":
        return None
    rank = os.environ.get("RANK", "unknown")
    local_rank = os.environ.get("LOCAL_RANK", "unknown")
    pid = os.getpid()
    path = path.format(rank=rank, local_rank=local_rank, pid=pid)
    if path.endswith(".jsonl"):
        return path
    return os.path.join(
        path, f"standard_ep_hot_expert_rank{rank}_local{local_rank}_pid{pid}.jsonl"
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _count_route_ids(route_ids: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Count discrete expert/route ids without floating-point histograms."""
    if num_bins <= 0:
        return torch.empty(0, device=route_ids.device, dtype=torch.int64)
    return torch.bincount(
        route_ids.reshape(-1).to(torch.int64),
        minlength=num_bins,
    )[:num_bins]


def _standard_ep_stage_timing_records() -> list[dict[str, object]] | None:
    """Opt-in stage timing for the standard-EP helper-execute path."""
    if not dsv4_profile_timing_enabled() or torch.compiler.is_compiling():
        return None
    if not _env_flag(
        "TORCHTITAN_STANDARD_EP_STAGE_TIMING",
        _env_flag("CANARY_STANDARD_EP_STAGE_TIMING", True),
    ):
        return None
    return []


def _standard_ep_hot_split_presort_mode() -> str:
    mode = (
        os.environ.get("TORCHTITAN_STANDARD_EP_HOT_SPLIT_PRESORT_MODE")
        or os.environ.get("CANARY_STANDARD_EP_HOT_SPLIT_PRESORT_MODE")
        or "off"
    )
    mode = mode.strip().lower()
    if mode == "":
        mode = "off"
    if mode not in {"off", "selected", "owner_compact"}:
        raise ValueError(
            "TORCHTITAN_STANDARD_EP_HOT_SPLIT_PRESORT_MODE must be "
            "'off', 'selected', or 'owner_compact', got "
            f"{mode!r}."
        )
    return mode


def _standard_ep_hot_split_materialize_rows_mode() -> str:
    mode = (
        os.environ.get("TORCHTITAN_STANDARD_EP_HOT_SPLIT_MATERIALIZE_ROWS")
        or os.environ.get("CANARY_STANDARD_EP_HOT_SPLIT_MATERIALIZE_ROWS")
        or "off"
    )
    mode = mode.strip().lower()
    if mode == "":
        mode = "off"
    if mode not in {"off", "selected", "owner_compact"}:
        raise ValueError(
            "TORCHTITAN_STANDARD_EP_HOT_SPLIT_MATERIALIZE_ROWS must be "
            "'off', 'selected', or 'owner_compact', got "
            f"{mode!r}."
        )
    return mode


def _standard_ep_hot_split_materialize_backend() -> str:
    mode = (
        os.environ.get("TORCHTITAN_STANDARD_EP_HOT_SPLIT_MATERIALIZE_BACKEND")
        or os.environ.get("CANARY_STANDARD_EP_HOT_SPLIT_MATERIALIZE_BACKEND")
        or "torch"
    )
    mode = mode.strip().lower()
    if mode == "":
        mode = "torch"
    if mode not in {"torch", "mori_compact"}:
        raise ValueError(
            "TORCHTITAN_STANDARD_EP_HOT_SPLIT_MATERIALIZE_BACKEND must be "
            "'torch' or 'mori_compact', got "
            f"{mode!r}."
        )
    return mode


def _standard_ep_mori_compact_pack_block_num() -> int:
    return _env_int(
        "TORCHTITAN_STANDARD_EP_MORI_COMPACT_PACK_BLOCKS",
        _env_int("CANARY_STANDARD_EP_MORI_COMPACT_PACK_BLOCKS", 512),
    )


def _standard_ep_mori_compact_pack_warp_per_block() -> int:
    return _env_int(
        "TORCHTITAN_STANDARD_EP_MORI_COMPACT_PACK_WARPS",
        _env_int("CANARY_STANDARD_EP_MORI_COMPACT_PACK_WARPS", 4),
    )


def _standard_ep_export_compact_route_positions() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_EXPORT_COMPACT_ROUTE_POSITIONS",
        _env_flag("CANARY_STANDARD_EP_EXPORT_COMPACT_ROUTE_POSITIONS", False),
    )


def _standard_ep_compact_collective_backend() -> str:
    backend = (
        os.environ.get("TORCHTITAN_STANDARD_EP_COMPACT_COLLECTIVE_BACKEND")
        or os.environ.get("CANARY_STANDARD_EP_COMPACT_COLLECTIVE_BACKEND")
        or "torch"
    )
    backend = backend.strip().lower()
    if backend == "":
        backend = "torch"
    if backend not in {"torch", "mori_native"}:
        raise ValueError(
            "TORCHTITAN_STANDARD_EP_COMPACT_COLLECTIVE_BACKEND must be "
            f"'torch' or 'mori_native', got {backend!r}."
        )
    return backend


def _standard_ep_use_backend_balanced_runtime_layout(
    *,
    materialize_backend: str,
    compact_collective_backend: str,
) -> bool:
    mode = (
        os.environ.get("TORCHTITAN_STANDARD_EP_BALANCED_MOE_RUNTIME_LAYOUT")
        or os.environ.get("CANARY_STANDARD_EP_BALANCED_MOE_RUNTIME_LAYOUT")
        or "auto"
    )
    mode = mode.strip().lower()
    if mode in {"", "auto"}:
        requested_weight_layout = (
            os.environ.get("TORCHTITAN_STANDARD_EP_HOT_SPLIT_WEIGHT_LAYOUT")
            or os.environ.get("CANARY_STANDARD_EP_HOT_SPLIT_WEIGHT_LAYOUT")
            or "selected"
        ).strip().lower()
        needed_exchange_impl = (
            os.environ.get("TORCHTITAN_STANDARD_EP_NEEDED_WEIGHT_EXCHANGE_IMPL")
            or os.environ.get("CANARY_STANDARD_EP_NEEDED_WEIGHT_EXCHANGE_IMPL")
            or "auto"
        ).strip().lower()
        return (
            materialize_backend == "mori_compact"
            or compact_collective_backend == "mori_native"
            or (
                requested_weight_layout == "needed"
                and needed_exchange_impl
                in {"auto", "backend", "native", "mori", "primus", "primus_turbo"}
            )
        )
    if mode in {
        "1",
        "true",
        "yes",
        "on",
        "backend",
        "native",
        "mori",
        "primus",
        "primus_turbo",
    }:
        return True
    if mode in {"0", "false", "no", "off", "torch"}:
        return False
    raise ValueError(
        "TORCHTITAN_STANDARD_EP_BALANCED_MOE_RUNTIME_LAYOUT must be "
        "'auto', 'backend', or 'off', got "
        f"{mode!r}."
    )


def _standard_ep_backend_normal_topk_dispatch_tensors() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_BACKEND_NORMAL_TOPK_DISPATCH_TENSORS",
        _env_flag(
            "CANARY_STANDARD_EP_BACKEND_NORMAL_TOPK_DISPATCH_TENSORS",
            False,
        ),
    )


def _standard_ep_backend_normal_topk_dispatch_validate() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_BACKEND_NORMAL_TOPK_DISPATCH_VALIDATE",
        _env_flag(
            "CANARY_STANDARD_EP_BACKEND_NORMAL_TOPK_DISPATCH_VALIDATE",
            True,
        ),
    )


def _standard_ep_backend_normal_topk_dispatch_permute() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_BACKEND_NORMAL_TOPK_DISPATCH_PERMUTE",
        _env_flag(
            "CANARY_STANDARD_EP_BACKEND_NORMAL_TOPK_DISPATCH_PERMUTE",
            False,
        ),
    )


def _standard_ep_mori_native_compact_fixed_peer_rows() -> int:
    return max(
        0,
        _env_int(
            "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_MAX_PEER_ROWS",
            _env_int("CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_MAX_PEER_ROWS", 0),
        ),
    )


def _standard_ep_mori_native_compact_capacity_rows(required_rows: int) -> int:
    """Capacity bucket for the native compact MORI row collective."""
    required_rows = max(1, int(required_rows))
    required_bucket = 1 << (required_rows - 1).bit_length()
    fixed_rows = _standard_ep_mori_native_compact_fixed_peer_rows()
    if fixed_rows > 0:
        if required_rows > fixed_rows:
            raise RuntimeError(
                "mori_native compact local peer-row requirement exceeds fixed "
                f"capacity: required={required_rows}, fixed={fixed_rows}. "
                "Increase TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_MAX_PEER_ROWS "
                "or unset it to use synchronized auto capacity."
            )
        return fixed_rows
    return required_bucket


def _standard_ep_mori_native_compact_remote_peer_rows(
    input_splits: Sequence[int] | torch.Tensor,
    output_splits: Sequence[int] | torch.Tensor,
    ep_rank: int,
) -> int:
    """Return the split max that actually needs remote staging.

    Native compact self rows are copied directly from the local compact input
    into the local output. They still affect output size, but they should not
    size the MORI peer shared-memory staging stride.
    """

    ep_rank = int(ep_rank)
    remote_values: list[int] = [1]
    for idx, value in enumerate(input_splits):
        if idx != ep_rank:
            remote_values.append(int(value))
    for idx, value in enumerate(output_splits):
        if idx != ep_rank:
            remote_values.append(int(value))
    return max(remote_values)


def _standard_ep_mori_native_compact_capacity_for_call(
    *,
    local_max_peer_rows: int,
    device: torch.device,
    ep_group,
) -> tuple[int, int]:
    """Return (max_peer_rows_for_debug, shared_capacity_peer_rows).

    In auto-cap mode all ranks must agree on the same MORI symmetric-buffer
    capacity, so we still need a tiny max all-reduce before handle lookup. In
    fixed-cap mode the capacity is already rank-invariant by construction; doing
    that scalar all-reduce for every dispatch/combine call only adds a
    synchronization edge to the hot path.
    """
    local_max_peer_rows = max(1, int(local_max_peer_rows))
    fixed_rows = _standard_ep_mori_native_compact_fixed_peer_rows()
    if fixed_rows > 0:
        return local_max_peer_rows, _standard_ep_mori_native_compact_capacity_rows(
            local_max_peer_rows
        )

    max_peer_rows_tensor = torch.tensor(
        [local_max_peer_rows], dtype=torch.int64, device=device
    )
    dist.all_reduce(max_peer_rows_tensor, op=dist.ReduceOp.MAX, group=ep_group)
    max_peer_rows = int(max_peer_rows_tensor.item())
    return max_peer_rows, _standard_ep_mori_native_compact_capacity_rows(max_peer_rows)


def _standard_ep_mori_native_compact_block_num() -> int:
    return _env_int(
        "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_BLOCKS",
        _env_int("CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_BLOCKS", 64),
    )


def _standard_ep_mori_native_compact_warp_per_block() -> int:
    return _env_int(
        "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_WARPS",
        _env_int("CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_WARPS", 8),
    )


def _standard_ep_mori_native_compact_prereset() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_PRERESET",
        _env_flag("CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_PRERESET", False),
    )


def _standard_ep_mori_native_compact_profile_init_detail() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_INIT_DETAIL",
        _env_flag(
            "CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_INIT_DETAIL",
            False,
        ),
    )


def _standard_ep_mori_native_compact_init_stage(
    timing_records: list[dict[str, object]] | None,
    stage: str,
):
    if _standard_ep_mori_native_compact_profile_init_detail():
        return dsv4_timed_stage(timing_records, stage)
    return nullcontext()


def _standard_ep_mori_native_compact_prelaunch_reset(
    *,
    stage_prefix: str,
    op,
    ep_group,
    timing_records: list[dict[str, object]] | None,
) -> None:
    if not _standard_ep_mori_native_compact_prereset():
        return
    with dsv4_timed_stage(timing_records, f"{stage_prefix}.prereset.reset"):
        op.reset()
    with dsv4_timed_stage(timing_records, f"{stage_prefix}.prereset.sync"):
        torch.cuda.synchronize()
    with dsv4_timed_stage(timing_records, f"{stage_prefix}.prereset.barrier"):
        dist.barrier(group=ep_group)


def _standard_ep_mori_native_compact_debug_cache() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_DEBUG_CACHE",
        _env_flag("CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_DEBUG_CACHE", False),
    )


def _standard_ep_mori_native_compact_gc_on_release() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_GC_ON_RELEASE",
        _env_flag("CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_GC_ON_RELEASE", False),
    )


def _standard_ep_mori_native_compact_cache_backward_handles() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_CACHE_BACKWARD_HANDLES",
        _env_flag(
            "CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_CACHE_BACKWARD_HANDLES",
            False,
        ),
    )


def _standard_ep_mori_native_compact_keep_released_handles() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_KEEP_RELEASED_HANDLES",
        _env_flag(
            "CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_KEEP_RELEASED_HANDLES",
            True,
        ),
    )


def _standard_ep_mori_native_compact_role_split() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_ROLE_SPLIT",
        _env_flag(
            "CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_ROLE_SPLIT",
            True,
        ),
    )


def _standard_ep_mori_native_compact_weighted_output() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_WEIGHTED_OUTPUT",
        _env_flag(
            "CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_WEIGHTED_OUTPUT",
            False,
        ),
    )


def _standard_ep_mori_native_compact_weighted_output_backward() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_WEIGHTED_OUTPUT_BACKWARD",
        _env_flag(
            "CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_WEIGHTED_OUTPUT_BACKWARD",
            False,
        ),
    )


def _standard_ep_mori_native_compact_reuse_forward_handles_for_backward() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_REUSE_FORWARD_HANDLES_FOR_BACKWARD",
        _env_flag(
            "CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_REUSE_FORWARD_HANDLES_FOR_BACKWARD",
            True,
        ),
    )


def _standard_ep_mori_native_compact_profile_dir() -> str | None:
    path = (
        os.environ.get("TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_DIR")
        or os.environ.get("CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_DIR")
    )
    if path is None or path.strip() == "":
        return None
    rank = os.environ.get("RANK", "unknown")
    local_rank = os.environ.get("LOCAL_RANK", "unknown")
    pid = os.getpid()
    return path.format(rank=rank, local_rank=local_rank, pid=pid)


def _standard_ep_mori_native_compact_profile_limit() -> int:
    return max(
        0,
        _env_int(
            "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_LIMIT",
            _env_int("CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_LIMIT", 2),
        ),
    )


def _standard_ep_mori_native_compact_profile_start() -> int:
    return max(
        0,
        _env_int(
            "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_START",
            _env_int("CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_START", 0),
        ),
    )


def _standard_ep_mori_native_compact_profile_stage() -> str:
    stage = (
        os.environ.get("TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_STAGE")
        or os.environ.get("CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_STAGE")
        or "both"
    ).strip().lower()
    if stage in {"", "all"}:
        stage = "both"
    if stage not in {
        "both",
        "forward",
        "backward",
        "dispatch",
        "combine",
        "dispatch_backward_combine",
        "combine_backward_dispatch",
    }:
        raise ValueError(
            "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_STAGE must be "
            "'both', 'forward', 'backward', 'dispatch', 'combine', "
            "'dispatch_backward_combine', or 'combine_backward_dispatch', got "
            f"{stage!r}."
        )
    return stage


def _standard_ep_mori_native_compact_profile_stage_matches(
    profile_stage: str, stage: str
) -> bool:
    if profile_stage == "both":
        return True
    if profile_stage == stage:
        return True
    if profile_stage == "forward":
        return stage in {"dispatch", "combine"}
    if profile_stage == "backward":
        return stage in {
            "dispatch_backward_combine",
            "combine_backward_dispatch",
        }
    return False


def _standard_ep_mori_native_compact_profile_defer_export() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_DEFER_EXPORT",
        _env_flag(
            "CANARY_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_DEFER_EXPORT",
            False,
        ),
    )


def _standard_ep_split_stats(values: tuple[int, ...] | list[int]) -> dict[str, object]:
    vals = [int(v) for v in values]
    total = int(sum(vals))
    count = len(vals)
    max_value = int(max(vals)) if vals else 0
    min_value = int(min(vals)) if vals else 0
    mean = float(total / count) if count else 0.0
    return {
        "sum": total,
        "count": count,
        "max": max_value,
        "min": min_value,
        "mean": mean,
        "nonzero": int(sum(1 for v in vals if v != 0)),
        "max_over_mean": float(max_value / mean) if mean > 0.0 else 0.0,
        "values": vals,
    }


def _standard_ep_mori_native_compact_profile_debug_event(
    profile_dir: str | None,
    stage: str,
    event: str,
    **fields,
) -> None:
    if profile_dir is None:
        return
    try:
        os.makedirs(profile_dir, exist_ok=True)
        rank = os.environ.get("RANK", "unknown")
        local_rank = os.environ.get("LOCAL_RANK", "unknown")
        pid = os.getpid()
        path = os.path.join(
            profile_dir,
            f"profile_debug_rank{rank}_local{local_rank}_pid{pid}.jsonl",
        )
        payload = {
            "stage": stage,
            "event": event,
            "rank": rank,
            "local_rank": local_rank,
            "pid": pid,
        }
        payload.update(fields)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        # Profiling diagnostics must not perturb the training path.
        pass


def _standard_ep_mori_native_compact_profile_export(
    profile_state,
    *,
    synchronize: bool,
    event_prefix: str,
) -> None:
    if profile_state is None:
        return
    stage, count, profile_dir, trace_buf, offset = profile_state
    if synchronize:
        torch.cuda.synchronize()
    try:
        import mori
        from mori.kernel_profiler import export_to_perfetto

        os.makedirs(profile_dir, exist_ok=True)
        rank = os.environ.get("RANK", "unknown")
        local_rank = os.environ.get("LOCAL_RANK", "unknown")
        pid = os.getpid()
        filename = (
            f"standard_ep_mori_native_compact_{stage}_"
            f"rank{rank}_local{local_rank}_pid{pid}_call{count}.json"
        )
        output_path = os.path.join(profile_dir, filename)
        slot_map = getattr(mori.cpp, "IntranodeSlots", None)
        try:
            offset_cpu = offset.detach().cpu()
            nonzero_offsets = int((offset_cpu != 0).sum().item())
            max_offset = int(offset_cpu.max().item()) if offset_cpu.numel() else 0
            offset_sum = int(offset_cpu.sum().item()) if offset_cpu.numel() else 0
            offset_sample = [int(v) for v in offset_cpu[:16].tolist()]
        except Exception as offset_exc:
            nonzero_offsets = -1
            max_offset = -1
            offset_sum = -1
            offset_sample = [f"offset_read_failed: {offset_exc!r}"]
        export_to_perfetto(
            trace_buf,
            output_path,
            slot_map=slot_map,
        )
        _standard_ep_mori_native_compact_profile_debug_event(
            profile_dir,
            stage,
            f"{event_prefix}_export_done",
            count=count,
            output_path=output_path,
            output_exists=os.path.exists(output_path),
            output_size=os.path.getsize(output_path)
            if os.path.exists(output_path)
            else 0,
            slot_map_type=type(slot_map).__name__,
            offset_nonzero=nonzero_offsets,
            offset_max=max_offset,
            offset_sum=offset_sum,
            offset_sample=offset_sample,
        )
    except Exception as exc:
        _standard_ep_mori_native_compact_profile_debug_event(
            profile_dir,
            stage,
            f"{event_prefix}_export_failed",
            count=count,
            error=repr(exc),
        )
        _STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_COUNTS[stage] = (
            _standard_ep_mori_native_compact_profile_start()
            + _standard_ep_mori_native_compact_profile_limit()
        )
        if _standard_ep_mori_native_compact_debug_cache():
            print(
                "[standard_ep_mori_native_compact_profile] export_failed "
                f"stage={stage} error={exc}",
                flush=True,
            )


def _standard_ep_mori_native_compact_profile_flush_deferred() -> None:
    if not _STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_PENDING:
        return
    pending = list(_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_PENDING)
    _STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_PENDING.clear()
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        for stage, count, profile_dir, _trace_buf, _offset in pending:
            _standard_ep_mori_native_compact_profile_debug_event(
                profile_dir,
                stage,
                "deferred_synchronize_failed",
                count=count,
                error=repr(exc),
            )
        return
    for profile_state in pending:
        _standard_ep_mori_native_compact_profile_export(
            profile_state,
            synchronize=False,
            event_prefix="deferred",
        )


def _standard_ep_mori_native_compact_profile_register_atexit() -> None:
    global _STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_ATEXIT_REGISTERED
    if _STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_ATEXIT_REGISTERED:
        return
    atexit.register(_standard_ep_mori_native_compact_profile_flush_deferred)
    _STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_ATEXIT_REGISTERED = True


def _standard_ep_mori_native_compact_profile_begin(op, stage: str):
    profile_dir = _standard_ep_mori_native_compact_profile_dir()
    if profile_dir is None:
        return None
    profile_stage = _standard_ep_mori_native_compact_profile_stage()
    if not _standard_ep_mori_native_compact_profile_stage_matches(
        profile_stage, stage
    ):
        return None
    limit = _standard_ep_mori_native_compact_profile_limit()
    if limit <= 0:
        return None
    start = _standard_ep_mori_native_compact_profile_start()
    stop = start + limit
    count = _STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_COUNTS.get(stage, 0)
    if count < start:
        _STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_COUNTS[stage] = count + 1
        return None
    if count >= stop:
        return None
    _STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_COUNTS[stage] = count + 1
    _standard_ep_mori_native_compact_profile_debug_event(
        profile_dir,
        stage,
        "begin",
        count=count,
        start=start,
        limit=limit,
        op_type=type(op).__name__,
        has_debug_time_buf=hasattr(op, "get_debug_time_buf"),
        has_debug_time_offset=hasattr(op, "get_debug_time_offset"),
    )
    try:
        trace_buf = op.get_debug_time_buf()
        offset = op.get_debug_time_offset()
    except Exception as exc:
        _standard_ep_mori_native_compact_profile_debug_event(
            profile_dir,
            stage,
            "begin_failed",
            count=count,
            error=repr(exc),
        )
        if _standard_ep_mori_native_compact_debug_cache():
            print(
                "[standard_ep_mori_native_compact_profile] unavailable "
                f"stage={stage} error={exc}",
                flush=True,
            )
        _STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_COUNTS[stage] = stop
        return None
    trace_buf.zero_()
    offset.zero_()
    _standard_ep_mori_native_compact_profile_debug_event(
        profile_dir,
        stage,
        "buffer_zeroed",
        count=count,
        trace_numel=int(trace_buf.numel()),
        offset_numel=int(offset.numel()),
    )
    return stage, count, profile_dir, trace_buf, offset


def _standard_ep_mori_native_compact_profile_end(profile_state) -> None:
    if profile_state is None:
        return
    stage, count, profile_dir, _trace_buf, _offset = profile_state
    if _standard_ep_mori_native_compact_profile_defer_export():
        _standard_ep_mori_native_compact_profile_register_atexit()
        _STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_PENDING.append(profile_state)
        _standard_ep_mori_native_compact_profile_debug_event(
            profile_dir,
            stage,
            "deferred_export_queued",
            count=count,
            pending=len(_STANDARD_EP_MORI_NATIVE_COMPACT_PROFILE_PENDING),
        )
        return
    _standard_ep_mori_native_compact_profile_export(
        profile_state,
        synchronize=True,
        event_prefix="immediate",
    )


def _standard_ep_mori_native_compact_op_cache_key(
    *,
    stage: str,
    device_index: int,
    ep_rank: int,
    ep_size: int,
    hidden_dim: int,
    element_size: int,
    capacity_peer_rows: int,
) -> tuple[object, ...]:
    # DeviceMesh.get_group() may return a fresh Python wrapper for the same
    # process group; object id would create duplicate MORI symmetric buffers.
    #
    # Dispatch and combine must not share the same MORI op handle. The native
    # compact kernels reuse the handle's recvTokenNumMemObj as their peer-count
    # signal buffer, and the fixed-cap fast path no longer has a scalar
    # all-reduce fence before every call. Keeping stage in the key prevents
    # dispatch/combine signal aliasing without restoring that synchronization
    # edge.
    return (
        str(stage),
        int(device_index),
        int(ep_rank),
        int(ep_size),
        int(hidden_dim),
        int(element_size),
        int(capacity_peer_rows),
    )


def _standard_ep_mori_native_compact_op_cache_prefix(
    key: tuple[object, ...],
) -> tuple[object, ...]:
    return key[:6]


def _standard_ep_mori_native_compact_role(stage: str) -> int:
    if not _standard_ep_mori_native_compact_role_split():
        return 0
    stage = str(stage)
    if stage in {"dispatch", "backward_dispatch"}:
        return 1
    if stage in {"combine", "backward_combine"}:
        return 2
    return 0


def _standard_ep_mori_native_compact_find_cached_op(
    key: tuple[object, ...],
):
    exact = _STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE.get(key)
    if exact is not None:
        return key, exact

    prefix = _standard_ep_mori_native_compact_op_cache_prefix(key)
    required_capacity = int(key[6])
    compatible = [
        cached_key
        for cached_key in _STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE
        if _standard_ep_mori_native_compact_op_cache_prefix(cached_key) == prefix
        and int(cached_key[6]) >= required_capacity
    ]
    if not compatible:
        return key, None
    best_key = min(compatible, key=lambda cached_key: int(cached_key[6]))
    return best_key, _STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE[best_key]


def _standard_ep_mori_native_compact_store_cached_op(
    key: tuple[object, ...],
    op,
) -> None:
    prefix = _standard_ep_mori_native_compact_op_cache_prefix(key)
    capacity = int(key[6])
    stale_keys = [
        cached_key
        for cached_key in _STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE
        if _standard_ep_mori_native_compact_op_cache_prefix(cached_key) == prefix
        and int(cached_key[6]) < capacity
        and _STANDARD_EP_MORI_NATIVE_COMPACT_OP_REFS.get(cached_key, 0) <= 0
    ]
    for stale_key in stale_keys:
        del _STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE[stale_key]
    if stale_keys and _standard_ep_mori_native_compact_gc_on_release():
        gc.collect()
    _STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE[key] = op


def _standard_ep_mori_native_compact_prune_stale_capacity_handles(
    key: tuple[object, ...],
) -> int:
    """Drop released lower-capacity handles once a compatible larger handle exists.

    During a step's first forward pass, smaller capacity handles can be created
    before the max-capacity bucket appears. Autograd refs correctly keep those
    handles alive until their inverse collective has run, but the perf path's
    keep-released behavior would otherwise retain all lower-capacity MORI
    symmetric buffers. A larger compatible handle can serve future smaller calls,
    so the released smaller handles are pure residency after their refs reach 0.
    """

    key = tuple(key)
    prefix = _standard_ep_mori_native_compact_op_cache_prefix(key)
    compatible_capacities = [
        int(cached_key[6])
        for cached_key in _STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE
        if _standard_ep_mori_native_compact_op_cache_prefix(cached_key) == prefix
    ]
    if not compatible_capacities:
        return 0
    max_capacity = max(compatible_capacities)
    stale_keys = [
        cached_key
        for cached_key in _STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE
        if _standard_ep_mori_native_compact_op_cache_prefix(cached_key) == prefix
        and int(cached_key[6]) < max_capacity
        and _STANDARD_EP_MORI_NATIVE_COMPACT_OP_REFS.get(cached_key, 0) <= 0
    ]
    for stale_key in stale_keys:
        del _STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE[stale_key]
    return len(stale_keys)


def _standard_ep_mori_native_compact_retain_key(key: tuple[object, ...]) -> None:
    if _standard_ep_mori_native_compact_cache_backward_handles():
        return
    key = tuple(key)
    _STANDARD_EP_MORI_NATIVE_COMPACT_OP_REFS[key] = (
        _STANDARD_EP_MORI_NATIVE_COMPACT_OP_REFS.get(key, 0) + 1
    )


def _standard_ep_mori_native_compact_release_key(key: tuple[object, ...]) -> None:
    if _standard_ep_mori_native_compact_cache_backward_handles():
        return
    key = tuple(key)
    refs = _STANDARD_EP_MORI_NATIVE_COMPACT_OP_REFS.get(key, 0)
    if refs > 1:
        _STANDARD_EP_MORI_NATIVE_COMPACT_OP_REFS[key] = refs - 1
        return
    if refs == 1:
        del _STANDARD_EP_MORI_NATIVE_COMPACT_OP_REFS[key]
    pruned = _standard_ep_mori_native_compact_prune_stale_capacity_handles(key)
    if pruned and _standard_ep_mori_native_compact_gc_on_release():
        gc.collect()
    if pruned and _standard_ep_mori_native_compact_debug_cache():
        print(
            "[standard_ep_mori_native_compact_cache] "
            f"pruned_stale_capacity={pruned} after_release_key={key} "
            f"cache_size={len(_STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE)}",
            flush=True,
        )
    # Destroying a MORI op synchronizes in the C++ handle destructor. Keep the
    # perf path as a logical ref release; physically dropping handles is an
    # explicit memory-pressure diagnostic.
    if _standard_ep_mori_native_compact_keep_released_handles():
        return
    op = _STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE.pop(key, None)
    if op is None:
        return
    if _standard_ep_mori_native_compact_gc_on_release():
        gc.collect()
    if _standard_ep_mori_native_compact_debug_cache():
        print(
            "[standard_ep_mori_native_compact_cache] "
            f"released_key={key} cache_size={len(_STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE)}",
            flush=True,
        )


def _standard_ep_mori_native_compact_release_stage(stage: str) -> None:
    if _standard_ep_mori_native_compact_cache_backward_handles():
        return
    stage = str(stage)
    release_keys = [
        key
        for key in _STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE
        if str(key[0]) == stage
        and _STANDARD_EP_MORI_NATIVE_COMPACT_OP_REFS.get(key, 0) <= 0
    ]
    if not release_keys:
        return
    for key in release_keys:
        del _STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE[key]
    if _standard_ep_mori_native_compact_gc_on_release():
        gc.collect()
    if _standard_ep_mori_native_compact_debug_cache():
        print(
            "[standard_ep_mori_native_compact_cache] "
            f"released_stage={stage} released={len(release_keys)} "
            f"cache_size={len(_STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE)}",
            flush=True,
        )


def _standard_ep_mori_native_compact_config(
    *,
    tensor: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    hidden_dim: int,
    num_local_experts: int,
    capacity_peer_rows: int,
    native_block_num: int,
    stage: str,
    EpDispatchCombineConfig,
    EpDispatchCombineKernelType,
):
    return EpDispatchCombineConfig(
        data_type=tensor.dtype,
        rank=int(ep_rank),
        world_size=int(ep_size),
        hidden_dim=int(hidden_dim),
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=int(tensor.element_size()),
        max_num_inp_token_per_rank=int(capacity_peer_rows),
        num_experts_per_rank=max(1, int(num_local_experts)),
        num_experts_per_token=1,
        max_total_recv_tokens=0,
        warp_num_per_block=8,
        block_num=max(1, int(native_block_num)),
        use_external_inp_buf=True,
        kernel_type=EpDispatchCombineKernelType.IntraNode,
        gpu_per_node=int(ep_size),
        rdma_block_num=0,
        num_qp_per_pe=1,
        quant_type="none",
        standard_ep_compact_only=True,
        standard_ep_compact_role=_standard_ep_mori_native_compact_role(stage),
    )


def _standard_ep_mori_native_compact_ensure_stage_ops(
    *,
    tensor: torch.Tensor,
    layout: "StandardEPCompactDispatchLayout",
    ep_rank: int,
    capacity_peer_rows: int,
    local_max_peer_rows: int,
    max_peer_rows: int,
    native_block_num: int,
    EpDispatchCombineConfig,
    EpDispatchCombineKernelType,
    EpDispatchCombineOp,
    stages: tuple[str, ...] = ("dispatch", "combine"),
) -> dict[str, object]:
    """Create native compact handles at a known all-rank point.

    MORI's Python `EpDispatchCombineOp` constructor contains a distributed
    barrier. Creating a missing handle inside autograd backward can deadlock if
    one rank already has that handle and another rank does not.

    Forward dispatch/combine and backward inverse dispatch/combine must not
    share handles. The compact kernels use handle-owned peer-count/signal
    buffers, and the backward inverse collectives run later in the autograd
    graph with different stream/lifetime ordering than the forward pair.

    Forward handles are created at the normal forward all-rank call sites. The
    backward-dispatch handle is created at combine-forward because combine
    backward needs it first. The backward-combine handle is created later by
    combine backward, immediately before dispatch backward consumes it. That
    keeps the autograd graph connected without making all inverse-role MORI
    buffers resident throughout the forward/backward window.
    """

    ops: dict[str, object] = {}
    for stage in stages:
        key = _standard_ep_mori_native_compact_op_cache_key(
            stage=stage,
            device_index=int(tensor.device.index or 0),
            ep_rank=int(ep_rank),
            ep_size=int(layout.ep_size),
            hidden_dim=int(layout.hidden_dim),
            element_size=int(tensor.element_size()),
            capacity_peer_rows=int(capacity_peer_rows),
        )
        cache_key, op = _standard_ep_mori_native_compact_find_cached_op(key)
        if op is None:
            if _standard_ep_mori_native_compact_debug_cache():
                print(
                    "[standard_ep_mori_native_compact_cache] "
                    f"miss={stage} key={key} local_max_peer_rows={local_max_peer_rows} "
                    f"max_peer_rows={max_peer_rows} capacity_peer_rows={capacity_peer_rows} "
                    f"cache_size={len(_STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE)}",
                    flush=True,
                )
            config = _standard_ep_mori_native_compact_config(
                tensor=tensor,
                ep_rank=int(ep_rank),
                ep_size=int(layout.ep_size),
                hidden_dim=int(layout.hidden_dim),
                num_local_experts=int(layout.num_local_experts),
                capacity_peer_rows=int(capacity_peer_rows),
                native_block_num=int(native_block_num),
                stage=str(stage),
                EpDispatchCombineConfig=EpDispatchCombineConfig,
                EpDispatchCombineKernelType=EpDispatchCombineKernelType,
            )
            op = EpDispatchCombineOp(config)
            _standard_ep_mori_native_compact_store_cached_op(key, op)
        elif cache_key != key and _standard_ep_mori_native_compact_debug_cache():
            print(
                "[standard_ep_mori_native_compact_cache] "
                f"reuse={stage} requested_key={key} cached_key={cache_key} "
                f"local_max_peer_rows={local_max_peer_rows} max_peer_rows={max_peer_rows}",
                flush=True,
            )
        ops[stage] = op
    return ops


def _standard_ep_mori_native_compact_ensure_key_op(
    *,
    tensor: torch.Tensor,
    key: tuple[object, ...],
    num_local_experts: int,
    native_block_num: int,
    local_max_peer_rows: int,
    max_peer_rows: int,
    EpDispatchCombineConfig,
    EpDispatchCombineKernelType,
    EpDispatchCombineOp,
):
    """Create one role handle from a cache key at an all-rank call site."""

    cache_key, op = _standard_ep_mori_native_compact_find_cached_op(key)
    if op is not None:
        if cache_key != key and _standard_ep_mori_native_compact_debug_cache():
            print(
                "[standard_ep_mori_native_compact_cache] "
                f"reuse={key[0]} requested_key={key} cached_key={cache_key} "
                f"local_max_peer_rows={local_max_peer_rows} max_peer_rows={max_peer_rows}",
                flush=True,
            )
        return cache_key, op

    if _standard_ep_mori_native_compact_debug_cache():
        print(
            "[standard_ep_mori_native_compact_cache] "
            f"miss={key[0]} key={key} local_max_peer_rows={local_max_peer_rows} "
            f"max_peer_rows={max_peer_rows} capacity_peer_rows={int(key[6])} "
            f"cache_size={len(_STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE)}",
            flush=True,
        )
    config = _standard_ep_mori_native_compact_config(
        tensor=tensor,
        ep_rank=int(key[2]),
        ep_size=int(key[3]),
        hidden_dim=int(key[4]),
        num_local_experts=int(num_local_experts),
        capacity_peer_rows=int(key[6]),
        native_block_num=int(native_block_num),
        stage=str(key[0]),
        EpDispatchCombineConfig=EpDispatchCombineConfig,
        EpDispatchCombineKernelType=EpDispatchCombineKernelType,
    )
    op = EpDispatchCombineOp(config)
    _standard_ep_mori_native_compact_store_cached_op(key, op)
    return key, op


def _build_standard_ep_compact_dispatch_layout(
    *,
    x: torch.Tensor,
    top_k: int,
    ep_size: int,
    num_local_experts: int,
    normal_flat_positions_experts_sorted: torch.Tensor | None,
    num_tokens_per_expert: torch.Tensor,
    num_tokens_per_expert_group_rank_major: torch.Tensor,
    input_splits_list: list[int],
    output_splits_list: list[int],
    rank_major_input_shape: tuple[int, ...],
) -> "StandardEPCompactDispatchLayout | None":
    if normal_flat_positions_experts_sorted is None:
        return None
    return StandardEPCompactDispatchLayout(
        local_flat_positions_experts_sorted=normal_flat_positions_experts_sorted,
        local_num_tokens_per_expert=num_tokens_per_expert.to(torch.int64).contiguous(),
        recv_counts_rank_major=num_tokens_per_expert_group_rank_major.to(
            torch.int64
        ).contiguous(),
        input_splits_tensor=torch.tensor(
            input_splits_list, dtype=torch.int64, device=x.device
        ),
        output_splits_tensor=torch.tensor(
            output_splits_list, dtype=torch.int64, device=x.device
        ),
        input_splits=tuple(int(v) for v in input_splits_list),
        output_splits=tuple(int(v) for v in output_splits_list),
        ep_size=int(ep_size),
        num_local_experts=int(num_local_experts),
        top_k=int(top_k),
        local_token_count=int(x.shape[0]),
        hidden_dim=int(x.shape[-1]),
        rank_major_input_shape=rank_major_input_shape,
    )


def _standard_ep_mori_native_compact_runtime():
    """Return the required native compact MORI symbols, or raise.

    This deliberately rejects the existing raw-route/route-mask MORI API. The
    standard-EP hot-helper path has already formed compact cold/normal rows and
    split metadata; a native MORI backend must consume that layout directly.
    """

    global _STANDARD_EP_MORI_NATIVE_COMPACT_RUNTIME_CACHE
    if _STANDARD_EP_MORI_NATIVE_COMPACT_RUNTIME_CACHE is not None:
        return _STANDARD_EP_MORI_NATIVE_COMPACT_RUNTIME_CACHE

    try:
        from mori import cpp as mori_cpp
    except ImportError as exc:
        raise RuntimeError(
            "mori_native compact EP requires the MORI Python extension in the "
            "active runtime."
        ) from exc

    required = (
        "launch_standard_ep_compact_dispatch",
        "launch_standard_ep_compact_combine",
        "get_standard_ep_compact_dispatch_output",
        "get_standard_ep_compact_combine_output",
    )
    missing = [name for name in required if getattr(mori_cpp, name, None) is None]
    if missing:
        raise RuntimeError(
            "mori_native compact EP was requested, but the active MORI build "
            "does not expose the native compact-row ABI. Missing symbols: "
            f"{missing}. This path must be implemented as a MORI primitive "
            "that consumes StandardEPCompactDispatchLayout; do not replace it "
            "with raw route_mask dispatch or a post-plan row materializer."
        )

    _STANDARD_EP_MORI_NATIVE_COMPACT_RUNTIME_CACHE = tuple(
        getattr(mori_cpp, name) for name in required
    )
    return _STANDARD_EP_MORI_NATIVE_COMPACT_RUNTIME_CACHE


def _standard_ep_compact_rows_balanced_moe_module(
    preferred_module_name: str | None,
    *,
    backend_label: str,
):
    """Return a backend module that owns already-compact EP row movement."""

    cache_key = str(preferred_module_name or "")
    cached = _STANDARD_EP_COMPACT_ROWS_BALANCED_MOE_MODULE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    module = _select_standard_ep_balanced_moe_module_with_capability(
        preferred_module_name,
        "balanced_moe_compact_rows_dispatch",
    )
    if module is None:
        raise RuntimeError(
            f"{backend_label} compact EP requires a balanced-MoE backend module "
            "advertising balanced_moe_compact_rows_dispatch. This is the "
            "already-compact row ABI; do not route it through a high-level "
            "raw top-k compact wrapper."
        )
    module_name = str(getattr(module, "__name__", ""))
    if not _standard_ep_balanced_moe_module_supports_capability(
        module_name,
        module,
        "balanced_moe_compact_rows_combine",
    ):
        raise RuntimeError(
            f"{backend_label} compact EP selected {module_name!r}, but that "
            "backend does not advertise balanced_moe_compact_rows_combine."
        )

    required = (
        "dispatch_balanced_moe_compact_rows",
        "combine_balanced_moe_compact_rows",
    )
    missing = [name for name in required if getattr(module, name, None) is None]
    if missing:
        raise RuntimeError(
            f"{backend_label} compact EP selected {module_name!r}, but the "
            f"module is missing compact-row wrappers {missing}; rebuild or "
            "restage the backend."
        )

    _STANDARD_EP_COMPACT_ROWS_BALANCED_MOE_MODULE_CACHE[cache_key] = module
    return module


def _standard_ep_mori_balanced_moe_module():
    """Return the MORI balanced-MoE module that owns compact row movement."""

    return _standard_ep_compact_rows_balanced_moe_module(
        "mori.ops.balanced_moe",
        backend_label="mori_native",
    )


class _StandardEPNativeCompactDispatchAutograd(torch.autograd.Function):
    """Autograd bridge for native compact MORI dispatch.

    MORI's native compact launch is a pybind data-pointer call, so PyTorch does
    not see the output rows as a differentiable view of the input rows unless we
    explicitly define the reverse collective.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        local_rows: torch.Tensor,
        local_flat_positions: torch.Tensor,
        local_num_tokens_per_expert: torch.Tensor,
        recv_counts_rank_major: torch.Tensor,
        input_splits_tensor: torch.Tensor,
        output_splits_tensor: torch.Tensor,
        expert_major_to_rank_major_indices: torch.Tensor | None,
        dispatch_op,
        backward_combine_key,
        backward_combine_op,
        num_output_rows: int,
        flat_position_rank_stride: int,
        block_num: int,
        warp_per_block: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expert_major_rows, expert_major_flat_positions = (
            _standard_ep_mori_balanced_moe_module().dispatch_balanced_moe_compact_rows(
                dispatch_op,
                local_rows,
                local_flat_positions,
                local_num_tokens_per_expert,
                recv_counts_rank_major,
                input_splits_tensor,
                output_splits_tensor,
                int(num_output_rows),
                int(flat_position_rank_stride),
                block_num=int(block_num),
                warp_per_block=int(warp_per_block),
            )
        )
        ctx.backward_combine_key = backward_combine_key
        ctx.backward_combine_op = backward_combine_op
        _standard_ep_mori_native_compact_retain_key(backward_combine_key)
        ctx.num_input_rows = int(local_rows.shape[0])
        ctx.block_num = int(block_num)
        ctx.warp_per_block = int(warp_per_block)
        ctx.has_expert_major_to_rank_major_indices = (
            expert_major_to_rank_major_indices is not None
        )
        if expert_major_to_rank_major_indices is None:
            expert_major_to_rank_major_indices = recv_counts_rank_major.new_empty((0,))
        ctx.save_for_backward(
            expert_major_to_rank_major_indices,
            recv_counts_rank_major,
            output_splits_tensor,
            input_splits_tensor,
        )
        return expert_major_rows, expert_major_flat_positions

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_expert_major_rows: torch.Tensor | None,
        _grad_expert_major_flat_positions: torch.Tensor | None,
    ):
        def release_backward_combine() -> None:
            _standard_ep_mori_native_compact_release_key(ctx.backward_combine_key)
            ctx.backward_combine_op = None

        if grad_expert_major_rows is None:
            grad_local_rows = None
            release_backward_combine()
        else:
            (
                expert_major_to_rank_major_indices,
                recv_counts_rank_major,
                output_splits_tensor,
                input_splits_tensor,
            ) = ctx.saved_tensors
            if not bool(ctx.has_expert_major_to_rank_major_indices):
                expert_major_to_rank_major_indices = None
            records = _standard_ep_stage_timing_records()
            grad_rows_for_native = grad_expert_major_rows
            grad_rows_was_contiguous = bool(grad_rows_for_native.is_contiguous())
            if not grad_rows_was_contiguous:
                with dsv4_timed_stage(
                    records,
                    "standard_ep.dispatch.mori_native_compact."
                    "backward_combine.grad_contiguous",
                ):
                    grad_rows_for_native = grad_rows_for_native.contiguous()
            with dsv4_timed_stage(
                records, "standard_ep.dispatch.mori_native_compact.backward_combine"
            ):
                backward_combine_op = ctx.backward_combine_op
                if backward_combine_op is None:
                    _, backward_combine_op = (
                        _standard_ep_mori_native_compact_find_cached_op(
                            ctx.backward_combine_key
                        )
                    )
                if backward_combine_op is None:
                    raise RuntimeError(
                        "mori_native compact dispatch backward reached without "
                        "a cached backward combine handle. Native combine "
                        "backward must create the handle before dispatch "
                        "backward consumes its gradient."
                    )
                profile_state = _standard_ep_mori_native_compact_profile_begin(
                    backward_combine_op, "dispatch_backward_combine"
                )
                try:
                    grad_local_rows, _ = (
                        _standard_ep_mori_balanced_moe_module().combine_balanced_moe_compact_rows(
                            backward_combine_op,
                            grad_rows_for_native,
                            None,
                            expert_major_to_rank_major_indices,
                            recv_counts_rank_major,
                            output_splits_tensor,
                            input_splits_tensor,
                            int(ctx.num_input_rows),
                            block_num=int(ctx.block_num),
                            warp_per_block=int(ctx.warp_per_block),
                            return_flat_positions=False,
                        )
                    )
                finally:
                    _standard_ep_mori_native_compact_profile_end(profile_state)
            release_backward_combine()
            flush_dsv4_profile_timing(
                records,
                {
                    "kind": "standard_ep_mori_native_compact_autograd",
                    "phase": "backward_dispatch_inverse",
                    "num_input_rows": int(ctx.num_input_rows),
                    "grad_rows": int(grad_expert_major_rows.shape[0]),
                    "grad_rows_was_contiguous": bool(grad_rows_was_contiguous),
                },
            )
        return (
            grad_local_rows,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _StandardEPNativeCompactCombineAutograd(torch.autograd.Function):
    """Autograd bridge for native compact MORI combine."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        expert_major_rows: torch.Tensor,
        expert_major_flat_positions: torch.Tensor,
        expert_major_to_rank_major_indices: torch.Tensor | None,
        local_num_tokens_per_expert: torch.Tensor,
        recv_counts_rank_major: torch.Tensor,
        input_splits_tensor: torch.Tensor,
        output_splits_tensor: torch.Tensor,
        combine_op,
        backward_dispatch_op,
        backward_dispatch_key,
        backward_combine_key,
        reuse_forward_handles_for_backward: bool,
        num_local_experts: int,
        local_max_peer_rows: int,
        max_peer_rows: int,
        num_output_rows: int,
        num_expert_major_rows: int,
        flat_position_rank_stride: int,
        ep_rank: int,
        block_num: int,
        warp_per_block: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_rank_rows, source_rank_flat_positions = (
            _standard_ep_mori_balanced_moe_module().combine_balanced_moe_compact_rows(
                combine_op,
                expert_major_rows,
                expert_major_flat_positions,
                expert_major_to_rank_major_indices,
                recv_counts_rank_major,
                input_splits_tensor,
                output_splits_tensor,
                int(num_output_rows),
                block_num=int(block_num),
                warp_per_block=int(warp_per_block),
            )
        )
        ctx.backward_dispatch_op = backward_dispatch_op
        ctx.backward_dispatch_key = backward_dispatch_key
        ctx.backward_combine_key = backward_combine_key
        ctx.reuse_forward_handles_for_backward = bool(
            reuse_forward_handles_for_backward
        )
        _standard_ep_mori_native_compact_retain_key(backward_dispatch_key)
        ctx.num_local_experts = int(num_local_experts)
        ctx.local_max_peer_rows = int(local_max_peer_rows)
        ctx.max_peer_rows = int(max_peer_rows)
        ctx.num_expert_major_rows = int(num_expert_major_rows)
        ctx.block_num = int(block_num)
        ctx.warp_per_block = int(warp_per_block)
        ctx.save_for_backward(
            local_num_tokens_per_expert,
            recv_counts_rank_major,
            output_splits_tensor,
            input_splits_tensor,
        )
        return source_rank_rows, source_rank_flat_positions

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_source_rank_rows: torch.Tensor | None,
        _grad_source_rank_flat_positions: torch.Tensor | None,
    ):
        def release_backward_dispatch() -> None:
            ctx.backward_dispatch_op = None
            _standard_ep_mori_native_compact_release_key(ctx.backward_dispatch_key)

        if grad_source_rank_rows is None:
            grad_expert_major_rows = None
            release_backward_dispatch()
        else:
            (
                local_num_tokens_per_expert,
                recv_counts_rank_major,
                output_splits_tensor,
                input_splits_tensor,
            ) = ctx.saved_tensors
            records = _standard_ep_stage_timing_records()
            grad_rows_for_native = grad_source_rank_rows
            grad_rows_was_contiguous = bool(grad_rows_for_native.is_contiguous())
            if not grad_rows_was_contiguous:
                with dsv4_timed_stage(
                    records,
                    "standard_ep.combine.mori_native_compact."
                    "backward_dispatch.grad_contiguous",
                ):
                    grad_rows_for_native = grad_rows_for_native.contiguous()
            with dsv4_timed_stage(
                records, "standard_ep.combine.mori_native_compact.backward_dispatch"
            ):
                if not ctx.reuse_forward_handles_for_backward:
                    from mori.ops.dispatch_combine import (
                        EpDispatchCombineConfig,
                        EpDispatchCombineKernelType,
                        EpDispatchCombineOp,
                    )

                    _standard_ep_mori_native_compact_ensure_key_op(
                        tensor=grad_source_rank_rows,
                        key=ctx.backward_combine_key,
                        num_local_experts=int(ctx.num_local_experts),
                        native_block_num=int(ctx.block_num),
                        local_max_peer_rows=int(ctx.local_max_peer_rows),
                        max_peer_rows=int(ctx.max_peer_rows),
                        EpDispatchCombineConfig=EpDispatchCombineConfig,
                        EpDispatchCombineKernelType=EpDispatchCombineKernelType,
                        EpDispatchCombineOp=EpDispatchCombineOp,
                    )
                profile_state = _standard_ep_mori_native_compact_profile_begin(
                    ctx.backward_dispatch_op, "combine_backward_dispatch"
                )
                try:
                    grad_expert_major_rows, _ = (
                        _standard_ep_mori_balanced_moe_module().dispatch_balanced_moe_compact_rows(
                            ctx.backward_dispatch_op,
                            grad_rows_for_native,
                            None,
                            local_num_tokens_per_expert,
                            recv_counts_rank_major,
                            output_splits_tensor,
                            input_splits_tensor,
                            int(ctx.num_expert_major_rows),
                            0,
                            block_num=int(ctx.block_num),
                            warp_per_block=int(ctx.warp_per_block),
                            return_flat_positions=False,
                        )
                    )
                finally:
                    _standard_ep_mori_native_compact_profile_end(profile_state)
            release_backward_dispatch()
            flush_dsv4_profile_timing(
                records,
                {
                    "kind": "standard_ep_mori_native_compact_autograd",
                    "phase": "backward_combine_inverse",
                    "num_expert_major_rows": int(ctx.num_expert_major_rows),
                    "grad_rows": int(grad_source_rank_rows.shape[0]),
                    "grad_rows_was_contiguous": bool(grad_rows_was_contiguous),
                },
            )
        return (
            grad_expert_major_rows,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _StandardEPNativeCompactCombineWeightedOutputAutograd(torch.autograd.Function):
    """Native compact combine with MORI-owned forward weighted output.

    MORI writes token-major weighted output directly while still returning
    source-rank rows and flat positions for the existing autograd-correct
    backward path.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        expert_major_rows: torch.Tensor,
        expert_major_flat_positions: torch.Tensor,
        expert_major_to_rank_major_indices: torch.Tensor | None,
        local_num_tokens_per_expert: torch.Tensor,
        recv_counts_rank_major: torch.Tensor,
        input_splits_tensor: torch.Tensor,
        output_splits_tensor: torch.Tensor,
        top_scores_flat: torch.Tensor,
        combine_op,
        backward_dispatch_op,
        backward_dispatch_key,
        backward_combine_key,
        reuse_forward_handles_for_backward: bool,
        num_local_experts: int,
        local_max_peer_rows: int,
        max_peer_rows: int,
        num_output_rows: int,
        num_expert_major_rows: int,
        flat_position_offset: int,
        top_k: int,
        token_output_rows: int,
        block_num: int,
        warp_per_block: int,
    ) -> torch.Tensor:
        source_rank_rows, source_rank_flat_positions, token_output = (
            _standard_ep_mori_balanced_moe_module().combine_balanced_moe_compact_rows(
                combine_op,
                expert_major_rows,
                expert_major_flat_positions,
                expert_major_to_rank_major_indices,
                recv_counts_rank_major,
                input_splits_tensor,
                output_splits_tensor,
                int(num_output_rows),
                block_num=int(block_num),
                warp_per_block=int(warp_per_block),
                top_scores_flat=top_scores_flat,
                top_k=int(top_k),
                flat_position_offset=int(flat_position_offset),
                token_output_rows=int(token_output_rows),
                return_token_output=True,
            )
        )

        ctx.backward_dispatch_op = backward_dispatch_op
        ctx.backward_dispatch_key = backward_dispatch_key
        ctx.backward_combine_key = backward_combine_key
        ctx.reuse_forward_handles_for_backward = bool(
            reuse_forward_handles_for_backward
        )
        _standard_ep_mori_native_compact_retain_key(backward_dispatch_key)
        ctx.num_local_experts = int(num_local_experts)
        ctx.local_max_peer_rows = int(local_max_peer_rows)
        ctx.max_peer_rows = int(max_peer_rows)
        ctx.num_expert_major_rows = int(num_expert_major_rows)
        ctx.flat_position_offset = int(flat_position_offset)
        ctx.top_k = int(top_k)
        ctx.block_num = int(block_num)
        ctx.warp_per_block = int(warp_per_block)
        ctx.save_for_backward(
            local_num_tokens_per_expert,
            recv_counts_rank_major,
            output_splits_tensor,
            input_splits_tensor,
            source_rank_rows,
            source_rank_flat_positions,
            top_scores_flat,
        )
        return token_output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_token_output: torch.Tensor | None,
    ):
        def release_backward_dispatch() -> None:
            ctx.backward_dispatch_op = None
            _standard_ep_mori_native_compact_release_key(ctx.backward_dispatch_key)

        grad_top_scores_flat = None
        if grad_token_output is None:
            grad_expert_major_rows = None
            release_backward_dispatch()
        else:
            (
                local_num_tokens_per_expert,
                recv_counts_rank_major,
                output_splits_tensor,
                input_splits_tensor,
                source_rank_rows,
                source_rank_flat_positions,
                top_scores_flat,
            ) = ctx.saved_tensors
            records = _standard_ep_stage_timing_records()
            native_backward_fn = getattr(
                ctx.backward_dispatch_op,
                "standard_ep_compact_weighted_output_backward_native",
                None,
            )
            use_native_backward = bool(
                _standard_ep_mori_native_compact_weighted_output_backward()
                and native_backward_fn is not None
            )
            if use_native_backward:
                grad_token_for_native = grad_token_output
                grad_token_was_contiguous = bool(grad_token_for_native.is_contiguous())
                if not grad_token_was_contiguous:
                    with dsv4_timed_stage(
                        records,
                        "standard_ep.combine.mori_native_compact."
                        "weighted_output_backward_native.grad_contiguous",
                    ):
                        grad_token_for_native = grad_token_for_native.contiguous()
                with dsv4_timed_stage(
                    records,
                    "standard_ep.combine.mori_native_compact."
                    "weighted_output_backward_native_scatter",
                ):
                    grad_source_rank_rows, grad_top_scores_flat = native_backward_fn(
                        source_rank_rows,
                        source_rank_flat_positions,
                        top_scores_flat,
                        grad_token_for_native,
                        top_k=int(ctx.top_k),
                        flat_position_offset=int(ctx.flat_position_offset),
                        block_num=int(ctx.block_num),
                        warp_per_block=int(ctx.warp_per_block),
                    )
            else:
                grad_token_was_contiguous = True
                with dsv4_timed_stage(
                    records,
                    "standard_ep.combine.mori_native_compact."
                    "weighted_output_backward_scatter",
                ):
                    grad_source_rank_rows, grad_top_scores_flat = (
                        _weighted_scatter_flat_backward(
                            source_rank_flat_positions,
                            source_rank_rows,
                            top_scores_flat,
                            grad_token_output,
                            top_k=int(ctx.top_k),
                            flat_position_offset=int(ctx.flat_position_offset),
                        )
                    )

            grad_rows_for_native = grad_source_rank_rows
            grad_rows_was_contiguous = bool(grad_rows_for_native.is_contiguous())
            if not grad_rows_was_contiguous:
                with dsv4_timed_stage(
                    records,
                    "standard_ep.combine.mori_native_compact."
                    "weighted_output_backward_dispatch.grad_contiguous",
                ):
                    grad_rows_for_native = grad_rows_for_native.contiguous()
            with dsv4_timed_stage(
                records,
                "standard_ep.combine.mori_native_compact."
                "weighted_output_backward_dispatch",
            ):
                if not ctx.reuse_forward_handles_for_backward:
                    from mori.ops.dispatch_combine import (
                        EpDispatchCombineConfig,
                        EpDispatchCombineKernelType,
                        EpDispatchCombineOp,
                    )

                    _standard_ep_mori_native_compact_ensure_key_op(
                        tensor=grad_source_rank_rows,
                        key=ctx.backward_combine_key,
                        num_local_experts=int(ctx.num_local_experts),
                        native_block_num=int(ctx.block_num),
                        local_max_peer_rows=int(ctx.local_max_peer_rows),
                        max_peer_rows=int(ctx.max_peer_rows),
                        EpDispatchCombineConfig=EpDispatchCombineConfig,
                        EpDispatchCombineKernelType=EpDispatchCombineKernelType,
                        EpDispatchCombineOp=EpDispatchCombineOp,
                    )
                profile_state = _standard_ep_mori_native_compact_profile_begin(
                    ctx.backward_dispatch_op,
                    "combine_weighted_output_backward_dispatch",
                )
                try:
                    grad_expert_major_rows, _ = (
                        _standard_ep_mori_balanced_moe_module().dispatch_balanced_moe_compact_rows(
                            ctx.backward_dispatch_op,
                            grad_rows_for_native,
                            None,
                            local_num_tokens_per_expert,
                            recv_counts_rank_major,
                            output_splits_tensor,
                            input_splits_tensor,
                            int(ctx.num_expert_major_rows),
                            0,
                            block_num=int(ctx.block_num),
                            warp_per_block=int(ctx.warp_per_block),
                            return_flat_positions=False,
                        )
                    )
                finally:
                    _standard_ep_mori_native_compact_profile_end(profile_state)
            release_backward_dispatch()
            flush_dsv4_profile_timing(
                records,
                {
                    "kind": "standard_ep_mori_native_compact_autograd",
                    "phase": "backward_combine_weighted_output_inverse",
                    "num_expert_major_rows": int(ctx.num_expert_major_rows),
                    "grad_rows": int(grad_source_rank_rows.shape[0]),
                    "grad_rows_was_contiguous": bool(grad_rows_was_contiguous),
                },
            )

        return (
            grad_expert_major_rows,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_top_scores_flat,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _standard_ep_native_mori_compact_dispatch(
    *,
    routed_input: torch.Tensor,
    layout: "StandardEPCompactDispatchLayout",
    expert_major_to_rank_major_indices: torch.Tensor | None,
    ep_mesh: DeviceMesh,
    timing_records: list[dict[str, object]] | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object] | None]:
    """Native compact-row MORI dispatch hook.

    The body is intentionally guarded until MORI exposes the required ABI. The
    contract is:

    - input rows are already source-rank local, expert-sorted compact rows;
    - `layout.input_splits`/`layout.output_splits` are the token-collective ABI;
    - `layout.local_flat_positions_experts_sorted` is the source scatter key;
    - MORI must return rank-major compact rows and retain source positions for
      native compact combine.
    """

    with dsv4_timed_stage(timing_records, "standard_ep.dispatch.mori_native_compact.init"):
        with _standard_ep_mori_native_compact_init_stage(
            timing_records, "standard_ep.dispatch.mori_native_compact.init.runtime"
        ):
            _standard_ep_mori_native_compact_runtime()
        if not routed_input.is_contiguous():
            raise ValueError(
                "mori_native compact dispatch requires already-contiguous compact rows; "
                "do not insert a hidden contiguous() copy in this path."
            )
        if layout.local_flat_positions_experts_sorted.dtype != torch.int64:
            raise TypeError("mori_native compact dispatch requires int64 flat positions")
        if not layout.local_flat_positions_experts_sorted.is_contiguous():
            raise ValueError("mori_native compact flat positions must be contiguous")
        if not layout.local_num_tokens_per_expert.is_contiguous():
            raise ValueError("mori_native compact local expert counts must be contiguous")
        if layout.input_splits_tensor.device != routed_input.device:
            raise ValueError("mori_native compact input splits must be on the row device")
        if layout.output_splits_tensor.device != routed_input.device:
            raise ValueError("mori_native compact output splits must be on the row device")

        with _standard_ep_mori_native_compact_init_stage(
            timing_records, "standard_ep.dispatch.mori_native_compact.init.imports"
        ):
            import mori
            from mori.ops.dispatch_combine import (
                EpDispatchCombineConfig,
                EpDispatchCombineKernelType,
                EpDispatchCombineOp,
            )

            from .mori_aiter_moe import _ensure_mori_process_group

        ep_group = ep_mesh.get_group()
        ep_rank = int(dist.get_rank(group=ep_group))

        with _standard_ep_mori_native_compact_init_stage(
            timing_records, "standard_ep.dispatch.mori_native_compact.init.process_group"
        ):
            _ensure_mori_process_group(mori, "torchtitan_mori_ep", ep_group)

        with _standard_ep_mori_native_compact_init_stage(
            timing_records, "standard_ep.dispatch.mori_native_compact.init.capacity"
        ):
            local_max_peer_rows = max(
                1,
                max((int(v) for v in layout.input_splits), default=0),
                max((int(v) for v in layout.output_splits), default=0),
            )
            remote_max_peer_rows = _standard_ep_mori_native_compact_remote_peer_rows(
                layout.input_splits,
                layout.output_splits,
                ep_rank,
            )
            native_block_num = _standard_ep_mori_native_compact_block_num()
            native_warp_per_block = _standard_ep_mori_native_compact_warp_per_block()
            max_peer_rows, capacity_peer_rows = (
                _standard_ep_mori_native_compact_capacity_for_call(
                    local_max_peer_rows=remote_max_peer_rows,
                    device=routed_input.device,
                    ep_group=ep_group,
                )
            )
        native_summary = None
        if timing_records is not None:
            with _standard_ep_mori_native_compact_init_stage(
                timing_records, "standard_ep.dispatch.mori_native_compact.init.summary"
            ):
                input_stats = _standard_ep_split_stats(layout.input_splits)
                output_stats = _standard_ep_split_stats(layout.output_splits)
                active_peer_rows = max(
                    1,
                    int(input_stats["max"]),
                    int(output_stats["max"]),
                )
                native_summary = {
                    "stage": "dispatch",
                    "ep_rank": int(ep_rank),
                    "ep_size": int(layout.ep_size),
                    "num_local_experts": int(layout.num_local_experts),
                    "hidden_dim": int(layout.hidden_dim),
                    "element_size": int(routed_input.element_size()),
                    "input_rows": int(routed_input.shape[0]),
                    "output_rows": int(layout.rank_major_input_shape[0]),
                    "input_splits": input_stats,
                    "output_splits": output_stats,
                    "local_max_peer_rows": int(local_max_peer_rows),
                    "remote_max_peer_rows": int(remote_max_peer_rows),
                    "active_peer_rows": int(active_peer_rows),
                    "max_peer_rows": int(max_peer_rows),
                    "capacity_peer_rows": int(capacity_peer_rows),
                    "capacity_over_active": (
                        float(capacity_peer_rows) / float(active_peer_rows)
                        if active_peer_rows > 0
                        else 0.0
                    ),
                    "capacity_over_remote": (
                        float(capacity_peer_rows) / float(remote_max_peer_rows)
                        if remote_max_peer_rows > 0
                        else 0.0
                    ),
                    "native_block_num": int(native_block_num),
                    "native_warp_per_block": int(native_warp_per_block),
                    "role_split": bool(_standard_ep_mori_native_compact_role_split()),
                    "reuse_forward_handles_for_backward": bool(
                        _standard_ep_mori_native_compact_reuse_forward_handles_for_backward()
                    ),
                    "keep_released_handles": bool(
                        _standard_ep_mori_native_compact_keep_released_handles()
                    ),
                }
        with _standard_ep_mori_native_compact_init_stage(
            timing_records, "standard_ep.dispatch.mori_native_compact.init.ensure_ops"
        ):
            ops = _standard_ep_mori_native_compact_ensure_stage_ops(
                tensor=routed_input,
                layout=layout,
                ep_rank=ep_rank,
                capacity_peer_rows=int(capacity_peer_rows),
                local_max_peer_rows=int(remote_max_peer_rows),
                max_peer_rows=int(max_peer_rows),
                native_block_num=int(native_block_num),
                EpDispatchCombineConfig=EpDispatchCombineConfig,
                EpDispatchCombineKernelType=EpDispatchCombineKernelType,
                EpDispatchCombineOp=EpDispatchCombineOp,
                stages=("dispatch", "combine"),
            )
        op = ops["dispatch"]
        reuse_forward_handles_for_backward = (
            _standard_ep_mori_native_compact_reuse_forward_handles_for_backward()
        )
        backward_combine_stage = (
            "combine" if reuse_forward_handles_for_backward else "backward_combine"
        )
        backward_combine_key = _standard_ep_mori_native_compact_op_cache_key(
            stage=backward_combine_stage,
            device_index=int(routed_input.device.index or 0),
            ep_rank=ep_rank,
            ep_size=int(layout.ep_size),
            hidden_dim=int(layout.hidden_dim),
            element_size=int(routed_input.element_size()),
            capacity_peer_rows=int(capacity_peer_rows),
        )
        backward_combine_op = None
        if reuse_forward_handles_for_backward:
            with _standard_ep_mori_native_compact_init_stage(
                timing_records,
                "standard_ep.dispatch.mori_native_compact.init.backward_handle",
            ):
                backward_combine_key, backward_combine_op = (
                    _standard_ep_mori_native_compact_find_cached_op(
                        backward_combine_key
                    )
                )
            if backward_combine_op is None:
                raise RuntimeError(
                    "mori_native compact shared-handle dispatch reached without "
                    "a cached forward combine handle."
                )

    _standard_ep_mori_native_compact_prelaunch_reset(
        stage_prefix="standard_ep.dispatch.mori_native_compact",
        op=op,
        ep_group=ep_mesh.get_group(),
        timing_records=timing_records,
    )
    with dsv4_timed_stage(timing_records, "standard_ep.dispatch.mori_native_compact.launch"):
        profile_state = _standard_ep_mori_native_compact_profile_begin(
            op, "dispatch"
        )
        result = _StandardEPNativeCompactDispatchAutograd.apply(
            routed_input,
            layout.local_flat_positions_experts_sorted,
            layout.local_num_tokens_per_expert,
            layout.recv_counts_rank_major,
            layout.input_splits_tensor,
            layout.output_splits_tensor,
            expert_major_to_rank_major_indices,
            op,
            backward_combine_key,
            backward_combine_op,
            int(layout.rank_major_input_shape[0]),
            int(layout.local_token_count) * int(layout.top_k),
            int(native_block_num),
            int(native_warp_per_block),
        )
        _standard_ep_mori_native_compact_profile_end(profile_state)
        expert_major_rows, expert_major_flat_positions = result
        return expert_major_rows, expert_major_flat_positions, native_summary


def _standard_ep_native_mori_compact_combine(
    *,
    routed_output: torch.Tensor,
    metadata: "AllToAllDispatchMetadata",
    ep_mesh: DeviceMesh,
    timing_records: list[dict[str, object]] | None,
    weighted_output: bool = False,
    top_scores_flat: torch.Tensor | None = None,
    top_k: int = 0,
    token_output_rows: int = 0,
    flat_position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, object] | None]:
    with dsv4_timed_stage(timing_records, "standard_ep.combine.mori_native_compact.init"):
        with _standard_ep_mori_native_compact_init_stage(
            timing_records, "standard_ep.combine.mori_native_compact.init.runtime"
        ):
            _standard_ep_mori_native_compact_runtime()
        layout = metadata.normal_compact_dispatch_layout
        if layout is None:
            raise RuntimeError("mori_native compact combine requires saved dispatch layout")
        if metadata.normal_recv_global_flat_positions_expert_major is None:
            raise RuntimeError(
                "mori_native compact combine requires native dispatch source positions"
            )
        if not routed_output.is_contiguous():
            raise ValueError(
                "mori_native compact combine requires contiguous expert-major rows"
            )
        if weighted_output:
            if top_scores_flat is None:
                raise RuntimeError(
                    "mori_native compact weighted output requires top_scores_flat"
                )
            if not top_scores_flat.is_contiguous():
                raise ValueError(
                    "mori_native compact weighted output requires contiguous scores"
                )
            if int(top_k) <= 0:
                raise ValueError("mori_native compact weighted output requires top_k > 0")
            if int(token_output_rows) <= 0:
                raise ValueError(
                    "mori_native compact weighted output requires token_output_rows > 0"
                )

        with _standard_ep_mori_native_compact_init_stage(
            timing_records, "standard_ep.combine.mori_native_compact.init.imports"
        ):
            import mori
            from mori.ops.dispatch_combine import (
                EpDispatchCombineConfig,
                EpDispatchCombineKernelType,
                EpDispatchCombineOp,
            )

            from .mori_aiter_moe import _ensure_mori_process_group

        ep_group = ep_mesh.get_group()
        ep_rank = int(dist.get_rank(group=ep_group))
        with _standard_ep_mori_native_compact_init_stage(
            timing_records, "standard_ep.combine.mori_native_compact.init.process_group"
        ):
            _ensure_mori_process_group(mori, "torchtitan_mori_ep", ep_group)

        with _standard_ep_mori_native_compact_init_stage(
            timing_records, "standard_ep.combine.mori_native_compact.init.capacity"
        ):
            local_max_peer_rows = max(
                1,
                max((int(v) for v in layout.input_splits), default=0),
                max((int(v) for v in layout.output_splits), default=0),
            )
            remote_max_peer_rows = _standard_ep_mori_native_compact_remote_peer_rows(
                layout.input_splits,
                layout.output_splits,
                ep_rank,
            )
            native_block_num = _standard_ep_mori_native_compact_block_num()
            native_warp_per_block = _standard_ep_mori_native_compact_warp_per_block()
            dispatch_summary = metadata.normal_compact_native_dispatch_summary or {}
            dispatch_capacity_peer_rows = int(
                dispatch_summary.get("capacity_peer_rows", 0) or 0
            )
            if dispatch_capacity_peer_rows > 0:
                capacity_peer_rows = dispatch_capacity_peer_rows
                max_peer_rows = int(
                    dispatch_summary.get("max_peer_rows", 0) or remote_max_peer_rows
                )
                capacity_source = "dispatch_summary"
            else:
                max_peer_rows, capacity_peer_rows = (
                    _standard_ep_mori_native_compact_capacity_for_call(
                        local_max_peer_rows=remote_max_peer_rows,
                        device=routed_output.device,
                        ep_group=ep_group,
                    )
                )
                capacity_source = "combine_all_reduce"
        native_summary = None
        if timing_records is not None:
            with _standard_ep_mori_native_compact_init_stage(
                timing_records, "standard_ep.combine.mori_native_compact.init.summary"
            ):
                input_stats = _standard_ep_split_stats(layout.output_splits)
                output_stats = _standard_ep_split_stats(layout.input_splits)
                active_peer_rows = max(
                    1,
                    int(input_stats["max"]),
                    int(output_stats["max"]),
                )
                native_summary = {
                    "stage": "combine",
                    "ep_rank": int(ep_rank),
                    "ep_size": int(layout.ep_size),
                    "num_local_experts": int(layout.num_local_experts),
                    "hidden_dim": int(layout.hidden_dim),
                    "element_size": int(routed_output.element_size()),
                    "input_rows": int(routed_output.shape[0]),
                    "output_rows": int(sum(metadata.input_splits)),
                    "input_splits": input_stats,
                    "output_splits": output_stats,
                    "local_max_peer_rows": int(local_max_peer_rows),
                    "remote_max_peer_rows": int(remote_max_peer_rows),
                    "active_peer_rows": int(active_peer_rows),
                    "max_peer_rows": int(max_peer_rows),
                    "capacity_peer_rows": int(capacity_peer_rows),
                    "capacity_source": str(capacity_source),
                    "capacity_over_active": (
                        float(capacity_peer_rows) / float(active_peer_rows)
                        if active_peer_rows > 0
                        else 0.0
                    ),
                    "capacity_over_remote": (
                        float(capacity_peer_rows) / float(remote_max_peer_rows)
                        if remote_max_peer_rows > 0
                        else 0.0
                    ),
                    "native_block_num": int(native_block_num),
                    "native_warp_per_block": int(native_warp_per_block),
                    "role_split": bool(_standard_ep_mori_native_compact_role_split()),
                    "reuse_forward_handles_for_backward": bool(
                        _standard_ep_mori_native_compact_reuse_forward_handles_for_backward()
                    ),
                    "keep_released_handles": bool(
                        _standard_ep_mori_native_compact_keep_released_handles()
                    ),
                }
        key = _standard_ep_mori_native_compact_op_cache_key(
            stage="combine",
            device_index=int(routed_output.device.index or 0),
            ep_rank=ep_rank,
            ep_size=int(layout.ep_size),
            hidden_dim=int(layout.hidden_dim),
            element_size=int(routed_output.element_size()),
            capacity_peer_rows=int(capacity_peer_rows),
        )
        with _standard_ep_mori_native_compact_init_stage(
            timing_records, "standard_ep.combine.mori_native_compact.init.find_combine"
        ):
            cache_key, op = _standard_ep_mori_native_compact_find_cached_op(key)
        if op is None:
            raise RuntimeError(
                "mori_native compact combine reached without a cached combine "
                "handle. Dispatch must create the forward dispatch/combine "
                "handles at the forward all-rank call site because MORI handle "
                "construction contains a distributed barrier. "
                f"requested_key={key} local_max_peer_rows={local_max_peer_rows} "
                f"max_peer_rows={max_peer_rows} capacity_peer_rows={capacity_peer_rows} "
                f"cache_size={len(_STANDARD_EP_MORI_NATIVE_COMPACT_OP_CACHE)}"
            )
        elif cache_key != key and _standard_ep_mori_native_compact_debug_cache():
            print(
                "[standard_ep_mori_native_compact_cache] "
                f"reuse=combine requested_key={key} cached_key={cache_key} "
                f"local_max_peer_rows={local_max_peer_rows} max_peer_rows={max_peer_rows}",
                flush=True,
            )
        reuse_forward_handles_for_backward = (
            _standard_ep_mori_native_compact_reuse_forward_handles_for_backward()
        )
        backward_dispatch_stage = (
            "dispatch" if reuse_forward_handles_for_backward else "backward_dispatch"
        )
        backward_dispatch_key = _standard_ep_mori_native_compact_op_cache_key(
            stage=backward_dispatch_stage,
            device_index=int(routed_output.device.index or 0),
            ep_rank=ep_rank,
            ep_size=int(layout.ep_size),
            hidden_dim=int(layout.hidden_dim),
            element_size=int(routed_output.element_size()),
            capacity_peer_rows=int(capacity_peer_rows),
        )
        backward_combine_stage = (
            "combine" if reuse_forward_handles_for_backward else "backward_combine"
        )
        backward_combine_key = _standard_ep_mori_native_compact_op_cache_key(
            stage=backward_combine_stage,
            device_index=int(routed_output.device.index or 0),
            ep_rank=ep_rank,
            ep_size=int(layout.ep_size),
            hidden_dim=int(layout.hidden_dim),
            element_size=int(routed_output.element_size()),
            capacity_peer_rows=int(capacity_peer_rows),
        )
        with _standard_ep_mori_native_compact_init_stage(
            timing_records,
            "standard_ep.combine.mori_native_compact.init.backward_handle",
        ):
            backward_dispatch_cache_key, backward_dispatch_op = (
                _standard_ep_mori_native_compact_find_cached_op(backward_dispatch_key)
            )
            if backward_dispatch_op is None:
                if reuse_forward_handles_for_backward:
                    raise RuntimeError(
                        "mori_native compact shared-handle combine reached without "
                        "a cached forward dispatch handle."
                    )
                backward_ops = _standard_ep_mori_native_compact_ensure_stage_ops(
                    tensor=routed_output,
                    layout=layout,
                    ep_rank=ep_rank,
                    capacity_peer_rows=int(capacity_peer_rows),
                    local_max_peer_rows=int(remote_max_peer_rows),
                    max_peer_rows=int(max_peer_rows),
                    native_block_num=int(native_block_num),
                    EpDispatchCombineConfig=EpDispatchCombineConfig,
                    EpDispatchCombineKernelType=EpDispatchCombineKernelType,
                    EpDispatchCombineOp=EpDispatchCombineOp,
                    stages=("backward_dispatch",),
                )
                backward_dispatch_cache_key, backward_dispatch_op = (
                    _standard_ep_mori_native_compact_find_cached_op(
                        backward_dispatch_key
                    )
                )
        if backward_dispatch_op is None:
            raise RuntimeError(
                "mori_native compact combine reached without a cached backward "
                "dispatch handle needed for autograd. Native combine forward "
                "must create the backward-dispatch handle before autograd "
                "backward begins."
            )
        elif (
            backward_dispatch_cache_key != backward_dispatch_key
            and _standard_ep_mori_native_compact_debug_cache()
        ):
            print(
                "[standard_ep_mori_native_compact_cache] "
                "reuse=combine_backward_dispatch "
                f"requested_key={backward_dispatch_key} "
                f"cached_key={backward_dispatch_cache_key}",
                flush=True,
            )
        backward_dispatch_key = backward_dispatch_cache_key

    _standard_ep_mori_native_compact_prelaunch_reset(
        stage_prefix="standard_ep.combine.mori_native_compact",
        op=op,
        ep_group=ep_mesh.get_group(),
        timing_records=timing_records,
    )
    with dsv4_timed_stage(timing_records, "standard_ep.combine.mori_native_compact.launch"):
        profile_state = _standard_ep_mori_native_compact_profile_begin(
            op, "combine"
        )
        if weighted_output:
            token_output = (
                _StandardEPNativeCompactCombineWeightedOutputAutograd.apply(
                    routed_output,
                    metadata.normal_recv_global_flat_positions_expert_major,
                    None,
                    layout.local_num_tokens_per_expert,
                    layout.recv_counts_rank_major,
                    layout.output_splits_tensor,
                    layout.input_splits_tensor,
                    top_scores_flat,
                    op,
                    backward_dispatch_op,
                    backward_dispatch_key,
                    backward_combine_key,
                    bool(reuse_forward_handles_for_backward),
                    int(layout.num_local_experts),
                    int(local_max_peer_rows),
                    int(max_peer_rows),
                    int(sum(metadata.input_splits)),
                    int(routed_output.shape[0]),
                    int(flat_position_offset),
                    int(top_k),
                    int(token_output_rows),
                    int(native_block_num),
                    int(native_warp_per_block),
                )
            )
            _standard_ep_mori_native_compact_profile_end(profile_state)
            return token_output, None, native_summary
        source_rank_rows, source_rank_flat_positions = (
            _StandardEPNativeCompactCombineAutograd.apply(
                routed_output,
                metadata.normal_recv_global_flat_positions_expert_major,
                None,
                layout.local_num_tokens_per_expert,
                layout.recv_counts_rank_major,
                layout.output_splits_tensor,
                layout.input_splits_tensor,
                op,
                backward_dispatch_op,
                backward_dispatch_key,
                backward_combine_key,
                bool(reuse_forward_handles_for_backward),
                int(layout.num_local_experts),
                int(local_max_peer_rows),
                int(max_peer_rows),
                int(sum(metadata.input_splits)),
                int(routed_output.shape[0]),
                int(metadata.normal_flat_position_rank_stride),
                int(ep_rank),
                int(native_block_num),
                int(native_warp_per_block),
            )
        )
        _standard_ep_mori_native_compact_profile_end(profile_state)
    return source_rank_rows, source_rank_flat_positions, native_summary


def _standard_ep_mori_compact_materialize_rows(
    *,
    x: torch.Tensor,
    flat_scores: torch.Tensor,
    remote_positions: torch.Tensor,
    hot_owner_slots: torch.Tensor,
    num_hot_slots: int,
    top_k: int,
    ep_mesh: DeviceMesh,
    timing_records: list[dict[str, object]] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize helper-hot rows with MORI's compact no-count ABI.

    The caller must already have the same presorted `remote_positions` and
    owner-slot layout used by the Torch baseline. Counts/group ends are carried
    from that layout, not recomputed with atomics inside the MORI row-copy
    kernel.
    """
    if remote_positions.numel() == 0:
        return (
            x.new_empty((0, x.shape[-1])),
            flat_scores.new_empty((0,)),
        )

    with dsv4_timed_stage(
        timing_records, "standard_ep.dispatch.hot_materialize_mori_compact.init"
    ):
        if not x.is_contiguous():
            raise ValueError("MORI compact hot-row materialization requires contiguous x")
        if flat_scores.dtype != torch.float32:
            raise TypeError("MORI compact hot-row materialization requires float32 scores")
        if hot_owner_slots.dtype != torch.int64:
            hot_owner_slots = hot_owner_slots.to(torch.int64)
        if not remote_positions.is_contiguous():
            remote_positions = remote_positions.contiguous()
        if not hot_owner_slots.is_contiguous():
            hot_owner_slots = hot_owner_slots.contiguous()

        ep_group = ep_mesh.get_group()
        runtime_key = id(ep_group)
        runtime = _STANDARD_EP_MORI_COMPACT_PACK_RUNTIME_CACHE.get(runtime_key)
        if runtime is None:
            import mori
            from mori import cpp as mori_cpp
            from mori.tensor_utils import dtype_to_int

            from .mori_aiter_moe import _ensure_mori_process_group

            _ensure_mori_process_group(mori, "torchtitan_mori_ep", ep_group)
            pack_fn = getattr(mori_cpp, "launch_hot_helper_compact_pack", None)
            if pack_fn is None:
                raise RuntimeError(
                    "MORI compact hot-row materialization requires "
                    "launch_hot_helper_compact_pack. Rebuild MORI."
                )
            runtime = (mori_cpp, dtype_to_int, pack_fn)
            _STANDARD_EP_MORI_COMPACT_PACK_RUNTIME_CACHE[runtime_key] = runtime
        mori_cpp, dtype_to_int, pack_fn = runtime

        hidden_dim = int(x.shape[-1])
        num_tokens = int(x.shape[0])
        key = (
            int(x.device.index or 0),
            hidden_dim,
            int(x.element_size()),
            num_tokens,
            int(top_k),
        )
        config = _STANDARD_EP_MORI_COMPACT_PACK_CONFIG_CACHE.get(key)
        if config is None:
            config = mori_cpp.EpDispatchCombineConfig(
                rank=0,
                world_size=1,
                hidden_dim=hidden_dim,
                scale_dim=0,
                scale_type_size=0,
                max_token_type_size=x.element_size(),
                max_num_inp_token_per_rank=num_tokens,
                num_experts_per_rank=1,
                num_experts_per_token=int(top_k),
                max_total_recv_tokens=0,
                warp_num_per_block=_standard_ep_mori_compact_pack_warp_per_block(),
                block_num=_standard_ep_mori_compact_pack_block_num(),
                use_external_inp_buf=True,
                kernel_type=mori_cpp.EpDispatchCombineKernelType.IntraNode,
                gpu_per_node=1,
                rdma_block_num=0,
                num_qp_per_pe=1,
                quant_type=mori_cpp.EpDispatchCombineQuantType.None_,
            )
            _STANDARD_EP_MORI_COMPACT_PACK_CONFIG_CACHE[key] = config
        hot_x = torch.empty(
            (int(remote_positions.numel()), hidden_dim),
            dtype=x.dtype,
            device=x.device,
        )
        hot_scores = torch.empty(
            (int(remote_positions.numel()),),
            dtype=torch.float32,
            device=x.device,
        )
        hot_src_info = torch.empty(
            (int(remote_positions.numel()),),
            dtype=torch.int64,
            device=x.device,
        )

    with dsv4_timed_stage(
        timing_records, "standard_ep.dispatch.hot_materialize_mori_compact.pack"
    ):
        pack_fn(
            config,
            dtype_to_int(x.dtype),
            x.data_ptr(),
            flat_scores.data_ptr(),
            remote_positions.data_ptr(),
            hot_owner_slots.data_ptr(),
            int(remote_positions.numel()),
            hot_x.data_ptr(),
            hot_scores.data_ptr(),
            hot_src_info.data_ptr(),
            0,
            int(num_hot_slots),
            _standard_ep_mori_compact_pack_block_num(),
            _standard_ep_mori_compact_pack_warp_per_block(),
            torch.cuda.current_stream().cuda_stream,
            hidden_dim,
        )
    return hot_x, hot_scores


def _standard_ep_defer_hot_reorder() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_DEFER_HOT_REORDER",
        _env_flag("CANARY_STANDARD_EP_DEFER_HOT_REORDER", False),
    )


def _standard_ep_balanced_moe_mode() -> str:
    mode = os.environ.get(
        "TORCHTITAN_MOE_BALANCED_MOE_MODE",
        os.environ.get("TORCHTITAN_MORI_AITER_BALANCED_MOE_MODE", "off"),
    ).strip().lower()
    if mode not in {"off", "plan", "execute"}:
        raise ValueError(
            "TORCHTITAN_MOE_BALANCED_MOE_MODE must be 'off', 'plan', or 'execute', "
            f"got {mode!r}"
        )
    return mode


def _standard_ep_balanced_moe_hot_experts(mode: str) -> int:
    default = 0 if mode == "off" else 1
    return max(
        0,
        _env_int(
            "TORCHTITAN_MOE_BALANCED_MOE_HOT_EXPERTS",
            _env_int("TORCHTITAN_MORI_AITER_BALANCED_MOE_HOT_EXPERTS", default),
        ),
    )


def _standard_ep_balanced_moe_min_reduction_pct() -> float:
    return max(
        0.0,
        _env_float(
            "TORCHTITAN_MOE_BALANCED_MOE_MIN_REDUCTION_PCT",
            _env_float("TORCHTITAN_MORI_AITER_BALANCED_MOE_MIN_REDUCTION_PCT", 0.0),
        ),
    )


def _standard_ep_balanced_moe_planner_backend_names() -> tuple[str, ...]:
    backend = os.environ.get(
        "TORCHTITAN_MOE_BALANCED_MOE_PLANNER_BACKEND",
        os.environ.get(
            "TORCHTITAN_MORI_AITER_BALANCED_MOE_PLANNER_BACKEND",
            os.environ.get(
                "TORCHTITAN_MORI_AITER_BALANCED_MOE_PLAN_BACKEND",
                "auto",
            ),
        ),
    )
    backend = backend.strip().lower().replace("-", "_")
    aliases = {
        "auto": ("mori", "primus_turbo"),
        "mori": ("mori",),
        "primus": ("primus_turbo",),
        "primus_turbo": ("primus_turbo",),
        "torch": (),
        "torchtitan": (),
        "local": (),
        "off": (),
    }
    if backend not in aliases:
        raise ValueError(
            "TORCHTITAN_MOE_BALANCED_MOE_PLANNER_BACKEND must be one of "
            "'auto', 'mori', 'primus_turbo', or 'local', got "
            f"{backend!r}"
        )
    return aliases[backend]


def _import_standard_ep_balanced_moe_planner_backend(backend_name: str):
    module_names = {
        "mori": "mori.ops.balanced_moe",
        "primus_turbo": "primus_turbo.pytorch.ops.moe.balanced_moe",
    }
    module_name = module_names.get(str(backend_name))
    if module_name is None:
        return None
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _import_standard_ep_balanced_moe_module(module_name: str | None):
    if not module_name:
        return None
    try:
        return importlib.import_module(str(module_name))
    except Exception:
        return None


def _iter_standard_ep_balanced_moe_modules(preferred_module_name: str | None = None):
    seen: set[str] = set()
    if preferred_module_name:
        module = _import_standard_ep_balanced_moe_module(preferred_module_name)
        if module is not None:
            seen.add(str(module.__name__))
            yield module
    for backend_name in _standard_ep_balanced_moe_planner_backend_names():
        module = _import_standard_ep_balanced_moe_planner_backend(backend_name)
        if module is None:
            continue
        module_name = str(module.__name__)
        if module_name in seen:
            continue
        seen.add(module_name)
        yield module


def _standard_ep_balanced_moe_module_supports_capability(
    module_name: str,
    backend_module,
    capability: str,
) -> bool:
    capabilities_fn = getattr(
        backend_module,
        "balanced_moe_backend_capabilities",
        None,
    )
    if capabilities_fn is None:
        return False
    try:
        capabilities = capabilities_fn()
    except Exception:
        return False
    if not isinstance(capabilities, dict):
        return False
    backend_label = str(capabilities.get("backend", "")).strip().lower()
    module_name_lower = str(module_name).strip().lower()
    if capability == "normal_topk_ep_dispatch_permute":
        # This capability is the Primus-Turbo raw top-k ABI.  MORI owns the
        # compact-row native ABI instead, so do not let auto-selection route
        # MORI into a Primus-shaped dispatch/permute path.
        if backend_label == "mori" or module_name_lower.startswith("mori."):
            return False
    return bool(capabilities.get(capability, False))


def _select_standard_ep_balanced_moe_module_with_capability(
    preferred_module_name: str | None,
    capability: str,
):
    for backend_module in _iter_standard_ep_balanced_moe_modules(
        preferred_module_name
    ):
        module_name = str(getattr(backend_module, "__name__", ""))
        if _standard_ep_balanced_moe_module_supports_capability(
            module_name,
            backend_module,
            capability,
        ):
            return backend_module
    return None


def _load_summary_from_ints(loads: list[int]) -> dict[str, float | int | list[int]]:
    total = int(sum(loads))
    mean = float(total / max(1, len(loads)))
    max_load = int(max(loads)) if loads else 0
    return {
        "loads": [int(v) for v in loads],
        "mean": mean,
        "max": max_load,
        "max_over_mean": float(max_load / mean) if mean > 0.0 else 0.0,
    }


def _standard_ep_owner_compact_fields_from_selected(
    *,
    selected_global_ids: list[int],
    selected_owner_ranks: list[int],
    ep_size: int,
    num_local_experts: int,
) -> dict[str, object]:
    """Build serializable owner-shard metadata for the local fallback planner."""

    owner_counts = [0 for _ in range(int(ep_size))]
    owner_slots: list[int] = []
    for owner_rank in selected_owner_ranks:
        owner_slot = owner_counts[int(owner_rank)]
        owner_counts[int(owner_rank)] += 1
        owner_slots.append(int(owner_slot))

    max_owned_per_rank = max(owner_counts, default=0)
    owner_shard_offsets = [
        int(owner_rank) * int(max_owned_per_rank) + int(owner_slot)
        for owner_rank, owner_slot in zip(selected_owner_ranks, owner_slots)
    ]
    owner_shard_active_offsets = tuple(sorted(int(v) for v in owner_shard_offsets))
    owner_shard_to_compact = {
        int(offset): int(compact_idx)
        for compact_idx, offset in enumerate(owner_shard_active_offsets)
    }
    owner_compact_offsets = [
        owner_shard_to_compact[int(offset)] for offset in owner_shard_offsets
    ]
    owner_compact_owner_ranks = [0 for _ in owner_shard_active_offsets]
    owner_compact_local_experts = [0 for _ in owner_shard_active_offsets]
    for compact_offset, global_id, owner_rank in zip(
        owner_compact_offsets, selected_global_ids, selected_owner_ranks
    ):
        owner_compact_owner_ranks[int(compact_offset)] = int(owner_rank)
        owner_compact_local_experts[int(compact_offset)] = int(
            global_id
        ) % int(num_local_experts)

    return {
        "standard_ep_balanced_moe_owner_counts": [int(v) for v in owner_counts],
        "standard_ep_balanced_moe_max_owned_per_rank": int(max_owned_per_rank),
        "standard_ep_balanced_moe_selected_owner_shard_offsets": [
            int(v) for v in owner_shard_offsets
        ],
        "standard_ep_balanced_moe_selected_owner_compact_offsets": [
            int(v) for v in owner_compact_offsets
        ],
        "standard_ep_balanced_moe_owner_shard_active_offsets": [
            int(v) for v in owner_shard_active_offsets
        ],
        "standard_ep_balanced_moe_owner_compact_owner_ranks": [
            int(v) for v in owner_compact_owner_ranks
        ],
        "standard_ep_balanced_moe_owner_compact_local_experts": [
            int(v) for v in owner_compact_local_experts
        ],
    }


def _standard_ep_owner_compact_need_masks(
    *,
    selected_source_rank_counts: list[list[int]],
    selected_owner_ranks: list[int],
    selected_owner_compact_offsets: list[int],
    ep_size: int,
    active_count: int,
    backend_module_name: str | None = None,
) -> tuple[tuple[bool, ...], ...]:
    for backend_module in _iter_standard_ep_balanced_moe_modules(backend_module_name):
        build_owner_compact_need_masks = getattr(
            backend_module,
            "build_owner_compact_need_masks",
            None,
        )
        if build_owner_compact_need_masks is None:
            continue
        try:
            return build_owner_compact_need_masks(
                selected_source_rank_counts=selected_source_rank_counts,
                selected_owner_ranks=selected_owner_ranks,
                selected_owner_compact_offsets=selected_owner_compact_offsets,
                world_size=int(ep_size),
                active_owner_compact_count=int(active_count),
            )
        except (TypeError, ValueError):
            continue

    need_masks = [
        [False for _ in range(int(active_count))] for _ in range(int(ep_size))
    ]
    for by_source, owner_rank, compact_offset in zip(
        selected_source_rank_counts,
        selected_owner_ranks,
        selected_owner_compact_offsets,
    ):
        compact_offset = int(compact_offset)
        if compact_offset < 0 or compact_offset >= int(active_count):
            continue
        for source_rank, count in enumerate(by_source):
            if int(source_rank) != int(owner_rank) and int(count) > 0:
                need_masks[int(source_rank)][compact_offset] = True
    return tuple(tuple(bool(v) for v in row) for row in need_masks)


def _standard_ep_owner_compact_exchange_plan(
    *,
    owner_compact_need_masks: tuple[tuple[bool, ...], ...],
    owner_compact_owner_ranks: list[int],
    owner_compact_local_experts: list[int],
    ep_rank: int,
    device: torch.device | None = None,
    backend_module_name: str | None = None,
) -> tuple[dict[str, object], dict[str, object] | None] | None:
    for backend_module in _iter_standard_ep_balanced_moe_modules(backend_module_name):
        build_owner_compact_exchange_plan = getattr(
            backend_module,
            "build_owner_compact_exchange_plan",
            None,
        )
        build_owner_compact_exchange_runtime_plan = getattr(
            backend_module,
            "build_owner_compact_exchange_runtime_plan",
            None,
        )
        if build_owner_compact_exchange_plan is None:
            continue
        try:
            if (
                build_owner_compact_exchange_runtime_plan is not None
                and device is not None
            ):
                plan, plan_tensors = build_owner_compact_exchange_runtime_plan(
                    owner_compact_need_masks=owner_compact_need_masks,
                    owner_compact_owner_ranks=owner_compact_owner_ranks,
                    compact_local_indices=owner_compact_local_experts,
                    rank=int(ep_rank),
                    device=device,
                    dtype=torch.int64,
                    split_dtype=torch.int64,
                )
            else:
                plan = build_owner_compact_exchange_plan(
                    owner_compact_need_masks=owner_compact_need_masks,
                    owner_compact_owner_ranks=owner_compact_owner_ranks,
                    rank=int(ep_rank),
                )
                plan_tensors = None
            if (
                plan_tensors is None
                and device is not None
                and hasattr(plan, "as_tensors")
            ):
                plan_tensors = plan.as_tensors(
                    device=device,
                    dtype=torch.int64,
                    split_dtype=torch.int64,
                )
            return plan.as_dict(), plan_tensors
        except (IndexError, TypeError, ValueError):
            continue
    return None


def _standard_ep_balanced_moe_payload_from_backend_plan(
    *,
    backend_plan: object,
    backend_module_name: str,
    planner_backend: str,
    hot_experts: int,
    include_native_plan: bool,
) -> dict[str, object]:
    before = _load_summary_from_ints(list(backend_plan.owner_load_before))
    after = _load_summary_from_ints(list(backend_plan.exec_load_after))
    payload = {
        "standard_ep_balanced_moe_mode": "plan",
        "standard_ep_balanced_moe_planner_backend": planner_backend,
        "standard_ep_balanced_moe_backend_module": backend_module_name,
        "standard_ep_balanced_moe_hot_experts_requested": int(hot_experts),
        "standard_ep_balanced_moe_hot_experts_selected": int(
            len(backend_plan.hot_experts)
        ),
        "standard_ep_balanced_moe_selected_global_ids": [
            int(v) for v in backend_plan.selected_global_experts
        ],
        "standard_ep_balanced_moe_selected_owner_ranks": [
            int(v) for v in backend_plan.selected_owner_ranks
        ],
        "standard_ep_balanced_moe_selected_source_rank_counts": [
            [int(v) for v in item.source_rank_counts]
            for item in backend_plan.hot_experts
        ],
        "standard_ep_balanced_moe_selected_remote_rows": [
            int(item.remote_rows) for item in backend_plan.hot_experts
        ],
        "standard_ep_balanced_moe_selected_owner_shard_offsets": [
            int(v) for v in backend_plan.owner_shard_offsets
        ],
        "standard_ep_balanced_moe_selected_owner_compact_offsets": [
            int(v) for v in backend_plan.owner_compact_offsets
        ],
        "standard_ep_balanced_moe_owner_counts": [
            int(v) for v in backend_plan.owner_counts
        ],
        "standard_ep_balanced_moe_max_owned_per_rank": int(
            backend_plan.max_owned_per_rank
        ),
        "standard_ep_balanced_moe_owner_shard_active_offsets": [
            int(v) for v in backend_plan.owner_shard_active_offsets
        ],
        "standard_ep_balanced_moe_owner_compact_owner_ranks": [
            int(v) for v in backend_plan.owner_compact_owner_ranks
        ],
        "standard_ep_balanced_moe_owner_compact_local_experts": [
            int(v) for v in backend_plan.owner_compact_local_experts
        ],
        "standard_ep_balanced_moe_owner_compact_need_masks": [
            [bool(v) for v in row] for row in backend_plan.owner_compact_need_masks
        ],
        "standard_ep_balanced_moe_remote_hot_rows_total": int(
            backend_plan.remote_rows_total
        ),
        "standard_ep_balanced_moe_selected_rows_total": int(
            backend_plan.selected_rows_total
        ),
        "standard_ep_balanced_moe_owner_load_before": before,
        "standard_ep_balanced_moe_exec_load_after": after,
        "standard_ep_balanced_moe_modeled_max_load_reduction_pct": float(
            backend_plan.modeled_max_load_reduction_pct
        ),
    }
    if include_native_plan:
        payload["_balanced_moe_backend_module"] = backend_module_name
        payload["_balanced_moe_plan"] = backend_plan
        if backend_module_name == "mori.ops.balanced_moe":
            payload["_mori_balanced_moe_plan"] = backend_plan
    return payload


def _build_standard_ep_balanced_moe_plan(
    *,
    num_tokens_per_expert_group: torch.Tensor,
    selected_experts_indices: torch.Tensor | None = None,
    ep_size: int,
    num_local_experts: int,
    hot_experts: int,
    ep_mesh: DeviceMesh,
    include_native_plan: bool = False,
) -> dict[str, object]:
    """Model MindSpeed balanced-MoE hot expert movement on standard EP counts.

    ``num_tokens_per_expert_group`` is local to the current owner rank after the
    count all-to-all and has shape ``[source_rank, local_expert]``. The plan
    gathers that tiny matrix from all owner ranks, reconstructs the global
    ``[owner, source, local_expert]`` count tensor, then applies the same greedy
    "max expert on max EP" hot-expert selection used by MindSpeed.
    """

    ep_size = int(ep_size)
    num_local_experts = int(num_local_experts)

    if selected_experts_indices is not None:
        for backend_name in _standard_ep_balanced_moe_planner_backend_names():
            backend_module = _import_standard_ep_balanced_moe_planner_backend(
                backend_name
            )
            build_balanced_moe_plan_from_topk_ids = getattr(
                backend_module,
                "build_balanced_moe_plan_from_topk_ids",
                None,
            )
            if build_balanced_moe_plan_from_topk_ids is None:
                continue
            try:
                backend_plan = build_balanced_moe_plan_from_topk_ids(
                    selected_experts_indices,
                    hot_expert_num=int(hot_experts),
                    num_local_experts=int(num_local_experts),
                    group=ep_mesh.get_group(),
                )
                return _standard_ep_balanced_moe_payload_from_backend_plan(
                    backend_plan=backend_plan,
                    backend_module_name=str(backend_module.__name__),
                    planner_backend=f"{backend_module.__name__}.topk",
                    hot_experts=hot_experts,
                    include_native_plan=include_native_plan,
                )
            except (RuntimeError, TypeError, ValueError):
                continue

    local = (
        num_tokens_per_expert_group.detach()
        .to(torch.int64)
        .view(ep_size, num_local_experts)
        .contiguous()
    )
    gathered = [torch.empty_like(local) for _ in range(ep_size)]
    dist.all_gather(gathered, local, group=ep_mesh.get_group())
    # [owner_rank, source_rank, local_expert]
    counts_owner_source_local = torch.stack(gathered, dim=0).cpu()

    for backend_name in _standard_ep_balanced_moe_planner_backend_names():
        backend_module = _import_standard_ep_balanced_moe_planner_backend(
            backend_name
        )
        build_balanced_moe_plan_from_global_counts = getattr(
            backend_module,
            "build_balanced_moe_plan_from_global_counts",
            None,
        )
        if build_balanced_moe_plan_from_global_counts is None:
            continue
        try:
            backend_plan = build_balanced_moe_plan_from_global_counts(
                counts_owner_source_local,
                hot_expert_num=int(hot_experts),
            )
            return _standard_ep_balanced_moe_payload_from_backend_plan(
                backend_plan=backend_plan,
                backend_module_name=str(backend_module.__name__),
                planner_backend=f"{backend_module.__name__}.counts",
                hot_experts=hot_experts,
                include_native_plan=include_native_plan,
            )
        except (RuntimeError, TypeError, ValueError):
            continue

    expert_counts = counts_owner_source_local.sum(dim=1).clone()
    owner_load_before = [int(v) for v in expert_counts.sum(dim=1).tolist()]
    modeled_owner_load = list(owner_load_before)
    selected: list[dict[str, object]] = []
    max_select = min(int(hot_experts), ep_size * num_local_experts)

    for _ in range(max_select):
        if not modeled_owner_load:
            break
        owner = int(max(range(ep_size), key=lambda idx: modeled_owner_load[idx]))
        local_counts = expert_counts[owner]
        if local_counts.numel() == 0:
            break
        local_expert = int(torch.argmax(local_counts).item())
        total_rows = int(local_counts[local_expert].item())
        if total_rows <= 0:
            break
        by_source = [
            int(v)
            for v in counts_owner_source_local[owner, :, local_expert].tolist()
        ]
        global_expert = owner * num_local_experts + local_expert

        modeled_owner_load[owner] -= total_rows
        for src_rank, count in enumerate(by_source):
            modeled_owner_load[src_rank] += int(count)
        expert_counts[owner, local_expert] = 0

        selected.append(
            {
                "global_expert": int(global_expert),
                "owner_rank": int(owner),
                "local_expert": int(local_expert),
                "source_rank_counts": by_source,
                "rows_total": int(total_rows),
                "remote_rows": int(total_rows - by_source[owner]),
            }
        )

    remote_rows_total = int(sum(int(item["remote_rows"]) for item in selected))
    selected_rows_total = int(sum(int(item["rows_total"]) for item in selected))
    before = _load_summary_from_ints(owner_load_before)
    after = _load_summary_from_ints(modeled_owner_load)
    selected_global_ids = [int(item["global_expert"]) for item in selected]
    selected_owner_ranks = [int(item["owner_rank"]) for item in selected]
    owner_compact_fields = _standard_ep_owner_compact_fields_from_selected(
        selected_global_ids=selected_global_ids,
        selected_owner_ranks=selected_owner_ranks,
        ep_size=int(ep_size),
        num_local_experts=int(num_local_experts),
    )
    owner_compact_fields["standard_ep_balanced_moe_owner_compact_need_masks"] = [
        list(row)
        for row in _standard_ep_owner_compact_need_masks(
            selected_source_rank_counts=[
                [int(v) for v in item["source_rank_counts"]] for item in selected
            ],
            selected_owner_ranks=selected_owner_ranks,
            selected_owner_compact_offsets=owner_compact_fields[
                "standard_ep_balanced_moe_selected_owner_compact_offsets"
            ],
            ep_size=int(ep_size),
            active_count=len(
                owner_compact_fields[
                    "standard_ep_balanced_moe_owner_shard_active_offsets"
                ]
            ),
        )
    ]
    payload = {
        "standard_ep_balanced_moe_mode": "plan",
        "standard_ep_balanced_moe_planner_backend": "torchtitan_fallback",
        "standard_ep_balanced_moe_hot_experts_requested": int(hot_experts),
        "standard_ep_balanced_moe_hot_experts_selected": int(len(selected)),
        "standard_ep_balanced_moe_selected_global_ids": selected_global_ids,
        "standard_ep_balanced_moe_selected_owner_ranks": selected_owner_ranks,
        "standard_ep_balanced_moe_selected_source_rank_counts": [
            item["source_rank_counts"] for item in selected
        ],
        "standard_ep_balanced_moe_selected_remote_rows": [
            int(item["remote_rows"]) for item in selected
        ],
        "standard_ep_balanced_moe_remote_hot_rows_total": remote_rows_total,
        "standard_ep_balanced_moe_selected_rows_total": selected_rows_total,
        "standard_ep_balanced_moe_owner_load_before": before,
        "standard_ep_balanced_moe_exec_load_after": after,
        "standard_ep_balanced_moe_modeled_max_load_reduction_pct": (
            float(
                (float(before["max"]) - float(after["max"]))
                / float(before["max"])
                * 100.0
            )
            if float(before["max"]) > 0.0
            else 0.0
        ),
    }
    payload.update(owner_compact_fields)
    return payload


def _record_standard_ep_hot_expert_stats(
    *,
    num_tokens_per_expert_group: torch.Tensor,
    ep_size: int,
    num_local_experts: int,
    input_splits: list[int],
    output_splits: list[int],
    ep_mesh: DeviceMesh | None = None,
) -> int | None:
    path = _standard_ep_hot_expert_timing_path()
    if path is None:
        return None

    counts_by_source = (
        num_tokens_per_expert_group.detach()
        .to(torch.int64)
        .view(int(ep_size), int(num_local_experts))
        .cpu()
    )
    owner_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "-1")))
    owner_ep_rank = owner_rank % int(ep_size) if owner_rank >= 0 else -1
    counts = counts_by_source.sum(dim=0)
    assignments = int(counts.sum().item())
    max_count = int(counts.max().item()) if counts.numel() else 0
    mean_count = float(assignments / max(1, int(num_local_experts)))
    topk = min(
        max(0, _env_int("TORCHTITAN_MORI_AITER_HOT_EXPERT_TOPK", 8)),
        int(num_local_experts),
    )
    hot_local_ids: list[int] = []
    hot_counts: list[int] = []
    source_counts: list[list[int]] = []
    remote_counts: list[int] = []
    if topk > 0 and counts.numel():
        hot_values, hot_indices = torch.topk(counts, k=topk)
        hot_local_ids = [int(v) for v in hot_indices.tolist()]
        hot_counts = [int(v) for v in hot_values.tolist()]
        for local_idx in hot_local_ids:
            by_source = [int(v) for v in counts_by_source[:, local_idx].tolist()]
            source_counts.append(by_source)
            local_owner_rows = (
                by_source[owner_ep_rank]
                if 0 <= owner_ep_rank < len(by_source)
                else 0
            )
            remote_counts.append(int(sum(by_source) - local_owner_rows))

    expert_start = owner_ep_rank * int(num_local_experts) if owner_ep_rank >= 0 else 0
    payload = {
        "kind": "standard_ep_hot_expert_stats",
        "rank": int(os.environ.get("RANK", "-1")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
        "pid": os.getpid(),
        "owner_ep_rank": int(owner_ep_rank),
        "ep_size": int(ep_size),
        "num_local_experts": int(num_local_experts),
        "expert_start": int(expert_start),
        "expert_end": int(expert_start + int(num_local_experts)),
        "local_vjp_assignments": assignments,
        "local_vjp_max_count": max_count,
        "local_vjp_mean_count": mean_count,
        "local_vjp_max_over_mean_count": (
            float(max_count / mean_count) if mean_count > 0.0 else None
        ),
        "local_vjp_hot_expert_topk": int(topk),
        "local_vjp_hot_expert_local_ids": hot_local_ids,
        "local_vjp_hot_expert_global_ids": [int(expert_start + idx) for idx in hot_local_ids],
        "local_vjp_hot_expert_counts": hot_counts,
        "local_vjp_hot_expert_topk_rows": int(sum(hot_counts)),
        "local_vjp_hot_expert_topk_share": (
            float(sum(hot_counts) / assignments) if assignments > 0 else None
        ),
        "local_vjp_hot_expert_source_rank_counts": source_counts,
        "local_vjp_hot_expert_remote_counts": remote_counts,
        "local_vjp_balanced_moe_top1_remote_rows": (
            int(remote_counts[0]) if remote_counts else 0
        ),
        "local_vjp_balanced_moe_topk_remote_rows": int(sum(remote_counts)),
        "local_vjp_balanced_moe_topk_remote_share": (
            float(sum(remote_counts) / assignments) if assignments > 0 else None
        ),
        "input_splits": [int(v) for v in input_splits],
        "output_splits": [int(v) for v in output_splits],
    }
    balanced_mode = _standard_ep_balanced_moe_mode()
    balanced_hot_experts = _standard_ep_balanced_moe_hot_experts(balanced_mode)
    if (
        balanced_mode in {"plan", "execute"}
        and balanced_hot_experts > 0
        and ep_mesh is not None
        and dist.is_available()
        and dist.is_initialized()
    ):
        if torch.compiler.is_compiling():
            payload["standard_ep_balanced_moe_mode"] = "plan_skipped_compiling"
        else:
            payload.update(
                _build_standard_ep_balanced_moe_plan(
                    num_tokens_per_expert_group=num_tokens_per_expert_group,
                    ep_size=ep_size,
                    num_local_experts=num_local_experts,
                    hot_experts=balanced_hot_experts,
                    ep_mesh=ep_mesh,
                )
            )
    if _env_flag("TORCHTITAN_MORI_AITER_RECORD_LOCAL_EXPERT_COUNTS", default=True):
        payload["local_vjp_expert_counts"] = [int(v) for v in counts.tolist()]

    global _STANDARD_EP_HOT_EXPERT_CALL_ID
    _STANDARD_EP_HOT_EXPERT_CALL_ID += 1
    payload["call_id"] = int(_STANDARD_EP_HOT_EXPERT_CALL_ID)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return int(_STANDARD_EP_HOT_EXPERT_CALL_ID)


def _record_standard_ep_hot_expert_execute_stats(
    *,
    call_id: int | None,
    plan: dict[str, object] | None,
    selected_hot_experts: tuple[int, ...],
    min_reduction_pct: float,
    modeled_reduction_pct: float,
    skipped_by_threshold: bool,
    remote_positions: torch.Tensor | None = None,
    remote_hot_offsets: torch.Tensor | None = None,
    keep_flat_mask: torch.Tensor | None = None,
    top_k: int | None = None,
) -> None:
    path = _standard_ep_hot_expert_timing_path()
    if path is None:
        return

    planned_global_ids: list[int] = []
    planned_remote_rows: list[int] = []
    planned_rows_total = 0
    planned_remote_rows_total = 0
    if plan is not None:
        planned_global_ids = [
            int(v)
            for v in plan.get("standard_ep_balanced_moe_selected_global_ids", [])
        ]
        planned_remote_rows = [
            int(v)
            for v in plan.get("standard_ep_balanced_moe_selected_remote_rows", [])
        ]
        planned_rows_total = int(
            plan.get("standard_ep_balanced_moe_selected_rows_total", 0)
        )
        planned_remote_rows_total = int(
            plan.get("standard_ep_balanced_moe_remote_hot_rows_total", 0)
        )

    remote_rows = 0
    remote_unique_tokens = 0
    remote_rows_by_hot_offset: list[int] = []
    if remote_positions is not None:
        remote_rows = int(remote_positions.numel())
        if top_k is not None and top_k > 0 and remote_rows > 0:
            remote_unique_tokens = int(
                torch.unique(remote_positions.detach() // int(top_k)).numel()
            )
    if remote_hot_offsets is not None and remote_hot_offsets.numel() > 0:
        rows_by_offset = torch.bincount(
            remote_hot_offsets.detach().to(torch.int64).cpu(),
            minlength=len(selected_hot_experts),
        )
        remote_rows_by_hot_offset = [int(v) for v in rows_by_offset.tolist()]
    elif selected_hot_experts:
        remote_rows_by_hot_offset = [0 for _ in selected_hot_experts]

    normal_routed_rows_after_hot_mask = None
    dropped_remote_hot_rows = None
    if keep_flat_mask is not None:
        kept_rows = int(keep_flat_mask.sum().item())
        normal_routed_rows_after_hot_mask = kept_rows
        dropped_remote_hot_rows = int(keep_flat_mask.numel() - kept_rows)

    reason = "executed"
    if not selected_hot_experts:
        reason = "skipped_threshold" if skipped_by_threshold else "skipped_no_hot_experts"

    payload = {
        "kind": "standard_ep_hot_expert_execute_stats",
        "rank": int(os.environ.get("RANK", "-1")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
        "pid": os.getpid(),
        "call_id": int(call_id) if call_id is not None else None,
        "standard_ep_balanced_moe_execute_reason": reason,
        "standard_ep_balanced_moe_execute_enabled": bool(selected_hot_experts),
        "standard_ep_balanced_moe_execute_skipped_by_threshold": bool(
            skipped_by_threshold
        ),
        "standard_ep_balanced_moe_min_reduction_pct": float(min_reduction_pct),
        "standard_ep_balanced_moe_modeled_max_load_reduction_pct": float(
            modeled_reduction_pct
        ),
        "standard_ep_balanced_moe_planned_global_ids": planned_global_ids,
        "standard_ep_balanced_moe_planned_remote_rows": planned_remote_rows,
        "standard_ep_balanced_moe_planned_rows_total": int(planned_rows_total),
        "standard_ep_balanced_moe_planned_remote_rows_total": int(
            planned_remote_rows_total
        ),
        "standard_ep_balanced_moe_executed_global_ids": [
            int(v) for v in selected_hot_experts
        ],
        "standard_ep_balanced_moe_executed_hot_experts_selected": int(
            len(selected_hot_experts)
        ),
        "standard_ep_balanced_moe_executed_remote_rows": int(remote_rows),
        "standard_ep_balanced_moe_executed_remote_unique_tokens": int(
            remote_unique_tokens
        ),
        "standard_ep_balanced_moe_executed_remote_rows_by_hot_offset": (
            remote_rows_by_hot_offset
        ),
        "standard_ep_balanced_moe_normal_routed_rows_after_hot_mask": (
            normal_routed_rows_after_hot_mask
        ),
        "standard_ep_balanced_moe_dropped_remote_hot_rows": dropped_remote_hot_rows,
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


@dataclass(frozen=True, kw_only=True)
class LocalDispatchMetadata:
    """Metadata returned by LocalTokenDispatcher.dispatch() for use in combine()."""

    token_indices_experts_sorted: torch.Tensor  # (N*top_k,)
    top_scores_experts_sorted: torch.Tensor  # (N*top_k,) scores in expert-sorted order


@dataclass(frozen=True, kw_only=True)
class StandardEPCompactDispatchLayout:
    """Normal/cold standard-EP compact-row ABI for a native MORI dispatcher.

    This layout is intentionally source-position-first.  The current standard
    EP path already builds compact rows by sorted flattened route position
    ``token * top_k + slot``; a native MORI compact dispatcher should consume
    this row stream and carry the source positions internally while moving the
    row data.  That avoids the diagnostic-only second int64 all-to-all used by
    ``TORCHTITAN_STANDARD_EP_EXPORT_COMPACT_ROUTE_POSITIONS``.
    """

    local_flat_positions_experts_sorted: torch.Tensor
    local_num_tokens_per_expert: torch.Tensor
    recv_counts_rank_major: torch.Tensor
    input_splits_tensor: torch.Tensor
    output_splits_tensor: torch.Tensor
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    ep_size: int
    num_local_experts: int
    top_k: int
    local_token_count: int
    hidden_dim: int
    rank_major_input_shape: tuple[int, ...]


@dataclass(frozen=True, kw_only=True)
class AllToAllDispatchMetadata(LocalDispatchMetadata):
    """Metadata returned by AllToAllTokenDispatcher.dispatch() for use in combine()."""

    input_shape: tuple  # for _unpermute
    permuted_indices: torch.Tensor  # for _unpermute
    input_splits: list[int]
    output_splits: list[int]
    standard_ep_hot_split: "StandardEPHotSplitMetadata | None" = None
    normal_flat_positions_experts_sorted: torch.Tensor | None = None
    normal_top_scores_flat: torch.Tensor | None = None
    normal_recv_global_flat_positions_expert_major: torch.Tensor | None = None
    normal_flat_position_rank_stride: int = 0
    normal_compact_dispatch_layout: StandardEPCompactDispatchLayout | None = None
    normal_topk_ids_for_backend: torch.Tensor | None = None
    normal_topk_weights_for_backend: torch.Tensor | None = None
    normal_topk_backend_module: str | None = None
    normal_topk_backend_state: object | None = None
    normal_compact_collective_backend: str = "torch"
    normal_compact_collective_native_active: bool = False
    normal_compact_native_dispatch_summary: dict[str, object] | None = None


@dataclass(frozen=True, kw_only=True)
class StandardEPHotSplitMetadata:
    """Source-local rows moved out of the standard EP owner path.

    ``selected_global_experts`` is present on every EP rank so GroupedExperts
    can run the same differentiable hot-weight broadcast/reduce collective even
    when a rank has no remote-hot rows in the current microbatch.
    """

    selected_global_experts: tuple[int, ...]
    remote_token_indices: torch.Tensor
    remote_flat_positions: torch.Tensor
    remote_top_scores: torch.Tensor
    remote_hot_offsets: torch.Tensor
    ep_rank: int
    ep_size: int
    num_local_experts: int
    remote_owner_shard_offsets: torch.Tensor | None = None
    remote_owner_compact_offsets: torch.Tensor | None = None
    owner_counts: tuple[int, ...] = ()
    owner_shard_active_offsets: tuple[int, ...] = ()
    owner_compact_owner_ranks: tuple[int, ...] = ()
    owner_compact_local_experts: tuple[int, ...] = ()
    owner_compact_need_masks: tuple[tuple[bool, ...], ...] = ()
    owner_compact_exchange_plan: dict[str, object] | None = None
    owner_compact_exchange_tensor_plan: dict[str, object] | None = None
    max_owned_per_rank: int = 0
    remote_offsets_presorted_by: str = ""
    remote_rows_materialized_by: str = ""
    remote_materialized_x: torch.Tensor | None = None
    remote_materialized_token_indices: torch.Tensor | None = None
    remote_materialized_top_scores: torch.Tensor | None = None
    remote_materialized_group_ends: torch.Tensor | None = None
    balanced_moe_backend_module: str | None = None


class LocalTokenDispatcher(Configurable):
    """Token dispatcher for EP=1. Handles local token reordering only.

    Also serves as the base class for EP dispatchers (AllToAllTokenDispatcher,
    DeepEPTokenDispatcher, HybridEPTokenDispatcher) which override
    dispatch() and combine().

    Not an nn.Module — dispatchers have no learnable parameters or buffers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        num_experts: int
        top_k: int
        score_before_experts: bool = True

    def __init__(self, config: Config):
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.score_before_experts = config.score_before_experts

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        """No-op for the EP=1 dispatcher. Subclasses override."""
        del ep_mesh, tp_mesh

    def _local_reorder(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder tokens by expert assignment for local expert computation.

        Groups tokens by expert index via histc + argsort, optionally
        applies routing scores (when ``score_before_experts`` is True).

        Args:
            x: (num_tokens, dim) input tokens
            top_scores: (num_tokens, top_k) routing scores
            selected_experts_indices: (num_tokens, top_k) expert indices

        Returns:
            routed_input: (num_tokens * top_k, dim) tokens in expert-sorted
                order, score-weighted if ``score_before_experts``
            num_tokens_per_expert: (num_experts,) token counts per expert
            token_indices_experts_sorted: (num_tokens * top_k,) token-to-original mapping
            top_scores_experts_sorted: (num_tokens * top_k,) scores in expert-sorted order
            flat_positions_experts_sorted: (num_tokens * top_k,) flattened token-route
                positions in the same expert-sorted order
        """
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = _count_route_ids(
            selected_experts_indices, self.num_experts
        )

        # Reorder the token indices to match the order of the experts
        # flat_positions_experts_sorted shape (bs*slen*top_k,)
        flat_positions_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        top_scores_experts_sorted = top_scores.view(-1)[flat_positions_experts_sorted]
        token_indices_experts_sorted = flat_positions_experts_sorted // self.top_k

        # shape (bs*slen*top_k, dim)
        routed_input = x[token_indices_experts_sorted]

        # Apply scores before expert computation if configured
        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        return (
            routed_input,
            num_tokens_per_expert,
            token_indices_experts_sorted,
            top_scores_experts_sorted,
            flat_positions_experts_sorted,
        )

    def _local_reorder_with_flat_mask(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        keep_flat_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like ``_local_reorder``, but drops flattened token-route rows first."""
        flat_experts = selected_experts_indices.reshape(-1)
        flat_scores = top_scores.reshape(-1)
        flat_positions = torch.arange(
            flat_experts.numel(), device=x.device, dtype=torch.long
        )

        kept_experts = flat_experts[keep_flat_mask]
        kept_scores = flat_scores[keep_flat_mask]
        kept_flat_positions = flat_positions[keep_flat_mask]

        num_tokens_per_expert = _count_route_ids(kept_experts, self.num_experts)
        sorted_positions = torch.argsort(kept_experts, stable=True)
        flat_positions_experts_sorted = kept_flat_positions[sorted_positions]
        token_indices_experts_sorted = flat_positions_experts_sorted // self.top_k
        top_scores_experts_sorted = kept_scores[sorted_positions]
        routed_input = x[token_indices_experts_sorted]

        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        return (
            routed_input,
            num_tokens_per_expert,
            token_indices_experts_sorted,
            top_scores_experts_sorted,
            flat_positions_experts_sorted,
        )

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, LocalDispatchMetadata]:
        """Reorder tokens by expert assignment for local expert computation.

        Args:
            x: (num_tokens, dim) all input tokens
            top_scores: (num_tokens, top_k) routing scores
            selected_experts_indices: (num_tokens, top_k) expert indices per token

        Returns:
            routed_input: (num_tokens * top_k, dim) tokens sorted by expert index
            num_tokens_per_expert: (num_experts,) token counts per expert
            metadata: LocalDispatchMetadata for combine()
        """
        (
            routed_input,
            num_tokens_per_expert,
            token_indices_experts_sorted,
            top_scores_experts_sorted,
            _flat_positions_experts_sorted,
        ) = self._local_reorder(x, top_scores, selected_experts_indices)

        metadata = LocalDispatchMetadata(
            token_indices_experts_sorted=token_indices_experts_sorted,
            top_scores_experts_sorted=top_scores_experts_sorted,
        )
        return routed_input, num_tokens_per_expert, metadata

    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: LocalDispatchMetadata,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Score and scatter_add routed expert outputs.

        Args:
            routed_output: (num_tokens * top_k, dim) expert outputs
            metadata: LocalDispatchMetadata from dispatch()
            x: (num_tokens, dim) original input tokens

        Returns:
            (num_tokens, dim) combined output.
        """
        out = torch.zeros_like(x)

        if not self.score_before_experts:
            routed_output = (
                routed_output.to(torch.float32)
                * metadata.top_scores_experts_sorted.reshape(-1, 1)
            ).to(routed_output.dtype)

        dim = x.shape[-1]
        out = deterministic_scatter_add(
            out,
            metadata.token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim),
            routed_output,
        )
        return out


class AllToAllTokenDispatcher(LocalTokenDispatcher):
    """Token dispatcher for EP>1. Handles token reorder + all-to-all dispatch/combine.

    Handles the full token routing lifecycle:
    dispatch (reorder + EP all-to-all) and combine (reverse).

    ``ep_mesh`` and the ``sp_size`` / ``sp_rank`` SP coordinates are wired
    by the owning ``GroupedExperts.parallelize`` override via
    ``wire_meshes``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalTokenDispatcher.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # DeviceMesh (not ProcessGroup) so that CooR precompile can use
        # torch.ops._dtensor.mesh_get_process_group to keep the FX graph
        # rank-agnostic. None when EP=1 so dispatch falls back to the
        # LocalTokenDispatcher path.
        self.ep_mesh: DeviceMesh | None = None
        # Sequence-parallel split coordinates derived from tp_mesh.
        # ``sp_rank`` uses ``DeviceMesh._sym_get_coordinate`` so it is a
        # ``SymInt`` under CooR precompile, keeping the FX graph
        # rank-agnostic. Defaults are the TP=1 values.
        self.sp_size: int = 1
        self.sp_rank: int | torch.SymInt = 0

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        """Install the EP mesh and SP coordinates used by dispatch / combine.

        Both arguments may be ``None`` when the corresponding parallelism
        dimension is disabled; ``dispatch`` / ``combine`` handle the
        disabled cases internally.
        """
        self.ep_mesh = ep_mesh
        if tp_mesh is not None:
            self.sp_size = tp_mesh.size()
            self.sp_rank = tp_mesh._sym_get_coordinate(0)

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, AllToAllDispatchMetadata | LocalDispatchMetadata
    ]:
        """Reorder tokens, then all-to-all dispatch to expert-parallel ranks.

        When ep_mesh is None (EP=1), falls back to local dispatch — no
        all-to-all communication, just local token reordering with padding.

        With SP, x/top_scores/selected_experts_indices are already the local
        SP shard (from DTensor Shard to_local via LocalMapConfig).

        Args:
            x: (num_local_tokens, dim) local token shard
            top_scores: (num_local_tokens, top_k) routing scores
            selected_experts_indices: (num_local_tokens, top_k) expert indices

        Returns:
            routed_input: (R, dim) tokens in expert-major order for local experts
            num_tokens_per_expert_local: (num_local_experts,) token counts
            metadata: dispatch metadata for combine()
        """
        # EP=1: fall back to local dispatch (no all-to-all needed)
        if self.ep_mesh is None:
            return super().dispatch(x, top_scores, selected_experts_indices)

        ep_size = self.ep_mesh.size()
        timing_records = _standard_ep_stage_timing_records()

        balanced_mode = _standard_ep_balanced_moe_mode()
        maybe_hot_execute = (
            balanced_mode == "execute"
            and not torch.compiler.is_compiling()
            and _standard_ep_balanced_moe_hot_experts(balanced_mode) > 0
        )
        use_backend_dispatch_permute = (
            _standard_ep_backend_normal_topk_dispatch_permute()
        )
        defer_hot_reorder = bool(
            (maybe_hot_execute and _standard_ep_defer_hot_reorder())
            or use_backend_dispatch_permute
        )
        normal_flat_positions_experts_sorted: torch.Tensor | None = None
        if defer_hot_reorder:
            with dsv4_timed_stage(timing_records, "standard_ep.dispatch.local_count"):
                num_tokens_per_expert = _count_route_ids(
                    selected_experts_indices, self.num_experts
                )
            routed_input = None
            token_indices_experts_sorted = None
            top_scores_experts_sorted = None
        else:
            with dsv4_timed_stage(timing_records, "standard_ep.dispatch.local_reorder"):
                (
                    routed_input,
                    num_tokens_per_expert,
                    token_indices_experts_sorted,
                    top_scores_experts_sorted,
                    normal_flat_positions_experts_sorted,
                ) = self._local_reorder(x, top_scores, selected_experts_indices)
        standard_ep_hot_split: StandardEPHotSplitMetadata | None = None
        standard_ep_hot_stats_call_id: int | None = None
        standard_ep_execute_plan: dict[str, object] | None = None
        standard_ep_execute_min_reduction_pct = 0.0
        standard_ep_execute_modeled_reduction_pct = 0.0
        standard_ep_execute_skipped_by_threshold = False
        selected_hot_experts: tuple[int, ...] = ()
        remote_hot_rows = 0
        materialize_backend = _standard_ep_hot_split_materialize_backend()
        normal_routed_rows = (
            int(x.shape[0] * self.top_k)
            if routed_input is None
            else int(routed_input.shape[0])
        )
        export_compact_route_positions = _standard_ep_export_compact_route_positions()
        compact_collective_backend = _standard_ep_compact_collective_backend()
        use_backend_runtime_layout = _standard_ep_use_backend_balanced_runtime_layout(
            materialize_backend=materialize_backend,
            compact_collective_backend=compact_collective_backend,
        )
        normal_recv_global_flat_positions_expert_major: torch.Tensor | None = None
        normal_flat_position_rank_stride = 0
        normal_topk_ids_for_backend: torch.Tensor | None = None
        normal_topk_weights_for_backend: torch.Tensor | None = None
        normal_topk_backend_module: str | None = None
        normal_topk_backend_state: object | None = None

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            with dsv4_timed_stage(
                timing_records, "standard_ep.dispatch.count_all_to_all"
            ):
                num_tokens_per_expert_group = all_to_all_single(
                    num_tokens_per_expert,
                    None,
                    None,
                    group=self.ep_mesh,
                )
                # Need to wait explicitly because it is used by a triton kernel later
                # which doesn't realize that AsyncCollectiveTensor needs unwrapping
                num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                    num_tokens_per_expert_group
                )
            with dsv4_timed_stage(
                timing_records, "standard_ep.dispatch.split_d2h"
            ):
                # non_blocking=True is safe in eager, but under torch.compile the
                # async D2H transfer can race with the subsequent .tolist()/.item()
                # calls, producing stale values and failing unbacked-symint guards.
                non_blocking = not torch.compiler.is_compiling()
                input_splits = (
                    num_tokens_per_expert.view(ep_size, -1)
                    .sum(dim=1)
                    .to(torch.device("cpu"), non_blocking=non_blocking)
                )
                # NOTE: this would incur a device-to-host sync
                output_splits = (
                    num_tokens_per_expert_group.view(ep_size, -1)
                    .sum(dim=1)
                    .to(torch.device("cpu"), non_blocking=False)
                )
                input_splits_list = [int(v) for v in input_splits.tolist()]
                output_splits_list = [int(v) for v in output_splits.tolist()]
            with dsv4_timed_stage(
                timing_records, "standard_ep.dispatch.record_hot_stats"
            ):
                standard_ep_hot_stats_call_id = _record_standard_ep_hot_expert_stats(
                    num_tokens_per_expert_group=num_tokens_per_expert_group,
                    ep_size=ep_size,
                    num_local_experts=num_tokens_per_expert_group.shape[0] // ep_size,
                    input_splits=input_splits_list,
                    output_splits=output_splits_list,
                    ep_mesh=self.ep_mesh,
                )
            if maybe_hot_execute:
                with dsv4_timed_stage(
                    timing_records, "standard_ep.dispatch.build_hot_plan"
                ):
                    standard_ep_execute_plan = _build_standard_ep_balanced_moe_plan(
                        num_tokens_per_expert_group=num_tokens_per_expert_group,
                        selected_experts_indices=selected_experts_indices,
                        ep_size=ep_size,
                        num_local_experts=num_tokens_per_expert_group.shape[0] // ep_size,
                        hot_experts=_standard_ep_balanced_moe_hot_experts(
                            balanced_mode
                        ),
                        ep_mesh=self.ep_mesh,
                        include_native_plan=use_backend_runtime_layout,
                    )
                selected_hot_experts = tuple(
                    int(v)
                    for v in standard_ep_execute_plan[
                        "standard_ep_balanced_moe_selected_global_ids"
                    ]
                )
                standard_ep_execute_min_reduction_pct = (
                    _standard_ep_balanced_moe_min_reduction_pct()
                )
                standard_ep_execute_modeled_reduction_pct = float(
                    standard_ep_execute_plan.get(
                        "standard_ep_balanced_moe_modeled_max_load_reduction_pct",
                        0.0,
                    )
                )
                if (
                    standard_ep_execute_modeled_reduction_pct
                    < standard_ep_execute_min_reduction_pct
                ):
                    selected_hot_experts = ()
                    standard_ep_execute_skipped_by_threshold = True

        if use_backend_dispatch_permute and not selected_hot_experts:
            if self.score_before_experts:
                raise RuntimeError(
                    "Backend normal top-k dispatch+permute is only wired for "
                    "score_before_experts=False because Primus-Turbo combine "
                    "owns the router-probability weighting."
                )
            if compact_collective_backend != "torch":
                raise RuntimeError(
                    "Backend normal top-k dispatch+permute replaces the "
                    "standard EP collective; do not combine it with mori_native "
                    "compact collectives in the same path."
                )
            if not _standard_ep_backend_normal_topk_dispatch_tensors():
                raise RuntimeError(
                    "Backend normal top-k dispatch+permute requires normal "
                    "top-k dispatch tensors. Enable "
                    "TORCHTITAN_STANDARD_EP_BACKEND_NORMAL_TOPK_DISPATCH_TENSORS=1 "
                    "or the CANARY alias."
                )
            with dsv4_timed_stage(
                timing_records,
                "standard_ep.dispatch.backend_normal_topk",
            ):
                normal_route_mask = torch.ones_like(
                    selected_experts_indices, dtype=torch.bool
                )
                selected_backend_module = (
                    _select_standard_ep_balanced_moe_module_with_capability(
                        None,
                        "normal_topk_dispatch_tensors",
                    )
                )
                build_normal_topk_dispatch_tensors = (
                    getattr(
                        selected_backend_module,
                        "build_normal_topk_dispatch_tensors",
                        None,
                    )
                    if selected_backend_module is not None
                    else None
                )
                if (
                    build_normal_topk_dispatch_tensors is None
                    or selected_backend_module is None
                ):
                    raise RuntimeError(
                        "Backend normal top-k dispatch tensor handoff was "
                        "requested, but no selected balanced-MoE backend "
                        "advertises normal_topk_dispatch_tensors and exposes "
                        "build_normal_topk_dispatch_tensors()."
                    )
                (
                    normal_topk_ids_for_backend,
                    normal_topk_weights_for_backend,
                ) = build_normal_topk_dispatch_tensors(
                    selected_experts_indices,
                    top_scores,
                    normal_route_mask=normal_route_mask,
                )
                if _standard_ep_backend_normal_topk_dispatch_validate():
                    torch.testing.assert_close(
                        normal_topk_ids_for_backend,
                        selected_experts_indices,
                        check_dtype=True,
                        rtol=0,
                        atol=0,
                    )
                    if normal_topk_weights_for_backend is None:
                        raise RuntimeError(
                            "Backend normal top-k dispatch tensor handoff must "
                            "preserve top-k weights when top_scores are provided."
                        )
                    torch.testing.assert_close(
                        normal_topk_weights_for_backend,
                        top_scores,
                        check_dtype=True,
                        rtol=0,
                        atol=0,
                    )
                normal_topk_backend_module = str(selected_backend_module.__name__)

            with dsv4_timed_stage(
                timing_records,
                "standard_ep.dispatch.backend_normal_topk_dispatch_permute",
            ):
                backend_dispatch_module = (
                    _select_standard_ep_balanced_moe_module_with_capability(
                        normal_topk_backend_module,
                        "normal_topk_ep_dispatch_permute",
                    )
                )
                dispatch_permute_normal_topk_tokens = (
                    getattr(
                        backend_dispatch_module,
                        "dispatch_permute_normal_topk_tokens",
                        None,
                    )
                    if backend_dispatch_module is not None
                    else None
                )
                if dispatch_permute_normal_topk_tokens is None:
                    raise RuntimeError(
                        "Backend normal top-k dispatch+permute was requested, "
                        "but no selected balanced-MoE backend advertises "
                        "normal_topk_ep_dispatch_permute and exposes "
                        "dispatch_permute_normal_topk_tokens()."
                    )
                (
                    routed_input,
                    num_tokens_per_expert_group,
                    top_scores_experts_sorted,
                    normal_topk_backend_state,
                ) = dispatch_permute_normal_topk_tokens(
                    x,
                    normal_topk_ids_for_backend,
                    normal_topk_weights_for_backend,
                    num_experts=int(self.num_experts),
                    num_local_experts=int(
                        num_tokens_per_expert_group.shape[0] // ep_size
                    ),
                    group=self.ep_mesh.get_group(),
                    num_topk=int(self.top_k),
                    use_cuda_num_tokens_per_expert=True,
                )
                token_indices_experts_sorted = torch.empty(
                    0, device=x.device, dtype=torch.long
                )
                normal_flat_positions_experts_sorted = None
                normal_routed_rows = int(x.shape[0] * self.top_k)

        if selected_hot_experts:
            ep_group = self.ep_mesh.get_group()
            ep_rank = int(dist.get_rank(group=ep_group))
            num_local_experts = int(num_tokens_per_expert_group.shape[0] // ep_size)
            if standard_ep_execute_plan is None:
                raise RuntimeError("balanced-MoE execute requires a hot-expert plan")
            hot_owner_ranks = [
                int(v)
                for v in standard_ep_execute_plan.get(
                    "standard_ep_balanced_moe_selected_owner_ranks", []
                )
            ]
            if len(hot_owner_ranks) != len(selected_hot_experts):
                hot_owner_ranks = [
                    int(expert) // int(num_local_experts)
                    for expert in selected_hot_experts
                ]
            owner_field_defaults = _standard_ep_owner_compact_fields_from_selected(
                selected_global_ids=[int(v) for v in selected_hot_experts],
                selected_owner_ranks=hot_owner_ranks,
                ep_size=int(ep_size),
                num_local_experts=int(num_local_experts),
            )
            owner_counts_list = [
                int(v)
                for v in standard_ep_execute_plan.get(
                    "standard_ep_balanced_moe_owner_counts",
                    owner_field_defaults["standard_ep_balanced_moe_owner_counts"],
                )
            ]
            if len(owner_counts_list) != int(ep_size):
                owner_counts_list = [
                    int(v)
                    for v in owner_field_defaults[
                        "standard_ep_balanced_moe_owner_counts"
                    ]
                ]
            max_owned_per_rank = int(
                standard_ep_execute_plan.get(
                    "standard_ep_balanced_moe_max_owned_per_rank",
                    owner_field_defaults[
                        "standard_ep_balanced_moe_max_owned_per_rank"
                    ],
                )
            )
            hot_owner_shard_offsets_list = [
                int(v)
                for v in standard_ep_execute_plan.get(
                    "standard_ep_balanced_moe_selected_owner_shard_offsets",
                    owner_field_defaults[
                        "standard_ep_balanced_moe_selected_owner_shard_offsets"
                    ],
                )
            ]
            hot_owner_compact_offsets_list = [
                int(v)
                for v in standard_ep_execute_plan.get(
                    "standard_ep_balanced_moe_selected_owner_compact_offsets",
                    owner_field_defaults[
                        "standard_ep_balanced_moe_selected_owner_compact_offsets"
                    ],
                )
            ]
            owner_shard_active_offsets = tuple(
                int(v)
                for v in standard_ep_execute_plan.get(
                    "standard_ep_balanced_moe_owner_shard_active_offsets",
                    owner_field_defaults[
                        "standard_ep_balanced_moe_owner_shard_active_offsets"
                    ],
                )
            )
            owner_compact_owner_ranks_list = [
                int(v)
                for v in standard_ep_execute_plan.get(
                    "standard_ep_balanced_moe_owner_compact_owner_ranks",
                    owner_field_defaults[
                        "standard_ep_balanced_moe_owner_compact_owner_ranks"
                    ],
                )
            ]
            owner_compact_local_experts_list = [
                int(v)
                for v in standard_ep_execute_plan.get(
                    "standard_ep_balanced_moe_owner_compact_local_experts",
                    owner_field_defaults[
                        "standard_ep_balanced_moe_owner_compact_local_experts"
                    ],
                )
            ]
            selected_source_rank_counts_list = [
                [int(v) for v in row]
                for row in standard_ep_execute_plan.get(
                    "standard_ep_balanced_moe_selected_source_rank_counts", []
                )
            ]
            owner_compact_need_masks = tuple(
                tuple(bool(v) for v in row)
                for row in standard_ep_execute_plan.get(
                    "standard_ep_balanced_moe_owner_compact_need_masks", []
                )
            )
            balanced_moe_backend_module_name = standard_ep_execute_plan.get(
                "_balanced_moe_backend_module",
                standard_ep_execute_plan.get(
                    "standard_ep_balanced_moe_backend_module",
                    None,
                ),
            )
            if (
                balanced_moe_backend_module_name is None
                and standard_ep_execute_plan.get("_mori_balanced_moe_plan") is not None
            ):
                balanced_moe_backend_module_name = "mori.ops.balanced_moe"
            active_owner_compact_count = len(owner_shard_active_offsets)
            if (
                len(owner_compact_need_masks) != int(ep_size)
                or any(
                    len(row) != active_owner_compact_count
                    for row in owner_compact_need_masks
                )
            ):
                owner_compact_need_masks = _standard_ep_owner_compact_need_masks(
                    selected_source_rank_counts=selected_source_rank_counts_list,
                    selected_owner_ranks=hot_owner_ranks,
                    selected_owner_compact_offsets=hot_owner_compact_offsets_list,
                    ep_size=int(ep_size),
                    active_count=active_owner_compact_count,
                    backend_module_name=(
                        str(balanced_moe_backend_module_name)
                        if balanced_moe_backend_module_name is not None
                        else None
                    ),
                )
            owner_compact_exchange_plan = None
            owner_compact_exchange_tensor_plan = None
            flat_experts = selected_experts_indices.reshape(-1)
            flat_scores = top_scores.reshape(-1)
            materialize_rows_mode = _standard_ep_hot_split_materialize_rows_mode()
            presort_mode = _standard_ep_hot_split_presort_mode()
            if materialize_rows_mode != "off":
                presort_mode = materialize_rows_mode
            source_partition = None
            balanced_moe_plan_obj = standard_ep_execute_plan.get(
                "_balanced_moe_plan",
                standard_ep_execute_plan.get("_mori_balanced_moe_plan"),
            )
            if use_backend_runtime_layout and balanced_moe_plan_obj is not None:
                with dsv4_timed_stage(
                    timing_records, "standard_ep.dispatch.hot_backend_runtime_layout"
                ):
                    try:
                        balanced_moe_backend_module = (
                            _import_standard_ep_balanced_moe_module(
                                str(balanced_moe_backend_module_name)
                                if balanced_moe_backend_module_name is not None
                                else None
                            )
                        )
                        build_balanced_moe_runtime_layout = getattr(
                            balanced_moe_backend_module,
                            "build_balanced_moe_runtime_layout",
                            None,
                        )
                        if build_balanced_moe_runtime_layout is None:
                            raise AttributeError("build_balanced_moe_runtime_layout")

                        compact_local_indices = torch.tensor(
                            owner_compact_local_experts_list,
                            device=x.device,
                            dtype=torch.long,
                        )
                        runtime_layout = build_balanced_moe_runtime_layout(
                            selected_experts_indices,
                            balanced_moe_plan_obj,
                            ep_rank=int(ep_rank),
                            compact_local_indices=compact_local_indices,
                            presort_by=presort_mode,
                            device=x.device,
                            dtype=torch.int64,
                            split_dtype=torch.int64,
                        )
                        source_partition = runtime_layout.source_partition
                        runtime_layout_payload = (
                            runtime_layout.as_dict()
                            if hasattr(runtime_layout, "as_dict")
                            else None
                        )
                        if runtime_layout_payload is not None:
                            owner_compact_exchange_plan = runtime_layout_payload[
                                "owner_compact_exchange_plan"
                            ]
                            owner_compact_exchange_tensor_plan = (
                                runtime_layout_payload[
                                    "owner_compact_exchange_runtime_plan"
                                ]
                            )
                        else:
                            owner_compact_exchange_plan = (
                                runtime_layout.owner_compact_exchange_plan.as_dict()
                            )
                            owner_compact_exchange_tensor_plan = (
                                runtime_layout.owner_compact_exchange_runtime_plan
                            )
                    except (
                        AttributeError,
                        ImportError,
                        TypeError,
                        ValueError,
                        IndexError,
                    ):
                        source_partition = None
            if owner_compact_exchange_plan is None:
                owner_compact_exchange_plans = _standard_ep_owner_compact_exchange_plan(
                    owner_compact_need_masks=owner_compact_need_masks,
                    owner_compact_owner_ranks=owner_compact_owner_ranks_list,
                    owner_compact_local_experts=owner_compact_local_experts_list,
                    ep_rank=ep_rank,
                    device=x.device,
                    backend_module_name=(
                        str(balanced_moe_backend_module_name)
                        if balanced_moe_backend_module_name is not None
                        else None
                    ),
                )
                if owner_compact_exchange_plans is not None:
                    (
                        owner_compact_exchange_plan,
                        owner_compact_exchange_tensor_plan,
                    ) = owner_compact_exchange_plans
            with dsv4_timed_stage(timing_records, "standard_ep.dispatch.hot_mask"):
                if source_partition is None:
                    build_source_partition_from_offsets = None
                    for balanced_moe_backend_module in _iter_standard_ep_balanced_moe_modules(
                        str(balanced_moe_backend_module_name)
                        if balanced_moe_backend_module_name is not None
                        else None
                    ):
                        build_source_partition_from_offsets = getattr(
                            balanced_moe_backend_module,
                            "build_source_partition_from_offsets",
                            None,
                        )
                        if build_source_partition_from_offsets is not None:
                            break
                else:
                    build_source_partition_from_offsets = None
                if (
                    source_partition is None
                    and build_source_partition_from_offsets is not None
                ):
                    try:
                        source_partition = build_source_partition_from_offsets(
                            selected_experts_indices,
                            selected_global_experts=selected_hot_experts,
                            selected_owner_ranks=hot_owner_ranks,
                            selected_owner_shard_offsets=hot_owner_shard_offsets_list,
                            selected_owner_compact_offsets=hot_owner_compact_offsets_list,
                            ep_rank=ep_rank,
                            presort_by=presort_mode,
                        )
                    except (AttributeError, TypeError, ValueError, IndexError):
                        source_partition = None
                if source_partition is not None:
                    remote_hot_mask = source_partition.remote_hot_mask
                    keep_flat_mask = source_partition.keep_flat_mask
                else:
                    remote_hot_mask = torch.zeros_like(flat_experts, dtype=torch.bool)
                    hot_offsets = torch.full_like(flat_experts, -1, dtype=torch.long)

                    for hot_offset, expert in enumerate(selected_hot_experts):
                        owner_rank = hot_owner_ranks[hot_offset]
                        if owner_rank == ep_rank:
                            continue
                        expert_mask = flat_experts == int(expert)
                        remote_hot_mask |= expert_mask
                        hot_offsets = torch.where(
                            expert_mask,
                            torch.full_like(hot_offsets, int(hot_offset)),
                            hot_offsets,
                        )

                    keep_flat_mask = ~remote_hot_mask
            if _standard_ep_backend_normal_topk_dispatch_tensors():
                with dsv4_timed_stage(
                    timing_records,
                    "standard_ep.dispatch.backend_normal_topk",
                ):
                    normal_route_mask = keep_flat_mask.reshape_as(
                        selected_experts_indices
                    )
                    selected_backend_module = (
                        _select_standard_ep_balanced_moe_module_with_capability(
                            str(balanced_moe_backend_module_name)
                            if balanced_moe_backend_module_name is not None
                            else None,
                            "normal_topk_dispatch_tensors",
                        )
                    )
                    build_normal_topk_dispatch_tensors = (
                        getattr(
                            selected_backend_module,
                            "build_normal_topk_dispatch_tensors",
                            None,
                        )
                        if selected_backend_module is not None
                        else None
                    )
                    if (
                        build_normal_topk_dispatch_tensors is None
                        or selected_backend_module is None
                    ):
                        raise RuntimeError(
                            "Backend normal top-k dispatch tensor handoff was "
                            "requested, but no selected balanced-MoE backend "
                            "advertises normal_topk_dispatch_tensors and "
                            "exposes build_normal_topk_dispatch_tensors()."
                        )
                    kwargs = (
                        {"partition": source_partition}
                        if source_partition is not None
                        else {"normal_route_mask": normal_route_mask}
                    )
                    (
                        normal_topk_ids_for_backend,
                        normal_topk_weights_for_backend,
                    ) = build_normal_topk_dispatch_tensors(
                        selected_experts_indices,
                        top_scores,
                        **kwargs,
                    )
                    if _standard_ep_backend_normal_topk_dispatch_validate():
                        expected_ids = torch.where(
                            normal_route_mask,
                            selected_experts_indices,
                            selected_experts_indices.new_full(
                                selected_experts_indices.shape,
                                -1,
                            ),
                        )
                        torch.testing.assert_close(
                            normal_topk_ids_for_backend,
                            expected_ids,
                            check_dtype=True,
                            rtol=0,
                            atol=0,
                        )
                        if normal_topk_weights_for_backend is None:
                            raise RuntimeError(
                                "Backend normal top-k dispatch tensor handoff "
                                "must preserve top-k weights when top_scores are "
                                "provided."
                            )
                        expected_weights = torch.where(
                            normal_route_mask,
                            top_scores,
                            top_scores.new_zeros(top_scores.shape),
                        )
                        torch.testing.assert_close(
                            normal_topk_weights_for_backend,
                            expected_weights,
                            check_dtype=True,
                            rtol=0,
                            atol=0,
                        )
                    normal_topk_backend_module = str(
                        selected_backend_module.__name__
                    )
            if use_backend_dispatch_permute:
                with dsv4_timed_stage(
                    timing_records,
                    "standard_ep.dispatch.backend_normal_topk_dispatch_permute",
                ):
                    if self.score_before_experts:
                        raise RuntimeError(
                            "Backend normal top-k dispatch+permute is only wired "
                            "for score_before_experts=False because Primus-Turbo "
                            "combine owns the router-probability weighting."
                        )
                    if compact_collective_backend != "torch":
                        raise RuntimeError(
                            "Backend normal top-k dispatch+permute replaces the "
                            "standard normal/cold EP collective; do not combine it "
                            "with mori_native compact collectives in the same path."
                        )
                    if (
                        normal_topk_ids_for_backend is None
                        or normal_topk_weights_for_backend is None
                    ):
                        raise RuntimeError(
                            "Backend normal top-k dispatch+permute requires "
                            "normal top-k dispatch tensors. Enable "
                            "TORCHTITAN_STANDARD_EP_BACKEND_NORMAL_TOPK_DISPATCH_TENSORS=1 "
                            "or the CANARY alias."
                        )
                    backend_dispatch_module = (
                        _select_standard_ep_balanced_moe_module_with_capability(
                            str(balanced_moe_backend_module_name)
                            if balanced_moe_backend_module_name is not None
                            else None,
                            "normal_topk_ep_dispatch_permute",
                        )
                    )
                    dispatch_permute_normal_topk_tokens = (
                        getattr(
                            backend_dispatch_module,
                            "dispatch_permute_normal_topk_tokens",
                            None,
                        )
                        if backend_dispatch_module is not None
                        else None
                    )
                    if dispatch_permute_normal_topk_tokens is None:
                        raise RuntimeError(
                            "Backend normal top-k dispatch+permute was requested, "
                            "but no selected balanced-MoE backend advertises "
                            "normal_topk_ep_dispatch_permute and exposes "
                            "dispatch_permute_normal_topk_tokens()."
                        )
                    (
                        routed_input,
                        num_tokens_per_expert_group,
                        top_scores_experts_sorted,
                        normal_topk_backend_state,
                    ) = dispatch_permute_normal_topk_tokens(
                        x,
                        normal_topk_ids_for_backend,
                        normal_topk_weights_for_backend,
                        num_experts=int(self.num_experts),
                        num_local_experts=int(num_local_experts),
                        group=self.ep_mesh.get_group(),
                        num_topk=int(self.top_k),
                        use_cuda_num_tokens_per_expert=True,
                    )
                    token_indices_experts_sorted = torch.empty(
                        0, device=x.device, dtype=torch.long
                    )
                    normal_flat_positions_experts_sorted = None
                    normal_routed_rows = int(routed_input.shape[0])
                    normal_topk_backend_module = str(
                        backend_dispatch_module.__name__
                    )
            else:
                with dsv4_timed_stage(
                    timing_records, "standard_ep.dispatch.hot_local_reorder"
                ):
                    (
                        routed_input,
                        num_tokens_per_expert,
                        token_indices_experts_sorted,
                        top_scores_experts_sorted,
                        normal_flat_positions_experts_sorted,
                    ) = self._local_reorder_with_flat_mask(
                        x, top_scores, selected_experts_indices, keep_flat_mask
                    )
                    normal_routed_rows = int(routed_input.shape[0])

            if not use_backend_dispatch_permute:
                with torch.no_grad():
                    with dsv4_timed_stage(
                        timing_records, "standard_ep.dispatch.hot_count_all_to_all"
                    ):
                        num_tokens_per_expert_group = all_to_all_single(
                            num_tokens_per_expert,
                            None,
                            None,
                            group=self.ep_mesh,
                        )
                        num_tokens_per_expert_group = (
                            torch.ops._c10d_functional.wait_tensor(
                                num_tokens_per_expert_group
                            )
                        )
                    with dsv4_timed_stage(
                        timing_records, "standard_ep.dispatch.hot_split_d2h"
                    ):
                        non_blocking = not torch.compiler.is_compiling()
                        input_splits = (
                            num_tokens_per_expert.view(ep_size, -1)
                            .sum(dim=1)
                            .to(torch.device("cpu"), non_blocking=non_blocking)
                        )
                        output_splits = (
                            num_tokens_per_expert_group.view(ep_size, -1)
                            .sum(dim=1)
                            .to(torch.device("cpu"), non_blocking=False)
                        )
                        input_splits_list = [int(v) for v in input_splits.tolist()]
                        output_splits_list = [int(v) for v in output_splits.tolist()]
            else:
                with dsv4_timed_stage(
                    timing_records,
                    "standard_ep.dispatch.backend_normal_topk_split_metadata",
                ):
                    input_splits_list = [0 for _ in range(int(ep_size))]
                    output_splits_list = [0 for _ in range(int(ep_size))]

            with dsv4_timed_stage(
                timing_records, "standard_ep.dispatch.hot_metadata"
            ):
                if source_partition is not None:
                    remote_positions = source_partition.remote_flat_positions
                    remote_hot_offsets = source_partition.remote_hot_offsets
                    remote_owner_shard_offsets = (
                        source_partition.remote_owner_shard_offsets
                    )
                    remote_owner_compact_offsets = (
                        source_partition.remote_owner_compact_offsets
                    )
                else:
                    remote_positions = torch.nonzero(
                        remote_hot_mask, as_tuple=False
                    ).flatten()
                    remote_hot_offsets = hot_offsets[remote_positions]
                    if max_owned_per_rank > 0:
                        hot_to_owner_shard_offsets = torch.tensor(
                            hot_owner_shard_offsets_list,
                            device=remote_hot_offsets.device,
                            dtype=remote_hot_offsets.dtype,
                        )
                        remote_owner_shard_offsets = (
                            hot_to_owner_shard_offsets.index_select(
                                0,
                                remote_hot_offsets.to(torch.long),
                            )
                        )
                        hot_to_owner_compact_offsets = torch.tensor(
                            hot_owner_compact_offsets_list,
                            device=remote_hot_offsets.device,
                            dtype=remote_hot_offsets.dtype,
                        )
                        remote_owner_compact_offsets = (
                            hot_to_owner_compact_offsets.index_select(
                                0,
                                remote_hot_offsets.to(torch.long),
                            )
                        )
                    else:
                        remote_owner_shard_offsets = remote_hot_offsets.clone()
                        remote_owner_compact_offsets = remote_hot_offsets.clone()
                remote_hot_rows = int(remote_positions.numel())
                remote_offsets_presorted_by = (
                    source_partition.remote_offsets_presorted_by
                    if source_partition is not None
                    else ""
                )
                remote_rows_materialized_by = ""
                remote_materialized_x = None
                remote_materialized_token_indices = None
                remote_materialized_top_scores = None
                remote_materialized_group_ends = (
                    source_partition.remote_group_ends
                    if source_partition is not None
                    else None
                )
                if (
                    remote_hot_rows > 0
                    and materialize_backend == "mori_compact"
                    and not self.score_before_experts
                    and materialize_rows_mode != "owner_compact"
                ):
                    raise ValueError(
                        "MORI compact hot-row materialization is only valid for "
                        "the retained owner-compact helper ABI. Set "
                        "TORCHTITAN_STANDARD_EP_HOT_SPLIT_MATERIALIZE_ROWS="
                        "owner_compact."
                    )
                if (
                    remote_hot_rows > 0
                    and presort_mode != "off"
                    and remote_offsets_presorted_by != presort_mode
                ):
                    if (
                        presort_mode == "owner_compact"
                        and remote_owner_compact_offsets is not None
                    ):
                        presort_offsets = remote_owner_compact_offsets
                    else:
                        presort_mode = "selected"
                        presort_offsets = remote_hot_offsets
                    presort_order = torch.argsort(presort_offsets, stable=True)
                    remote_positions = remote_positions.index_select(0, presort_order)
                    remote_hot_offsets = remote_hot_offsets.index_select(
                        0, presort_order
                    )
                    if remote_owner_shard_offsets is not None:
                        remote_owner_shard_offsets = (
                            remote_owner_shard_offsets.index_select(0, presort_order)
                        )
                    if remote_owner_compact_offsets is not None:
                        remote_owner_compact_offsets = (
                            remote_owner_compact_offsets.index_select(
                                0, presort_order
                            )
                        )
                    remote_offsets_presorted_by = presort_mode
                if remote_hot_rows > 0 and materialize_rows_mode != "off":
                    materialized_offsets = (
                        remote_owner_compact_offsets
                        if (
                            remote_offsets_presorted_by == "owner_compact"
                            and remote_owner_compact_offsets is not None
                        )
                        else remote_hot_offsets
                    )
                    materialized_group_count = (
                        len(owner_shard_active_offsets)
                        if remote_offsets_presorted_by == "owner_compact"
                        else len(selected_hot_experts)
                    )
                    if remote_materialized_group_ends is None:
                        remote_materialized_group_ends = torch.cumsum(
                            torch.bincount(
                                materialized_offsets.to(torch.long),
                                minlength=int(materialized_group_count),
                            ).to(torch.int64),
                            dim=0,
                            dtype=torch.int32,
                        )
                    remote_materialized_token_indices = (
                        remote_positions // self.top_k
                    ).to(torch.long)
                    if (
                        materialize_backend == "mori_compact"
                        and not self.score_before_experts
                    ):
                        (
                            remote_materialized_x,
                            remote_materialized_top_scores,
                        ) = _standard_ep_mori_compact_materialize_rows(
                            x=x,
                            flat_scores=flat_scores,
                            remote_positions=remote_positions.to(torch.int64).contiguous(),
                            hot_owner_slots=materialized_offsets.to(torch.int64).contiguous(),
                            num_hot_slots=int(materialized_group_count),
                            top_k=int(self.top_k),
                            ep_mesh=self.ep_mesh,
                            timing_records=timing_records,
                        )
                    else:
                        if (
                            materialize_backend == "mori_compact"
                            and self.score_before_experts
                        ):
                            # The active DeepSeek-V4 contract is score_after_experts.
                            # Fall back rather than silently changing helper-row math.
                            materialize_backend = "torch"
                        remote_materialized_top_scores = flat_scores[remote_positions]
                        remote_materialized_x = x[remote_materialized_token_indices]
                        if self.score_before_experts:
                            remote_materialized_x = (
                                remote_materialized_x.to(torch.float32)
                                * remote_materialized_top_scores.reshape(-1, 1)
                            ).to(x.dtype)
                    remote_rows_materialized_by = remote_offsets_presorted_by
                _record_standard_ep_hot_expert_execute_stats(
                    call_id=standard_ep_hot_stats_call_id,
                    plan=standard_ep_execute_plan,
                    selected_hot_experts=selected_hot_experts,
                    min_reduction_pct=standard_ep_execute_min_reduction_pct,
                    modeled_reduction_pct=standard_ep_execute_modeled_reduction_pct,
                    skipped_by_threshold=standard_ep_execute_skipped_by_threshold,
                    remote_positions=remote_positions,
                    remote_hot_offsets=remote_hot_offsets,
                    keep_flat_mask=keep_flat_mask,
                    top_k=self.top_k,
                )
            standard_ep_hot_split = StandardEPHotSplitMetadata(
                selected_global_experts=selected_hot_experts,
                remote_token_indices=(remote_positions // self.top_k).to(torch.long),
                remote_flat_positions=remote_positions.to(torch.long),
                remote_top_scores=flat_scores[remote_positions],
                remote_hot_offsets=remote_hot_offsets,
                ep_rank=ep_rank,
                ep_size=int(ep_size),
                num_local_experts=num_local_experts,
                remote_owner_shard_offsets=remote_owner_shard_offsets,
                remote_owner_compact_offsets=remote_owner_compact_offsets,
                owner_counts=tuple(int(v) for v in owner_counts_list),
                owner_shard_active_offsets=owner_shard_active_offsets,
                owner_compact_owner_ranks=tuple(
                    int(v) for v in owner_compact_owner_ranks_list
                ),
                owner_compact_local_experts=tuple(
                    int(v) for v in owner_compact_local_experts_list
                ),
                owner_compact_need_masks=owner_compact_need_masks,
                owner_compact_exchange_plan=owner_compact_exchange_plan,
                owner_compact_exchange_tensor_plan=owner_compact_exchange_tensor_plan,
                max_owned_per_rank=int(max_owned_per_rank),
                remote_offsets_presorted_by=remote_offsets_presorted_by,
                remote_rows_materialized_by=remote_rows_materialized_by,
                remote_materialized_x=remote_materialized_x,
                remote_materialized_token_indices=remote_materialized_token_indices,
                remote_materialized_top_scores=remote_materialized_top_scores,
                remote_materialized_group_ends=remote_materialized_group_ends,
                balanced_moe_backend_module=(
                    str(balanced_moe_backend_module_name)
                    if balanced_moe_backend_module_name is not None
                    else None
                ),
            )
        elif standard_ep_execute_plan is not None:
            with dsv4_timed_stage(
                timing_records, "standard_ep.dispatch.hot_metadata"
            ):
                _record_standard_ep_hot_expert_execute_stats(
                    call_id=standard_ep_hot_stats_call_id,
                    plan=standard_ep_execute_plan,
                    selected_hot_experts=selected_hot_experts,
                    min_reduction_pct=standard_ep_execute_min_reduction_pct,
                    modeled_reduction_pct=standard_ep_execute_modeled_reduction_pct,
                    skipped_by_threshold=standard_ep_execute_skipped_by_threshold,
                    top_k=self.top_k,
                )

        if routed_input is None:
            with dsv4_timed_stage(timing_records, "standard_ep.dispatch.local_reorder"):
                (
                    routed_input,
                    num_tokens_per_expert,
                    token_indices_experts_sorted,
                    top_scores_experts_sorted,
                    normal_flat_positions_experts_sorted,
                ) = self._local_reorder(x, top_scores, selected_experts_indices)
                normal_routed_rows = int(routed_input.shape[0])

        assert token_indices_experts_sorted is not None
        assert top_scores_experts_sorted is not None

        if normal_topk_backend_state is not None:
            if int(self.sp_size) != 1:
                raise NotImplementedError(
                    "Backend normal top-k dispatch+permute currently supports SP=1 "
                    "only; the active DSv4 Flash/Qwen/Kimi gates use TP1/CP1."
                )
            metadata = AllToAllDispatchMetadata(
                token_indices_experts_sorted=token_indices_experts_sorted,
                top_scores_experts_sorted=top_scores_experts_sorted,
                input_shape=tuple(routed_input.shape),
                permuted_indices=torch.empty(0, device=x.device, dtype=torch.long),
                input_splits=input_splits_list,
                output_splits=output_splits_list,
                standard_ep_hot_split=standard_ep_hot_split,
                normal_flat_positions_experts_sorted=None,
                normal_top_scores_flat=None,
                normal_recv_global_flat_positions_expert_major=None,
                normal_flat_position_rank_stride=0,
                normal_compact_dispatch_layout=None,
                normal_topk_ids_for_backend=normal_topk_ids_for_backend,
                normal_topk_weights_for_backend=normal_topk_weights_for_backend,
                normal_topk_backend_module=normal_topk_backend_module,
                normal_topk_backend_state=normal_topk_backend_state,
                normal_compact_collective_backend="backend_normal_topk",
                normal_compact_collective_native_active=False,
                normal_compact_native_dispatch_summary=None,
            )
            flush_dsv4_profile_timing(
                timing_records,
                {
                    "kind": "standard_ep_dispatch",
                    "phase": "forward",
                    "ep_size": int(ep_size),
                    "top_k": int(self.top_k),
                    "x_shape": list(x.shape),
                    "routed_input_shape": list(routed_input.shape),
                    "num_tokens_per_expert_shape": list(
                        num_tokens_per_expert_group.shape
                    ),
                    "has_standard_ep_hot_split": bool(
                        standard_ep_hot_split is not None
                    ),
                    "normal_routed_rows": int(normal_routed_rows),
                    "backend_normal_topk_dispatch_ready": True,
                    "backend_normal_topk_dispatch_backend": normal_topk_backend_module,
                    "backend_normal_topk_dispatch_permute_active": True,
                    "backend_normal_topk_dispatch_numel": (
                        int(normal_topk_ids_for_backend.numel())
                        if normal_topk_ids_for_backend is not None
                        else 0
                    ),
                    "compact_collective_backend": "backend_normal_topk",
                    "compact_collective_native_active": False,
                },
            )
            return routed_input, num_tokens_per_expert_group, metadata

        if export_compact_route_positions or compact_collective_backend == "mori_native":
            if normal_flat_positions_experts_sorted is None:
                raise RuntimeError(
                    "standard EP compact-route source positions require "
                    "normal_flat_positions_experts_sorted to be populated."
                )
            if int(self.sp_size) != 1:
                raise NotImplementedError(
                    "standard EP compact-route source positions currently support SP=1 "
                    "only; the active DSv4 Flash canary uses TP1/CP1."
                )
            ep_group = self.ep_mesh.get_group()
            ep_rank = int(dist.get_rank(group=ep_group))
            normal_flat_position_rank_stride = int(x.shape[0]) * int(self.top_k)
        if export_compact_route_positions:
            normal_global_flat_positions_experts_sorted = (
                normal_flat_positions_experts_sorted.to(torch.int64)
                + int(ep_rank) * int(normal_flat_position_rank_stride)
            )
        else:
            normal_global_flat_positions_experts_sorted = None

        rank_major_input_shape = (int(sum(output_splits_list)), int(x.shape[-1]))
        num_tokens_per_expert_group_rank_major = num_tokens_per_expert_group
        normal_compact_dispatch_layout: StandardEPCompactDispatchLayout | None = (
            _build_standard_ep_compact_dispatch_layout(
                x=x,
                top_k=int(self.top_k),
                ep_size=int(ep_size),
                num_local_experts=int(
                    num_tokens_per_expert_group_rank_major.shape[0] // ep_size
                ),
                normal_flat_positions_experts_sorted=normal_flat_positions_experts_sorted,
                num_tokens_per_expert=num_tokens_per_expert,
                num_tokens_per_expert_group_rank_major=(
                    num_tokens_per_expert_group_rank_major
                ),
                input_splits_list=input_splits_list,
                output_splits_list=output_splits_list,
                rank_major_input_shape=rank_major_input_shape,
            )
        )

        # num_tokens_per_expert_group layout before native dispatch:
        #   (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
        # Native MORI compact dispatch writes expert-major rows directly. When
        # count metadata is available, native combine can derive the inverse
        # layout from counts and does not need the full rank-major index vector.
        num_local_experts = num_tokens_per_expert_group.shape[0] // ep_size
        if compact_collective_backend == "mori_native":
            with dsv4_timed_stage(
                timing_records, "standard_ep.dispatch.native_count_metadata"
            ):
                count_matrix = num_tokens_per_expert_group.to(torch.int64).view(
                    ep_size, num_local_experts
                )
                num_tokens_per_expert_group = count_matrix.sum(0).contiguous()
                permuted_indices = num_tokens_per_expert_group.new_empty((0,))

        normal_compact_native_dispatch_summary = None

        # All-to-all dispatch tokens to EP ranks
        if compact_collective_backend == "mori_native":
            if normal_compact_dispatch_layout is None:
                raise RuntimeError(
                    "mori_native compact dispatch requires "
                    "StandardEPCompactDispatchLayout."
                )
            (
                routed_input,
                normal_recv_global_flat_positions,
                normal_compact_native_dispatch_summary,
            ) = _standard_ep_native_mori_compact_dispatch(
                routed_input=routed_input,
                layout=normal_compact_dispatch_layout,
                expert_major_to_rank_major_indices=None,
                ep_mesh=self.ep_mesh,
                timing_records=timing_records,
            )
        else:
            with dsv4_timed_stage(timing_records, "standard_ep.dispatch.all_to_all"):
                routed_input = all_to_all_single_autograd(
                    routed_input,
                    output_splits_list,
                    input_splits_list,
                    self.ep_mesh,
                )
            if normal_global_flat_positions_experts_sorted is not None:
                with dsv4_timed_stage(
                    timing_records, "standard_ep.dispatch.route_pos_all_to_all"
                ):
                    normal_recv_global_flat_positions = all_to_all_single(
                        normal_global_flat_positions_experts_sorted,
                        output_splits_list,
                        input_splits_list,
                        self.ep_mesh,
                    )
                    normal_recv_global_flat_positions = (
                        torch.ops._c10d_functional.wait_tensor(
                            normal_recv_global_flat_positions
                        )
                    )
            else:
                normal_recv_global_flat_positions = None

        if compact_collective_backend == "mori_native":
            input_shape = rank_major_input_shape
            normal_recv_global_flat_positions_expert_major = (
                normal_recv_global_flat_positions
            )
        else:
            # Reorder from rank-major to expert-major via _permute.
            #
            # num_tokens_per_expert_group layout after all-to-all:
            #   (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
            # _permute reshuffles to:
            #   (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
            with dsv4_timed_stage(timing_records, "standard_ep.dispatch.permute"):
                (
                    input_shape,
                    routed_input,
                    permuted_indices,
                    num_tokens_per_expert_group,
                ) = self._permute(
                    routed_input,
                    num_tokens_per_expert_group,
                    ep_size,
                    num_local_experts,
                )
                if normal_recv_global_flat_positions is not None:
                    normal_recv_global_flat_positions_expert_major = (
                        normal_recv_global_flat_positions[permuted_indices].contiguous()
                    )

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted=token_indices_experts_sorted,
            top_scores_experts_sorted=top_scores_experts_sorted,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
            standard_ep_hot_split=standard_ep_hot_split,
            normal_flat_positions_experts_sorted=normal_flat_positions_experts_sorted,
            normal_top_scores_flat=(
                top_scores.view(-1)
                if compact_collective_backend == "mori_native"
                else None
            ),
            normal_recv_global_flat_positions_expert_major=normal_recv_global_flat_positions_expert_major,
            normal_flat_position_rank_stride=normal_flat_position_rank_stride,
            normal_compact_dispatch_layout=normal_compact_dispatch_layout,
            normal_topk_ids_for_backend=normal_topk_ids_for_backend,
            normal_topk_weights_for_backend=normal_topk_weights_for_backend,
            normal_topk_backend_module=normal_topk_backend_module,
            normal_topk_backend_state=normal_topk_backend_state,
            normal_compact_collective_backend=compact_collective_backend,
            normal_compact_collective_native_active=(
                compact_collective_backend == "mori_native"
            ),
            normal_compact_native_dispatch_summary=(
                normal_compact_native_dispatch_summary
            ),
        )
        flush_dsv4_profile_timing(
            timing_records,
            {
                "kind": "standard_ep_dispatch",
                "phase": "forward",
                "ep_size": int(ep_size),
                "top_k": int(self.top_k),
                "x_shape": list(x.shape),
                "routed_rows_after_hot_mask": int(normal_routed_rows),
                "routed_rows_after_all_to_all": int(routed_input.shape[0]),
                "selected_hot_experts": [int(v) for v in selected_hot_experts],
                "num_selected_hot_experts": int(len(selected_hot_experts)),
                "remote_hot_rows": int(remote_hot_rows),
                "has_standard_ep_hot_split": bool(standard_ep_hot_split is not None),
                "has_normal_flat_positions": bool(
                    normal_flat_positions_experts_sorted is not None
                ),
                "normal_flat_positions_rows": (
                    int(normal_flat_positions_experts_sorted.numel())
                    if normal_flat_positions_experts_sorted is not None
                    else 0
                ),
                "export_compact_route_positions": bool(export_compact_route_positions),
                "has_recv_global_flat_positions": bool(
                    normal_recv_global_flat_positions_expert_major is not None
                ),
                "recv_global_flat_positions_rows": (
                    int(normal_recv_global_flat_positions_expert_major.numel())
                    if normal_recv_global_flat_positions_expert_major is not None
                    else 0
                ),
                "flat_position_rank_stride": int(normal_flat_position_rank_stride),
                "compact_dispatch_layout_ready": bool(
                    normal_compact_dispatch_layout is not None
                ),
                "backend_normal_topk_dispatch_ready": bool(
                    normal_topk_ids_for_backend is not None
                ),
                "backend_normal_topk_dispatch_backend": (
                    normal_topk_backend_module
                ),
                "backend_normal_topk_dispatch_numel": (
                    int(normal_topk_ids_for_backend.numel())
                    if normal_topk_ids_for_backend is not None
                    else 0
                ),
                "compact_dispatch_layout_requires_extra_collective": False,
                "compact_dispatch_rank_major_rows": int(rank_major_input_shape[0]),
                "compact_dispatch_num_segments": int(
                    ep_size * (num_tokens_per_expert_group_rank_major.shape[0] // ep_size)
                ),
                "compact_collective_backend": compact_collective_backend,
                "compact_collective_native_active": bool(
                    compact_collective_backend == "mori_native"
                ),
                "compact_collective_native_dispatch_summary": (
                    normal_compact_native_dispatch_summary
                ),
                "standard_ep_hot_split_materialize_backend": materialize_backend,
                "standard_ep_balanced_moe_runtime_layout": bool(
                    use_backend_runtime_layout
                ),
                "standard_ep_mori_balanced_runtime_layout": bool(
                    use_backend_runtime_layout
                ),
                "standard_ep_mori_compact_pack_blocks": int(
                    _standard_ep_mori_compact_pack_block_num()
                ),
                "standard_ep_mori_compact_pack_warps": int(
                    _standard_ep_mori_compact_pack_warp_per_block()
                ),
                "input_splits_sum": int(sum(input_splits_list)),
                "output_splits_sum": int(sum(output_splits_list)),
            },
        )
        return routed_input, num_tokens_per_expert_group, metadata

    def _permute_indices(self, num_tokens_per_expert_group, ep_size, num_local_experts):
        """Build the rank-major -> expert-major row index sidecar."""
        num_tokens_per_expert_group = num_tokens_per_expert_group.to(torch.int64)
        device = num_tokens_per_expert_group.device
        total = num_tokens_per_expert_group.sum()

        # [R, E] matrix of token counts per (rank, expert)
        t_mat = num_tokens_per_expert_group.view(ep_size, num_local_experts)

        # Where each (r, e) segment starts in the input (rank-major order)
        input_starts = (
            num_tokens_per_expert_group.cumsum(0) - num_tokens_per_expert_group
        ).view(ep_size, num_local_experts)

        # Transpose to expert-major [E, R] and flatten
        segment_lens = t_mat.t().reshape(-1)
        input_starts = input_starts.t().reshape(-1)

        # For each output position, find its input position:
        #   output[p] = input[input_starts[seg] + (p - output_starts[seg])]
        seg_ids = torch.arange(segment_lens.shape[0], device=device).repeat_interleave(
            segment_lens
        )
        output_starts = segment_lens.cumsum(0) - segment_lens
        permuted_indices = (
            input_starts[seg_ids]
            + torch.arange(total, device=device)
            - output_starts[seg_ids]
        )

        num_tokens_per_expert = t_mat.sum(0)
        return permuted_indices, num_tokens_per_expert

    def _permute(
        self, routed_input, num_tokens_per_expert_group, ep_size, num_local_experts
    ):
        """Reorder tokens from rank-major to expert-major layout.

        Input layout:  (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
        Output layout: (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
        """
        permuted_indices, num_tokens_per_expert = self._permute_indices(
            num_tokens_per_expert_group,
            ep_size,
            num_local_experts,
        )
        return (
            routed_input.shape,
            routed_input[permuted_indices, :],
            permuted_indices,
            num_tokens_per_expert,
        )

    def _unpermute(self, routed_output, input_shape, permuted_indices):
        """Reverse expert-major reordering."""
        out_unpermuted = routed_output.new_empty(input_shape)
        out_unpermuted[permuted_indices, :] = routed_output
        return out_unpermuted

    def _backend_normal_topk_unpermute_combine(
        self,
        routed_output: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        timing_records: list[dict[str, object]] | None,
    ) -> torch.Tensor:
        if metadata.normal_topk_backend_state is None:
            raise RuntimeError("missing backend normal top-k dispatch state")
        if not metadata.normal_topk_backend_module:
            raise RuntimeError("missing backend normal top-k module name")
        if self.score_before_experts:
            raise RuntimeError(
                "Backend normal top-k combine is only wired for "
                "score_before_experts=False because the backend combine owns "
                "router-probability weighting."
            )
        if int(self.sp_size) != 1:
            raise NotImplementedError(
                "Backend normal top-k combine currently supports SP=1 only."
            )
        backend_module = _import_standard_ep_balanced_moe_module(
            metadata.normal_topk_backend_module
        )
        unpermute_combine_normal_topk_tokens = (
            getattr(
                backend_module,
                "unpermute_combine_normal_topk_tokens",
                None,
            )
            if backend_module is not None
            else None
        )
        if unpermute_combine_normal_topk_tokens is None:
            raise RuntimeError(
                "Backend normal top-k combine reached a backend without "
                "unpermute_combine_normal_topk_tokens()."
            )
        with dsv4_timed_stage(
            timing_records,
            "standard_ep.combine.backend_normal_topk_unpermute_combine",
        ):
            return unpermute_combine_normal_topk_tokens(
                routed_output,
                metadata.normal_topk_backend_state,
                group=self.ep_mesh.get_group(),
            )

    # pyrefly: ignore [bad-override]
    def _finish_combine_after_all_to_all(
        self,
        routed_output: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x: torch.Tensor,
        routed_output_shape: tuple[int, ...],
        timing_records: list[dict[str, object]] | None,
        native_source_flat_positions: torch.Tensor | None = None,
        native_combine_summary: dict[str, object] | None = None,
    ) -> torch.Tensor:
        if (
            timing_records is not None
            and not metadata.normal_compact_collective_native_active
        ):
            with dsv4_timed_stage(timing_records, "standard_ep.combine.all_to_all_wait"):
                routed_output = torch.ops._c10d_functional.wait_tensor(routed_output)

        # With SP, x is the local shard. Create full-size buffer for scatter_add
        # so routed results from all SP ranks can be placed at global positions.
        with dsv4_timed_stage(timing_records, "standard_ep.combine.allocate_out"):
            out = torch.zeros(
                x.shape[0] * self.sp_size, x.shape[-1], device=x.device, dtype=x.dtype
            )

        native_flat_position_offset = 0
        if native_source_flat_positions is not None:
            if int(self.sp_size) != 1:
                raise NotImplementedError(
                    "mori_native compact source-position scatter currently supports SP=1 only"
                )
            if native_source_flat_positions.dim() != 1:
                raise ValueError(
                    "mori_native compact source positions must be a 1D tensor"
                )
            if int(native_source_flat_positions.numel()) != int(routed_output.shape[0]):
                raise RuntimeError(
                    "mori_native compact combine returned source positions with a "
                    "different row count than source rows"
                )
            if native_source_flat_positions.dtype != torch.int64:
                raise TypeError(
                    "mori_native compact source positions must be int64"
                )
            with dsv4_timed_stage(
                timing_records, "standard_ep.combine.native_source_positions"
            ):
                source_flat_positions = native_source_flat_positions
                if int(metadata.normal_flat_position_rank_stride) > 0:
                    ep_rank = int(dist.get_rank(group=self.ep_mesh.get_group()))
                    native_flat_position_offset = ep_rank * int(
                        metadata.normal_flat_position_rank_stride
                    )
                if self.score_before_experts:
                    source_flat_positions = (
                        source_flat_positions - int(native_flat_position_offset)
                    )
                    token_indices_experts_sorted = source_flat_positions // self.top_k
                elif metadata.normal_top_scores_flat is None:
                    raise RuntimeError(
                        "mori_native compact combine needs source-order top scores "
                        "for weighted scatter"
                    )
        else:
            # With SP, token indices are 0-based within the local shard.
            # Offset to global positions for the full-size scatter buffer.
            if self.sp_size > 1:
                token_indices_experts_sorted = (
                    metadata.token_indices_experts_sorted + x.shape[0] * self.sp_rank
                )
            else:
                token_indices_experts_sorted = metadata.token_indices_experts_sorted

        if not self.score_before_experts:
            with dsv4_timed_stage(
                timing_records, "standard_ep.combine.weighted_scatter_add"
            ):
                if native_source_flat_positions is not None:
                    assert metadata.normal_top_scores_flat is not None
                    out = weighted_scatter_add_from_flat_positions(
                        out,
                        native_source_flat_positions,
                        routed_output,
                        metadata.normal_top_scores_flat,
                        top_k=int(self.top_k),
                        flat_position_offset=int(native_flat_position_offset),
                    )
                else:
                    assert isinstance(token_indices_experts_sorted, torch.Tensor)
                    out = weighted_scatter_add(
                        out,
                        token_indices_experts_sorted,
                        routed_output,
                        metadata.top_scores_experts_sorted,
                    )
        else:
            assert isinstance(token_indices_experts_sorted, torch.Tensor)
            with dsv4_timed_stage(timing_records, "standard_ep.combine.scatter_add"):
                out = deterministic_scatter_add(
                    out,
                    token_indices_experts_sorted.reshape(-1, 1).expand(-1, out.shape[-1]),
                    routed_output,
                )
        flush_dsv4_profile_timing(
            timing_records,
            {
                "kind": "standard_ep_combine",
                "phase": "forward",
                "ep_size": int(self.ep_mesh.size()),
                "top_k": int(self.top_k),
                "x_shape": list(x.shape),
                "routed_output_shape_before_combine": list(routed_output_shape),
                "has_standard_ep_hot_split": bool(
                    metadata.standard_ep_hot_split is not None
                ),
                "compact_collective_backend": str(
                    metadata.normal_compact_collective_backend
                ),
                "compact_collective_native_active": bool(
                    metadata.normal_compact_collective_native_active
                ),
                "native_source_positions_active": bool(
                    native_source_flat_positions is not None
                ),
                "native_source_positions_rows": (
                    int(native_source_flat_positions.numel())
                    if native_source_flat_positions is not None
                    else 0
                ),
                "native_source_flat_position_offset": int(native_flat_position_offset),
                "native_source_scatter_backend": (
                    os.environ.get("TORCHTITAN_STANDARD_EP_SOURCE_SCATTER_BACKEND")
                    or os.environ.get("CANARY_STANDARD_EP_SOURCE_SCATTER_BACKEND")
                    or "materialize"
                ),
                "compact_collective_native_dispatch_summary": (
                    metadata.normal_compact_native_dispatch_summary
                ),
                "compact_collective_native_combine_summary": (
                    native_combine_summary
                ),
                "input_splits_sum": int(sum(metadata.input_splits)),
                "output_splits_sum": int(sum(metadata.output_splits)),
            },
        )
        return out

    def combine_with_overlap(
        self,
        routed_output: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x: torch.Tensor,
        overlap_callback,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Combine routed rows while running a source-local callback in the a2a window."""
        if self.ep_mesh is None:
            return super().combine(routed_output, metadata, x), overlap_callback()

        timing_records = _standard_ep_stage_timing_records()
        routed_output_shape = tuple(routed_output.shape)
        if metadata.normal_topk_backend_state is not None:
            combined = self._backend_normal_topk_unpermute_combine(
                routed_output,
                metadata,
                timing_records,
            )
            overlap_result = None
            if overlap_callback is not None:
                with dsv4_timed_stage(
                    timing_records, "standard_ep.combine.overlap_callback"
                ):
                    overlap_result = overlap_callback()
            flush_dsv4_profile_timing(
                timing_records,
                {
                    "kind": "standard_ep_combine",
                    "phase": "forward",
                    "ep_size": int(self.ep_mesh.size()),
                    "top_k": int(self.top_k),
                    "x_shape": list(x.shape),
                    "routed_output_shape_before_combine": list(routed_output_shape),
                    "has_standard_ep_hot_split": bool(
                        metadata.standard_ep_hot_split is not None
                    ),
                    "compact_collective_backend": str(
                        metadata.normal_compact_collective_backend
                    ),
                    "backend_normal_topk_dispatch_permute_active": True,
                    "backend_normal_topk_dispatch_backend": (
                        metadata.normal_topk_backend_module
                    ),
                    "input_splits_sum": int(sum(metadata.input_splits)),
                    "output_splits_sum": int(sum(metadata.output_splits)),
                },
            )
            return combined, overlap_result
        native_source_flat_positions = None
        native_combine_summary = None
        native_weighted_output = bool(
            metadata.normal_compact_collective_native_active
            and _standard_ep_mori_native_compact_weighted_output()
            and not self.score_before_experts
        )
        native_flat_position_offset = 0
        if native_weighted_output:
            if metadata.normal_top_scores_flat is None:
                raise RuntimeError(
                    "mori_native compact weighted output requires saved top scores"
                )
            if int(metadata.normal_flat_position_rank_stride) > 0:
                ep_rank = int(dist.get_rank(group=self.ep_mesh.get_group()))
                native_flat_position_offset = ep_rank * int(
                    metadata.normal_flat_position_rank_stride
                )

        if metadata.normal_compact_collective_native_active:
            (
                routed_output,
                native_source_flat_positions,
                native_combine_summary,
            ) = _standard_ep_native_mori_compact_combine(
                routed_output=routed_output,
                metadata=metadata,
                ep_mesh=self.ep_mesh,
                timing_records=timing_records,
                weighted_output=native_weighted_output,
                top_scores_flat=metadata.normal_top_scores_flat,
                top_k=int(self.top_k),
                token_output_rows=int(x.shape[0]) * int(self.sp_size),
                flat_position_offset=int(native_flat_position_offset),
            )
        else:
            with dsv4_timed_stage(timing_records, "standard_ep.combine.unpermute"):
                routed_output = self._unpermute(
                    routed_output, metadata.input_shape, metadata.permuted_indices
                )

            with dsv4_timed_stage(timing_records, "standard_ep.combine.all_to_all"):
                routed_output = all_to_all_single_autograd(
                    routed_output,
                    metadata.input_splits,
                    metadata.output_splits,
                    self.ep_mesh,
                )

        overlap_result = None
        if overlap_callback is not None:
            with dsv4_timed_stage(
                timing_records, "standard_ep.combine.overlap_callback"
            ):
                overlap_result = overlap_callback()
        if native_weighted_output:
            flush_dsv4_profile_timing(
                timing_records,
                {
                    "kind": "standard_ep_combine",
                    "phase": "forward",
                    "ep_size": int(self.ep_mesh.size()),
                    "top_k": int(self.top_k),
                    "x_shape": list(x.shape),
                    "routed_output_shape_before_combine": list(routed_output_shape),
                    "has_standard_ep_hot_split": bool(
                        metadata.standard_ep_hot_split is not None
                    ),
                    "compact_collective_backend": str(
                        metadata.normal_compact_collective_backend
                    ),
                    "compact_collective_native_active": True,
                    "native_weighted_output_active": True,
                    "native_weighted_output_true_mori": True,
                    "native_weighted_output_contract": (
                        "mori_compact_combine_weighted_token_output"
                    ),
                    "native_source_positions_active": False,
                    "native_source_positions_rows": 0,
                    "native_source_flat_position_offset": int(
                        native_flat_position_offset
                    ),
                    "native_source_scatter_backend": "mori_native_weighted_combine",
                    "compact_collective_native_dispatch_summary": (
                        metadata.normal_compact_native_dispatch_summary
                    ),
                    "compact_collective_native_combine_summary": native_combine_summary,
                    "input_splits_sum": int(sum(metadata.input_splits)),
                    "output_splits_sum": int(sum(metadata.output_splits)),
                },
            )
            return routed_output, overlap_result

        out = self._finish_combine_after_all_to_all(
            routed_output,
            metadata,
            x,
            routed_output_shape,
            timing_records,
            native_source_flat_positions=native_source_flat_positions,
            native_combine_summary=native_combine_summary,
        )
        return out, overlap_result

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse the dispatch: unpermute + all-to-all + score + scatter_add.

        When sp_size > 1, dispatch uses local token indices.
        Combine offsets them to global positions so scatter_add
        into full x is correct.

        Args:
            routed_output: (R, dim) expert outputs in expert-major order
            metadata: AllToAllDispatchMetadata from dispatch()
            x: (num_tokens, dim) original input tokens

        Returns:
            (num_tokens, dim) combined output.
        """
        # EP=1: fall back to local combine (no all-to-all needed)
        if self.ep_mesh is None:
            return super().combine(routed_output, metadata, x)

        timing_records = _standard_ep_stage_timing_records()
        routed_output_shape = tuple(routed_output.shape)
        if metadata.normal_topk_backend_state is not None:
            combined = self._backend_normal_topk_unpermute_combine(
                routed_output,
                metadata,
                timing_records,
            )
            flush_dsv4_profile_timing(
                timing_records,
                {
                    "kind": "standard_ep_combine",
                    "phase": "forward",
                    "ep_size": int(self.ep_mesh.size()),
                    "top_k": int(self.top_k),
                    "x_shape": list(x.shape),
                    "routed_output_shape_before_combine": list(routed_output_shape),
                    "has_standard_ep_hot_split": bool(
                        metadata.standard_ep_hot_split is not None
                    ),
                    "compact_collective_backend": str(
                        metadata.normal_compact_collective_backend
                    ),
                    "backend_normal_topk_dispatch_permute_active": True,
                    "backend_normal_topk_dispatch_backend": (
                        metadata.normal_topk_backend_module
                    ),
                    "input_splits_sum": int(sum(metadata.input_splits)),
                    "output_splits_sum": int(sum(metadata.output_splits)),
                },
            )
            return combined
        native_source_flat_positions = None
        native_combine_summary = None
        native_weighted_output = bool(
            metadata.normal_compact_collective_native_active
            and _standard_ep_mori_native_compact_weighted_output()
            and not self.score_before_experts
        )
        native_flat_position_offset = 0
        if native_weighted_output:
            if metadata.normal_top_scores_flat is None:
                raise RuntimeError(
                    "mori_native compact weighted output requires saved top scores"
                )
            if int(metadata.normal_flat_position_rank_stride) > 0:
                ep_rank = int(dist.get_rank(group=self.ep_mesh.get_group()))
                native_flat_position_offset = ep_rank * int(
                    metadata.normal_flat_position_rank_stride
                )

        if metadata.normal_compact_collective_native_active:
            (
                routed_output,
                native_source_flat_positions,
                native_combine_summary,
            ) = _standard_ep_native_mori_compact_combine(
                routed_output=routed_output,
                metadata=metadata,
                ep_mesh=self.ep_mesh,
                timing_records=timing_records,
                weighted_output=native_weighted_output,
                top_scores_flat=metadata.normal_top_scores_flat,
                top_k=int(self.top_k),
                token_output_rows=int(x.shape[0]) * int(self.sp_size),
                flat_position_offset=int(native_flat_position_offset),
            )
        else:
            # Reverse expert-major reordering
            with dsv4_timed_stage(timing_records, "standard_ep.combine.unpermute"):
                routed_output = self._unpermute(
                    routed_output, metadata.input_shape, metadata.permuted_indices
                )

            # All-to-all combine: returns AsyncCollectiveTensor — the a2a runs
            # on the NCCL stream and won't block until the tensor is accessed.
            with dsv4_timed_stage(timing_records, "standard_ep.combine.all_to_all"):
                routed_output = all_to_all_single_autograd(
                    routed_output,
                    metadata.input_splits,
                    metadata.output_splits,
                    self.ep_mesh,
                )
        if native_weighted_output:
            flush_dsv4_profile_timing(
                timing_records,
                {
                    "kind": "standard_ep_combine",
                    "phase": "forward",
                    "ep_size": int(self.ep_mesh.size()),
                    "top_k": int(self.top_k),
                    "x_shape": list(x.shape),
                    "routed_output_shape_before_combine": list(routed_output_shape),
                    "has_standard_ep_hot_split": bool(
                        metadata.standard_ep_hot_split is not None
                    ),
                    "compact_collective_backend": str(
                        metadata.normal_compact_collective_backend
                    ),
                    "compact_collective_native_active": True,
                    "native_weighted_output_active": True,
                    "native_weighted_output_true_mori": True,
                    "native_weighted_output_contract": (
                        "mori_compact_combine_weighted_token_output"
                    ),
                    "native_source_positions_active": False,
                    "native_source_positions_rows": 0,
                    "native_source_flat_position_offset": int(
                        native_flat_position_offset
                    ),
                    "native_source_scatter_backend": "mori_native_weighted_combine",
                    "compact_collective_native_dispatch_summary": (
                        metadata.normal_compact_native_dispatch_summary
                    ),
                    "compact_collective_native_combine_summary": native_combine_summary,
                    "input_splits_sum": int(sum(metadata.input_splits)),
                    "output_splits_sum": int(sum(metadata.output_splits)),
                },
            )
            return routed_output
        return self._finish_combine_after_all_to_all(
            routed_output,
            metadata,
            x,
            routed_output_shape,
            timing_records,
            native_source_flat_positions=native_source_flat_positions,
            native_combine_summary=native_combine_summary,
        )


class TorchAOTokenDispatcher(AllToAllTokenDispatcher):
    """Token dispatcher with token group padding for quantized grouped GEMMs.

    Uses torchao's ``permute_and_pad`` instead of the standard ``_permute`` to
    reorder tokens into expert-major order and pad each expert's token group to
    a multiple of ``pad_multiple``. This alignment is required by FP8/MXFP8
    quantized grouped GEMM kernels (e.g. 16 for FP8, 32 for MXFP8).

    Requires EP to be enabled (ep_mesh must be set). Raises ValueError
    if ep_mesh is None, since quantized grouped GEMMs need padded token
    groups which are only produced by the EP permute_and_pad path.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AllToAllTokenDispatcher.Config):
        pad_multiple: int

    def __init__(self, config: Config):
        super().__init__(config)
        self.pad_multiple = config.pad_multiple

    def dispatch(self, x, top_scores, selected_experts_indices):
        if self.ep_mesh is None:
            raise ValueError(
                "TorchAOTokenDispatcher requires expert parallelism (ep_mesh must be set). "
                "Quantized grouped GEMMs need padded token groups, which requires EP>1. "
            )
        return super().dispatch(x, top_scores, selected_experts_indices)

    def _permute(
        self, routed_input, num_tokens_per_expert_group, ep_size, num_local_experts
    ):
        # FP8/MXFP8 require groups to be permuted to expert major order AND
        # padded to nearest multiple of 16.
        # It also does padding to make sure the number of tokens each expert
        # gets locally is a multiple of `self.pad_multiple`.
        # Note that this will create side effects when wrapping the for-loop
        # implementation of GroupedExperts, as it does not need padding.
        from torchao.prototype.moe_training.ep.permute import permute_and_pad

        (
            input_shape,
            routed_input,
            permuted_indices,
            num_tokens_per_expert_group_padded,
            _group_offsets,
        ) = permute_and_pad(
            routed_input,
            num_tokens_per_expert_group,
            ep_size,
            num_local_experts,
            self.pad_multiple,
        )
        return (
            input_shape,
            routed_input,
            permuted_indices,
            num_tokens_per_expert_group_padded,
        )

    def _unpermute(self, routed_output, input_shape, permuted_indices):
        # Strip the padding sentinel row added by permute_and_pad
        out_unpermuted = routed_output.new_empty(input_shape)
        out_unpermuted[permuted_indices, :] = routed_output
        return out_unpermuted[:-1]


@dataclass(frozen=True, kw_only=True)
class DeepEPDispatchMetadata:
    """Metadata for DeepEP and HybridEP token dispatch."""

    state: object  # deepep.DispatchState or hybridep.DispatchState


class DeepEPTokenDispatcher(LocalTokenDispatcher):
    """Token dispatcher using DeepEP for efficient token dispatch/combine.

    Uses DeepEP library kernels (H100/NVLink Switch) instead of standard
    all-to-all collectives. Combine is asynchronous — callers must call
    sync_combine() before using the result.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalTokenDispatcher.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.ep_mesh: DeviceMesh | None = None

        # Import to register custom ops so SAC saves communication outputs
        # instead of recomputing them. This must happen before apply_ac.
        from torchtitan.distributed.deepep import deepep  # noqa: F401

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        """Install the EP mesh used by DeepEP dispatch / combine.

        ``tp_mesh`` provides SP coordinates so combine can expand its output
        to full sequence length (matching AllToAll's convention).
        """
        self.ep_mesh = ep_mesh
        if tp_mesh is not None:
            self.sp_size = tp_mesh.size()
            self.sp_rank = tp_mesh._sym_get_coordinate(0)

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchMetadata]:
        assert self.ep_mesh is not None, (
            "ep_mesh must be set before dispatch. "
            "ExpertParallel._partition_fn() should set it."
        )
        ep_group = self.ep_mesh.get_group()
        num_local_experts = self.num_experts // ep_group.size()

        from torchtitan.distributed.deepep.deepep import dispatch_tokens

        hidden_states, tokens_per_expert, state = dispatch_tokens(
            x,
            selected_experts_indices,
            top_scores,
            num_local_experts,
            self.num_experts,
            ep_group,
            score_before_experts=self.score_before_experts,
        )

        metadata = DeepEPDispatchMetadata(state=state)
        return hidden_states, tokens_per_expert, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Combine tokens via DeepEP.

        When sp_size == 1, combine is async — sync_combine() is deferred
        to MoE.forward, enabling overlap with shared_experts.
        When sp_size > 1, there is no overlap: sync is forced here because
        the SP expansion must read the combine result before returning.
        """
        from torchtitan.distributed.deepep.deepep import combine_tokens, sync_combine

        # pyrefly: ignore [bad-argument-type]
        routed_output = combine_tokens(routed_output, metadata.state)

        if self.sp_size > 1:
            sync_combine()
            out = torch.zeros(
                routed_output.shape[0] * self.sp_size,
                routed_output.shape[-1],
                device=routed_output.device,
                dtype=routed_output.dtype,
            )
            offset = routed_output.shape[0] * self.sp_rank
            out[offset : offset + routed_output.shape[0]] = routed_output
            return out

        return routed_output


class MoriEPTokenDispatcher(LocalTokenDispatcher):
    """MORI EP dispatcher contract for AMD training.

    This is intentionally a guarded integration boundary, not a silent
    fallback to PyTorch all-to-all.  MORI's Python API can dispatch/combine
    packed standard-MoE tensors, but TorchTitan still needs a training-safe
    autograd wrapper and score-gradient path before it can replace
    AllToAllTokenDispatcher in DSv4 pretraining.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalTokenDispatcher.Config):
        kernel_type: str = "intranode"
        max_num_inp_token_per_rank: int = 4096
        use_standard_moe_adapt: bool = True

    def __init__(self, config: Config):
        super().__init__(config)
        self.kernel_type = config.kernel_type
        self.max_num_inp_token_per_rank = config.max_num_inp_token_per_rank
        self.use_standard_moe_adapt = config.use_standard_moe_adapt
        self.ep_mesh: DeviceMesh | None = None
        self.sp_size: int = 1
        self.sp_rank: int | torch.SymInt = 0

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        self.ep_mesh = ep_mesh
        if tp_mesh is not None:
            self.sp_size = tp_mesh.size()
            self.sp_rank = tp_mesh._sym_get_coordinate(0)

    def training_bridge_enabled(self) -> bool:
        import os

        return os.environ.get("TORCHTITAN_EXPERIMENTAL_MORI_AITER_MOE", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _raise_not_training_ready(self) -> None:
        raise RuntimeError(
            "MoriEPTokenDispatcher was selected, but the MORI training path is "
            "not enabled. Set TORCHTITAN_EXPERIMENTAL_MORI_AITER_MOE=1 to "
            "exercise the guarded MORI dispatch + AITER fused_moe + MORI "
            "combine bridge. That bridge currently uses a slow PyTorch "
            "reference backward; use comm_backend='standard' for the current "
            "passing direct TorchTitan canary."
        )

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ):
        if self.ep_mesh is None:
            return super().dispatch(x, top_scores, selected_experts_indices)
        self._raise_not_training_ready()

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: LocalDispatchMetadata,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.ep_mesh is None:
            return super().combine(routed_output, metadata, x)
        self._raise_not_training_ready()


class HybridEPTokenDispatcher(LocalTokenDispatcher):
    """Token dispatcher using HybridEP for efficient token dispatch/combine.

    Uses HybridEP library kernels (GB200/NVLink72) instead of standard
    all-to-all collectives.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalTokenDispatcher.Config):
        """Config for HybridEP token dispatcher.

        Args:
            non_blocking_capacity_factor: Enable non-blocking HybridEP dispatch
                with a given capacity factor.

                Setting this to a float in (0, 1] enables CPU-free non-blocking
                dispatch and controls num_permuted_tokens — the fused-permute
                output capacity, estimated as:
                num_tokens × ep_size × min(num_local_experts, top_k) × cf,
                aligned for MXFP8.  Tokens whose permuted offset exceeds this
                limit are silently dropped (overflow_flag is set on GPU).

                - None = blocking mode (default).  HybridEP calls
                  cudaStreamSynchronize after dispatch, copies
                  tokens_per_expert to pinned CPU memory, and computes the
                  exact num_permuted_tokens on the host.  No token dropping.
                - 1.0 = non-blocking, worst-case sizing: every token can reach
                  every local expert, no drops, highest memory.
                - < 1.0 = non-blocking, reduced memory; controls the
                  fused-permute output tensor size (num_permuted_tokens).
                  Safe in practice when forced load balancing (e.g. aux-loss /
                  round-robin) keeps distribution roughly uniform.

                Note: this factor has no lasting effect on the all-to-all
                communication buffer.  HybridEP's dispatch_with_permute
                internally passes the actual num_tokens to
                update_template_config, which auto-grows the buffer to the
                full token count on the first dispatch regardless of this
                setting.
        """

        non_blocking_capacity_factor: float | None = None
        pad_multiple: int | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.non_blocking_capacity_factor = config.non_blocking_capacity_factor
        self.pad_multiple = config.pad_multiple
        self.ep_mesh: DeviceMesh | None = None
        self.sp_size: int = 1
        self.sp_rank: int | torch.SymInt = 0

        # Import to register custom ops so SAC saves communication outputs
        # instead of recomputing them. This must happen before apply_ac.
        from torchtitan.distributed.deepep import hybridep  # noqa: F401

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        """Install the EP mesh used by HybridEP dispatch / combine.

        ``tp_mesh`` provides SP coordinates so combine can expand its output
        to full sequence length (matching AllToAll's convention).
        """
        self.ep_mesh = ep_mesh
        if tp_mesh is not None:
            self.sp_size = tp_mesh.size()
            self.sp_rank = tp_mesh._sym_get_coordinate(0)

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchMetadata]:
        assert self.ep_mesh is not None, (
            "ep_mesh must be set before dispatch. "
            "ExpertParallel._partition_fn() should set it."
        )
        ep_group = self.ep_mesh.get_group()
        num_local_experts = self.num_experts // ep_group.size()

        from torchtitan.distributed.deepep.hybridep import dispatch_tokens

        hidden_states, tokens_per_expert, state = dispatch_tokens(
            x,
            selected_experts_indices,
            top_scores,
            num_local_experts,
            self.num_experts,
            ep_group,
            score_before_experts=self.score_before_experts,
            non_blocking_expert_capacity_factor=self.non_blocking_capacity_factor,
            pad_multiple=self.pad_multiple,
        )

        metadata = DeepEPDispatchMetadata(state=state)
        return hidden_states, tokens_per_expert, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Combine tokens via HybridEP."""
        from torchtitan.distributed.deepep import hybridep

        routed_output = hybridep.combine_tokens(
            routed_output,
            metadata.state,  # pyrefly: ignore [bad-argument-type]
            pad_multiple=self.pad_multiple,
        )

        if self.sp_size > 1:
            out = torch.zeros(
                routed_output.shape[0] * self.sp_size,
                routed_output.shape[-1],
                device=routed_output.device,
                dtype=routed_output.dtype,
            )
            offset = routed_output.shape[0] * self.sp_rank
            out[offset : offset + routed_output.shape[0]] = routed_output
            return out

        return routed_output
