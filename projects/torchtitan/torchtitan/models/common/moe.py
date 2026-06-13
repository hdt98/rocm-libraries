# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import lru_cache
import importlib
import os
import time
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.dsv4_profile_timing import (
    dsv4_profile_timing_enabled,
    dsv4_timed_stage,
    flush_dsv4_profile_timing,
)
from torchtitan.models.common.nn_modules import Linear
from torchtitan.ops.scatter_add import deterministic_scatter_add, weighted_scatter_add
from torchtitan.protocols.module import Module

from .token_dispatcher import (
    AllToAllDispatchMetadata,
    DeepEPTokenDispatcher,
    LocalTokenDispatcher,
    MoriEPTokenDispatcher,
    StandardEPHotSplitMetadata,
)

_STANDARD_EP_BALANCED_MOE_MODULE_CANDIDATES = (
    "mori.ops.balanced_moe",
    "primus_turbo.pytorch.ops.moe.balanced_moe",
)


@lru_cache(maxsize=None)
def _import_standard_ep_balanced_moe_module(module_name: str):
    return importlib.import_module(module_name)


def _iter_standard_ep_balanced_moe_modules(preferred_module_name: str | None = None):
    seen: set[str] = set()
    candidates: list[str] = []
    if preferred_module_name:
        candidates.append(str(preferred_module_name))
    candidates.extend(_STANDARD_EP_BALANCED_MOE_MODULE_CANDIDATES)
    for module_name in candidates:
        module_name = str(module_name).strip()
        if not module_name or module_name in seen:
            continue
        seen.add(module_name)
        try:
            yield module_name, _import_standard_ep_balanced_moe_module(module_name)
        except Exception:
            continue


def _standard_ep_balanced_moe_module_matches_impl(
    module_name: str,
    backend_module,
    exchange_impl: str,
) -> bool:
    if exchange_impl in {"auto", "backend"}:
        return True

    capabilities_fn = getattr(
        backend_module,
        "balanced_moe_backend_capabilities",
        None,
    )
    capabilities = {}
    backend_label = ""
    if capabilities_fn is not None:
        try:
            capabilities = capabilities_fn()
            backend_label = str(capabilities.get("backend", "")).strip().lower()
        except Exception:
            backend_label = ""

    if exchange_impl == "native":
        return bool(capabilities.get("native_hot_helper_transport", False))

    module_name_lower = str(module_name).strip().lower()
    if exchange_impl == "mori":
        return (
            backend_label == "mori"
            or module_name_lower == "mori.ops.balanced_moe"
            or module_name_lower.startswith("mori.")
        )
    if exchange_impl in {"primus", "primus_turbo"}:
        return (
            backend_label == "primus_turbo"
            or "primus_turbo" in module_name_lower
            or "primus-turbo" in module_name_lower
        )
    return True


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _count_route_ids(route_ids: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Count discrete route ids without relying on floating-point histograms."""
    if num_bins <= 0:
        return torch.empty(0, device=route_ids.device, dtype=torch.int64)
    return torch.bincount(
        route_ids.reshape(-1).to(torch.int64),
        minlength=num_bins,
    )[:num_bins]


def _profile_cumulative_offsets_summary(
    offsets: torch.Tensor | None,
) -> dict[str, object]:
    """Compact row-mix fingerprint for profile-only helper VJP comparisons."""
    if offsets is None or not dsv4_profile_timing_enabled():
        return {}
    try:
        ends = [int(v) for v in offsets.detach().to("cpu").tolist()]
    except Exception as exc:
        return {"row_group_summary_error": type(exc).__name__}
    counts: list[int] = []
    prev = 0
    for end in ends:
        count = max(0, int(end) - int(prev))
        counts.append(count)
        prev = int(end)
    nonzero = [count for count in counts if count > 0]
    sorted_nonzero = sorted(nonzero, reverse=True)
    total = int(sum(counts))
    return {
        "row_group_count": len(counts),
        "row_group_nonzero_count": len(nonzero),
        "row_group_total_rows": total,
        "row_group_max_rows": int(sorted_nonzero[0]) if sorted_nonzero else 0,
        "row_group_min_nonzero_rows": int(sorted_nonzero[-1]) if sorted_nonzero else 0,
        "row_group_top8_rows": [int(v) for v in sorted_nonzero[:8]],
        "row_group_counts": [int(v) for v in counts],
    }


def _standard_ep_local_expert_backend() -> str:
    backend = (
        os.environ.get("TORCHTITAN_STANDARD_EP_LOCAL_EXPERT_BACKEND")
        or os.environ.get("CANARY_STANDARD_EP_LOCAL_EXPERT_BACKEND")
        or "grouped_mm"
    )
    backend = backend.strip().lower()
    if backend == "":
        backend = "grouped_mm"
    if backend not in {"grouped_mm", "aiter_stage1_recompute", "aiter_compact"}:
        raise ValueError(
            "TORCHTITAN_STANDARD_EP_LOCAL_EXPERT_BACKEND must be "
            "'grouped_mm', 'aiter_stage1_recompute', or 'aiter_compact', "
            f"got {backend!r}."
        )
    return backend


def _standard_ep_hot_split_weight_layout() -> str:
    layout = (
        os.environ.get("TORCHTITAN_STANDARD_EP_HOT_SPLIT_WEIGHT_LAYOUT")
        or os.environ.get("CANARY_STANDARD_EP_HOT_SPLIT_WEIGHT_LAYOUT")
        or "selected"
    )
    layout = layout.strip().lower()
    if layout == "":
        layout = "selected"
    if layout not in {"selected", "owner_sharded", "needed"}:
        raise ValueError(
            "TORCHTITAN_STANDARD_EP_HOT_SPLIT_WEIGHT_LAYOUT must be "
            f"'selected', 'owner_sharded', or 'needed', got {layout!r}."
        )
    return layout


def _standard_ep_hot_split_weight_order() -> str:
    order = (
        os.environ.get("TORCHTITAN_STANDARD_EP_HOT_SPLIT_WEIGHT_ORDER")
        or os.environ.get("CANARY_STANDARD_EP_HOT_SPLIT_WEIGHT_ORDER")
        or "owner_compact"
    )
    order = order.strip().lower()
    if order == "":
        order = "selected"
    if order not in {"selected", "owner_compact"}:
        raise ValueError(
            "TORCHTITAN_STANDARD_EP_HOT_SPLIT_WEIGHT_ORDER must be "
            f"'selected' or 'owner_compact', got {order!r}."
        )
    return order


def _standard_ep_needed_weight_max_density() -> float:
    return _env_float(
        "TORCHTITAN_STANDARD_EP_NEEDED_WEIGHT_MAX_DENSITY",
        _env_float("CANARY_STANDARD_EP_NEEDED_WEIGHT_MAX_DENSITY", 0.5),
    )


def _standard_ep_needed_weight_exchange_impl() -> str:
    impl = (
        os.environ.get("TORCHTITAN_STANDARD_EP_NEEDED_WEIGHT_EXCHANGE_IMPL")
        or os.environ.get("CANARY_STANDARD_EP_NEEDED_WEIGHT_EXCHANGE_IMPL")
        or "auto"
    )
    impl = impl.strip().lower()
    if impl == "":
        impl = "auto"
    if impl not in {
        "auto",
        "backend",
        "local",
        "mori",
        "native",
        "primus",
        "primus_turbo",
    }:
        raise ValueError(
            "TORCHTITAN_STANDARD_EP_NEEDED_WEIGHT_EXCHANGE_IMPL must be "
            "'auto', 'local', 'backend', 'mori', or 'primus_turbo', got "
            f"{impl!r}."
        )
    return impl


def _standard_ep_needed_weight_exchange_transport() -> str | None:
    transport = (
        os.environ.get("TORCHTITAN_STANDARD_EP_NEEDED_WEIGHT_EXCHANGE_TRANSPORT")
        or os.environ.get("CANARY_STANDARD_EP_NEEDED_WEIGHT_EXCHANGE_TRANSPORT")
    )
    if transport is None:
        return None
    transport = transport.strip().lower()
    if transport == "":
        return None
    transport_aliases = {
        "torch": "torch_distributed_all_to_all_single",
        "all_to_all_single": "torch_distributed_all_to_all_single",
        "torch_distributed_all_to_all_single": "torch_distributed_all_to_all_single",
        "native": "native",
        "primus_turbo_moe_dispatch": "primus_turbo_moe_dispatch",
        "mori_sdma": "mori_sdma_padded_all2all",
        "mori_sdma_padded": "mori_sdma_padded_all2all",
        "mori_sdma_padded_all2all": "mori_sdma_padded_all2all",
    }
    if transport not in transport_aliases:
        raise ValueError(
            "TORCHTITAN_STANDARD_EP_NEEDED_WEIGHT_EXCHANGE_TRANSPORT must be "
            "'torch_distributed_all_to_all_single', 'native', "
            "'primus_turbo_moe_dispatch', or 'mori_sdma_padded_all2all', "
            f"got {transport!r}."
        )
    return transport_aliases[transport]


def _standard_ep_balanced_moe_capabilities(
    *,
    backend_name: str,
    backend_module,
) -> dict[str, Any]:
    capabilities_fn = getattr(
        backend_module,
        "balanced_moe_backend_capabilities",
        None,
    )
    if capabilities_fn is None:
        return {}
    capabilities = capabilities_fn()
    if not isinstance(capabilities, dict):
        raise TypeError(
            f"Balanced-MoE backend {backend_name!r} returned non-dict "
            f"capabilities {type(capabilities)!r}."
        )
    return capabilities


def _validate_standard_ep_needed_weight_exchange_transport(
    transport: str | None,
    *,
    backend_name: str,
    backend_module,
    exchange_impl: str = "auto",
) -> str | None:
    capabilities = _standard_ep_balanced_moe_capabilities(
        backend_name=backend_name,
        backend_module=backend_module,
    )
    if transport == "native" or (transport is None and exchange_impl == "native"):
        native_transport = capabilities.get(
            "native_owner_compact_exchange_transport",
            None,
        )
        if native_transport is None or str(native_transport).strip() == "":
            backend_label = str(capabilities.get("backend", backend_name))
            raise ValueError(
                "Native owner-compact hot-row exchange was requested, but "
                f"balanced-MoE backend {backend_label!r} ({backend_name}) does "
                "not advertise native_owner_compact_exchange_transport."
            )
        transport = str(native_transport)
    if transport is None:
        return None
    if not capabilities:
        if transport == "torch_distributed_all_to_all_single":
            return transport
        raise ValueError(
            f"Balanced-MoE backend {backend_name!r} does not expose "
            "balanced_moe_backend_capabilities(); refusing requested "
            f"owner-compact hot-row transport {transport!r}."
        )
    supported = tuple(
        str(v)
        for v in capabilities.get("owner_compact_exchange_transports", ())
    )
    if transport not in supported:
        backend_label = str(capabilities.get("backend", backend_name))
        raise ValueError(
            "Requested owner-compact hot-row transport "
            f"{transport!r} is not supported by balanced-MoE backend "
            f"{backend_label!r} ({backend_name}). Supported transports: "
            f"{supported!r}."
        )
    return transport


def _standard_ep_hot_split_backward_mode() -> str:
    mode = (
        os.environ.get("TORCHTITAN_STANDARD_EP_HOT_SPLIT_BACKWARD_MODE")
        or os.environ.get("CANARY_STANDARD_EP_HOT_SPLIT_BACKWARD_MODE")
        or "autograd"
    )
    mode = mode.strip().lower()
    if mode == "":
        mode = "autograd"
    if mode not in {"autograd", "manual_loop", "manual_tgmm"}:
        raise ValueError(
            "TORCHTITAN_STANDARD_EP_HOT_SPLIT_BACKWARD_MODE must be "
            "'autograd', 'manual_loop', or 'manual_tgmm', got "
            f"{mode!r}."
        )
    return mode


def _standard_ep_hot_split_swiglu_backend() -> str:
    backend = (
        os.environ.get("TORCHTITAN_STANDARD_EP_HOT_SPLIT_SWIGLU_BACKEND")
        or os.environ.get("CANARY_STANDARD_EP_HOT_SPLIT_SWIGLU_BACKEND")
        or "torch"
    )
    backend = backend.strip().lower()
    if backend == "":
        backend = "torch"
    if backend not in {"torch", "triton"}:
        raise ValueError(
            "TORCHTITAN_STANDARD_EP_HOT_SPLIT_SWIGLU_BACKEND must be "
            f"'torch' or 'triton', got {backend!r}."
        )
    return backend


def _standard_ep_hot_split_weighted_scatter_backend() -> str | None:
    backend = (
        os.environ.get("TORCHTITAN_STANDARD_EP_HOT_SPLIT_WEIGHTED_SCATTER_BACKEND")
        or os.environ.get("CANARY_STANDARD_EP_HOT_SPLIT_WEIGHTED_SCATTER_BACKEND")
    )
    if backend is None:
        return None
    backend = backend.strip().lower()
    if backend == "":
        return None
    if backend not in {
        "torch",
        "deterministic",
        "off",
        "torch_custom",
        "custom",
        "torch_chunked",
        "chunked",
        "triton_bwd",
        "torch_triton_bwd",
        "torch_forward_triton_backward",
        "triton",
    }:
        raise ValueError(
            "TORCHTITAN_STANDARD_EP_HOT_SPLIT_WEIGHTED_SCATTER_BACKEND must be "
            "'torch', 'torch_custom', 'torch_chunked', 'triton_bwd', or "
            "'triton', got "
            f"{backend!r}."
        )
    return backend


def _standard_ep_hot_split_attach_materialized_grad() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_HOT_SPLIT_ATTACH_MATERIALIZED_GRAD",
        _env_flag("CANARY_STANDARD_EP_HOT_SPLIT_ATTACH_MATERIALIZED_GRAD", True),
    )


def _standard_ep_overlap_hot_split_forward() -> bool:
    return _env_flag(
        "TORCHTITAN_STANDARD_EP_OVERLAP_HOT_SPLIT_FORWARD",
        _env_flag("CANARY_STANDARD_EP_OVERLAP_HOT_SPLIT_FORWARD", False),
    )


def _standard_ep_hot_weight_backend(env_name: str) -> str:
    backend = (
        os.environ.get(env_name)
        or os.environ.get(env_name.replace("TORCHTITAN_", "CANARY_"))
        or "torch"
    )
    backend = backend.strip().lower()
    if backend == "":
        backend = "torch"
    allowed = {"torch", "mori_sdma"}
    if "FORWARD" in env_name:
        allowed.add("torch_allgather")
    if backend not in allowed:
        allowed_str = "', '".join(sorted(allowed))
        raise ValueError(f"{env_name} must be '{allowed_str}', got {backend!r}.")
    return backend


def _standard_ep_hot_weight_forward_backend() -> str:
    return _standard_ep_hot_weight_backend(
        "TORCHTITAN_STANDARD_EP_HOT_WEIGHT_FORWARD_BACKEND"
    )


def _standard_ep_hot_weight_reduce_backend() -> str:
    return _standard_ep_hot_weight_backend(
        "TORCHTITAN_STANDARD_EP_HOT_WEIGHT_REDUCE_BACKEND"
    )


def _standard_ep_hot_split_backward_timing_enabled() -> bool:
    return dsv4_profile_timing_enabled() and (
        _env_flag("TORCHTITAN_STANDARD_EP_HOT_SPLIT_BACKWARD_TIMING")
        or _env_flag("CANARY_STANDARD_EP_HOT_SPLIT_BACKWARD_TIMING")
    )


def _standard_ep_hot_split_autograd_vjp_detail_enabled() -> bool:
    return _standard_ep_hot_split_backward_timing_enabled() and (
        _env_flag("TORCHTITAN_STANDARD_EP_HOT_SPLIT_AUTOGRAD_VJP_DETAIL")
        or _env_flag("CANARY_STANDARD_EP_HOT_SPLIT_AUTOGRAD_VJP_DETAIL")
    )


_STANDARD_EP_HOT_SPLIT_BWD_SPAN_ID = 0
_STANDARD_EP_HOT_SPLIT_BWD_SPANS: dict[int, dict[str, Any]] = {}


def _new_standard_ep_hot_split_bwd_span(
    stage: str,
    meta: dict[str, Any],
) -> int:
    global _STANDARD_EP_HOT_SPLIT_BWD_SPAN_ID
    _STANDARD_EP_HOT_SPLIT_BWD_SPAN_ID += 1
    span_id = _STANDARD_EP_HOT_SPLIT_BWD_SPAN_ID
    _STANDARD_EP_HOT_SPLIT_BWD_SPANS[span_id] = {
        "stage": stage,
        "meta": meta,
    }
    return span_id


def _start_standard_ep_hot_split_bwd_span(span_id: int) -> None:
    state = _STANDARD_EP_HOT_SPLIT_BWD_SPANS.get(int(span_id))
    if state is None or state.get("started"):
        return
    state["started"] = True
    state["wall_start"] = time.perf_counter()
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        state["start_event"] = start_event


def _end_standard_ep_hot_split_bwd_span(span_id: int) -> None:
    state = _STANDARD_EP_HOT_SPLIT_BWD_SPANS.pop(int(span_id), None)
    if state is None or not state.get("started"):
        return
    record: dict[str, Any] = {
        "stage": state["stage"],
        "wall_issue_ms": (time.perf_counter() - float(state["wall_start"])) * 1000.0,
    }
    if torch.cuda.is_available():
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        record["_start_event"] = state["start_event"]
        record["_end_event"] = end_event
    flush_dsv4_profile_timing(
        [record],
        {
            "kind": "standard_ep_hot_split_backward",
            "phase": "backward",
            **state["meta"],
        },
    )


class _StandardEPHotSplitBackwardSpanStart(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
        span_id: int,
    ) -> torch.Tensor:
        ctx.span_id = int(span_id)
        return tensor

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        _start_standard_ep_hot_split_bwd_span(ctx.span_id)
        return grad_output, None


class _StandardEPHotSplitBackwardBoundary(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
        end_span_id: int,
        start_span_id: int,
    ) -> torch.Tensor:
        ctx.end_span_id = int(end_span_id)
        ctx.start_span_id = int(start_span_id)
        return tensor

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        _end_standard_ep_hot_split_bwd_span(ctx.end_span_id)
        _start_standard_ep_hot_split_bwd_span(ctx.start_span_id)
        return grad_output, None, None


class _StandardEPHotSplitBackwardSpanEnd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
        span_id: int,
    ) -> torch.Tensor:
        ctx.span_id = int(span_id)
        return tensor

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        _end_standard_ep_hot_split_bwd_span(ctx.span_id)
        return grad_output, None


class _StandardEPHotSplitMaterializedRows(torch.autograd.Function):
    """Return saved hot-helper rows while routing compact dX back to source tokens."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        materialized_x: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        ctx.x_shape = tuple(x.shape)
        ctx.save_for_backward(token_indices.to(torch.long).contiguous())
        return materialized_x

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        (token_indices,) = ctx.saved_tensors
        timing_records = (
            []
            if _standard_ep_hot_split_backward_timing_enabled()
            else None
        )
        with dsv4_timed_stage(
            timing_records,
            "standard_ep.hot_split.backward.compact_dx_return.index_add",
        ):
            grad_x = grad_output.new_zeros(ctx.x_shape)
            grad_x.index_add_(0, token_indices, grad_output)
        if timing_records is not None:
            flush_dsv4_profile_timing(
                timing_records,
                {
                    "kind": "standard_ep_hot_split_compact_return",
                    "phase": "backward",
                    "target": "x",
                    "rows": int(token_indices.numel()),
                    "grad_output_shape": list(grad_output.shape),
                    "x_shape": list(ctx.x_shape),
                },
            )
        return grad_x, None, None


class _StandardEPHotSplitMaterializedScores(torch.autograd.Function):
    """Return saved hot-helper scores while routing compact dTopK to top_scores."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        top_scores: torch.Tensor,
        materialized_scores: torch.Tensor,
        flat_positions: torch.Tensor,
    ) -> torch.Tensor:
        ctx.top_scores_shape = tuple(top_scores.shape)
        ctx.save_for_backward(flat_positions.to(torch.long).contiguous())
        return materialized_scores

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        (flat_positions,) = ctx.saved_tensors
        timing_records = (
            []
            if _standard_ep_hot_split_backward_timing_enabled()
            else None
        )
        with dsv4_timed_stage(
            timing_records,
            "standard_ep.hot_split.backward.compact_dtopk_return.index_add",
        ):
            grad_scores = grad_output.new_zeros(ctx.top_scores_shape)
            grad_scores_flat = grad_scores.reshape(-1)
            grad_scores_flat.index_add_(0, flat_positions, grad_output.reshape(-1))
        if timing_records is not None:
            flush_dsv4_profile_timing(
                timing_records,
                {
                    "kind": "standard_ep_hot_split_compact_return",
                    "phase": "backward",
                    "target": "top_scores",
                    "rows": int(flat_positions.numel()),
                    "grad_output_shape": list(grad_output.shape),
                    "top_scores_shape": list(ctx.top_scores_shape),
                },
            )
        return grad_scores, None, None


def _standard_ep_swiglu_backward(
    dhidden: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_f = gate.float()
    up_f = up.float()
    dh_f = dhidden.float()
    sigmoid_gate = torch.sigmoid(gate_f)
    silu_gate = gate_f * sigmoid_gate
    silu_grad = sigmoid_gate * (1.0 + gate_f * (1.0 - sigmoid_gate))
    dup = dh_f * silu_gate
    dgate = dh_f * up_f * silu_grad
    return dgate.to(dhidden.dtype), dup.to(dhidden.dtype)


if triton is not None:

    @triton.jit
    def _standard_ep_triton_swiglu_forward_kernel(
        gate_ptr,
        up_ptr,
        hidden_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ) -> None:
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-gate))
        hidden = gate * sig * up
        tl.store(hidden_ptr + offsets, hidden, mask=mask)

    @triton.jit
    def _standard_ep_triton_swiglu_backward_kernel(
        dhidden_ptr,
        gate_ptr,
        up_ptr,
        dgate_ptr,
        dup_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ) -> None:
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        dhidden = tl.load(dhidden_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-gate))
        silu = gate * sig
        silu_grad = sig * (1.0 + gate * (1.0 - sig))
        dup = dhidden * silu
        dgate = dhidden * up * silu_grad
        tl.store(dgate_ptr + offsets, dgate, mask=mask)
        tl.store(dup_ptr + offsets, dup, mask=mask)

else:
    _standard_ep_triton_swiglu_forward_kernel = None
    _standard_ep_triton_swiglu_backward_kernel = None


class _StandardEPHotSplitTritonSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        if triton is None or _standard_ep_triton_swiglu_forward_kernel is None:
            raise RuntimeError(
                "Triton hot-split SwiGLU backend requested, but triton is not importable."
            )
        if not gate.is_cuda or not up.is_cuda:
            raise RuntimeError("Triton hot-split SwiGLU backend requires CUDA/ROCm tensors.")
        if not gate.is_contiguous():
            gate = gate.contiguous()
        if not up.is_contiguous():
            up = up.contiguous()
        hidden = torch.empty_like(gate)
        block = 1024
        grid = (triton.cdiv(gate.numel(), block),)
        _standard_ep_triton_swiglu_forward_kernel[grid](
            gate,
            up,
            hidden,
            gate.numel(),
            BLOCK=block,
        )
        ctx.save_for_backward(gate, up)
        return hidden

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors
        if not grad_hidden.is_contiguous():
            grad_hidden = grad_hidden.contiguous()
        dgate = torch.empty_like(gate)
        dup = torch.empty_like(up)
        block = 1024
        grid = (triton.cdiv(grad_hidden.numel(), block),)
        _standard_ep_triton_swiglu_backward_kernel[grid](
            grad_hidden,
            gate,
            up,
            dgate,
            dup,
            grad_hidden.numel(),
            BLOCK=block,
        )
        return dgate, dup


def _standard_ep_hot_split_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if _standard_ep_hot_split_swiglu_backend() == "triton":
        return _StandardEPHotSplitTritonSwiGLU.apply(gate, up)
    return F.silu(gate) * up


def _standard_ep_hot_split_tgmm_op():
    try:
        from aiter.ops.triton.gmm import ptgmm
    except ImportError as exc:
        raise RuntimeError(
            "TORCHTITAN_STANDARD_EP_HOT_SPLIT_BACKWARD_MODE=manual_tgmm "
            "requires aiter.ops.triton.gmm.ptgmm in the active runtime."
        ) from exc
    return ptgmm


def _standard_ep_hot_split_wgrad_loop(
    *,
    sorted_x: torch.Tensor,
    hidden: torch.Tensor,
    dgate: torch.Tensor,
    dup: torch.Tensor,
    grad_y: torch.Tensor,
    offsets: torch.Tensor,
    w1_hot: torch.Tensor,
    w3_hot: torch.Tensor,
    w2_hot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dw1 = torch.zeros_like(w1_hot)
    dw3 = torch.zeros_like(w3_hot)
    dw2 = torch.zeros_like(w2_hot)
    counts_cpu = [int(v) for v in offsets.detach().cpu().tolist()]
    start = 0
    for expert_idx, end in enumerate(counts_cpu):
        if end <= start:
            continue
        x_e = sorted_x[start:end]
        hidden_e = hidden[start:end]
        dgate_e = dgate[start:end]
        dup_e = dup[start:end]
        grad_y_e = grad_y[start:end]
        dw2[expert_idx] = torch.mm(grad_y_e.transpose(0, 1), hidden_e)
        dw1[expert_idx] = torch.mm(dgate_e.transpose(0, 1), x_e)
        dw3[expert_idx] = torch.mm(dup_e.transpose(0, 1), x_e)
        start = end
    return dw1, dw3, dw2


def _standard_ep_hot_split_wgrad_tgmm(
    *,
    sorted_x: torch.Tensor,
    hidden: torch.Tensor,
    dgate: torch.Tensor,
    dup: torch.Tensor,
    grad_y: torch.Tensor,
    offsets: torch.Tensor,
    w1_hot: torch.Tensor,
    w3_hot: torch.Tensor,
    w2_hot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ptgmm = _standard_ep_hot_split_tgmm_op()
    group_sizes = offsets.to(torch.int32).contiguous()
    if group_sizes.numel() > 1:
        group_sizes = torch.cat(
            [group_sizes[:1], group_sizes[1:] - group_sizes[:-1]]
        ).contiguous()
    dw2 = ptgmm(
        grad_y.transpose(0, 1),
        hidden,
        group_sizes,
        preferred_element_type=sorted_x.dtype,
        existing_out=torch.zeros_like(w2_hot),
    )
    dw1 = ptgmm(
        dgate.transpose(0, 1),
        sorted_x,
        group_sizes,
        preferred_element_type=sorted_x.dtype,
        existing_out=torch.zeros_like(w1_hot),
    )
    dw3 = ptgmm(
        dup.transpose(0, 1),
        sorted_x,
        group_sizes,
        preferred_element_type=sorted_x.dtype,
        existing_out=torch.zeros_like(w3_hot),
    )
    return dw1, dw3, dw2


class _StandardEPHotSplitGroupedMLP(torch.autograd.Function):
    """Opt-in single-owner autograd node for helper-side grouped MoE rows."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        sorted_x: torch.Tensor,
        w1_hot: torch.Tensor,
        w3_hot: torch.Tensor,
        w2_hot: torch.Tensor,
        offsets: torch.Tensor,
        backward_mode: str,
    ) -> torch.Tensor:
        gate = torch._grouped_mm(
            sorted_x.bfloat16(),
            w1_hot.bfloat16().transpose(-2, -1),
            offs=offsets,
        )
        up = torch._grouped_mm(
            sorted_x.bfloat16(),
            w3_hot.bfloat16().transpose(-2, -1),
            offs=offsets,
        )
        hidden = F.silu(gate) * up
        out = torch._grouped_mm(
            hidden,
            w2_hot.bfloat16().transpose(-2, -1),
            offs=offsets,
        ).type_as(sorted_x)
        ctx.backward_mode = str(backward_mode)
        ctx.save_for_backward(sorted_x, w1_hot, w3_hot, w2_hot, offsets, gate, up, hidden)
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        sorted_x, w1_hot, w3_hot, w2_hot, offsets, gate, up, hidden = ctx.saved_tensors
        mode = str(ctx.backward_mode)
        timing_records = (
            []
            if (
                dsv4_profile_timing_enabled()
                and _env_flag(
                    "TORCHTITAN_STANDARD_EP_HOT_SPLIT_MANUAL_VJP_DETAIL",
                    _env_flag("CANARY_STANDARD_EP_HOT_SPLIT_MANUAL_VJP_DETAIL", False),
                )
            )
            else None
        )
        grad_y = grad_output.contiguous()
        with dsv4_timed_stage(timing_records, "standard_ep.hot_split.manual_vjp.w2_dgrad"):
            dhidden = torch._grouped_mm(
                grad_y.bfloat16(),
                w2_hot.bfloat16(),
                offs=offsets,
            )
        with dsv4_timed_stage(timing_records, "standard_ep.hot_split.manual_vjp.swiglu_backward"):
            dgate, dup = _standard_ep_swiglu_backward(dhidden, gate, up)
        with dsv4_timed_stage(timing_records, "standard_ep.hot_split.manual_vjp.xgrad"):
            dx = torch._grouped_mm(
                dgate.bfloat16(),
                w1_hot.bfloat16(),
                offs=offsets,
            )
            dx = dx + torch._grouped_mm(
                dup.bfloat16(),
                w3_hot.bfloat16(),
                offs=offsets,
            )
        with dsv4_timed_stage(timing_records, "standard_ep.hot_split.manual_vjp.wgrad"):
            if mode == "manual_tgmm":
                dw1, dw3, dw2 = _standard_ep_hot_split_wgrad_tgmm(
                    sorted_x=sorted_x,
                    hidden=hidden,
                    dgate=dgate,
                    dup=dup,
                    grad_y=grad_y,
                    offsets=offsets,
                    w1_hot=w1_hot,
                    w3_hot=w3_hot,
                    w2_hot=w2_hot,
                )
            else:
                dw1, dw3, dw2 = _standard_ep_hot_split_wgrad_loop(
                    sorted_x=sorted_x,
                    hidden=hidden,
                    dgate=dgate,
                    dup=dup,
                    grad_y=grad_y,
                    offsets=offsets,
                    w1_hot=w1_hot,
                    w3_hot=w3_hot,
                    w2_hot=w2_hot,
                )
        flush_dsv4_profile_timing(
            timing_records,
            {
                "kind": "standard_ep_hot_split_manual_vjp",
                "phase": "backward",
                "backward_mode": mode,
                "sorted_x_shape": list(sorted_x.shape),
                "num_hot_groups": int(w1_hot.shape[0]),
            },
        )
        return (
            dx.to(sorted_x.dtype),
            dw1.to(w1_hot.dtype),
            dw3.to(w3_hot.dtype),
            dw2.to(w2_hot.dtype),
            None,
            None,
        )


def _standard_ep_hot_to_owner_compact_order(
    *,
    selected_global_experts: tuple[int, ...],
    num_local_experts: int,
    ep_size: int,
    max_owned_per_rank: int,
    owner_shard_active_offsets: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    owner_counts = [0 for _ in range(int(ep_size))]
    offset_to_compact = {
        int(offset): int(compact)
        for compact, offset in enumerate(owner_shard_active_offsets)
    }
    compact_to_hot = [0 for _ in range(len(owner_shard_active_offsets))]
    for hot_offset, expert in enumerate(selected_global_experts):
        owner_rank = int(expert) // int(num_local_experts)
        owner_slot = owner_counts[owner_rank]
        owner_counts[owner_rank] += 1
        owner_shard_offset = int(owner_rank) * int(max_owned_per_rank) + int(owner_slot)
        compact_offset = offset_to_compact[int(owner_shard_offset)]
        compact_to_hot[compact_offset] = int(hot_offset)
    return torch.tensor(compact_to_hot, device=device, dtype=torch.long)


def _standard_ep_owner_compact_global_experts(
    *,
    owner_compact_owner_ranks: tuple[int, ...],
    owner_compact_local_experts: tuple[int, ...],
    num_local_experts: int,
    selected_global_experts: tuple[int, ...],
) -> tuple[int, ...]:
    if len(owner_compact_owner_ranks) != len(owner_compact_local_experts):
        raise ValueError(
            "owner_compact_owner_ranks and owner_compact_local_experts must "
            "have the same length."
        )
    compact_experts = tuple(
        int(owner_rank) * int(num_local_experts) + int(local_expert)
        for owner_rank, local_expert in zip(
            owner_compact_owner_ranks,
            owner_compact_local_experts,
        )
    )
    if sorted(compact_experts) != sorted(int(e) for e in selected_global_experts):
        raise ValueError(
            "Owner-compact hot-weight order does not match selected hot experts."
        )
    return compact_experts


def _standard_ep_hot_weight_all_reduce_(
    tensor: torch.Tensor,
    *,
    ep_rank: int,
    ep_group: dist.ProcessGroup,
    backend: str,
) -> None:
    if backend == "torch":
        dist.all_reduce(tensor, group=ep_group)
        return
    from .mori_aiter_moe import _mori_hot_weight_all_reduce_

    _mori_hot_weight_all_reduce_(
        tensor,
        ep_rank=ep_rank,
        ep_group=ep_group,
        backend=backend,
    )


def _standard_ep_hot_weight_owner_gather(
    local_weight: torch.Tensor,
    selected_global_experts: tuple[int, ...],
    *,
    num_local_experts: int,
    ep_rank: int,
    ep_group: dist.ProcessGroup,
    timing_records: list[dict[str, Any]] | None = None,
    stage_prefix: str = "standard_ep_hot_weight.forward.owner_gather_hot_weight",
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    world_size = dist.get_world_size(ep_group)
    num_hot = len(selected_global_experts)
    with dsv4_timed_stage(timing_records, f"{stage_prefix}.plan_owner_slots"):
        owner_slots: list[tuple[int, int]] = []
        owner_counts = [0 for _ in range(world_size)]
        for expert in selected_global_experts:
            owner_rank = int(expert) // int(num_local_experts)
            if owner_rank < 0 or owner_rank >= world_size:
                raise ValueError(
                    f"Hot expert {expert} maps to owner rank {owner_rank}, "
                    f"outside EP world size {world_size}."
                )
            owner_slot = owner_counts[owner_rank]
            owner_counts[owner_rank] += 1
            owner_slots.append((owner_rank, owner_slot))

    max_owned = max(owner_counts, default=0)
    padded_hot_rows = world_size * max_owned
    max_pad_factor = _env_float(
        "TORCHTITAN_STANDARD_EP_HOT_WEIGHT_GATHER_MAX_PAD_FACTOR",
        _env_float("CANARY_STANDARD_EP_HOT_WEIGHT_GATHER_MAX_PAD_FACTOR", 2.0),
    )
    metadata: dict[str, Any] = {
        "owner_counts": owner_counts,
        "max_owned_per_rank": int(max_owned),
        "padded_hot_rows": int(padded_hot_rows),
        "pad_factor": (
            float(padded_hot_rows) / float(num_hot) if num_hot > 0 else 0.0
        ),
        "max_pad_factor": float(max_pad_factor),
    }
    if num_hot == 0:
        return local_weight.new_empty((0, *local_weight.shape[1:])), metadata
    if max_owned == 0 or padded_hot_rows > num_hot * max_pad_factor:
        metadata["fallback_reason"] = "owner_gather_padding_exceeds_threshold"
        return None, metadata

    with dsv4_timed_stage(timing_records, f"{stage_prefix}.allocate_local_shard"):
        local_shard = local_weight.new_zeros((max_owned, *local_weight.shape[1:]))
    with dsv4_timed_stage(timing_records, f"{stage_prefix}.pack_owner_shard"):
        for hot_offset, expert in enumerate(selected_global_experts):
            owner_rank, owner_slot = owner_slots[hot_offset]
            if owner_rank == ep_rank:
                local_shard[owner_slot].copy_(
                    local_weight[int(expert) % int(num_local_experts)]
                )

    with dsv4_timed_stage(timing_records, f"{stage_prefix}.allocate_gathered"):
        gathered = local_weight.new_empty(
            (padded_hot_rows, *local_weight.shape[1:])
        )
    with dsv4_timed_stage(timing_records, f"{stage_prefix}.all_gather_hot_weight"):
        dist.all_gather_into_tensor(gathered, local_shard.contiguous(), group=ep_group)

    with dsv4_timed_stage(timing_records, f"{stage_prefix}.unpack_hot_weight"):
        hot_weight = local_weight.new_empty((num_hot, *local_weight.shape[1:]))
        for hot_offset, (owner_rank, owner_slot) in enumerate(owner_slots):
            hot_weight[hot_offset].copy_(
                gathered[owner_rank * max_owned + owner_slot]
            )
    return hot_weight, metadata


class _StandardEPOwnerShardedHotWeightGather(torch.autograd.Function):
    """Gather selected hot weights in compact owner-sharded order.

    The all-gather payload is padded by owner rank, but the returned tensor is
    compacted to active ``owner_rank * max_owned_per_rank + owner_slot`` rows.
    This avoids dense selected-hot order and avoids empty grouped-mm groups.
    """

    @staticmethod
    def forward(
        ctx,
        local_weight: torch.Tensor,
        selected_global_experts: tuple[int, ...],
        num_local_experts: int,
        ep_rank: int,
        ep_group,
        max_owned_per_rank: int,
        owner_shard_active_offsets: tuple[int, ...],
        label: str,
    ) -> torch.Tensor:
        ctx.selected_global_experts = tuple(int(e) for e in selected_global_experts)
        ctx.num_local_experts = int(num_local_experts)
        ctx.ep_rank = int(ep_rank)
        ctx.ep_group = ep_group
        ctx.local_weight_shape = tuple(local_weight.shape)
        ctx.max_owned_per_rank = int(max_owned_per_rank)
        ctx.owner_shard_active_offsets = tuple(
            int(v) for v in owner_shard_active_offsets
        )
        ctx.label = str(label)
        ctx.reduce_backend = _standard_ep_hot_weight_reduce_backend()

        world_size = dist.get_world_size(ep_group)
        timing_records = [] if dsv4_profile_timing_enabled() else None
        with dsv4_timed_stage(
            timing_records,
            f"standard_ep_hot_weight.{ctx.label}.forward.owner_shard.allocate",
        ):
            local_shard = local_weight.new_zeros(
                (ctx.max_owned_per_rank, *local_weight.shape[1:])
            )
        with dsv4_timed_stage(
            timing_records,
            f"standard_ep_hot_weight.{ctx.label}.forward.owner_shard.pack",
        ):
            owner_counts = [0 for _ in range(world_size)]
            for expert in ctx.selected_global_experts:
                owner_rank = int(expert) // ctx.num_local_experts
                owner_slot = owner_counts[owner_rank]
                owner_counts[owner_rank] += 1
                if owner_rank == ctx.ep_rank:
                    local_shard[owner_slot].copy_(
                        local_weight[int(expert) % ctx.num_local_experts]
                    )
            ctx.owner_counts = tuple(int(v) for v in owner_counts)

        with dsv4_timed_stage(
            timing_records,
            f"standard_ep_hot_weight.{ctx.label}.forward.owner_shard.allocate_gathered",
        ):
            gathered = local_weight.new_empty(
                (world_size * ctx.max_owned_per_rank, *local_weight.shape[1:])
            )
        with dsv4_timed_stage(
            timing_records,
            f"standard_ep_hot_weight.{ctx.label}.forward.owner_shard.all_gather",
        ):
            dist.all_gather_into_tensor(gathered, local_shard.contiguous(), group=ep_group)
        with dsv4_timed_stage(
            timing_records,
            f"standard_ep_hot_weight.{ctx.label}.forward.owner_shard.compact_active",
        ):
            active_offsets = torch.tensor(
                ctx.owner_shard_active_offsets,
                device=gathered.device,
                dtype=torch.long,
            )
            owner_weight = gathered.index_select(0, active_offsets)
        flush_dsv4_profile_timing(
            timing_records,
            {
                "kind": "standard_ep_owner_sharded_hot_weight",
                "phase": "forward",
                "label": ctx.label,
                "selected_global_experts": list(ctx.selected_global_experts),
                "num_selected_hot_experts": len(ctx.selected_global_experts),
                "num_local_experts": ctx.num_local_experts,
                "ep_rank": ctx.ep_rank,
                "owner_counts": list(ctx.owner_counts),
                "max_owned_per_rank": ctx.max_owned_per_rank,
                "padded_hot_rows": int(world_size * ctx.max_owned_per_rank),
                "owner_shard_active_offsets": list(ctx.owner_shard_active_offsets),
                "local_weight_shape": list(local_weight.shape),
                "owner_sharded_gather_shape": list(gathered.shape),
                "owner_sharded_weight_shape": list(owner_weight.shape),
                "hot_weight_reduce_backend": ctx.reduce_backend,
            },
        )
        return owner_weight

    @staticmethod
    def backward(ctx, grad_owner_weight: torch.Tensor):
        grad_owner_weight = grad_owner_weight.contiguous()
        grad_local = grad_owner_weight.new_zeros(ctx.local_weight_shape)
        timing_records = [] if dsv4_profile_timing_enabled() else None
        with dsv4_timed_stage(
            timing_records,
            f"standard_ep_hot_weight.{ctx.label}.backward.owner_shard.all_reduce_grad",
        ):
            _standard_ep_hot_weight_all_reduce_(
                grad_owner_weight,
                ep_rank=ctx.ep_rank,
                ep_group=ctx.ep_group,
                backend=ctx.reduce_backend,
            )
        with dsv4_timed_stage(
            timing_records,
            f"standard_ep_hot_weight.{ctx.label}.backward.owner_shard.scatter_owner_grad",
        ):
            owner_counts = [0 for _ in range(dist.get_world_size(ctx.ep_group))]
            active_offset_to_compact = {
                int(offset): int(compact_idx)
                for compact_idx, offset in enumerate(ctx.owner_shard_active_offsets)
            }
            for expert in ctx.selected_global_experts:
                owner_rank = int(expert) // ctx.num_local_experts
                owner_slot = owner_counts[owner_rank]
                owner_counts[owner_rank] += 1
                if owner_rank == ctx.ep_rank:
                    owner_shard_offset = (
                        int(owner_rank) * int(ctx.max_owned_per_rank)
                        + int(owner_slot)
                    )
                    compact_idx = active_offset_to_compact[owner_shard_offset]
                    grad_local[int(expert) % ctx.num_local_experts].add_(
                        grad_owner_weight[compact_idx]
                    )
        flush_dsv4_profile_timing(
            timing_records,
            {
                "kind": "standard_ep_owner_sharded_hot_weight",
                "phase": "backward",
                "label": ctx.label,
                "selected_global_experts": list(ctx.selected_global_experts),
                "num_selected_hot_experts": len(ctx.selected_global_experts),
                "num_local_experts": ctx.num_local_experts,
                "ep_rank": ctx.ep_rank,
                "owner_counts": list(getattr(ctx, "owner_counts", ())),
                "max_owned_per_rank": ctx.max_owned_per_rank,
                "owner_shard_active_offsets": list(ctx.owner_shard_active_offsets),
                "local_weight_shape": list(ctx.local_weight_shape),
                "grad_owner_sharded_weight_shape": list(grad_owner_weight.shape),
                "hot_weight_reduce_backend": ctx.reduce_backend,
            },
        )
        return grad_local, None, None, None, None, None, None, None


class _StandardEPNeededHotWeightExchange(torch.autograd.Function):
    """Exchange only compact owner hot weights needed by this helper rank."""

    @staticmethod
    def _compact_owner_maps(
        selected_global_experts: tuple[int, ...],
        *,
        num_local_experts: int,
        ep_size: int,
        max_owned_per_rank: int,
        owner_shard_active_offsets: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        active_to_compact = {
            int(offset): int(compact_idx)
            for compact_idx, offset in enumerate(owner_shard_active_offsets)
        }
        compact_owner_ranks = [0 for _ in owner_shard_active_offsets]
        compact_local_experts = [0 for _ in owner_shard_active_offsets]
        owner_counts = [0 for _ in range(int(ep_size))]
        for expert in selected_global_experts:
            owner_rank = int(expert) // int(num_local_experts)
            owner_slot = owner_counts[owner_rank]
            owner_counts[owner_rank] += 1
            owner_offset = int(owner_rank) * int(max_owned_per_rank) + int(owner_slot)
            compact_idx = active_to_compact.get(owner_offset)
            if compact_idx is None:
                continue
            compact_owner_ranks[compact_idx] = int(owner_rank)
            compact_local_experts[compact_idx] = (
                int(expert) % int(num_local_experts)
            )
        return tuple(compact_owner_ranks), tuple(compact_local_experts)

    @staticmethod
    def forward(
        ctx,
        local_weight: torch.Tensor,
        selected_global_experts: tuple[int, ...],
        num_local_experts: int,
        ep_rank: int,
        ep_group,
        max_owned_per_rank: int,
        owner_shard_active_offsets: tuple[int, ...],
        needed_owner_compact_offsets: tuple[int, ...],
        compact_owner_ranks: tuple[int, ...],
        compact_local_experts: tuple[int, ...],
        need_mask: tuple[bool, ...],
        need_masks: tuple[tuple[bool, ...], ...],
        send_owner_compact_offsets_by_rank: tuple[tuple[int, ...], ...],
        recv_owner_compact_offsets_by_rank: tuple[tuple[int, ...], ...],
        send_owner_compact_offsets: tuple[int, ...],
        recv_owner_compact_offsets: tuple[int, ...],
        recv_to_needed_index: tuple[int, ...],
        grad_send_needed_indices: tuple[int, ...],
        grad_recv_owner_compact_offsets: tuple[int, ...],
        label: str,
    ) -> torch.Tensor:
        ctx.selected_global_experts = tuple(int(e) for e in selected_global_experts)
        ctx.num_local_experts = int(num_local_experts)
        ctx.ep_rank = int(ep_rank)
        ctx.ep_group = ep_group
        ctx.local_weight_shape = tuple(local_weight.shape)
        ctx.max_owned_per_rank = int(max_owned_per_rank)
        ctx.owner_shard_active_offsets = tuple(
            int(v) for v in owner_shard_active_offsets
        )
        ctx.needed_owner_compact_offsets = tuple(
            int(v) for v in needed_owner_compact_offsets
        )
        ctx.label = str(label)
        ctx.reduce_backend = _standard_ep_hot_weight_reduce_backend()

        world_size = dist.get_world_size(ep_group)
        active_count = len(ctx.owner_shard_active_offsets)
        ctx.compact_owner_ranks = tuple(int(v) for v in compact_owner_ranks)
        ctx.compact_local_experts = tuple(int(v) for v in compact_local_experts)
        ctx.need_mask = tuple(bool(v) for v in need_mask)
        ctx.need_masks = tuple(tuple(bool(v) for v in row) for row in need_masks)
        ctx.send_owner_compact_offsets_by_rank = tuple(
            tuple(int(v) for v in row)
            for row in send_owner_compact_offsets_by_rank
        )
        ctx.recv_owner_compact_offsets_by_rank = tuple(
            tuple(int(v) for v in row)
            for row in recv_owner_compact_offsets_by_rank
        )
        ctx.send_owner_compact_offsets = tuple(
            int(v) for v in send_owner_compact_offsets
        )
        ctx.recv_owner_compact_offsets = tuple(
            int(v) for v in recv_owner_compact_offsets
        )
        ctx.recv_to_needed_index = tuple(int(v) for v in recv_to_needed_index)
        ctx.grad_send_needed_indices = tuple(
            int(v) for v in grad_send_needed_indices
        )
        ctx.grad_recv_owner_compact_offsets = tuple(
            int(v) for v in grad_recv_owner_compact_offsets
        )
        if len(ctx.compact_owner_ranks) != active_count:
            raise ValueError(
                "compact_owner_ranks must match owner_shard_active_offsets"
            )
        if len(ctx.compact_local_experts) != active_count:
            raise ValueError(
                "compact_local_experts must match owner_shard_active_offsets"
            )
        if len(ctx.need_mask) != active_count:
            raise ValueError("need_mask must match owner_shard_active_offsets")
        if len(ctx.need_masks) != world_size or any(
            len(row) != active_count for row in ctx.need_masks
        ):
            raise ValueError(
                "need_masks must be shaped [ep_world_size, active_owner_compact_count]"
            )
        if len(ctx.send_owner_compact_offsets_by_rank) != world_size:
            raise ValueError("send_owner_compact_offsets_by_rank must be per-rank")
        if len(ctx.recv_owner_compact_offsets_by_rank) != world_size:
            raise ValueError("recv_owner_compact_offsets_by_rank must be per-rank")
        if ctx.send_owner_compact_offsets and len(
            ctx.send_owner_compact_offsets
        ) != sum(len(row) for row in ctx.send_owner_compact_offsets_by_rank):
            raise ValueError("send_owner_compact_offsets must match input splits")
        if ctx.recv_owner_compact_offsets and len(
            ctx.recv_owner_compact_offsets
        ) != sum(len(row) for row in ctx.recv_owner_compact_offsets_by_rank):
            raise ValueError("recv_owner_compact_offsets must match output splits")
        if ctx.recv_to_needed_index and len(ctx.recv_to_needed_index) != sum(
            len(row) for row in ctx.recv_owner_compact_offsets_by_rank
        ):
            raise ValueError("recv_to_needed_index must match output rows")
        if ctx.grad_send_needed_indices and len(
            ctx.grad_send_needed_indices
        ) != sum(len(row) for row in ctx.recv_owner_compact_offsets_by_rank):
            raise ValueError("grad_send_needed_indices must match backward input rows")
        if ctx.grad_recv_owner_compact_offsets and len(
            ctx.grad_recv_owner_compact_offsets
        ) != sum(len(row) for row in ctx.send_owner_compact_offsets_by_rank):
            raise ValueError(
                "grad_recv_owner_compact_offsets must match backward output rows"
            )

        timing_records = [] if dsv4_profile_timing_enabled() else None
        with dsv4_timed_stage(
            timing_records,
            f"standard_ep_hot_weight.{ctx.label}.forward.needed.use_cached_plan",
        ):
            compact_owner_ranks = ctx.compact_owner_ranks
            compact_local_experts = ctx.compact_local_experts
            send_by_rank = ctx.send_owner_compact_offsets_by_rank
            recv_by_rank = ctx.recv_owner_compact_offsets_by_rank

        with dsv4_timed_stage(
            timing_records,
            f"standard_ep_hot_weight.{ctx.label}.forward.needed.pack_send",
        ):
            input_splits = [len(row) for row in send_by_rank]
            output_splits = [len(row) for row in recv_by_rank]
            if ctx.send_owner_compact_offsets:
                send_compact_offsets = torch.tensor(
                    ctx.send_owner_compact_offsets,
                    device=local_weight.device,
                    dtype=torch.long,
                )
                compact_to_local = torch.tensor(
                    compact_local_experts,
                    device=local_weight.device,
                    dtype=torch.long,
                )
                send_local_experts = compact_to_local.index_select(
                    0, send_compact_offsets
                )
                send = local_weight.index_select(0, send_local_experts).contiguous()
            else:
                send_rows: list[torch.Tensor] = []
                for offsets in send_by_rank:
                    for compact_idx in offsets:
                        send_rows.append(
                            local_weight[compact_local_experts[compact_idx]]
                        )
                if send_rows:
                    send = torch.stack(send_rows, dim=0).contiguous()
                else:
                    send = local_weight.new_empty((0, *local_weight.shape[1:]))
            recv = local_weight.new_empty((sum(output_splits), *local_weight.shape[1:]))

        with dsv4_timed_stage(
            timing_records,
            f"standard_ep_hot_weight.{ctx.label}.forward.needed.all_to_all",
        ):
            dist.all_to_all_single(
                recv,
                send,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=ep_group,
            )

        with dsv4_timed_stage(
            timing_records,
            f"standard_ep_hot_weight.{ctx.label}.forward.needed.unpack",
        ):
            needed_weight = local_weight.new_empty(
                (len(ctx.needed_owner_compact_offsets), *local_weight.shape[1:])
            )
            if ctx.recv_to_needed_index:
                recv_to_needed = torch.tensor(
                    ctx.recv_to_needed_index,
                    device=recv.device,
                    dtype=torch.long,
                )
                needed_weight.index_copy_(0, recv_to_needed, recv)
            else:
                needed_to_row = {
                    int(compact_idx): int(row_idx)
                    for row_idx, compact_idx in enumerate(
                        ctx.needed_owner_compact_offsets
                    )
                }
                cursor = 0
                for offsets in recv_by_rank:
                    for compact_idx in offsets:
                        needed_weight[needed_to_row[compact_idx]].copy_(recv[cursor])
                        cursor += 1

        flush_dsv4_profile_timing(
            timing_records,
            {
                "kind": "standard_ep_needed_hot_weight",
                "phase": "forward",
                "label": ctx.label,
                "selected_global_experts": list(ctx.selected_global_experts),
                "num_selected_hot_experts": len(ctx.selected_global_experts),
                "num_needed_owner_compact_offsets": len(
                    ctx.needed_owner_compact_offsets
                ),
                "num_local_experts": ctx.num_local_experts,
                "ep_rank": ctx.ep_rank,
                "max_owned_per_rank": ctx.max_owned_per_rank,
                "owner_shard_active_offsets": list(ctx.owner_shard_active_offsets),
                "compact_owner_ranks": list(compact_owner_ranks),
                "needed_owner_compact_offsets": list(
                    ctx.needed_owner_compact_offsets
                ),
                "needed_plan_cached": True,
                "needed_tensor_abi": bool(ctx.recv_to_needed_index),
                "input_splits": list(input_splits),
                "output_splits": list(output_splits),
                "local_weight_shape": list(local_weight.shape),
                "needed_weight_shape": list(needed_weight.shape),
                "hot_weight_reduce_backend": ctx.reduce_backend,
            },
        )
        return needed_weight

    @staticmethod
    def backward(ctx, grad_needed_weight: torch.Tensor):
        grad_needed_weight = grad_needed_weight.contiguous()
        world_size = dist.get_world_size(ctx.ep_group)
        compact_owner_ranks = tuple(int(v) for v in ctx.compact_owner_ranks)
        compact_local_experts = tuple(int(v) for v in ctx.compact_local_experts)
        send_by_rank = tuple(
            tuple(int(v) for v in row)
            for row in ctx.send_owner_compact_offsets_by_rank
        )
        recv_by_rank = tuple(
            tuple(int(v) for v in row)
            for row in ctx.recv_owner_compact_offsets_by_rank
        )
        timing_records = [] if dsv4_profile_timing_enabled() else None

        with dsv4_timed_stage(
            timing_records,
            f"standard_ep_hot_weight.{ctx.label}.backward.needed.pack_send",
        ):
            input_splits = [len(row) for row in recv_by_rank]
            output_splits = [len(row) for row in send_by_rank]
            if ctx.grad_send_needed_indices:
                grad_send_needed = torch.tensor(
                    ctx.grad_send_needed_indices,
                    device=grad_needed_weight.device,
                    dtype=torch.long,
                )
                send = grad_needed_weight.index_select(
                    0, grad_send_needed
                ).contiguous()
            else:
                needed_to_row = {
                    int(compact_idx): int(row_idx)
                    for row_idx, compact_idx in enumerate(
                        ctx.needed_owner_compact_offsets
                    )
                }
                send_rows: list[torch.Tensor] = []
                for offsets in recv_by_rank:
                    for compact_idx in offsets:
                        send_rows.append(
                            grad_needed_weight[needed_to_row[compact_idx]]
                        )
                if send_rows:
                    send = torch.stack(send_rows, dim=0).contiguous()
                else:
                    send = grad_needed_weight.new_empty(
                        (0, *grad_needed_weight.shape[1:])
                    )
            recv = grad_needed_weight.new_empty(
                (sum(output_splits), *grad_needed_weight.shape[1:])
            )

        with dsv4_timed_stage(
            timing_records,
            f"standard_ep_hot_weight.{ctx.label}.backward.needed.all_to_all_reduce",
        ):
            dist.all_to_all_single(
                recv,
                send,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=ctx.ep_group,
            )

        with dsv4_timed_stage(
            timing_records,
            f"standard_ep_hot_weight.{ctx.label}.backward.needed.scatter_owner_grad",
        ):
            grad_local = grad_needed_weight.new_zeros(ctx.local_weight_shape)
            if ctx.grad_recv_owner_compact_offsets:
                grad_recv_compact = torch.tensor(
                    ctx.grad_recv_owner_compact_offsets,
                    device=recv.device,
                    dtype=torch.long,
                )
                compact_to_local = torch.tensor(
                    compact_local_experts,
                    device=recv.device,
                    dtype=torch.long,
                )
                grad_local_indices = compact_to_local.index_select(
                    0, grad_recv_compact
                )
                grad_local.index_add_(0, grad_local_indices, recv)
            else:
                cursor = 0
                for offsets in send_by_rank:
                    for compact_idx in offsets:
                        grad_local[compact_local_experts[compact_idx]].add_(
                            recv[cursor]
                        )
                        cursor += 1

        flush_dsv4_profile_timing(
            timing_records,
            {
                "kind": "standard_ep_needed_hot_weight",
                "phase": "backward",
                "label": ctx.label,
                "selected_global_experts": list(ctx.selected_global_experts),
                "num_selected_hot_experts": len(ctx.selected_global_experts),
                "num_needed_owner_compact_offsets": len(
                    ctx.needed_owner_compact_offsets
                ),
                "num_local_experts": ctx.num_local_experts,
                "ep_rank": ctx.ep_rank,
                "max_owned_per_rank": ctx.max_owned_per_rank,
                "owner_shard_active_offsets": list(ctx.owner_shard_active_offsets),
                "compact_owner_ranks": list(compact_owner_ranks),
                "needed_owner_compact_offsets": list(
                    ctx.needed_owner_compact_offsets
                ),
                "needed_plan_cached": True,
                "needed_tensor_abi": bool(ctx.grad_send_needed_indices),
                "input_splits": list(input_splits),
                "output_splits": list(output_splits),
                "local_weight_shape": list(ctx.local_weight_shape),
                "grad_needed_weight_shape": list(grad_needed_weight.shape),
                "hot_weight_reduce_backend": ctx.reduce_backend,
            },
        )
        return (
            grad_local,
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


class _ProfiledLinear(torch.autograd.Function):
    """Profiling-only linear wrapper that names MoE/router backward matmuls."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias, label: str):
        ctx.has_bias = bias is not None
        ctx.label = str(label)
        if bias is None:
            ctx.save_for_backward(x, weight)
        else:
            ctx.save_for_backward(x, weight, bias)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        label = ctx.label
        timing_records = [] if dsv4_profile_timing_enabled() else None
        with torch.profiler.record_function(f"{label}.bwd"):
            saved = ctx.saved_tensors
            if ctx.has_bias:
                x, weight, _bias = saved
            else:
                x, weight = saved

            grad_x = grad_weight = grad_bias = None
            grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
            x_flat = x.reshape(-1, x.shape[-1])

            if ctx.needs_input_grad[0]:
                with torch.profiler.record_function(f"{label}.bwd.dinput"):
                    with dsv4_timed_stage(timing_records, f"{label}.bwd.dinput"):
                        grad_x = grad_output_flat.to(dtype=weight.dtype).matmul(weight)
                        grad_x = grad_x.reshape_as(x).to(x.dtype)
            if ctx.needs_input_grad[1]:
                with torch.profiler.record_function(f"{label}.bwd.dweight"):
                    with dsv4_timed_stage(timing_records, f"{label}.bwd.dweight"):
                        grad_weight = grad_output_flat.to(dtype=x.dtype).transpose(0, 1).matmul(x_flat)
                        grad_weight = grad_weight.reshape_as(weight).to(weight.dtype)
            if ctx.has_bias and ctx.needs_input_grad[2]:
                with torch.profiler.record_function(f"{label}.bwd.dbias"):
                    with dsv4_timed_stage(timing_records, f"{label}.bwd.dbias"):
                        grad_bias = grad_output_flat.to(dtype=saved[-1].dtype).sum(dim=0)
        flush_dsv4_profile_timing(
            timing_records,
            {
                "kind": "linear_backward",
                "label": label,
                "x_shape": list(x.shape),
                "weight_shape": list(weight.shape),
                "grad_output_shape": list(grad_output.shape),
                "has_bias": bool(ctx.has_bias),
            },
        )
        return grad_x, grad_weight, grad_bias, None


def _profiled_linear(module: Linear, x: torch.Tensor, label: str) -> torch.Tensor:
    if not _env_flag("TORCHTITAN_DSV4_PROFILE_DENSE_LINEAR_BACKWARD", default=False):
        return module(x)
    return _ProfiledLinear.apply(x, module.weight, module.bias, label)


class _StandardEPHotWeightBroadcast(torch.autograd.Function):
    """Replicate selected hot expert rows, then reduce their grads to owners."""

    @staticmethod
    def forward(
        ctx,
        local_weight: torch.Tensor,
        selected_global_experts: tuple[int, ...],
        num_local_experts: int,
        ep_rank: int,
        ep_group,
        label: str,
    ) -> torch.Tensor:
        ctx.selected_global_experts = tuple(int(e) for e in selected_global_experts)
        ctx.num_local_experts = int(num_local_experts)
        ctx.ep_rank = int(ep_rank)
        ctx.ep_group = ep_group
        ctx.local_weight_shape = tuple(local_weight.shape)
        ctx.label = str(label)
        ctx.reduce_backend = _standard_ep_hot_weight_reduce_backend()
        forward_backend = _standard_ep_hot_weight_forward_backend()
        actual_forward_backend = forward_backend
        gather_metadata: dict[str, Any] | None = None

        timing_records = [] if dsv4_profile_timing_enabled() else None
        if forward_backend == "torch_allgather":
            hot_weight, gather_metadata = _standard_ep_hot_weight_owner_gather(
                local_weight,
                ctx.selected_global_experts,
                num_local_experts=ctx.num_local_experts,
                ep_rank=ctx.ep_rank,
                ep_group=ep_group,
                timing_records=timing_records,
                stage_prefix=(
                    f"standard_ep_hot_weight.{ctx.label}"
                    ".forward.owner_gather_hot_weight"
                ),
            )
            if hot_weight is None:
                actual_forward_backend = "torch_allgather_fallback_torch_allreduce"
                hot_weight = local_weight.new_zeros(
                    (len(ctx.selected_global_experts), *local_weight.shape[1:])
                )
                with dsv4_timed_stage(timing_records, f"standard_ep_hot_weight.{ctx.label}.forward.copy_owner"):
                    for hot_offset, expert in enumerate(ctx.selected_global_experts):
                        owner_rank = int(expert) // ctx.num_local_experts
                        if owner_rank == ctx.ep_rank:
                            hot_weight[hot_offset].copy_(
                                local_weight[int(expert) % ctx.num_local_experts]
                            )
                with dsv4_timed_stage(timing_records, f"standard_ep_hot_weight.{ctx.label}.forward.all_reduce_hot_weight"):
                    _standard_ep_hot_weight_all_reduce_(
                        hot_weight,
                        ep_rank=ctx.ep_rank,
                        ep_group=ep_group,
                        backend="torch",
                    )
        else:
            hot_weight = local_weight.new_zeros(
                (len(ctx.selected_global_experts), *local_weight.shape[1:])
            )
            with dsv4_timed_stage(timing_records, f"standard_ep_hot_weight.{ctx.label}.forward.copy_owner"):
                for hot_offset, expert in enumerate(ctx.selected_global_experts):
                    owner_rank = int(expert) // ctx.num_local_experts
                    if owner_rank == ctx.ep_rank:
                        hot_weight[hot_offset].copy_(
                            local_weight[int(expert) % ctx.num_local_experts]
                        )
            with dsv4_timed_stage(timing_records, f"standard_ep_hot_weight.{ctx.label}.forward.all_reduce_hot_weight"):
                _standard_ep_hot_weight_all_reduce_(
                    hot_weight,
                    ep_rank=ctx.ep_rank,
                    ep_group=ep_group,
                    backend=forward_backend,
                )
        payload = {
            "kind": "standard_ep_hot_weight",
            "phase": "forward",
            "label": ctx.label,
            "selected_global_experts": list(ctx.selected_global_experts),
            "num_selected_hot_experts": len(ctx.selected_global_experts),
            "num_local_experts": ctx.num_local_experts,
            "ep_rank": ctx.ep_rank,
            "local_weight_shape": list(local_weight.shape),
            "hot_weight_shape": list(hot_weight.shape),
            "hot_weight_forward_backend": forward_backend,
            "hot_weight_forward_actual_backend": actual_forward_backend,
            "hot_weight_reduce_backend": ctx.reduce_backend,
        }
        if gather_metadata is not None:
            payload["owner_gather"] = gather_metadata
        flush_dsv4_profile_timing(
            timing_records,
            payload,
        )
        return hot_weight

    @staticmethod
    def backward(ctx, grad_hot_weight: torch.Tensor):
        grad_hot_weight = grad_hot_weight.contiguous()

        grad_local = grad_hot_weight.new_zeros(ctx.local_weight_shape)
        timing_records = [] if dsv4_profile_timing_enabled() else None
        with dsv4_timed_stage(timing_records, f"standard_ep_hot_weight.{ctx.label}.backward.all_reduce_grad_hot_weight"):
            _standard_ep_hot_weight_all_reduce_(
                grad_hot_weight,
                ep_rank=ctx.ep_rank,
                ep_group=ctx.ep_group,
                backend=ctx.reduce_backend,
            )
        with dsv4_timed_stage(timing_records, f"standard_ep_hot_weight.{ctx.label}.backward.scatter_owner_grad"):
            for hot_offset, expert in enumerate(ctx.selected_global_experts):
                owner_rank = int(expert) // ctx.num_local_experts
                if owner_rank == ctx.ep_rank:
                    grad_local[int(expert) % ctx.num_local_experts].add_(
                        grad_hot_weight[hot_offset]
                    )
        flush_dsv4_profile_timing(
            timing_records,
            {
                "kind": "standard_ep_hot_weight",
                "phase": "backward",
                "label": ctx.label,
                "selected_global_experts": list(ctx.selected_global_experts),
                "num_selected_hot_experts": len(ctx.selected_global_experts),
                "num_local_experts": ctx.num_local_experts,
                "ep_rank": ctx.ep_rank,
                "local_weight_shape": list(ctx.local_weight_shape),
                "grad_hot_weight_shape": list(grad_hot_weight.shape),
                "hot_weight_reduce_backend": ctx.reduce_backend,
            },
        )
        return grad_local, None, None, None, None, None


@dataclass(slots=True)
class MoEScheduleNodeState:
    """State threaded through explicit MoE schedule nodes.

    The default forward path still runs these nodes in the legacy order. The
    point is to expose the same callable boundaries a future 1F1B/A2A scheduler
    needs: dispatch, local expert compute, combine, and source-rank hot helper.
    """

    flat_x: torch.Tensor
    top_scores: torch.Tensor
    selected_experts_indices: torch.Tensor
    original_shape: tuple[Any, Any, Any]
    routed_input: torch.Tensor | None = None
    num_tokens_local: torch.Tensor | None = None
    metadata: Any | None = None
    routed_output: torch.Tensor | None = None


class GroupedExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int
        token_dispatcher: LocalTokenDispatcher.Config

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.w1 = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.w2 = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )
        self.w3 = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.token_dispatcher = config.token_dispatcher.build()

    def _experts_forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Raw expert computation without dispatch/combine."""
        if isinstance(self.w1, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1 = self.w1.to_local()
            # pyrefly: ignore [missing-attribute]
            w2 = self.w2.to_local()
            # pyrefly: ignore [missing-attribute]
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        backend = _standard_ep_local_expert_backend()
        if backend != "grouped_mm":
            from .mori_aiter_moe import standard_ep_aiter_local_experts_forward

            return standard_ep_aiter_local_experts_forward(
                x=x,
                num_tokens_per_expert=num_tokens_per_expert,
                w1_local=w1,
                w3_local=w3,
                w2_local=w2,
                backend=backend,
            )

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        with torch.profiler.record_function("moe.experts.w1_silu"):
            h = F.silu(
                torch._grouped_mm(
                    x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets
                )
            )
        with torch.profiler.record_function("moe.experts.w3_mul"):
            h = h * torch._grouped_mm(
                x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
            )
        with torch.profiler.record_function("moe.experts.w2"):
            return torch._grouped_mm(
                h, w2.bfloat16().transpose(-2, -1), offs=offsets
            ).type_as(x)

    def _standard_ep_hot_split_forward(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        metadata: StandardEPHotSplitMetadata,
    ) -> torch.Tensor:
        """Compute standard-EP remote-hot rows on their source rank."""
        ep_mesh = getattr(self.token_dispatcher, "ep_mesh", None)
        if ep_mesh is None or not metadata.selected_global_experts:
            return torch.zeros_like(x)

        if isinstance(self.w1, DTensor):
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()  # pyrefly: ignore [missing-attribute]
            w3 = self.w3.to_local()  # pyrefly: ignore [missing-attribute]
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        ep_group = ep_mesh.get_group()
        ep_rank = int(dist.get_rank(group=ep_group))
        num_local_experts = int(metadata.num_local_experts)
        timing_records = [] if dsv4_profile_timing_enabled() else None
        requested_weight_layout = _standard_ep_hot_split_weight_layout()
        effective_weight_layout = "selected"
        owner_sharded_pad_factor = 0.0
        padded_hot_rows = 0
        needed_owner_compact_offsets: tuple[int, ...] = ()
        needed_owner_compact_offsets_tensor: torch.Tensor | None = None
        needed_need_mask: tuple[bool, ...] = ()
        needed_need_masks: tuple[tuple[bool, ...], ...] = ()
        needed_send_owner_compact_offsets_by_rank: tuple[tuple[int, ...], ...] = ()
        needed_recv_owner_compact_offsets_by_rank: tuple[tuple[int, ...], ...] = ()
        needed_send_owner_compact_offsets: tuple[int, ...] = ()
        needed_recv_owner_compact_offsets: tuple[int, ...] = ()
        needed_recv_to_needed_index: tuple[int, ...] = ()
        needed_grad_send_needed_indices: tuple[int, ...] = ()
        needed_grad_recv_owner_compact_offsets: tuple[int, ...] = ()
        needed_compact_to_needed_index: tuple[int, ...] = ()
        needed_exchange_tensor_plan: dict[str, object] | None = None
        needed_exchange_runtime_tensor_plan: dict[str, object] | None = None
        owner_compact_local_experts_tensor: torch.Tensor | None = None
        compact_to_needed_index_tensor: torch.Tensor | None = None
        needed_local_count = 0
        needed_max_count = 0
        needed_max_density = _standard_ep_needed_weight_max_density()
        needed_fallback_reason = ""
        max_pad_factor = _env_float(
            "TORCHTITAN_STANDARD_EP_HOT_WEIGHT_GATHER_MAX_PAD_FACTOR",
            _env_float("CANARY_STANDARD_EP_HOT_WEIGHT_GATHER_MAX_PAD_FACTOR", 2.0),
        )
        if (
            requested_weight_layout == "needed"
            and metadata.remote_owner_compact_offsets is not None
            and metadata.owner_shard_active_offsets
            and int(metadata.max_owned_per_rank) > 0
        ):
            active_count = len(metadata.owner_shard_active_offsets)
            cached_need_masks = tuple(
                tuple(bool(v) for v in row)
                for row in getattr(metadata, "owner_compact_need_masks", ())
            )
            has_cached_need_masks = (
                len(cached_need_masks) == int(metadata.ep_size)
                and all(len(row) == active_count for row in cached_need_masks)
            )
            if has_cached_need_masks:
                with dsv4_timed_stage(
                    timing_records,
                    "standard_ep.hot_split.needed_weight_plan.use_mori_plan",
                ):
                    exchange_plan = getattr(
                        metadata, "owner_compact_exchange_plan", None
                    )
                    tensor_exchange_plan = getattr(
                        metadata, "owner_compact_exchange_tensor_plan", None
                    )
                    needed_need_masks = cached_need_masks
                    needed_need_mask = tuple(
                        bool(v) for v in needed_need_masks[int(ep_rank)]
                    )
                    if isinstance(exchange_plan, dict):
                        if isinstance(tensor_exchange_plan, dict):
                            needed_exchange_tensor_plan = tensor_exchange_plan
                        needed_owner_compact_offsets = tuple(
                            int(v)
                            for v in exchange_plan.get(
                                "needed_owner_compact_offsets", []
                            )
                        )
                        compact_to_needed_index = tuple(
                            int(v)
                            for v in exchange_plan.get(
                                "compact_to_needed_index", []
                            )
                        )
                        needed_compact_to_needed_index = compact_to_needed_index
                        needed_send_owner_compact_offsets_by_rank = tuple(
                            tuple(int(v) for v in row)
                            for row in exchange_plan.get(
                                "send_owner_compact_offsets_by_rank", []
                            )
                        )
                        needed_recv_owner_compact_offsets_by_rank = tuple(
                            tuple(int(v) for v in row)
                            for row in exchange_plan.get(
                                "recv_owner_compact_offsets_by_rank", []
                            )
                        )
                        needed_send_owner_compact_offsets = tuple(
                            int(v)
                            for v in exchange_plan.get(
                                "send_owner_compact_offsets", []
                            )
                        )
                        needed_recv_owner_compact_offsets = tuple(
                            int(v)
                            for v in exchange_plan.get(
                                "recv_owner_compact_offsets", []
                            )
                        )
                        needed_recv_to_needed_index = tuple(
                            int(v)
                            for v in exchange_plan.get("recv_to_needed_index", [])
                        )
                        needed_grad_send_needed_indices = tuple(
                            int(v)
                            for v in exchange_plan.get(
                                "grad_send_needed_indices", []
                            )
                        )
                        needed_grad_recv_owner_compact_offsets = tuple(
                            int(v)
                            for v in exchange_plan.get(
                                "grad_recv_owner_compact_offsets", []
                            )
                        )
                        needed_max_count = int(
                            exchange_plan.get("max_needed_count", 0)
                        )
                    else:
                        needed_owner_compact_offsets = tuple(
                            int(compact_idx)
                            for compact_idx, needed in enumerate(needed_need_mask)
                            if bool(needed)
                        )
                        needed_row_by_compact = {
                            int(compact_idx): int(row_idx)
                            for row_idx, compact_idx in enumerate(
                                needed_owner_compact_offsets
                            )
                        }
                        compact_to_needed_index = tuple(
                            needed_row_by_compact.get(int(compact_idx), -1)
                            for compact_idx in range(active_count)
                        )
                        needed_compact_to_needed_index = compact_to_needed_index
                    needed_owner_compact_offsets_tensor = torch.tensor(
                        needed_owner_compact_offsets,
                        device=metadata.remote_owner_compact_offsets.device,
                        dtype=torch.long,
                    )
                    if len(compact_to_needed_index) == active_count:
                        compact_to_needed_index_tensor = torch.tensor(
                            compact_to_needed_index,
                            device=metadata.remote_owner_compact_offsets.device,
                            dtype=torch.long,
                        )
                    needed_local_count = len(needed_owner_compact_offsets)
                    if needed_max_count <= 0:
                        needed_max_count = max(
                            (
                                sum(1 for needed in row if bool(needed))
                                for row in needed_need_masks
                            ),
                            default=0,
                        )
            else:
                if metadata.remote_owner_compact_offsets.numel() > 0:
                    needed_owner_compact_offsets_tensor = torch.unique(
                        metadata.remote_owner_compact_offsets.to(torch.long),
                        sorted=True,
                    )
                    needed_owner_compact_offsets = tuple(
                        int(v)
                        for v in needed_owner_compact_offsets_tensor.detach()
                        .cpu()
                        .tolist()
                    )
                with dsv4_timed_stage(
                    timing_records,
                    "standard_ep.hot_split.needed_weight_plan.build_need_mask",
                ):
                    need_mask_tensor = torch.zeros(
                        active_count,
                        device=metadata.remote_owner_compact_offsets.device,
                        dtype=torch.bool,
                    )
                    if needed_owner_compact_offsets_tensor is not None:
                        need_mask_tensor.index_fill_(
                            0,
                            needed_owner_compact_offsets_tensor.to(torch.long),
                            True,
                        )
                    needed_local_count = int(need_mask_tensor.sum().item())
                with dsv4_timed_stage(
                    timing_records,
                    "standard_ep.hot_split.needed_weight_plan.max_count_all_reduce",
                ):
                    max_needed_tensor = torch.tensor(
                        [needed_local_count],
                        device=need_mask_tensor.device,
                        dtype=torch.int32,
                    )
                    dist.all_reduce(
                        max_needed_tensor,
                        op=dist.ReduceOp.MAX,
                        group=ep_group,
                    )
                    needed_max_count = int(max_needed_tensor.item())
                if (
                    active_count > 0
                    and float(needed_max_count) / float(active_count)
                    <= float(needed_max_density)
                ):
                    with dsv4_timed_stage(
                        timing_records,
                        "standard_ep.hot_split.needed_weight_plan.all_gather_need_mask",
                    ):
                        gathered_need_masks = [
                            torch.empty_like(need_mask_tensor)
                            for _ in range(dist.get_world_size(ep_group))
                        ]
                        dist.all_gather(
                            gathered_need_masks, need_mask_tensor, group=ep_group
                        )
                        need_masks_tensor = torch.stack(
                            gathered_need_masks, dim=0
                        ).to(torch.bool)
                    with dsv4_timed_stage(
                        timing_records,
                        "standard_ep.hot_split.needed_weight_plan.d2h",
                    ):
                        needed_need_mask = tuple(
                            bool(v) for v in need_mask_tensor.detach().cpu().tolist()
                        )
                        needed_need_masks = tuple(
                            tuple(bool(v) for v in row)
                            for row in need_masks_tensor.detach().cpu().tolist()
                        )
                    effective_weight_layout = "needed"
                else:
                    needed_fallback_reason = "needed_density_exceeds_threshold"
                    needed_owner_compact_offsets = ()
                    needed_owner_compact_offsets_tensor = None
                if effective_weight_layout == "needed":
                    needed_row_by_compact = {
                        int(compact_idx): int(row_idx)
                        for row_idx, compact_idx in enumerate(
                            needed_owner_compact_offsets
                        )
                    }
                    needed_compact_to_needed_index = tuple(
                        needed_row_by_compact.get(int(compact_idx), -1)
                        for compact_idx in range(active_count)
                    )
            if (
                has_cached_need_masks
                and active_count > 0
                and float(needed_max_count) / float(active_count)
                <= float(needed_max_density)
            ):
                effective_weight_layout = "needed"
            elif has_cached_need_masks:
                needed_fallback_reason = "needed_density_exceeds_threshold"
                needed_owner_compact_offsets = ()
                needed_owner_compact_offsets_tensor = None
            if effective_weight_layout != "needed":
                needed_owner_compact_offsets = ()
                needed_owner_compact_offsets_tensor = None
                needed_compact_to_needed_index = ()
                compact_to_needed_index_tensor = None
            elif (
                not needed_send_owner_compact_offsets_by_rank
                or not needed_recv_owner_compact_offsets_by_rank
            ):
                compact_owner_ranks_for_plan = tuple(
                    int(v) for v in metadata.owner_compact_owner_ranks
                )
                needed_send_owner_compact_offsets_by_rank = tuple(
                    tuple(
                        compact_idx
                        for compact_idx, owner_rank in enumerate(
                            compact_owner_ranks_for_plan
                        )
                        if int(owner_rank) == int(ep_rank)
                        and bool(needed_need_masks[dst][compact_idx])
                    )
                    for dst in range(int(metadata.ep_size))
                )
                needed_recv_owner_compact_offsets_by_rank = tuple(
                    tuple(
                        compact_idx
                        for compact_idx, owner_rank in enumerate(
                            compact_owner_ranks_for_plan
                        )
                        if int(owner_rank) == src
                        and bool(needed_need_mask[compact_idx])
                    )
                    for src in range(int(metadata.ep_size))
                )
            if not needed_send_owner_compact_offsets:
                needed_send_owner_compact_offsets = tuple(
                    int(v)
                    for row in needed_send_owner_compact_offsets_by_rank
                    for v in row
                )
            if not needed_recv_owner_compact_offsets:
                needed_recv_owner_compact_offsets = tuple(
                    int(v)
                    for row in needed_recv_owner_compact_offsets_by_rank
                    for v in row
                )
            if not needed_recv_to_needed_index and needed_compact_to_needed_index:
                needed_recv_to_needed_index = tuple(
                    int(needed_compact_to_needed_index[int(v)])
                    for v in needed_recv_owner_compact_offsets
                )
            if not needed_grad_send_needed_indices:
                if needed_compact_to_needed_index:
                    needed_grad_send_needed_indices = tuple(
                        int(needed_compact_to_needed_index[int(v)])
                        for v in needed_recv_owner_compact_offsets
                    )
                else:
                    needed_row_by_compact = {
                        int(compact_idx): int(row_idx)
                        for row_idx, compact_idx in enumerate(
                            needed_owner_compact_offsets
                        )
                    }
                    needed_grad_send_needed_indices = tuple(
                        int(needed_row_by_compact[int(v)])
                        for v in needed_recv_owner_compact_offsets
                    )
            if not needed_grad_recv_owner_compact_offsets:
                needed_grad_recv_owner_compact_offsets = (
                    needed_send_owner_compact_offsets
                )
        if (
            requested_weight_layout == "owner_sharded"
            and metadata.remote_owner_compact_offsets is not None
            and metadata.owner_shard_active_offsets
            and int(metadata.max_owned_per_rank) > 0
        ):
            padded_hot_rows = int(metadata.ep_size) * int(metadata.max_owned_per_rank)
            selected_hot_rows = max(1, len(metadata.selected_global_experts))
            owner_sharded_pad_factor = (
                float(padded_hot_rows) / float(selected_hot_rows)
            )
            if owner_sharded_pad_factor <= float(max_pad_factor):
                effective_weight_layout = "owner_sharded"

        if effective_weight_layout == "needed":
            hot_split_row_group_layout = "needed_owner_compact"
        elif (
            metadata.remote_owner_compact_offsets is not None
            and metadata.owner_shard_active_offsets
            and int(metadata.max_owned_per_rank) > 0
            and (
                effective_weight_layout == "owner_sharded"
                or metadata.remote_offsets_presorted_by == "owner_compact"
                or metadata.remote_rows_materialized_by == "owner_compact"
            )
        ):
            hot_split_row_group_layout = "owner_compact"
        else:
            hot_split_row_group_layout = "selected"

        requested_weight_order = _standard_ep_hot_split_weight_order()
        hot_weight_order_layout = "selected"
        hot_weight_order_fallback_reason = ""
        hot_weight_global_experts = metadata.selected_global_experts
        if requested_weight_order == "owner_compact":
            if effective_weight_layout != "selected":
                hot_weight_order_fallback_reason = (
                    f"weight_layout_{effective_weight_layout}"
                )
            elif hot_split_row_group_layout != "owner_compact":
                hot_weight_order_fallback_reason = (
                    f"row_group_layout_{hot_split_row_group_layout}"
                )
            elif (
                not metadata.owner_compact_owner_ranks
                or not metadata.owner_compact_local_experts
            ):
                hot_weight_order_fallback_reason = "missing_owner_compact_metadata"
            else:
                hot_weight_global_experts = _standard_ep_owner_compact_global_experts(
                    owner_compact_owner_ranks=tuple(
                        int(v) for v in metadata.owner_compact_owner_ranks
                    ),
                    owner_compact_local_experts=tuple(
                        int(v) for v in metadata.owner_compact_local_experts
                    ),
                    num_local_experts=num_local_experts,
                    selected_global_experts=metadata.selected_global_experts,
                )
                hot_weight_order_layout = "owner_compact"

        def _hot_weight(local_weight: torch.Tensor, label: str) -> torch.Tensor:
            return _StandardEPHotWeightBroadcast.apply(
                local_weight,
                hot_weight_global_experts,
                num_local_experts,
                ep_rank,
                ep_group,
                label,
            )

        def _owner_sharded_hot_weight(
            local_weight: torch.Tensor, label: str
        ) -> torch.Tensor:
            return _StandardEPOwnerShardedHotWeightGather.apply(
                local_weight,
                metadata.selected_global_experts,
                num_local_experts,
                ep_rank,
                ep_group,
                int(metadata.max_owned_per_rank),
                tuple(int(v) for v in metadata.owner_shard_active_offsets),
                label,
            )

        needed_exchange_backend_name = ""
        needed_exchange_backend_module = None
        needed_exchange_fn = None
        needed_prepare_runtime_plan_fn = None
        needed_exchange_impl = _standard_ep_needed_weight_exchange_impl()
        preferred_balanced_moe_backend = getattr(
            metadata,
            "balanced_moe_backend_module",
            None,
        )
        if requested_weight_layout == "needed" and needed_exchange_impl != "local":
            for module_name, balanced_moe_backend_module in (
                _iter_standard_ep_balanced_moe_modules(
                    str(preferred_balanced_moe_backend)
                    if preferred_balanced_moe_backend is not None
                    else None
                )
            ):
                if not _standard_ep_balanced_moe_module_matches_impl(
                    module_name,
                    balanced_moe_backend_module,
                    needed_exchange_impl,
                ):
                    continue
                exchange_owner_compact_needed_rows = getattr(
                    balanced_moe_backend_module,
                    "exchange_owner_compact_needed_rows",
                    None,
                )
                if exchange_owner_compact_needed_rows is None:
                    continue
                needed_exchange_backend_name = module_name
                needed_exchange_backend_module = balanced_moe_backend_module
                needed_exchange_fn = exchange_owner_compact_needed_rows
                needed_prepare_runtime_plan_fn = getattr(
                    balanced_moe_backend_module,
                    "prepare_owner_compact_needed_rows_runtime_plan",
                    None,
                )
                break

        def _needed_hot_weight(
            local_weight: torch.Tensor, label: str
        ) -> torch.Tensor:
            nonlocal needed_exchange_runtime_tensor_plan
            nonlocal owner_compact_local_experts_tensor
            exchange_transport = _standard_ep_needed_weight_exchange_transport()
            if needed_exchange_impl in {
                "auto",
                "backend",
                "native",
                "mori",
                "primus",
                "primus_turbo",
            }:
                if needed_exchange_fn is not None:
                    exchange_transport = (
                        _validate_standard_ep_needed_weight_exchange_transport(
                            exchange_transport,
                            backend_name=needed_exchange_backend_name,
                            backend_module=needed_exchange_backend_module,
                            exchange_impl=needed_exchange_impl,
                        )
                    )
                    with dsv4_timed_stage(
                        timing_records,
                        f"standard_ep_hot_weight.{label}.forward.needed.backend_exchange",
                    ):
                        if needed_exchange_tensor_plan is not None:
                            if needed_exchange_runtime_tensor_plan is None:
                                if (
                                    isinstance(needed_exchange_tensor_plan, dict)
                                    and "send_local_indices"
                                    in needed_exchange_tensor_plan
                                    and "grad_recv_local_indices"
                                    in needed_exchange_tensor_plan
                                ):
                                    needed_exchange_runtime_tensor_plan = (
                                        needed_exchange_tensor_plan
                                    )
                                else:
                                    if owner_compact_local_experts_tensor is None:
                                        owner_compact_local_experts_tensor = torch.tensor(
                                            tuple(
                                                int(v)
                                                for v in metadata.owner_compact_local_experts
                                            ),
                                            device=local_weight.device,
                                            dtype=torch.long,
                                        )
                                    if needed_prepare_runtime_plan_fn is not None:
                                        needed_exchange_runtime_tensor_plan = (
                                            needed_prepare_runtime_plan_fn(
                                                compact_local_indices=owner_compact_local_experts_tensor,
                                                plan=needed_exchange_tensor_plan,
                                                device=local_weight.device,
                                            )
                                        )
                                    else:
                                        needed_exchange_runtime_tensor_plan = dict(
                                            needed_exchange_tensor_plan
                                        )
                                        send_offsets = (
                                            needed_exchange_runtime_tensor_plan.get(
                                                "send_owner_compact_offsets"
                                            )
                                        )
                                        if isinstance(send_offsets, torch.Tensor):
                                            send_offsets = send_offsets.to(
                                                device=local_weight.device,
                                                dtype=torch.long,
                                            )
                                            if send_offsets.numel() > 0:
                                                needed_exchange_runtime_tensor_plan[
                                                    "send_local_indices"
                                                ] = owner_compact_local_experts_tensor.index_select(
                                                    0, send_offsets
                                                )
                                            else:
                                                needed_exchange_runtime_tensor_plan[
                                                    "send_local_indices"
                                                ] = torch.empty(
                                                    0,
                                                    device=local_weight.device,
                                                    dtype=torch.long,
                                                )
                                        grad_recv_offsets = (
                                            needed_exchange_runtime_tensor_plan.get(
                                                "grad_recv_owner_compact_offsets"
                                            )
                                        )
                                        if isinstance(grad_recv_offsets, torch.Tensor):
                                            grad_recv_offsets = grad_recv_offsets.to(
                                                device=local_weight.device,
                                                dtype=torch.long,
                                            )
                                            if grad_recv_offsets.numel() > 0:
                                                needed_exchange_runtime_tensor_plan[
                                                    "grad_recv_local_indices"
                                                ] = owner_compact_local_experts_tensor.index_select(
                                                    0, grad_recv_offsets
                                                )
                                            else:
                                                needed_exchange_runtime_tensor_plan[
                                                    "grad_recv_local_indices"
                                                ] = torch.empty(
                                                    0,
                                                    device=local_weight.device,
                                                    dtype=torch.long,
                                                )
                            compact_local_indices_arg = None
                            if not (
                                isinstance(needed_exchange_runtime_tensor_plan, dict)
                                and "send_local_indices"
                                in needed_exchange_runtime_tensor_plan
                                and "grad_recv_local_indices"
                                in needed_exchange_runtime_tensor_plan
                            ):
                                if owner_compact_local_experts_tensor is None:
                                    owner_compact_local_experts_tensor = torch.tensor(
                                        tuple(
                                            int(v)
                                            for v in metadata.owner_compact_local_experts
                                        ),
                                        device=local_weight.device,
                                        dtype=torch.long,
                                    )
                                compact_local_indices_arg = (
                                    owner_compact_local_experts_tensor
                                )
                            return needed_exchange_fn(
                                local_weight,
                                compact_local_indices=compact_local_indices_arg,
                                plan=needed_exchange_runtime_tensor_plan,
                                group=ep_group,
                                transport=exchange_transport,
                            )
                        if owner_compact_local_experts_tensor is None:
                            owner_compact_local_experts_tensor = torch.tensor(
                                tuple(
                                    int(v)
                                    for v in metadata.owner_compact_local_experts
                                ),
                                device=local_weight.device,
                                dtype=torch.long,
                            )
                        return needed_exchange_fn(
                            local_weight,
                            compact_local_indices=owner_compact_local_experts_tensor,
                            group=ep_group,
                            needed_owner_compact_offsets=needed_owner_compact_offsets,
                            send_owner_compact_offsets=(
                                needed_send_owner_compact_offsets
                            ),
                            recv_to_needed_index=needed_recv_to_needed_index,
                            grad_send_needed_indices=(
                                needed_grad_send_needed_indices
                            ),
                            grad_recv_owner_compact_offsets=(
                                needed_grad_recv_owner_compact_offsets
                            ),
                            input_splits=tuple(
                                len(row)
                                for row in needed_send_owner_compact_offsets_by_rank
                            ),
                            output_splits=tuple(
                                len(row)
                                for row in needed_recv_owner_compact_offsets_by_rank
                            ),
                            transport=exchange_transport,
                        )
                if needed_exchange_impl in {
                    "backend",
                    "native",
                    "mori",
                    "primus",
                    "primus_turbo",
                }:
                    raise RuntimeError(
                        "Backend owner-compact needed-row exchange was requested "
                        "but no selected balanced-MoE backend exposes "
                        "exchange_owner_compact_needed_rows in the active Python "
                        "environment. Requested "
                        f"backend={preferred_balanced_moe_backend!r}, "
                        f"exchange_impl={needed_exchange_impl!r}."
                    )
            return _StandardEPNeededHotWeightExchange.apply(
                local_weight,
                metadata.selected_global_experts,
                num_local_experts,
                ep_rank,
                ep_group,
                int(metadata.max_owned_per_rank),
                tuple(int(v) for v in metadata.owner_shard_active_offsets),
                needed_owner_compact_offsets,
                tuple(int(v) for v in metadata.owner_compact_owner_ranks),
                tuple(int(v) for v in metadata.owner_compact_local_experts),
                needed_need_mask,
                needed_need_masks,
                needed_send_owner_compact_offsets_by_rank,
                needed_recv_owner_compact_offsets_by_rank,
                needed_send_owner_compact_offsets,
                needed_recv_owner_compact_offsets,
                needed_recv_to_needed_index,
                needed_grad_send_needed_indices,
                needed_grad_recv_owner_compact_offsets,
                label,
            )

        use_w13_hot_split = _env_flag(
            "TORCHTITAN_STANDARD_EP_HOT_SPLIT_W13", default=False
        ) or _env_flag("CANARY_STANDARD_EP_HOT_SPLIT_W13", default=False)
        with dsv4_timed_stage(
            timing_records, "standard_ep.hot_split.broadcast_hot_weights"
        ):
            if effective_weight_layout == "needed":
                w2_hot = _needed_hot_weight(w2, "w2")
            elif effective_weight_layout == "owner_sharded":
                w2_hot = _owner_sharded_hot_weight(w2, "w2")
            else:
                w2_hot = _hot_weight(w2, "w2")
            if use_w13_hot_split:
                if effective_weight_layout == "needed":
                    w13_hot = _needed_hot_weight(torch.cat([w1, w3], dim=1), "w13")
                elif effective_weight_layout == "owner_sharded":
                    w13_hot = _owner_sharded_hot_weight(
                        torch.cat([w1, w3], dim=1), "w13"
                    )
                else:
                    w13_hot = _hot_weight(torch.cat([w1, w3], dim=1), "w13")
                w1_hot = None
                w3_hot = None
            else:
                if effective_weight_layout == "needed":
                    w1_hot = _needed_hot_weight(w1, "w1")
                    w3_hot = _needed_hot_weight(w3, "w3")
                elif effective_weight_layout == "owner_sharded":
                    w1_hot = _owner_sharded_hot_weight(w1, "w1")
                    w3_hot = _owner_sharded_hot_weight(w3, "w3")
                else:
                    w1_hot = _hot_weight(w1, "w1")
                    w3_hot = _hot_weight(w3, "w3")
                w13_hot = None

        with dsv4_timed_stage(timing_records, "standard_ep.hot_split.allocate_out"):
            out = torch.zeros(
                x.shape[0] * getattr(self.token_dispatcher, "sp_size", 1),
                x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )
        remote_rows = int(metadata.remote_token_indices.numel())
        attach_materialized_grad = _standard_ep_hot_split_attach_materialized_grad()
        num_hot_weight_groups = 0
        num_remote_hot_groups = 0
        num_remote_owner_compact_groups = 0
        num_remote_owner_shard_groups = 0
        used_materialized_rows = False
        bwd_scatter_span_id: int | None = None
        bwd_mlp_span_id: int | None = None
        bwd_w2_span_id: int | None = None
        bwd_activation_gate_span_id: int | None = None
        bwd_activation_up_span_id: int | None = None
        bwd_w1_span_id: int | None = None
        bwd_w3_span_id: int | None = None
        hot_split_backward_mode = _standard_ep_hot_split_backward_mode()
        compact_to_hot_for_selected_weight: torch.Tensor | None = None

        def _selected_weight_compact_to_hot(device: torch.device) -> torch.Tensor:
            nonlocal compact_to_hot_for_selected_weight
            if (
                compact_to_hot_for_selected_weight is None
                or compact_to_hot_for_selected_weight.device != device
            ):
                with dsv4_timed_stage(
                    timing_records,
                    "standard_ep.hot_split.align_selected_hot_weight.build_index",
                ):
                    compact_to_hot_for_selected_weight = (
                        _standard_ep_hot_to_owner_compact_order(
                            selected_global_experts=metadata.selected_global_experts,
                            num_local_experts=num_local_experts,
                            ep_size=int(metadata.ep_size),
                            max_owned_per_rank=int(metadata.max_owned_per_rank),
                            owner_shard_active_offsets=tuple(
                                int(v) for v in metadata.owner_shard_active_offsets
                            ),
                            device=device,
                        )
                    )
            return compact_to_hot_for_selected_weight

        def _align_selected_hot_weight_to_row_groups(
            hot_weight: torch.Tensor,
            label: str,
        ) -> torch.Tensor:
            if hot_weight_order_layout == hot_split_row_group_layout:
                return hot_weight
            if (
                hot_split_row_group_layout not in {"owner_compact", "needed_owner_compact"}
                or effective_weight_layout == "owner_sharded"
                or effective_weight_layout == "needed"
            ):
                return hot_weight
            compact_to_hot = _selected_weight_compact_to_hot(hot_weight.device)
            with dsv4_timed_stage(
                timing_records,
                f"standard_ep.hot_split.align_selected_hot_weight.{label}",
            ):
                return hot_weight.index_select(0, compact_to_hot)

        if use_w13_hot_split:
            if w13_hot is not None:
                w13_hot = _align_selected_hot_weight_to_row_groups(w13_hot, "w13")
        else:
            if w1_hot is not None:
                w1_hot = _align_selected_hot_weight_to_row_groups(w1_hot, "w1")
            w2_hot = _align_selected_hot_weight_to_row_groups(w2_hot, "w2")
            if w3_hot is not None:
                w3_hot = _align_selected_hot_weight_to_row_groups(w3_hot, "w3")
        if use_w13_hot_split:
            w2_hot = _align_selected_hot_weight_to_row_groups(w2_hot, "w2")

        if remote_rows > 0:
            hot_split_weighted_scatter_backend = (
                _standard_ep_hot_split_weighted_scatter_backend()
            )
            expected_presort = (
                "owner_compact"
                if hot_split_row_group_layout == "needed_owner_compact"
                else hot_split_row_group_layout
            )
            materialized_needed_remap = (
                hot_split_row_group_layout == "needed_owner_compact"
            )
            materialized_ready = (
                metadata.remote_rows_materialized_by == expected_presort
                and metadata.remote_materialized_x is not None
                and metadata.remote_materialized_token_indices is not None
                and metadata.remote_materialized_top_scores is not None
                and (
                    materialized_needed_remap
                    or metadata.remote_materialized_group_ends is not None
                )
            )
            if attach_materialized_grad and self.token_dispatcher.score_before_experts:
                materialized_ready = False
            if hot_split_row_group_layout == "needed_owner_compact":
                assert metadata.remote_owner_compact_offsets is not None
                assert needed_owner_compact_offsets_tensor is not None
                if compact_to_needed_index_tensor is not None:
                    compact_to_needed_index_tensor = compact_to_needed_index_tensor.to(
                        device=metadata.remote_owner_compact_offsets.device,
                        dtype=torch.long,
                    )
                    group_offsets = compact_to_needed_index_tensor.index_select(
                        0,
                        metadata.remote_owner_compact_offsets.to(torch.long).contiguous(),
                    )
                else:
                    needed_owner_compact_offsets_tensor = (
                        needed_owner_compact_offsets_tensor.to(
                            device=metadata.remote_owner_compact_offsets.device,
                            dtype=metadata.remote_owner_compact_offsets.dtype,
                        )
                    )
                    group_offsets = torch.searchsorted(
                        needed_owner_compact_offsets_tensor,
                        metadata.remote_owner_compact_offsets.contiguous(),
                    ).to(torch.long)
            else:
                group_offsets = (
                    metadata.remote_owner_compact_offsets
                    if hot_split_row_group_layout == "owner_compact"
                    else metadata.remote_hot_offsets
                )
                assert group_offsets is not None
            if dsv4_profile_timing_enabled() and not torch.compiler.is_compiling():
                num_remote_hot_groups = int(
                    torch.unique(metadata.remote_hot_offsets).numel()
                )
                if metadata.remote_owner_compact_offsets is not None:
                    num_remote_owner_compact_groups = int(
                        torch.unique(metadata.remote_owner_compact_offsets).numel()
                    )
                if metadata.remote_owner_shard_offsets is not None:
                    num_remote_owner_shard_groups = int(
                        torch.unique(metadata.remote_owner_shard_offsets).numel()
                    )
            num_hot_weight_groups = (
                len(needed_owner_compact_offsets)
                if hot_split_row_group_layout == "needed_owner_compact"
                else (
                    len(metadata.owner_shard_active_offsets)
                    if hot_split_row_group_layout == "owner_compact"
                    else len(metadata.selected_global_experts)
                )
            )
            if materialized_ready:
                with dsv4_timed_stage(
                    timing_records, "standard_ep.hot_split.use_materialized_rows"
                ):
                    inverse_order = None
                    sorted_offsets = group_offsets
                    sorted_x = metadata.remote_materialized_x
                    sorted_scores = metadata.remote_materialized_top_scores
                    token_indices = metadata.remote_materialized_token_indices
                    if materialized_needed_remap:
                        num_tokens_per_hot = _count_route_ids(
                            sorted_offsets, num_hot_weight_groups
                        )
                        offsets = torch.cumsum(
                            num_tokens_per_hot, dim=0, dtype=torch.int32
                        )
                    else:
                        offsets = metadata.remote_materialized_group_ends
                    if attach_materialized_grad:
                        sorted_x = _StandardEPHotSplitMaterializedRows.apply(
                            x,
                            sorted_x,
                            token_indices,
                        )
                        sorted_scores = _StandardEPHotSplitMaterializedScores.apply(
                            top_scores,
                            sorted_scores,
                            metadata.remote_flat_positions,
                        )
                    used_materialized_rows = True
            else:
                with dsv4_timed_stage(timing_records, "standard_ep.hot_split.gather_remote"):
                    remote_x = x[metadata.remote_token_indices]
                    remote_scores = metadata.remote_top_scores
                    if self.token_dispatcher.score_before_experts:
                        remote_x = (
                            remote_x.to(torch.float32) * remote_scores.reshape(-1, 1)
                        ).to(x.dtype)

                with dsv4_timed_stage(timing_records, "standard_ep.hot_split.sort_offsets"):
                    if metadata.remote_offsets_presorted_by == expected_presort:
                        inverse_order = None
                        sorted_offsets = group_offsets
                        sorted_x = remote_x
                        sorted_scores = remote_scores
                    else:
                        order = torch.argsort(group_offsets, stable=True)
                        inverse_order = torch.empty_like(order)
                        inverse_order[order] = torch.arange(order.numel(), device=order.device)
                        sorted_offsets = group_offsets[order]
                        sorted_x = remote_x[order]
                        sorted_scores = remote_scores[order]
                with dsv4_timed_stage(timing_records, "standard_ep.hot_split.count_offsets"):
                    num_tokens_per_hot = _count_route_ids(
                        sorted_offsets, num_hot_weight_groups
                    )
                    offsets = torch.cumsum(num_tokens_per_hot, dim=0, dtype=torch.int32)

            if _standard_ep_hot_split_backward_timing_enabled():
                span_meta = {
                    "ep_rank": int(ep_rank),
                    "ep_size": int(metadata.ep_size),
                    "num_local_experts": int(num_local_experts),
                    "num_selected_hot_experts": len(metadata.selected_global_experts),
                    "remote_rows": int(remote_rows),
                    "num_hot_weight_groups": int(num_hot_weight_groups),
                    "hot_split_weight_layout_effective": effective_weight_layout,
                    "hot_split_row_group_layout": hot_split_row_group_layout,
                    "hot_split_weight_order_requested": requested_weight_order,
                    "hot_split_weight_order_effective": hot_weight_order_layout,
                    "hot_split_weight_order_fallback_reason": (
                        hot_weight_order_fallback_reason
                    ),
                    "needed_owner_compact_offsets": list(
                        needed_owner_compact_offsets
                    ),
                    "needed_exchange_backend": needed_exchange_backend_name,
                    "remote_offsets_presorted_by": metadata.remote_offsets_presorted_by,
                    "remote_rows_materialized_by": metadata.remote_rows_materialized_by,
                    "used_materialized_rows": bool(used_materialized_rows),
                    "score_before_experts": bool(self.token_dispatcher.score_before_experts),
                    "use_w13_hot_split": bool(use_w13_hot_split),
                    "hot_split_backward_mode": hot_split_backward_mode,
                    "x_shape": list(x.shape),
                    "sorted_x_shape": list(sorted_x.shape),
                    "row_group_summary": _profile_cumulative_offsets_summary(offsets),
                }
                bwd_scatter_span_id = _new_standard_ep_hot_split_bwd_span(
                    "standard_ep.hot_split.backward.scatter_to_remote_y",
                    span_meta,
                )
                bwd_mlp_span_id = _new_standard_ep_hot_split_bwd_span(
                    "standard_ep.hot_split.backward.helper_mlp_vjp_to_sorted_x",
                    span_meta,
                )
                if (
                    _standard_ep_hot_split_autograd_vjp_detail_enabled()
                    and hot_split_backward_mode == "autograd"
                    and not use_w13_hot_split
                ):
                    bwd_w2_span_id = _new_standard_ep_hot_split_bwd_span(
                        "standard_ep.hot_split.backward.autograd_vjp.w2",
                        {**span_meta, "autograd_vjp_detail": True},
                    )
                    bwd_activation_gate_span_id = _new_standard_ep_hot_split_bwd_span(
                        "standard_ep.hot_split.backward.autograd_vjp.activation_gate",
                        {**span_meta, "autograd_vjp_detail": True},
                    )
                    bwd_activation_up_span_id = _new_standard_ep_hot_split_bwd_span(
                        "standard_ep.hot_split.backward.autograd_vjp.activation_up",
                        {**span_meta, "autograd_vjp_detail": True},
                    )
                    bwd_w1_span_id = _new_standard_ep_hot_split_bwd_span(
                        "standard_ep.hot_split.backward.autograd_vjp.w1",
                        {**span_meta, "autograd_vjp_detail": True},
                    )
                    bwd_w3_span_id = _new_standard_ep_hot_split_bwd_span(
                        "standard_ep.hot_split.backward.autograd_vjp.w3",
                        {**span_meta, "autograd_vjp_detail": True},
                    )
                sorted_x = _StandardEPHotSplitBackwardSpanEnd.apply(
                    sorted_x,
                    bwd_mlp_span_id,
                )

            use_manual_helper_vjp = (
                hot_split_backward_mode != "autograd"
                and not use_w13_hot_split
                and w1_hot is not None
                and w3_hot is not None
            )
            if use_manual_helper_vjp:
                assert w1_hot is not None and w3_hot is not None
                with dsv4_timed_stage(
                    timing_records,
                    "standard_ep.hot_split.manual_grouped_mlp",
                ):
                    remote_y = _StandardEPHotSplitGroupedMLP.apply(
                        sorted_x,
                        w1_hot,
                        w3_hot,
                        w2_hot,
                        offsets,
                        hot_split_backward_mode,
                    )
            elif use_w13_hot_split:
                assert w13_hot is not None
                with dsv4_timed_stage(
                    timing_records, "standard_ep.hot_split.grouped_w13"
                ):
                    h13 = torch._grouped_mm(
                        sorted_x.bfloat16(),
                        w13_hot.bfloat16().transpose(-2, -1),
                        offs=offsets,
                    )
                    h1, h3 = torch.split(h13, int(w1.shape[1]), dim=-1)
                    h = _standard_ep_hot_split_swiglu(h1, h3)
            else:
                assert w1_hot is not None and w3_hot is not None
                sorted_x_w1 = sorted_x
                sorted_x_w3 = sorted_x
                if bwd_w1_span_id is not None:
                    sorted_x_w1 = _StandardEPHotSplitBackwardSpanEnd.apply(
                        sorted_x_w1,
                        bwd_w1_span_id,
                    )
                if bwd_w3_span_id is not None:
                    sorted_x_w3 = _StandardEPHotSplitBackwardSpanEnd.apply(
                        sorted_x_w3,
                        bwd_w3_span_id,
                    )
                with dsv4_timed_stage(
                    timing_records, "standard_ep.hot_split.grouped_w1"
                ):
                    gate = torch._grouped_mm(
                        sorted_x_w1.bfloat16(),
                        w1_hot.bfloat16().transpose(-2, -1),
                        offs=offsets,
                    )
                    if (
                        bwd_activation_gate_span_id is not None
                        and bwd_w1_span_id is not None
                    ):
                        gate = _StandardEPHotSplitBackwardBoundary.apply(
                            gate,
                            bwd_activation_gate_span_id,
                            bwd_w1_span_id,
                        )
                with dsv4_timed_stage(
                    timing_records, "standard_ep.hot_split.grouped_w3"
                ):
                    up = torch._grouped_mm(
                        sorted_x_w3.bfloat16(),
                        w3_hot.bfloat16().transpose(-2, -1),
                        offs=offsets,
                    )
                    if (
                        bwd_activation_up_span_id is not None
                        and bwd_w3_span_id is not None
                    ):
                        up = _StandardEPHotSplitBackwardBoundary.apply(
                            up,
                            bwd_activation_up_span_id,
                            bwd_w3_span_id,
                        )
                    h = _standard_ep_hot_split_swiglu(gate, up)
            if not use_manual_helper_vjp:
                if (
                    bwd_w2_span_id is not None
                    and bwd_activation_gate_span_id is not None
                    and bwd_activation_up_span_id is not None
                ):
                    h = _StandardEPHotSplitBackwardSpanStart.apply(
                        h,
                        bwd_activation_up_span_id,
                    )
                    h = _StandardEPHotSplitBackwardBoundary.apply(
                        h,
                        bwd_w2_span_id,
                        bwd_activation_gate_span_id,
                    )
                with dsv4_timed_stage(timing_records, "standard_ep.hot_split.grouped_w2"):
                    remote_y = torch._grouped_mm(
                        h,
                        w2_hot.bfloat16().transpose(-2, -1),
                        offs=offsets,
                    ).type_as(x)
                if bwd_w2_span_id is not None:
                    remote_y = _StandardEPHotSplitBackwardSpanStart.apply(
                        remote_y,
                        bwd_w2_span_id,
                    )
            if bwd_scatter_span_id is not None and bwd_mlp_span_id is not None:
                remote_y = _StandardEPHotSplitBackwardBoundary.apply(
                    remote_y,
                    bwd_scatter_span_id,
                    bwd_mlp_span_id,
                )
            with dsv4_timed_stage(timing_records, "standard_ep.hot_split.inverse_order"):
                if inverse_order is not None:
                    remote_y = remote_y[inverse_order]
                    scatter_scores = sorted_scores[inverse_order]
                else:
                    scatter_scores = sorted_scores

            if not used_materialized_rows:
                token_indices = metadata.remote_token_indices
            if getattr(self.token_dispatcher, "sp_size", 1) > 1:
                token_indices = (
                    token_indices + x.shape[0] * self.token_dispatcher.sp_rank
                )
            if not self.token_dispatcher.score_before_experts:
                with dsv4_timed_stage(
                    timing_records, "standard_ep.hot_split.weighted_scatter_tokens"
                ):
                    out = weighted_scatter_add(
                        out,
                        token_indices,
                        remote_y,
                        scatter_scores,
                        backend=hot_split_weighted_scatter_backend,
                    )
            else:
                with dsv4_timed_stage(
                    timing_records, "standard_ep.hot_split.scatter_tokens"
                ):
                    out = deterministic_scatter_add(
                        out,
                        token_indices.reshape(-1, 1).expand(-1, out.shape[-1]),
                        remote_y,
                    )
            if bwd_scatter_span_id is not None:
                out = _StandardEPHotSplitBackwardSpanStart.apply(
                    out,
                    bwd_scatter_span_id,
                )

        with dsv4_timed_stage(timing_records, "standard_ep.hot_split.participation"):
            if use_w13_hot_split:
                assert w13_hot is not None
                participation = (w2_hot.sum() + w13_hot.sum()) * 0.0
            else:
                assert w1_hot is not None and w3_hot is not None
                participation = (w1_hot.sum() + w2_hot.sum() + w3_hot.sum()) * 0.0
        flush_dsv4_profile_timing(
            timing_records,
            {
                "kind": "standard_ep_hot_split",
                "phase": "forward",
                "selected_global_experts": list(metadata.selected_global_experts),
                "num_selected_hot_experts": len(metadata.selected_global_experts),
                "remote_rows": int(remote_rows),
                "ep_rank": int(ep_rank),
                "ep_size": int(metadata.ep_size),
                "num_local_experts": int(num_local_experts),
                "hot_split_weight_layout_requested": requested_weight_layout,
                "hot_split_weight_layout_effective": effective_weight_layout,
                "hot_split_row_group_layout": hot_split_row_group_layout,
                "hot_split_weight_order_requested": requested_weight_order,
                "hot_split_weight_order_effective": hot_weight_order_layout,
                "hot_split_weight_order_fallback_reason": (
                    hot_weight_order_fallback_reason
                ),
                "hot_weight_global_experts": list(hot_weight_global_experts),
                "owner_counts": list(metadata.owner_counts),
                "owner_shard_active_offsets": list(metadata.owner_shard_active_offsets),
                "owner_shard_active_count": len(metadata.owner_shard_active_offsets),
                "max_owned_per_rank": int(metadata.max_owned_per_rank),
                "padded_hot_rows": int(padded_hot_rows),
                "owner_sharded_pad_factor": float(owner_sharded_pad_factor),
                "owner_sharded_max_pad_factor": float(max_pad_factor),
                "owner_sharded_fell_back_to_selected": bool(
                    requested_weight_layout == "owner_sharded"
                    and effective_weight_layout != "owner_sharded"
                ),
                "needed_fell_back_to_selected": bool(
                    requested_weight_layout == "needed"
                    and effective_weight_layout != "needed"
                ),
                "needed_fallback_reason": needed_fallback_reason,
                "needed_local_count": int(needed_local_count),
                "needed_max_count": int(needed_max_count),
                "needed_max_density": float(needed_max_density),
                "needed_owner_compact_offsets": list(needed_owner_compact_offsets),
                "needed_owner_compact_count": len(needed_owner_compact_offsets),
                "num_hot_weight_groups": int(num_hot_weight_groups),
                "num_remote_hot_groups": int(num_remote_hot_groups),
                "num_remote_owner_compact_groups": int(num_remote_owner_compact_groups),
                "num_remote_owner_shard_groups": int(num_remote_owner_shard_groups),
                "remote_offsets_presorted_by": metadata.remote_offsets_presorted_by,
                "remote_rows_materialized_by": metadata.remote_rows_materialized_by,
                "used_materialized_rows": bool(used_materialized_rows),
                "attach_materialized_grad": bool(attach_materialized_grad),
                "x_shape": list(x.shape),
                "use_w13_hot_split": bool(use_w13_hot_split),
                "hot_split_backward_mode": hot_split_backward_mode,
                "row_group_summary": _profile_cumulative_offsets_summary(
                    offsets if remote_rows > 0 else None
                ),
            },
        )
        return out + participation

    def _schedule_prepare_forward(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> MoEScheduleNodeState:
        """Flatten token-major inputs for dispatch schedule nodes."""
        bs, slen, dim = x.shape
        top_k = top_scores.size(-1)
        return MoEScheduleNodeState(
            flat_x=x.view(bs * slen, dim),
            top_scores=top_scores.view(bs * slen, top_k),
            selected_experts_indices=selected_experts_indices.view(bs * slen, top_k),
            original_shape=(bs, slen, dim),
        )

    def _schedule_mori_aiter_forward(
        self,
        state: MoEScheduleNodeState,
    ) -> torch.Tensor | None:
        """Run the retained fused MORI/AITER bridge when that backend is active."""
        if not (
            isinstance(self.token_dispatcher, MoriEPTokenDispatcher)
            and self.token_dispatcher.ep_mesh is not None
            and self.token_dispatcher.training_bridge_enabled()
        ):
            return None

        from .mori_aiter_moe import mori_aiter_grouped_experts_forward

        layer_idx = getattr(self, "layer_idx", None)
        profile_meta: dict[str, Any] = {"moe_module": type(self).__name__}
        if layer_idx is not None:
            profile_meta["layer_idx"] = int(layer_idx)
        with torch.profiler.record_function("moe.mori_aiter_bridge"):
            return mori_aiter_grouped_experts_forward(
                x=state.flat_x,
                top_scores=state.top_scores,
                selected_experts_indices=state.selected_experts_indices,
                w1=self.w1,
                w3=self.w3,
                w2=self.w2,
                dispatcher=self.token_dispatcher,
                profile_meta=profile_meta,
            )

    def _schedule_dispatch_forward(
        self,
        state: MoEScheduleNodeState,
    ) -> MoEScheduleNodeState:
        """Forward A2A dispatch node."""
        with torch.profiler.record_function("moe.dispatch"):
            routed_input, num_tokens_local, metadata = self.token_dispatcher.dispatch(
                state.flat_x,
                state.top_scores,
                state.selected_experts_indices,
            )
        state.routed_input = routed_input
        state.num_tokens_local = num_tokens_local
        state.metadata = metadata
        return state

    def _schedule_experts_forward(
        self,
        state: MoEScheduleNodeState,
    ) -> MoEScheduleNodeState:
        """Local expert compute node."""
        if state.routed_input is None or state.num_tokens_local is None:
            raise RuntimeError("MoE experts node requires dispatch output.")
        with torch.profiler.record_function("moe.local_experts"):
            state.routed_output = self._experts_forward(
                state.routed_input,
                state.num_tokens_local,
            )
        return state

    def _schedule_hot_split_forward(
        self,
        state: MoEScheduleNodeState,
    ) -> torch.Tensor | None:
        """Source-rank hot-helper node used by standard EP execute mode."""
        metadata = state.metadata
        if not (
            isinstance(metadata, AllToAllDispatchMetadata)
            and metadata.standard_ep_hot_split is not None
        ):
            return None
        with torch.profiler.record_function("moe.standard_ep_hot_split"):
            return self._standard_ep_hot_split_forward(
                state.flat_x,
                state.top_scores,
                metadata.standard_ep_hot_split,
            )

    def _schedule_combine_forward(
        self,
        state: MoEScheduleNodeState,
    ) -> torch.Tensor:
        """Forward A2A combine node plus optional standard-EP hot-helper output."""
        metadata = state.metadata
        if state.routed_output is None or metadata is None:
            raise RuntimeError("MoE combine node requires expert output and metadata.")
        if (
            isinstance(metadata, AllToAllDispatchMetadata)
            and metadata.standard_ep_hot_split is not None
            and _standard_ep_overlap_hot_split_forward()
        ):
            combine_with_overlap = getattr(
                self.token_dispatcher, "combine_with_overlap", None
            )
            if combine_with_overlap is not None:
                with torch.profiler.record_function(
                    "moe.combine_overlap_standard_ep_hot_split"
                ):
                    combined, hot_split_out = combine_with_overlap(
                        state.routed_output,
                        metadata,
                        state.flat_x,
                        lambda: self._standard_ep_hot_split_forward(
                            state.flat_x,
                            state.top_scores,
                            metadata.standard_ep_hot_split,
                        ),
                    )
                if hot_split_out is not None:
                    combined = combined + hot_split_out
                return combined

        with torch.profiler.record_function("moe.combine"):
            combined = self.token_dispatcher.combine(
                state.routed_output,
                metadata,
                state.flat_x,
            )
        hot_split_out = self._schedule_hot_split_forward(state)
        if hot_split_out is not None:
            combined = combined + hot_split_out
        return combined

    def forward(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, compute, combine, and scatter_add.

        When parallelized, ``local_map`` (from ``sharding_config``) handles
        DTensor→local conversion on entry and local→DTensor(Partial) wrapping
        on exit. The forward body operates on plain local tensors.
        """
        state = self._schedule_prepare_forward(
            x,
            top_scores,
            selected_experts_indices,
        )
        bridge_output = self._schedule_mori_aiter_forward(state)
        if bridge_output is not None:
            return bridge_output
        state = self._schedule_dispatch_forward(state)
        state = self._schedule_experts_forward(state)
        return self._schedule_combine_forward(state)

    def parallelize(self, parallel_dims) -> None:
        """Parallelize expert weights, then wire EP/TP meshes on the dispatcher
        so dispatch/combine see the right meshes at runtime."""
        super().parallelize(parallel_dims)
        # TODO(@pianpwk): With spmd_types and set_current_mesh, replace wire_meshes
        # with current_mesh calls inside AllToAllTokenDispatcher and
        # DeepEPTokenDispatcher.
        self.token_dispatcher.wire_meshes(
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
        )


class TokenChoiceTopKRouter(Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Optionally supports node-limited (group-limited) routing where experts are divided into groups
    (e.g., by node), and only num_limited_groups groups are considered before selecting top_k experts.
    This reduces cross-node communication in distributed settings.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        gate: Linear.Config
        num_expert_groups: int | None = None  # must be a divisor of num_experts
        num_limited_groups: int | None = None
        top_k: int = 1
        score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sigmoid"
        route_norm: bool = False
        route_scale: float = 1.0
        _debug_force_load_balance: bool = False
        hash_routing_vocab_size: int | None = None
        hash_routing_chunk_size: int = 8192

    def __init__(self, config: Config):
        super().__init__()
        self.gate = config.gate.build()
        self.num_experts = config.num_experts
        self.num_expert_groups = config.num_expert_groups
        self.num_limited_groups = config.num_limited_groups
        self.top_k = config.top_k
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self._debug_force_load_balance = config._debug_force_load_balance
        self.hash_routing_vocab_size = config.hash_routing_vocab_size
        self.hash_routing_chunk_size = int(config.hash_routing_chunk_size)
        if self.hash_routing_vocab_size is not None:
            self.register_buffer(
                "tid2eid",
                self._build_hash_routing_table(
                    int(self.hash_routing_vocab_size),
                    self.num_experts,
                    self.top_k,
                    chunk_size=self.hash_routing_chunk_size,
                ),
                persistent=True,
            )
        else:
            self.tid2eid = None

    @staticmethod
    def _build_hash_routing_table(
        vocab_size: int,
        num_experts: int,
        top_k: int,
        *,
        chunk_size: int = 8192,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
        tid2eid = torch.empty((vocab_size, top_k), dtype=torch.long, device=device)
        for start in range(0, vocab_size, chunk_size):
            stop = min(start + chunk_size, vocab_size)
            tid2eid[start:stop] = (
                torch.rand((stop - start, num_experts), device=device)
                .topk(top_k, dim=-1)
                .indices
            )
        return tid2eid

    def _debug_force_load_balance_routing(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.
        Returns (selected_experts_indices (bs, slen, top_k) LongTensor, top_scores (bs, slen, top_k) FloatTensor).
        """
        bs, slen, _ = scores.shape
        # Round-robin indices with exact balance
        selected_experts_indices = (
            torch.arange(
                bs * slen * self.top_k, device=scores.device, dtype=torch.int64
            ).reshape(bs, slen, self.top_k)
            % self.num_experts
        )
        top_scores = scores.gather(dim=-1, index=selected_experts_indices)
        return selected_experts_indices, top_scores

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice: torch.Tensor,
    ) -> torch.Tensor:
        """Select num_limited_groups groups based on group scores,
            and set expert scores in non-selected groups as -inf

        Args:
            scores_for_choice: Router scores with expert_bias (if any), shape (bs, slen, num_experts)

        Returns:
            scores_for_choice: shape (bs, slen, num_experts)
        """
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        assert self.num_expert_groups is not None
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by num_expert_groups ({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group ({experts_per_group}) must be >= 2")
        scores_grouped = scores_for_choice.unflatten(
            -1, (self.num_expert_groups, experts_per_group)
        )
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)
        _, group_idx = torch.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(-1, group_idx, False)  # False = selected groups (keep)
        # Mask out experts from non-selected groups
        scores_for_choice = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).flatten(-2)

        return scores_for_choice

    def forward(
        self,
        x: torch.Tensor,
        expert_bias: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs, slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs, slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # Compute gate in float32 to help stability of expert load balancing.
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            scores = _profiled_linear(
                self.gate,
                x,
                "dsv4.linear.moe.router_gate",
            )

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        # scored is already float32 from the autocast above.
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        elif self.score_func == "softmax":
            scores = F.softmax(scores, dim=-1)
        elif self.score_func == "sqrtsoftplus":
            scores = torch.sqrt(F.softplus(scores))
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if self.tid2eid is not None:
            if input_ids is None:
                raise ValueError("hash-routed TokenChoiceTopKRouter requires input_ids")
            if input_ids.shape != scores.shape[:-1]:
                raise ValueError(
                    "hash-routed TokenChoiceTopKRouter expected input_ids shape "
                    f"{tuple(scores.shape[:-1])}, got {tuple(input_ids.shape)}"
                )
            selected_experts_indices = self.tid2eid[
                input_ids.to(device=scores.device, dtype=torch.long).reshape(-1)
            ].view(*scores.shape[:-1], self.top_k)
        else:
            scores_for_choice = scores if expert_bias is None else scores + expert_bias
            # Apply node-limited routing if configured
            if self.num_expert_groups is not None:
                scores_for_choice = self._get_node_limited_routing_scores(scores_for_choice)
            _, selected_experts_indices = torch.topk(
                scores_for_choice, k=self.top_k, dim=-1, sorted=False
            )

        # top scores shape (bs, slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        top_scores = scores.gather(dim=-1, index=selected_experts_indices)

        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = _count_route_ids(
            selected_experts_indices, self.num_experts
        ).to(torch.float32)

        return top_scores, selected_experts_indices, num_tokens_per_expert


class MoE(Module):
    """Mixture of Experts layer.

    The forward pass proceeds as:
    1. Router computes expert assignments (stays on DTensor)
    2. GroupedExperts.forward() converts DTensor to local, then handles:
       a. dispatch (TokenDispatcher) — reorder tokens by expert assignment.
          With EP, also performs all-to-all communication to send tokens
          to expert-owning ranks.
       b. expert computation (local tensors)
       c. combine (TokenDispatcher) — reverse the dispatch reordering.
          - LocalTokenDispatcher (no EP): scatter_add only.
          - AllToAll: all-to-all communication, then scatter_add.
          - DeepEP: async combine_tokens (sync deferred to step 4 when
            sp_size == 1; forced inside combine when sp_size > 1).
          - HybridEP: synchronous combine_tokens.
    3. Shared experts run on DTensor. Overlaps with DeepEP async combine
       when sp_size == 1; no overlap otherwise.
    4. Routed and shared expert outputs are summed.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int = 8
        experts: GroupedExperts.Config
        router: TokenChoiceTopKRouter.Config
        load_balance_coeff: float | None = 1e-3
        shared_experts: FeedForward.Config | None = None

    def __init__(self, config: Config):
        super().__init__()

        num_experts = config.num_experts
        self.experts = config.experts.build()
        self.router = config.router.build()
        self.shared_experts = (
            config.shared_experts.build() if config.shared_experts is not None else None
        )
        if self.shared_experts is not None:
            self.shared_experts.profile_label_prefix = (
                "dsv4.linear.moe.shared_experts"
            )

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = config.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.

        Under TP, the MoE wrapper's ``sharding_config`` (set by
        ``set_moe_sharding_config``) handles input/output redistribution:
        input is redistributed from sp_layout to desired_input_layouts;
        output (Partial) is redistributed to sp_layout. MoE.forward()
        operates on DTensors — the DTensor→local conversion happens at
        the GroupedExperts boundary.
        """
        bs, slen, dim = x.shape

        # top_scores and selected_experts_indices shape (bs, slen, top_k)
        # num_tokens_per_expert shape (num_experts,)
        with torch.profiler.record_function("moe.router"):
            (
                top_scores,
                selected_experts_indices,
                num_tokens_per_expert,
            ) = self.router(x, self.expert_bias, input_ids=input_ids)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        with torch.profiler.record_function("moe.tokens_per_expert_update"):
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)

        with torch.profiler.record_function("moe.routed_experts"):
            out = self.experts(x, top_scores, selected_experts_indices)

        # shared_experts runs in parallel with deepep combine communication.
        with torch.profiler.record_function("moe.shared_experts"):
            shared_out = self.shared_experts(x) if self.shared_experts is not None else None

        if (
            isinstance(self.experts.token_dispatcher, DeepEPTokenDispatcher)
            and self.experts.token_dispatcher.sp_size == 1
        ):
            # Sync the combine operation before using routed_output.
            # This inserts a CUDA stream wait, ensuring combine is complete before
            # the subsequent addition or view operations read routed output.
            from torchtitan.distributed.deepep.deepep import sync_combine

            with torch.profiler.record_function("moe.deepep_sync_combine"):
                sync_combine()

        with torch.profiler.record_function("moe.output_sum"):
            out = out.view(bs, slen, dim)
            if shared_out is not None:
                out = out + shared_out
            return out

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        assert isinstance(buffer_device, torch.device)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
