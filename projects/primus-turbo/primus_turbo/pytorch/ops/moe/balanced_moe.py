###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Balanced-MoE planning utilities for Primus-Turbo PyTorch MoE.

This module is deliberately framework-neutral.  A training stack such as
TorchTitan can choose when balanced MoE is enabled and how many hot experts to
relocate, while Primus-Turbo owns the concrete layout contract that its EP
dispatch/combine path should consume:

* normal/cold routes remain in the raw EP path;
* remote-hot routes are helper-executed on source ranks;
* hot expert weights are addressed by owner-compact row id.

Keeping this ABI inside Primus-Turbo avoids growing a second, incompatible
owner-compact format in every framework integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch

__all__ = [
    "BALANCED_MOE_BACKEND_ABI_VERSION",
    "BALANCED_MOE_BACKEND_CAPABILITIES",
    "BalancedMoeHotExpert",
    "BalancedMoeCompactCombineOutput",
    "BalancedMoeCompactDispatchOutput",
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

BALANCED_MOE_BACKEND_ABI_VERSION = 1
OWNER_COMPACT_TORCH_A2A_TRANSPORT = "torch_distributed_all_to_all_single"
OWNER_COMPACT_PRIMUS_TURBO_DISPATCH_TRANSPORT = "primus_turbo_moe_dispatch"
OWNER_COMPACT_MORI_SDMA_PADDED_A2A_TRANSPORT = "mori_sdma_padded_all2all"
BALANCED_MOE_COMPACT_ROWS_TORCH_REFERENCE_BACKEND = (
    "torch_distributed_all_to_all_single_reference"
)
BALANCED_MOE_BACKEND_CAPABILITIES = {
    "backend": "primus_turbo",
    "hot_expert_planner": True,
    "source_partition": True,
    "owner_compact_exchange_plan": True,
    "owner_compact_runtime_layout": True,
    "owner_compact_exchange_autograd": True,
    "normal_topk_dispatch_tensors": True,
    "normal_topk_ep_dispatch": True,
    "normal_topk_ep_dispatch_backend": "primus_turbo_moe_dispatch",
    "normal_topk_ep_dispatch_permute": True,
    "normal_topk_ep_dispatch_permute_backend": "primus_turbo_moe_dispatch_permute",
    "balanced_moe_compact_dispatch": True,
    "balanced_moe_compact_dispatch_backend": (
        "primus_turbo_raw_topk_ep_plus_source_compact_hot_rows"
    ),
    "balanced_moe_compact_combine": True,
    "balanced_moe_compact_combine_backend": (
        "primus_turbo_raw_topk_ep_plus_torch_index_add"
    ),
    "balanced_moe_compact_rows_api": True,
    "balanced_moe_compact_rows_reference_dispatch": True,
    "balanced_moe_compact_rows_reference_dispatch_backend": (
        BALANCED_MOE_COMPACT_ROWS_TORCH_REFERENCE_BACKEND
    ),
    "balanced_moe_compact_rows_reference_combine": True,
    "balanced_moe_compact_rows_reference_combine_backend": (
        BALANCED_MOE_COMPACT_ROWS_TORCH_REFERENCE_BACKEND
    ),
    "balanced_moe_compact_rows_dispatch": False,
    "balanced_moe_compact_rows_dispatch_backend": "not_implemented",
    "balanced_moe_compact_rows_combine": False,
    "balanced_moe_compact_rows_combine_backend": "not_implemented",
    "balanced_moe_compact_rows_native": False,
    "balanced_moe_compact_rows_native_status": (
        "compact-row dispatch/combine still use the torch reference path; "
        "owner-compact hot-weight exchange can use MORI SDMA"
    ),
    "owner_compact_exchange_transport": OWNER_COMPACT_TORCH_A2A_TRANSPORT,
    "owner_compact_exchange_transports": (
        OWNER_COMPACT_TORCH_A2A_TRANSPORT,
        OWNER_COMPACT_PRIMUS_TURBO_DISPATCH_TRANSPORT,
        OWNER_COMPACT_MORI_SDMA_PADDED_A2A_TRANSPORT,
    ),
    "native_hot_helper_transport": True,
    "native_hot_helper_transport_status": (
        "owner_compact_exchange_can_delegate_to_mori_sdma_padded_all2all"
    ),
    "native_owner_compact_exchange_transport": (
        OWNER_COMPACT_MORI_SDMA_PADDED_A2A_TRANSPORT
    ),
}


def balanced_moe_backend_capabilities() -> dict[str, object]:
    """Return the balanced-MoE backend ABI surface implemented here.

    This module owns the Primus-Turbo hot/cold/helper layout ABI.  Compact-row
    dispatch/combine is exposed here so framework integrations can import the
    balanced-MoE feature from Primus-Turbo instead of reimplementing the ABI,
    but the current compact-row implementation is still a torch-distributed
    reference.  Primus-Turbo also exposes an opt-in synthetic-rank TurboEP
    transport for the owner-compact hot-weight edge; throughput gates should
    require ``balanced_moe_compact_rows_native`` before claiming a fully native
    compact-row promotion.
    """

    return {
        "abi_version": BALANCED_MOE_BACKEND_ABI_VERSION,
        **BALANCED_MOE_BACKEND_CAPABILITIES,
    }


@dataclass(frozen=True)
class BalancedMoeHotExpert:
    """One hot expert selected for helper execution."""

    global_expert: int
    owner_rank: int
    local_expert: int
    source_rank_counts: tuple[int, ...]
    rows_total: int
    remote_rows: int
    owner_slot: int
    owner_shard_offset: int
    owner_compact_offset: int


@dataclass(frozen=True)
class BalancedMoePlan:
    """Rank-global balanced-MoE plan.

    ``owner_shard_offset`` is stable for owner-sharded hot-weight tensors shaped
    ``[world_size, max_owned_per_rank, ...]``. ``owner_compact_offset`` is
    stable for compact active-hot tensors shaped ``[num_hot, ...]``. Keeping
    both lets Primus-Turbo consume owner-shaped or compact helper layouts
    without converting back to selected-hot order.
    """

    world_size: int
    num_local_experts: int
    hot_experts: tuple[BalancedMoeHotExpert, ...]
    owner_counts: tuple[int, ...]
    max_owned_per_rank: int
    owner_shard_active_offsets: tuple[int, ...]
    owner_compact_owner_ranks: tuple[int, ...]
    owner_compact_local_experts: tuple[int, ...]
    owner_compact_need_masks: tuple[tuple[bool, ...], ...]
    owner_load_before: tuple[int, ...]
    exec_load_after: tuple[int, ...]
    selected_rows_total: int
    remote_rows_total: int
    modeled_max_load_reduction_pct: float

    @property
    def selected_global_experts(self) -> tuple[int, ...]:
        return tuple(item.global_expert for item in self.hot_experts)

    @property
    def selected_owner_ranks(self) -> tuple[int, ...]:
        return tuple(item.owner_rank for item in self.hot_experts)

    @property
    def selected_local_experts(self) -> tuple[int, ...]:
        return tuple(item.local_expert for item in self.hot_experts)

    @property
    def owner_shard_offsets(self) -> tuple[int, ...]:
        return tuple(item.owner_shard_offset for item in self.hot_experts)

    @property
    def owner_compact_offsets(self) -> tuple[int, ...]:
        return tuple(item.owner_compact_offset for item in self.hot_experts)

    def needed_owner_compact_offsets(self, rank: int) -> tuple[int, ...]:
        rank = int(rank)
        if rank < 0 or rank >= self.world_size:
            raise IndexError(f"rank {rank} is outside world_size={self.world_size}")
        return tuple(
            compact_idx
            for compact_idx, needed in enumerate(self.owner_compact_need_masks[rank])
            if bool(needed)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "world_size": self.world_size,
            "num_local_experts": self.num_local_experts,
            "selected_global_experts": list(self.selected_global_experts),
            "selected_owner_ranks": list(self.selected_owner_ranks),
            "selected_local_experts": list(self.selected_local_experts),
            "owner_counts": list(self.owner_counts),
            "max_owned_per_rank": self.max_owned_per_rank,
            "owner_shard_offsets": list(self.owner_shard_offsets),
            "owner_shard_active_offsets": list(self.owner_shard_active_offsets),
            "owner_compact_offsets": list(self.owner_compact_offsets),
            "owner_compact_owner_ranks": list(self.owner_compact_owner_ranks),
            "owner_compact_local_experts": list(self.owner_compact_local_experts),
            "owner_compact_need_masks": [
                list(row) for row in self.owner_compact_need_masks
            ],
            "needed_owner_compact_offsets_by_rank": [
                list(self.needed_owner_compact_offsets(rank))
                for rank in range(self.world_size)
            ],
            "owner_load_before": list(self.owner_load_before),
            "exec_load_after": list(self.exec_load_after),
            "selected_rows_total": self.selected_rows_total,
            "remote_rows_total": self.remote_rows_total,
            "modeled_max_load_reduction_pct": self.modeled_max_load_reduction_pct,
            "hot_experts": [
                {
                    "global_expert": item.global_expert,
                    "owner_rank": item.owner_rank,
                    "local_expert": item.local_expert,
                    "source_rank_counts": list(item.source_rank_counts),
                    "rows_total": item.rows_total,
                    "remote_rows": item.remote_rows,
                    "owner_slot": item.owner_slot,
                    "owner_shard_offset": item.owner_shard_offset,
                    "owner_compact_offset": item.owner_compact_offset,
                }
                for item in self.hot_experts
            ],
        }


@dataclass(frozen=True)
class BalancedMoeSourcePartition:
    """Source-rank route partition for native balanced-MoE execution."""

    keep_flat_mask: torch.Tensor
    remote_hot_mask: torch.Tensor
    normal_route_mask: torch.Tensor
    remote_flat_positions: torch.Tensor
    remote_hot_offsets: torch.Tensor
    remote_owner_shard_offsets: torch.Tensor
    remote_owner_compact_offsets: torch.Tensor
    remote_token_indices: torch.Tensor
    remote_group_ends: torch.Tensor | None = None
    remote_offsets_presorted_by: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return the tensor layout contract consumed by framework adapters."""

        return {
            "keep_flat_mask": self.keep_flat_mask,
            "remote_hot_mask": self.remote_hot_mask,
            "normal_route_mask": self.normal_route_mask,
            "remote_flat_positions": self.remote_flat_positions,
            "remote_hot_offsets": self.remote_hot_offsets,
            "remote_owner_shard_offsets": self.remote_owner_shard_offsets,
            "remote_owner_compact_offsets": self.remote_owner_compact_offsets,
            "remote_token_indices": self.remote_token_indices,
            "remote_group_ends": self.remote_group_ends,
            "remote_offsets_presorted_by": self.remote_offsets_presorted_by,
        }


@dataclass(frozen=True)
class BalancedMoeOwnerCompactExchangePlan:
    """Rank-local hot-weight movement plan in owner-compact row order."""

    rank: int
    world_size: int
    active_owner_compact_count: int
    needed_owner_compact_offsets: tuple[int, ...]
    compact_to_needed_index: tuple[int, ...]
    send_owner_compact_offsets_by_rank: tuple[tuple[int, ...], ...]
    recv_owner_compact_offsets_by_rank: tuple[tuple[int, ...], ...]
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    max_needed_count: int
    needed_density: float

    @property
    def send_owner_compact_offsets(self) -> tuple[int, ...]:
        return tuple(
            offset
            for per_rank in self.send_owner_compact_offsets_by_rank
            for offset in per_rank
        )

    @property
    def recv_owner_compact_offsets(self) -> tuple[int, ...]:
        return tuple(
            offset
            for per_rank in self.recv_owner_compact_offsets_by_rank
            for offset in per_rank
        )

    @property
    def recv_to_needed_index(self) -> tuple[int, ...]:
        return tuple(
            self.compact_to_needed_index[int(offset)]
            for offset in self.recv_owner_compact_offsets
        )

    @property
    def grad_send_needed_indices_by_rank(self) -> tuple[tuple[int, ...], ...]:
        """Backward send rows in peer-rank order.

        Forward receives needed hot weights from their owners.  Backward sends
        the corresponding partial dW rows back to those owners.
        """

        return tuple(
            tuple(
                self.compact_to_needed_index[int(offset)]
                for offset in per_rank
            )
            for per_rank in self.recv_owner_compact_offsets_by_rank
        )

    @property
    def grad_send_needed_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for per_rank in self.grad_send_needed_indices_by_rank
            for index in per_rank
        )

    @property
    def grad_recv_owner_compact_offsets(self) -> tuple[int, ...]:
        return self.send_owner_compact_offsets

    @property
    def input_split_offsets(self) -> tuple[int, ...]:
        return _exclusive_prefix_offsets(self.input_splits)

    @property
    def output_split_offsets(self) -> tuple[int, ...]:
        return _exclusive_prefix_offsets(self.output_splits)

    @property
    def grad_input_split_offsets(self) -> tuple[int, ...]:
        return self.output_split_offsets

    @property
    def grad_output_split_offsets(self) -> tuple[int, ...]:
        return self.input_split_offsets

    def as_tensors(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.int64,
        split_dtype: torch.dtype = torch.int64,
    ) -> dict[str, object]:
        """Return small tensor metadata for owner-compact row movement."""

        return {
            "needed_owner_compact_offsets": torch.tensor(
                self.needed_owner_compact_offsets,
                dtype=dtype,
                device=device,
            ),
            "compact_to_needed_index": torch.tensor(
                self.compact_to_needed_index,
                dtype=dtype,
                device=device,
            ),
            "send_owner_compact_offsets": torch.tensor(
                self.send_owner_compact_offsets,
                dtype=dtype,
                device=device,
            ),
            "recv_owner_compact_offsets": torch.tensor(
                self.recv_owner_compact_offsets,
                dtype=dtype,
                device=device,
            ),
            "recv_to_needed_index": torch.tensor(
                self.recv_to_needed_index,
                dtype=dtype,
                device=device,
            ),
            "grad_send_needed_indices": torch.tensor(
                self.grad_send_needed_indices,
                dtype=dtype,
                device=device,
            ),
            "grad_recv_owner_compact_offsets": torch.tensor(
                self.grad_recv_owner_compact_offsets,
                dtype=dtype,
                device=device,
            ),
            "input_splits": torch.tensor(
                self.input_splits,
                dtype=split_dtype,
                device=device,
            ),
            "output_splits": torch.tensor(
                self.output_splits,
                dtype=split_dtype,
                device=device,
            ),
            "input_split_offsets": torch.tensor(
                self.input_split_offsets,
                dtype=split_dtype,
                device=device,
            ),
            "output_split_offsets": torch.tensor(
                self.output_split_offsets,
                dtype=split_dtype,
                device=device,
            ),
            "grad_input_split_offsets": torch.tensor(
                self.grad_input_split_offsets,
                dtype=split_dtype,
                device=device,
            ),
            "grad_output_split_offsets": torch.tensor(
                self.grad_output_split_offsets,
                dtype=split_dtype,
                device=device,
            ),
            "needed_row_count": int(len(self.needed_owner_compact_offsets)),
            "input_splits_tuple": self.input_splits,
            "output_splits_tuple": self.output_splits,
            "input_split_offsets_tuple": self.input_split_offsets,
            "output_split_offsets_tuple": self.output_split_offsets,
            "grad_input_split_offsets_tuple": self.grad_input_split_offsets,
            "grad_output_split_offsets_tuple": self.grad_output_split_offsets,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": int(self.rank),
            "world_size": int(self.world_size),
            "active_owner_compact_count": int(self.active_owner_compact_count),
            "needed_owner_compact_offsets": [
                int(v) for v in self.needed_owner_compact_offsets
            ],
            "compact_to_needed_index": [
                int(v) for v in self.compact_to_needed_index
            ],
            "send_owner_compact_offsets_by_rank": [
                [int(v) for v in row]
                for row in self.send_owner_compact_offsets_by_rank
            ],
            "recv_owner_compact_offsets_by_rank": [
                [int(v) for v in row]
                for row in self.recv_owner_compact_offsets_by_rank
            ],
            "send_owner_compact_offsets": [
                int(v) for v in self.send_owner_compact_offsets
            ],
            "recv_owner_compact_offsets": [
                int(v) for v in self.recv_owner_compact_offsets
            ],
            "recv_to_needed_index": [
                int(v) for v in self.recv_to_needed_index
            ],
            "grad_send_needed_indices_by_rank": [
                [int(v) for v in row]
                for row in self.grad_send_needed_indices_by_rank
            ],
            "grad_send_needed_indices": [
                int(v) for v in self.grad_send_needed_indices
            ],
            "grad_recv_owner_compact_offsets": [
                int(v) for v in self.grad_recv_owner_compact_offsets
            ],
            "input_splits": [int(v) for v in self.input_splits],
            "output_splits": [int(v) for v in self.output_splits],
            "input_split_offsets": [int(v) for v in self.input_split_offsets],
            "output_split_offsets": [int(v) for v in self.output_split_offsets],
            "grad_input_split_offsets": [
                int(v) for v in self.grad_input_split_offsets
            ],
            "grad_output_split_offsets": [
                int(v) for v in self.grad_output_split_offsets
            ],
            "max_needed_count": int(self.max_needed_count),
            "needed_density": float(self.needed_density),
        }


@dataclass(frozen=True)
class BalancedMoeRuntimeLayout:
    """Combined source partition and owner-compact exchange runtime metadata."""

    source_partition: BalancedMoeSourcePartition | Mapping[str, object]
    owner_compact_exchange_plan: BalancedMoeOwnerCompactExchangePlan
    owner_compact_exchange_runtime_plan: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        """Return the complete runtime ABI for native balanced-MoE execution."""

        return {
            "source_partition": self.source_partition.as_dict(),
            "owner_compact_exchange_plan": self.owner_compact_exchange_plan.as_dict(),
            "owner_compact_exchange_runtime_plan": dict(
                self.owner_compact_exchange_runtime_plan
            ),
        }


@dataclass(frozen=True)
class BalancedMoeCompactDispatchOutput:
    """Backend-owned balanced-MoE dispatch result.

    Normal/cold routes are already in Primus-Turbo's expert-major raw top-k EP
    layout. Remote-hot helper rows are kept in the source partition's compact
    owner order, so framework code does not need to invent another hot-row ABI.
    """

    normal_expert_input: torch.Tensor
    normal_tokens_per_expert: torch.Tensor
    normal_expert_weights: torch.Tensor
    normal_state: BalancedMoeNormalTopKDispatchState
    hot_expert_input: torch.Tensor
    hot_expert_weights: torch.Tensor
    hot_owner_compact_offsets: torch.Tensor
    hot_token_indices: torch.Tensor
    hot_group_ends: torch.Tensor | None
    hot_offsets_presorted_by: str
    source_partition: BalancedMoeSourcePartition


@dataclass(frozen=True)
class BalancedMoeCompactCombineOutput:
    """Backend-owned balanced-MoE combine result."""

    combined: torch.Tensor
    normal_combined: torch.Tensor
    hot_token_indices: torch.Tensor
    hot_offsets_presorted_by: str


def _tensor_or_empty(
    value: torch.Tensor | Sequence[int] | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.empty(0, device=device, dtype=torch.long)
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.long)
    return torch.tensor(tuple(int(v) for v in value), device=device, dtype=torch.long)


def _split_tuple(value: torch.Tensor | Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return tuple(int(v) for v in value)


def _offset_tuple(
    value: torch.Tensor | Sequence[int] | None,
    splits: tuple[int, ...],
    *,
    name: str,
) -> tuple[int, ...]:
    offsets = _split_tuple(value)
    if not offsets:
        offsets = _exclusive_prefix_offsets(splits)
    if len(offsets) != len(splits) + 1:
        raise ValueError(
            f"{name} must have len(splits)+1 entries, got {len(offsets)} "
            f"for {len(splits)} splits"
        )
    if offsets[0] != 0:
        raise ValueError(f"{name} must start at 0")
    for peer, count in enumerate(splits):
        span = int(offsets[peer + 1]) - int(offsets[peer])
        if span != int(count):
            raise ValueError(
                f"{name} span for peer {peer} is {span}, expected split count {count}"
            )
    return offsets


def _row_count(value: torch.Tensor | Sequence[int] | None) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    return len(value)


def _plan_get(
    plan: BalancedMoeOwnerCompactExchangePlan | Mapping[str, object] | None,
    name: str,
    default: object = (),
) -> object:
    if plan is None:
        return default
    if isinstance(plan, BalancedMoeOwnerCompactExchangePlan):
        return getattr(plan, name, default)
    if isinstance(plan, Mapping):
        return plan.get(name, default)
    return getattr(plan, name, default)


def build_normal_topk_dispatch_tensors(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    *,
    partition: BalancedMoeSourcePartition | Mapping[str, object] | None = None,
    normal_route_mask: torch.Tensor | None = None,
    masked_expert_id: int = -1,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Mask helper-executed routes out of raw top-k dispatch tensors.

    Primus-Turbo's DeepEP/TurboEP dispatcher consumes token-major
    ``[num_tokens, top_k]`` expert ids and weights.  A balanced-MoE caller
    should first split remote-hot routes into the helper path, then pass only
    the remaining normal/cold routes to the raw dispatcher.  This helper makes
    that tensor contract backend-owned: remote-hot slots become ``-1`` and
    their weights become zero, while the original shape is preserved.
    """

    if not isinstance(topk_ids, torch.Tensor):
        raise TypeError("topk_ids must be a torch.Tensor")
    if partition is not None:
        normal_route_mask = _plan_get(
            partition,
            "normal_route_mask",
            normal_route_mask,
        )
    if normal_route_mask is None:
        raise ValueError("normal_route_mask or partition is required")
    if not isinstance(normal_route_mask, torch.Tensor):
        normal_route_mask = torch.as_tensor(normal_route_mask)
    if tuple(normal_route_mask.shape) != tuple(topk_ids.shape):
        raise ValueError(
            "normal_route_mask must have the same shape as topk_ids, got "
            f"{tuple(normal_route_mask.shape)} vs {tuple(topk_ids.shape)}"
        )

    id_mask = normal_route_mask.to(device=topk_ids.device, dtype=torch.bool)
    normal_topk_ids = torch.where(
        id_mask,
        topk_ids,
        topk_ids.new_full(topk_ids.shape, int(masked_expert_id)),
    )
    if topk_weights is None:
        return normal_topk_ids, None
    if not isinstance(topk_weights, torch.Tensor):
        raise TypeError("topk_weights must be a torch.Tensor when provided")
    if tuple(topk_weights.shape) != tuple(topk_ids.shape):
        raise ValueError(
            "topk_weights must have the same shape as topk_ids, got "
            f"{tuple(topk_weights.shape)} vs {tuple(topk_ids.shape)}"
        )
    weight_mask = normal_route_mask.to(
        device=topk_weights.device,
        dtype=torch.bool,
    )
    normal_topk_weights = torch.where(
        weight_mask,
        topk_weights,
        topk_weights.new_zeros(topk_weights.shape),
    )
    return normal_topk_ids, normal_topk_weights


@dataclass(frozen=True)
class BalancedMoeNormalTopKDispatchState:
    """State returned by the Primus-Turbo normal/cold raw top-k EP path."""

    recv_topk_ids: torch.Tensor | None
    recv_topk_weights: torch.Tensor | None
    tokens_per_expert: torch.Tensor
    handle: object
    num_experts: int
    async_finish: bool = False
    allocate_on_comm_stream: bool = False
    row_id_map: torch.Tensor | None = None
    num_dispatched_token_tensor: torch.Tensor | None = None
    hidden_shape_before_permute: tuple[int, ...] | None = None
    num_local_experts: int | None = None
    num_topk: int | None = None
    pad_multiple: int = 0


def dispatch_normal_topk_tokens(
    x: torch.Tensor,
    normal_topk_ids: torch.Tensor,
    normal_topk_weights: torch.Tensor,
    *,
    num_experts: int,
    group,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
    num_worst_tokens: int = 0,
) -> tuple[torch.Tensor, BalancedMoeNormalTopKDispatchState]:
    """Dispatch normal/cold rows through Primus-Turbo's raw top-k EP backend.

    Balanced-MoE policy and hot-helper row selection stay outside this helper.
    The backend contract starts after remote-hot rows have been masked with
    :func:`build_normal_topk_dispatch_tensors`, preserving the token-major
    ``[tokens, top_k]`` ABI expected by TurboEP.
    """

    if normal_topk_weights is None:
        raise ValueError("normal_topk_weights is required for TurboEP dispatch")
    if tuple(normal_topk_ids.shape) != tuple(normal_topk_weights.shape):
        raise ValueError("normal_topk_ids and normal_topk_weights must align")
    if normal_topk_ids.dtype != torch.int64:
        normal_topk_ids = normal_topk_ids.to(torch.int64)
    if normal_topk_weights.dtype != torch.float32:
        normal_topk_weights = normal_topk_weights.float()

    from primus_turbo.pytorch.ops.moe.moe_dispatch_combine import moe_dispatch

    (
        recv_x,
        recv_topk_ids,
        recv_topk_weights,
        tokens_per_expert,
        handle,
    ) = moe_dispatch(
        x,
        token_indices=normal_topk_ids,
        token_probs=normal_topk_weights,
        num_experts=int(num_experts),
        group=group,
        async_finish=bool(async_finish),
        allocate_on_comm_stream=bool(allocate_on_comm_stream),
        num_worst_tokens=int(num_worst_tokens),
    )
    return recv_x, BalancedMoeNormalTopKDispatchState(
        recv_topk_ids=recv_topk_ids,
        recv_topk_weights=recv_topk_weights,
        tokens_per_expert=tokens_per_expert,
        handle=handle,
        num_experts=int(num_experts),
        async_finish=bool(async_finish),
        allocate_on_comm_stream=bool(allocate_on_comm_stream),
    )


def combine_normal_topk_tokens(
    expert_output: torch.Tensor,
    state: BalancedMoeNormalTopKDispatchState,
    *,
    group,
    async_finish: bool | None = None,
    allocate_on_comm_stream: bool | None = None,
) -> torch.Tensor:
    """Combine outputs for :func:`dispatch_normal_topk_tokens`."""

    from primus_turbo.pytorch.ops.moe.moe_dispatch_combine import moe_combine

    return moe_combine(
        expert_output,
        group,
        state.handle,
        async_finish=(
            state.async_finish if async_finish is None else bool(async_finish)
        ),
        allocate_on_comm_stream=(
            state.allocate_on_comm_stream
            if allocate_on_comm_stream is None
            else bool(allocate_on_comm_stream)
        ),
    )


def dispatch_permute_normal_topk_tokens(
    x: torch.Tensor,
    normal_topk_ids: torch.Tensor,
    normal_topk_weights: torch.Tensor,
    *,
    num_experts: int,
    num_local_experts: int,
    group,
    num_topk: int | None = None,
    pad_multiple: int = 0,
    num_permuted_tokens: int = -1,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
    num_worst_tokens: int = 0,
    use_cuda_num_tokens_per_expert: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BalancedMoeNormalTopKDispatchState]:
    """Dispatch normal/cold top-k routes and return expert-major rows.

    This mirrors :class:`DeepEPTokenDispatcher`'s internal sequence:
    ``moe_dispatch -> moe_permute``.  Keeping it in Primus-Turbo lets
    TorchTitan treat the normal/cold EP path as a backend-owned ABI instead of
    reconstructing Primus-specific row maps around a generic dispatch output.
    """

    recv_x, state = dispatch_normal_topk_tokens(
        x,
        normal_topk_ids,
        normal_topk_weights,
        num_experts=num_experts,
        group=group,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
        num_worst_tokens=num_worst_tokens,
    )
    if state.recv_topk_ids is None or state.recv_topk_weights is None:
        raise RuntimeError("Primus-Turbo dispatch did not return top-k metadata")
    if num_topk is None:
        num_topk = int(state.recv_topk_ids.shape[-1])

    import primus_turbo.pytorch as turbo

    hidden_shape_before_permute = tuple(recv_x.shape)
    (
        permuted_x,
        row_id_map,
        tokens_per_expert,
        _,
        num_dispatched_token_tensor,
        _,
        permuted_weights,
    ) = turbo.ops.moe_permute(
        recv_x,
        state.recv_topk_ids,
        num_local_experts=int(num_local_experts),
        num_topk=int(num_topk),
        pad_multiple=int(pad_multiple),
        num_permuted_tokens=int(num_permuted_tokens),
        probs=state.recv_topk_weights,
        probs_layout="topk",
    )
    if not use_cuda_num_tokens_per_expert:
        tokens_per_expert = tokens_per_expert.cpu()

    return (
        permuted_x,
        tokens_per_expert,
        permuted_weights,
        BalancedMoeNormalTopKDispatchState(
            recv_topk_ids=state.recv_topk_ids,
            recv_topk_weights=state.recv_topk_weights,
            tokens_per_expert=tokens_per_expert,
            handle=state.handle,
            num_experts=state.num_experts,
            async_finish=state.async_finish,
            allocate_on_comm_stream=state.allocate_on_comm_stream,
            row_id_map=row_id_map,
            num_dispatched_token_tensor=num_dispatched_token_tensor,
            hidden_shape_before_permute=hidden_shape_before_permute,
            num_local_experts=int(num_local_experts),
            num_topk=int(num_topk),
            pad_multiple=int(pad_multiple),
        ),
    )


def unpermute_combine_normal_topk_tokens(
    expert_output: torch.Tensor,
    state: BalancedMoeNormalTopKDispatchState,
    *,
    group,
    async_finish: bool | None = None,
    allocate_on_comm_stream: bool | None = None,
) -> torch.Tensor:
    """Invert :func:`dispatch_permute_normal_topk_tokens` and combine rows."""

    if state.row_id_map is None:
        raise ValueError("state.row_id_map is required for normal top-k unpermute")
    if state.num_dispatched_token_tensor is None:
        raise ValueError(
            "state.num_dispatched_token_tensor is required for normal top-k unpermute"
        )
    if state.hidden_shape_before_permute is None:
        raise ValueError(
            "state.hidden_shape_before_permute is required for normal top-k unpermute"
        )
    if state.num_local_experts is None:
        raise ValueError("state.num_local_experts is required for normal top-k unpermute")

    import primus_turbo.pytorch as turbo

    unpermuted_x, _ = turbo.ops.moe_unpermute(
        expert_output,
        state.row_id_map,
        state.num_dispatched_token_tensor,
        restore_shape=state.hidden_shape_before_permute,
        num_local_experts=int(state.num_local_experts),
        pad_multiple=int(state.pad_multiple),
    )
    return combine_normal_topk_tokens(
        unpermuted_x,
        state,
        group=group,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )


def _resolve_source_partition(
    *,
    layout: BalancedMoeRuntimeLayout | Mapping[str, object] | None = None,
    partition: BalancedMoeSourcePartition | Mapping[str, object] | None = None,
) -> BalancedMoeSourcePartition | Mapping[str, object]:
    if partition is not None:
        return partition
    if layout is None:
        raise ValueError("layout or partition is required")
    if isinstance(layout, BalancedMoeRuntimeLayout):
        return layout.source_partition
    if isinstance(layout, Mapping):
        source_partition = layout.get("source_partition")
        if source_partition is not None:
            return source_partition
    source_partition = getattr(layout, "source_partition", None)
    if source_partition is not None:
        return source_partition
    raise ValueError("layout does not contain a source_partition")


def dispatch_balanced_moe_compact(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    num_local_experts: int,
    group,
    layout: BalancedMoeRuntimeLayout | Mapping[str, object] | None = None,
    partition: BalancedMoeSourcePartition | Mapping[str, object] | None = None,
    pad_multiple: int = 0,
    num_permuted_tokens: int = -1,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
    num_worst_tokens: int = 0,
    use_cuda_num_tokens_per_expert: bool = False,
) -> BalancedMoeCompactDispatchOutput:
    """Dispatch normal/cold routes and expose compact helper-hot rows.

    Policy stays in the caller: this function consumes an already-built
    balanced-MoE source partition.  The normal/cold side is sent through
    Primus-Turbo's raw top-k EP dispatch+permute path, while helper-hot rows
    are gathered in the partition's compact row order for grouped helper MLP.
    """

    source_partition = _resolve_source_partition(
        layout=layout,
        partition=partition,
    )
    normal_topk_ids, normal_topk_weights = build_normal_topk_dispatch_tensors(
        topk_ids,
        topk_weights,
        partition=source_partition,
    )
    (
        normal_expert_input,
        normal_tokens_per_expert,
        normal_expert_weights,
        normal_state,
    ) = dispatch_permute_normal_topk_tokens(
        x,
        normal_topk_ids,
        normal_topk_weights,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        group=group,
        num_topk=int(topk_ids.shape[-1]),
        pad_multiple=pad_multiple,
        num_permuted_tokens=num_permuted_tokens,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
        num_worst_tokens=num_worst_tokens,
        use_cuda_num_tokens_per_expert=use_cuda_num_tokens_per_expert,
    )

    remote_token_indices = _plan_get(
        source_partition,
        "remote_token_indices",
        None,
    )
    remote_flat_positions = _plan_get(
        source_partition,
        "remote_flat_positions",
        None,
    )
    remote_owner_compact_offsets = _plan_get(
        source_partition,
        "remote_owner_compact_offsets",
        None,
    )
    if remote_token_indices is None or remote_flat_positions is None:
        raise ValueError("source partition must include remote hot row indices")

    hot_token_indices = _tensor_or_empty(
        remote_token_indices,
        device=x.device,
    ).to(torch.long)
    hot_flat_positions = _tensor_or_empty(
        remote_flat_positions,
        device=topk_weights.device,
    ).to(torch.long)
    hot_owner_compact_offsets = _tensor_or_empty(
        remote_owner_compact_offsets,
        device=x.device,
    ).to(torch.long)

    hot_expert_input = x.index_select(0, hot_token_indices)
    hot_expert_weights = topk_weights.reshape(-1).index_select(0, hot_flat_positions)

    return BalancedMoeCompactDispatchOutput(
        normal_expert_input=normal_expert_input,
        normal_tokens_per_expert=normal_tokens_per_expert,
        normal_expert_weights=normal_expert_weights,
        normal_state=normal_state,
        hot_expert_input=hot_expert_input,
        hot_expert_weights=hot_expert_weights,
        hot_owner_compact_offsets=hot_owner_compact_offsets,
        hot_token_indices=hot_token_indices,
        hot_group_ends=_plan_get(source_partition, "remote_group_ends", None),
        hot_offsets_presorted_by=str(
            _plan_get(source_partition, "remote_offsets_presorted_by", "")
        ),
        source_partition=source_partition,
    )


def combine_balanced_moe_compact(
    normal_expert_output: torch.Tensor,
    dispatch_output: BalancedMoeCompactDispatchOutput,
    *,
    group,
    hot_expert_output: torch.Tensor | None = None,
    hot_expert_weights: torch.Tensor | None = None,
    hot_token_indices: torch.Tensor | None = None,
    async_finish: bool | None = None,
    allocate_on_comm_stream: bool | None = None,
) -> BalancedMoeCompactCombineOutput:
    """Combine normal/cold output and helper-hot output in token order."""

    normal_combined = unpermute_combine_normal_topk_tokens(
        normal_expert_output,
        dispatch_output.normal_state,
        group=group,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    combined = normal_combined

    if hot_expert_output is not None:
        if hot_expert_weights is None:
            hot_expert_weights = dispatch_output.hot_expert_weights
        if hot_token_indices is None:
            hot_token_indices = dispatch_output.hot_token_indices
        if int(hot_expert_output.shape[0]) != int(hot_expert_weights.numel()):
            raise ValueError("hot_expert_output rows must match hot weights")
        if int(hot_expert_output.shape[0]) != int(hot_token_indices.numel()):
            raise ValueError("hot_expert_output rows must match hot token ids")

        view_shape = (int(hot_expert_weights.numel()),) + (
            1,
        ) * (hot_expert_output.dim() - 1)
        weighted_hot = hot_expert_output * hot_expert_weights.to(
            dtype=hot_expert_output.dtype,
            device=hot_expert_output.device,
        ).reshape(view_shape)
        combined = normal_combined.clone()
        combined.index_add_(
            0,
            hot_token_indices.to(device=combined.device, dtype=torch.long),
            weighted_hot.to(dtype=combined.dtype, device=combined.device),
        )

    return BalancedMoeCompactCombineOutput(
        combined=combined,
        normal_combined=normal_combined,
        hot_token_indices=dispatch_output.hot_token_indices,
        hot_offsets_presorted_by=dispatch_output.hot_offsets_presorted_by,
    )


def _dist_all_to_all_single(
    recv: torch.Tensor,
    send: torch.Tensor,
    *,
    output_splits: tuple[int, ...],
    input_splits: tuple[int, ...],
    group,
) -> None:
    if len(input_splits) != len(output_splits):
        raise ValueError("input_splits and output_splits must have the same length")
    if (
        len(input_splits) == 1
        and (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        )
    ):
        if send.numel() != recv.numel():
            raise RuntimeError("single-rank exchange send/recv sizes differ")
        recv.copy_(send.reshape_as(recv))
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized for owner-compact exchange"
        )
    torch.distributed.all_to_all_single(
        recv,
        send,
        output_split_sizes=list(output_splits),
        input_split_sizes=list(input_splits),
        group=group,
    )


def _normalize_owner_compact_exchange_transport(transport: str | None) -> str:
    if transport is None:
        return OWNER_COMPACT_TORCH_A2A_TRANSPORT
    normalized = str(transport).strip().lower()
    if normalized in {"", "auto", "torch", "all_to_all_single"}:
        return OWNER_COMPACT_TORCH_A2A_TRANSPORT
    if normalized == OWNER_COMPACT_TORCH_A2A_TRANSPORT:
        return OWNER_COMPACT_TORCH_A2A_TRANSPORT
    if normalized in {
        "native",
        "native_hot_helper",
        "mori_sdma",
        "mori_sdma_padded",
        "mori_sdma_padded_all2all",
        "mori_native",
        OWNER_COMPACT_MORI_SDMA_PADDED_A2A_TRANSPORT,
    }:
        return OWNER_COMPACT_MORI_SDMA_PADDED_A2A_TRANSPORT
    if normalized in {
        OWNER_COMPACT_PRIMUS_TURBO_DISPATCH_TRANSPORT,
    }:
        return OWNER_COMPACT_PRIMUS_TURBO_DISPATCH_TRANSPORT
    raise ValueError(
        "unsupported owner-compact hot-row transport "
        f"{transport!r}; Primus-Turbo balanced_moe currently supports "
        f"{BALANCED_MOE_BACKEND_CAPABILITIES['owner_compact_exchange_transports']}"
    )


def _compact_rows_group(op):
    if op is None:
        return None
    if isinstance(op, Mapping):
        return op.get("group")
    return getattr(op, "group", None)


def _compact_rows_use_reference(op) -> bool:
    if op is None:
        return True
    if isinstance(op, str):
        normalized = op.strip().lower()
        return normalized in {
            "",
            "auto",
            "torch",
            "reference",
            "torch_reference",
            BALANCED_MOE_COMPACT_ROWS_TORCH_REFERENCE_BACKEND,
        }
    if isinstance(op, Mapping):
        backend = op.get("backend", op.get("transport"))
        if backend is not None:
            return _compact_rows_use_reference(str(backend))
    return False


def _ensure_1d_long_tensor(
    value: torch.Tensor | None,
    *,
    device: torch.device,
    name: str,
    allow_none: bool = False,
) -> torch.Tensor | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} is required")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dim() != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(value.shape)}")
    return value.to(device=device, dtype=torch.long)


def _counts_rank_major_tensor(
    recv_counts_rank_major: torch.Tensor | None,
    *,
    world_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    counts = _ensure_1d_long_tensor(
        recv_counts_rank_major,
        device=device,
        name=name,
    )
    assert counts is not None
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if int(counts.numel()) % world_size != 0:
        raise ValueError(
            f"{name} length {int(counts.numel())} is not divisible by "
            f"world_size {world_size}"
        )
    return counts.contiguous().view(world_size, -1)


def _rank_major_to_expert_major_indices(counts_rank_major: torch.Tensor) -> torch.Tensor:
    """Build the compact row permutation from rank-major to expert-major order."""

    if counts_rank_major.dim() != 2:
        raise ValueError(
            "counts_rank_major must have shape [world_size, num_local_experts]"
        )
    device = counts_rank_major.device
    counts = counts_rank_major.to(dtype=torch.long, device=device).contiguous()
    world_size, num_local_experts = (int(counts.shape[0]), int(counts.shape[1]))
    flat_counts = counts.reshape(-1)
    starts = torch.empty_like(flat_counts)
    if flat_counts.numel() > 0:
        starts[0] = 0
        if flat_counts.numel() > 1:
            starts[1:] = torch.cumsum(flat_counts[:-1], dim=0)

    pieces: list[torch.Tensor] = []
    for expert_idx in range(num_local_experts):
        for rank_idx in range(world_size):
            flat_idx = int(rank_idx * num_local_experts + expert_idx)
            count = int(flat_counts[flat_idx].item())
            if count <= 0:
                continue
            start = int(starts[flat_idx].item())
            pieces.append(
                torch.arange(
                    start,
                    start + count,
                    device=device,
                    dtype=torch.long,
                )
            )
    if not pieces:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.cat(pieces, dim=0)


def _expert_major_to_rank_major(
    expert_major: torch.Tensor,
    rank_to_expert_indices: torch.Tensor,
) -> torch.Tensor:
    if int(expert_major.shape[0]) != int(rank_to_expert_indices.numel()):
        raise ValueError(
            "expert-major row count must match rank/expert permutation length, "
            f"got {int(expert_major.shape[0])} and "
            f"{int(rank_to_expert_indices.numel())}"
        )
    rank_major = torch.empty_like(expert_major)
    if rank_to_expert_indices.numel() > 0:
        rank_major.index_copy_(0, rank_to_expert_indices, expert_major)
    return rank_major


def _reference_dispatch_balanced_moe_compact_rows(
    op,
    local_rows: torch.Tensor,
    local_flat_positions: torch.Tensor | None,
    local_num_tokens_per_expert: torch.Tensor,
    recv_counts_rank_major: torch.Tensor,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    num_output_rows: int,
    *,
    return_flat_positions: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Torch reference for the backend-owned compact-row dispatch ABI."""

    device = local_rows.device
    input_splits_t = _split_tuple(input_splits)
    output_splits_t = _split_tuple(output_splits)
    expected_send = int(sum(input_splits_t))
    expected_recv = int(sum(output_splits_t))
    if int(local_rows.shape[0]) != expected_send:
        raise ValueError(
            f"local_rows has {int(local_rows.shape[0])} rows, expected "
            f"input_splits total {expected_send}"
        )
    if int(num_output_rows) != expected_recv:
        raise ValueError(
            f"num_output_rows={int(num_output_rows)} does not match "
            f"output_splits total {expected_recv}"
        )

    local_counts = _ensure_1d_long_tensor(
        local_num_tokens_per_expert,
        device=device,
        name="local_num_tokens_per_expert",
    )
    assert local_counts is not None
    if int(local_counts.sum().item()) != expected_send:
        raise ValueError(
            "local_num_tokens_per_expert must sum to local send rows, got "
            f"{int(local_counts.sum().item())} and {expected_send}"
        )

    rank_major_rows = local_rows.new_empty((expected_recv, *local_rows.shape[1:]))
    _dist_all_to_all_single(
        rank_major_rows,
        local_rows.contiguous(),
        output_splits=output_splits_t,
        input_splits=input_splits_t,
        group=_compact_rows_group(op),
    )

    counts_rank_major = _counts_rank_major_tensor(
        recv_counts_rank_major,
        world_size=len(output_splits_t),
        device=device,
        name="recv_counts_rank_major",
    )
    if int(counts_rank_major.sum().item()) != expected_recv:
        raise ValueError(
            "recv_counts_rank_major must sum to received rows, got "
            f"{int(counts_rank_major.sum().item())} and {expected_recv}"
        )
    rank_to_expert_indices = _rank_major_to_expert_major_indices(counts_rank_major)
    expert_major_rows = rank_major_rows.index_select(0, rank_to_expert_indices)

    expert_major_positions = None
    if return_flat_positions:
        local_positions = _ensure_1d_long_tensor(
            local_flat_positions,
            device=device,
            name="local_flat_positions",
        )
        assert local_positions is not None
        if int(local_positions.numel()) != expected_send:
            raise ValueError(
                "local_flat_positions must match local send rows, got "
                f"{int(local_positions.numel())} and {expected_send}"
            )
        rank_major_positions = local_positions.new_empty((expected_recv,))
        _dist_all_to_all_single(
            rank_major_positions,
            local_positions.contiguous(),
            output_splits=output_splits_t,
            input_splits=input_splits_t,
            group=_compact_rows_group(op),
        )
        expert_major_positions = rank_major_positions.index_select(
            0,
            rank_to_expert_indices,
        )
    return expert_major_rows, expert_major_positions


def _reference_combine_balanced_moe_compact_rows(
    op,
    expert_major_rows: torch.Tensor,
    expert_major_flat_positions: torch.Tensor | None,
    expert_major_to_rank_major_indices: torch.Tensor | None,
    recv_counts_rank_major: torch.Tensor | None,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    num_output_rows: int,
    *,
    return_flat_positions: bool,
    top_scores_flat: torch.Tensor | None,
    top_k: int,
    flat_position_offset: int,
    token_output_rows: int,
    return_token_output: bool,
):
    """Torch reference for the backend-owned compact-row combine ABI."""

    device = expert_major_rows.device
    input_splits_t = _split_tuple(input_splits)
    output_splits_t = _split_tuple(output_splits)
    expected_send = int(sum(input_splits_t))
    expected_recv = int(sum(output_splits_t))
    if int(expert_major_rows.shape[0]) != expected_send:
        raise ValueError(
            f"expert_major_rows has {int(expert_major_rows.shape[0])} rows, "
            f"expected input_splits total {expected_send}"
        )
    if int(num_output_rows) != expected_recv:
        raise ValueError(
            f"num_output_rows={int(num_output_rows)} does not match "
            f"output_splits total {expected_recv}"
        )

    if expert_major_to_rank_major_indices is not None:
        rank_indices = _ensure_1d_long_tensor(
            expert_major_to_rank_major_indices,
            device=device,
            name="expert_major_to_rank_major_indices",
        )
        assert rank_indices is not None
        rank_major_rows = expert_major_rows.index_select(0, rank_indices)
        rank_major_positions = (
            expert_major_flat_positions.to(device=device, dtype=torch.long).index_select(
                0,
                rank_indices,
            )
            if expert_major_flat_positions is not None
            else None
        )
    else:
        counts_rank_major = _counts_rank_major_tensor(
            recv_counts_rank_major,
            world_size=len(input_splits_t),
            device=device,
            name="recv_counts_rank_major",
        )
        if int(counts_rank_major.sum().item()) != expected_send:
            raise ValueError(
                "recv_counts_rank_major must sum to combine send rows, got "
                f"{int(counts_rank_major.sum().item())} and {expected_send}"
            )
        rank_to_expert_indices = _rank_major_to_expert_major_indices(
            counts_rank_major
        )
        rank_major_rows = _expert_major_to_rank_major(
            expert_major_rows,
            rank_to_expert_indices,
        )
        rank_major_positions = (
            _expert_major_to_rank_major(
                expert_major_flat_positions.to(device=device, dtype=torch.long),
                rank_to_expert_indices,
            )
            if expert_major_flat_positions is not None
            else None
        )

    source_rank_rows = expert_major_rows.new_empty(
        (expected_recv, *expert_major_rows.shape[1:])
    )
    _dist_all_to_all_single(
        source_rank_rows,
        rank_major_rows.contiguous(),
        output_splits=output_splits_t,
        input_splits=input_splits_t,
        group=_compact_rows_group(op),
    )

    source_rank_flat_positions = None
    if return_flat_positions or return_token_output:
        if rank_major_positions is None:
            raise ValueError(
                "expert_major_flat_positions is required when returning "
                "flat positions or token output"
            )
        source_rank_flat_positions = rank_major_positions.new_empty((expected_recv,))
        _dist_all_to_all_single(
            source_rank_flat_positions,
            rank_major_positions.contiguous(),
            output_splits=output_splits_t,
            input_splits=input_splits_t,
            group=_compact_rows_group(op),
        )

    if return_token_output:
        if source_rank_flat_positions is None:
            raise AssertionError("source positions must be available for token output")
        if top_scores_flat is None:
            raise ValueError("top_scores_flat is required for token output")
        if int(top_k) <= 0:
            raise ValueError("top_k must be positive for token output")
        if int(token_output_rows) <= 0:
            raise ValueError("token_output_rows must be positive for token output")

        source_offsets = source_rank_flat_positions.to(device=device, dtype=torch.long)
        local_flat = source_offsets - int(flat_position_offset)
        if bool((local_flat < 0).any().item()):
            raise ValueError("source flat positions are below flat_position_offset")
        token_indices = torch.div(local_flat, int(top_k), rounding_mode="floor")
        if bool((token_indices >= int(token_output_rows)).any().item()):
            raise ValueError("source flat positions exceed token_output_rows")
        weights = top_scores_flat.to(device=device).reshape(-1).index_select(
            0,
            local_flat,
        )
        view_shape = (int(weights.numel()),) + (1,) * (source_rank_rows.dim() - 1)
        token_output = source_rank_rows.new_zeros(
            (int(token_output_rows), *source_rank_rows.shape[1:])
        )
        if source_rank_rows.numel() > 0:
            token_output.index_add_(
                0,
                token_indices,
                source_rank_rows * weights.to(source_rank_rows.dtype).reshape(view_shape),
            )
        return source_rank_rows, source_rank_flat_positions, token_output

    return source_rank_rows, source_rank_flat_positions


def _require_compact_rows_method(op, *names: str):
    """Return the first compact-row primitive exposed by ``op``.

    The module also provides a torch reference implementation when ``op`` is
    ``None``/``"torch_reference"``. That reference is useful for ABI parity and
    correctness, while a production run should still replace it with a real
    Primus-Turbo compact-row primitive before claiming a throughput promotion.
    """

    if _compact_rows_use_reference(op):
        return None
    for name in names:
        method = getattr(op, name, None)
        if method is not None:
            return method
    expected = ", ".join(f"{name}()" for name in names)
    raise TypeError(
        "Primus-Turbo balanced-MoE compact-row API requires a backend op "
        f"exposing one of: {expected}, or op=None/'torch_reference' for the "
        "backend-owned reference transport."
    )


def dispatch_balanced_moe_compact_rows(
    op,
    local_rows: torch.Tensor,
    local_flat_positions: torch.Tensor | None,
    local_num_tokens_per_expert: torch.Tensor,
    recv_counts_rank_major: torch.Tensor,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    num_output_rows: int,
    flat_position_rank_stride: int,
    *,
    block_num: int = -1,
    warp_per_block: int = -1,
    return_flat_positions: bool = True,
):
    """Dispatch already-compact balanced-MoE rows through a backend primitive.

    This is the same high-level ABI used by MORI's native compact path:
    policy/layout decisions happen above the backend, while the backend op owns
    movement of an already-compact row stream.  Primus-Turbo currently exposes
    the wrapper for ABI parity; capability flags intentionally remain false
    until a Primus op provides the method below and passes throughput gates.
    """

    method = _require_compact_rows_method(
        op,
        "dispatch_balanced_moe_compact_rows",
        "dispatch_standard_ep_compact_native",
    )
    if method is None:
        return _reference_dispatch_balanced_moe_compact_rows(
            op,
            local_rows,
            local_flat_positions,
            local_num_tokens_per_expert,
            recv_counts_rank_major,
            input_splits,
            output_splits,
            int(num_output_rows),
            return_flat_positions=return_flat_positions,
        )
    return method(
        local_rows,
        local_flat_positions,
        local_num_tokens_per_expert,
        recv_counts_rank_major,
        input_splits,
        output_splits,
        int(num_output_rows),
        int(flat_position_rank_stride),
        block_num=block_num,
        warp_per_block=warp_per_block,
        return_flat_positions=return_flat_positions,
    )


def combine_balanced_moe_compact_rows(
    op,
    expert_major_rows: torch.Tensor,
    expert_major_flat_positions: torch.Tensor | None,
    expert_major_to_rank_major_indices: torch.Tensor | None,
    recv_counts_rank_major: torch.Tensor | None,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    num_output_rows: int,
    *,
    block_num: int = -1,
    warp_per_block: int = -1,
    return_flat_positions: bool = True,
    top_scores_flat: torch.Tensor | None = None,
    top_k: int = 0,
    flat_position_offset: int = 0,
    token_output_rows: int = 0,
    return_token_output: bool = False,
):
    """Combine already-compact balanced-MoE rows through a backend primitive."""

    method = _require_compact_rows_method(
        op,
        "combine_balanced_moe_compact_rows",
        "combine_standard_ep_compact_native",
    )
    if method is None:
        return _reference_combine_balanced_moe_compact_rows(
            op,
            expert_major_rows,
            expert_major_flat_positions,
            expert_major_to_rank_major_indices,
            recv_counts_rank_major,
            input_splits,
            output_splits,
            int(num_output_rows),
            return_flat_positions=return_flat_positions,
            top_scores_flat=top_scores_flat,
            top_k=int(top_k),
            flat_position_offset=int(flat_position_offset),
            token_output_rows=int(token_output_rows),
            return_token_output=return_token_output,
        )
    return method(
        expert_major_rows,
        expert_major_flat_positions,
        expert_major_to_rank_major_indices,
        recv_counts_rank_major,
        input_splits,
        output_splits,
        int(num_output_rows),
        block_num=block_num,
        warp_per_block=warp_per_block,
        return_flat_positions=return_flat_positions,
        top_scores_flat=top_scores_flat,
        top_k=top_k,
        flat_position_offset=flat_position_offset,
        token_output_rows=token_output_rows,
        return_token_output=return_token_output,
    )


def _as_int_tuple(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def _exclusive_prefix_offsets(values: Sequence[int]) -> tuple[int, ...]:
    offsets = [0]
    total = 0
    for value in values:
        total += int(value)
        offsets.append(int(total))
    return tuple(offsets)


def _load_reduction_pct(before: Sequence[int], after: Sequence[int]) -> float:
    if not before:
        return 0.0
    before_max = max(int(v) for v in before)
    after_max = max(int(v) for v in after) if after else 0
    if before_max <= 0:
        return 0.0
    return float((before_max - after_max) / before_max * 100.0)


def build_balanced_moe_plan_from_global_counts(
    counts_owner_source_local: torch.Tensor,
    hot_expert_num: int,
    *,
    min_reduction_pct: float = 0.0,
) -> BalancedMoePlan:
    """Select hot experts with the MindSpeed-style greedy load model.

    Args:
        counts_owner_source_local: int tensor with shape
            ``[owner_rank, source_rank, local_expert]``.
        hot_expert_num: maximum number of hot experts to helper-execute.
        min_reduction_pct: reject the plan if modeled max-load reduction is
            below this threshold.
    """

    if not isinstance(counts_owner_source_local, torch.Tensor):
        raise TypeError("counts_owner_source_local must be a torch.Tensor")
    if counts_owner_source_local.dim() != 3:
        raise ValueError(
            "counts_owner_source_local must have shape "
            "[owner_rank, source_rank, local_expert]"
        )

    counts = counts_owner_source_local.detach().to(torch.int64).cpu().contiguous()
    world_size, source_world_size, num_local_experts = counts.shape
    if int(world_size) != int(source_world_size):
        raise ValueError(
            "owner and source dimensions must match EP world size, got "
            f"{world_size} and {source_world_size}"
        )

    max_select = min(max(0, int(hot_expert_num)), int(world_size * num_local_experts))
    expert_counts = counts.sum(dim=1).clone()
    owner_load_before = _as_int_tuple(expert_counts.sum(dim=1).tolist())
    modeled_owner_load = [int(v) for v in owner_load_before]

    selected_raw: list[tuple[int, int, int, tuple[int, ...], int, int]] = []
    for _ in range(max_select):
        owner = int(max(range(int(world_size)), key=lambda idx: modeled_owner_load[idx]))
        local_counts = expert_counts[owner]
        local_expert = int(torch.argmax(local_counts).item())
        rows_total = int(local_counts[local_expert].item())
        if rows_total <= 0:
            break

        by_source = _as_int_tuple(counts[owner, :, local_expert].tolist())
        global_expert = int(owner * int(num_local_experts) + local_expert)

        modeled_owner_load[owner] -= rows_total
        for source_rank, row_count in enumerate(by_source):
            modeled_owner_load[source_rank] += int(row_count)
        expert_counts[owner, local_expert] = 0
        selected_raw.append(
            (
                global_expert,
                owner,
                local_expert,
                by_source,
                rows_total,
                int(rows_total - by_source[owner]),
            )
        )

    owner_counts = [0 for _ in range(int(world_size))]
    owner_slots: list[int] = []
    for _global_expert, owner, _local_expert, _by_source, _rows_total, _remote_rows in selected_raw:
        owner_slot = owner_counts[int(owner)]
        owner_counts[int(owner)] += 1
        owner_slots.append(int(owner_slot))

    max_owned = max(owner_counts, default=0)
    owner_shard_offsets = [
        int(owner) * int(max_owned) + int(owner_slot)
        for owner_slot, (_global_expert, owner, *_rest) in zip(owner_slots, selected_raw)
    ]
    owner_shard_active_offsets = tuple(sorted(int(v) for v in owner_shard_offsets))
    owner_shard_to_compact = {
        int(offset): int(compact_idx)
        for compact_idx, offset in enumerate(owner_shard_active_offsets)
    }
    compact_offsets = [
        owner_shard_to_compact[int(offset)] for offset in owner_shard_offsets
    ]
    owner_compact_owner_ranks = [0 for _ in owner_shard_active_offsets]
    owner_compact_local_experts = [0 for _ in owner_shard_active_offsets]
    for compact_offset, (_global_expert, owner, local_expert, *_rest) in zip(
        compact_offsets, selected_raw
    ):
        owner_compact_owner_ranks[int(compact_offset)] = int(owner)
        owner_compact_local_experts[int(compact_offset)] = int(local_expert)

    need_masks = build_owner_compact_need_masks(
        selected_source_rank_counts=[row[3] for row in selected_raw],
        selected_owner_ranks=[row[1] for row in selected_raw],
        selected_owner_compact_offsets=compact_offsets,
        world_size=int(world_size),
        active_owner_compact_count=len(owner_compact_owner_ranks),
    )
    reduction_pct = _load_reduction_pct(owner_load_before, modeled_owner_load)
    if float(reduction_pct) < float(min_reduction_pct):
        return BalancedMoePlan(
            world_size=int(world_size),
            num_local_experts=int(num_local_experts),
            hot_experts=(),
            owner_counts=tuple(0 for _ in range(int(world_size))),
            max_owned_per_rank=0,
            owner_shard_active_offsets=(),
            owner_compact_owner_ranks=(),
            owner_compact_local_experts=(),
            owner_compact_need_masks=tuple(() for _ in range(int(world_size))),
            owner_load_before=owner_load_before,
            exec_load_after=owner_load_before,
            selected_rows_total=0,
            remote_rows_total=0,
            modeled_max_load_reduction_pct=0.0,
        )

    hot_experts = tuple(
        BalancedMoeHotExpert(
            global_expert=int(global_expert),
            owner_rank=int(owner),
            local_expert=int(local_expert),
            source_rank_counts=by_source,
            rows_total=int(rows_total),
            remote_rows=int(remote_rows),
            owner_slot=int(owner_slot),
            owner_shard_offset=int(owner_shard_offset),
            owner_compact_offset=int(compact_offset),
        )
        for (
            (global_expert, owner, local_expert, by_source, rows_total, remote_rows),
            owner_slot,
            owner_shard_offset,
            compact_offset,
        ) in zip(selected_raw, owner_slots, owner_shard_offsets, compact_offsets)
    )
    return BalancedMoePlan(
        world_size=int(world_size),
        num_local_experts=int(num_local_experts),
        hot_experts=hot_experts,
        owner_counts=_as_int_tuple(owner_counts),
        max_owned_per_rank=int(max_owned),
        owner_shard_active_offsets=_as_int_tuple(owner_shard_active_offsets),
        owner_compact_owner_ranks=_as_int_tuple(owner_compact_owner_ranks),
        owner_compact_local_experts=_as_int_tuple(owner_compact_local_experts),
        owner_compact_need_masks=need_masks,
        owner_load_before=owner_load_before,
        exec_load_after=_as_int_tuple(modeled_owner_load),
        selected_rows_total=int(sum(row[4] for row in selected_raw)),
        remote_rows_total=int(sum(row[5] for row in selected_raw)),
        modeled_max_load_reduction_pct=float(reduction_pct),
    )


def _distributed_world_size(group=None) -> int | None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size(group=group))
    return None


def count_local_routes_by_owner_expert(
    topk_ids: torch.Tensor,
    *,
    num_local_experts: int,
    world_size: int | None = None,
    group=None,
    ignore_negative: bool = True,
) -> torch.Tensor:
    """Count this source rank's raw top-k routes by owner rank/local expert."""

    if not isinstance(topk_ids, torch.Tensor):
        raise TypeError("topk_ids must be a torch.Tensor")
    num_local_experts = int(num_local_experts)
    if num_local_experts <= 0:
        raise ValueError(
            f"num_local_experts must be positive, got {num_local_experts}"
        )
    if world_size is None:
        world_size = _distributed_world_size(group=group)
    if world_size is None:
        raise RuntimeError(
            "world_size must be provided when torch.distributed is not initialized"
        )
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")

    flat = topk_ids.detach().to(torch.int64).reshape(-1)
    if bool(ignore_negative):
        flat = flat[flat >= 0]
    elif bool((flat < 0).any().item()):
        raise ValueError("topk_ids contains negative route ids")

    num_global_experts = int(world_size * num_local_experts)
    if flat.numel() > 0 and bool((flat >= num_global_experts).any().item()):
        max_id = int(flat.max().item())
        raise ValueError(
            f"topk_ids contains expert id {max_id}, but only "
            f"{num_global_experts} global experts are addressable"
        )

    counts = torch.bincount(flat, minlength=num_global_experts)
    return counts.to(torch.int64).reshape(world_size, num_local_experts)


def gather_local_counts_to_global(local_counts_source_local: torch.Tensor, group=None) -> torch.Tensor:
    """All-gather local ``[owner_rank, local_expert]`` counts into ``[owner, source, local]``."""

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized to gather counts")
    local = local_counts_source_local.detach().to(torch.int64).contiguous()
    if local.dim() != 2:
        raise ValueError(
            "local_counts_source_local must have shape [owner_rank, local_expert]"
        )
    world_size = int(torch.distributed.get_world_size(group=group))
    if int(local.shape[0]) != world_size:
        raise ValueError(
            "local_counts_source_local owner dimension must match world size, "
            f"got {int(local.shape[0])} and {world_size}"
        )
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local, group=group)
    return torch.stack(gathered, dim=0).permute(1, 0, 2).contiguous()


def build_balanced_moe_plan_from_topk_ids(
    topk_ids: torch.Tensor,
    hot_expert_num: int,
    *,
    num_local_experts: int,
    group=None,
    world_size: int | None = None,
    min_reduction_pct: float = 0.0,
) -> BalancedMoePlan:
    """Build a balanced-MoE plan directly from raw top-k route ids."""

    if world_size is None:
        world_size = _distributed_world_size(group=group)
    local_counts = count_local_routes_by_owner_expert(
        topk_ids,
        num_local_experts=num_local_experts,
        world_size=world_size,
        group=group,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        global_counts = gather_local_counts_to_global(local_counts, group=group)
    else:
        if int(local_counts.shape[0]) != 1:
            raise RuntimeError(
                "torch.distributed must be initialized to build a multi-rank "
                "balanced-MoE plan from topk_ids"
            )
        global_counts = local_counts.unsqueeze(1).contiguous()
    return build_balanced_moe_plan_from_global_counts(
        global_counts,
        hot_expert_num,
        min_reduction_pct=min_reduction_pct,
    )


def build_balanced_moe_plan(
    local_counts_source_local: torch.Tensor,
    hot_expert_num: int,
    *,
    group=None,
    min_reduction_pct: float = 0.0,
) -> BalancedMoePlan:
    global_counts = gather_local_counts_to_global(local_counts_source_local, group=group)
    return build_balanced_moe_plan_from_global_counts(
        global_counts,
        hot_expert_num,
        min_reduction_pct=min_reduction_pct,
    )


def build_owner_compact_need_masks(
    *,
    selected_source_rank_counts: Sequence[Sequence[int]],
    selected_owner_ranks: Sequence[int],
    selected_owner_compact_offsets: Sequence[int],
    world_size: int,
    active_owner_compact_count: int,
) -> tuple[tuple[bool, ...], ...]:
    world_size = int(world_size)
    active_owner_compact_count = int(active_owner_compact_count)
    if (
        len(selected_source_rank_counts) != len(selected_owner_ranks)
        or len(selected_owner_ranks) != len(selected_owner_compact_offsets)
    ):
        raise ValueError("selected source counts, owners, and offsets must align")
    need_masks = [
        [False for _ in range(active_owner_compact_count)]
        for _ in range(world_size)
    ]
    for by_source, owner_rank, compact_offset in zip(
        selected_source_rank_counts,
        selected_owner_ranks,
        selected_owner_compact_offsets,
    ):
        owner_rank = int(owner_rank)
        compact_offset = int(compact_offset)
        if len(by_source) != world_size:
            raise ValueError("each source-count row must have world_size entries")
        if compact_offset < 0 or compact_offset >= active_owner_compact_count:
            continue
        for source_rank, count in enumerate(by_source):
            if source_rank != owner_rank and int(count) > 0:
                need_masks[source_rank][compact_offset] = True
    return tuple(tuple(row) for row in need_masks)


def build_owner_compact_exchange_plan(
    *,
    owner_compact_need_masks: Sequence[Sequence[bool]],
    owner_compact_owner_ranks: Sequence[int],
    rank: int,
) -> BalancedMoeOwnerCompactExchangePlan:
    world_size = int(len(owner_compact_need_masks))
    rank = int(rank)
    if rank < 0 or rank >= world_size:
        raise IndexError(f"rank {rank} is outside world_size={world_size}")
    owner_ranks = _as_int_tuple(owner_compact_owner_ranks)
    active_count = len(owner_ranks)

    normalized_masks: list[tuple[bool, ...]] = []
    for source_rank, row in enumerate(owner_compact_need_masks):
        if len(row) != active_count:
            raise ValueError(
                "owner_compact_need_masks must have shape "
                f"[world_size, {active_count}], row {source_rank} has {len(row)}"
            )
        normalized_masks.append(tuple(bool(v) for v in row))

    needed_offsets = tuple(
        compact_idx
        for compact_idx, needed in enumerate(normalized_masks[rank])
        if bool(needed)
    )
    compact_to_needed = [-1 for _ in range(active_count)]
    for needed_idx, compact_idx in enumerate(needed_offsets):
        compact_to_needed[int(compact_idx)] = int(needed_idx)

    send_by_rank: list[tuple[int, ...]] = []
    recv_by_rank: list[tuple[int, ...]] = []
    for peer in range(world_size):
        send_by_rank.append(
            tuple(
                compact_idx
                for compact_idx, owner_rank in enumerate(owner_ranks)
                if int(owner_rank) == rank and bool(normalized_masks[peer][compact_idx])
            )
        )
        recv_by_rank.append(
            tuple(
                compact_idx
                for compact_idx, owner_rank in enumerate(owner_ranks)
                if int(owner_rank) == peer and bool(normalized_masks[rank][compact_idx])
            )
        )
    max_needed_count = max(
        (sum(1 for needed in row if bool(needed)) for row in normalized_masks),
        default=0,
    )
    needed_density = (
        float(max_needed_count) / float(active_count)
        if active_count > 0
        else 0.0
    )
    return BalancedMoeOwnerCompactExchangePlan(
        rank=rank,
        world_size=world_size,
        active_owner_compact_count=active_count,
        needed_owner_compact_offsets=_as_int_tuple(needed_offsets),
        compact_to_needed_index=_as_int_tuple(compact_to_needed),
        send_owner_compact_offsets_by_rank=tuple(send_by_rank),
        recv_owner_compact_offsets_by_rank=tuple(recv_by_rank),
        input_splits=_as_int_tuple(len(row) for row in send_by_rank),
        output_splits=_as_int_tuple(len(row) for row in recv_by_rank),
        max_needed_count=int(max_needed_count),
        needed_density=float(needed_density),
    )


def build_owner_compact_exchange_plan_from_plan(
    plan: BalancedMoePlan,
    *,
    rank: int,
) -> BalancedMoeOwnerCompactExchangePlan:
    return build_owner_compact_exchange_plan(
        owner_compact_need_masks=plan.owner_compact_need_masks,
        owner_compact_owner_ranks=plan.owner_compact_owner_ranks,
        rank=int(rank),
    )


def prepare_owner_compact_needed_rows_runtime_plan(
    *,
    compact_local_indices: torch.Tensor | Sequence[int],
    plan: BalancedMoeOwnerCompactExchangePlan | Mapping[str, object],
    device: torch.device | str | None = None,
    transport: str | None = None,
) -> dict[str, object]:
    """Precompute rank-local row indices for owner-compact hot-weight movement."""

    if device is None:
        device = (
            compact_local_indices.device
            if isinstance(compact_local_indices, torch.Tensor)
            else torch.device("cpu")
        )
    device_t = device if isinstance(device, torch.device) else torch.device(device)
    if isinstance(plan, BalancedMoeOwnerCompactExchangePlan):
        runtime_plan: dict[str, object] = plan.as_tensors(device=device_t)
    elif hasattr(plan, "as_tensors"):
        runtime_plan = plan.as_tensors(device=device_t)  # type: ignore[assignment]
    elif isinstance(plan, Mapping):
        runtime_plan = dict(plan)
    else:
        raise TypeError(
            "plan must be a BalancedMoeOwnerCompactExchangePlan or mapping"
        )

    compact_local_indices_t = _tensor_or_empty(
        compact_local_indices,
        device=device_t,
    )
    send_offsets = _tensor_or_empty(
        _plan_get(runtime_plan, "send_owner_compact_offsets", None),
        device=device_t,
    )
    runtime_plan["send_local_indices"] = (
        compact_local_indices_t.index_select(0, send_offsets)
        if send_offsets.numel() > 0
        else torch.empty(0, device=device_t, dtype=torch.long)
    )

    grad_recv_offsets = _tensor_or_empty(
        _plan_get(runtime_plan, "grad_recv_owner_compact_offsets", None),
        device=device_t,
    )
    runtime_plan["grad_recv_local_indices"] = (
        compact_local_indices_t.index_select(0, grad_recv_offsets)
        if grad_recv_offsets.numel() > 0
        else torch.empty(0, device=device_t, dtype=torch.long)
    )
    runtime_plan["transport"] = _normalize_owner_compact_exchange_transport(
        str(_plan_get(runtime_plan, "transport", transport))
        if _plan_get(runtime_plan, "transport", transport) is not None
        else None
    )
    return runtime_plan


def build_owner_compact_exchange_runtime_plan(
    *,
    owner_compact_need_masks: Sequence[Sequence[bool]],
    owner_compact_owner_ranks: Sequence[int],
    rank: int,
    compact_local_indices: torch.Tensor | Sequence[int],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.int64,
    split_dtype: torch.dtype = torch.int64,
    transport: str | None = None,
) -> tuple[BalancedMoeOwnerCompactExchangePlan, dict[str, object]]:
    """Build serial and runtime hot-weight exchange plans."""

    plan = build_owner_compact_exchange_plan(
        owner_compact_need_masks=owner_compact_need_masks,
        owner_compact_owner_ranks=owner_compact_owner_ranks,
        rank=int(rank),
    )
    runtime_plan = prepare_owner_compact_needed_rows_runtime_plan(
        compact_local_indices=compact_local_indices,
        plan=plan.as_tensors(
            device=device,
            dtype=dtype,
            split_dtype=split_dtype,
        ),
        device=device,
        transport=transport,
    )
    return plan, runtime_plan


def build_owner_compact_exchange_runtime_plan_from_plan(
    plan: BalancedMoePlan,
    *,
    rank: int,
    compact_local_indices: torch.Tensor | Sequence[int],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.int64,
    split_dtype: torch.dtype = torch.int64,
    transport: str | None = None,
) -> tuple[BalancedMoeOwnerCompactExchangePlan, dict[str, object]]:
    """Build owner-compact exchange metadata from a full balanced-MoE plan."""

    exchange_plan = build_owner_compact_exchange_plan_from_plan(
        plan,
        rank=int(rank),
    )
    runtime_plan = prepare_owner_compact_needed_rows_runtime_plan(
        compact_local_indices=compact_local_indices,
        plan=exchange_plan.as_tensors(
            device=device,
            dtype=dtype,
            split_dtype=split_dtype,
        ),
        device=device,
        transport=transport,
    )
    return exchange_plan, runtime_plan


class _OwnerCompactNeededRowsExchange(torch.autograd.Function):
    """Autograd exchange for owner-compact hot rows.

    Forward moves compact owner rows to helper ranks. Backward sends helper
    partial gradients back to true owners and accumulates by local expert row.
    """

    @staticmethod
    def forward(
        ctx,
        local_rows: torch.Tensor,
        compact_local_indices: torch.Tensor,
        send_local_indices: torch.Tensor,
        grad_recv_local_indices: torch.Tensor,
        needed_row_count: int,
        send_owner_compact_offsets: torch.Tensor,
        recv_to_needed_index: torch.Tensor,
        grad_send_needed_indices: torch.Tensor,
        grad_recv_owner_compact_offsets: torch.Tensor,
        input_splits: tuple[int, ...],
        output_splits: tuple[int, ...],
        input_split_offsets: tuple[int, ...],
        output_split_offsets: tuple[int, ...],
        grad_input_split_offsets: tuple[int, ...],
        grad_output_split_offsets: tuple[int, ...],
        group,
        transport: str,
    ) -> torch.Tensor:
        compact_local_indices = compact_local_indices.to(
            device=local_rows.device,
            dtype=torch.long,
        )
        send_local_indices = send_local_indices.to(
            device=local_rows.device,
            dtype=torch.long,
        )
        grad_recv_local_indices = grad_recv_local_indices.to(
            device=local_rows.device,
            dtype=torch.long,
        )
        send_owner_compact_offsets = send_owner_compact_offsets.to(
            device=local_rows.device,
            dtype=torch.long,
        )
        recv_to_needed_index = recv_to_needed_index.to(
            device=local_rows.device,
            dtype=torch.long,
        )
        grad_send_needed_indices = grad_send_needed_indices.to(
            device=local_rows.device,
            dtype=torch.long,
        )
        grad_recv_owner_compact_offsets = grad_recv_owner_compact_offsets.to(
            device=local_rows.device,
            dtype=torch.long,
        )
        input_splits = tuple(int(v) for v in input_splits)
        output_splits = tuple(int(v) for v in output_splits)
        input_split_offsets = _offset_tuple(
            input_split_offsets,
            input_splits,
            name="input_split_offsets",
        )
        output_split_offsets = _offset_tuple(
            output_split_offsets,
            output_splits,
            name="output_split_offsets",
        )
        grad_input_split_offsets = _offset_tuple(
            grad_input_split_offsets,
            output_splits,
            name="grad_input_split_offsets",
        )
        grad_output_split_offsets = _offset_tuple(
            grad_output_split_offsets,
            input_splits,
            name="grad_output_split_offsets",
        )
        if int(send_owner_compact_offsets.numel()) != int(sum(input_splits)):
            raise ValueError("send_owner_compact_offsets must match input_splits")
        if int(recv_to_needed_index.numel()) != int(sum(output_splits)):
            raise ValueError("recv_to_needed_index must match output_splits")
        if int(grad_send_needed_indices.numel()) != int(sum(output_splits)):
            raise ValueError("grad_send_needed_indices must match output_splits")
        if int(grad_recv_owner_compact_offsets.numel()) != int(sum(input_splits)):
            raise ValueError("grad_recv_owner_compact_offsets must match input_splits")

        ctx.local_rows_shape = tuple(local_rows.shape)
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.input_split_offsets = input_split_offsets
        ctx.output_split_offsets = output_split_offsets
        ctx.grad_input_split_offsets = grad_input_split_offsets
        ctx.grad_output_split_offsets = grad_output_split_offsets
        ctx.group = group
        _normalize_owner_compact_exchange_transport(transport)

        if send_owner_compact_offsets.numel() > 0:
            if int(send_local_indices.numel()) != int(
                send_owner_compact_offsets.numel()
            ):
                send_local_indices = compact_local_indices.index_select(
                    0,
                    send_owner_compact_offsets,
                )
            send = local_rows.index_select(0, send_local_indices).contiguous()
        else:
            send = local_rows.new_empty((0, *local_rows.shape[1:]))
            send_local_indices = torch.empty(
                0,
                device=local_rows.device,
                dtype=torch.long,
            )

        if grad_recv_owner_compact_offsets.numel() > 0:
            if int(grad_recv_local_indices.numel()) != int(
                grad_recv_owner_compact_offsets.numel()
            ):
                grad_recv_local_indices = compact_local_indices.index_select(
                    0,
                    grad_recv_owner_compact_offsets,
                )
        else:
            grad_recv_local_indices = torch.empty(
                0,
                device=local_rows.device,
                dtype=torch.long,
            )
        ctx.save_for_backward(grad_send_needed_indices, grad_recv_local_indices)

        recv = local_rows.new_empty((sum(output_splits), *local_rows.shape[1:]))
        _dist_all_to_all_single(
            recv,
            send,
            output_splits=output_splits,
            input_splits=input_splits,
            group=group,
        )

        needed_rows = local_rows.new_empty(
            (int(needed_row_count), *local_rows.shape[1:])
        )
        if recv_to_needed_index.numel() > 0:
            needed_rows.index_copy_(0, recv_to_needed_index, recv)
        return needed_rows

    @staticmethod
    def backward(ctx, grad_needed_rows: torch.Tensor):
        grad_send_needed_indices, grad_recv_local_indices = ctx.saved_tensors
        grad_needed_rows = grad_needed_rows.contiguous()
        if grad_send_needed_indices.numel() > 0:
            send = grad_needed_rows.index_select(
                0,
                grad_send_needed_indices.to(grad_needed_rows.device),
            ).contiguous()
        else:
            send = grad_needed_rows.new_empty((0, *grad_needed_rows.shape[1:]))

        recv = grad_needed_rows.new_empty(
            (sum(ctx.input_splits), *grad_needed_rows.shape[1:])
        )
        _dist_all_to_all_single(
            recv,
            send,
            output_splits=ctx.input_splits,
            input_splits=ctx.output_splits,
            group=ctx.group,
        )

        grad_local_rows = grad_needed_rows.new_zeros(ctx.local_rows_shape)
        if grad_recv_local_indices.numel() > 0:
            grad_local_rows.index_add_(
                0,
                grad_recv_local_indices.to(grad_needed_rows.device),
                recv,
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
            None,
            None,
            None,
        )


def _exchange_owner_compact_needed_rows_via_turboep(
    local_rows: torch.Tensor,
    *,
    compact_local_indices: torch.Tensor,
    send_local_indices: torch.Tensor,
    grad_recv_local_indices: torch.Tensor,
    needed_row_count: int,
    send_owner_compact_offsets: torch.Tensor,
    recv_to_needed_index: torch.Tensor,
    grad_send_needed_indices: torch.Tensor,
    grad_recv_owner_compact_offsets: torch.Tensor,
    input_splits: tuple[int, ...],
    output_splits: tuple[int, ...],
    group,
) -> torch.Tensor:
    """Move owner-compact rows through Primus-Turbo's raw TurboEP dispatcher.

    Each local send row becomes one synthetic top-k token.  The synthetic
    expert id is the destination rank, with one synthetic local expert per
    rank.  Backward is provided by ``moe_dispatch`` itself: its VJP calls
    TurboEP combine, returning helper partial dW rows to the true owners.
    """

    if len(input_splits) != len(output_splits):
        raise ValueError("input_splits and output_splits must have the same length")

    device = local_rows.device
    compact_local_indices = compact_local_indices.to(device=device, dtype=torch.long)
    send_local_indices = send_local_indices.to(device=device, dtype=torch.long)
    grad_recv_local_indices = grad_recv_local_indices.to(device=device, dtype=torch.long)
    send_owner_compact_offsets = send_owner_compact_offsets.to(
        device=device,
        dtype=torch.long,
    )
    recv_to_needed_index = recv_to_needed_index.to(device=device, dtype=torch.long)

    # Fast local path keeps CPU-only unit tests useful and avoids invoking
    # TurboEP when a single-rank caller only needs a deterministic gather.
    if len(input_splits) == 1 and (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
    ):
        if send_owner_compact_offsets.numel() > 0:
            if int(send_local_indices.numel()) != int(
                send_owner_compact_offsets.numel()
            ):
                send_local_indices = compact_local_indices.index_select(
                    0,
                    send_owner_compact_offsets,
                )
            recv_rows = local_rows.index_select(0, send_local_indices)
        else:
            recv_rows = local_rows.new_empty((0, *local_rows.shape[1:]))
        needed_rows = local_rows.new_empty((int(needed_row_count), *local_rows.shape[1:]))
        if recv_to_needed_index.numel() > 0:
            needed_rows.index_copy_(0, recv_to_needed_index, recv_rows)
        return needed_rows

    if group is None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "Primus-Turbo owner-compact exchange requires a distributed "
                "process group when world_size > 1"
            )
        group = torch.distributed.group.WORLD
    if not local_rows.is_cuda:
        raise RuntimeError(
            "Primus-Turbo owner-compact exchange requires CUDA/ROCm tensors"
        )

    if send_owner_compact_offsets.numel() > 0:
        if int(send_local_indices.numel()) != int(send_owner_compact_offsets.numel()):
            send_local_indices = compact_local_indices.index_select(
                0,
                send_owner_compact_offsets,
            )
        send_rows = local_rows.index_select(0, send_local_indices).contiguous()
    else:
        send_rows = local_rows.new_empty((0, *local_rows.shape[1:]))

    expected_send = int(sum(input_splits))
    expected_recv = int(sum(output_splits))
    if int(send_rows.shape[0]) != expected_send:
        raise ValueError(
            f"send row count {int(send_rows.shape[0])} does not match "
            f"input_splits total {expected_send}"
        )
    if int(recv_to_needed_index.numel()) != expected_recv:
        raise ValueError(
            f"recv_to_needed_index has {int(recv_to_needed_index.numel())} "
            f"entries, expected output_splits total {expected_recv}"
        )

    target_ranks = [
        peer
        for peer, count in enumerate(input_splits)
        for _ in range(int(count))
    ]
    token_indices = torch.tensor(
        target_ranks,
        device=device,
        dtype=torch.long,
    ).reshape(-1, 1)
    token_weights = torch.ones(
        (expected_send, 1),
        device=device,
        dtype=torch.float32,
    )

    from primus_turbo.pytorch.ops.moe.moe_dispatch_combine import moe_dispatch

    recv_rows, _recv_token_indices, _recv_token_probs, _tokens_per_expert, _handle = (
        moe_dispatch(
            send_rows,
            token_indices=token_indices,
            token_probs=token_weights,
            num_experts=len(input_splits),
            group=group,
        )
    )
    needed_rows = local_rows.new_empty((int(needed_row_count), *local_rows.shape[1:]))
    if recv_to_needed_index.numel() > 0:
        needed_rows.index_copy_(
            0,
            recv_to_needed_index.to(device=device, dtype=torch.long),
            recv_rows,
        )
    return needed_rows


def _exchange_owner_compact_needed_rows_via_mori_sdma(
    local_rows: torch.Tensor,
    *,
    compact_local_indices: torch.Tensor,
    plan: BalancedMoeOwnerCompactExchangePlan | Mapping[str, object] | None,
    group,
    needed_owner_compact_offsets: torch.Tensor | Sequence[int] | None,
    send_owner_compact_offsets: torch.Tensor | Sequence[int] | None,
    recv_to_needed_index: torch.Tensor | Sequence[int] | None,
    grad_send_needed_indices: torch.Tensor | Sequence[int] | None,
    grad_recv_owner_compact_offsets: torch.Tensor | Sequence[int] | None,
    input_splits: torch.Tensor | Sequence[int] | None,
    output_splits: torch.Tensor | Sequence[int] | None,
    input_split_offsets: torch.Tensor | Sequence[int] | None,
    output_split_offsets: torch.Tensor | Sequence[int] | None,
    grad_input_split_offsets: torch.Tensor | Sequence[int] | None,
    grad_output_split_offsets: torch.Tensor | Sequence[int] | None,
) -> torch.Tensor:
    """Delegate owner-compact hot-row movement to MORI SDMA.

    Primus-Turbo owns the hot/cold/helper layout and normal TurboEP edge.
    MORI owns the SDMA transport primitive.  Keeping the call here lets
    TorchTitan select one Primus-Turbo balanced-MoE ABI without rebuilding a
    dense selected-hot tensor or carrying transport-specific logic in the
    layer wrapper.
    """

    try:
        from mori.ops.balanced_moe import (
            exchange_owner_compact_needed_rows as mori_exchange_owner_compact_needed_rows,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime image.
        raise RuntimeError(
            "Primus-Turbo native hot-helper transport requested "
            "mori_sdma_padded_all2all, but mori.ops.balanced_moe is not "
            "available in the active runtime."
        ) from exc

    return mori_exchange_owner_compact_needed_rows(
        local_rows,
        compact_local_indices=compact_local_indices,
        plan=plan,
        group=group,
        needed_owner_compact_offsets=needed_owner_compact_offsets,
        send_owner_compact_offsets=send_owner_compact_offsets,
        recv_to_needed_index=recv_to_needed_index,
        grad_send_needed_indices=grad_send_needed_indices,
        grad_recv_owner_compact_offsets=grad_recv_owner_compact_offsets,
        input_splits=input_splits,
        output_splits=output_splits,
        input_split_offsets=input_split_offsets,
        output_split_offsets=output_split_offsets,
        grad_input_split_offsets=grad_input_split_offsets,
        grad_output_split_offsets=grad_output_split_offsets,
        transport=OWNER_COMPACT_MORI_SDMA_PADDED_A2A_TRANSPORT,
    )


def exchange_owner_compact_needed_rows(
    local_rows: torch.Tensor,
    *,
    compact_local_indices: torch.Tensor | Sequence[int] | None = None,
    plan: BalancedMoeOwnerCompactExchangePlan | Mapping[str, object] | None = None,
    group=None,
    needed_owner_compact_offsets: torch.Tensor | Sequence[int] | None = None,
    send_owner_compact_offsets: torch.Tensor | Sequence[int] | None = None,
    recv_to_needed_index: torch.Tensor | Sequence[int] | None = None,
    grad_send_needed_indices: torch.Tensor | Sequence[int] | None = None,
    grad_recv_owner_compact_offsets: torch.Tensor | Sequence[int] | None = None,
    input_splits: torch.Tensor | Sequence[int] | None = None,
    output_splits: torch.Tensor | Sequence[int] | None = None,
    input_split_offsets: torch.Tensor | Sequence[int] | None = None,
    output_split_offsets: torch.Tensor | Sequence[int] | None = None,
    grad_input_split_offsets: torch.Tensor | Sequence[int] | None = None,
    grad_output_split_offsets: torch.Tensor | Sequence[int] | None = None,
    transport: str | None = None,
) -> torch.Tensor:
    """Exchange owner-compact hot rows needed by this helper rank.

    This is the Primus-Turbo balanced-MoE hot-weight movement ABI: callers
    pass compact owner-row metadata from
    :class:`BalancedMoeOwnerCompactExchangePlan`, and the autograd inverse
    returns partial dW rows to the true owners without selected-order adapters.
    """

    needed_owner_compact_offsets = _plan_get(
        plan,
        "needed_owner_compact_offsets",
        needed_owner_compact_offsets,
    )
    send_owner_compact_offsets = _plan_get(
        plan,
        "send_owner_compact_offsets",
        send_owner_compact_offsets,
    )
    recv_to_needed_index = _plan_get(plan, "recv_to_needed_index", recv_to_needed_index)
    grad_send_needed_indices = _plan_get(
        plan,
        "grad_send_needed_indices",
        grad_send_needed_indices,
    )
    grad_recv_owner_compact_offsets = _plan_get(
        plan,
        "grad_recv_owner_compact_offsets",
        grad_recv_owner_compact_offsets,
    )
    send_local_indices = _plan_get(plan, "send_local_indices", None)
    grad_recv_local_indices = _plan_get(plan, "grad_recv_local_indices", None)
    input_splits = _plan_get(
        plan,
        "input_splits_tuple",
        _plan_get(plan, "input_splits", input_splits),
    )
    output_splits = _plan_get(
        plan,
        "output_splits_tuple",
        _plan_get(plan, "output_splits", output_splits),
    )
    input_split_offsets = _plan_get(
        plan,
        "input_split_offsets_tuple",
        _plan_get(plan, "input_split_offsets", input_split_offsets),
    )
    output_split_offsets = _plan_get(
        plan,
        "output_split_offsets_tuple",
        _plan_get(plan, "output_split_offsets", output_split_offsets),
    )
    grad_input_split_offsets = _plan_get(
        plan,
        "grad_input_split_offsets_tuple",
        _plan_get(plan, "grad_input_split_offsets", grad_input_split_offsets),
    )
    grad_output_split_offsets = _plan_get(
        plan,
        "grad_output_split_offsets_tuple",
        _plan_get(plan, "grad_output_split_offsets", grad_output_split_offsets),
    )

    device = local_rows.device
    if compact_local_indices is None:
        has_send_local = send_local_indices is not None
        has_grad_recv_local = grad_recv_local_indices is not None
        if not has_send_local or not has_grad_recv_local:
            raise ValueError(
                "compact_local_indices is required unless plan carries both "
                "send_local_indices and grad_recv_local_indices"
            )
    compact_local_indices_t = _tensor_or_empty(
        compact_local_indices,
        device=device,
    )
    needed_count_value = _plan_get(plan, "needed_row_count", None)
    needed_count = (
        int(needed_count_value)
        if needed_count_value is not None
        else _row_count(needed_owner_compact_offsets)
    )
    input_splits_t = _split_tuple(input_splits)
    output_splits_t = _split_tuple(output_splits)
    input_split_offsets_t = _offset_tuple(
        input_split_offsets,
        input_splits_t,
        name="input_split_offsets",
    )
    output_split_offsets_t = _offset_tuple(
        output_split_offsets,
        output_splits_t,
        name="output_split_offsets",
    )
    grad_input_split_offsets_t = _offset_tuple(
        grad_input_split_offsets,
        output_splits_t,
        name="grad_input_split_offsets",
    )
    grad_output_split_offsets_t = _offset_tuple(
        grad_output_split_offsets,
        input_splits_t,
        name="grad_output_split_offsets",
    )
    requested_transport = (
        transport
        if transport is not None
        else _plan_get(plan, "transport", None)
    )
    transport = _normalize_owner_compact_exchange_transport(
        str(requested_transport) if requested_transport is not None else None
    )
    if transport == OWNER_COMPACT_MORI_SDMA_PADDED_A2A_TRANSPORT:
        return _exchange_owner_compact_needed_rows_via_mori_sdma(
            local_rows,
            compact_local_indices=compact_local_indices_t,
            plan=plan,
            group=group,
            needed_owner_compact_offsets=needed_owner_compact_offsets,
            send_owner_compact_offsets=send_owner_compact_offsets,
            recv_to_needed_index=recv_to_needed_index,
            grad_send_needed_indices=grad_send_needed_indices,
            grad_recv_owner_compact_offsets=grad_recv_owner_compact_offsets,
            input_splits=input_splits_t,
            output_splits=output_splits_t,
            input_split_offsets=input_split_offsets_t,
            output_split_offsets=output_split_offsets_t,
            grad_input_split_offsets=grad_input_split_offsets_t,
            grad_output_split_offsets=grad_output_split_offsets_t,
        )
    if transport == OWNER_COMPACT_PRIMUS_TURBO_DISPATCH_TRANSPORT:
        return _exchange_owner_compact_needed_rows_via_turboep(
            local_rows,
            compact_local_indices=compact_local_indices_t,
            send_local_indices=_tensor_or_empty(send_local_indices, device=device),
            grad_recv_local_indices=_tensor_or_empty(
                grad_recv_local_indices,
                device=device,
            ),
            needed_row_count=needed_count,
            send_owner_compact_offsets=_tensor_or_empty(
                send_owner_compact_offsets,
                device=device,
            ),
            recv_to_needed_index=_tensor_or_empty(recv_to_needed_index, device=device),
            grad_send_needed_indices=_tensor_or_empty(
                grad_send_needed_indices,
                device=device,
            ),
            grad_recv_owner_compact_offsets=_tensor_or_empty(
                grad_recv_owner_compact_offsets,
                device=device,
            ),
            input_splits=input_splits_t,
            output_splits=output_splits_t,
            group=group,
        )
    return _OwnerCompactNeededRowsExchange.apply(
        local_rows,
        compact_local_indices_t,
        _tensor_or_empty(send_local_indices, device=device),
        _tensor_or_empty(grad_recv_local_indices, device=device),
        needed_count,
        _tensor_or_empty(send_owner_compact_offsets, device=device),
        _tensor_or_empty(recv_to_needed_index, device=device),
        _tensor_or_empty(grad_send_needed_indices, device=device),
        _tensor_or_empty(grad_recv_owner_compact_offsets, device=device),
        input_splits_t,
        output_splits_t,
        input_split_offsets_t,
        output_split_offsets_t,
        grad_input_split_offsets_t,
        grad_output_split_offsets_t,
        group,
        transport,
    )


def build_source_partition_from_offsets(
    selected_experts_indices: torch.Tensor,
    *,
    selected_global_experts: Sequence[int],
    selected_owner_ranks: Sequence[int],
    selected_owner_shard_offsets: Sequence[int] | None = None,
    selected_owner_compact_offsets: Sequence[int] | None = None,
    ep_rank: int,
    presort_by: str = "off",
) -> BalancedMoeSourcePartition:
    """Build source-side masks and owner-compact offsets for helper rows."""

    if selected_experts_indices.dim() < 1:
        raise ValueError("selected_experts_indices must have at least one dimension")
    hot_global = _as_int_tuple(selected_global_experts)
    hot_owner = _as_int_tuple(selected_owner_ranks)
    if len(hot_global) != len(hot_owner):
        raise ValueError("selected expert ids and owner ranks must align")
    if selected_owner_shard_offsets is None:
        hot_to_shard = tuple(range(len(hot_global)))
    else:
        hot_to_shard = _as_int_tuple(selected_owner_shard_offsets)
    if selected_owner_compact_offsets is None:
        hot_to_compact = tuple(range(len(hot_global)))
    else:
        hot_to_compact = _as_int_tuple(selected_owner_compact_offsets)
    if len(hot_to_shard) != len(hot_global) or len(hot_to_compact) != len(hot_global):
        raise ValueError("selected owner offsets must align with experts")
    presort_by = presort_by.strip().lower() or "off"
    if presort_by not in {"off", "selected", "owner_compact"}:
        raise ValueError("presort_by must be 'off', 'selected', or 'owner_compact'")

    flat_experts = selected_experts_indices.reshape(-1)
    device = flat_experts.device
    remote_hot_mask = torch.zeros_like(flat_experts, dtype=torch.bool)
    hot_offsets = torch.full_like(flat_experts, -1, dtype=torch.long)
    for hot_offset, (global_expert, owner_rank) in enumerate(zip(hot_global, hot_owner)):
        if int(owner_rank) == int(ep_rank):
            continue
        expert_mask = flat_experts == int(global_expert)
        remote_hot_mask |= expert_mask
        hot_offsets = torch.where(
            expert_mask,
            torch.full_like(hot_offsets, int(hot_offset)),
            hot_offsets,
        )

    keep_flat_mask = ~remote_hot_mask
    remote_flat_positions = torch.nonzero(remote_hot_mask, as_tuple=False).flatten()
    remote_hot_offsets = hot_offsets.index_select(0, remote_flat_positions)
    if remote_hot_offsets.numel() > 0:
        shard_lookup = torch.tensor(hot_to_shard, dtype=torch.long, device=device)
        compact_lookup = torch.tensor(hot_to_compact, dtype=torch.long, device=device)
        remote_owner_shard_offsets = shard_lookup.index_select(
            0, remote_hot_offsets.to(torch.long)
        )
        remote_owner_compact_offsets = compact_lookup.index_select(
            0, remote_hot_offsets.to(torch.long)
        )
    else:
        remote_owner_shard_offsets = torch.empty(0, dtype=torch.long, device=device)
        remote_owner_compact_offsets = torch.empty(0, dtype=torch.long, device=device)

    remote_group_ends = None
    remote_offsets_presorted_by = ""
    if presort_by != "off" and remote_hot_offsets.numel() > 0:
        if presort_by == "owner_compact":
            presort_offsets = remote_owner_compact_offsets
            group_count = max(hot_to_compact) + 1 if hot_to_compact else 0
        else:
            presort_offsets = remote_hot_offsets
            group_count = len(hot_global)
        order = torch.argsort(presort_offsets, stable=True)
        remote_flat_positions = remote_flat_positions.index_select(0, order)
        remote_hot_offsets = remote_hot_offsets.index_select(0, order)
        remote_owner_shard_offsets = remote_owner_shard_offsets.index_select(
            0, order
        )
        remote_owner_compact_offsets = remote_owner_compact_offsets.index_select(
            0, order
        )
        remote_group_ends = torch.cumsum(
            torch.bincount(presort_offsets.to(torch.long), minlength=int(group_count)).to(
                torch.int64
            ),
            dim=0,
            dtype=torch.int32,
        )
        remote_offsets_presorted_by = presort_by

    top_k = int(selected_experts_indices.shape[-1])
    remote_token_indices = remote_flat_positions // max(1, top_k)
    return BalancedMoeSourcePartition(
        keep_flat_mask=keep_flat_mask,
        remote_hot_mask=remote_hot_mask,
        normal_route_mask=keep_flat_mask.reshape_as(selected_experts_indices).to(
            torch.uint8
        ),
        remote_flat_positions=remote_flat_positions,
        remote_hot_offsets=remote_hot_offsets,
        remote_owner_shard_offsets=remote_owner_shard_offsets,
        remote_owner_compact_offsets=remote_owner_compact_offsets,
        remote_token_indices=remote_token_indices,
        remote_group_ends=remote_group_ends,
        remote_offsets_presorted_by=remote_offsets_presorted_by,
    )


def build_source_partition(
    selected_experts_indices: torch.Tensor,
    plan: BalancedMoePlan,
    *,
    ep_rank: int,
    presort_by: str = "off",
) -> BalancedMoeSourcePartition:
    return build_source_partition_from_offsets(
        selected_experts_indices,
        selected_global_experts=plan.selected_global_experts,
        selected_owner_ranks=plan.selected_owner_ranks,
        selected_owner_shard_offsets=plan.owner_shard_offsets,
        selected_owner_compact_offsets=plan.owner_compact_offsets,
        ep_rank=ep_rank,
        presort_by=presort_by,
    )


def build_balanced_moe_runtime_layout(
    selected_experts_indices: torch.Tensor,
    plan: BalancedMoePlan,
    *,
    ep_rank: int,
    compact_local_indices: torch.Tensor | Sequence[int],
    presort_by: str = "owner_compact",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.int64,
    split_dtype: torch.dtype = torch.int64,
    transport: str | None = None,
) -> BalancedMoeRuntimeLayout:
    """Build source partition plus reusable owner-compact exchange metadata."""

    source_partition = build_source_partition(
        selected_experts_indices,
        plan,
        ep_rank=int(ep_rank),
        presort_by=presort_by,
    )
    exchange_plan, runtime_plan = build_owner_compact_exchange_runtime_plan_from_plan(
        plan,
        rank=int(ep_rank),
        compact_local_indices=compact_local_indices,
        device=device,
        dtype=dtype,
        split_dtype=split_dtype,
        transport=transport,
    )
    return BalancedMoeRuntimeLayout(
        source_partition=source_partition,
        owner_compact_exchange_plan=exchange_plan,
        owner_compact_exchange_runtime_plan=runtime_plan,
    )
