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
"""Balanced-MoE planning utilities for MORI EP.

This module owns the deterministic hot-expert plan used by a native MORI
balanced-MoE path.  It mirrors the MindSpeed/Ascend policy:

1. Build global route counts as ``[owner_rank, source_rank, local_expert]``.
2. Repeatedly pick the busiest expert on the busiest owner rank.
3. Model that expert as helper-executed on source ranks, reducing owner-rank
   load while preserving the original top-k routing semantics.

The returned plan is intentionally framework-neutral: training and inference
callers can use it to form normal/cold MORI dispatch rows, helper-hot rows,
and hot-weight broadcast/reduce groups without rebuilding the policy outside
MORI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch

from mori.ops.dispatch_combine import (
    BalancedMoeCompactCombineOutput,
    BalancedMoeCompactDispatchOutput,
)

__all__ = [
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

BALANCED_MOE_BACKEND_ABI_VERSION = 1
OWNER_COMPACT_TORCH_A2A_TRANSPORT = "torch_distributed_all_to_all_single"
OWNER_COMPACT_MORI_SDMA_PADDED_A2A_TRANSPORT = "mori_sdma_padded_all2all"
OWNER_COMPACT_NATIVE_A2A_TRANSPORT = OWNER_COMPACT_MORI_SDMA_PADDED_A2A_TRANSPORT
BALANCED_MOE_BACKEND_CAPABILITIES = {
    "backend": "mori",
    "hot_expert_planner": True,
    "source_partition": True,
    "owner_compact_exchange_plan": True,
    "owner_compact_runtime_layout": True,
    "owner_compact_exchange_autograd": True,
    "normal_topk_dispatch_tensors": True,
    "normal_topk_ep_dispatch": False,
    "normal_topk_ep_dispatch_backend": "not_implemented",
    "normal_topk_ep_dispatch_permute": False,
    "normal_topk_ep_dispatch_permute_backend": "not_implemented",
    "balanced_moe_compact_dispatch": True,
    "balanced_moe_compact_dispatch_backend": "mori_ep_dispatch_combine_op",
    "balanced_moe_compact_combine": True,
    "balanced_moe_compact_combine_backend": "mori_ep_dispatch_combine_op",
    "balanced_moe_compact_rows_dispatch": True,
    "balanced_moe_compact_rows_dispatch_backend": "mori_ep_dispatch_combine_op",
    "balanced_moe_compact_rows_combine": True,
    "balanced_moe_compact_rows_combine_backend": "mori_ep_dispatch_combine_op",
    "balanced_moe_compact_rows_native": True,
    "balanced_moe_compact_rows_native_status": "mori_ep_dispatch_combine_op",
    "owner_compact_exchange_transport": OWNER_COMPACT_TORCH_A2A_TRANSPORT,
    "owner_compact_exchange_transports": (
        OWNER_COMPACT_TORCH_A2A_TRANSPORT,
        OWNER_COMPACT_MORI_SDMA_PADDED_A2A_TRANSPORT,
    ),
    "native_hot_helper_transport": True,
    "native_hot_helper_transport_status": "opt_in_padded_sdma_all2all",
    "native_owner_compact_exchange_transport": OWNER_COMPACT_NATIVE_A2A_TRANSPORT,
}


def balanced_moe_backend_capabilities() -> dict[str, object]:
    """Return the balanced-MoE backend ABI surface implemented here.

    This is deliberately explicit because a backend can own the hot/cold/helper
    layout contract before its row movement is implemented with native MORI
    transport.  Throughput promotions should check this before calling a result
    fully backend-native.
    """

    return {
        "abi_version": BALANCED_MOE_BACKEND_ABI_VERSION,
        **BALANCED_MOE_BACKEND_CAPABILITIES,
    }


@dataclass(frozen=True)
class BalancedMoeHotExpert:
    """One helper-executed hot expert selected by the greedy planner."""

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
    """Native MORI balanced-MoE plan.

    ``owner_shard_offset`` is stable for an owner-sharded hot-weight tensor of
    shape ``[world_size, max_owned_per_rank, ...]``.  ``owner_compact_offset``
    is stable for compact active-hot tensors of shape ``[num_hot, ...]``.
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
            "owner_compact_offsets": list(self.owner_compact_offsets),
            "owner_shard_active_offsets": list(self.owner_shard_active_offsets),
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

    def needed_owner_compact_offsets(self, rank: int) -> tuple[int, ...]:
        """Return compact hot-weight rows needed by a source/helper rank."""

        rank = int(rank)
        if rank < 0 or rank >= int(self.world_size):
            raise IndexError(f"rank {rank} is outside world_size={self.world_size}")
        if not self.owner_compact_need_masks:
            return ()
        return tuple(
            int(compact_idx)
            for compact_idx, needed in enumerate(self.owner_compact_need_masks[rank])
            if bool(needed)
        )


@dataclass(frozen=True)
class BalancedMoeSourcePartition:
    """Source-rank route partition derived from a :class:`BalancedMoePlan`."""

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
class BalancedMoeRuntimeLayout:
    """Native source-rank balanced-MoE runtime layout.

    This groups the three framework-facing products of a balanced-MoE plan:
    the normal/cold route mask, the helper-hot source partition, and the
    owner-compact hot-weight exchange runtime plan.  Keeping them together
    makes hot/cold/helper rows a MORI ABI instead of scattered framework glue.
    """

    source_partition: BalancedMoeSourcePartition
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
class BalancedMoeOwnerCompactExchangePlan:
    """Per-rank hot-weight exchange plan for owner-compact balanced MoE.

    The plan is intentionally row-offset based.  A helper rank receives only
    the compact hot-weight rows it actually needs, while an owner rank sends
    only rows requested by helpers.  This is the ABI needed to avoid expanding
    back into selected-hot order around the helper grouped-MLP path.
    """

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
        """Forward send rows flattened in peer-rank order.

        These compact offsets identify rows this rank owns and must send to
        helper/source ranks.  Backward receives partial dW rows in the same
        compact-offset order.
        """

        return tuple(
            int(offset)
            for row in self.send_owner_compact_offsets_by_rank
            for offset in row
        )

    @property
    def recv_owner_compact_offsets(self) -> tuple[int, ...]:
        """Forward receive rows flattened in peer-rank order."""

        return tuple(
            int(offset)
            for row in self.recv_owner_compact_offsets_by_rank
            for offset in row
        )

    @property
    def recv_to_needed_index(self) -> tuple[int, ...]:
        """Map flattened forward receives into this rank's compact needed buffer."""

        return tuple(
            int(self.compact_to_needed_index[int(offset)])
            for offset in self.recv_owner_compact_offsets
        )

    @property
    def grad_send_needed_indices_by_rank(self) -> tuple[tuple[int, ...], ...]:
        """Backward send needed-buffer rows grouped by owner peer.

        Forward receives hot weights from the true owner; backward sends the
        corresponding partial hot-weight gradients back to that same owner.
        The helper-side gradient buffer is naturally indexed by
        ``needed_owner_compact_offsets``, so this mapping avoids rebuilding a
        selected-order adapter in callers.
        """

        return tuple(
            tuple(
                int(self.compact_to_needed_index[int(offset)])
                for offset in row
            )
            for row in self.recv_owner_compact_offsets_by_rank
        )

    @property
    def grad_send_needed_indices(self) -> tuple[int, ...]:
        """Backward send rows flattened in peer-rank order."""

        return tuple(
            int(index)
            for row in self.grad_send_needed_indices_by_rank
            for index in row
        )

    @property
    def grad_recv_owner_compact_offsets(self) -> tuple[int, ...]:
        """Backward receive compact owner rows flattened in peer-rank order."""

        return self.send_owner_compact_offsets

    @property
    def input_split_offsets(self) -> tuple[int, ...]:
        """Exclusive row-prefix offsets for forward send peer spans."""

        return _exclusive_prefix_offsets(self.input_splits)

    @property
    def output_split_offsets(self) -> tuple[int, ...]:
        """Exclusive row-prefix offsets for forward receive peer spans."""

        return _exclusive_prefix_offsets(self.output_splits)

    @property
    def grad_input_split_offsets(self) -> tuple[int, ...]:
        """Exclusive row-prefix offsets for backward send peer spans."""

        return self.output_split_offsets

    @property
    def grad_output_split_offsets(self) -> tuple[int, ...]:
        """Exclusive row-prefix offsets for backward receive peer spans."""

        return self.input_split_offsets

    def as_tensors(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.int64,
        split_dtype: torch.dtype = torch.int64,
    ) -> dict[str, object]:
        """Return the owner-compact exchange ABI as tensor row indices.

        The returned tensors are deliberately small metadata tensors.  They are
        the direct ABI a native MORI balanced-MoE weight movement primitive
        needs: forward owner rows, forward helper receives, backward helper
        partial-grad sends, and backward owner receives.
        """

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
        return getattr(plan, name)
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

    Native balanced-MoE first removes remote-hot routes for helper execution.
    The remaining normal/cold routes can still use a raw top-k dispatcher if
    remote-hot slots are converted to the backend sentinel ``-1`` and their
    weights are zeroed.  Keeping this conversion in MORI makes the hot/cold
    layout ABI explicit instead of rebuilding route masks in each framework.
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
    """State for backends that implement raw top-k normal/cold EP dispatch."""

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
    """MORI does not implement the TurboEP-style raw top-k dispatch ABI."""

    del (
        x,
        normal_topk_ids,
        normal_topk_weights,
        num_experts,
        group,
        async_finish,
        allocate_on_comm_stream,
        num_worst_tokens,
    )
    raise NotImplementedError(
        "MORI balanced_moe uses the compact-row MORI native path; "
        "raw top-k normal/cold dispatch is a Primus-Turbo TurboEP ABI."
    )


def combine_normal_topk_tokens(
    expert_output: torch.Tensor,
    state: BalancedMoeNormalTopKDispatchState,
    *,
    group,
    async_finish: bool | None = None,
    allocate_on_comm_stream: bool | None = None,
) -> torch.Tensor:
    """MORI does not implement the TurboEP-style raw top-k combine ABI."""

    del expert_output, state, group, async_finish, allocate_on_comm_stream
    raise NotImplementedError(
        "MORI balanced_moe uses the compact-row MORI native path; "
        "raw top-k normal/cold combine is a Primus-Turbo TurboEP ABI."
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
    """MORI does not implement the TurboEP-style raw top-k permute ABI."""

    del (
        x,
        normal_topk_ids,
        normal_topk_weights,
        num_experts,
        num_local_experts,
        group,
        num_topk,
        pad_multiple,
        num_permuted_tokens,
        async_finish,
        allocate_on_comm_stream,
        num_worst_tokens,
        use_cuda_num_tokens_per_expert,
    )
    raise NotImplementedError(
        "MORI balanced_moe uses the compact-row MORI native path; "
        "raw top-k normal/cold dispatch+permute is a Primus-Turbo TurboEP ABI."
    )


def unpermute_combine_normal_topk_tokens(
    expert_output: torch.Tensor,
    state: BalancedMoeNormalTopKDispatchState,
    *,
    group,
    async_finish: bool | None = None,
    allocate_on_comm_stream: bool | None = None,
) -> torch.Tensor:
    """MORI does not implement the TurboEP-style raw top-k unpermute ABI."""

    del expert_output, state, group, async_finish, allocate_on_comm_stream
    raise NotImplementedError(
        "MORI balanced_moe uses the compact-row MORI native path; "
        "raw top-k normal/cold unpermute+combine is a Primus-Turbo TurboEP ABI."
    )


def _require_compact_op_method(op, name: str):
    method = getattr(op, name, None)
    if method is None:
        raise TypeError(
            "MORI balanced-MoE compact API requires an EpDispatchCombineOp-like "
            f"object exposing {name}()."
        )
    return method


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
    """Dispatch already-compact normal/cold rows through the MORI EP primitive.

    TorchTitan owns the hot/cold policy and may materialize the compact row
    stream before invoking MORI.  Keep this import boundary in
    ``mori.ops.balanced_moe`` so framework code can consume the balanced-MoE
    feature surface without reaching directly into lower-level op methods.
    """

    return _require_compact_op_method(op, "dispatch_standard_ep_compact_native")(
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
    """Combine already-compact MORI EP rows through the balanced-MoE module."""

    return _require_compact_op_method(op, "combine_standard_ep_compact_native")(
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


def dispatch_balanced_moe_compact(
    op,
    input: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor | None,
    indices: torch.Tensor,
    source_partition: BalancedMoeSourcePartition | Mapping[str, object],
    num_hot_slots: int | None = None,
    *,
    hot_slot_kind: str = "owner_compact",
    block_num: int = -1,
    rdma_block_num: int = -1,
    warp_per_block: int = -1,
    compute_hot_counts: bool = False,
):
    """Dispatch normal/cold routes plus compact helper-hot rows through MORI.

    The policy object and source partition are built in this module.  The row
    movement stays in :class:`mori.ops.dispatch_combine.EpDispatchCombineOp`,
    which owns the native MORI kernels.  This wrapper exists so framework
    integrations can import the complete balanced-MoE contract from
    ``mori.ops.balanced_moe`` instead of unpacking the partition into
    lower-level dispatch-combine calls.
    """

    return _require_compact_op_method(op, "dispatch_balanced_moe_compact")(
        input,
        weights,
        scales,
        indices,
        source_partition,
        num_hot_slots,
        hot_slot_kind=hot_slot_kind,
        block_num=block_num,
        rdma_block_num=rdma_block_num,
        warp_per_block=warp_per_block,
        compute_hot_counts=compute_hot_counts,
    )


def combine_balanced_moe_compact(
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
    """Combine MORI balanced-MoE compact rows back to source-rank order.

    This is the matching module-level API for
    :func:`dispatch_balanced_moe_compact`.  It delegates directly to the native
    dispatch-combine operator and does not materialize an intermediate
    selected-hot or token-major route layout.
    """

    return _require_compact_op_method(op, "combine_balanced_moe_compact")(
        expert_major_rows,
        expert_major_flat_positions,
        expert_major_to_rank_major_indices,
        recv_counts_rank_major,
        input_splits,
        output_splits,
        num_output_rows,
        block_num=block_num,
        warp_per_block=warp_per_block,
        return_flat_positions=return_flat_positions,
        top_scores_flat=top_scores_flat,
        top_k=top_k,
        flat_position_offset=flat_position_offset,
        token_output_rows=token_output_rows,
        return_token_output=return_token_output,
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
        and (not torch.distributed.is_available() or not torch.distributed.is_initialized())
    ):
        if send.numel() != recv.numel():
            raise RuntimeError("single-rank exchange send/recv sizes differ")
        recv.copy_(send.reshape_as(recv))
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized for owner-compact exchange")
    torch.distributed.all_to_all_single(
        recv,
        send,
        output_split_sizes=list(output_splits),
        input_split_sizes=list(input_splits),
        group=group,
    )


_SDMA_ALL2ALL_HANDLE_CACHE: dict[tuple[int, int, int, int], object] = {}


def _normalize_owner_compact_exchange_transport(transport: str | None) -> str:
    if transport is None:
        return OWNER_COMPACT_TORCH_A2A_TRANSPORT
    normalized = str(transport).strip().lower()
    if normalized in {
        "",
        "auto",
        "torch",
        "all_to_all_single",
        OWNER_COMPACT_TORCH_A2A_TRANSPORT,
    }:
        return OWNER_COMPACT_TORCH_A2A_TRANSPORT
    if normalized in {
        "mori_sdma",
        "mori_sdma_padded",
        "mori_native",
        "native",
        "native_hot_helper",
        OWNER_COMPACT_MORI_SDMA_PADDED_A2A_TRANSPORT,
    }:
        return OWNER_COMPACT_MORI_SDMA_PADDED_A2A_TRANSPORT
    raise ValueError(
        "unsupported owner-compact hot-row transport "
        f"{transport!r}; expected one of "
        f"{BALANCED_MOE_BACKEND_CAPABILITIES['owner_compact_exchange_transports']}"
    )


def _pack_peer_major_rows(
    rows: torch.Tensor,
    splits: tuple[int, ...],
    *,
    padded_rows_per_peer: int,
    split_offsets: tuple[int, ...] | None = None,
) -> torch.Tensor:
    split_offsets = _offset_tuple(
        split_offsets,
        splits,
        name="input_split_offsets",
    )
    packed = rows.new_zeros(
        (len(splits) * int(padded_rows_per_peer), *rows.shape[1:])
    )
    for peer, count in enumerate(splits):
        count = int(count)
        if count > int(padded_rows_per_peer):
            raise ValueError(
                f"split for peer {peer} has {count} rows, exceeding padded "
                f"rows per peer {padded_rows_per_peer}"
            )
        if count > 0:
            dst_start = int(peer) * int(padded_rows_per_peer)
            src_start = int(split_offsets[peer])
            src_end = int(split_offsets[peer + 1])
            packed[dst_start : dst_start + count].copy_(rows[src_start:src_end])
    if int(split_offsets[-1]) != int(rows.shape[0]):
        raise ValueError(
            f"splits describe {int(split_offsets[-1])} rows, but rows tensor has "
            f"{int(rows.shape[0])}"
        )
    return packed


def _unpack_peer_major_rows(
    packed: torch.Tensor,
    splits: tuple[int, ...],
    *,
    padded_rows_per_peer: int,
    split_offsets: tuple[int, ...] | None = None,
) -> torch.Tensor:
    split_offsets = _offset_tuple(
        split_offsets,
        splits,
        name="output_split_offsets",
    )
    rows = packed.new_empty((sum(int(v) for v in splits), *packed.shape[1:]))
    for peer, count in enumerate(splits):
        count = int(count)
        if count > int(padded_rows_per_peer):
            raise ValueError(
                f"split for peer {peer} has {count} rows, exceeding padded "
                f"rows per peer {padded_rows_per_peer}"
            )
        if count > 0:
            src_start = int(peer) * int(padded_rows_per_peer)
            dst_start = int(split_offsets[peer])
            dst_end = int(split_offsets[peer + 1])
            rows[dst_start:dst_end].copy_(packed[src_start : src_start + count])
    return rows


def _u32_view_flat(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    num_bytes = int(tensor.numel()) * int(tensor.element_size())
    if num_bytes % 4 != 0:
        raise ValueError(
            "mori_sdma_padded_all2all requires a packed hot-row byte size "
            f"divisible by 4, got {num_bytes} bytes"
        )
    return tensor.view(torch.uint32).reshape(-1)


def _device_cache_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(
            "mori_sdma_padded_all2all requires CUDA/ROCm tensors; got "
            f"device {device}"
        )
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _cached_all2all_sdma(
    *,
    rank: int,
    world_size: int,
    count_u32_per_peer: int,
    device: torch.device,
):
    from mori.ccl import All2allSdma

    device_idx = _device_cache_index(device)
    key = (
        int(rank),
        int(world_size),
        int(count_u32_per_peer),
        int(device_idx),
    )
    handle = _SDMA_ALL2ALL_HANDLE_CACHE.get(key)
    if handle is None:
        total_bytes = int(world_size) * int(count_u32_per_peer) * 4
        handle = All2allSdma(
            int(rank),
            int(world_size),
            input_buffer_size=total_bytes,
            output_buffer_size=total_bytes,
            copy_output_to_user=True,
        )
        _SDMA_ALL2ALL_HANDLE_CACHE[key] = handle
    return handle


def _mori_sdma_padded_all_to_all(
    send: torch.Tensor,
    *,
    input_splits: tuple[int, ...],
    output_splits: tuple[int, ...],
    input_split_offsets: tuple[int, ...] | None = None,
    output_split_offsets: tuple[int, ...] | None = None,
    group,
) -> torch.Tensor:
    if len(input_splits) != len(output_splits):
        raise ValueError("input_splits and output_splits must have the same length")
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
    world_size = len(input_splits)
    if send.device.type != "cuda":
        raise ValueError(
            "mori_sdma_padded_all2all requires CUDA/ROCm tensors; got "
            f"{send.device}"
        )
    if world_size == 1 and (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
    ):
        return send.reshape_as(send)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized for MORI SDMA owner-row exchange"
        )
    rank = int(torch.distributed.get_rank(group=group))
    distributed_world_size = int(torch.distributed.get_world_size(group=group))
    if distributed_world_size != world_size:
        raise ValueError(
            "split metadata world size does not match distributed group size: "
            f"{world_size} != {distributed_world_size}"
        )
    padded_rows_per_peer = max(
        max((int(v) for v in input_splits), default=0),
        max((int(v) for v in output_splits), default=0),
    )
    recv_row_count = sum(int(v) for v in output_splits)
    if padded_rows_per_peer <= 0:
        return send.new_empty((recv_row_count, *send.shape[1:]))

    send_padded = _pack_peer_major_rows(
        send,
        input_splits,
        padded_rows_per_peer=padded_rows_per_peer,
        split_offsets=input_split_offsets,
    )
    recv_padded = send_padded.new_empty(send_padded.shape)
    send_u32 = _u32_view_flat(send_padded)
    recv_u32 = _u32_view_flat(recv_padded)
    if send_u32.numel() != recv_u32.numel():
        raise RuntimeError("packed SDMA send/recv u32 views are not the same size")
    count_u32_per_peer = int(send_u32.numel()) // int(world_size)
    if count_u32_per_peer * int(world_size) != int(send_u32.numel()):
        raise RuntimeError("packed SDMA buffer is not evenly divisible by world size")

    handle = _cached_all2all_sdma(
        rank=rank,
        world_size=world_size,
        count_u32_per_peer=count_u32_per_peer,
        device=send.device,
    )
    handle(send_u32, recv_u32, count_u32_per_peer)
    return _unpack_peer_major_rows(
        recv_padded,
        output_splits,
        padded_rows_per_peer=padded_rows_per_peer,
        split_offsets=output_split_offsets,
    )


def prepare_owner_compact_needed_rows_runtime_plan(
    *,
    compact_local_indices: torch.Tensor | Sequence[int],
    plan: BalancedMoeOwnerCompactExchangePlan | Mapping[str, object],
    device: torch.device | str | None = None,
    transport: str | None = None,
) -> dict[str, object]:
    """Prepare reusable local-index metadata for needed-row exchange.

    ``BalancedMoeOwnerCompactExchangePlan`` is compact-owner-offset based so it
    is portable across frameworks.  The hot-weight movement kernel ultimately
    needs local expert row indices for the current rank.  This helper performs
    that rank-local translation once, then callers can reuse the returned
    mapping for W2/W1/W3 hot-weight banks without rebuilding the same tiny
    index tensors.
    """

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
    if send_offsets.numel() > 0:
        runtime_plan["send_local_indices"] = compact_local_indices_t.index_select(
            0,
            send_offsets,
        )
    else:
        runtime_plan["send_local_indices"] = torch.empty(
            0,
            device=device_t,
            dtype=torch.long,
        )

    grad_recv_offsets = _tensor_or_empty(
        _plan_get(runtime_plan, "grad_recv_owner_compact_offsets", None),
        device=device_t,
    )
    if grad_recv_offsets.numel() > 0:
        runtime_plan["grad_recv_local_indices"] = compact_local_indices_t.index_select(
            0,
            grad_recv_offsets,
        )
    else:
        runtime_plan["grad_recv_local_indices"] = torch.empty(
            0,
            device=device_t,
            dtype=torch.long,
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
    """Build serial and runtime owner-compact hot-weight exchange plans.

    Framework callers normally derive the hot/cold/helper partition once in
    forward.  This helper keeps the next rank-local translation inside MORI:
    compact owner-row offsets become local expert row indices exactly once,
    then the returned runtime mapping can be reused for multiple hot-weight
    banks.
    """

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
    """Build serial and runtime owner-compact exchange plans from a MoE plan.

    This is the direct native MORI handoff for callers that already built a
    :class:`BalancedMoePlan` in forward.  It keeps the compact-offset to local
    row-index translation inside MORI and returns a reusable runtime mapping
    for all hot-weight banks.
    """

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

    Forward moves compact owner rows to helper ranks.  Backward sends helper
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
        ctx.transport = _normalize_owner_compact_exchange_transport(transport)

        if send_owner_compact_offsets.numel() > 0:
            if int(send_local_indices.numel()) != int(
                send_owner_compact_offsets.numel()
            ):
                send_local_indices = compact_local_indices.index_select(
                    0, send_owner_compact_offsets
                )
            send = local_rows.index_select(0, send_local_indices).contiguous()
        else:
            send = local_rows.new_empty((0, *local_rows.shape[1:]))
            send_local_indices = torch.empty(
                0, device=local_rows.device, dtype=torch.long
            )

        if grad_recv_owner_compact_offsets.numel() > 0:
            if int(grad_recv_local_indices.numel()) != int(
                grad_recv_owner_compact_offsets.numel()
            ):
                grad_recv_local_indices = compact_local_indices.index_select(
                    0, grad_recv_owner_compact_offsets
                )
        else:
            grad_recv_local_indices = torch.empty(
                0, device=local_rows.device, dtype=torch.long
            )
        ctx.save_for_backward(grad_send_needed_indices, grad_recv_local_indices)

        if ctx.transport == OWNER_COMPACT_MORI_SDMA_PADDED_A2A_TRANSPORT:
            recv = _mori_sdma_padded_all_to_all(
                send,
                output_splits=output_splits,
                input_splits=input_splits,
                output_split_offsets=output_split_offsets,
                input_split_offsets=input_split_offsets,
                group=group,
            )
        else:
            recv = local_rows.new_empty((sum(output_splits), *local_rows.shape[1:]))
            _dist_all_to_all_single(
                recv,
                send,
                output_splits=output_splits,
                input_splits=input_splits,
                group=group,
            )

        needed_rows = local_rows.new_empty((int(needed_row_count), *local_rows.shape[1:]))
        if recv_to_needed_index.numel() > 0:
            needed_rows.index_copy_(0, recv_to_needed_index, recv)
        return needed_rows

    @staticmethod
    def backward(ctx, grad_needed_rows: torch.Tensor):
        grad_send_needed_indices, grad_recv_local_indices = ctx.saved_tensors
        grad_needed_rows = grad_needed_rows.contiguous()
        if grad_send_needed_indices.numel() > 0:
            send = grad_needed_rows.index_select(
                0, grad_send_needed_indices.to(grad_needed_rows.device)
            ).contiguous()
        else:
            send = grad_needed_rows.new_empty((0, *grad_needed_rows.shape[1:]))

        if ctx.transport == OWNER_COMPACT_MORI_SDMA_PADDED_A2A_TRANSPORT:
            recv = _mori_sdma_padded_all_to_all(
                send,
                output_splits=ctx.input_splits,
                input_splits=ctx.output_splits,
                output_split_offsets=ctx.grad_output_split_offsets,
                input_split_offsets=ctx.grad_input_split_offsets,
                group=ctx.group,
            )
        else:
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

    This is the first-class MORI balanced-MoE ABI for hot-weight movement:
    callers pass compact owner-row metadata from
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
    transport = _normalize_owner_compact_exchange_transport(
        str(_plan_get(plan, "transport", transport))
        if _plan_get(plan, "transport", transport) is not None
        else None
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
    """Build a MindSpeed-style hot-expert relocation plan.

    Args:
        counts_owner_source_local: route counts with shape
            ``[owner_rank, source_rank, local_expert]``.  Element
            ``[o, s, e]`` is the number of rows sourced by rank ``s`` and
            normally owned by local expert ``e`` on rank ``o``.
        hot_expert_num: maximum number of hot experts to helper-execute.
        min_reduction_pct: if the modeled max-load reduction is below this
            threshold, return an empty plan.  This keeps the decision inside
            MORI so callers do not all invent different threshold behavior.
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
        if not modeled_owner_load:
            break
        owner = int(max(range(int(world_size)), key=lambda idx: modeled_owner_load[idx]))
        local_counts = expert_counts[owner]
        if local_counts.numel() == 0:
            break
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
        owner_slot = owner_counts[owner]
        owner_counts[owner] += 1
        owner_slots.append(owner_slot)

    max_owned_per_rank = max(owner_counts, default=0)
    owner_shard_offsets = [
        int(owner) * int(max_owned_per_rank) + int(owner_slot)
        for owner_slot, (_global_expert, owner, *_rest) in zip(owner_slots, selected_raw)
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
    for compact_offset, (_global_expert, owner, local_expert, *_rest) in zip(
        owner_compact_offsets, selected_raw
    ):
        owner_compact_owner_ranks[compact_offset] = int(owner)
        owner_compact_local_experts[compact_offset] = int(local_expert)

    modeled_reduction = _load_reduction_pct(owner_load_before, modeled_owner_load)
    if modeled_reduction < float(min_reduction_pct):
        selected_raw = []
        owner_counts = [0 for _ in range(int(world_size))]
        owner_slots = []
        owner_shard_offsets = []
        owner_shard_active_offsets = ()
        owner_compact_offsets = []
        owner_compact_owner_ranks = []
        owner_compact_local_experts = []
        max_owned_per_rank = 0
        modeled_owner_load = [int(v) for v in owner_load_before]
        modeled_reduction = 0.0

    hot_items = tuple(
        BalancedMoeHotExpert(
            global_expert=int(global_expert),
            owner_rank=int(owner),
            local_expert=int(local_expert),
            source_rank_counts=by_source,
            rows_total=int(rows_total),
            remote_rows=int(remote_rows),
            owner_slot=int(owner_slot),
            owner_shard_offset=int(owner_shard_offset),
            owner_compact_offset=int(owner_compact_offset),
        )
        for (
            (global_expert, owner, local_expert, by_source, rows_total, remote_rows),
            owner_slot,
            owner_shard_offset,
            owner_compact_offset,
        ) in zip(selected_raw, owner_slots, owner_shard_offsets, owner_compact_offsets)
    )
    owner_compact_need_masks = [
        [False for _ in owner_shard_active_offsets] for _ in range(int(world_size))
    ]
    for item in hot_items:
        compact_offset = int(item.owner_compact_offset)
        for source_rank, count in enumerate(item.source_rank_counts):
            if int(source_rank) != int(item.owner_rank) and int(count) > 0:
                owner_compact_need_masks[int(source_rank)][compact_offset] = True

    return BalancedMoePlan(
        world_size=int(world_size),
        num_local_experts=int(num_local_experts),
        hot_experts=hot_items,
        owner_counts=_as_int_tuple(owner_counts),
        max_owned_per_rank=int(max_owned_per_rank),
        owner_shard_active_offsets=_as_int_tuple(owner_shard_active_offsets),
        owner_compact_owner_ranks=_as_int_tuple(owner_compact_owner_ranks),
        owner_compact_local_experts=_as_int_tuple(owner_compact_local_experts),
        owner_compact_need_masks=tuple(
            tuple(bool(v) for v in row) for row in owner_compact_need_masks
        ),
        owner_load_before=owner_load_before,
        exec_load_after=_as_int_tuple(modeled_owner_load),
        selected_rows_total=int(sum(item.rows_total for item in hot_items)),
        remote_rows_total=int(sum(item.remote_rows for item in hot_items)),
        modeled_max_load_reduction_pct=float(modeled_reduction),
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
    """Count this source rank's raw top-k routes by owner rank/local expert.

    Args:
        topk_ids: raw selected expert ids with any leading route shape.  Negative
            ids are ignored when ``ignore_negative`` is true, matching padded
            route tensors.
        num_local_experts: experts owned by each EP rank.
        world_size: EP world size.  If omitted, a distributed process group must
            be initialized so the backend can read it.
        group: optional distributed group used only to infer world size.
        ignore_negative: ignore padded negative ids instead of rejecting them.

    Returns:
        Int64 tensor shaped ``[owner_rank, local_expert]``.  This is the local
        source-rank slice consumed by :func:`gather_local_counts_to_global`.
    """

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


def gather_local_counts_to_global(
    local_counts_source_local: torch.Tensor,
    group=None,
) -> torch.Tensor:
    """Gather local owner counts into ``[owner, source, local_expert]`` form.

    ``local_counts_source_local`` is this source rank's
    ``[owner_rank, local_expert]`` count slice.  ``all_gather`` naturally stacks
    source ranks first, so this helper transposes the stack before returning the
    planner's canonical owner-major shape.
    """

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
    """Build a balanced-MoE plan directly from raw top-k route ids.

    This is the backend-owned front door a framework should call before forming
    hot/cold/helper partitions.  It keeps raw route counting, owner/source count
    layout, and greedy hot-expert selection inside MORI.
    """

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
    """Gather local EP counts and build a native MORI balanced-MoE plan."""

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
    """Build source-rank hot-weight need masks for owner-compact layout.

    A source/helper rank needs compact hot-weight row ``c`` when it owns one or
    more routed rows for the corresponding hot expert, but is not the true
    expert owner.  This is the small policy object used by balanced-MoE forward
    hot-weight movement and backward hot-dW reduction planning.
    """

    world_size = int(world_size)
    active_owner_compact_count = int(active_owner_compact_count)
    if world_size < 0:
        raise ValueError(f"world_size must be non-negative, got {world_size}")
    if active_owner_compact_count < 0:
        raise ValueError(
            "active_owner_compact_count must be non-negative, got "
            f"{active_owner_compact_count}"
        )
    if (
        len(selected_source_rank_counts) != len(selected_owner_ranks)
        or len(selected_owner_ranks) != len(selected_owner_compact_offsets)
    ):
        raise ValueError(
            "selected_source_rank_counts, selected_owner_ranks, and "
            "selected_owner_compact_offsets must have matching lengths"
        )

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
        if compact_offset < 0 or compact_offset >= active_owner_compact_count:
            continue
        if owner_rank < 0 or owner_rank >= world_size:
            raise ValueError(
                f"owner_rank {owner_rank} is outside world_size={world_size}"
            )
        if len(by_source) != world_size:
            raise ValueError(
                "each selected_source_rank_counts row must have world_size "
                f"entries, got {len(by_source)} and world_size={world_size}"
            )
        for source_rank, count in enumerate(by_source):
            if int(source_rank) != owner_rank and int(count) > 0:
                need_masks[int(source_rank)][compact_offset] = True

    return tuple(tuple(bool(v) for v in row) for row in need_masks)


def build_owner_compact_exchange_plan(
    *,
    owner_compact_need_masks: Sequence[Sequence[bool]],
    owner_compact_owner_ranks: Sequence[int],
    rank: int,
) -> BalancedMoeOwnerCompactExchangePlan:
    """Build the per-rank owner-compact hot-weight exchange ABI.

    ``owner_compact_need_masks[source_rank][compact_idx]`` says that a source
    rank needs compact hot-weight row ``compact_idx``.  The corresponding
    owner rank for that row comes from ``owner_compact_owner_ranks``.

    Forward sends rows from their true owner to helper ranks; backward uses the
    same offsets in reverse to return partial weight gradients to the owner.
    """

    world_size = int(len(owner_compact_need_masks))
    rank = int(rank)
    if rank < 0 or rank >= world_size:
        raise IndexError(f"rank {rank} is outside world_size={world_size}")

    owner_ranks = _as_int_tuple(owner_compact_owner_ranks)
    active_count = int(len(owner_ranks))
    normalized_masks: list[tuple[bool, ...]] = []
    for source_rank, row in enumerate(owner_compact_need_masks):
        if len(row) != active_count:
            raise ValueError(
                "owner_compact_need_masks must be shaped "
                "[world_size, active_owner_compact_count]; row "
                f"{source_rank} has {len(row)} entries, expected {active_count}"
            )
        normalized_masks.append(tuple(bool(v) for v in row))
    for compact_idx, owner_rank in enumerate(owner_ranks):
        if int(owner_rank) < 0 or int(owner_rank) >= world_size:
            raise ValueError(
                f"owner rank {owner_rank} for compact row {compact_idx} "
                f"is outside world_size={world_size}"
            )

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
    """Build a per-rank hot-weight exchange plan from a full MoE plan."""

    return build_owner_compact_exchange_plan(
        owner_compact_need_masks=plan.owner_compact_need_masks,
        owner_compact_owner_ranks=plan.owner_compact_owner_ranks,
        rank=int(rank),
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
    """Build source-side normal/hot masks and compact owner offsets.

    ``remote_hot_mask`` marks routes selected for helper execution on this
    source rank.  ``normal_route_mask`` is the uint8 mask that existing MORI
    standard-MoE dispatch can use to keep only cold plus local-hot rows in the
    normal owner path.
    """

    if not isinstance(selected_experts_indices, torch.Tensor):
        raise TypeError("selected_experts_indices must be a torch.Tensor")
    if selected_experts_indices.dim() < 1:
        raise ValueError("selected_experts_indices must have at least one dimension")

    hot_global = _as_int_tuple(selected_global_experts)
    hot_owner = _as_int_tuple(selected_owner_ranks)
    if len(hot_global) != len(hot_owner):
        raise ValueError(
            "selected_global_experts and selected_owner_ranks must have "
            f"matching lengths, got {len(hot_global)} and {len(hot_owner)}"
        )
    if selected_owner_shard_offsets is None:
        hot_to_shard = tuple(range(len(hot_global)))
    else:
        hot_to_shard = _as_int_tuple(selected_owner_shard_offsets)
    if selected_owner_compact_offsets is None:
        hot_to_compact = tuple(range(len(hot_global)))
    else:
        hot_to_compact = _as_int_tuple(selected_owner_compact_offsets)
    if len(hot_to_shard) != len(hot_global) or len(hot_to_compact) != len(hot_global):
        raise ValueError(
            "selected owner offset lists must match selected_global_experts length"
        )
    presort_by = presort_by.strip().lower()
    if presort_by == "":
        presort_by = "off"
    if presort_by not in {"off", "selected", "owner_compact"}:
        raise ValueError(
            "presort_by must be 'off', 'selected', or 'owner_compact', "
            f"got {presort_by!r}"
        )

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

    remote_offsets_presorted_by = ""
    remote_group_ends = None
    if presort_by != "off" and remote_hot_offsets.numel() > 0:
        if presort_by == "owner_compact":
            presort_offsets = remote_owner_compact_offsets
            group_count = (max(hot_to_compact) + 1) if hot_to_compact else 0
        else:
            presort_offsets = remote_hot_offsets
            group_count = len(hot_global)
        presort_order = torch.argsort(presort_offsets, stable=True)
        remote_flat_positions = remote_flat_positions.index_select(0, presort_order)
        remote_hot_offsets = remote_hot_offsets.index_select(0, presort_order)
        remote_owner_shard_offsets = remote_owner_shard_offsets.index_select(
            0, presort_order
        )
        remote_owner_compact_offsets = remote_owner_compact_offsets.index_select(
            0, presort_order
        )
        remote_group_ends = torch.cumsum(
            torch.bincount(
                presort_offsets.to(torch.long),
                minlength=int(group_count),
            ).to(torch.int64),
            dim=0,
            dtype=torch.int32,
        )
        remote_offsets_presorted_by = presort_by

    top_k = int(selected_experts_indices.shape[-1])
    remote_token_indices = remote_flat_positions // max(1, top_k)
    normal_route_mask = keep_flat_mask.reshape_as(selected_experts_indices).to(torch.uint8)

    return BalancedMoeSourcePartition(
        keep_flat_mask=keep_flat_mask,
        remote_hot_mask=remote_hot_mask,
        normal_route_mask=normal_route_mask,
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
    """Build source-side normal/hot masks from a full balanced-MoE plan."""

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
    """Build the native MORI source layout for balanced-MoE execution.

    Callers that already have a :class:`BalancedMoePlan` can use this one
    helper to derive the normal route mask, helper-hot row partition, and
    reusable owner-compact hot-weight exchange metadata.  The default
    ``presort_by='owner_compact'`` matches the compact helper grouped-MLP ABI.
    """

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
