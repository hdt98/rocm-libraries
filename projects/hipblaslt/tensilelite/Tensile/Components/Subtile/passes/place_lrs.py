# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pass 0: Place MFMAs and Local Reads based on read granularities."""

from __future__ import annotations
from typing import List

from .._types import (
    LRPlacement,
    MFMAPlacement,
    MFMATileRange,
    SchedulerConfig,
    SubIterKSlot,
)
from ..ScheduleTypes import LogicalSchedule
from ._utils import partition_tile_range


def place_lrs(config: SchedulerConfig) -> LogicalSchedule:
    """Place MFMAs and LRs based on read granularities.

    Returns a list of partitions, each containing a list of SubIterKSlots.

    Each LR prefetches data for the next subIterK group. Within-partition
    prefetches use current partition tiles; cross-partition prefetches
    (wrapping) use next partition tiles.

    Two tracking mechanisms:
    - loaded_ranges: tracks tile ranges in VGPR per side. Wrapping LRs
      are only placed when the next partition's tiles aren't already loaded.
    - placed: tracks (tensor, k-range, tile-range) of non-wrapping LRs
      placed so far across partitions. Skips redundant K-prefetch when
      the same data was already loaded by an earlier partition.
    """
    numP = config.numPartitions
    part_ranges = [partition_tile_range(config, pi) for pi in range(numP)]

    # Track which tile ranges are currently loaded in VGPR (for wrapping decisions).
    loaded_ranges = {'A': {part_ranges[0]['A']},
                     'B': {part_ranges[0]['B']}}

    # Track placed K-prefetch LRs across partitions (for dedup).
    placed = set()

    partitions = []
    for pi in range(numP):
        cur, nxt = part_ranges[pi], part_ranges[(pi + 1) % numP]
        is_last = (pi == numP - 1)

        load = {}
        for side in ('A', 'B'):
            load[side] = is_last or nxt[side] not in loaded_ranges[side]

        slots = _place_lrs_for_partition(config, cur, nxt, is_last, load, placed)
        for slot in slots:
            for lr in slot.lrs:
                lr.partition = pi
        partitions.append(slots)

        for side in ('A', 'B'):
            if load[side]:
                loaded_ranges[side] = {cur[side], nxt[side]}

    return partitions


def _place_lrs_for_partition(config: SchedulerConfig, cur: tuple, nxt: tuple,
                             is_last: bool,
                             load: dict,
                             placed: set) -> List[SubIterKSlot]:
    """Place MFMAs and LRs for one partition."""
    numK = config.numSubIterK
    multi_part = config.numPartitions > 1

    slots = [SubIterKSlot(subIterK=k) for k in range(numK)]
    slot_mt = {}  # slot_k -> lr_mt, for MT-homogeneity enforcement

    # MFMAs
    for k in range(numK):
        slots[k].mfma = MFMAPlacement(
            subIterK=k,
            tileA=MFMATileRange(k, k + 1, cur['A'][0], cur['A'][1]),
            tileB=MFMATileRange(k, k + 1, cur['B'][0], cur['B'][1]),
        )

    # All tensors that can participate.
    all_tensors = [('A', config.lrA), ('B', config.lrB)]
    if config.hasScale:
        all_tensors.append(('SA', config.lrSA))
        all_tensors.append(('SB', config.lrSB))

    # Place LRs grouped by k_gran.
    # - Non-wrapping (K-prefetch): all tensors, deduped by placed set.
    # - Wrapping (cross-partition): only tensors whose side needs loading.
    for k_gran in sorted(set(g.k for _, g in all_tensors)):
        group_all = [(t, g) for t, g in all_tensors if g.k == k_gran]
        num_chunks = numK // k_gran
        for chunk_idx in range(num_chunks):
            next_chunk = (chunk_idx + 1) % num_chunks
            is_wrap = (next_chunk == 0)
            lr_mt = 1 if is_last and is_wrap else 0
            lr_k_start = next_chunk * k_gran
            lr_k_end = lr_k_start + k_gran
            base_slot = chunk_idx * k_gran

            # For wrapping chunks, only include tensors whose side is
            # loading so that slot assignment reflects active tensors.
            if is_wrap and multi_part:
                group = [(t, g) for t, g in group_all
                         if t in ('A', 'B') or load['A' if t in ('A', 'SA') else 'B']]
            else:
                group = group_all

            # Group by side (A/SA together, B/SB together) for slot assignment
            sides = [[(t, g) for t, g in group if t in ('A', 'SA')],
                     [(t, g) for t, g in group if t in ('B', 'SB')]]
            sides = [s for s in sides if s]

            for side_idx, side in enumerate(sides):
                slot_k = base_slot + (side_idx % k_gran)
                # Redirect LRs away from slots committed to a different MT,
                # keeping each slot MT-homogeneous.
                committed = slot_mt.get(slot_k)
                if committed is not None and committed != lr_mt:
                    slot_k = numK - 1

                for tensor, gran in side:
                    tile_range = nxt if (is_wrap or not multi_part) else cur
                    side_key = 'A' if tensor in ('A', 'SA') else 'B'
                    ts, te = tile_range[side_key]

                    # Wrapping: use load dict. Non-wrapping: use placed set.
                    if is_wrap and multi_part:
                        if not load[side_key]:
                            continue
                    else:
                        lr_key = (tensor, lr_k_start, lr_k_end, ts, te)
                        if lr_key in placed:
                            continue
                        placed.add(lr_key)

                    lr = LRPlacement(
                        tensor=tensor,
                        mtIteration=lr_mt,
                        tiles=MFMATileRange(lr_k_start, lr_k_end, ts, te),
                        subIterK_slot=slot_k,
                    )
                    slots[slot_k].lrs.append(lr)
                    slot_mt[slot_k] = lr_mt

    return slots
