# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pass 1: Assign physical VGPR tile IDs to all placements."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Dict, List

from .._types import (
    TENSOR_SIDE,
    SchedulerConfig,
)
from ..ScheduleTypes import LogicalSchedule


@dataclass
class VgprAllocation:
    """Result of the assign_vgpr_tiles pass."""
    unroll_factor: int
    needs_unrolling: bool
    tile_peaks: Dict[str, int]


def assign_vgpr_tiles(config: SchedulerConfig, schedule: LogicalSchedule) -> VgprAllocation:
    """Assign physical vgprTileIds to all placements (A, B, SA, SB).

    Modifies the schedule in-place (appends to vgpr_tile_maps and vgpr_tile_map).

    Free-list allocator with per-tensor FIFO queues, iterated until
    convergence (or max 8 unroll iterations).

    Three phases:
      1. Scan all MFMAs to find last read position for each
         (tensor, tileId, k_data_group) key.
      2. Walk execution order in a loop: each iteration feeds the
         previous next_iter as the starting active state.  Appends
         one tile-map dict per iteration to each placement's list.
         Stops when next_iter matches the seeded state (convergence).
      3. Record unroll_factor, needs_unrolling, and max tile_peaks.

    Returns a VgprAllocation with the allocation metadata.
    """
    tensors = config.tensors
    numK = config.numSubIterK
    MAX_UNROLL = 8

    lr_grans = {'A': config.lrA, 'B': config.lrB}
    if config.hasScale:
        lr_grans['SA'] = config.lrSA
        lr_grans['SB'] = config.lrSB

    # ── Phase 1: find last MFMA read for each key ──
    last_read = {}  # key -> flat position
    for pi, slots in enumerate(schedule):
        for slot in slots:
            if not slot.mfma:
                continue
            pos = pi * numK + slot.subIterK
            k = slot.subIterK
            for tensor in tensors:
                side = TENSOR_SIDE[tensor]
                tileRange = slot.mfma.tileA if side == 'A' else slot.mfma.tileB
                gran = lr_grans[tensor]
                for t in tileRange.tileId_list:
                    group = (t // gran.mn) * gran.mn
                    k_chunk = (k // gran.k) * gran.k
                    last_read[(tensor, group, k_chunk)] = pos

    # ── Phase 2: iterate until convergence ──
    class _FreeList:
        __slots__ = ('free', 'next_id', 'active_count', 'peak')
        def __init__(self):
            self.free = deque()
            self.next_id = 0
            self.active_count = 0
            self.peak = 0
        def alloc(self):
            if self.free:
                vid = self.free.popleft()  # FIFO for convergence
            else:
                vid = self.next_id
                self.next_id += 1
            self.active_count += 1
            self.peak = max(self.peak, self.active_count)
            return vid
        def release(self, vid):
            self.free.append(vid)
            self.active_count -= 1

    max_peaks = {t: 0 for t in tensors}
    carry_active = {}
    all_next_iters = []  # next_iter from each iteration, for cycle detection

    pools = {t: _FreeList() for t in tensors}

    for unroll_iter in range(MAX_UNROLL):
        if unroll_iter == 0:
            active = {}
        else:
            active = dict(carry_active)
            # Reset active_count to match carry_active (tiles that survived
            # as live from the previous iteration's wrapping LRs).
            for t in tensors:
                pools[t].active_count = sum(
                    1 for key in active if key[0] == t)

        next_iter = {}

        for pi, slots in enumerate(schedule):
            for slot in slots:
                pos = pi * numK + slot.subIterK
                k = slot.subIterK

                # ── MFMA reads: look up or seed ──
                if slot.mfma:
                    for tensor in tensors:
                        side = TENSOR_SIDE[tensor]
                        tileRange = slot.mfma.tileA if side == 'A' else slot.mfma.tileB
                        gran = lr_grans[tensor]
                        tile_map = {}
                        for t in tileRange.tileId_list:
                            group = (t // gran.mn) * gran.mn
                            k_chunk = (k // gran.k) * gran.k
                            key = (tensor, group, k_chunk)
                            if key not in active:
                                active[key] = pools[tensor].alloc()
                            tile_map[group] = active[key]
                        slot.mfma.vgpr_tile_maps.setdefault(tensor, []).append(tile_map)

                # ── LR writes: allocate new tiles ──
                for lr in slot.lrs:
                    tensor = lr.tensor
                    is_wrapping = lr.mtIteration != 0
                    target = next_iter if is_wrapping else active

                    gran = lr_grans[tensor]
                    tile_map = {}
                    seen_keys = set()
                    for t in lr.tiles.tileId_list:
                        group = (t // gran.mn) * gran.mn
                        for lk in lr.tiles.subIterK_list:
                            k_chunk = (lk // gran.k) * gran.k
                            key = (tensor, group, k_chunk)
                            if key in seen_keys:
                                continue
                            seen_keys.add(key)
                            if key in target:
                                pools[tensor].release(target[key])
                            vid = pools[tensor].alloc()
                            target[key] = vid
                            tile_map[group] = vid
                    lr.vgpr_tile_map.append(tile_map)

                # ── Release tiles whose last read was at this position ──
                to_release = [key for key, lr_pos in last_read.items()
                              if lr_pos == pos and key in active]
                for key in to_release:
                    pools[key[0]].release(active[key])
                    del active[key]

        # Track max peaks across iterations
        for t in tensors:
            max_peaks[t] = max(max_peaks[t], pools[t].peak)

        # Check convergence: if this iteration's next_iter matches
        # any previous iteration's next_iter, we found a cycle.
        converged = False
        for prev_idx, prev_ni in enumerate(all_next_iters):
            if next_iter == prev_ni:
                # Strip tile maps from the redundant convergence iteration.
                for pi2, slots2 in enumerate(schedule):
                    for slot2 in slots2:
                        if slot2.mfma:
                            for tensor in tensors:
                                if tensor in slot2.mfma.vgpr_tile_maps:
                                    slot2.mfma.vgpr_tile_maps[tensor].pop()
                        for lr2 in slot2.lrs:
                            lr2.vgpr_tile_map.pop()
                converged = True
                break
        if converged:
            break

        # Carry next_iter forward as active for next iteration
        all_next_iters.append(next_iter)
        carry_active = next_iter
    else:
        assert False, (f"assign_vgpr_tiles did not converge after "
                       f"{MAX_UNROLL} unroll iterations")

    # ── Phase 3: return results ──
    return VgprAllocation(
        unroll_factor=unroll_iter,
        needs_unrolling=unroll_iter > 1,
        tile_peaks=max_peaks,
    )
