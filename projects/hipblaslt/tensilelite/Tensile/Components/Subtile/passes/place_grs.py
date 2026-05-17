# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pass 2: Place Global Reads based on GR granularities."""

from __future__ import annotations
from typing import Dict, List, Set, Tuple

from .._types import (
    GRPlacement,
    MFMATileRange,
    SchedulerConfig,
    fmt_mt,
)
from ..ScheduleTypes import LogicalSchedule
from ._utils import partition_tile_range


def place_grs(config: SchedulerConfig, schedule: LogicalSchedule) -> LogicalSchedule:
    """Place Global Reads by iterating MFMAs across partitions.

    Phase 1: Build ordered GR list from partition traversal respecting gr granularities.
    Phase 2: Distribute evenly GR atoms across all (partition, subIterK) slots.

    Returns the modified schedule.
    """
    part_ranges = [partition_tile_range(config, pi)
                   for pi in range(config.numPartitions)]

    # TODO: cover PGR3 (offsetMT and offsetPartition may differ)
    offsetMT = 1
    offsetPartition = 1

    gr_list = _build_gr_list(config, schedule, part_ranges, offsetMT, offsetPartition)
    lr_mt_n_info = _build_lr_conflict_map(schedule)
    _distribute_grs(config, schedule, gr_list, lr_mt_n_info)
    return schedule


def _build_gr_list(config: SchedulerConfig, schedule: LogicalSchedule,
                   part_ranges, offsetMT, offsetPartition, debug=False):
    """Phase 1: Build ordered GR list from placed MFMAs.

    For each partition × subIterK, derive target partition/MT from
    the MFMA and offsets. Add GRs (A, B, SA, SB) with tile and K
    ranges snapped to GR granularity. Dedup within same MT level,
    then remove n+1 entries that also appear at n+2 (cross-MT dedup).

    Returns list of (tensor, mt_str, tile_start, tile_end,
                     k_start, k_end, gr_gran).
    """
    numP = config.numPartitions

    seen: Set[tuple] = set()
    gr_list = []

    for pi in range(numP):
        partition_slots = schedule[pi]

        target_pi = (pi + offsetPartition) % numP
        wraps = (pi + offsetPartition) >= numP
        mt_val = offsetMT + (1 if wraps else 0)

        target_range = part_ranges[target_pi]

        for slot in partition_slots:
            k = slot.mfma.subIterK

            items = [('A', target_range['A'], config.grA),
                     ('B', target_range['B'], config.grB)]
            if config.hasScale:
                items.append(('SA', target_range['A'], config.grSA))
                items.append(('SB', target_range['B'], config.grSB))

            for tensor, (t_start, t_end), gr_gran in items:
                mn = gr_gran.mn
                k_gran = gr_gran.k

                gr_tile_start = (t_start // mn) * mn
                gr_tile_end = ((t_end + mn - 1) // mn) * mn

                gr_k_start = (k // k_gran) * k_gran
                gr_k_end = gr_k_start + k_gran

                key = (tensor, mt_val, gr_tile_start, gr_tile_end,
                       gr_k_start, gr_k_end)
                if key in seen:
                    continue
                seen.add(key)
                gr_list.append((tensor, mt_val, gr_tile_start,
                                gr_tile_end, gr_k_start, gr_k_end,
                                gr_gran))

    # Cross-MT dedup: if a tile/k range appears at both n+1 and n+2,
    # the n+1 load is redundant — the previous iteration's n+2 already
    # wrote the same data into LDS.  Remove the n+1 duplicate.
    base_mt = offsetMT
    n2_keys = {(t, ts, te, ks, ke)
               for t, mt, ts, te, ks, ke, _ in gr_list
               if mt != base_mt}
    gr_list = [entry for entry in gr_list
               if entry[1] != base_mt or
               (entry[0], entry[2], entry[3], entry[4], entry[5])
               not in n2_keys]

    if debug:
        print(f"Phase 1: {len(gr_list)} GR entries")
        for i, (t, mt, ts, te, ks, ke, g) in enumerate(gr_list):
            loads = ((te - ts) // g.mn) * ((ke - ks) // g.k)
            print(f"  [{i}] {t:2s} {fmt_mt(mt)} tiles[{ts},{te - 1}] k[{ks},{ke - 1}] "
                  f"gr_gran(mn={g.mn},k={g.k}) loads={loads}")

    return gr_list


def _build_lr_conflict_map(schedule: LogicalSchedule):
    """Build per-partition LR(MT n) info for LDS conflict checking.

    Returns dict: (partition_idx, tensor) -> list of
                  (subIterK_slot, k_start, k_end).
    """
    lr_mt_n_info: Dict[Tuple[int, str], List[Tuple[int, int, int]]] = {}
    for pi, partition_slots in enumerate(schedule):
        for slot in partition_slots:
            for lr in slot.lrs:
                if lr.mtIteration == 0:
                    lr_mt_n_info.setdefault((pi, lr.tensor), []).append(
                        (slot.subIterK,
                         lr.tiles.subIterK_start,
                         lr.tiles.subIterK_end))
    return lr_mt_n_info


def _has_lr_conflict(lr_mt_n_info, tensor, mt_val, pi, subIterK,
                     gr_k_start, gr_k_end):
    """Return True if placing GR(mt_val) at (pi, subIterK) conflicts.

    GR(MT n+2) writes the same LDS buffer as MT n, so it conflicts
    only if a later LR(MT n) in the same partition accesses an
    overlapping subIterK range.
    """
    if mt_val != 2:
        return False
    for lr_slot, lr_ks, lr_ke in lr_mt_n_info.get((pi, tensor), []):
        if lr_slot > subIterK and gr_k_start < lr_ke and lr_ks < gr_k_end:
            return True
    return False


def _distribute_grs(config: SchedulerConfig, schedule: LogicalSchedule,
                    gr_list, lr_mt_n_info, debug=False):
    """Phase 2: Distribute GR atoms across partition × subIterK slots.

    Explodes GR entries into atomic loads, distributes them into flat
    buckets respecting LDS conflict constraints and load balance,
    then remerges consecutive atoms and places them into partitions.
    """
    numK = config.numSubIterK
    numP = config.numPartitions
    numSlots = numP * numK

    # 2a. Explode GR entries into atomic loads (1 load each)
    atoms = []
    for tensor, mt_val, t_start, t_end, k_start, k_end, gr_gran in gr_list:
        mn = gr_gran.mn
        for pos in range(t_start, t_end, mn):
            atoms.append((tensor, mt_val, pos, pos + mn, k_start, k_end))

    loads_per_slot = len(atoms) // numSlots

    # 2b. Distribute atoms into flat buckets [0..numSlots),
    #     each bucket maps to (partition=flat//numK, subIterK=flat%numK)
    buckets: List[list] = [[] for _ in range(numSlots)]
    for atom in atoms:
        tensor, mt_val, _, _, ks, ke = atom
        cur = 0
        while cur < numSlots - 1:
            pi = cur // numK
            subK = cur % numK
            if (not _has_lr_conflict(lr_mt_n_info, tensor, mt_val,
                                     pi, subK, ks, ke) and
                    len(buckets[cur]) < loads_per_slot):
                break
            cur += 1
        buckets[cur].append(atom)

    if debug:
        print(f"Phase 2b: {len(atoms)} atoms, {numSlots} slots, "
              f"{loads_per_slot} per slot")
        for flat, bucket in enumerate(buckets):
            pi = flat // numK
            si = flat % numK
            if bucket:
                items = ", ".join(
                    f"{t} {fmt_mt(mt)} tile[{ts},{te-1}] k[{ks},{ke-1}]"
                    for t, mt, ts, te, ks, ke in bucket)
                print(f"  P{pi} s{si}: {len(bucket)} atoms — {items}")
            else:
                print(f"  P{pi} s{si}: empty")

    # 2c. Remerge consecutive atoms and place into partitions
    for flat, bucket in enumerate(buckets):
        pi = flat // numK
        si = flat % numK
        target_slot = schedule[pi][si]
        for atom in bucket:
            tensor, mt_val, ts, te, ks, ke = atom
            if target_slot.grs:
                prev = target_slot.grs[-1]
                if (prev.tensor == tensor and
                        prev.mtIteration == mt_val and
                        prev.tiles.subIterK_start == ks and
                        prev.tiles.subIterK_end == ke and
                        prev.tiles.tileId_end == ts):
                    prev.tiles = MFMATileRange(ks, ke, prev.tiles.tileId_start, te)
                    continue
            target_slot.grs.append(GRPlacement(
                tensor=tensor, mtIteration=mt_val,
                tiles=MFMATileRange(ks, ke, ts, te),
                subIterK_slot=si,
                partition=pi))
