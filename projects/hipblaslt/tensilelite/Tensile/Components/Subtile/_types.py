# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared type definitions for the LogicalScheduler pipeline.

All dataclasses, constants, and utility functions used across the scheduler
and its passes live here to avoid circular imports.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


TENSOR_SIDE = {'A': 'A', 'B': 'B', 'SA': 'A', 'SB': 'B'}

def fmt_mt(mt: int) -> str:
    """Format MT iteration integer as display string: 0 → 'n', 1 → 'n+1', 2 → 'n+2'."""
    return "n" if mt == 0 else f"n+{mt}"

# ── Core primitives ─────────────────────────────────────────

@dataclass
class MFMATileRange:
    """A rectangular range of MFMA tile coordinates for one read."""
    subIterK_start: int
    subIterK_end: int          # exclusive
    tileId_start: int
    tileId_end: int            # exclusive

    @property
    def subIterK_list(self) -> List[int]:
        return list(range(self.subIterK_start, self.subIterK_end))

    @property
    def tileId_list(self) -> List[int]:
        return list(range(self.tileId_start, self.tileId_end))

    def fmt_k(self) -> str:
        ids = self.subIterK_list
        if len(ids) == 1:
            return f"[{ids[0]}]"
        return f"[{ids[0]},{ids[-1]}]"

    def fmt_tiles(self) -> str:
        return f"[{self.tileId_start}-{self.tileId_end - 1}]"


# ── Config ──────────────────────────────────────────────────

@dataclass
class ReadGranularity:
    """Load granularity for one operation on one tensor, measured in MFMA tiles.

    mn: how many MFMA tiles in the M (for A/SA) or N (for B/SB) dimension
    k:  how many subIterK steps one read covers
    """
    mn: int
    k: int


@dataclass
class SchedulerConfig:
    """Configuration for the MFMATile-based scheduler."""
    numMFMATilesM: int    # MFMA tiles in M dimension (for A)
    numMFMATilesN: int    # MFMA tiles in N dimension (for B)
    numSubIterK: int      # subIterK steps within the macrotile
    lrA: ReadGranularity
    lrB: ReadGranularity
    grA: ReadGranularity
    grB: ReadGranularity
    lrSA: Optional[ReadGranularity] = None
    lrSB: Optional[ReadGranularity] = None
    grSA: Optional[ReadGranularity] = None
    grSB: Optional[ReadGranularity] = None
    numPartitionsM: int = 1   # partition grid in M dimension
    numPartitionsN: int = 1   # partition grid in N dimension

    @property
    def hasScale(self) -> bool:
        return self.lrSA is not None and self.lrSB is not None

    @property
    def tensors(self) -> List[str]:
        return ['A', 'B'] + (['SA', 'SB'] if self.hasScale else [])

    @property
    def numPartitions(self) -> int:
        return self.numPartitionsM * self.numPartitionsN

    @property
    def partitionSizeM(self) -> int:
        assert self.numMFMATilesM % self.numPartitionsM == 0
        return self.numMFMATilesM // self.numPartitionsM

    @property
    def partitionSizeN(self) -> int:
        assert self.numMFMATilesN % self.numPartitionsN == 0
        return self.numMFMATilesN // self.numPartitionsN

    @staticmethod
    def get_partition_candidates(tileInfoA, tileInfoB) -> list:
        """Return partition candidates as [(numPartitionsM, numPartitionsN), ...].

        Enumerates all divisors of MAX(M, N) in ascending order and
        partitions the larger dimension. Starts with (1, 1).
        This will only produces 1xN or Nx1 partitions to allow VGPR pressure reduction.
        """
        M = tileInfoA.localMMATileGrid[0]
        N = tileInfoB.localMMATileGrid[0]
        maxDim = max(M, N)

        divisors = sorted(d for d in range(1, maxDim + 1) if maxDim % d == 0)

        candidates = []
        for d in divisors:
            if N >= M:
                candidates.append((1, d))
            else:
                candidates.append((d, 1))

        return candidates



# ── Schedule operation types ────────────────────────────────

@dataclass
class Emittable:
    """Base for anything placed in an EmittedModule."""
    kind: str = field(init=False, default="")


@dataclass
class MFMAPlacement(Emittable):
    """MFMA operation consuming data for one subIterK."""
    subIterK: int
    tileA: MFMATileRange       # A tiles consumed
    tileB: MFMATileRange       # B tiles consumed
    deps: List['Dep'] = field(default_factory=list)      # populated by annotate_deps()
    preOps: List['BaseOp'] = field(default_factory=list)     # populated by remove_cross_deps()
    vgpr_tile_maps: Dict[str, List[dict]] = field(default_factory=dict)  # {tensor: [{groupIdx: vgprTileId}]} per unroll iter

    def __post_init__(self):
        self.kind = 'mfma'

    def __str__(self):
        return (f"MFMAs (MT n, subIterK {self.subIterK}  ) "
                f"A : {self.tileA.fmt_tiles()} , B : {self.tileB.fmt_tiles()}")


@dataclass
class LRPlacement(Emittable):
    """Local Read placement for one tensor in one subIterK slot."""
    tensor: str                # 'A', 'B', 'SA', 'SB'
    mtIteration: int           # 0 = current MT, 1 = next MT
    tiles: MFMATileRange
    subIterK_slot: int         # which subIterK this LR is placed in
    partition: int = 0         # which partition this LR belongs to
    deps: List['Dep'] = field(default_factory=list)      # populated by annotate_deps()
    preOps: List['BaseOp'] = field(default_factory=list)     # populated by remove_cross_deps()
    vgpr_tile_map: List[dict] = field(default_factory=list)  # [{tileId: vgprTileId}] per unroll iter

    def __post_init__(self):
        self.kind = 'lr'

    def __str__(self):
        return (f"LR {self.tensor.ljust(2)} (MT {fmt_mt(self.mtIteration)}, "
                f"subIterK {self.tiles.fmt_k()}) {self.tiles.fmt_tiles()}")


@dataclass
class GRPlacement(Emittable):
    """Global Read placement for one tensor in one subIterK slot."""
    tensor: str                # 'A', 'B', 'SA', 'SB'
    mtIteration: int           # 1 = next MT, 2 = two MTs ahead
    tiles: MFMATileRange
    subIterK_slot: int         # which subIterK this GR is placed in
    partition: int = 0         # which partition this GR belongs to
    deps: List['Dep'] = field(default_factory=list)      # populated by annotate_deps()
    preOps: List['BaseOp'] = field(default_factory=list)     # populated by remove_cross_deps()

    def __post_init__(self):
        self.kind = 'gr'

    def __str__(self):
        return (f"GR {self.tensor} (MT {fmt_mt(self.mtIteration)}, "
                f"subIterK {self.tiles.fmt_k()}) ids {self.tiles.fmt_tiles()}")


# ── Per-subIterK container ──────────────────────────────────

@dataclass
class SubIterKSlot:
    """All operations placed in one subIterK step."""
    subIterK: int
    mfma: Optional[MFMAPlacement] = None
    lrs: List[LRPlacement] = field(default_factory=list)
    grs: List[GRPlacement] = field(default_factory=list)


# ── Dependency types ────────────────────────────────────────

@dataclass
class WaitGRCounts:
    """Per-tensor inflight load counts for wait_gr preOp."""
    A: int = 0
    B: int = 0
    SA: int = 0
    SB: int = 0

    def __str__(self):
        parts = []
        for t in ('A', 'B', 'SA', 'SB'):
            v = getattr(self, t)
            if v:
                parts.append(f"{t}={v}")
        return ",".join(parts) if parts else "0"


@dataclass
class BaseOp(Emittable):
    """Base class for typed dependency operations in a before-chain."""

    def __str__(self):
        return self.kind


@dataclass
class WaitGROp(BaseOp):
    """Wait for global reads to complete. Optionally includes a sync barrier."""
    wait_gr_counts: Optional[WaitGRCounts] = None
    has_sync: bool = False

    def __post_init__(self):
        self.kind = 'wait_gr'

    def __str__(self):
        if self.wait_gr_counts:
            return f"{self.kind}({self.wait_gr_counts})"
        return self.kind


@dataclass
class WaitLROp(BaseOp):
    """Wait for local reads to complete. Optionally includes a sync barrier."""
    has_sync: bool = False

    def __post_init__(self):
        self.kind = 'wait_lr'


@dataclass
class SyncOp(BaseOp):
    """Standalone sync barrier."""
    def __post_init__(self):
        self.kind = 'sync'


@dataclass
class LRIncOp(BaseOp):
    """LDS buffer swap for local reads on a specific tensor."""
    tensor: str = ""

    def __post_init__(self):
        self.kind = 'lr_inc'

    def __str__(self):
        return f"lr_inc({self.tensor})"


@dataclass
class GRIncOp(BaseOp):
    """Pointer update + LDS swap for global reads on a specific tensor."""
    tensor: str = ""

    def __post_init__(self):
        self.kind = 'gr_inc'

    def __str__(self):
        return f"gr_inc({self.tensor})"


@dataclass
class SkipOp(BaseOp):
    """Skip guard: compare LoopCounter and branch."""
    compare: str = ""
    value: int = 0
    target: str = ""

    def __post_init__(self):
        self.kind = 'skip'

    @property
    def tensor(self) -> str:
        return f"{self.compare}:{self.value}:{self.target}"

    def __str__(self):
        return f"skip({self.tensor})"


@dataclass
class Dep:
    """Dependency on another placement (annotate_deps output)."""
    ref: Union[LRPlacement, GRPlacement]
    mt_offset: int = 0  # 0 = same MT, -1 = prev MT, -2 = two MTs back, ...




# ── Emitted output ─────────────────────────────────────────

@dataclass
class EmittedModule:
    """One emitted module with before-link for instruction scheduling.

    Compatible with SubtileBasedInstructionScheduler.instructionSchedule().
    Instructions are left empty at the logical level — filled during emission.
    """
    moduleId: int = -1
    instructions: list = field(default_factory=list)
    before: Optional[int] = None   # moduleId that must complete before this module
    source: Optional[Emittable] = None

    @property
    def opType(self) -> str:
        return self.source.kind if self.source else ""
