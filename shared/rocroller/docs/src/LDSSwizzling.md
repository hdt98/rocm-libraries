# LDS Swizzle for Bank Conflict Elimination

Intra-wave column rotation + pair-swap to eliminate 4x LDS bank conflict
serialization during Local Read (`ds_read_b128`). Only column indices
are permuted; row assignments are unchanged.

Reference: Tensile's `SubtileBasedKernel.py` functions `_grSwizzleColIds` (GR)
and `lraTileAssignment` (LR). See the LDS swizzling presentation for background
on bank conflicts, `ds_read_b128` phases, and correctness proofs.

## Hardware Background (GFX950)

LDS has 64 banks x 4 bytes = 256 bytes per bank row, holding exactly
16 dwordx4 columns per bank row. When a tile row's K dimension spans
fewer than 16 columns, multiple tile rows pack into the same bank row
and reads from different rows at the same column offset hit the same
banks, causing serialization.

Other architectures have 32 banks (128-byte bank rows, 8 dwordx4 columns).

## Parameters

In the code (`LDSSwizzleParams` struct in `LowerTile.cpp`):

```
numColumns      = tileK / (128 / elementBits)     -- dwordx4 chunks per tile row
rowsPerBankRow  = columnsPerBankRow / numColumns   -- tile rows per bank row
elementsPerChunk = 128 / elementBits               -- elements per dwordx4 chunk
columnsPerBankRow = 16                             -- GFX950 constant
```

Per-lane terms:

```
bankRowIdx = row / rowsPerBankRow    -- which bank row a tile row belongs to
```

Examples (tile rows x columns = 16 dwordx4 per bank row):

```
FP4  macK=128:  4 cols/row x 4 rows/bank-row
FP4  macK=256:  8 cols/row x 2 rows/bank-row
FP8  macK=128:  8 cols/row x 2 rows/bank-row
FP16 macK=128: 16 cols/row x 1 row/bank-row  (no conflicts, swizzle skipped)
FP16 macK=64:   8 cols/row x 2 rows/bank-row
FP16 macK=32:   4 cols/row x 4 rows/bank-row
```

When `numColumns >= columnsPerBankRow`, each tile row fills an entire bank
row and there are no conflicts. The code skips swizzle in this case
(`LDSSwizzleParams::noConflicts()`).

## `ds_read_b128` Thread Phases

The LDS unit processes a `ds_read_b128` from a 64-lane wave in 4 phases,
each executing 16 threads simultaneously. The 16 threads in a phase access
LDS in parallel, so bank conflicts only occur between threads within the
same phase. The phase assignment follows the hardware SIMD layout:

| Phase | Threads                                   |
|-------|-------------------------------------------|
| 0     | T0-3, T12-15, T20-23, T24-27              |
| 1     | T32-35, T44-47, T52-55, T56-59            |
| 2     | T4-7, T8-11, T16-19, T28-31               |
| 3     | T36-39, T40-43, T48-51, T60-63            |

Each `ds_read_b128` reads 16 bytes (4 banks) per thread. With 16 threads
per phase and 64 banks total (16 bank groups of 4), zero bank conflicts
means each thread in a phase accesses a distinct bank group.

Without swizzling, threads within a phase that read the same K-column chunk
hit the same bank group, causing 4x serialization. The swizzle permutes
column assignments so that the 16 threads in each phase spread across all
16 bank groups.

## Permutation: Swap-then-Rotate

Both GR and LR use a swap-then-rotate permutation within `numColumns`:

```
bankRowIdx   = row / rowsPerBankRow
swapOnEvenRow = (bankRowIdx ^ 1) & 1           -- 1 on even, 0 on odd
swappedCol   = col ^ swapOnEvenRow
rotation     = numColumns - (bankRowIdx / 2) * 2
swizzledCol  = (swappedCol + rotation) & (numColumns - 1)
```

The LR side applies the inverse (un-rotate then un-swap):

```
invRotation    = (bankRowIdx / 2) * 2
unrotatedCol   = (col + invRotation) & (numColumns - 1)
swizzledCol    = unrotatedCol ^ swapOnEvenRow
```

This produces `numColumns` distinct permutations, enough to differentiate
all tile rows sharing a bank row.

## Implementation in rocRoller

Gated by `LDSBankSwizzleMode` kernel option (None, Swizzle). Both GR
and LR transforms are expressed as `ExpressionTransform`
edges in the coordinate graph. No new instruction emission or cross-lane
operations needed.

### ExpressionTransform edge

A coordinate transform edge carrying arbitrary forward and reverse
`ExpressionPtr` trees. Uses `PositionalArgument` slots (`$0`, `$1`, ...)
as placeholders for input dimension indexes. The visitor calls
`positionalArgumentPropagation(expr, indexes)` to substitute actual index
expressions at transform time.

The row coordinate is passed as a downstream neighbor (`positionalArgument(1)`)
on the graph edge, not via `DataFlowTag`.

### GR ExpressionTransform

Positional arguments:
- `$0` = input K-column chunk index (ThreadTileNumber for K dimension)
- `$1` = tile row coordinate (ThreadTileNumber for the non-K dimension)

The expression includes group splitting (`columnInGroup` / `groupIndex`)
so that the output has the form `groupIndex * numColumns + swizzledInGroup`.
This algebraic structure is required for the `Tile` edge to correctly invert
the expression during coordinate graph traversal. Without it, the Tile edge
falls back to an incorrect bitmask.

Graph topology (MATRIX_B, swizzle enabled):

```
  Workitem --Flatten--> {nThrX, nThrY}
  {nThrX, nThrY} --ExpressionTransform--> grSwizzleNThrX
  grSwizzleNThrX, iThrX --Tile--> iMacX
```

For MATRIX_A, the roles of X/Y are swapped (K is dim 1).

### LR ExpressionTransform

Positional arguments:
- `$0` = element index within K (iMacY for MATRIX_B, iMacX for MATRIX_A)
- `$1` = tile row coordinate (iMacX for MATRIX_B, iMacY for MATRIX_A)

The element index is decomposed into a dwordx4 chunk index and sub-element
offset; only the chunk's within-group bits are swizzled, then reassembled:

```
fullChunk      = elemIdx / elementsPerChunk
subElement     = elemIdx % elementsPerChunk
chunkInGroup   = fullChunk % numColumns
chunkGroupIdx  = fullChunk / numColumns
// ... apply inverse permutation to chunkInGroup ...
swizzledElem   = (chunkGroupIdx * numColumns + swizzledInGroup)
                 * elementsPerChunk + subElement
```

Graph topology (swizzle enabled):

```
  ldsTag --Tile--> {iMacX, rawIMacY}
  {rawIMacY} --ExpressionTransform--> {colCoord, rowCoord}
```

### Verification tables

GR verification (numColumns=8, rowsPerBankRow=2):

| lane | base_col | bankRowIdx | swap_col | gr_rotation | gr_col | expected   |
|------|----------|------------|----------|-------------|--------|------------|
| 0    | 0        | 0 (even)   | 1        | 8           | 1      | K1 at L0   |
| 1    | 1        | 0 (even)   | 0        | 8           | 0      | K0 at L1   |
| 8    | 0        | 0 (even)   | 1        | 8           | 1      | K1 at L8   |
| 9    | 1        | 0 (even)   | 0        | 8           | 0      | K0 at L9   |
| 16   | 0        | 1 (odd)    | 0        | 8           | 0      | K0 at L16  |
| 17   | 1        | 1 (odd)    | 1        | 8           | 1      | K1 at L17  |
| 32   | 0        | 2 (even)   | 1        | 6           | 7      | K7 at L32  |
| 33   | 1        | 2 (even)   | 0        | 6           | 6      | K6 at L33  |
| 35   | 3        | 2 (even)   | 2        | 6           | 0      | K0 at L35  |
| 48   | 0        | 3 (odd)    | 0        | 6           | 6      | K6 at L48  |
| 50   | 2        | 3 (odd)    | 2        | 6           | 0      | K0 at L50  |
| 56   | 0        | 3 (odd)    | 0        | 6           | 6      | K6 at L56  |
| 63   | 7        | 3 (odd)    | 7        | 6           | 5      | K5 at L63  |

### Follow-up: Inter-wave rotation

For certain wave group configurations (e.g., 2x2 `MIWaveGroup` where multiple waves
contribute to the same LDS region), an additional per-wave column rotation may be
needed on the GR side:

```
wave_rotation = (waveId & 1) * (2 * numRowsPerBankRow)
col = (col + intra_rotation - wave_rotation) % numColumns
```

This applies when `loadRatioGR != 0.5` in Tensile's terminology. For the primary
target (256x256x256 with 4x1 wave group, `loadRatioGR = 0.5`), intra-wave
rotation alone is sufficient.

Reference: `_grSwizzleColIds` in `SubtileBasedKernel.py` lines 778-812.

### Follow-up: Non-FP4 data types

The swizzle logic is data-type agnostic -- `numColumns = tileK / (128 / elementBits)`
generalizes across FP4, FP8, FP16, etc. The `noConflicts()` check automatically
skips swizzle when `numColumns >= 16` (e.g., FP16 macK=128). Testing with non-FP4
data types is the next step.
