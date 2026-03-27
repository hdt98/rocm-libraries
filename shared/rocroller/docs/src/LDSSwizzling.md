# LDS Swizzle for Bank Conflict Elimination

Intra-wave column rotation + pair-swap to eliminate 4x LDS bank conflict
serialization during Local Read (`ds_read_b128`). The GR row assignment stays
the same (8 contiguous rows per wave, no half-wave split); only column indices
are permuted.

Reference: Tensile's `SubtileBasedKernel.py` functions `_grSwizzleColIds` (GR)
and `lraTileAssignment` (LR). See the LDS swizzling presentation for background
on bank conflicts, `ds_read_b128` phases, and correctness proofs.

## Parameters

```
macKBytes         = macK * bpe
loadWidth         = 16 B   (dwordx4)
kChunksPerRow     = macKBytes / loadWidth           (K-columns per row, in dwordx4 units)
ldsRowBankSize    = 64 banks * 4 B = 256 B          (one full LDS bank row)
numRowsPerBankRow = ldsRowBankSize / macKBytes      (matrix rows per LDS bank row)
rowsPerWave       = 64 / kChunksPerRow
```

Per-lane terms (rocRoller coordinate graph names in parentheses):

```
laneInSIMD  = lane % 16    (Lane dimension, position within 16-lane SIMD)
SIMDIndex   = lane // 16   (Adhoc("SIMDIndex"), which SIMD, 0-3)
ldsRowId    = laneInSIMD / numRowsPerBankRow              (for LR)
            = lane / (kChunksPerRow * numRowsPerBankRow)  (for GR)
```

Example (macK=256 fp4, `v_mfma_scale_f32_16x16x128_f8f6f4`):

```
macKBytes         = 256 * 0.5 = 128 B
kChunksPerRow     = 128 / 16 = 8
numRowsPerBankRow = 256 / 128 = 2
rowsPerWave       = 64 / 8 = 8
```

## Implementing in rocRoller

Gated by a new compile-time kernel option (`LDSBankSwizzleMode`). Two sides need to change:
GR column index and LR column index. Both can be expressed as pure per-lane arithmetic
in the coordinate graph using a new `ExpressionTransform` edge type. No new instruction
emission or cross-lane operations needed -- the pair-swap reduces to `col ^ 1` since
column values are sequential. All parameters are derived from the tile config.

**Note:** The 256x256x256 tile requires Direct2LDS; without it the kernel runs out of VGPRs
and must use a smaller tile. However, the implementation should work for any tile size that
exhibits this bank conflict pattern. Development strategy: implement and verify on a smaller
tile first (without Direct2LDS), then confirm it scales up to 256x256x256 with Direct2LDS.

### ExpressionTransform edge

A new coordinate transform edge type that carries arbitrary forward and reverse
`ExpressionPtr` trees. Uses `PositionalArgument` slots (`$0`, `$1`, ...) as
placeholders for input dimension indexes. The visitor calls
`positionalArgumentPropagation(expr, indexes)` to substitute actual index
expressions at transform time.

```cpp
struct ExpressionTransform
{
    using ExpressionPtr = Expression::ExpressionPtr;

    std::vector<ExpressionPtr> forward;  // y = forward($0, $1, ...)
    std::vector<ExpressionPtr> reverse;  // x = reverse($0, $1, ...)

    // ...name(), toString()
};
```

This is the same mechanism used by `PiecewiseAffineJoin::condition`, generalized
to the entire transform. Diff visitors throw (not needed for swizzling).

### Swap-if-even pattern

Both GR and LR use the same conditional pair-swap:

```
swap_if_even(col, ldsRowId) = (ldsRowId % 2 == 0) ? (col ^ 1) : col
```

On the GR side, `quad_perm([1,0,3,2])` on sequential column values is equivalent
to `col ^ 1`. On the LR side, this replaces the `v_permlane16_swap_b32` -- the
swap between SIMDIndex groups produces the same result as `col ^ 1` because the
pre-swap column values differ by exactly 1 between adjacent groups.

### GR ExpressionTransform

Positional arguments (source dimensions):
- `$0` = `base_col`: sequential column assignment (= `lane % kChunksPerRow`).
  Coordinate graph dimension: the fast-moving component after tiling the
  Workitem/Lane into row and column within the wave.
- `$1` = `ldsRowId`: LDS bank row index (= `(lane / kChunksPerRow) / numRowsPerBankRow`).
  Coordinate graph dimension: derived from the row component of the lane
  decomposition.

Forward expression (1 output: swizzled GR column):

```
swap_col    = ($1 % 2 == 0) ? ($0 ^ 1) : $0
gr_rotation = kChunksPerRow - ($1 / 2) * 2
gr_col      = (swap_col + gr_rotation) % kChunksPerRow
```

As an `ExpressionPtr` tree (using rocRoller Expression API):

```
$0 = PositionalArgument(0, VGPR, Int32)    // base_col
$1 = PositionalArgument(1, VGPR, Int32)    // ldsRowId
K  = literal(kChunksPerRow)

swap_col    = Conditional($1 & 1 == 0, $0 ^ 1, $0)
gr_rotation = K - ($1 >> 1) << 1
gr_col      = (swap_col + gr_rotation) & (K - 1)   // K is power of 2
```

Verification:

| lane | base_col | ldsRowId | swap_col | gr_rotation | gr_col | expected   |
|------|----------|----------|----------|-------------|--------|------------|
| 0    | 0        | 0 (even) | 1        | 8           | 1      | K1 at L0   |
| 1    | 1        | 0 (even) | 0        | 8           | 0      | K0 at L1   |
| 8    | 0        | 0 (even) | 1        | 8           | 1      | K1 at L8   |
| 9    | 1        | 0 (even) | 0        | 8           | 0      | K0 at L9   |
| 16   | 0        | 1 (odd)  | 0        | 8           | 0      | K0 at L16  |
| 17   | 1        | 1 (odd)  | 1        | 8           | 1      | K1 at L17  |
| 32   | 0        | 2 (even) | 1        | 6           | 7      | K7 at L32  |
| 33   | 1        | 2 (even) | 0        | 6           | 6      | K6 at L33  |
| 35   | 3        | 2 (even) | 2        | 6           | 0      | K0 at L35  |
| 48   | 0        | 3 (odd)  | 0        | 6           | 6      | K6 at L48  |
| 50   | 2        | 3 (odd)  | 2        | 6           | 0      | K0 at L50  |
| 56   | 0        | 3 (odd)  | 0        | 6           | 6      | K6 at L56  |
| 63   | 7        | 3 (odd)  | 7        | 6           | 5      | K5 at L63  |

### LR ExpressionTransform

Positional arguments (source dimensions):
- `$0` = `SIMDIndex`: SIMD index within wave (= `lane / 16`, range 0-3).
  Coordinate graph dimension: `Adhoc("SIMDIndex")` from the Lane
  decomposition in `addLoadSwizzleTileCT` / `addLoadWaveTileCT`.
- `$1` = `ldsRowId`: LDS bank row index (= `laneInSIMD / numRowsPerBankRow`).
  Coordinate graph dimension: derived from `Lane` (laneInSIMD = `lane % 16`).

Forward expression (1 output: swizzled LR column):

```
rotation = ($1 / 2) * 2
base_col = ($0 + rotation) % kChunksPerRow
lr_col   = ($1 % 2 == 0) ? (base_col ^ 1) : base_col
```

As an `ExpressionPtr` tree:

```
$0 = PositionalArgument(0, VGPR, Int32)    // SIMDIndex
$1 = PositionalArgument(1, VGPR, Int32)    // ldsRowId
K  = literal(kChunksPerRow)

rotation = ($1 >> 1) << 1
base_col = ($0 + rotation) & (K - 1)       // K is power of 2
lr_col   = Conditional($1 & 1 == 0, base_col ^ 1, base_col)
```

Verification:

| laneInSIMD | SIMDIndex ($0) | ldsRowId ($1) | rotation | base_col | lr_col | expected |
|------------|----------------|---------------|----------|----------|--------|----------|
| 0          | 0              | 0 (even)      | 0        | 0        | 1      | 1        |
| 0          | 1              | 0 (even)      | 0        | 1        | 0      | 0        |
| 0          | 2              | 0 (even)      | 0        | 2        | 3      | 3        |
| 0          | 3              | 0 (even)      | 0        | 3        | 2      | 2        |
| 2          | 0              | 1 (odd)       | 0        | 0        | 0      | 0        |
| 3          | 2              | 1 (odd)       | 0        | 2        | 2      | 2        |
| 4          | 0              | 2 (even)      | 2        | 2        | 3      | 3        |
| 4          | 1              | 2 (even)      | 2        | 3        | 2      | 2        |
| 4          | 3              | 2 (even)      | 2        | 5        | 4      | 4        |
| 6          | 0              | 3 (odd)       | 2        | 2        | 2      | 2        |
| 8          | 0              | 4 (even)      | 4        | 4        | 5      | 5        |
| 8          | 2              | 4 (even)      | 4        | 6        | 7      | 7        |
| 10         | 1              | 5 (odd)       | 4        | 5        | 5      | 5        |
| 12         | 0              | 6 (even)      | 6        | 6        | 7      | 7        |
| 14         | 0              | 7 (odd)       | 6        | 6        | 6      | 6        |
| 14         | 3              | 7 (odd)       | 6        | 1        | 1      | 1        |

### Coordinate graph integration: GR side (StoreLDSTile)

`addStoreThreadTileCT` in `LowerTile.cpp:1258` decomposes the Workitem into
the tile's row/column structure. For `rightmostFastest=true` and
`isGlobalToLDS=true` (the MATRIX_A path):

```
Current graph (relevant edges):

  Workitem ──Tile──→ {nThrX, nThrY}     // nThrX=row (slow), nThrY=col (fast)
  nThrX, iThrX ──Flatten──→ iMacX       // M-row within tile
  nThrY, iThrY ──Flatten──→ iMacY       // K-column within tile
  iMacX, iMacY ──Flatten──→ ldsTag      // linear LDS offset (in StoreLDSTile visitor)
```

Key dimensions:
- `nThrY` (ThreadTileNumber(1)) = `workitem % kChunksPerRow` = sequential column (base_col)
- `nThrX` (ThreadTileNumber(0)) = `workitem / kChunksPerRow` = row within tile

With swizzle, insert an `ExpressionTransform` that remaps `nThrY` based on
`nThrX`:

```
With swizzle:

  Workitem ──Tile──→ {nThrX, nThrY}
  {nThrY, nThrX} ──ExpressionTransform──→ swizzled_nThrY   // NEW
  nThrX, iThrX ──Flatten──→ iMacX                          // unchanged
  swizzled_nThrY, iThrY ──Flatten──→ iMacY                 // was nThrY
  iMacX, iMacY ──Flatten──→ ldsTag
```

`nThrX` is input to both the Flatten for `iMacX` and the ExpressionTransform
(a dimension can be input to multiple edges).

The ExpressionTransform forward expression:

```
$0 = nThrY (base_col, = workitem % kChunksPerRow)
$1 = nThrX (row, = workitem / kChunksPerRow)
K  = kChunksPerRow, R = numRowsPerBankRow

ldsRowId     = $1 / R
swap_col     = (ldsRowId % 2 == 0) ? ($0 ^ 1) : $0
gr_rotation  = K - (ldsRowId / 2) * 2
swizzled_col = (swap_col + gr_rotation) & (K - 1)
```

The modular arithmetic makes this periodic across waves (rotation wraps every
`kChunksPerRow` rows), so the same expression handles multi-wave workgroups.

### Coordinate graph integration: LR side (LoadLDSTile)

`addLoadWaveTileCT` in `LowerTile.cpp:529` decomposes the Lane into MFMA
sub-dimensions. For `MATRIX_A` with standard (non-transposed, non-sub-dword-
non-contiguous) layout:

```
Current graph (relevant edges):

  iWaveY ──Tile──→ {blockNumber, blockIndex}   // K-column decomposition
  blockNumber, iWaveX ──Flatten──→ lane        // lane = blockNumber*16 + iWaveX
  blockIndex ──PassThrough──→ vgpr
```

Key dimensions:
- `blockNumber` (VGPRBlockNumber) = `lane / 16` = SIMDIndex, size = `mi.k / K_L`
- `iWaveX` (WaveTileIndex(0)) = `lane % 16` = laneInSIMD
- `blockIndex` (VGPRBlockIndex) = VGPR register index, size = `K_L`

With swizzle, insert an `ExpressionTransform` that remaps `blockNumber` to
produce the swizzled K-column group for LDS addressing:

```
With swizzle:

  {blockNumber, iWaveX} ──ExpressionTransform──→ swizzled_blockNumber  // NEW
  iWaveY ──Tile──→ {swizzled_blockNumber, blockIndex}                  // was blockNumber
  blockNumber, iWaveX ──Flatten──→ lane                                // unchanged
  blockIndex ──PassThrough──→ vgpr
```

`blockNumber` remains the input to the lane Flatten (physical lane assignment
stays the same). The swizzled version only affects the K-column group used for
LDS addressing via `iWaveY`.

The ExpressionTransform forward expression:

```
$0 = blockNumber (SIMDIndex, = lane / 16)
$1 = iWaveX     (laneInSIMD, = lane % 16)
K  = kChunksPerRow, R = numRowsPerBankRow

ldsRowId     = $1 / R
rotation     = (ldsRowId / 2) * 2
base_col     = ($0 + rotation) & (K - 1)
swizzled_col = (ldsRowId % 2 == 0) ? (base_col ^ 1) : base_col
```

Note: the MATRIX_B case follows the same pattern but with X/Y swapped
(iWaveX ↔ iWaveY, waveX ↔ waveY). The swizzle expressions are identical.

### TODO

1. Add `ExpressionTransform` edge type
   - Add struct to `CoordinateEdge.hpp` with `forward` and `reverse` expression vectors
   - Add to `CoordinateTransformEdge` variant in `CoordinateEdge_fwd.hpp`
   - Add `ForwardEdgeVisitor` / `ReverseEdgeVisitor` operators using `positionalArgumentPropagation`
   - Diff visitors: throw (not needed)
   - Add unit tests for the edge: verify forward/reverse propagation with known
     expression trees (e.g., identity, XOR, shift patterns)

2. End-to-end smoke test with trivial expression
   - Wire an `ExpressionTransform` with `f(x) = x + 1 - 1` into an existing
     tile path (e.g., the LDS column index for a small GEMM)
   - Run the kernel and verify correct results. The `+1 -1` may be simplified
     away by constant expression folding, but should still appear in assembly
     comments confirming the expression propagated through codegen
   - This validates the full pipeline (coordinate graph -> AssignIndexExpressions
     -> code generation) before adding any swizzle complexity

3. Add kernel option
   - Add `LDSBankSwizzleMode` kernel option (None, Swizzle) -- model on `ScaleSkipPermlaneMode.hpp`

4. Add unit GEMM test for smaller tile (128x128x256, no Direct2LDS)
   - Model on `GPU_GEMM_LoadPath_Direct2LDS_FP4_MT256x256x128_TN` in `GEMMTest.cpp:468`
     which uses `GEMMProblemF8F6F4{32, 32, 64}` (from `GEMMF8F6F4.hpp`)
   - Adjust: `macM=128, macN=128, macK=256`, use `BufferToLDSViaVGPR` (not Direct2LDS),
     enable `LDSBankSwizzleMode::Swizzle`
   - This serves as the correctness baseline throughout development

5. GR column rotation
   - In `addStoreThreadTileCT` (`LowerTile.cpp:1258`), when swizzle enabled:
     add `ExpressionTransform({nThrY, nThrX} → swizzled_nThrY)` and use
     `swizzled_nThrY` in the Flatten to `iMacY` (replacing `nThrY`)
   - Expression inputs: `$0=nThrY` (base_col), `$1=nThrX` (row within tile)
   - Derive `ldsRowId = $1 / numRowsPerBankRow` inside the expression
   - Verify: check assembly for XOR + rotation arithmetic, trace GR pattern

6. LR column rotation
   - In `addLoadWaveTileCT` (`LowerTile.cpp:529`), MATRIX_A case, when swizzle
     enabled: add `ExpressionTransform({blockNumber, iWaveX} → swizzled_blockNumber)`
     and use `swizzled_blockNumber` in the Tile from `iWaveY` (replacing `blockNumber`)
   - Expression inputs: `$0=blockNumber` (SIMDIndex), `$1=iWaveX` (laneInSIMD)
   - Derive `ldsRowId = $1 / numRowsPerBankRow` inside the expression
   - Handle MATRIX_B symmetrically (X/Y swapped)
   - Verify: check assembly, traces, norm passes

7. Benchmark (128x128x256)
   - rocgdb tracing to confirm rotation pattern eliminates bank conflicts
   - Benchmark LDS read cycles (expect 4x improvement)

8. Scale up to 256x256x256 with Direct2LDS
   - Enable Direct2LDS path with `LDSBankSwizzleMode::Swizzle`
   - Verify the same rotation logic applies correctly at the larger tile size
   - Benchmark to confirm bank conflict elimination

### Follow-up: Inter-wave rotation

For certain wave group configurations (e.g., 2x2 `MIWaveGroup` where multiple waves
contribute to the same LDS region), an additional per-wave column rotation may be
needed on the GR side:

```
wave_rotation = (waveId & 1) * (2 * numRowsPerBankRow)
col = (col + intra_rotation - wave_rotation) % kChunksPerRow
```

This applies when `loadRatioGR != 0.5` in Tensile's terminology. For the primary
target (256x256x256 with 4x1 wave group, `loadRatioGR = 0.5`), intra-wave
rotation alone is sufficient.

Reference: `_grSwizzleColIds` in `SubtileBasedKernel.py` lines 778-812.
