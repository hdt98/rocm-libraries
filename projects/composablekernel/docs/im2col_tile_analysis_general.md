# Im2col Tile Awareness Analysis — Generalized

## Notation

### Convolution Parameters (2D)

| Symbol | Meaning |
|--------|---------|
| G | Number of groups |
| N | Batch size |
| Hi, Wi | Input spatial dimensions |
| Ho, Wo | Output spatial dimensions |
| Y, X | Filter spatial dimensions |
| C | Input channels per group |
| K | Output channels per group |
| SH, SW | ConvStrideH, ConvStrideW |
| DH, DW | ConvDilationH, ConvDilationW |
| PH, PW | Input left padding (H, W) |

Output size: `Ho = (Hi + 2·PH − DH·(Y−1) − 1) / SH + 1`, similarly for Wo.

### Tensor Strides (NHWGC input layout)

```
NStride  = Hi × Wi × G × C
HiStride = Wi × G × C
WiStride = G × C
GStride  = C
CStride  = 1
```

### GEMM Dimensions and Tile Sizes

| Matrix | Size | Tile |
|--------|------|------|
| A (input, im2col) | M_gemm × K_gemm | M_tile × K_tile |
| B (weight)        | N_gemm × K_gemm | N_tile × K_tile |
| C (output)        | M_gemm × N_gemm | M_tile × N_tile |

```
M_gemm = N × Ho × Wo
K_gemm = Y × X × C
N_gemm = K
```

---

## Index Mappings

### m_gemm → conv input spatial indices

```
n_conv =  m_gemm / (Ho × Wo)
ho     = (m_gemm % (Ho × Wo)) / Wo
wo     =  m_gemm % Wo
```

Range: `m_gemm ∈ [0, M_gemm)`, `n_conv ∈ [0,N)`, `ho ∈ [0,Ho)`, `wo ∈ [0,Wo)`.

> Tests: [MappingBounds](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L251) · [MappingBijection](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L269) · [MDecodeSpecificValues](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L292)

### k_gemm → conv filter indices

```
y      =  k_gemm / (X × C)
x      = (k_gemm % (X × C)) / C
c_conv =  k_gemm % C
```

Range: `k_gemm ∈ [0, K_gemm)`, `y ∈ [0,Y)`, `x ∈ [0,X)`, `c_conv ∈ [0,C)`.

> Tests: [MappingBounds](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L251) · [MappingBijection](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L269) · [KDecodeSpecificValues](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L307)

### n_gemm → output filter index

```
k_conv = n_gemm
```

---

## A Matrix Access: Address Decomposition

`A[m_gemm, k_gemm]` reads the input tensor at physical location:

```
input[n_conv, ho, wo, g, c_conv]  with filter offset (y, x)
```

The actual input coordinates with stride and dilation are:

```
input_h = ho × SH + y × DH − PH      (must satisfy 0 ≤ input_h < Hi)
input_w = wo × SW + x × DW − PW      (must satisfy 0 ≤ input_w < Wi)
```

The linear byte offset into the input tensor is:

```
offset(m_gemm, k_gemm) =
      n_conv × NStride
    + (ho × SH + y × DH − PH) × HiStride
    + (wo × SW + x × DW − PW) × WiStride
    + g × GStride
    + c_conv
```

Expanding and separating M and K contributions:

```
offset(m_gemm, k_gemm) = M_base(m_gemm)  +  K_offset(k_gemm)  +  const_g
```

where:

```
M_base(m_gemm)   = n_conv × NStride
                 + ho × SH × HiStride
                 + wo × SW × WiStride
                 − PH × HiStride
                 − PW × WiStride           ← absorb padding into M_base

K_offset(k_gemm) = y × DH × HiStride
                 + x × DW × WiStride
                 + c_conv

const_g          = g × GStride             ← constant per kernel launch
```

**The M and K address contributions are independent and additive.**
This holds for any stride, dilation, or padding. Padding affects only the
validity check (see below), not the address structure.

> Tests (exhaustive decomposition vs. `TransformConvFwdToGemm` reference):
> [UnitStrideZeroPad](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L331) · [Stride2ZeroPad](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L336) · [Dilation2ZeroPad](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L341) · [WithPadding](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L346)
>
> Tests (independence of M and K contributions):
> [MBaseIndependentOfK](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L376) · [KOffsetIndependentOfM](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L388)
>
> Tests (individual contributions vs. reference):
> [MBaseMatchesDescriptorForK0](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L351) · [KOffsetMatchesDescriptorForM0](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L363)

---

## Validity Check (Padding)

When padding is non-zero, an element may map to a padded (invalid) input
location. The validity check also decomposes:

```
valid(m_gemm, k_gemm) = valid_h(ho, y)  AND  valid_w(wo, x)

valid_h(ho, y) = (0 ≤ ho × SH + y × DH − PH < Hi)
valid_w(wo, x) = (0 ≤ wo × SW + x × DW − PW < Wi)
```

- `valid_h` depends on (ho, y): ho from m_gemm, y from k_gemm.
- `valid_w` depends on (wo, x): wo from m_gemm, x from k_gemm.

With non-zero padding these cannot be fully separated, but:
- The `y` and `x` ranges that are invalid for a given (ho, wo) tile can be
  **precomputed once per M-tile** and stored as a small validity mask over k_gemm.
- For zero padding, the check is trivially true and can be eliminated entirely.

> Tests: [ValidityMatchesReference](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L722) · [CornerElementsInvalid](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L730) · [CenterFilterValid](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L738) · [ValidElementsDecomposeCorrectly](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L746)

---

## M-tile Pattern Analysis

A block covers rows `m_gemm ∈ [m_start, m_start + M_tile − 1]` where:
```
m_start = block_M_idx × M_tile
```

The tile origin (i=0):
```
n_conv(i=0) = m_start / (Ho × Wo)
ho(i=0)     = (m_start % (Ho × Wo)) / Wo
wo(i=0)     = m_start % Wo
```

```
m_gemm(i) = m_start + i, where i ∈ [0, M_tile)
```

### n_conv across the tile

`n_conv` changes only when `m_gemm` crosses a multiple of `Ho × Wo`.

```
P(n_conv changes within tile) ≈ M_tile / (Ho × Wo)
```

For typical values this is negligible (e.g., 16 / (100 x 100) = 0.16%). In practice,
`n_conv` is constant within every tile.

> Test: [NConvConstantWithinTile](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L526)

However, it remains to be seen if this is a good approximation. 

### ho across the tile — general formula

For row `i ∈ [0, M_tile)`, the number of ho-boundaries (wo-row wraps) crossed is:

```
B(i) = ⌊(wo_0 + i) / Wo⌋        (number of complete wo-rows since tile start)
```

The ho index and wo index for row i are:
```
ho_i = ho_0 + B(i)
wo_i = (wo_0 + i) % Wo
```

**Number of ho-boundaries crossed across the whole tile:**
```
B_total = ⌊(wo_0 + M_tile − 1) / Wo⌋
```

| Condition | B_total | Probability (uniform wo_0) |
|-----------|---------|---------------------------|
| M_tile ≤ Wo, wo_0 ≤ Wo − M_tile | 0 | (Wo − M_tile + 1) / Wo |
| M_tile ≤ Wo, wo_0 > Wo − M_tile | 1 | (M_tile − 1) / Wo |
| M_tile > Wo | ⌊M_tile/Wo⌋ or ⌊M_tile/Wo⌋+1 | — |

> Tests: [BoundaryCountFormula](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L478) · [BoundaryProbability](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L549)

### M_base across the tile

Define the per-step and per-wrap increments:
```
step_w        = SW × WiStride         (M_base change per +1 in wo)
wrap_delta    = SH × HiStride − Wo × SW × WiStride
              = SH × HiStride − Wo × step_w   (M_base change at each ho-boundary)
```

For unit stride (SH=SW=1) which is comon for the RetinaNet shapes we have: 
`step_w = WiStride`, `wrap_delta = HiStride − Wo × WiStride`.

Then for row i within the tile:
```
M_base[i] = M_base[0]  +  i × step_w  +  B(i) × wrap_delta
```

where `B(i) = ⌊(wo_0 + i) / Wo⌋`.

- **When B_total=0**: `B(i) = 0` for all i → `M_base[i] = M_base[0] + i × step_w`.
  Pure linear: no division required after tile setup.

- **When B_total=1**: `B(i) = 0` for `i < Wo − wo_0`, then 1.
  One conditional add of `wrap_delta`.

- **When B_total≥2** (M_tile > Wo only): general formula applies, computed incrementally.

> Tests: [StepWAndWrapDeltaValues](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L415) · [SingleHoTileIsLinear](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L427) · [DualHoTileHasWrapDelta](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L450) · [GeneralFormulaWithNWraps](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L499)

---

## K-tile Pattern Analysis

A K-tile covers `k_gemm ∈ [k_start, k_start + K_tile − 1]` where:
```
k_start = k_tile_idx × K_tile
```

Decoding: `y_0 = k_start / (X×C)`, `x_0 = (k_start % (X×C)) / C`, `c_0 = k_start % C`.

### y across the K-tile

`y` changes when `k_gemm` crosses a multiple of `X × C`.

```
P(y constant within K-tile) = 1   when K_tile ≤ X × C
                                   AND k_start is aligned to X×C boundary
```

For general K_tile and alignment:
```
y-boundaries within K-tile = ⌊(k_start/C + K_tile/C − 1) / X⌋ − ⌊k_start/C / X⌋
```

When `K_tile = X × C` (one full filter row per K-tile): exactly one `y` value, all X `x`-values.
When `K_tile = C` (one filter column per K-tile): exactly one `y`, one `x` value. Simplest case.
When `K_tile = 2C`: one `y`, two `x` values.

> Tests: [YConstantForKTileEqualC](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L577) · [YConstantForKTileEqual2C](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L593)

### x across the K-tile

Number of distinct x values within a K-tile: `K_tile / C` (when K_tile is a multiple of C).

The general count of x-values:
```
x_count = ⌈K_tile / C⌉    (may be non-integer if K_tile not multiple of C)
```

### c_conv across the K-tile

`c_conv = k_gemm % C` cycles through `[0, C)` repeatedly.
Full cycles per K-tile: `K_tile / C` (when K_tile is a multiple of C).

### K_offset incremental steps

```
Δ(c_conv +=1, x unchanged):  ΔK_offset = DW × WiStride · 0  + 1 = 1
Δ(c_conv wraps, x +=1):       ΔK_offset = DW × WiStride − (C − 1)
Δ(x wraps, y +=1):             ΔK_offset = DH × HiStride − X × DW × WiStride − (C − 1)
```

For unit dilation (DH=DW=1): `ΔK_x = WiStride − C`, `ΔK_y = HiStride − X × WiStride − C`.

The K_offset can be computed incrementally by tracking only these deltas — no
division or modulo required in the inner loop.

> Tests: [KOffsetIncrementsByOnePerC](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L622) · [KOffsetXTransitionStep](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L635) · [KOffsetYTransitionStep](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L650) · [Dilation2KOffsetYStep](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L701)

---

## Precomputation Strategy (General)

### At block start — once per block, amortized over K_tiles_per_block K-tiles

```
Step 1.  n_conv_0 = m_start / (Ho × Wo)         \
Step 2.  ho_0     = (m_start % (Ho × Wo)) / Wo   |  3 integer divisions
Step 3.  wo_0     = m_start % Wo                 /

Step 4.  M_base[0] = n_conv_0 × NStride
                   + ho_0 × SH × HiStride
                   + wo_0 × SW × WiStride
                   + g × GStride
                   − PH × HiStride − PW × WiStride

Step 5.  M_base[i] for i = 1..M_tile-1:
             accumulate M_base[i] = M_base[i-1] + step_w + (new_ho_row ? wrap_delta : 0)
             → M_tile additions + at most B_total comparisons
             → store in register array of M_tile integers
```

Amortization cost: 3 divisions amortized over `M_tile × K_tiles_per_block` element loads.

> Test: [PrecomputedArrayMatchesMBase](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L782)

### At each K-tile start — once per K-tile

```
Step 6.  y_0     = k_start / (X × C)             \  2 integer divisions
Step 7.  x_0     = (k_start % (X × C)) / C        /  (or derived from k_tile_idx cheaply)

Step 8.  K_base  = y_0 × DH × HiStride
                 + x_0 × DW × WiStride

         (c_conv starts at 0 if K_tile is aligned to C boundaries)
```

If K_tile is a multiple of C and k_start is a multiple of C (common):
Steps 6–7 simplify to integer shifts/masks if C and X×C are powers of 2.

### Per element — hot path

```
offset = M_base[local_m] + K_base + local_x × DW × WiStride + local_c
```

where `local_x = local_k / C` (0 or 1 for K_tile = 2C) and `local_c = local_k % C`
advance by 1 each step with periodic x-increment.

**Hot path cost: 2 register loads + 2 additions.** No division or modulo.

> Test (full round-trip precomputed M_base + K_offset vs. reference): [PrecomputedPlusKOffsetMatchesReference](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L794)

---

## B Matrix Access (Weight — No Im2col)

`B[n_gemm, k_gemm]` reads weight tensor at `weight[g, k_conv, y, x, c_conv]`.

For GKYXC layout with strides `KStride = Y×X×C`, `ZYXStride = X×C`, etc.:

```
offset(n_gemm, k_gemm) = n_gemm × KStride + k_gemm
```

The B matrix is **purely linear** — no im2col transform, no precomputation needed.
The weight tiles load with trivial strided access.

---

## General Formulae Summary

### Boundary crossing probability

For a tile of size M_tile along m_gemm with output width Wo:

| Tile size vs Wo | Expected ho-boundaries per tile |
|----------------|--------------------------------|
| M_tile ≤ Wo | (M_tile − 1) / Wo |
| M_tile > Wo | (M_tile − 1) / Wo (same formula, may be > 1) |

For a K-tile of size K_tile along k_gemm with filter X×C:

| K_tile vs X×C | y constant? | x-values per K-tile |
|---------------|------------|---------------------|
| K_tile ≤ X×C | Yes (when aligned) | K_tile / C |
| K_tile = X×C | Yes (always) | X |
| K_tile > X×C | No | > X |

### Precomputation cost summary

| Computation | Cost | Frequency |
|-------------|------|-----------|
| M_base[0..M_tile-1] | 3 divmod + M_tile adds | Once per block |
| K_base (per K-tile) | 2 divmod (or cheaper) | Once per K-tile |
| Per-element offset | 2 additions + 2 register reads | Every load |

### Amortized division cost per element load

```
cost_per_load = (3 + K_tiles × 2) / (M_tile × K_tile)
              = 3 / (M_tile × K_tile) + 2 / M_tile
```

For analyzed case (M_tile=16, K_tile=64, K_tiles=8):
```
= 3 / (16 × 512) + 2 / 16 = 0.000366 + 0.125 ≈ 0.125 divisions/load
```

Compare with current (full unmerge on every move_tensor_coordinate call):
```
≈ 4 divisions per load (2 for M, 2 for K)
```

**Reduction factor: ~32× fewer integer divisions in the hot path.**

---

## Practical Constraints and Edge Cases

### When n_conv spans the tile

Probability ≈ M_tile / (Ho × Wo). When it occurs (e.g., last few M-tiles):
- n_conv changes once at the boundary
- Add `NStride` to M_base for subsequent rows, similar to `wrap_delta`
- Negligible frequency; can be handled with a branch

> Test: [NConvConstantWithinTile](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L526)

### When K_tile > X×C (y changes within K-tile)

y-boundaries within K-tile: `⌊(x_0 + K_tile/C − 1) / X⌋`

When y changes: add `DH × HiStride − X × DW × WiStride` to K_base at the boundary.
This is the K-dimension equivalent of `wrap_delta`.

### Non-unit stride with large strides

Large SH or SW reduce Ho/Wo, which increases P(n_conv spans tile) and
P(ho is constant per tile). Large strides favour the precomputed approach
since M_base changes more slowly per output element.

> Tests: [Stride2DecompositionVsReference](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L670) · [Stride2StepW](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp#L680)

### Alignment of K_tile to C

If K_tile is not a multiple of C, the K-tile may start mid-cycle in c_conv.
Account for by initialising `c_start = k_start % C` and adjusting K_base
and the x-transition point accordingly.


## Testing

Unit tests for the im2col index calculations:
[test_im2col_index_mapping.cpp](../test/ck_tile/image_to_column/test_im2col_index_mapping.cpp)
