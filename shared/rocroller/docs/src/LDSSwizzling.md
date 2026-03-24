# LDS Swizzle for Bank Conflict Elimination

mxfp4, 256x256x256 tile, `v_mfma_scale_f32_16x16x128_f8f6f4`, direct to lds.

This document describes how intra-wave column rotation + pair-swap eliminates 4x
LDS bank conflict serialization during Local Read (`ds_read_b128`). The approach
follows Tensile's `SubtileBasedKernel.py`: the GR row assignment stays the same
(8 contiguous rows per wave, no half-wave split), and only the column indices are
permuted.

Reference: `SubtileBasedKernel.py` functions `_grSwizzleColIds` (GR side) and
`lraTileAssignment` (LR side).

## Background

LDS has 64 banks, each 1 dword wide. A bank conflict occurs when multiple threads in the
same execution phase read from the same bank; the accesses are serialized, stalling the
wavefront. For `ds_read_b128`, the effective granularity is 16 dwordx4 bank groups.

`ds_read_b128` occurs in 4 phases, with 16 threads each:

```
Phase 0:  T0-T3,   T12-T15,  T20-T23,  T24-T27
Phase 1:  T32-T35, T44-T47,  T52-T55,  T56-T59
Phase 2:  T4-T7,   T8-T11,   T16-T19,  T28-T31
Phase 3:  T36-T39, T40-T43,  T48-T51,  T60-T63
```

Only banks accessed by threads in the same phase can cause conflicts. For zero conflicts,
each phase's 16 threads must hit 16 different bank groups.

Per-lane terms used throughout:

```
lane16      = lane % 16    (position within the 16-lane SIMD, a.k.a. laneInSIMD)
lane16Group = lane // 16   (which SIMD, 0-3, a.k.a. SIMDIndex)
```

## Parameters

For the 256x256x256 fp4 tile with `v_mfma_scale_f32_16x16x128_f8f6f4`:

```
depthU           = 256
bpe              = 0.5 B  (fp4)
depthUBytes      = depthU * bpe = 128 B
loadWidth        = 16 B   (dwordx4)
blockSize        = depthUBytes / loadWidth = 8    (K-columns per row)
ldsRowBankSize   = 64 banks * 4 B = 256 B         (one full LDS bank row)
numRowsPerBankRow= ldsRowBankSize / depthUBytes = 2 (matrix rows per LDS bank row)
rowsPerWave      = 64 / blockSize = 8
```

Each wave's 64 lanes map to an 8-row x 8-column grid. Each column is one 16B
K-chunk (32 fp4 elements).

## Current Problem (no rotation)

Without rotation, every wave uses `col = lane % 8` (sequential columns).
The LR bank group for lane `l` reading column `col` at row `lane16` is:

```
bank_group = (col + lane16 * 8) % 16
```

Since `col = lane16Group` (0-3) and `lane16 * 8` contributes only 0 or 8 (mod 16),
each phase's 16 threads hit only 4 of 16 bank groups, with 4 threads per group:

```
  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
---------------------------------------------------------------
 T0 T20                          T1 T21
 T2 T22                          T3 T23
T12 T24                         T13 T25
T14 T26                         T15 T27
---------------------------------------------------------------
...
```

Result: **4x serialization**.

## Solution: Intra-Wave Rotation + Pair-Swap

The key idea: rotate the column assignment based on the LDS row that each lane's
data falls into. Different LDS rows get different rotations, breaking the bank
conflict symmetry. A conditional pair-swap provides additional diversity to cover
all 16 bank groups.

No half-wave split is needed. The GR keeps 8 contiguous rows per wave.

### Derived quantities

```
ldsRowId  = lane16 / numRowsPerBankRow           (for LR: lane16 = lane % 16)
          = lane / (blockSize * numRowsPerBankRow)  (for GR: across all 64 lanes)
```

For our parameters (`numRowsPerBankRow = 2`):

| lane16 range | ldsRowId | rotation = `(ldsRowId // 2) * 2` |
|---|---|---|
| 0-1 | 0 | 0 |
| 2-3 | 1 | 0 |
| 4-5 | 2 | 2 |
| 6-7 | 3 | 2 |
| 8-9 | 4 | 4 |
| 10-11 | 5 | 4 |
| 12-13 | 6 | 6 |
| 14-15 | 7 | 6 |

Four distinct rotations (0, 0, 2, 2, 4, 4, 6, 6) across 16 rows. Combined with
4 `lane16Group` values and the pair-swap, this produces 16 unique bank groups per phase.

### GR Column Assignment

Each GR lane fetches a K-column from global memory and writes it to a fixed LDS
position (`wave_base + lane * 16`). The rotation permutes which K-column goes to
which lane.

```
base_col = lane % blockSize

# Conditional pair-swap: applied to even ldsRowId groups only
if ldsRowId % 2 == 0:
    col = quad_perm(base_col, [1,0,3,2])   # v_mov_b32 dpp:quad_perm:[1,0,3,2]
else:
    col = base_col

# Intra-wave rotation (additive inverse of LR rotation)
gr_rotation = blockSize - (ldsRowId // 2) * 2
col = (col + gr_rotation) % blockSize
```

Instruction sequence (from `_grSwizzleColIds`):
```
v_lshrrev_b32  ldsRowId, log2(blockSize), laneId         ; row within wave
v_lshrrev_b32  ldsRowId, log2(numRowsPerBankRow), ldsRowId ; LDS row id
v_and_b32      tmp, ldsRowId, 1
v_cmpx_eq_u32  vcc, 0, tmp                               ; even ldsRowId?
v_mov_b32      colId, colId dpp:quad_perm:[1,0,3,2]      ; pair-swap
s_mov_b64      exec, -1
; rotation
v_lshrrev_b32  tmp, 1, ldsRowId                           ; ldsRowId // 2
v_lshlrev_b32  tmp, 1, tmp                                ; * 2
v_sub_u32      tmp, blockSize, tmp                         ; blockSize - (ldsRowId//2)*2
v_add_u32      colId, colId, tmp
v_and_b32      colId, colId, blockSize-1                  ; mod blockSize
```

### GR layout -- which lane loads each (M-row, K-chunk)

All waves have the same column pattern (no inter-wave rotation). Wave changes
every 8 rows.

| M-row | K0 | K1 | K2 | K3 | K4 | K5 | K6 | K7 | rotation | swap |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | L1 | L0 | L3 | L2 | L5 | L4 | L7 | L6 | 0 | yes |
| 1 | L9 | L8 | L11 | L10 | L13 | L12 | L15 | L14 | 0 | yes |
| 2 | L16 | L17 | L18 | L19 | L20 | L21 | L22 | L23 | 0 | no |
| 3 | L24 | L25 | L26 | L27 | L28 | L29 | L30 | L31 | 0 | no |
| 4 | L35 | L34 | L37 | L36 | L39 | L38 | L33 | L32 | 6 | yes |
| 5 | L43 | L42 | L45 | L44 | L47 | L46 | L41 | L40 | 6 | yes |
| 6 | L50 | L51 | L52 | L53 | L54 | L55 | L48 | L49 | 6 | no |
| 7 | L58 | L59 | L60 | L61 | L62 | L63 | L56 | L57 | 6 | no |

Four distinct column permutations across 8 rows:

| Row group | Permutation (K-col at LDS positions 0-7) |
|---|---|
| 0-1 (swap, rotation 0) | 1 0 3 2 5 4 7 6 |
| 2-3 (no swap, rotation 0) | 0 1 2 3 4 5 6 7 |
| 4-5 (swap, rotation 6) | 7 6 1 0 3 2 5 4 |
| 6-7 (no swap, rotation 6) | 6 7 0 1 2 3 4 5 |

### LR Column Assignment

The LR rotation is the additive complement of the GR rotation: where GR uses
`blockSize - (ldsRowId // 2) * 2`, LR uses `(ldsRowId // 2) * 2`. This ensures
the LR reads the correct K-column data from the rotated LDS layout.

```
rotation = (ldsRowId // 2) * 2          # ldsRowId = lane16 / numRowsPerBankRow
col = (lane16Group + rotation) % blockSize

# Pair-swap: exchange col with lane +/-16 partner for lanes where lane%4 < 2
v_permlane16_swap_b32 with exec mask 0x33333333
```

Instruction sequence (from `lraTileAssignment`):
```
v_and_b32      lane16Group, Serial, wavefrontSize-1
v_lshrrev_b32  lane16Group, log2(MFMA_M), lane16Group    ; lane // 16
v_and_b32      lane16, Serial, MFMA_M-1                  ; lane % 16
; rotation
v_lshrrev_b32  rotation, log2(numRowsPerBankRow), lane16  ; ldsRowId
v_lshrrev_b32  rotation, 1, rotation                      ; ldsRowId // 2
v_lshlrev_b32  rotation, 1, rotation                      ; * 2
v_add_u32      colOffset, rotation, lane16Group
v_and_b32      colOffset, colOffset, blockSize-1          ; mod blockSize
; pair-swap
s_mov_b32      exec_lo, 0x33333333
s_mov_b32      exec_hi, 0x33333333
v_permlane16_swap_b32 colOffset, colOffset
s_mov_b64      exec, -1
```

The `v_permlane16_swap_b32` with mask `0x33333333` swaps `colOffset` between
lane `i` and lane `i+16` (i.e., between `lane16Group` 0 and 1, or 2 and 3), but
only for lanes where `lane % 4 < 2`. Lanes with `lane % 4 >= 2` keep their
original value.

### LR column values per lane (after rotation + swap)

| lane16 | G=0 | G=1 | G=2 | G=3 |
|---|---|---|---|---|
| 0 | **1** | **0** | **3** | **2** |
| 1 | **1** | **0** | **3** | **2** |
| 2 | 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 | 3 |
| 4 | **3** | **2** | **5** | **4** |
| 5 | **3** | **2** | **5** | **4** |
| 6 | 2 | 3 | 4 | 5 |
| 7 | 2 | 3 | 4 | 5 |
| 8 | **5** | **4** | **7** | **6** |
| 9 | **5** | **4** | **7** | **6** |
| 10 | 4 | 5 | 6 | 7 |
| 11 | 4 | 5 | 6 | 7 |
| 12 | **7** | **6** | **1** | **0** |
| 13 | **7** | **6** | **1** | **0** |
| 14 | 6 | 7 | 0 | 1 |
| 15 | 6 | 7 | 0 | 1 |

Bold = swapped by `v_permlane16_swap_b32`. The swap exchanges values between
G=0/G=1 and G=2/G=3 for lanes where `lane % 4 < 2`.

### Correctness check (GR-LR consistency)

For the LR to read the correct data, `LR_col` must point to the LDS position
containing the K-column that each lane needs.

GR lane `l` writes K-column `GR_col[l]` to LDS position `l * 16` (within the wave).
LR lane `l` needs K-group `lane16Group` and reads from LDS column `LR_col`.

Spot checks for wave 0 (first MFMA tile, K-columns 0-3):

| LR lane | lane16 | G | LR col | LDS addr | GR lane at that addr | GR K-col | match? |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 1 | 16 | 1 | 0 | K0 = G0 |
| 16 | 0 | 1 | 0 | 0 | 0 | 1 | K1 = G1 |
| 32 | 0 | 2 | 3 | 48 | 3 | 2 | K2 = G2 |
| 48 | 0 | 3 | 2 | 32 | 2 | 3 | K3 = G3 |
| 4 | 4 | 0 | 3 | 560 | 35 | 0 | K0 = G0 |
| 36 | 4 | 2 | 5 | 592 | 37 | 2 | K2 = G2 |

All correct. The second MFMA tile (K-columns 4-7) uses `col = (LR_col + 4) % 8`,
which shifts to the corresponding K-columns and remains consistent.

## Bank Conflict Analysis

`bank_group = (LR_col + lane16 * 8) % 16`

### Bank groups per lane

| lane16 | G=0 | G=1 | G=2 | G=3 |
|---|---|---|---|---|
| 0 | 1 | 0 | 3 | 2 |
| 1 | 9 | 8 | 11 | 10 |
| 2 | 0 | 1 | 2 | 3 |
| 3 | 8 | 9 | 10 | 11 |
| 4 | 3 | 2 | 5 | 4 |
| 5 | 11 | 10 | 13 | 12 |
| 6 | 2 | 3 | 4 | 5 |
| 7 | 10 | 11 | 12 | 13 |
| 8 | 5 | 4 | 7 | 6 |
| 9 | 13 | 12 | 15 | 14 |
| 10 | 4 | 5 | 6 | 7 |
| 11 | 12 | 13 | 14 | 15 |
| 12 | 7 | 6 | 1 | 0 |
| 13 | 15 | 14 | 9 | 8 |
| 14 | 6 | 7 | 0 | 1 |
| 15 | 14 | 15 | 8 | 9 |

### Bank groups per phase

```
  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
---------------------------------------------------------------
Phase 0: T2  T0 T20 T22 T24 T26 T14 T12  T3  T1 T21 T23 T25 T27 T15 T13
---------------------------------------------------------------
Phase 1: T46 T44 T34 T32 T52 T54 T56 T58 T47 T45 T35 T33 T53 T55 T57 T59
---------------------------------------------------------------
Phase 2: T16 T18  T6  T4 T10  T8 T28 T30 T17 T19  T7  T5 T11  T9 T29 T31
---------------------------------------------------------------
Phase 3: T60 T62 T48 T50 T38 T36 T42 T40 T61 T63 T49 T51 T39 T37 T43 T41
```

Every phase: 16 threads across 16 unique bank groups. **Zero bank conflicts.**

<details>
<summary>first_bank per lane</summary>

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 9 | 0 | 8 | 3 | 11 | 2 | 10 | 5 | 13 | 4 | 12 | 7 | 15 | 6 | 14 | 0 | 8 | 1 | 9 | 2 | 10 | 3 | 11 | 4 | 12 | 5 | 13 | 6 | 14 | 7 | 15 | 3 | 11 | 2 | 10 | 5 | 13 | 4 | 12 | 7 | 15 | 6 | 14 | 1 | 9 | 0 | 8 | 2 | 10 | 3 | 11 | 4 | 12 | 5 | 13 | 6 | 14 | 7 | 15 | 0 | 8 | 1 | 9 |

</details>

## Why it works

The rotation creates 4 distinct column offsets (0, 2, 4, 6) across 16 rows, changing
every `2 * numRowsPerBankRow = 4` rows. The pair-swap further differentiates
adjacent row pairs within each rotation group. Together:

- 4 rotation values x 2 swap states = 8 distinct column permutations across 16 rows
- Combined with the `lane16 * 8` term in the bank group formula (which alternates
  between +0 and +8 mod 16 for even/odd rows), this produces 16 unique bank groups
  per `ds_read_b128` phase

The critical property: GR and LR rotations are additive complements modulo
`blockSize`. GR uses `blockSize - (ldsRowId // 2) * 2`, LR uses
`(ldsRowId // 2) * 2`. They sum to `blockSize` (= 0 mod blockSize), ensuring the
LR reads the correct K-column data from the rotated LDS positions.

## Implementing in rocRoller

Gated by a new compile-time kernel option (`LDSBankSwizzleMode`). Two sides need to change:
GR voffset and LR address. Both can be expressed as pure per-lane arithmetic in the
coordinate graph -- no new instruction emission needed. All parameters are derived from
the tile config.

**Note:** The 256x256x256 tile requires Direct2LDS; without it the kernel runs out of VGPRs
and must use a smaller tile. However, the implementation should work for any tile size that
exhibits this bank conflict pattern. Development strategy: implement and verify on a smaller
tile first (without Direct2LDS), then confirm it scales up to 256x256x256 with Direct2LDS.

### TODO

The GR row assignment (8 contiguous rows per wave) stays unchanged. Only column
indices need rotation. The current row tiling from `createInternalTile` in
`LowerTile.cpp` can remain as-is.

1. Add kernel option
   - Add `LDSBankSwizzleMode` kernel option (None, Swizzle) -- model on `ScaleSkipPermlaneMode.hpp`

2. GR column rotation (start with 128x128x256 tile, no Direct2LDS)
   - Modify GR voffset column computation to apply intra-wave rotation + conditional pair-swap
   - The column transform: `col = (swap_if_even(lane % blockSize) + blockSize - (ldsRowId // 2) * 2) % blockSize`
   - Where: coordinate transforms that compute `voffset` in `LowerTile.cpp`
   - Verify: check assembly for rotation + swap instructions, trace GR pattern

3. LR column rotation
   - Modify LR address computation to apply matching rotation + `v_permlane16_swap_b32`
   - The column transform: `col = swap_0x33(((lane16Group + (ldsRowId // 2) * 2) % blockSize))`
   - Where: coordinate transforms for local read addresses
   - Verify: check assembly, traces, norm passes

4. Benchmark (128x128x256)
   - rocgdb tracing to confirm rotation pattern eliminates bank conflicts
   - Benchmark LDS read cycles (expect 4x improvement)

5. Scale up to 256x256x256 with Direct2LDS
   - Enable Direct2LDS path with `LDSBankSwizzleMode::Swizzle`
   - Verify the same rotation logic applies correctly at the larger tile size
   - Benchmark to confirm bank conflict elimination

### Follow-up: Inter-wave rotation

For certain wave group configurations (e.g., 2x2 `MIWaveGroup` where multiple waves
contribute to the same LDS region), an additional per-wave column rotation may be
needed on the GR side:

```
wave_rotation = (waveId & 1) * (2 * numRowsPerBankRow)
col = (col + intra_rotation - wave_rotation) % blockSize
```

This applies when `loadRatioGR != 0.5` in Tensile's terminology. For the primary
target (256x256x256 with 4x1 wave group, `loadRatioGR = 0.5`), intra-wave
rotation alone is sufficient.

Reference: `_grSwizzleColIds` in `SubtileBasedKernel.py` lines 778-812.
