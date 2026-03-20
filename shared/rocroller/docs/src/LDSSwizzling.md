# LDS Swizzle for Bank Conflict Elimination

mxfp4, 256x256x256 tile, `v_mfma_scale_f32_16x16x128_f8f6f4`, direct to lds.

This document analyzes LDS bank conflicts in the mxfp4 GEMM kernel and explains how a
swizzled Global Read (GR) layout eliminates 4x Local Read (LR) serialization. It compares
three variants (rocRoller, no-swizzle-no-rotate, and swizzled) and outlines the steps
to implement the conflict-free layout in rocRoller.

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
SIMDIndex  = lane // 16   (which SIMD, 0-3)
laneInSIMD = lane % 16    (position within the 16-lane SIMD)
```

## Global Load to LDS

All three kernels have the same LDS destination address pattern:

```
lds_addr[lane] = wave * 1024 + lane * 16
```

The main difference is `voffset` (per-lane global memory offset):
`voffset = col*colStride + row*rowStride`, where `rowStride = LDA * dt_size / 8`.

<details>
<summary>Variable values for this kernel</summary>

```
colStride  = load_width = 16 B       (1 dwordx4 = 32 fp4 elements in K)
rowStride  = LDA * dt_size / 8       (leading dimension in bytes; = 16384 for this kernel)
macM       = 256                     (M-dimension of the workgroup tile, in rows)
```

</details>

| | rocRoller | No-swizzle-no-rotate | Swizzled |
|---|---|---|---|
| `col` | `lane % 8` | same as rocRoller | `(col_xor + (8-2*wave)) % 8` where `col_xor = (lane%8)^1` for rows 0-1 of each half-wave (XOR swap), `lane%8` for rows 2-3 (no XOR). `newserial = (lane%32) + wave*32` |
| `row` | `lane // 8` | `(lane//8 % 4) + (lane//32)*(macM/2)` | same as no-swizzle-no-rotate |
| `voffset` | `col*colStride + row*rowStride` | same as rocRoller | same as rocRoller |
| Unique rows/instr | 8 (rows 0-7) | 8 (rows 0-3 and 128-131) | same as no-swizzle-no-rotate |

The `% 4` in the no-swizzle/swizzled `row` formula creates a half-wave split (32 lanes -> 4 rows x 8 cols) so that:

- The swizzled LR reads from all 4 wave LDS regions (`laneInSIMD // 4` selects region)
- Each region has a different GR col rotation (0, -2, -4, -6)
- 4 rotations x 4 SIMDIndex values = 16 unique bank groups per phase
- With 8 rows x 2 regions instead, only 2 rotations -> not enough

**No-swizzle-no-rotate:** same half-wave split, sequential col (`lane % 8`), all waves identical.

**Swizzled:** adds per-wave col rotation + conditional XOR pair-swap (`(lane%8)^1`) on
rows 0-1 of each half-wave only (rows 2-3 get rotation without XOR).

The tables below show which `wave.lane` loads each (M-row, K-chunk) from global memory.
Pattern repeats every 16 M-rows.

### rocRoller -- 8 contiguous rows per wave, sequential columns

| M-row | K0 | K1 | K2 | K3 | K4 | K5 | K6 | K7 |
|---|---|---|---|---|---|---|---|---|
| 0 | w0.L0 | w0.L1 | w0.L2 | w0.L3 | w0.L4 | w0.L5 | w0.L6 | w0.L7 |
| 1 | w0.L8 | w0.L9 | w0.L10 | w0.L11 | w0.L12 | w0.L13 | w0.L14 | w0.L15 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 7 | w0.L56 | w0.L57 | w0.L58 | w0.L59 | w0.L60 | w0.L61 | w0.L62 | w0.L63 |
| 8 | w1.L0 | w1.L1 | w1.L2 | w1.L3 | w1.L4 | w1.L5 | w1.L6 | w1.L7 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

All waves have identical col pattern (`lane % 8`). Wave changes every 8 rows.

### No-swizzle-no-rotate -- half-wave split, sequential columns

| M-row | K0 | K1 | K2 | K3 | K4 | K5 | K6 | K7 |
|---|---|---|---|---|---|---|---|---|
| 0 | w0.L0 | w0.L1 | w0.L2 | w0.L3 | w0.L4 | w0.L5 | w0.L6 | w0.L7 |
| 1 | w0.L8 | w0.L9 | w0.L10 | w0.L11 | w0.L12 | w0.L13 | w0.L14 | w0.L15 |
| 2 | w0.L16 | w0.L17 | w0.L18 | w0.L19 | w0.L20 | w0.L21 | w0.L22 | w0.L23 |
| 3 | w0.L24 | w0.L25 | w0.L26 | w0.L27 | w0.L28 | w0.L29 | w0.L30 | w0.L31 |
| 4 | w1.L0 | w1.L1 | w1.L2 | w1.L3 | w1.L4 | w1.L5 | w1.L6 | w1.L7 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 128 | w0.L32 | w0.L33 | w0.L34 | w0.L35 | w0.L36 | w0.L37 | w0.L38 | w0.L39 |
| 129 | w0.L40 | w0.L41 | w0.L42 | w0.L43 | w0.L44 | w0.L45 | w0.L46 | w0.L47 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

Same sequential col pattern as rocRoller. Wave changes every 4 rows (half-wave split).
Lanes 32-63 load the same pattern but from M-rows offset by macM/2 (= 128).
All 4 waves identical (no rotation).

### Swizzled -- half-wave split + XOR swap + per-wave rotation

Same row structure as no-swizzle-no-rotate, but K-chunk columns are permuted per wave:

| M-row | K0 | K1 | K2 | K3 | K4 | K5 | K6 | K7 |
|---|---|---|---|---|---|---|---|---|
| 0 | w0.L1 | w0.L0 | w0.L3 | w0.L2 | w0.L5 | w0.L4 | w0.L7 | w0.L6 |
| 1 | w0.L9 | w0.L8 | w0.L11 | w0.L10 | w0.L13 | w0.L12 | w0.L15 | w0.L14 |
| 2 | w0.L16 | w0.L17 | w0.L18 | w0.L19 | w0.L20 | w0.L21 | w0.L22 | w0.L23 |
| 3 | w0.L24 | w0.L25 | w0.L26 | w0.L27 | w0.L28 | w0.L29 | w0.L30 | w0.L31 |
| 4 | w1.L3 | w1.L2 | w1.L5 | w1.L4 | w1.L7 | w1.L6 | w1.L1 | w1.L0 |
| 5 | w1.L11 | w1.L10 | w1.L13 | w1.L12 | w1.L15 | w1.L14 | w1.L9 | w1.L8 |
| 6 | w1.L18 | w1.L19 | w1.L20 | w1.L21 | w1.L22 | w1.L23 | w1.L16 | w1.L17 |
| 7 | w1.L26 | w1.L27 | w1.L28 | w1.L29 | w1.L30 | w1.L31 | w1.L24 | w1.L25 |
| 8 | w2.L5 | w2.L4 | w2.L7 | w2.L6 | w2.L1 | w2.L0 | w2.L3 | w2.L2 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 12 | w3.L7 | w3.L6 | w3.L1 | w3.L0 | w3.L3 | w3.L2 | w3.L5 | w3.L4 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

Per-wave col rotation (shifts by -2 per wave):

| Wave | K0 | K1 | K2 | K3 | K4 | K5 | K6 | K7 | rotation |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 0 | 3 | 2 | 5 | 4 | 7 | 6 | 0 |
| 1 | 7 | 6 | 1 | 0 | 3 | 2 | 5 | 4 | -2 |
| 2 | 5 | 4 | 7 | 6 | 1 | 0 | 3 | 2 | -4 |
| 3 | 3 | 2 | 5 | 4 | 7 | 6 | 1 | 0 | -6 |

## LDS Read (`ds_read_b128`)

`ds_read_b128` has 4 phases of 16 threads each. For zero bank conflicts, each phase's
16 threads must hit 16 different bank groups. The bank group a thread hits depends on the
LDS address it reads from -- specifically the `col` (low bits of the address).

The key constraint: 16 threads per phase need 16 unique `col` values. `col` is derived
from `SIMDIndex` (4 values) combined with a rotation term. The rotation term must provide
enough diversity:

- **rocRoller**: reads from 2 wave LDS regions -> 2 rotation values -> only 4 unique bank
  groups per phase (4x serialization)
- **Swizzled**: reads from 4 wave LDS regions -> 4 rotation values (0, 2, 4, 6) -> combined
  with 4 SIMDIndex values + permlane swap -> 16 unique bank groups (no serialization)

The rotation is naturally per-wave because each wave writes to its own 1024B LDS region,
and the LR (Local Read) reads each region as a unit (`laneInSIMD // 4` selects region). All 4 rows
read from a region share the same rotation compensation, so the GR rotation must be
uniform within each region.

### LR Formula Comparison

| | rocRoller | No-swizzle-no-rotate | Swizzled |
|---|---|---|---|
| `addr` | `SIMDIndex*16 + laneInSIMD*128` | `SIMDIndex*16 + (laneInSIMD%4)*128 + (laneInSIMD//4)*1024` | `col*16 + (laneInSIMD%4)*128 + (laneInSIMD//4)*1024` |
| `col_base` | `(SIMDIndex + laneInSIMD*8) % 16` | `(SIMDIndex + (laneInSIMD%4)*8) % 16` | `(SIMDIndex + floor(laneInSIMD/4)*2) % 8` |
| `col` (swap) | col_base | col_base | exchange col_base with lane `^16` partner if `(lane%4) < 2`, via `v_permlane16_swap_b32` (mask `0x33333333`) |

LR lane mapping (same for all three kernels): `SIMDIndex` selects K-chunk column,
`laneInSIMD` selects M-row (cycles every 16). Wave partition splits at M-row 128.

<details>
<summary>Reproduce from traces</summary>

```bash
# rocRoller (LDA = K = 32768)
scripts/trace_tile_map.py build_release/trace_fp4.jsonl --lda 32768

# No-swizzle-no-rotate (LDA = 256 = macM)
scripts/trace_tile_map.py trace_noswizzle.jsonl --lda 256

# Swizzled (same LDA)
scripts/trace_tile_map.py trace_swizzle.jsonl --lda 256
```

</details>

Static reference tables: `scripts/print_macro_tiles.py`

### LR A tile -- which wave+lane reads each (M-row, K-chunk)

| M-row | K0 | K1 | K2 | K3 | K4 | K5 | K6 | K7 |
|---|---|---|---|---|---|---|---|---|
| 0 | w0,2 L0 | w0,2 L16 | w0,2 L32 | w0,2 L48 | w0,2 L0 | w0,2 L16 | w0,2 L32 | w0,2 L48 |
| 1 | w0,2 L1 | w0,2 L17 | w0,2 L33 | w0,2 L49 | w0,2 L1 | w0,2 L17 | w0,2 L33 | w0,2 L49 |
| 2 | w0,2 L2 | w0,2 L18 | w0,2 L34 | w0,2 L50 | w0,2 L2 | w0,2 L18 | w0,2 L34 | w0,2 L50 |
| 3 | w0,2 L3 | w0,2 L19 | w0,2 L35 | w0,2 L51 | w0,2 L3 | w0,2 L19 | w0,2 L35 | w0,2 L51 |
| 4 | w0,2 L4 | w0,2 L20 | w0,2 L36 | w0,2 L52 | w0,2 L4 | w0,2 L20 | w0,2 L36 | w0,2 L52 |
| 5 | w0,2 L5 | w0,2 L21 | w0,2 L37 | w0,2 L53 | w0,2 L5 | w0,2 L21 | w0,2 L37 | w0,2 L53 |
| 6 | w0,2 L6 | w0,2 L22 | w0,2 L38 | w0,2 L54 | w0,2 L6 | w0,2 L22 | w0,2 L38 | w0,2 L54 |
| 7 | w0,2 L7 | w0,2 L23 | w0,2 L39 | w0,2 L55 | w0,2 L7 | w0,2 L23 | w0,2 L39 | w0,2 L55 |
| 8 | w0,2 L8 | w0,2 L24 | w0,2 L40 | w0,2 L56 | w0,2 L8 | w0,2 L24 | w0,2 L40 | w0,2 L56 |
| 9 | w0,2 L9 | w0,2 L25 | w0,2 L41 | w0,2 L57 | w0,2 L9 | w0,2 L25 | w0,2 L41 | w0,2 L57 |
| 10 | w0,2 L10 | w0,2 L26 | w0,2 L42 | w0,2 L58 | w0,2 L10 | w0,2 L26 | w0,2 L42 | w0,2 L58 |
| 11 | w0,2 L11 | w0,2 L27 | w0,2 L43 | w0,2 L59 | w0,2 L11 | w0,2 L27 | w0,2 L43 | w0,2 L59 |
| 12 | w0,2 L12 | w0,2 L28 | w0,2 L44 | w0,2 L60 | w0,2 L12 | w0,2 L28 | w0,2 L44 | w0,2 L60 |
| 13 | w0,2 L13 | w0,2 L29 | w0,2 L45 | w0,2 L61 | w0,2 L13 | w0,2 L29 | w0,2 L45 | w0,2 L61 |
| 14 | w0,2 L14 | w0,2 L30 | w0,2 L46 | w0,2 L62 | w0,2 L14 | w0,2 L30 | w0,2 L46 | w0,2 L62 |
| 15 | w0,2 L15 | w0,2 L31 | w0,2 L47 | w0,2 L63 | w0,2 L15 | w0,2 L31 | w0,2 L47 | w0,2 L63 |
| 16 | w0,2 L0 | w0,2 L16 | w0,2 L32 | w0,2 L48 | w0,2 L0 | w0,2 L16 | w0,2 L32 | w0,2 L48 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 128 | w1,3 L0 | w1,3 L16 | w1,3 L32 | w1,3 L48 | w1,3 L0 | w1,3 L16 | w1,3 L32 | w1,3 L48 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 255 | w1,3 L15 | w1,3 L31 | w1,3 L47 | w1,3 L63 | w1,3 L15 | w1,3 L31 | w1,3 L47 | w1,3 L63 |

Each LR wave reads from all 4 GR (Global Read) wave regions (`laneInSIMD // 4` selects region),
providing 4 distinct rotations and 16/16 unique bank groups per phase.

### Bank group access per phase

The diagrams below show which threads hit which bank group (columns 0-15) per `ds_read_b128`
phase. Each row group is one phase; threads in the same phase sharing a column would conflict.

#### rocRoller / No-swizzle-no-rotate

```
  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
---------------------------------------------------------------
 T0 T20                          T1 T21
 T2 T22                          T3 T23
T12 T24                         T13 T25
T14 T26                         T15 T27
---------------------------------------------------------------
        T32 T52                         T33 T53
        T34 T54                         T35 T55
        T44 T56                         T45 T57
        T46 T58                         T47 T59
---------------------------------------------------------------
 T4 T16                          T5 T17
 T6 T18                          T7 T19
 T8 T28                          T9 T29
T10 T30                         T11 T31
---------------------------------------------------------------
        T36 T48                         T37 T49
        T38 T50                         T39 T51
        T40 T60                         T41 T61
        T42 T62                         T43 T63
```

<details>
<summary>first_bank per lane (all waves identical)</summary>

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 8 | 0 | 8 | 0 | 8 | 0 | 8 | 0 | 8 | 0 | 8 | 0 | 8 | 0 | 8 | 1 | 9 | 1 | 9 | 1 | 9 | 1 | 9 | 1 | 9 | 1 | 9 | 1 | 9 | 1 | 9 | 2 | 10 | 2 | 10 | 2 | 10 | 2 | 10 | 2 | 10 | 2 | 10 | 2 | 10 | 2 | 10 | 3 | 11 | 3 | 11 | 3 | 11 | 3 | 11 | 3 | 11 | 3 | 11 | 3 | 11 | 3 | 11 |

Only 4 of 16 bank groups active per phase; each hit by 4 threads -> 4x serialization.

</details>

#### Swizzled

```
  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
---------------------------------------------------------------
 T2  T0 T20 T22 T24 T26 T14 T12  T3  T1 T21 T23 T25 T27 T15 T13
---------------------------------------------------------------
T46 T44 T34 T32 T52 T54 T56 T58 T47 T45 T35 T33 T53 T55 T57 T59
---------------------------------------------------------------
T16 T18  T6  T4 T10  T8 T28 T30 T17 T19  T7  T5 T11  T9 T29 T31
---------------------------------------------------------------
T60 T62 T48 T50 T38 T36 T42 T40 T61 T63 T49 T51 T39 T37 T43 T41
```

<details>
<summary>first_bank per lane (wave 0)</summary>

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 9 | 0 | 8 | 3 | 11 | 2 | 10 | 5 | 13 | 4 | 12 | 7 | 15 | 6 | 14 | 0 | 8 | 1 | 9 | 2 | 10 | 3 | 11 | 4 | 12 | 5 | 13 | 6 | 14 | 7 | 15 | 3 | 11 | 2 | 10 | 5 | 13 | 4 | 12 | 7 | 15 | 6 | 14 | 1 | 9 | 0 | 8 | 2 | 10 | 3 | 11 | 4 | 12 | 5 | 13 | 6 | 14 | 7 | 15 | 0 | 8 | 1 | 9 |

All 16 of 16 bank groups active per phase; each hit by exactly 1 thread -> no serialization.

</details>

## Implementing in rocRoller

Gated by a new compile-time kernel option (`LDSBankSwizzleMode`). Two sides need to change:
GR voffset and LR address. Both can be expressed as pure per-lane arithmetic in the
coordinate graph -- no new instruction emission needed. All parameters are derived from
the tile config.

### TODO

Step 1: Enable half-wave GR pattern
- Add `LDSBankSwizzleMode` kernel option (None, Swizzle) -- model on `ScaleSkipPermlaneMode.hpp`
- New coordinate transform: remap GR voffset to use half-wave split (sequential cols, no rotation or XOR); LDS dest unchanged
- Update LR address to match: with the half-wave split, each wave's LDS region now holds 2 rows per wave (not 4), so LR must read 2 rows from each of the 4 waves' regions rather than 4 contiguous rows from 4 waves' regions

Step 2: Enable swizzle
- Extend transform: add per-wave col rotation + conditional XOR pair-swap on GR; add inverse col rotation (closed-form per-lane arithmetic) on LR
