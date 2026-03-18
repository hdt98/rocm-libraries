# rocRoller fp4 vs. LDS Swizzled Kernel

mxfp4, 256x256x256 tile, `v_mfma_scale_f32_16x16x128_f8f6f4`, direct to lds.

## Background

LDS: 64 banks, each 1 dword wide.

For `ds_read_b128`, the effective granularity is 16 dwordx4 bank groups.

`ds_read_b128` occurs in 4 phases, with 16 threads each:

```
Phase 0:  T0-T3,   T12-T15,  T20-T23,  T24-T27
Phase 1:  T32-T35, T44-T47,  T52-T55,  T56-T59
Phase 2:  T4-T7,   T8-T11,   T16-T19,  T28-T31
Phase 3:  T36-T39, T40-T43,  T48-T51,  T60-T63
```

Only banks accessed by threads in the same phase can cause conflicts.

### rocRoller / No-swizzle

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
<summary>Commands to reproduce</summary>

```bash
ADDR=$(jq -r 'select(.instruction | test("ds_read_b128")) | .address' trace_fp4.jsonl | head -1)
jq "select(.address == \"$ADDR\" and .wave == 0) | .derived.first_bank" trace_fp4.jsonl | head -1 | \
    python3 scripts/lds_bank_table.py
```

</details>

### Swizzled

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
<summary>Commands to reproduce</summary>

```bash
ADDR=$(jq -r 'select(.instruction | test("ds_read_b128")) | .address' trace_swizzle.jsonl | head -1)
jq "select(.address == \"$ADDR\" and .wave == 0) | .derived.first_bank" trace_swizzle.jsonl | head -1 | \
    python3 scripts/lds_bank_table.py
```

</details>

---

## LDS Read (`ds_read_b128`)

### LR Formula Comparison

For `v_mfma_scale_f32_16x16x128_f8f6f4`, one MFMA instruction executes across the full
64-lane wave, split into 4 SIMDs of 16 lanes each. Each SIMD produces a 16x16 output block
covering a different 16-row slice of the M-dimension. `SIMDIndex` indexes which of these
4 SIMDs a lane belongs to; `laneInSIMD` is the lane's position within its SIMD
(rocRoller's names from `LowerTile.cpp`):

```
SIMDIndex  = lane // 16   (which SIMD, 0-3; maps to 4 disjoint row slices)
laneInSIMD = lane % 16    (position within the 16-lane SIMD)
```

| | rocRoller | No-swizzle | Swizzled |
|---|---|---|---|
| `addr` | `SIMDIndex*16 + laneInSIMD*128` | same | `col*16 + (laneInSIMD//4)*1024 + ((laneInSIMD%4)//2)*256` |
| `col_base` | `(SIMDIndex + laneInSIMD*8) % 16` | same | `(SIMDIndex + floor(laneInSIMD/4)*2) % 8` |
| `col` (swap) | col_base | col_base | exchange col_base with lane `^16` partner if `(lane%4) < 2`, via `v_permlane16_swap_b32` (mask `0x33333333`) |

rocRoller and no-swizzle have identical LR -- the LDS read address depends only on lane
position, not on the GR pattern. Conflict-free LR requires the swizzle/rotation in GR to
lay data out in LDS in the specific arrangement the two-step LR formula expects.

### LDS Read -- Result for wave 0

| Lanes | laneInSIMD | SIMDIndex | col_base | col |
|---|---|---|---|---|
| 0-3 | 0-3 | 0 | [0,0,0,0] | [1,1,0,0] |
| 4-7 | 4-7 | 0 | [2,2,2,2] | [3,3,2,2] |
| 8-11 | 8-11 | 0 | [4,4,4,4] | [5,5,4,4] |
| 12-15 | 12-15 | 0 | [6,6,6,6] | [7,7,6,6] |
| 16-19 | 0-3 | 1 | [1,1,1,1] | [0,0,1,1] |
| 20-23 | 4-7 | 1 | [3,3,3,3] | [2,2,3,3] |
| 24-27 | 8-11 | 1 | [5,5,5,5] | [4,4,5,5] |
| 28-31 | 12-15 | 1 | [7,7,7,7] | [6,6,7,7] |
| 32-35 | 0-3 | 2 | [2,2,2,2] | [3,3,2,2] |
| 36-39 | 4-7 | 2 | [4,4,4,4] | [5,5,4,4] |
| 40-43 | 8-11 | 2 | [6,6,6,6] | [7,7,6,6] |
| 44-47 | 12-15 | 2 | [0,0,0,0] | [1,1,0,0] |
| 48-51 | 0-3 | 3 | [3,3,3,3] | [2,2,3,3] |
| 52-55 | 4-7 | 3 | [5,5,5,5] | [4,4,5,5] |
| 56-59 | 8-11 | 3 | [7,7,7,7] | [6,6,7,7] |
| 60-63 | 12-15 | 3 | [1,1,1,1] | [0,0,1,1] |

Each final `col` maps to bank group `col` (since each col is 16 B = 4 dwords). The 16 lanes
in each phase land on all 16 distinct bank groups -- one thread per bank, zero serialization.

The GR and LR transforms are designed as inverses: data written to LDS at offset
`col_GR * 16` by lane `l` during global load is read back at the same offset by the lane
whose LR `col` equals `col_GR`, ensuring every thread gets the data it needs while
maximizing bank utilization on both the write and read sides.

### rocRoller fp4 Kernel

`relative_addr` (wave 0):

| laneInSIMD | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0-15 | 0 | 128 | 256 | 384 | 512 | 640 | 768 | 896 | 1024 | 1152 | 1280 | 1408 | 1536 | 1664 | 1792 | 1920 |
| L16-31 | 16 | 144 | 272 | 400 | 528 | 656 | 784 | 912 | 1040 | 1168 | 1296 | 1424 | 1552 | 1680 | 1808 | 1936 |
| L32-47 | 32 | 160 | 288 | 416 | 544 | 672 | 800 | 928 | 1056 | 1184 | 1312 | 1440 | 1568 | 1696 | 1824 | 1952 |
| L48-63 | 48 | 176 | 304 | 432 | 560 | 688 | 816 | 944 | 1072 | 1200 | 1328 | 1456 | 1584 | 1712 | 1840 | 1968 |

`first_bank` (all waves identical -- `ds_read_b128` address depends only on lane position, not wave):

| laneInSIMD | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0-15 | 0 | 8 | 0 | 8 | 0 | 8 | 0 | 8 | 0 | 8 | 0 | 8 | 0 | 8 | 0 | 8 |
| L16-31 | 1 | 9 | 1 | 9 | 1 | 9 | 1 | 9 | 1 | 9 | 1 | 9 | 1 | 9 | 1 | 9 |
| L32-47 | 2 | 10 | 2 | 10 | 2 | 10 | 2 | 10 | 2 | 10 | 2 | 10 | 2 | 10 | 2 | 10 |
| L48-63 | 3 | 11 | 3 | 11 | 3 | 11 | 3 | 11 | 3 | 11 | 3 | 11 | 3 | 11 | 3 | 11 |

<details>
<summary>Commands to reproduce</summary>

```bash
ADDR=$(jq -r 'select(.instruction | test("ds_read_b128 v\\[92:95\\]")) | .address' trace_fp4.jsonl | head -1)
jq "select(.address == \"$ADDR\" and .wave == 0) | .derived.first_bank" trace_fp4.jsonl | head -1 | \
    python3 scripts/lds_bank_table.py
```

</details>

Only **4 of 16** bank groups active per phase; each hit by **4 threads** -> **4x serialization**.

### No-swizzle Kernel

`first_bank` (wave 0; all waves identical):

Identical to rocRoller -- sequential col in GR produces the same LDS layout, so the same
LR address formula applies and the same 4 bank groups are active per phase.

<details>
<summary>Commands to reproduce</summary>

```bash
ADDR=$(jq -r 'select(.instruction == "ds_read_b128 v[36:39], v218") | .address' trace_noswizzle.jsonl | head -1)
jq "select(.address == \"$ADDR\" and .wave == 0) | .derived.first_bank" trace_noswizzle.jsonl | head -1 | \
    python3 scripts/lds_bank_table.py
```

</details>

### Swizzled Kernel

`relative_addr` (wave 0):

| laneInSIMD | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0-15 | 16 | 144 | 256 | 384 | 1072 | 1200 | 1312 | 1440 | 2128 | 2256 | 2368 | 2496 | 3184 | 3312 | 3424 | 3552 |
| L16-31 | 0 | 128 | 272 | 400 | 1056 | 1184 | 1328 | 1456 | 2112 | 2240 | 2384 | 2512 | 3168 | 3296 | 3440 | 3568 |
| L32-47 | 48 | 176 | 288 | 416 | 1104 | 1232 | 1344 | 1472 | 2160 | 2288 | 2400 | 2528 | 3088 | 3216 | 3328 | 3456 |
| L48-63 | 32 | 160 | 304 | 432 | 1088 | 1216 | 1360 | 1488 | 2144 | 2272 | 2416 | 2544 | 3072 | 3200 | 3344 | 3472 |

`first_bank` (wave 0):

| laneInSIMD | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0-15 | 1 | 9 | 0 | 8 | 3 | 11 | 2 | 10 | 5 | 13 | 4 | 12 | 7 | 15 | 6 | 14 |
| L16-31 | 0 | 8 | 1 | 9 | 2 | 10 | 3 | 11 | 4 | 12 | 5 | 13 | 6 | 14 | 7 | 15 |
| L32-47 | 3 | 11 | 2 | 10 | 5 | 13 | 4 | 12 | 7 | 15 | 6 | 14 | 1 | 9 | 0 | 8 |
| L48-63 | 2 | 10 | 3 | 11 | 4 | 12 | 5 | 13 | 6 | 14 | 7 | 15 | 0 | 8 | 1 | 9 |

<details>
<summary>Commands to reproduce</summary>

```bash
ADDR=$(jq -r 'select(.instruction == "ds_read_b128 v[36:39], v218") | .address' trace_swizzle.jsonl | head -1)
jq "select(.address == \"$ADDR\" and .wave == 0) | .derived.first_bank" trace_swizzle.jsonl | head -1 | \
    python3 scripts/lds_bank_table.py
```

</details>

All **16 of 16** bank groups active per phase; each hit by exactly **1 thread** -> no serialization.

---

## Global Read to LDS

Both kernels have same LDS dest address pattern:

```
lds_addr[lane] = wave * 1024 + lane * 16
```

Main difference is `voffset`, the per-lane offset.

<details>
<summary>Variable values for this kernel</summary>

```
colStride  = 16 B            (1 dwordx4 = 32 fp4 elements in K)
rowStride  = K/2 = 16384 B   (K fp4 elements packed across one row of A^T)
macM       = 256             (M-dimension of the workgroup tile, in rows)
```

</details>

| | rocRoller | No-swizzle | Swizzled |
|---|---|---|---|
| `col` | `lane % 8` | `lane % 8` | `((lane%8)^1 + (8-2*wave)) % 8` |
| `row` | `lane // 8` | `(lane//8 % 4) + (lane//32)*(macM/2)` | `(lane//8 % 4) + (lane//32)*(macM/2)` |
| `voffset` | `col*colStride + row*rowStride` | `col*colStride + row*rowStride` | `col*colStride + row*rowStride` |
| Unique rows/instr | 8 (rows 0-7) | 8 (rows 0-3 and 128-131) | 8 (rows 0-3 and 128-131) |

```
rocRoller (split at 64):
  lanes  0- 7: row 0
  lanes  8-15: row 1
  lanes 16-23: row 2
  lanes 24-31: row 3
  lanes 32-39: row 4
  lanes 40-47: row 5
  lanes 48-55: row 6
  lanes 56-63: row 7

No-swizzle / Swizzled (split at 32):
  lanes  0- 7: row 0
  lanes  8-15: row 1
  lanes 16-23: row 2
  lanes 24-31: row 3
  lanes 32-39: row macM/2     (= 128)
  lanes 40-47: row macM/2 + 1 (= 129)
  lanes 48-55: row macM/2 + 2 (= 130)
  lanes 56-63: row macM/2 + 3 (= 131)
```

The `% 4` in the row term serves two purposes:

**MFMA SIMD alignment:** The LDS read uses `SIMDIndex = lane // 16` (0-3) as the row
selector, so each SIMD must load from a distinct row. This requires the GR to cover exactly
4 rows -- one per SIMD.

**Enables per-wave col rotation:** With exactly 4 rows, each wave loads a 4 row x 8 col
subtile. The 4 waves rotate col by -2 each, cycling through all 8 positions in steps of 2
without collision across waves. With 8 rows (rocRoller's pattern), each SIMD would span
2 rows instead of 1, breaking the alignment the LDS read requires.

**No-swizzle:** adopts this row structure, keeps sequential col (`lane % 8`). All 4 waves identical.

**Swizzled:** same row structure, adds XOR swap + per-wave col rotation (`((lane%8)^1 + (8-2*wave)) % 8`).

### rocRoller -- sequential columns

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| K-row 0 (L0-7) | 0 | 16 | 32 | 48 | 64 | 80 | 96 | 112 |
| K-row 1 (L8-15) | 16384 | 16400 | 16416 | 16432 | 16448 | 16464 | 16480 | 16496 |
| K-row 2 (L16-23) | 32768 | 32784 | 32800 | 32816 | 32832 | 32848 | 32864 | 32880 |
| K-row 3 (L24-31) | 49152 | 49168 | 49184 | 49200 | 49216 | 49232 | 49248 | 49264 |
| K-row 4 (L32-39) | 65536 | 65552 | 65568 | 65584 | 65600 | 65616 | 65632 | 65648 |
| K-row 5 (L40-47) | 81920 | 81936 | 81952 | 81968 | 81984 | 82000 | 82016 | 82032 |
| K-row 6 (L48-55) | 98304 | 98320 | 98336 | 98352 | 98368 | 98384 | 98400 | 98416 |
| K-row 7 (L56-63) | 114688 | 114704 | 114720 | 114736 | 114752 | 114768 | 114784 | 114800 |

<details>
<summary>Commands to reproduce</summary>

```bash
ADDR=$(jq -r 'select(.family == "buffer" and (.instruction | test("offen lds"))) | .address' trace_fp4.jsonl | head -1)
jq -c "select(.address == \"$ADDR\" and .wave == 0) |
    {global_rel: .derived.relative_addr, lds_rel: .derived.lds_relative_addr}" trace_fp4.jsonl | head -1
```

</details>

### No-swizzle -- half-wave split, sequential columns

Sequential col assignment identical to rocRoller, but with the half-wave split.
All 4 waves are identical (no rotation).

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | |
|---|---|---|---|---|---|---|---|---|---|
| K-row 0 (L0-7) | 0 | 16 | 32 | 48 | 64 | 80 | 96 | 112 | |
| K-row 1 (L8-15) | 16384 | 16400 | 16416 | 16432 | 16448 | 16464 | 16480 | 16496 | |
| K-row 2 (L16-23) | 32768 | 32784 | 32800 | 32816 | 32832 | 32848 | 32864 | 32880 | |
| K-row 3 (L24-31) | 49152 | 49168 | 49184 | 49200 | 49216 | 49232 | 49248 | 49264 | |
| K-row 0 (L32-39) | 0 | 16 | 32 | 48 | 64 | 80 | 96 | 112 | half-wave 1 (global base +2 097 152 B) |
| K-row 1 (L40-47) | 16384 | 16400 | 16416 | 16432 | 16448 | 16464 | 16480 | 16496 | |
| K-row 2 (L48-55) | 32768 | 32784 | 32800 | 32816 | 32832 | 32848 | 32864 | 32880 | |
| K-row 3 (L56-63) | 49152 | 49168 | 49184 | 49200 | 49216 | 49232 | 49248 | 49264 | |

<details>
<summary>Commands to reproduce</summary>

```bash
ADDR=$(jq -r 'select(.family == "buffer" and (.instruction | test("offen lds"))) | .address' trace_noswizzle.jsonl | head -1)
for wave in 0 1 2 3; do
    jq -c "select(.address == \"$ADDR\" and .wave == $wave) |
        {wave: .wave, global_rel: .derived.relative_addr, lds_rel: .derived.lds_relative_addr}" trace_noswizzle.jsonl | head -1
done
```

</details>

### Swizzled -- half-wave split + XOR swap + per-wave rotation

The rotation is zero for wave 0 and shifts by -2 columns (mod 8) per subsequent wave:

| Wave | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | rotation |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 0 | 3 | 2 | 5 | 4 | 7 | 6 | 0 |
| 1 | 7 | 6 | 1 | 0 | 3 | 2 | 5 | 4 | -2 |
| 2 | 5 | 4 | 7 | 6 | 1 | 0 | 3 | 2 | -4 |
| 3 | 3 | 2 | 5 | 4 | 7 | 6 | 1 | 0 | -6 |

Each wave writes to its own 1 024 B region of LDS (`absolute_lds_base = wave * 1024 B`). The
voffset tables below show offsets relative to each wave's LDS base -- the pattern is the same
for both half-waves within a wave (half-wave 1 global addresses are shifted by +2 097 152 B).

**Wave 0** (LDS base 0 B; half-wave 0: A cols 0-127 | half-wave 1: A cols 128-255, global base +2 097 152 B):

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| K-row 0 (L0-7) | 16 | 0 | 48 | 32 | 80 | 64 | 112 | 96 |
| K-row 1 (L8-15) | 16400 | 16384 | 16432 | 16416 | 16464 | 16448 | 16496 | 16480 |
| K-row 2 (L16-23) | 32768 | 32784 | 32800 | 32816 | 32832 | 32848 | 32864 | 32880 |
| K-row 3 (L24-31) | 49152 | 49168 | 49184 | 49200 | 49216 | 49232 | 49248 | 49264 |
| K-row 0 (L32-39) | 16 | 0 | 48 | 32 | 80 | 64 | 112 | 96 |
| K-row 1 (L40-47) | 16400 | 16384 | 16432 | 16416 | 16464 | 16448 | 16496 | 16480 |
| K-row 2 (L48-55) | 32768 | 32784 | 32800 | 32816 | 32832 | 32848 | 32864 | 32880 |
| K-row 3 (L56-63) | 49152 | 49168 | 49184 | 49200 | 49216 | 49232 | 49248 | 49264 |

**Wave 1** (LDS base 1 024 B, rotation -2):

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| K-row 0 (L0-7) | 112 | 96 | 16 | 0 | 48 | 32 | 80 | 64 |
| K-row 1 (L8-15) | 16496 | 16480 | 16400 | 16384 | 16432 | 16416 | 16464 | 16448 |
| K-row 2 (L16-23) | 32864 | 32880 | 32768 | 32784 | 32800 | 32816 | 32832 | 32848 |
| K-row 3 (L24-31) | 49248 | 49264 | 49152 | 49168 | 49184 | 49200 | 49216 | 49232 |
| K-row 0 (L32-39) | 112 | 96 | 16 | 0 | 48 | 32 | 80 | 64 |
| K-row 1 (L40-47) | 16496 | 16480 | 16400 | 16384 | 16432 | 16416 | 16464 | 16448 |
| K-row 2 (L48-55) | 32864 | 32880 | 32768 | 32784 | 32800 | 32816 | 32832 | 32848 |
| K-row 3 (L56-63) | 49248 | 49264 | 49152 | 49168 | 49184 | 49200 | 49216 | 49232 |

**Wave 2** (LDS base 2 048 B, rotation -4):

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| K-row 0 (L0-7) | 80 | 64 | 112 | 96 | 16 | 0 | 48 | 32 |
| K-row 1 (L8-15) | 16464 | 16448 | 16496 | 16480 | 16400 | 16384 | 16432 | 16416 |
| K-row 2 (L16-23) | 32832 | 32848 | 32864 | 32880 | 32768 | 32784 | 32800 | 32816 |
| K-row 3 (L24-31) | 49216 | 49232 | 49248 | 49264 | 49152 | 49168 | 49184 | 49200 |
| K-row 0 (L32-39) | 80 | 64 | 112 | 96 | 16 | 0 | 48 | 32 |
| K-row 1 (L40-47) | 16464 | 16448 | 16496 | 16480 | 16400 | 16384 | 16432 | 16416 |
| K-row 2 (L48-55) | 32832 | 32848 | 32864 | 32880 | 32768 | 32784 | 32800 | 32816 |
| K-row 3 (L56-63) | 49216 | 49232 | 49248 | 49264 | 49152 | 49168 | 49184 | 49200 |

**Wave 3** (LDS base 3 072 B, rotation -6):

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| K-row 0 (L0-7) | 48 | 32 | 80 | 64 | 112 | 96 | 16 | 0 |
| K-row 1 (L8-15) | 16432 | 16416 | 16464 | 16448 | 16496 | 16480 | 16400 | 16384 |
| K-row 2 (L16-23) | 32800 | 32816 | 32832 | 32848 | 32864 | 32880 | 32768 | 32784 |
| K-row 3 (L24-31) | 49184 | 49200 | 49216 | 49232 | 49248 | 49264 | 49152 | 49168 |
| K-row 0 (L32-39) | 48 | 32 | 80 | 64 | 112 | 96 | 16 | 0 |
| K-row 1 (L40-47) | 16432 | 16416 | 16464 | 16448 | 16496 | 16480 | 16400 | 16384 |
| K-row 2 (L48-55) | 32800 | 32816 | 32832 | 32848 | 32864 | 32880 | 32768 | 32784 |
| K-row 3 (L56-63) | 49184 | 49200 | 49216 | 49232 | 49248 | 49264 | 49152 | 49168 |

<details>
<summary>Commands to reproduce</summary>

```bash
ADDR=$(jq -r 'select(.family == "buffer" and (.instruction | test("offen lds"))) | .address' trace_swizzle.jsonl | head -1)
for wave in 0 1 2 3; do
    jq -c "select(.address == \"$ADDR\" and .wave == $wave) |
        {wave: .wave, global_rel: .derived.relative_addr, lds_rel: .derived.lds_relative_addr}" trace_swizzle.jsonl | head -1
done
```

</details>
