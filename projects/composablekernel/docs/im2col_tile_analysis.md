# Im2col Tile Awareness Analysis

## Problem Setup

2D grouped convolution, forward pass, FP16, gfx950 (MI350).

| Parameter | Value |
|-----------|-------|
| G (groups) | 2 |
| N (batch) | 32 |
| H × W (input spatial) | 64 × 64 |
| Ho × Wo (output spatial) | 61 × 61 |
| Y × X (filter) | 4 × 4 |
| C (input channels per group) | 32 |
| K (output channels per group) | 32 |
| Stride / Dilation / Padding | 1 / 1 / 0 |
| Layout (input / weight / output) | NHWGC / GKYXC / NHWGK |

Kernel: `gemm_pipeline_AgBgCrCompV3_16x64x64` — confirmed from runtime output.

---

## GEMM Problem Dimensions

The convolution is mapped to GEMM as follows:

| Matrix | GEMM size | Tile size | Role |
|--------|-----------|-----------|------|
| A | M_gemm × K_gemm = **119,072 × 512** | M_tile × K_tile = **16 × 64** | Input (im2col) |
| B | N_gemm × K_gemm = **32 × 512**       | N_tile × K_tile = **64 × 64** | Weight |
| C | M_gemm × N_gemm = **119,072 × 32**   | M_tile × N_tile = **16 × 64** | Output |

Grid: `{7,442, 2, 1}` — 7,442 M-tiles per group, 2 groups (gridDim.y = G = 2).

Verified from debug data: `m_start = blockIdx.x × 16` exactly for all 117,484 sampled blocks.

---

## Index Mappings

### m_gemm → conv input indices

```
n_conv  =  m_gemm / (Ho × Wo)      =  m_gemm / 3721
ho      = (m_gemm % 3721) / Wo     = (m_gemm % 3721) / 61
wo      =  m_gemm % Wo             =  m_gemm % 61
```

### k_gemm → conv filter indices

```
y       =  k_gemm / (X × C)        =  k_gemm / 128
x       = (k_gemm % 128) / C       = (k_gemm % 128) / 32
c_conv  =  k_gemm % C              =  k_gemm % 32
```

### n_gemm → output filter index

```
k_conv  =  n_gemm
```

---

## A Matrix Access (Im2col — the Expensive Part)

`A[m_gemm, k_gemm]` reads input tensor at `input[n_conv, ho + y, wo + x, g, c_conv]`.

Linear byte offset (NHWGC layout, Wi = Hi = 64, G = 2, C = 32):

```
offset = n_conv × 262144  +  (ho + y) × 4096  +  (wo + x) × 64  +  g × 32  +  c_conv
```

Strides: `NStride = 262144`, `HiStride = 4096`, `WiStride = 64`, `CStride = 1`.

### Key decomposition: M and K contributions are independent

```
offset = [n_conv × 262144  +  ho × 4096  +  wo × 64]    ← M_base(m_gemm)
       + [y × 4096         +  x × 64    +  c_conv  ]    ← K_offset(k_gemm)
       + g × 32                                          ← constant per kernel launch
```

**M_base depends only on m_gemm. K_offset depends only on k_gemm. They add independently.**

This decomposition is the foundation for eliminating im2col arithmetic from the hot path.

---

## Pattern Analysis: M-tile

Each block covers `m_gemm ∈ [m_start, m_start + 15]` where `m_start = blockIdx.x × 16`.

Decoding the tile origin:
```
n_conv  = m_start / 3721     ← constant for ALL 16 rows (16 << 3721)
ho_0    = (m_start % 3721) / 61
wo_0    = m_start % 61
```

For row `i` within the tile (`i = 0..15`), two cases arise:

### Case A — 83.6% of tiles (`wo_0 + 15 < 61`, no ho-row wrap)

```
ho_i = ho_0        (constant across all 16 rows)
wo_i = wo_0 + i    (linear increment)

M_base[i] = M_base[0] + i × 64    ← pure addition, no division needed
```

### Case B — 16.4% of tiles (`wo_0 + 15 ≥ 61`, one ho-row boundary crossed)

Let `boundary = 61 - wo_0` (the row index where ho wraps).

```
for i < boundary:   ho_i = ho_0,     wo_i = wo_0 + i
for i ≥ boundary:   ho_i = ho_0 + 1, wo_i = i - boundary

M_base[i] = M_base[0] + i × 64 + (i ≥ boundary ? 192 : 0)
```

where `192 = HiStride − Wo × WiStride = 4096 − 61 × 64`.

**n_conv is constant in effectively 100% of tiles** (a tile would need to cross an N-image
boundary, which requires `m_start % 3721 + 16 ≥ 3721` — probability < 0.5%).

---

## Pattern Analysis: K-tile

Each K-tile covers `k_gemm ∈ [k_start, k_start + 63]` where `k_start = k_tile × 64`.

From the data, every K-tile has:

| K-tile | k_gemm range | y | x sequence         |
|--------|-------------|---|--------------------|
| 0      | [0,   63]   | 0 | 0 (×32), 1 (×32)  |
| 1      | [64,  127]  | 0 | 2 (×32), 3 (×32)  |
| 2      | [128, 191]  | 1 | 0 (×32), 1 (×32)  |
| 3      | [192, 255]  | 1 | 2 (×32), 3 (×32)  |
| 4      | [256, 319]  | 2 | 0 (×32), 1 (×32)  |
| 5      | [320, 383]  | 2 | 2 (×32), 3 (×32)  |
| 6      | [384, 447]  | 3 | 0 (×32), 1 (×32)  |
| 7      | [448, 511]  | 3 | 2 (×32), 3 (×32)  |

Within each K-tile:
- **y is constant** for all 64 elements
- **x changes exactly once** at the midpoint (every 32 steps = every C elements)
- **c_conv cycles 0..31 twice** (full sweep, stride 1)

K_offset advances incrementally:
```
c_conv  += 1  every step         (stride = 1)
x       += 1  at k_gemm % 32    (stride = 64, once per K-tile)
y       += 1  between K-tiles    (stride = 4096, never within a K-tile)
```

---

## Precomputation Strategy

### At block start — amortized over 8 K-tiles (16 × 8 = 128 A element loads)

```
1. n_conv = m_start / 3721                  \
2. ho_0   = (m_start % 3721) / 61           |  3 integer divisions (once per block)
3. wo_0   = m_start % 61                    /

4. M_base[0..15]:
       base0    = n_conv × 262144 + ho_0 × 4096 + wo_0 × 64
       boundary = 61 - wo_0
       M_base[i] = base0 + i × 64 + (i >= boundary ? 192 : 0)
   → 16 additions + 1 comparison  (stored in registers)
```

### At each K-tile start — once per K-tile (8 times per block)

```
5. y      = k_tile >> 1
   x_0    = (k_tile & 1) ? 2 : 0
   K_base = y × 4096 + x_0 × 64     ← no division, known from k_tile index
```

### Per element access — hot path

```
offset = M_base[local_m] + K_base + (local_k >= 32 ? 64 : 0) + c_conv
```

where `c_conv = local_k % 32` increments by 1 each step.

---

## Impact Assessment

### Current cost (tensor transform machinery)

For every `move_tensor_coordinate` call in the K-loop, the transform chain re-evaluates:
- Unmerge `m_gemm → (n_conv, ho, wo)`: **2 integer divisions + 2 modulos**
- Unmerge `k_gemm → (y, x, c_conv)`: **2 integer divisions + 2 modulos**
- Embed transforms (multiply + add for each spatial dimension)

These execute for every element of A loaded, every K-loop step.

From the profiler (rocprof-compute on gfx950):
- **VALU IOPs: 17,778 Giop/s at 24.7% of peak**
- **INT32 instructions: 19.4M per kernel** — ~42% of all VALU instructions
- **INT64 instructions: 2.9M per kernel**
- **MFMA utilization: only 8.4%** — the integer overhead is the bottleneck

### With precomputation

| Operation | Current (per element) | With precomputation |
|-----------|----------------------|---------------------|
| M_base divisions | 2 divmod pairs | 0 (precomputed at block start) |
| K_offset divisions | 2 divmod pairs | 0 (increment only) |
| Address computation | ~8–12 integer ops | 2 additions + 1 register load |

The 3 integer divisions at block start are amortized over 128 A-element loads per block,
reducing their per-element cost to ~0.02 divisions per load.

---

## Profiling Reference

Profiled on AMD MI350 (gfx950), ROCm 7.1.1, rocprof-compute v3.

| Metric | Value | Peak | % of Peak |
|--------|-------|------|-----------|
| MFMA FLOPs (F16) | 193,129 Gflop/s | 2,306,867 | 8.37% |
| VALU IOPs | 17,779 Giop/s | 72,090 | 24.66% |
| SALU Utilization | 41.6% | — | — |
| VALU Utilization | 104.4% | — | — |
| MFMA Utilization | 8.3% | — | — |
| INT32 instructions | 19.4M / kernel | — | — |
| INT64 instructions | 2.9M / kernel | — | — |
| MFMA instructions | 0.95M / kernel | — | — |
| Ratio VALU:MFMA | 49:1 | — | — |
