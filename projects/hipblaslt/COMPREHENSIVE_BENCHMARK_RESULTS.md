# Multi-MacroTile Benchmark Results

**Device:** AMD Instinct MI355X (256 CUs, gfx950)  
**Precision:** FP16  
**hipBLASLt Version:** 100202  
**Date:** 2026-04-20  
**Config:** `--device 7 --api_method c -i 100 -j 100 --multi_macrotile --split_strategy 17 --num_splits 2 --l2_cache_hints`

---

## Summary

| Metric | Value |
|--------|-------|
| **Problems tested** | **46** |
| Auto-disabled (safe, 0% change) | 6 |
| Active comparisons | 40 |
| **Wins (>0.5%)** | **33 / 40 (82%)** |
| Losses (>0.5%) | 7 / 40 (18%) |
| **Best gain** | **+64.8%** (15360x15360x8192) |
| Worst loss | -11.8% (10240x6144x4096) |
| **Average gain (all active)** | **+22.9%** |
| **Average gain (wins only)** | **+28.7%** |

---

## K Scaling (M=N=10240, K varies)

| K | Baseline (TF) | S17 (TF) | Gain | Winning Split |
|------|--------------|----------|------|---------------|
| 1024 | 0.883 | --- | 0% | auto-disabled |
| 2048 | 1.007 | --- | 0% | auto-disabled |
| 3072 | 1.062 | --- | 0% | auto-disabled |
| 4096 | 1.094 | 1.180 | **+7.9%** | asym-40/60 [4096,6144] |
| 5120 | 1.086 | 1.192 | **+9.8%** | asym-60/40 [6144,4096] |
| 6144 | 1.045 | 1.213 | **+16.1%** | asym-40/60 [4096,6144] |
| 7168 | 1.002 | 1.209 | **+20.7%** | asym-40/60 [4096,6144] |
| 8192 | 0.967 | 1.209 | **+25.0%** | asym-60/40 [6144,4096] |
| 10240 | 0.981 | 1.199 | **+22.2%** | asym-60/40 [6144,4096] |
| 12288 | 0.944 | 1.196 | **+26.7%** | asym-40/60 [4096,6144] |
| 14336 | 0.886 | 1.191 | **+34.4%** | asym-40/60 [4096,6144] |
| 16384 | 0.920 | 1.178 | **+28.0%** | asym-40/60 [4096,6144] |
| 32768 | 0.859 | 1.176 | **+36.9%** | asym-40/60 [4096,6144] |

S17 wins for **all K >= 4096** with gains from +7.9% to +36.9%. The [4096, 6144] asymmetric split is consistently optimal for the 10240 M-dimension.

---

## Square Matrix Scaling (K=8192, M=N varies)

| M=N | Baseline (TF) | S17 (TF) | Gain | Winning Split |
|------|--------------|----------|------|---------------|
| 10752 | 1.182 | 1.193 | **+0.9%** | asym-70/30 [7424,3328] |
| 11264 | 1.250 | 1.289 | **+3.1%** | uniform [5632,5632] |
| 11520 | 1.228 | 1.126 | -8.3% | asym-60/40 [6912,4608] |
| 11776 | 0.987 | 1.197 | **+21.3%** | asym-30/70 [3456,8320] |
| 12288 | 0.890 | 1.182 | **+32.8%** | pow2-2k [2048,10240] |
| 12800 | 1.244 | 1.236 | -0.6% | uniform [6400,6400] |
| 13056 | 0.988 | 1.172 | **+18.6%** | pow2-2k [2048,11008] |
| 13312 | 1.186 | 1.146 | -3.4% | pow2-2k [2048,11264] |
| 13824 | 0.791 | 1.201 | **+51.8%** | uniform [6912,6912] |
| 14080 | 1.217 | 1.183 | -2.8% | pow2-8k [8192,5888] |
| 14336 | 0.809 | 1.135 | **+40.3%** | pow2-2k [2048,12288] |
| 14848 | 0.799 | 1.236 | **+54.7%** | uniform [7424,7424] |
| 15104 | 1.180 | 1.165 | -1.3% | uniform [7552,7552] |
| 15360 | 0.679 | 1.119 | **+64.8%** | asym-30/70 [4608,10752] |
| 16384 | 0.839 | 1.222 | **+45.6%** | asym-40/60 [6528,9856] |

Massive gains on "performance valley" dimensions (0.68-0.89 TF baseline). Regressions only where baseline is already efficient (> 1.15 TF).

---

## Rectangular Matrices (K=8192)

| Problem | Baseline (TF) | S17 (TF) | Gain | Winning Split |
|---------|--------------|----------|------|---------------|
| 12288x6144 | 1.041 | 1.197 | **+15.0%** | pow2-2k [2048,10240] |
| 6144x12288 | 1.046 | 1.160 | **+10.9%** | asym-40/60 [2432,3712] |
| 16384x8192 | 0.896 | 1.179 | **+31.6%** | asym-70/30 [11392,4992] |
| 8192x16384 | 0.900 | 1.147 | **+27.4%** | asym-30/70 [2432,5760] |
| 20480x10240 | 0.830 | 1.238 | **+49.2%** | asym-30/70 [6144,14336] |
| 10240x20480 | 0.859 | 1.241 | **+44.5%** | asym-30/70 [3072,7168] |
| 12288x10240 | 0.884 | 1.291 | **+46.0%** | uniform [6144,6144] |
| 10240x12288 | 0.891 | 1.260 | **+41.4%** | uniform [5120,5120] |
| 10240x5120 | 1.053 | 1.125 | **+6.8%** | asym-30/70 [3072,7168] |
| 5120x10240 | 1.024 | 1.016 | -0.8% | asym-40/60 [2048,3072] |
| 15360x5120 | 1.187 | 1.203 | **+1.3%** | asym-60/40 [9216,6144] |
| 5120x15360 | 0.816 | 1.075 | **+31.7%** | uniform [2560,2560] |

Large rectangular problems show the highest gains (up to +49.2%) because the baseline kernel is poorly optimized for these shapes.

---

## Cubic Problems (M=N=K)

| Problem | Baseline (TF) | S17 (TF) | Gain | Winning Split |
|---------|--------------|----------|------|---------------|
| 10240^3 | 0.981 | 1.199 | **+22.2%** | asym-60/40 [6144,4096] |
| 12288^3 | 0.839 | 1.144 | **+36.4%** | pow2-2k [2048,10240] |
| 16384^3 | 0.857 | 1.218 | **+42.1%** | asym-40/60 [6528,9856] |

---

## Auto-Disabled Cases

| Problem | Baseline (TF) | Reason |
|---------|--------------|--------|
| 10240x10240x1024 | 0.883 | K < 4096 |
| 10240x10240x2048 | 1.007 | K < 4096 |
| 10240x10240x3072 | 1.062 | K < 4096 |
| 10240x8192x8192 | 0.931 | Both M,N < 10240 |
| 8192x10240x8192 | 0.929 | M < 10240 (M-split) |
| 10240x8192x4096 | 0.928 | K < 4096 |

---

## Key Patterns

**1. "Performance Valley" Dimensions: +20% to +65%**

Dimensions like 13824, 14848, 15360, 16384 have very low baseline efficiency (0.68-0.84 TF). The single-kernel heuristic selects a poor MacroTile. Splitting into sub-problems that avoid these valleys produces massive gains.

**2. 10240 with any K >= 4096: +8% to +37%**

10240 consistently benefits from [4096, 6144] or [6144, 4096]. Both dimensions get well-tuned kernels.

**3. Large Rectangular: +27% to +49%**

20480x10240, 10240x20480, 16384x8192, 8192x16384 have poor baselines and benefit greatly.

**4. Already-Efficient Baselines: -12% to +3%**

Dimensions where baseline exceeds ~1.15 TF (11520, 12800, 14080) see slight regressions because splitting adds overhead without improving kernel selection.

---

## Recommended Usage

```bash
# Enable multi-MacroTile with Origami search
./hipblaslt-bench -m $M -n $N -k $K \
  --precision f16_r --device 7 \
  --multi_macrotile --split_strategy 17 --num_splits 2 \
  --l2_cache_hints --api_method c -i 100 -j 100
```

**When to use**: M or N >= 10240 AND K >= 4096  
**Expected gain**: +22.9% average, up to +64.8%  
**Overhead**: ~20-60 ms for candidate search (~7 candidates x 3 iterations each)  
**Safety**: Problems below thresholds are auto-disabled (0% impact)

---

## Detailed Kernel Information

For each problem size: the baseline MacroTile selected by the heuristic, the Origami split, and the MacroTile selected for each sub-problem. This shows exactly *how* multi-MacroTile changes kernel selection.

#### 10240x10240x4096

**Baseline**: 1.094 TF, MacroTile: `MT256x240x64`

**Multi-MT (S17)**: 1.180 TF (+7.9%), Split: asym-40/60 [4096,6144]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 4096x10240x4096 | 0 | `MT256x208x64` |
| [1] | 6144x10240x4096 | 4096 | `MT256x208x64` |

Baseline uses MT256x240 but sub-problems get MT256x208 -- a smaller N-tile that runs faster for these M-sizes because it has better CU utilization (fewer tail workgroups).

---

#### 10240x10240x8192

**Baseline**: 0.967 TF, MacroTile: `MT256x240x64`

**Multi-MT (S17)**: 1.209 TF (+25.0%), Split: asym-60/40 [6144,4096]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 6144x10240x8192 | 0 | `MT256x208x64` |
| [1] | 4096x10240x8192 | 6144 | `MT256x208x64` |

---

#### 10240x10240x16384

**Baseline**: 0.920 TF, MacroTile: `MT256x240x64`

**Multi-MT (S17)**: 1.178 TF (+28.0%), Split: asym-40/60 [4096,6144]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 4096x10240x16384 | 0 | `MT256x208x64` |
| [1] | 6144x10240x16384 | 4096 | `MT256x208x64` |

---

#### 10240x10240x32768

**Baseline**: 0.859 TF, MacroTile: `MT256x240x64`

**Multi-MT (S17)**: 1.176 TF (+36.9%), Split: asym-40/60 [4096,6144]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 4096x10240x32768 | 0 | `MT256x208x64` |
| [1] | 6144x10240x32768 | 4096 | `MT256x208x64` |

---

#### 11264x11264x8192

**Baseline**: 1.250 TF, MacroTile: `MT256x256x64`

**Multi-MT (S17)**: 1.289 TF (+3.1%), Split: uniform [5632,5632]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 5632x11264x8192 | 0 | `MT256x256x64` |
| [1] | 5632x11264x8192 | 5632 | `MT256x256x64` |

Same MT for baseline and sub-problems. Gain comes from better WG distribution across CUs.

---

#### 11776x11776x8192

**Baseline**: 0.987 TF, MacroTile: `MT256x240x64`

**Multi-MT (S17)**: 1.197 TF (+21.3%), Split: asym-70/30 [8192,3584]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 8192x11776x8192 | 0 | `MT256x224x64` |
| [1] | 3584x11776x8192 | 8192 | `MT256x224x64` |

Splits get MT256x224 (different from baseline MT256x240). The 8192-wide sub-problem is a pow2 that maps perfectly to this tile.

---

#### 12288x12288x8192

**Baseline**: 0.890 TF, MacroTile: `MT256x256x64`

**Multi-MT (S17)**: 1.182 TF (+32.8%), Split: pow2-2k [2048,10240]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 2048x12288x8192 | 0 | `MT256x240x64` |
| [1] | 10240x12288x8192 | 2048 | `MT256x240x64` |

Baseline uses MT256x256 but sub-problems get MT256x240 which is significantly faster for these shapes.

---

#### 13824x13824x8192

**Baseline**: 0.791 TF, MacroTile: `MT256x256x64`

**Multi-MT (S17)**: 1.201 TF (+51.8%), Split: uniform [6912,6912]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 6912x13824x8192 | 0 | `MT256x256x64` |
| [1] | 6912x13824x8192 | 6912 | `MT256x256x64` |

Same MT -- the gain comes entirely from better WG distribution. 13824 is a "dead zone" dimension.

---

#### 14336x14336x8192

**Baseline**: 0.809 TF, MacroTile: `MT256x256x64`

**Multi-MT (S17)**: 1.135 TF (+40.3%), Split: pow2-2k [2048,12288]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 2048x14336x8192 | 0 | `MT256x224x64` |
| [1] | 12288x14336x8192 | 2048 | `MT256x224x64` |

Sub-problems get MT256x224 instead of baseline MT256x256. This different tile is better tuned for these sub-problem sizes.

---

#### 14848x14848x8192

**Baseline**: 0.799 TF, MacroTile: `MT256x256x64`

**Multi-MT (S17)**: 1.236 TF (+54.7%), Split: uniform [7424,7424]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 7424x14848x8192 | 0 | `MT256x256x64` |
| [1] | 7424x14848x8192 | 7424 | `MT256x256x64` |

Same MT, gain from WG distribution. Another "dead zone" dimension.

---

#### 15360x15360x8192

**Baseline**: 0.679 TF, MacroTile: `MT256x240x64`

**Multi-MT (S17)**: 1.119 TF (+64.8%), Split: asym-30/70 [4608,10752]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 4608x15360x8192 | 0 | `MT256x240x64` |
| [1] | 10752x15360x8192 | 4608 | `MT256x240x64` |

Same MT -- the massive +64.8% gain comes purely from CU utilization improvement. 15360 is the worst "dead zone" found (only 44% of peak).

---

#### 16384x16384x8192

**Baseline**: 0.839 TF, MacroTile: `MT256x256x64`

**Multi-MT (S17)**: 1.222 TF (+45.6%), Split: asym-30/70 [4864,11520]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 4864x16384x8192 | 0 | `MT256x256x64` |
| [1] | 11520x16384x8192 | 4864 | `MT256x256x64` |

---

#### 12288x6144x8192

**Baseline**: 1.041 TF, MacroTile: `MT256x240x64`

**Multi-MT (S17)**: 1.197 TF (+15.0%), Split: pow2-2k [2048,10240]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 2048x6144x8192 | 0 | `MT256x192x64` |
| [1] | 10240x6144x8192 | 2048 | `MT256x192x64` |

Sub-problems get MT256x192 (smaller N-tile), well-suited for the N=6144 dimension.

---

#### 6144x12288x8192

**Baseline**: 1.046 TF, MacroTile: `MT256x240x64`

**Multi-MT (S17)**: 1.160 TF (+10.9%), Split: asym-40/60 [2432,3712]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 2432x12288x8192 | 0 | `MT256x192x64` |
| [1] | 3712x12288x8192 | 2432 | `MT256x192x64` |

---

#### 20480x10240x8192

**Baseline**: 0.830 TF, MacroTile: `MT256x256x64`

**Multi-MT (S17)**: 1.238 TF (+49.2%), Split: asym-30/70 [6144,14336]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 6144x10240x8192 | 0 | `MT256x240x64` |
| [1] | 14336x10240x8192 | 6144 | `MT256x240x64` |

Sub-problems get MT256x240 instead of baseline MT256x256, which is faster for this N=10240 shape.

---

#### 12288x10240x8192

**Baseline**: 0.884 TF, MacroTile: `MT256x256x64`

**Multi-MT (S17)**: 1.291 TF (+46.0%), Split: uniform [6144,6144]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 6144x10240x8192 | 0 | `MT256x256x64` |
| [1] | 6144x10240x8192 | 6144 | `MT256x256x64` |

---

#### 10240x12288x8192

**Baseline**: 0.891 TF, MacroTile: `MT256x256x64`

**Multi-MT (S17)**: 1.260 TF (+41.4%), Split: uniform [5120,5120]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 5120x12288x8192 | 0 | `MT256x256x64` |
| [1] | 5120x12288x8192 | 5120 | `MT256x256x64` |

---

#### 16384x16384x16384

**Baseline**: 0.857 TF, MacroTile: `MT256x256x64`

**Multi-MT (S17)**: 1.218 TF (+42.1%), Split: asym-30/70 [4864,11520]

| Sub | Dims | M-offset | MacroTile |
|-----|------|----------|-----------|
| [0] | 4864x16384x16384 | 0 | `MT256x256x64` |
| [1] | 11520x16384x16384 | 4864 | `MT256x256x64` |

---

### MacroTile Change Pattern Summary

| Baseline MT | Sub-problem MT | Occurrence | Typical Gain |
|------------|----------------|------------|-------------|
| MT256x240x64 | **MT256x208x64** | 10240xNxK problems | +8% to +37% |
| MT256x256x64 | **MT256x256x64** (same) | 13K-16K square | +32% to +55% |
| MT256x256x64 | **MT256x224x64** | 11776, 14336 | +21% to +40% |
| MT256x256x64 | **MT256x240x64** | 12288, 20480x10240 | +33% to +49% |
| MT256x240x64 | **MT256x192x64** | rectangular (N=6144) | +11% to +15% |

**Key insight**: Multi-MacroTile gains come from two mechanisms:
1. **MT change** (10240-class): sub-problems get a different, more efficient MacroTile (MT256x208 instead of MT256x240)
2. **WG redistribution** (13K-16K "dead zones"): same MT but better workgroup distribution across CUs, avoiding tail effects
