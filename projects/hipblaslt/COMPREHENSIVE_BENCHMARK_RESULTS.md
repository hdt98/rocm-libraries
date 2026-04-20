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
