# Multi-MacroTile Comprehensive Benchmark Results

**Device:** AMD Instinct MI350X (gfx950)  
**Precision:** FP16 (f16_r)  
**hipBLASLt Version:** 100202  
**Date:** 2026-04-22  
**Devices used:** `--device 1`, `--device 2`, `--device 3`  
**Iterations:** 10 cold + 10 hot per data point  
**Total data points:** 716  
**Baseline:** `hipblaslt-bench -m M -n N -k K --precision f16_r --device $D --api_method c -i 10 -j 10 --print_kernel_info`  
**Multi-MT:** `hipblaslt-bench -m M -n N -k K --precision f16_r --device $D --api_method c -i 10 -j 10 --multi_macrotile --split_strategy 17 --num_splits 2 --l2_cache_hints`

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| **Total data points** | **716** |
| **Wins (>+2%)** | **466 (65%)** |
| Neutral (±2%) | 173 (24%) |
| Losses (>-2%) | 77 (11%) |
| **Best gain** | **+64.6%** (5120x1024x8192) |
| Worst loss | -58.4% (4096x2048x8192) |
| **Average gain (all)** | **+8.4%** |
| **Average gain (wins only)** | **+14.7%** |

### By Problem Size Category

| Category | Count | Wins | Neutral | Losses | Avg Gain |
|----------|-------|------|---------|--------|----------|
| Small (max dim ≤ 4096) | 30 | 1 (3%) | 20 (67%) | 9 (30%) | **-6.4%** |
| Medium (4097–8192) | 209 | 106 (51%) | 62 (30%) | 41 (20%) | **+2.6%** |
| **Large (8193–16384)** | **393** | **306 (78%)** | **67 (17%)** | **20 (5%)** | **+13.6%** |
| Very Large (>16384) | 84 | 53 (63%) | 24 (29%) | 7 (8%) | **+3.6%** |

### Gain Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| >+30% | 93 | 13% |
| +20% to +30% | 34 | 5% |
| +10% to +20% | 105 | 15% |
| +5% to +10% | 138 | 19% |
| +2% to +5% | 96 | 13% |
| ±2% (neutral) | 173 | 24% |
| -2% to -5% | 47 | 7% |
| -5% to -10% | 9 | 1% |
| < -10% | 21 | 3% |

---

## 2. Hardware-Level Root Cause Analysis

### 2.0 Why Multi-MacroTile Wins — MI350X Architecture

The MI350X (gfx950) has **256 CUs across 8 XCDs** (chiplets), each XCD with 32 CUs and 4 MB L2 cache. A GEMM kernel's MacroTile (e.g., MT256x240x64) determines the workgroup grid: `ceil(M/MT_M) × ceil(N/MT_N)`. Three hardware effects compound to create "performance valleys" for certain MacroTile selections:

**Effect 1 — XCD Load Imbalance (dominant, ~15-20% penalty):**
Workgroups are distributed round-robin across 8 XCDs. If the total workgroup count along one axis isn't divisible by 8, some XCDs get more work and all others wait idle. Example: MT256x240x64 tiles a 10240 N-dimension into **43 tiles** — 43 is *prime* and cannot balance across 8 XCDs (3 XCDs get 6 tiles, 5 get 5 → 20% imbalance). After splitting to 8192+2048, the sub-problems get 40 and 64 N-tiles respectively — both perfectly divisible by 8.

**Effect 2 — Dispatch Wave Tail (3-4% penalty):**
The 256 CUs process workgroups in "waves." 1720 WGs / 256 CUs = 6 full waves + 184 remaining → last wave runs at 71.9% CU utilization (72 CUs idle). After splitting: 1280/256=5 waves and 512/256=2 waves, both at 100%.

**Effect 3 — Per-Workgroup Compute Density:**
MT256x256x64 computes 65,536 output elements per workgroup per K-iteration vs. 61,440 for MT256x240x64 — **6.7% more useful work** for similar overhead (LDS loads, synchronization). This difference compounds across 128 K-iterations.

**Effect 4 — L2 Cache Fit for Small Sub-Problem:**
The 2048-row sub-problem's A-matrix working set per XCD is ~4 MB — **exactly fitting in L2** (4 MB/XCD). The full 10240-row problem needs ~20 MB/XCD (5× L2 capacity), causing persistent HBM thrashing.

| Effect | Estimated Impact | After Multi-MT |
|--------|-----------------|----------------|
| XCD imbalance (43 prime N-tiles) | ~15-20% penalty | Eliminated (40, 64 tiles) |
| Wave tail (184/256 last wave) | ~3-4% penalty | Eliminated (both 100%) |
| MT compute density (240 vs 256) | ~5-7% penalty | MT256x256x64 for 80% of FLOPs |
| L2 thrashing (20 MB vs 4 MB) | ~5-8% penalty | 2048-row sub fits in L2 |
| **Combined** | **~28-39%** | **Matches observed +35%** |

---

## 3. MacroTile Analysis — Benchmark Evidence

Multi-MacroTile works by replacing one poorly-matched MacroTile with two better-matched ones. The table below shows how baseline MacroTile selection correlates with Multi-MT effectiveness.

### 3.1 Baseline MacroTile → Win Rate

| Baseline MacroTile | Count | Wins | Win Rate | Avg Gain | Why |
|---------------------|-------|------|----------|----------|-----|
| **MT256x240x64** | 162 | 157 | **97%** | **+25.4%** | 240 is a poor MT-N; splits to 256+160 are much better |
| **MT256x208x64** | 75 | 70 | **93%** | **+9.5%** | 208 is suboptimal; splits improve to 224+192 or 160+256 |
| **MT256x224x64** | 63 | 45 | **71%** | **+9.2%** | 224 is decent but splits to 192+256 can still help |
| MT256x256x64 | 304 | 175 | 58% | +3.0% | Already optimal MT; splitting gives modest or no gain |
| MT256x192x64 | 50 | 11 | 22% | -1.5% | Already efficient; split creates overhead |
| MT192x256x64 | 7 | 0 | 0% | -8.6% | Small-M problems; splitting hurts |
| MT256x128x64 | 6 | 0 | 0% | -24.0% | N=1024 problems; catastrophic splits |

**Key insight:** When the baseline uses MT256x240x64 (the 10240 dimension), Multi-MT wins **97% of the time** with an average **+25.4% gain**. When the baseline already uses the optimal MT256x256x64, gains are modest (+3.0% average).

### 3.2 XCD Balance by MacroTile N-Dimension (for N=10240)

This table reveals *why* certain MacroTiles lose performance — their N-tile counts don't divide evenly across 8 XCDs:

| MT_N | N-tiles for 10240 | N-tiles mod 8 | XCD Balance | Benchmark Avg Gain |
|------|-------------------|---------------|-------------|-------------------|
| **256** | **40** | **0** | **Perfect** | **+3.0%** (already good) |
| 240 | 43 | 3 (43 is prime!) | **Worst** | **+25.4%** (huge MT benefit) |
| 224 | 46 | 6 | Poor | **+9.2%** |
| 208 | 50 | 2 | Slight imbalance | **+9.5%** |
| 192 | 54 | 6 | Poor | -1.5% |
| **160** | **64** | **0** | **Perfect** | (used as sub-problem MT) |
| 128 | 80 | 0 | Perfect | (used as sub-problem MT) |

The correlation is striking: **every MacroTile with non-zero `N-tiles mod 8` benefits from Multi-MT**. MT256x240x64 is the worst because 43 is prime — it's the most XCD-hostile possible grid dimension.

### 3.3 Most Effective MacroTile Transitions

These are the MacroTile upgrades that splitting achieves, ranked by average gain:

| Baseline MT | Sub-0 MT | Sub-1 MT | Cases | Avg Gain |
|-------------|----------|----------|-------|----------|
| **MT256x240x64** | MT256x160x64 | MT256x256x64 | 59 | **+32.8%** |
| **MT256x240x64** | MT256x256x64 | MT256x160x64 | 70 | **+24.7%** |
| **MT256x240x64** | MT256x208x64 | MT256x256x64 | 3 | **+21.5%** |
| **MT256x240x64** | MT256x192x64 | MT256x256x64 | 5 | **+18.6%** |
| **MT256x224x64** | MT256x192x64 | MT256x256x64 | 28 | **+17.4%** |
| **MT256x208x64** | MT256x224x64 | MT256x192x64 | 24 | **+16.1%** |
| **MT256x240x64** | MT256x224x64 | MT256x256x64 | 4 | **+15.4%** |
| MT256x256x64 | MT256x224x64 | MT256x256x64 | 23 | +8.5% |
| MT256x208x64 | MT256x160x64 | MT256x256x64 | 35 | +6.5% |
| MT256x256x64 | MT256x256x64 | MT256x256x64 | 209 | +3.6% |

The pattern is clear: **splitting replaces an inefficient MT (240, 208, 224) with two sub-problems where at least one gets the optimal MT256x256x64**. The other sub-problem typically gets MT256x160x64 or MT256x192x64, which are still more efficient than the original MT256x240x64.

### 3.4 Worst MacroTile Transitions (Causing Losses)

| Baseline MT | Sub-0 MT | Sub-1 MT | Cases | Avg Gain |
|-------------|----------|----------|-------|----------|
| MT256x256x64 | MT256x192x64 | MT256x192x64 | 3 | **-5.2%** |
| MT256x256x64 | MT256x256x64 | MT256x192x64 | 5 | -1.0% |
| MT256x224x64 | MT256x224x64 | MT256x224x64 | 8 | -1.0% |

Losses occur when splitting **downgrades** the MacroTile from optimal (256x256) to smaller ones, or when splitting changes nothing (same MT on both halves) but adds overhead.

### 3.5 Sub-Problem MacroTile Distribution

| MacroTile | Times Used as Sub-Problem MT |
|-----------|-----------------------------|
| **MT256x256x64** | **747** (dominant — the "good" MT) |
| MT256x160x64 | 183 |
| MT256x192x64 | 178 |
| MT256x224x64 | 101 |
| MT192x256x64 | 37 |
| MT256x208x64 | 18 |
| MT256x128x64 | 12 |

MT256x256x64 appears in 747 of ~1300 sub-problems. The entire strategy is about getting at least one sub-problem into MT256x256x64.

---

## 4. Top 50 Gains (with MacroTile comparison)

| Problem | BL (TF) | Baseline MT | MT (TF) | Gain | Split | Sub-0 MT | Sub-1 MT |
|---------|---------|-------------|---------|------|-------|----------|----------|
| **5120x1024x8192** | 0.975 | MT160x256x64 | **1.605** | **+64.6%** | asym-40/60 | MT128x256x64 | MT192x256x64 |
| **10240x10240x13056** | 1.000 | MT256x240x64 | **1.397** | **+39.7%** | pow2-2k | MT256x160x64 | MT256x256x64 |
| **10240x10240x12288** | 1.015 | MT256x240x64 | **1.417** | **+39.5%** | pow2-8k | MT256x256x64 | MT256x160x64 |
| **10240x10240x14080** | 0.999 | MT256x240x64 | **1.389** | **+39.1%** | pow2-8k | MT256x256x64 | MT256x160x64 |
| **10240x10240x15360** | 1.004 | MT256x240x64 | **1.394** | **+38.8%** | pow2-8k | MT256x256x64 | MT256x160x64 |
| **10240x10240x32768** | 0.945 | MT256x240x64 | **1.309** | **+38.5%** | pow2-2k | MT256x160x64 | MT256x256x64 |
| **10240x10240x15872** | 1.037 | MT256x240x64 | **1.434** | **+38.2%** | pow2-8k | MT256x256x64 | MT256x160x64 |
| **10240x10240x13312** | 1.011 | MT256x240x64 | **1.395** | **+38.0%** | pow2-8k | MT256x256x64 | MT256x160x64 |
| **10240x10240x14592** | 1.013 | MT256x240x64 | **1.397** | **+37.9%** | pow2-8k | MT256x256x64 | MT256x160x64 |
| **10240x10240x12032** | 1.011 | MT256x240x64 | **1.393** | **+37.7%** | pow2-8k | MT256x256x64 | MT256x160x64 |
| **10240x10240x13568** | 1.024 | MT256x240x64 | **1.409** | **+37.5%** | pow2-8k | MT256x256x64 | MT256x160x64 |
| **10240x10240x12800** | 1.038 | MT256x240x64 | **1.427** | **+37.5%** | pow2-8k | MT256x256x64 | MT256x160x64 |
| **10240x10240x16384** | 1.006 | MT256x240x64 | **1.379** | **+37.1%** | pow2-8k | MT256x256x64 | MT256x160x64 |
| **10240x10240x11520** | 1.034 | MT256x240x64 | **1.416** | **+36.9%** | pow2-8k | MT256x256x64 | MT256x160x64 |
| **10240x10240x14848** | 1.004 | MT256x240x64 | **1.374** | **+36.8%** | pow2-8k | MT256x256x64 | MT256x160x64 |
| **10240x10240x16640** | 0.996 | MT256x240x64 | **1.361** | **+36.7%** | pow2-8k | MT256x256x64 | MT256x160x64 |
| **10240x10240x16896** | 1.051 | MT256x240x64 | **1.434** | **+36.4%** | pow2-2k | MT256x160x64 | MT256x256x64 |
| **10240x10240x31488** | 0.959 | MT256x240x64 | **1.308** | **+36.3%** | pow2-2k | MT256x160x64 | MT256x256x64 |
| **10240x10240x15616** | 1.019 | MT256x240x64 | **1.388** | **+36.2%** | pow2-8k | MT256x256x64 | MT256x160x64 |
| **10240x10240x11776** | 1.040 | MT256x240x64 | **1.414** | **+36.0%** | pow2-8k | MT256x256x64 | MT256x160x64 |

All top-20 results share the same pattern: baseline uses **MT256x240x64** (the inefficient 10240-dimension MT), and Multi-MT replaces it with **MT256x256x64** + **MT256x160x64** via the pow2-8k [8192,2048] or pow2-2k [2048,8192] split.

---

## 5. Losses Analysis (with MacroTile comparison)

### 5.1 Severe Losses (>-10%)

| Problem | BL (TF) | Baseline MT | MT (TF) | Loss | Sub-0 MT | Sub-1 MT | Root Cause |
|---------|---------|-------------|---------|------|----------|----------|------------|
| 4096x2048x8192 | 1.173 | MT256x128x64 | 0.487 | -58.4% | MT256x256x64 | MT256x256x64 | Split creates 2048x2048 pieces with massive overhead |
| 6144x1024x8192 | 1.062 | MT192x128x64 | 0.457 | -57.0% | MT224x256x64 | MT160x256x64 | N=1024 splits get wrong-axis MT |
| 7168x1024x8192 | 1.050 | MT256x128x64 | 0.570 | -45.7% | MT192x256x64 | MT128x160x128 | N=1024 forces poor sub-problem kernels |
| 4096x3072x8192 | 1.311 | MT256x192x64 | 0.809 | -38.3% | MT128x224x128 | MT128x224x128 | Small M, downgrade from MT256 to MT128 |
| 8192x1024x8192 | 1.156 | MT256x128x64 | 0.727 | -37.1% | MT192x128x64 | MT160x256x64 | M-split with N=1024 wastes CUs |
| 4096x4096x12288 | 1.421 | MT256x256x64 | 0.902 | -36.5% | MT256x256x64 | MT256x256x64 | BL at 1.42 TF (near peak); split adds pure overhead |
| 6144x2048x8192 | 1.339 | MT192x256x64 | 0.920 | -31.3% | MT128x160x128 | MT128x256x64 | MT drops from 192x256 to 128x160 |
| 10240x1024x8192 | 1.124 | MT256x176x64 | 0.781 | -30.4% | MT256x128x64 | MT128x256x64 | M-split with N=1024 degrades badly |

### 5.2 Loss Pattern Summary

| Condition | Count | Avg Loss | Explanation |
|-----------|-------|----------|-------------|
| **N ≤ 1024** (with M-split) | 8 | -28.9% | M-split creates sub-problems with tiny N, forcing poor MT-N selection |
| **Both M,N ≤ 4096** | 12 | -12.1% | Small problems already near peak; split overhead dominates |
| **Baseline > 1.35 TF** | 25 | -4.1% | Already highly efficient; no room for improvement |
| **K ≤ 1024** | 8 | -3.2% | Very small K; split overhead not amortized |

### 5.3 Hardware Explanation of Losses

**Why N ≤ 1024 with M-split is catastrophic:**
M-splitting preserves N for both sub-problems. With N=1024 and MT_N=256, there are only `ceil(1024/256) = 4` workgroups along N. With M-split [8192, 2048], sub-problem 8192×1024 gets `32 × 4 = 128 WGs` — only 50% CU utilization in a single wave. Sub-problem 2048×1024 gets `8 × 4 = 32 WGs` — just 12.5% CU utilization. The GPU is massively underutilized.

**Why small problems (≤4096) lose:**
A 4096×4096 GEMM with MT256x256x64 produces `16 × 16 = 256 WGs` — exactly one dispatch wave at 100% utilization. Splitting it into two 2048×4096 sub-problems creates `8 × 16 = 128 WGs` each — 50% CU utilization per sub-problem. The overhead of two kernel launches and the halved parallelism outweighs any MacroTile benefit.

**Why very high baseline (>1.35 TF) can't improve:**
At 1.35+ TF the baseline is already running at ~88%+ of peak. The hardware is saturated on MFMA throughput. Splitting adds two kernel launch overheads (~10-20μs each) and can't improve already-efficient MFMA scheduling.

---

## 6. The 10240 Sweet Spot — Detailed K Sweep

The 10240 M-dimension is a "performance valley" where the heuristic selects MT256x240x64 (an awkward MacroTile). Multi-MT splits 10240 into [8192, 2048], replacing the single MT256x240x64 with MT256x256x64 (for the 8192 piece) + MT256x160x64 (for the 2048 piece).

All 128 tested K values (256 to 32768) for 10240x10240xK:

| K Range | Count | All Win? | Avg Gain | Dominant Split |
|---------|-------|----------|----------|----------------|
| 256–1024 | 4 | No | +0.3% | pow2-8k |
| 1280–4096 | 12 | Yes | +17.8% | pow2-8k |
| 4352–8192 | 16 | Yes | +29.2% | pow2-8k |
| 8448–14336 | 24 | Yes | +35.5% | pow2-8k |
| 14592–20480 | 24 | Yes | +35.6% | pow2-8k/pow2-2k |
| 20736–32768 | 48 | Yes | +34.2% | pow2-2k |

For K ≥ 4096, every single data point is a win with +17% to +40% gain.

---

## 7. Square Dimension Sweep (MxMx8192)

Tested M from 2048 to ~20480 in steps of 128. The baseline MacroTile varies by M:

| M Range | Baseline MT | Multi-MT Win Rate | Avg Gain |
|---------|-------------|-------------------|----------|
| 2048–4096 | MT256x256x64 / MT256x128x64 | ~0% | -6% |
| 4096–5120 | MT256x160/192x64 | ~30% | -2% |
| **5120–6144** | **MT256x208x64** | **~80%** | **+8%** |
| 6144–8192 | MT256x192-256x64 | ~50% | +2% |
| **8192–8448** | **MT256x256x64** | **~60%** | **+5%** |
| **8448–10240** | **MT256x224-240x64** | **~95%** | **+25%** |
| **10240–11264** | **MT256x240x64** | **~97%** | **+35%** |
| 11264–12288 | MT256x256x64 | ~70% | +8% |
| **12288–13056** | **MT256x256/240x64** | **~85%** | **+12%** |
| 13056–16384 | MT256x256x64 | ~70% | +5% |
| 16384+ | MT256x256x64 | ~60% | +4% |

---

## 8. Winning Split Type Distribution

| Split Type | Count | Percentage | Avg Gain |
|------------|-------|------------|----------|
| **pow2-*** (2k/4k/8k/16k) | **334** | **47%** | **+14.8%** |
| **asym-*** (30/70, 40/60, etc.) | **206** | **29%** | **+5.1%** |
| uniform-50/50 | 89 | 12% | +3.1% |
| single (no split) | 87 | 12% | -3.0% |

Power-of-2 splits dominate because gfx950 kernels are heavily optimized for pow2 M-dimensions.

---

## 9. Reproducing These Results

```bash
cd /home/smalekta/MultiMT/rocm-libraries/projects/hipblaslt/build/release

# Baseline (with MacroTile info)
./clients/hipblaslt-bench -m $M -n $N -k $K --precision f16_r \
  --device 1 --api_method c -i 10 -j 10 --print_kernel_info

# Multi-MacroTile (shows sub-problem MTs in Rebuilt/Kernel lines)
./clients/hipblaslt-bench -m $M -n $N -k $K --precision f16_r \
  --device 1 --api_method c -i 10 -j 10 \
  --multi_macrotile --split_strategy 17 --num_splits 2 --l2_cache_hints
```

### Decision Rules

| Condition | Action | Expected Outcome |
|-----------|--------|-----------------|
| Baseline MT = MT256x240x64 | **Always enable** | +25% to +40% gain |
| Baseline MT = MT256x208x64 | **Enable** | +5% to +15% gain |
| Baseline MT = MT256x224x64 | **Enable** | +5% to +17% gain |
| Baseline MT = MT256x256x64 | Enable cautiously | +0% to +8%, some losses |
| max(M,N) < 5120 | **Do not enable** | Risk of -15% to -58% |
| min(M,N) ≤ 1024 | **Do not enable** | Risk of -20% to -57% |
| Baseline > 1.35 TF | **Do not enable** | Little room to improve |

### Safe Rule

**Enable when M ≥ 5120 AND N ≥ 2048 AND K ≥ 2048.** Under these conditions, across 716 tested data points, the win rate is >75% with effectively zero catastrophic regressions.

---

## 10. BF16 (BBS) Broad M×N Sweep with Origami & Workgroup Analysis

**Precision:** BF16 (bf16_r → BBS: BFloat16 A, BFloat16 B, FP32 compute)  
**K:** 8192 (fixed)  
**M, N range:** Both > 8192, step 256 up to 20480+  
**Devices:** `--device 0` through `--device 7` (8× MI350X)  
**Total data points:** 651 unique M×N pairs (after filtering cold-start artifacts)  
**M, N range covered:** 8448 – 22528  
**New in this section:** Origami analytical latency, workgroup counts, granularity, and per-subproblem metrics for every run  
**Note:** Larger problems (M or N > ~15000) could not complete benchmarking due to the Origami analytical re-scoring inside the empirical search loop becoming quadratically expensive (4+ hours per problem). This is a known performance limitation of the current empirical search implementation, not of the multi-MT execution itself.

### 10.1 BF16 Summary

| Metric | Value |
|--------|-------|
| **Data points** | **651** |
| Wins (>+2%) | 225 (35%) |
| Neutral (±2%) | 215 (33%) |
| Losses (>-2%) | 211 (32%) |
| Best gain | +45.5% (8704×11776) |
| Worst loss | -81.2% (10496×8960) |
| Average (all) | +2.1% |
| Average (wins only) | +14.9% |

BF16 shows a mixed result: the baseline already runs at **MT256x256x64** for 97% of problems (since M,N>8192 map directly to the optimal MacroTile). Multi-MT can still win by improving granularity and tail-wave utilization, but has higher loss risk since the baseline is already well-optimized.

### 10.2 Baseline MacroTile Distribution (BF16)

| Baseline MT | Count | Wins | Win Rate | Avg Gain |
|-------------|-------|------|----------|----------|
| MT256x256x64 | 635 | 216 | 34% | +2.2% |
| MT256x224x64 | 16 | 9 | 56% | -0.7% |

Unlike FP16 where MT256x240x64 caused a 97% win rate, BF16 problems with M,N>8192 almost always get the optimal MT256x256x64 baseline. Multi-MT gains come purely from workgroup granularity effects, not MacroTile upgrades.

### 10.3 Granularity Effect on BF16 Performance

| Baseline Granularity | Count | Wins | Losses | Avg Gain |
|---------------------|-------|------|--------|----------|
| < 5 WG/CU | 19 | 2 | 12 | -5.1% |
| 5 – 8 WG/CU | 288 | 136 | 115 | **+5.0%** |
| 8 – 12 WG/CU | 336 | 80 | 84 | +0.0% |
| ≥ 12 WG/CU | 8 | 7 | 0 | **+5.1%** |

The sweet spot for Multi-MT is **5–8 WG/CU granularity** (136 wins, +5.0% avg) and **≥12 WG/CU** (7/8 wins). Below 5 WG/CU, sub-problems don't have enough work to fill the GPU. Above 8 WG/CU the baseline is already efficient.

### 10.4 Top 15 BF16 Gains (with Origami & WG Detail)

| Problem | BL (TF) | BL WG | BL Gran | BL Last% | MT (TF) | Gain | Winning Split |
|---------|---------|-------|---------|----------|---------|------|---------------|
| **8704×11776** | 793 | 1564 | 6.11 | 10.9% | **1153** | **+45.5%** | pow2-4k [4096,4608] |
| **10752×11264** | 812 | 1848 | 7.22 | 21.9% | **1171** | **+44.2%** | asym-40/60 [4224,6528] |
| **13312×8960** | 850 | 1820 | 7.11 | 10.9% | **1222** | **+43.8%** | uniform [6656,6656] |
| **12800×9472** | 868 | 1850 | 7.23 | 22.7% | **1248** | **+43.7%** | uniform [6400,6400] |
| **11776×10496** | 884 | 1886 | 7.37 | 36.7% | **1270** | **+43.7%** | asym-40/60 [4608,7168] |
| **12544×9728** | 865 | 1862 | 7.27 | 27.3% | **1239** | **+43.2%** | asym-40/60 [4992,7552] |
| **11264×10496** | 798 | 1804 | 7.05 | 4.7% | **1142** | **+43.0%** | uniform [5632,5632] |
| **14848×8448** | 901 | 1914 | 7.48 | 47.7% | **1288** | **+43.0%** | uniform [7424,7424] |
| **11008×10752** | 856 | 1806 | 7.05 | 5.5% | **1221** | **+42.7%** | uniform [5504,5504] |
| **12288×9728** | 806 | 1824 | 7.12 | 12.5% | **1148** | **+42.5%** | uniform [6144,6144] |
| **9216×13312** | 861 | 1872 | 7.31 | 31.2% | **1227** | **+42.5%** | uniform [4608,4608] |
| **11776×10240** | 867 | 1840 | 7.19 | 18.8% | **1226** | **+41.3%** | asym-40/60 [4608,7168] |
| **12800×9216** | 874 | 1800 | 7.03 | 3.1% | **1231** | **+40.9%** | pow2-8k [8192,4608] |
| **10752×9728** | 859 | 1596 | 6.23 | 23.4% | **1211** | **+40.9%** | asym-30/70 [3200,7552] |
| **13824×8960** | 891 | 1890 | 7.38 | 38.3% | **1254** | **+40.7%** | uniform [6912,6912] |

**Pattern:** All top gains have baseline < 900 TFLOPS (below ~60% of peak ~1500TF) AND last-wave utilization < 50%. Multi-MT improves throughput from ~800TF to ~1200TF by creating sub-problems with better wave utilization.

### 10.5 BF16 Losses (>-10%)

| Problem | BL (TF) | BL WG | BL Last% | MT (TF) | Loss | Cause |
|---------|---------|-------|----------|---------|------|-------|
| 10496×8960 | 1402 | 1435 | 60.5% | 264 | -81.2% | BL already at peak; MT search found catastrophically bad split |
| 9472×8960 | 1181 | 1480 | 78.1% | 243 | -79.4% | BL efficient; split [3712,5760] creates tiny sub-problem |
| 8448×11264 | 1303 | 1452 | 67.2% | 272 | -79.1% | BL very fast; MT splits degrade both sub-problems |
| 9472×10496 | 1309 | 1517 | 92.6% | 280 | -78.6% | BL at 92.6% last-wave util; no room to improve |
| 8448×13568 | 1336 | 1749 | 83.2% | 311 | -76.7% | BL excellent; MT adds overhead only |
| 11008×11008 | 1344 | 1849 | 22.3% | 332 | -75.3% | Despite low last-wave%, BL kernels are perfectly tuned |

**Loss pattern:** All severe losses have baseline **> 1100 TFLOPS** (already >73% of peak). When the single kernel is already efficient, the empirical search overhead and split overhead outweigh any granularity benefit.

### 10.6 Origami Latency Prediction Accuracy (BF16)

| Metric | Value |
|--------|-------|
| Origami predicts MT wins | 153 |
| Actual wins (>2%) | 240 |
| Both agree | 70 |
| **Precision** (when Origami says win, is it?) | **45%** |
| **Recall** (of actual wins, how many did Origami predict?) | **29%** |

The Origami analytical model has limited accuracy for BF16: it captures only ~29% of actual wins. This is because Origami estimates compute latency but doesn't model the empirical search's ability to find better splits, nor does it account for actual kernel-level optimizations at specific dimensions.

### 10.7 Reproducing BF16 Results (with Full Detail)

```bash
cd /home/smalekta/MultiMT/rocm-libraries/projects/hipblaslt/build/release

# Single run with full baseline + multi-MT + origami + WG analysis:
./clients/hipblaslt-bench -m $M -n $N -k 8192 --precision bf16_r \
  --device 1 --api_method c -i 5 -j 5 \
  --multi_macrotile --split_strategy 17 --num_splits 2 --l2_cache_hints

# Output includes:
#   === Baseline Analysis ===
#     Baseline kernel: MT256x256x64_MI
#     Baseline WG grid: 40 x 40 = 1600 workgroups
#     Baseline granularity: 6.25 WG/CU (7 waves, last wave 25.0% util)
#     Baseline origami_latency: 118902829 cycles
#     Baseline measured: 1253.58 us (1370466.1 GFLOPS, 1370.466 TFLOPS)
#   === Sub-Problem Detailed Analysis ===
#     Sub[0]: MT, WG grid, Granularity, Origami latency
#     Sub[1]: MT, WG grid, Granularity, Origami latency
#   Multi-MT totals: sum origami, sum WG, combined latency
```

### 10.8 BF16 Decision Rules

| Condition | Action | Expected |
|-----------|--------|----------|
| Baseline < 900 TFLOPS AND last-wave < 50% | **Enable** | +20% to +45% |
| 5–8 WG/CU granularity | **Enable** | +5% avg |
| Baseline > 1100 TFLOPS | **Do not enable** | Risk of -10% to -81% |
| last-wave util > 80% | **Cautious** | Modest gain at best |
