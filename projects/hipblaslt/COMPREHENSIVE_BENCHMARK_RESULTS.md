# Multi-MacroTile Comprehensive Benchmark Results

**Device:** AMD Instinct MI350X (gfx950)  
**Precision:** FP16 (f16_r / HHS) and BF16 (bf16_r / BBS)  
**hipBLASLt Version:** 100202  
**Date:** 2026-04-23  
**Devices used:** `--device 0` through `--device 7` (8× MI350X)  
**Methodology:** Adaptive iterations — each problem runs ≥ 3 seconds cold + ≥ 3 seconds hot (iterations computed from `ceil(3,000,000 / estimated_us_per_iter)`)  
**Total data points collected so far:** 787 (394 FP16 + 393 BF16), of 2835 planned (benchmarks continuing)  
**Baseline:** `hipblaslt-bench -m M -n N -k K --precision $P --device $D --api_method c -i $ITERS -j $ITERS`  
**Multi-MT:** `hipblaslt-bench -m M -n N -k K --precision $P --device $D --api_method c -i $ITERS -j $ITERS --multi_macrotile --split_strategy 17 --num_splits 2 --l2_cache_hints`

---

## 1. Executive Summary

### Combined (FP16 + BF16)

| Metric | Value |
|--------|-------|
| **Total data points** | **787** |
| **Wins (>+2%)** | **266 (34%)** |
| Neutral (±2%) | 346 (44%) |
| Losses (>-2%) | 175 (22%) |
| **Best gain** | **+25.6%** (10240×10240×11264, FP16) |
| Worst loss | -51.8% (6144×1024×8192, FP16) |
| **Average gain (all)** | **+1.5%** |
| **Average gain (wins only)** | **+8.7%** |

### By Precision

| Precision | Points | Wins | Neutral | Losses | Avg Gain | Avg Win | Best | Worst |
|-----------|--------|------|---------|--------|----------|---------|------|-------|
| **FP16 (HHS)** | 394 | 152 (38%) | 177 (45%) | 65 (17%) | **+2.5%** | **+11.2%** | +25.6% | -51.8% |
| **BF16 (BBS)** | 393 | 114 (29%) | 169 (43%) | 110 (28%) | **+0.6%** | **+5.4%** | +24.9% | -7.0% |

---

## 2. Hardware-Level Root Cause Analysis

### 2.1 Why Multi-MacroTile Wins — MI350X Architecture

The MI350X (gfx950) has **256 CUs across 8 XCDs** (chiplets), each XCD with 32 CUs and 4 MB L2 cache. A GEMM kernel's MacroTile (e.g., MT256x240x64) determines the workgroup grid: `ceil(M/MT_M) × ceil(N/MT_N)`. Three hardware effects compound to create "performance valleys":

**Effect 1 — XCD Load Imbalance (dominant, ~15-20% penalty):**
Workgroups distribute round-robin across 8 XCDs. If workgroup count along one axis isn't divisible by 8, some XCDs get more work. Example: MT256x240x64 tiles 10240 into **43 N-tiles** — 43 is *prime*, creating 20% XCD imbalance. After splitting to [8192,2048], sub-problems get 40 and 64 N-tiles — both perfectly divisible by 8.

**Effect 2 — Dispatch Wave Tail (3-4% penalty):**
1720 WGs / 256 CUs = 6 full waves + 184 remaining → last wave at 71.9% CU utilization. After splitting: 1280/256=5 and 512/256=2 waves, both 100%.

**Effect 3 — Per-Workgroup Compute Density:**
MT256x256x64 computes 6.7% more FLOPs per K-iteration than MT256x240x64.

**Effect 4 — L2 Cache Fit:**
The 2048-row sub-problem's A-matrix working set per XCD is ~4 MB — exactly fits in L2 (4 MB/XCD). The 10240-row problem needs ~20 MB/XCD.

| Effect | Estimated Impact | After Multi-MT |
|--------|-----------------|----------------|
| XCD imbalance | ~15-20% penalty | Eliminated |
| Wave tail | ~3-4% penalty | Eliminated |
| MT compute density | ~5-7% penalty | MT256x256x64 for 80% of FLOPs |
| L2 thrashing | ~5-8% penalty | Sub fits in L2 |
| **Combined** | **~28-39%** | **Matches observed +25%** |

### 2.2 XCD Balance by MacroTile N-Dimension (for N=10240)

| MT_N | N-tiles for 10240 | mod 8 | XCD Balance | Avg Gain |
|------|-------------------|-------|-------------|----------|
| **256** | **40** | **0** | **Perfect** | -1.5% (already optimal) |
| 240 | 43 | 3 (prime!) | **Worst** | **+13.0%** |
| 224 | 46 | 6 | Poor | +2.9% |
| 208 | 50 | 2 | Slight | +7.6% |
| 192 | 54 | 6 | Poor | -1.1% |
| **160** | **64** | **0** | **Perfect** | (sub-problem MT) |

---

## 3. FP16 (HHS) Results — Baseline MacroTile Analysis

### 3.1 FP16 Baseline MacroTile → Win Rate

| Baseline MacroTile | Count | Wins | Win Rate | Avg Gain |
|---------------------|-------|------|----------|----------|
| **MT256x240x64** | 70 | 60 | **86%** | **+13.0%** |
| **MT256x208x64** | 55 | 46 | **84%** | **+7.6%** |
| MT256x224x64 | 42 | 17 | 40% | +2.9% |
| MT256x256x64 | 124 | 20 | 16% | -1.5% |
| MT256x192x64 | 44 | 1 | 2% | -1.1% |
| MT192x256x64 | 8 | 0 | 0% | -8.5% |
| MT256x176x64 | 7 | 3 | 43% | -1.4% |
| MT256x160x64 | 6 | 0 | 0% | -2.2% |

**Key insight:** Multi-MT wins **86% of the time when baseline uses MT256x240x64** (avg +13.0%), and **84% with MT256x208x64** (avg +7.6%). When baseline already uses optimal MT256x256x64, win rate drops to 16%.

### 3.2 Top 10 FP16 Gains

| Problem | BL (TF) | Baseline MT | MT (TF) | Gain | Winner |
|---------|---------|-------------|---------|------|--------|
| **10240×10240×11264** | 919 | MT256x240x64 | **1154** | **+25.6%** | pow2-8k [8192,2048] |
| **10240×10240×12544** | 916 | MT256x240x64 | **1150** | **+25.6%** | pow2-8k [8192,2048] |
| **10240×10240×12032** | 909 | MT256x240x64 | **1140** | **+25.5%** | pow2-8k [8192,2048] |
| **10240×10240×9472** | 888 | MT256x240x64 | **1113** | **+25.3%** | pow2-8k [8192,2048] |
| **10240×10240×11008** | 918 | MT256x240x64 | **1151** | **+25.3%** | pow2-8k [8192,2048] |
| **10240×10240×12288** | 912 | MT256x240x64 | **1142** | **+25.3%** | pow2-8k [8192,2048] |
| **10240×10240×9984** | 887 | MT256x240x64 | **1107** | **+24.9%** | pow2-8k [8192,2048] |
| **10240×10240×9216** | 928 | MT256x240x64 | **1158** | **+24.8%** | pow2-8k [8192,2048] |
| **10240×10240×9728** | 898 | MT256x240x64 | **1120** | **+24.6%** | pow2-8k [8192,2048] |
| **10240×10240×8704** | 925 | MT256x240x64 | **1151** | **+24.5%** | pow2-8k [8192,2048] |

All top gains are 10240×10240 problems with MT256x240x64 baseline, split to [8192,2048] → MT256x256x64 + MT256x160x64.

### 3.3 FP16 Losses (worst 10)

| Problem | BL (TF) | MT (TF) | Loss | Root Cause |
|---------|---------|---------|------|------------|
| 6144×1024×8192 | 937 | 452 | -51.8% | N=1024 with M-split → sub-problems severely underutilize CUs |
| 4096×2048×8192 | 1004 | 514 | -48.8% | Both dims ≤ 4096; split halves parallelism |
| 7168×1024×8192 | 949 | 629 | -33.8% | N=1024 forces poor sub-problem kernels |
| 4096×3072×8192 | 1123 | 751 | -33.1% | Small M; MT drops from 256 to 128 |
| 8192×1024×8192 | 997 | 706 | -29.2% | N=1024 M-split wastes CUs |
| 12288×1024×8192 | 848 | 604 | -28.8% | N=1024 |
| 6144×2048×8192 | 1010 | 729 | -27.9% | Small N=2048 |
| 5120×3072×8192 | 787 | 602 | -23.5% | Both dims small |
| 4096×1024×8192 | 771 | 590 | -23.4% | Both dims ≤ 4096 |
| 4096×4096×12288 | 1068 | 821 | -23.1% | Baseline already efficient |

**Pattern:** All severe FP16 losses have N ≤ 1024 or both M,N ≤ 4096.

---

## 4. BF16 (BBS) Results

### 4.1 BF16 Summary

| Metric | Value |
|--------|-------|
| Data points | 393 |
| Wins (>+2%) | 114 (29%) |
| Neutral (±2%) | 169 (43%) |
| Losses (>-2%) | 110 (28%) |
| Best gain | +24.9% |
| Worst loss | -7.0% |
| Average (all) | +0.6% |
| Average (wins) | +5.4% |

BF16 baselines mostly use MT256x256x64 (already optimal), so Multi-MT gains come purely from workgroup granularity effects, not MacroTile upgrades. Loss severity is much lower than FP16 (-7% max vs -52%).

### 4.2 Top 10 BF16 Gains

| Problem | BL (TF) | MT (TF) | Gain | Winner |
|---------|---------|---------|------|--------|
| **11264×9216** | 827 | **1034** | **+24.9%** | asym-40/60 [4480,6784] |
| **8704×8448** | 832 | **1036** | **+24.6%** | uniform [4352,4352] |
| **11776×8960** | 816 | **999** | **+22.4%** | asym-40/60 [4608,7168] |
| **8960×12032** | 882 | **1063** | **+20.6%** | pow2-4k [4096,4864] |
| **9216×11008** | 869 | **1030** | **+18.6%** | pow2-4k [4096,5120] |
| **9472×11264** | 829 | **983** | **+18.6%** | asym-40/60 [3712,5760] |
| **10240×9984** | 865 | **1018** | **+17.7%** | pow2-8k [8192,2048] |
| **8960×9728** | 834 | **975** | **+17.0%** | asym-40/60 [3584,5376] |
| **9984×9216** | 858 | **998** | **+16.2%** | uniform [4992,4992] |
| **9984×10496** | 858 | **982** | **+14.4%** | asym-40/60 [3968,6016] |

**Pattern:** BF16 gains occur when baseline is below ~870 TFLOPS (sub-60% of peak ~1500 TF).

---

## 5. Empirical Search Optimization

The original empirical search had an O(N²) bottleneck from `getAllAlgos()` re-scoring. Fixes applied:

| Fix | Impact |
|-----|--------|
| Remove O(N²) Origami scoring | 4+ hour hangs → seconds |
| Direct sub-problem construction | Skip `splitGemmProblem()` re-entry |
| Reduce candidates 7→4 | 43% fewer empirical trials |
| Reduce micro-bench iters 3→1 | 67% less GPU time per candidate |

**Result:** 10240² completes in 1.4s, 20480² in 1.9s (FP16). BF16 remains slower due to hipBLASLt kernel code object loading overhead.

---

## 6. Decision Rules

### FP16

| Condition | Action | Expected |
|-----------|--------|----------|
| Baseline MT = MT256x240x64 | **Always enable** | +13% avg, 86% win rate |
| Baseline MT = MT256x208x64 | **Enable** | +7.6% avg, 84% win rate |
| Baseline MT = MT256x224x64 | **Enable cautiously** | +2.9% avg, 40% win rate |
| Baseline MT = MT256x256x64 | **Do not enable** | -1.5% avg, 16% win rate |
| min(M,N) ≤ 1024 | **Do not enable** | Risk of -29% to -52% |
| Both M,N ≤ 4096 | **Do not enable** | Risk of -23% to -49% |

### BF16

| Condition | Action | Expected |
|-----------|--------|----------|
| Baseline < 870 TFLOPS | **Enable** | +5% to +25% |
| Baseline > 1000 TFLOPS | **Do not enable** | Risk of -2% to -7% |
| M,N in 8704–12288 range | **Best chance** | Highest BF16 win rate |

### Safe Rule (Both Precisions)

**Enable when: M ≥ 5120 AND N ≥ 2048 AND K ≥ 2048 AND baseline MacroTile ≠ MT256x256x64.**

---

## 7. Reproducing These Results

```bash
cd /home/smalekta/MultiMT/rocm-libraries/projects/hipblaslt/build/release

# Calculate iterations for 3s target: ITERS = ceil(3000000 / estimated_us_per_iter)
# FP16: us_per_iter ≈ 2*M*N*K / 1.2e12 * 1e6
# BF16: us_per_iter ≈ 2*M*N*K / 1.4e12 * 1e6

# Example: 10240x10240x8192 FP16 → us≈1430 → ITERS=ceil(3e6/1430)=2098
./clients/hipblaslt-bench -m 10240 -n 10240 -k 8192 --precision f16_r \
  --device 1 --api_method c -i 2098 -j 2098 \
  --multi_macrotile --split_strategy 17 --num_splits 2 --l2_cache_hints

# Output includes:
#   === Baseline Analysis === (MT, WG grid, granularity, origami latency, measured us)
#   === Origami Empirical Split Search === (4 candidates, 1 iter each)
#   === Sub-Problem Detailed Analysis === (per-sub MT, WG, granularity, origami)
#   Multi-MT Performance: combined TFLOPS
```

**Note:** Benchmarks for 2835 total problems (787 completed so far) are continuing on all 8 GPUs. BF16 problems take significantly longer due to hipBLASLt kernel code object loading overhead. Results will be updated as more data arrives.
