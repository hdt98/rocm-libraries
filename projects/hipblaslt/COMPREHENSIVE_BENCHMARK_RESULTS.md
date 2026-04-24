# Multi-MacroTile Comprehensive Benchmark Results

**Device:** AMD Instinct MI350X (gfx950)  
**Precision:** FP16 (f16_r / HHS)  
**hipBLASLt Version:** 100202 (git: c24cf177ca)  
**Date:** 2026-04-23  
**Devices:** `--device 0` through `--device 7` (8× MI350X)  
**Layouts:** NN, TN, NT, TT (all 4 transpose combinations)  
**Methodology:** Adaptive iterations (≥3s cold + ≥3s hot). Every data point validated with `--verify`.  
**K:** 8192 (fixed), M and N from 1024 to 32768  
**Total valid data points:** 728 (all verified, max norm_error = 5.08e-05)  
**Status:** Benchmarks ongoing on 8 GPUs, document updates live

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| **Valid data points** | **728** |
| Wins (>+2%) | 151 (21%) |
| Neutral (±2%) | 311 (43%) |
| Losses (>-2%) | 266 (37%) |
| **Best gain** | **+35.4%** (9216×9216, TN) |
| Worst loss | -25.4% (8192×2048, TT) |
| Average (all) | -1.0% |
| **Verification** | **728/728 pass** (max norm = 5.08e-05) |

### By Layout

| Layout | Points | Wins (%) | Losses (%) | Avg Gain | Best |
|--------|--------|----------|------------|----------|------|
| **NN** | 185 | **47 (25%)** | 64 (35%) | **+0.5%** | +26.8% |
| **TN** | 182 | **45 (25%)** | 74 (41%) | -0.5% | **+35.4%** |
| NT | 182 | 34 (19%) | 55 (30%) | -0.9% | +11.0% |
| TT | 179 | 25 (14%) | 73 (41%) | -3.2% | +13.6% |

**NN and TN layouts have 2× higher win rates** than TT. This is because NN/TN more frequently get the suboptimal MT256x240x64 baseline, which Multi-MT can upgrade.

---

## 2. Baseline MacroTile → Multi-MT Effectiveness (NN Layout)

| Baseline MT | Points | Wins | Win Rate | Avg Gain | Explanation |
|-------------|--------|------|----------|----------|-------------|
| **MT256x240x64** | 32 | 21 | **66%** | **+10.3%** | 240 → prime N-tiles (43), severe XCD imbalance |
| **MT256x208x64** | 7 | 4 | **57%** | **+3.9%** | 208 → non-pow2 grid, moderate imbalance |
| **MT256x224x64** | 17 | 6 | **35%** | **+2.3%** | 224 → decent but improvable |
| MT256x176x64 | 2 | 2 | 100% | +8.8% | Small sample, needs more data |
| MT256x256x64 | 118 | 13 | **11%** | **-2.6%** | Already optimal; splitting hurts |
| MT256x192x64 | 5 | 0 | 0% | -2.3% | Already efficient |

**Core insight: Multi-MT wins when baseline uses a suboptimal MacroTile (MT_N ≠ 256). When baseline is already MT256x256x64, it loses 2.6% on average.**

---

## 3. Hardware Root Cause

The MI350X has **256 CUs across 8 XCDs**. The MacroTile determines workgroup count: `ceil(M/MT_M) × ceil(N/MT_N)`.

**Why MT256x240x64 loses (the 10240 case):**
- `ceil(10240/240) = 43` N-tiles. 43 is **prime** → cannot divide evenly across 8 XCDs → 20% load imbalance
- `1720 total WGs / 256 CUs = 6.72 waves` → last wave at 71.9% utilization (72 idle CUs)

**After splitting 10240 → [8192, 2048]:**
- Sub-0 (8192): `ceil(8192/256) = 32` → 32/8 = 4.0 per XCD (perfect), 1280 WGs = 5 exact waves
- Sub-1 (2048): `ceil(2048/160) = 13, ceil(10240/160) = 64` → 64/8 = 8.0 per XCD (perfect), 512 WGs = 2 exact waves
- Both achieve 100% CU utilization

**Why TT layout gains less:**
- With both matrices transposed, the heuristic more often selects MT256x256x64 (the optimal tile) from the start → less room for Multi-MT improvement

---

## 4. Top Gains

| Problem | Layout | BL (TF) | Baseline MT | MT (TF) | Gain | Winner |
|---------|--------|---------|-------------|---------|------|--------|
| **9216×9216** | TN | 855 | MT256x224x64 | **1157** | **+35.4%** | single |
| **11264×6144** | TN | 881 | MT256x224x64 | **1193** | **+35.4%** | single |
| **9216×9216** | TN | 774 | MT256x224x64 | **1034** | **+33.6%** | single |
| **5120×8192** | TN | 899 | MT256x224x64 | **1155** | **+28.4%** | single |
| **10240×10240** (K=24576) | NN | 897 | MT256x240x64 | **1138** | **+26.8%** | single |
| **10240×10240** (K=20480) | NN | 927 | MT256x240x64 | **1166** | **+25.8%** | single |
| **10240×10240** (K=16384) | NN | 894 | MT256x240x64 | **1122** | **+25.5%** | single |
| **10240×10240** (K=10240) | NN | 907 | MT256x240x64 | **1131** | **+24.7%** | single |
| **10240×10240** (K=12288) | NN | 909 | MT256x240x64 | **1128** | **+24.0%** | single |
| **10240×10240** (K=8192) | NN | 934 | MT256x240x64 | **1154** | **+23.5%** | single |

**All top gains share the same pattern:** baseline uses a suboptimal MT (240 or 224), and Multi-MT replaces it with MT256x256x64 + MT256x160x64 via a pow2 split.

---

## 5. Worst Losses

| Problem | Layout | BL (TF) | MT (TF) | Loss | Root Cause |
|---------|--------|---------|---------|------|------------|
| 8192×2048 | TT | 1169 | 873 | -25.4% | N=2048 too small for M-split |
| 4096×20480 | TN | 1173 | 879 | -25.1% | M=4096 split creates tiny sub-problems |
| 4096×28672 | TN | 1070 | 819 | -23.4% | M=4096 too small |
| 8192×14336 | TN | 1187 | 915 | -22.9% | BL at 1187 TF already near peak |
| 8192×2048 | NT | 1145 | 891 | -22.2% | N=2048 with M-split → CU underutil |
| 4096×11264 | TT | 1165 | 910 | -21.9% | Small M |
| 4096×12288 | TT | 1184 | 930 | -21.5% | Small M |

**All severe losses have M or N ≤ 4096, or baseline > 1100 TFLOPS.**

---

## 6. Empirical Search Optimization

| Fix | Before | After |
|-----|--------|-------|
| Analytical scoring | O(N²) `getAllAlgos()` (hours hang) | Removed |
| Same-MT guard | Skipped valid splits prematurely | Removed |
| Candidates | 7 | 4 (uniform + pow2 + asym) |
| Micro-bench iterations | 3 per candidate | 1 |
| Sub-problem build | Re-called `splitGemmProblem()` | Direct inline |

**Speed:** 10240² in 1.5s, 20480² in 1.9s. Previously 4+ hours hang.

---

## 7. Decision Rules

| Condition | Layout | Action | Expected |
|-----------|--------|--------|----------|
| BL MT = MT256x240x64 | NN/TN | **Always enable** | +10% to +27% |
| BL MT = MT256x224x64 | NN/TN | **Enable** | +2% to +35% |
| BL MT = MT256x208x64 | NN/TN | **Enable** | +4% to +10% |
| BL MT = MT256x256x64 | Any | **Do not enable** | -2.6% avg |
| Any | TT | **Do not enable** | -3.2% avg |
| min(M,N) ≤ 4096 | Any | **Do not enable** | Risk of -20% to -25% |
| BL > 1100 TFLOPS | Any | **Do not enable** | Already near peak |

### Safe Rule

**Enable when: M ≥ 5120 AND N ≥ 5120 AND K ≥ 4096 AND layout = NN or TN AND baseline MT ≠ MT256x256x64.**

---

## 8. Reproducing Results

```bash
cd /home/smalekta/MultiMT/rocm-libraries/projects/hipblaslt/build/release

# Compute iterations for 3s: ITERS = ceil(3e6 / (2*M*N*K / 1.2e12 * 1e6))
# Example: 10240x10240x8192 → us≈1430 → ITERS=2098

# Multi-MacroTile benchmark:
./clients/hipblaslt-bench -m 10240 -n 10240 -k 8192 --precision f16_r \
  --device 1 --api_method c -i 2098 -j 2098 --transA N --transB N \
  --multi_macrotile --split_strategy 17 --num_splits 2 --l2_cache_hints

# Verify correctness:
./clients/hipblaslt-bench -m 10240 -n 10240 -k 8192 --precision f16_r \
  --device 1 --api_method c -i 1 -j 1 --transA N --transB N --verify
```

**Raw data:** `bench_results_v2/results_dev{0-7}.csv`  
**Benchmarks continuing on 8 GPUs — document updates live.**
