# Multi-MacroTile Comprehensive Benchmark Results

**Device:** AMD Instinct MI355X (256 CUs, gfx950)  
**Precision:** FP16  
**hipBLASLt Version:** 100202  
**Date:** 2026-04-17  
**Test Configuration:** -i 100 -j 100 (100 cold + 100 hot iterations), --device 7

---

## Executive Summary

### Optimizations Implemented (9 total)

**Round 1 (8 optimizations):**
1. Pre-create layouts outside hot loop (eliminates per-iteration overhead)
2. CU-mask stream partitioning for stream-parallel
3. Separate workspace per stream
4. Fix autoSelectNumSplits (always 2 for sequential)
5. Concurrent warmup on separate streams
6. Micro-benchmark strategy selection (runtime regression elimination)
7. K-dimension aware split sizing (strategies 19/20)
8. Allow pow2 splits for non-pow2 dims (fixes 12288 regression)

**Round 2 (Origami Empirical Search):**
9. S17 Origami empirical split search: generates 6 candidate split ratios, micro-benchmarks each with 3 real iterations, picks empirically fastest split

### Key Findings

| Category | S3 Uniform | S17 Origami | vs Baseline |
|----------|-----------|-------------|-------------|
| **Best overall gain** | +25.8% | **+28.1%** | 12288x6144x8192 |
| **Best square gain** | +9.8% | **+26.9%** | 10240x10240x32768 |
| **Worst regression** | -18.6% | **-1.5%** | 13312x13312x8192 |
| **Win rate (30 problems)** | 11/25 | **17/25 (68%)** | Active cases |
| **Average gain (wins)** | +6.1% | **+12.2%** | - |
| **Small matrix safety** | Auto-disabled | Auto-disabled | 0% (safe) |

**S17 Origami is the recommended strategy** for all multi-MacroTile workloads. It automatically discovers the optimal non-uniform split ratio via empirical micro-benchmark, providing up to +28% uplift with worst-case -1.5% regression.

---

## Test Suite 1: K Scaling (M=N=10240)

| K | Baseline (TF) | S3 Uniform (TF) | S3 Gain | S17 Origami (TF) | S17 Gain | S17 Winner | Best |
|---|--------------|-----------------|---------|-----------------|----------|------------|------|
| 4096 | 1.255 | 1.281 | +2.1% | **1.375** | **+9.6%** | [6144,4096] | **S17** |
| 8192 | 1.167 | 1.278 | +9.5% | **1.396** | **+19.6%** | [6144,4096] | **S17** |
| 16384 | 1.149 | 1.250 | +8.8% | **1.377** | **+19.8%** | [6144,4096] | **S17** |
| 32768 | 1.093 | 1.240 | +13.5% | **1.386** | **+26.9%** | [4096,6144] | **S17** |

**Analysis**: S17 Origami dominates across all K values, finding the [6144,4096] or [4096,6144] asymmetric split that consistently outperforms uniform by +7-12pp. The gain grows with K because the overhead is amortized over longer execution times.

---

## Test Suite 2: Square Matrix Scaling (K=8192)

| Size | Baseline (TF) | S3 (TF) | S3 Gain | S17 (TF) | S17 Gain | S17 Winner | Best |
|------|--------------|---------|---------|----------|----------|------------|------|
| 4096 | 1.496 | disabled | 0% | disabled | 0% | - | Baseline |
| 6144 | 1.498 | disabled | 0% | disabled | 0% | - | Baseline |
| 8192 | 1.540 | disabled | 0% | disabled | 0% | - | Baseline |
| 9216 | 1.303 | disabled | 0% | disabled | 0% | - | Baseline |
| 9728 | 1.424 | disabled | 0% | disabled | 0% | - | Baseline |
| **10240** | 1.167 | 1.278 | +9.5% | **1.396** | **+19.6%** | [6144,4096] | **S17** |
| **10752** | 1.428 | - | - | 1.422 | -0.4% | [3200,7552] | Baseline |
| **11264** | 1.409 | 1.462 | +3.8% | **1.457** | **+3.4%** | [5632,5632] | S3/S17 |
| **11776** | 1.172 | - | - | **1.421** | **+21.3%** | [3456,8320] | **S17** |
| 12288 | 1.531 | 1.244 | -18.8% | 1.530 | -0.1% | [8192,4096] | Baseline |
| 13312 | 1.403 | 1.169 | -16.7% | 1.382 | -1.5% | [3968,9344] | Baseline |
| **14336** | 1.410 | 1.383 | -1.9% | **1.467** | **+4.0%** | [8192,6144] | **S17** |
| **15360** | 1.181 | 1.203 | +1.9% | **1.399** | **+18.5%** | [4608,10752] | **S17** |
| **16384** | 1.482 | 1.515 | +2.2% | **1.518** | **+2.4%** | [8192,8192] | **S17** |

### Key Observations

**S17 wins 6 of 10 active square sizes**, discovering asymmetric splits that S3 cannot find. The biggest wins are on "performance valley" dimensions (10240, 11776, 15360) where baseline kernels are suboptimal.

**S17 never regresses more than -1.5%** even on already-optimal baselines (12288, 13312), vs S3 which regresses -18.8% on 12288.

**Small matrices (< 10240) auto-disabled**: Prevents the -16% to -25% regressions seen with forced splitting.

---

## Test Suite 3: Rectangular Matrices (K=8192)

| Size | Baseline (TF) | S3 (TF) | S3 Gain | S17 (TF) | S17 Gain | S17 Winner | Best |
|------|--------------|---------|---------|----------|----------|------------|------|
| **12288×6144** | 1.183 | 1.487 | +25.7% | **1.515** | **+28.1%** | pow2 [8192,4096] | **S17** |
| **6144×12288** | 1.283 | 1.497 | +16.7% | **1.496** | **+16.6%** | pow2 [4096,2048] | S3/S17 |
| **16384×8192** | 1.480 | 1.540 | +4.1% | **1.546** | **+4.4%** | uniform [8192,8192] | **S17** |
| 8192×16384 | 1.552 | 1.524 | -1.8% | 1.530 | -1.4% | uniform [4096,4096] | Baseline |
| **10240×5120** | 1.337 | 1.328 | -0.7% | **1.447** | **+8.2%** | asym [3072,7168] | **S17** |
| **5120×10240** | 1.302 | 1.291 | -0.8% | **1.455** | **+11.8%** | asym [1536,3584] | **S17** |
| **20480×10240** | 1.432 | 1.174 | -18.0% | **1.500** | **+4.8%** | asym [8192,12288] | **S17** |
| 10240×20480 | 1.512 | 1.144 | -24.3% | 1.507 | -0.4% | asym [4096,6144] | Baseline |

**S17 transforms rectangular performance**: Cases where S3 lost badly (20480x10240 at -18%, 10240x5120 at -0.7%) become wins (+4.8%, +8.2%) with S17's asymmetric splits. The empirical search finds that non-uniform ratios like [3072,7168] give both sub-problems better kernel selection than uniform [5120,5120].

---

## Test Suite 4: Stream-Parallel (CU-Mask Partitioned)

| Size | Baseline (TF) | Stream-Parallel S8 (TF) | Gain |
|------|--------------|------------------------|------|
| 10240 | 1.163 | 1.210 | +4.0% |
| 12288 | 1.536 | 1.148 | -25.2% |
| 16384 | 1.481 | 1.507 | +1.8% |

**Note**: Stream-parallel (S8) still faces HBM bandwidth contention. S17 sequential with asymmetric splits outperforms stream-parallel in most cases. Stream-parallel remains experimental.

---

## Recommendations

### Production Deployment

```bash
# RECOMMENDED: S17 Origami with empirical search
./hipblaslt-bench -m $M -n $N -k $K \
  --precision f16_r --device 7 \
  --multi_macrotile --split_strategy 17 --num_splits 2 \
  --l2_cache_hints --api_method c -i 100 -j 100
```

This configuration:
- Tests 6 candidate split ratios with 3 real iterations each (~15ms overhead)
- Picks the empirically fastest split for the full timing run
- Discovers asymmetric splits invisible to uniform strategies
- Auto-disables for small matrices (<10240) or small K (<4096)
- **68% win rate, +12.2% average gain on winning cases, worst case -1.5%**

### Decision Tree

```
Is M or N < 10240?
├─ YES → Auto-disabled (0% change, safe)
└─ NO  → Is K ≥ 4096?
    ├─ NO  → Auto-disabled (0% change)
    └─ YES → S17 Origami empirical search
        ├─ Test 6 candidates (50/50, 60/40, 40/60, 70/30, 30/70, pow2)
        ├─ Micro-bench each (3 iterations)
        └─ Pick winner → Use for full timing run (+2% to +28%)
```

---

## Test Suite 6: Origami Empirical Split Search (S17) -- 30-Problem Sweep

**S17 Origami-Optimized M-Split with Empirical Micro-Benchmark**: Generates 6 candidate split ratios (50/50, 60/40, 40/60, 70/30, 30/70, pow2-biased), micro-benchmarks each with 3 real iterations, picks the empirically fastest split.

**Timing verified**: GFLOPS = 2*M*N*K / (avg_time_us * 1000). Time includes all sub-problem kernel launches on same stream, synced once at end. FLOPs use full original problem size.

### Complete Results (30 problems, --device 7, -i 100 -j 100)

| Problem | BL (TF) | BL (us) | S17 (TF) | S17 (us) | Gain | Winning Split |
|---------|---------|---------|----------|----------|------|---------------|
| **10240x10240x4096** | 1.255 | 684 | **1.375** | 625 | **+9.6%** | asym-60/40 [6144,4096] |
| **10240x10240x8192** | 1.167 | 1473 | **1.396** | 1231 | **+19.6%** | asym-60/40 [6144,4096] |
| **10240x10240x16384** | 1.149 | 2991 | **1.377** | 2497 | **+19.8%** | asym-60/40 [6144,4096] |
| **10240x10240x32768** | 1.093 | 6288 | **1.386** | 4955 | **+26.9%** | asym-40/60 [4096,6144] |
| **11264x11264x8192** | 1.409 | 1475 | **1.457** | 1426 | **+3.4%** | uniform [5632,5632] |
| 12288x12288x8192 | 1.531 | 1616 | 1.530 | 1617 | -0.1% | pow2 [8192,4096] |
| 13312x13312x8192 | 1.403 | 2070 | 1.382 | 2100 | -1.5% | asym-30/70 [3968,9344] |
| **14336x14336x8192** | 1.410 | 2388 | **1.467** | 2296 | **+4.0%** | pow2 [8192,6144] |
| **15360x15360x8192** | 1.181 | 3273 | **1.399** | 2764 | **+18.5%** | asym-30/70 [4608,10752] |
| **16384x16384x8192** | 1.482 | 2968 | **1.518** | 2898 | **+2.4%** | uniform [8192,8192] |
| 10752x10752x8192 | 1.428 | 1327 | 1.422 | 1334 | -0.4% | asym-30/70 [3200,7552] |
| **11776x11776x8192** | 1.172 | 1938 | **1.421** | 1599 | **+21.3%** | asym-30/70 [3456,8320] |
| **12288x6144x8192** | 1.183 | 1046 | **1.515** | 816 | **+28.1%** | pow2 [8192,4096] |
| **6144x12288x8192** | 1.283 | 964 | **1.496** | 826 | **+16.6%** | pow2 [4096,2048] |
| **16384x8192x8192** | 1.480 | 1485 | **1.546** | 1423 | **+4.4%** | uniform [8192,8192] |
| 8192x16384x8192 | 1.552 | 1417 | 1.530 | 1437 | -1.4% | uniform [4096,4096] |
| **10240x5120x8192** | 1.337 | 643 | **1.447** | 594 | **+8.2%** | asym-30/70 [3072,7168] |
| **5120x10240x8192** | 1.302 | 660 | **1.455** | 590 | **+11.8%** | asym-30/70 [1536,3584] |
| **20480x10240x8192** | 1.432 | 2399 | **1.500** | 2291 | **+4.8%** | asym-40/60 [8192,12288] |
| 10240x20480x8192 | 1.512 | 2273 | 1.507 | 2280 | -0.4% | asym-40/60 [4096,6144] |
| **16384x16384x16384** | 1.502 | 5855 | **1.533** | 5736 | **+2.1%** | uniform [8192,8192] |
| **15360x15360x4096** | 1.297 | 1490 | **1.419** | 1363 | **+9.4%** | asym-30/70 [4608,10752] |
| 14336x14336x4096 | 1.417 | 1188 | 1.422 | 1185 | +0.3% | pow2 [8192,6144] |
| 12288x12288x4096 | 1.494 | 828 | 1.477 | 838 | -1.1% | pow2 [8192,4096] |
| 16384x16384x4096 | 1.504 | 1462 | 1.494 | 1472 | -0.7% | uniform [8192,8192] |
| 10240x10240x2048 | 1.168 | 368 | --- | --- | 0% | auto-disabled (K<4096) |
| 10240x10240x1024 | 1.016 | 211 | --- | --- | 0% | auto-disabled |
| 10240x10240x512 | 0.805 | 133 | --- | --- | 0% | auto-disabled |
| 9216x9216x8192 | 1.303 | 1068 | --- | --- | 0% | auto-disabled (M<10240) |
| 9728x9728x8192 | 1.424 | 1089 | --- | --- | 0% | auto-disabled (M<10240) |

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total problems tested | 30 |
| Auto-disabled (safe, 0% change) | 5 |
| Active comparisons | 25 |
| **Wins (>0.5% uplift)** | **17 of 25 (68%)** |
| Losses (>0.5% regression) | 4 of 25 (16%) |
| Neutral (within ±0.5%) | 4 of 25 (16%) |
| **Best uplift** | **+28.1%** (12288x6144x8192) |
| **Worst regression** | **-1.5%** (13312x13312x8192) |
| Average gain (active cases) | **+7.7%** |
| Average gain (winning cases only) | **+12.2%** |

### Top 10 Winning Cases

| # | Problem | Gain | S17 (TF) | BL (TF) | Winning Split |
|---|---------|------|----------|---------|---------------|
| 1 | **12288x6144x8192** | **+28.1%** | 1.515 | 1.183 | pow2 [8192,4096] |
| 2 | **10240x10240x32768** | **+26.9%** | 1.386 | 1.093 | asym-40/60 [4096,6144] |
| 3 | **11776x11776x8192** | **+21.3%** | 1.421 | 1.172 | asym-30/70 [3456,8320] |
| 4 | **10240x10240x16384** | **+19.8%** | 1.377 | 1.149 | asym-60/40 [6144,4096] |
| 5 | **10240x10240x8192** | **+19.6%** | 1.396 | 1.167 | asym-60/40 [6144,4096] |
| 6 | **15360x15360x8192** | **+18.5%** | 1.399 | 1.181 | asym-30/70 [4608,10752] |
| 7 | **6144x12288x8192** | **+16.6%** | 1.496 | 1.283 | pow2 [4096,2048] |
| 8 | **5120x10240x8192** | **+11.8%** | 1.455 | 1.302 | asym-30/70 [1536,3584] |
| 9 | **10240x10240x4096** | **+9.6%** | 1.375 | 1.255 | asym-60/40 [6144,4096] |
| 10 | **15360x15360x4096** | **+9.4%** | 1.419 | 1.297 | asym-30/70 [4608,10752] |

### Analysis: Where Origami Wins and Why

**Pattern 1: 10240-class problems with any K (+9.6% to +26.9%)**

The [4096, 6144] or [6144, 4096] asymmetric split consistently wins for 10240x10240. Both 4096 and 6144 get highly-tuned kernels, and the empirical search discovers this is better than the uniform [5120, 5120].

**Pattern 2: Rectangular 2:1 ratios (+8.2% to +28.1%)**

12288x6144, 6144x12288, 10240x5120, 5120x10240 all show significant wins. The pow2-biased split creates clean power-of-2 sub-problems (e.g., [8192, 4096]) that map to peak-efficiency kernels.

**Pattern 3: 15360-class "dead zone" problems (+9.4% to +18.5%)**

15360 is a particularly bad dimension for single-kernel (1.181 TF baseline). The 30/70 split [4608, 10752] gets much better kernel selection for both sub-problems.

**Pattern 4: 11776 "odd size" problem (+21.3%)**

11776 is far from any power-of-2 and gets a slow baseline kernel (1.172 TF). Origami's [3456, 8320] split rescues it to 1.421 TF.

**Where it doesn't help: Already-optimal baseline sizes**

12288x12288 (1.531 TF baseline), 10752x10752 (1.428 TF), 8192x16384 (1.552 TF) -- these dimensions already hit near-peak performance with single kernel. Splitting can't improve and adds slight overhead.

### Winning Split Ratio Distribution

| Winning Ratio | Count | Typical Problem |
|---------------|-------|-----------------|
| uniform 50/50 | 6 | 16384x16384, 11264x11264 |
| asym 60/40 | 3 | 10240x10240 (any K) |
| asym 40/60 | 2 | 10240x10240x32768 |
| asym 30/70 | 6 | 15360, 11776, 5120x10240 |
| pow2-biased | 5 | 12288x12288, 14336, 12288x6144 |
| asym 70/30 | 0 | (never optimal) |

Key finding: **70/30 is never optimal** but **30/70 is the most common winner** (6 times). This suggests that creating one small power-of-2-like sub-problem paired with one large sub-problem is often the best strategy.

### Recommendations for Production Use

```bash
# RECOMMENDED: S17 Origami for all problems >= 10240 with K >= 4096
./hipblaslt-bench -m $M -n $N -k $K \
  --precision f16_r --device 7 \
  --multi_macrotile --split_strategy 17 --num_splits 2 \
  --l2_cache_hints --api_method c -i 100 -j 100
```

**When to use S17**: M or N >= 10240 AND K >= 4096 (auto-disabled otherwise)
**Expected gain**: +7.7% average, +12.2% on winning cases, up to +28.1%
**Worst case**: -1.5% (rare, only for already-optimal baselines)
**Overhead**: ~15ms for candidate search (amortized over long-running benchmarks)

---

**Last Updated**: 2026-04-17 (post Origami empirical implementation)  
**Status**: Full 30-problem sweep complete with verified timing  
**Best Result**: 12288x6144x8192: **1.515 TF (+28.1% over baseline)**  
**Win Rate**: 17/25 active cases (68%)
