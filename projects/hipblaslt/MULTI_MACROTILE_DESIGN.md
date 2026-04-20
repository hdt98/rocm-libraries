# Multi-MacroTile: Origami-Based Split Search

**hipBLASLt Feature Documentation**  
**Version:** 3.0  
**Date:** 2026-04-20

---

## Overview

Multi-MacroTile splits a single GEMM into two sub-problems solved sequentially with per-subproblem kernel selection. An empirical micro-benchmark ("Origami search") tests multiple candidate split ratios at runtime and picks the fastest.

Many problem sizes sit in "performance valleys" where the single-kernel baseline achieves only 44-76% of peak. Splitting avoids these valleys by creating sub-problems at dimensions with better-tuned kernels. The best case shows **+64.8%** improvement over baseline.

---

## How It Works

### Step 1: Guard Check

```
shouldUseMultiMacroTile(M, N, K):
  - K < 4096       → disabled (overhead dominates)
  - M,N both <10240 → disabled (small problems lose performance)
```

### Step 2: MacroTile Preservation

Before splitting, the Origami path queries `getAllAlgos` for both the original problem and the uniform split (M/2). If the split's best kernel uses a MacroTile < 75% of the baseline's MacroTile, splitting is rejected (returns to single kernel).

### Step 3: Candidate Generation

`generateOrigamiCandidates(total_size, macrotile_size)` produces ~7-8 candidates:

| Source | Candidates | Example (M=10240) |
|--------|------------|-------------------|
| Uniform | 50/50 | [5120, 5120] |
| Ratio-based | 60/40, 40/60, 70/30, 30/70 | [6144, 4096], [4096, 6144], [7168, 3072], [3072, 7168] |
| Exhaustive pow2 | [2048, rem], [4096, rem], [8192, rem] | [2048, 8192], [4096, 6144], [8192, 2048] |

All candidates are MacroTile-aligned and enforce a minimum sub-problem size of `max(2048, 15% of total)`.

### Step 4: Empirical Micro-Benchmark

In `testing_matmul.hpp`, for each candidate:
1. Build sub-problems with candidate split sizes and compute matrix offsets
2. Pre-create matrix layouts, query heuristic for each sub-problem
3. Run 3 iterations of sequential `hipblasLtMatmul` calls
4. Measure wall-clock time

The candidate with the lowest measured time wins.

### Step 5: Timing Run

The winning split is used for the full timing run (100+ iterations). Matrix layouts are pre-created once before the loop; only `hipblasLtMatmul` calls are in the hot path.

---

## File Structure

```
clients/common/include/
├── multi_macrotile.hpp                   # Core: GemmSubProblem, SubProblemContext,
│                                         #   offset calculations, splitGemmProblem()
├── multi_macrotile_origami_improved.hpp  # Origami: candidate generation,
│                                         #   MacroTile preservation check,
│                                         #   getOrigamiCandidates()

Integration points:
├── testing_matmul.hpp          # Origami empirical search loop + timing
├── hipblaslt_arguments.hpp     # CLI fields (multi_macrotile, split_strategy, num_splits)
└── client.cpp                  # --multi_macrotile --split_strategy --num_splits options
```

### `multi_macrotile.hpp` (~200 lines)

| Component | Purpose |
|-----------|---------|
| `GemmSubProblem` | Sub-problem dimensions, offsets, workgroup estimate |
| `SubProblemContext` | Pre-created layouts + algo + pointers for hot loop |
| `getDataTypeSize()` | Element size lookup |
| `calculateOffset{A,B,CD,Bias}()` | Byte offset for split matrices |
| `shouldUseMultiMacroTile()` | Guard: K >= 4096 and M or N >= 10240 |
| `splitGemmProblem()` | Entry point: calls Origami, builds sub-problems |

### `multi_macrotile_origami_improved.hpp` (~160 lines)

| Component | Purpose |
|-----------|---------|
| `OrigamiCandidate` | Split sizes + label string |
| `getOrigamiCandidates()` | Thread-local candidate storage |
| `parseMacroTileFromName()` | Extract MT dims from solution name |
| `isMacroTilePreserved()` | Check split doesn't degrade MacroTile |
| `generateOrigamiCandidates()` | Produce ratio + pow2 candidates |
| `computeOrigamiOptimizedSplitsWithHandle()` | Entry: check MT, generate candidates, return default |

---

## CLI Usage

```bash
./hipblaslt-bench -m 10240 -n 10240 -k 8192 \
  --precision f16_r --device 7 \
  --multi_macrotile --split_strategy 17 --num_splits 2 \
  --l2_cache_hints --api_method c -i 100 -j 100
```

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--multi_macrotile` | flag | false | Enable splitting |
| `--split_strategy` | 17 (M-split) or 18 (N-split) | 17 | Split dimension |
| `--num_splits` | 2 | 2 | Number of sub-problems |
| `--l2_cache_hints` | flag | true | L2 persistence for shared matrix |

---

## Benchmark Results (46-Problem Sweep)

**Device:** AMD Instinct MI355X (256 CUs, gfx950)  
**Precision:** FP16, --device 7, -i 100 -j 100

### Aggregate

| Metric | Value |
|--------|-------|
| Problems tested | 46 |
| Auto-disabled (safe, 0%) | 6 |
| Active comparisons | 40 |
| **Wins (>0.5%)** | **33 (82%)** |
| Losses (>0.5%) | 7 (18%) |
| **Best gain** | **+64.8%** |
| Worst loss | -11.8% |
| **Avg gain (all active)** | **+22.9%** |
| **Avg gain (wins only)** | **+28.7%** |

### Top 15

| Problem | BL (TF) | S17 (TF) | Gain | Split |
|---------|---------|----------|------|-------|
| 15360x15360x8192 | 0.679 | 1.119 | **+64.8%** | asym-30/70 [4608,10752] |
| 14848x14848x8192 | 0.799 | 1.236 | **+54.7%** | uniform [7424,7424] |
| 13824x13824x8192 | 0.791 | 1.201 | **+51.8%** | uniform [6912,6912] |
| 20480x10240x8192 | 0.830 | 1.238 | **+49.2%** | asym-30/70 [6144,14336] |
| 12288x10240x8192 | 0.884 | 1.291 | **+46.0%** | uniform [6144,6144] |
| 16384x16384x8192 | 0.839 | 1.222 | **+45.6%** | asym-40/60 [6528,9856] |
| 10240x20480x8192 | 0.859 | 1.241 | **+44.5%** | asym-30/70 [3072,7168] |
| 16384x16384x16384 | 0.857 | 1.218 | **+42.1%** | asym-40/60 [6528,9856] |
| 10240x12288x8192 | 0.891 | 1.260 | **+41.4%** | uniform [5120,5120] |
| 14336x14336x8192 | 0.809 | 1.135 | **+40.3%** | pow2-2k [2048,12288] |
| 10240x10240x32768 | 0.859 | 1.176 | **+36.9%** | asym-40/60 [4096,6144] |
| 12288x12288x12288 | 0.839 | 1.144 | **+36.4%** | pow2-2k [2048,10240] |
| 10240x10240x14336 | 0.886 | 1.191 | **+34.4%** | asym-40/60 [4096,6144] |
| 12288x12288x8192 | 0.890 | 1.182 | **+32.8%** | pow2-2k [2048,10240] |
| 5120x15360x8192 | 0.816 | 1.075 | **+31.7%** | uniform [2560,2560] |

### Losses

| Problem | BL (TF) | S17 (TF) | Loss | Cause |
|---------|---------|----------|------|-------|
| 10240x6144x4096 | 1.255 | 1.107 | -11.8% | Small K + rectangular |
| 11520x11520x8192 | 1.228 | 1.126 | -8.3% | Baseline already efficient |
| 13312x13312x8192 | 1.186 | 1.146 | -3.4% | Non-ideal split found |
| 14080x14080x8192 | 1.217 | 1.183 | -2.8% | Baseline efficient |
| 15104x15104x8192 | 1.180 | 1.165 | -1.3% | Marginal |
| 5120x10240x8192 | 1.024 | 1.016 | -0.8% | Marginal |
| 12800x12800x8192 | 1.244 | 1.236 | -0.6% | Baseline efficient |

Common pattern: regressions occur when the baseline is already efficient (> 1.1 TF).

---

## Why It Works

Kernel profiling reveals dramatic efficiency variation by sub-problem M-dimension on MI355X:

| M size | TFLOPS | Efficiency |
|--------|--------|------------|
| 8192 | 1.536 | 99.8% |
| 6144 | 1.482 | 96.2% |
| 3072 | 1.465 | 95.1% |
| 4096 | 1.333 | 86.6% |
| 2048 | 1.335 | 86.7% |
| 5120 | 1.294 | 84.0% |
| 10240 | 1.165 | 75.6% |
| 15360 | 0.679 | 44.1% |

The 10240 single-kernel baseline runs at only 75.6% efficiency. Splitting into [6144, 4096] gives each sub-problem a better kernel (96.2% and 86.6%), for a combined time that's ~20% faster.

The 15360 baseline is catastrophically slow (44.1%). Even a uniform [7680, 7680] split dramatically improves it because each half avoids the dead-zone kernel.

---

## Removed Code

The following were removed during the cleanup to simplify the codebase:

| Removed | Lines | Reason |
|---------|-------|--------|
| `multi_macrotile_fused.hpp` | 278 | Fused dispatch: blocked by platform |
| `multi_macrotile_fused_kernel.hpp` | 289 | Device-side dispatch: never worked |
| `multi_macrotile_fused_host.hpp` | 323 | Host-side batch: never worked |
| `multi_macrotile_kernel_extraction.hpp` | 410 | Kernel extraction: never worked |
| `multi_macrotile_origami.hpp` | 520 | Old Origami with estimation: replaced |
| Strategies 0-10, 15-16, 19-20 | ~1000 | Legacy uniform/heuristic: outperformed by S17 |
| Stream-parallel path | ~150 | CU contention issues: underperformed |

**Net reduction**: ~3200 lines -> ~360 lines (89% reduction).
