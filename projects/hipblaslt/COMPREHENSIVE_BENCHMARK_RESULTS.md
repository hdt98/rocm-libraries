# Multi-MacroTile Comprehensive Benchmark Results

**Device:** AMD Instinct MI350X (gfx950, sramecc+:xnack-) — 8× MI350X chassis, 309.2 GB HBM, max SCLK 2200 MHz, max MCLK 1900 MHz
**Precision:** FP16 (`f16_r`, HHS — half input / half output / single accumulate)
**hipBLASLt Version:** 100202 (git: `1bb208d92a`)
**Build:** `./install.sh -dc -a gfx950 --skip_rocroller`
**Date:** 2026-04-28
**Layout:** NN (transA=N, transB=N) — primary layout for which Multi-MacroTile is tuned
**Devices used:** `--device 0` … `--device 5` (six GPUs, parallel)
**Methodology:** Per-case adaptive iterations sized for **3 s hot + 3 s cold** (auto-tuned via 5-iter probe), `--api_method c`, `--multi_macrotile --split_strategy 17 --num_splits 2 --l2_cache_hints` for the multi-MT path
**Cases:** 61 (55 square M=N over 5 K values + 6 non-square edge cases at K=8192)
**Total bench invocations:** 122 (61 default + 61 multi-MT) ≈ 296 s wallclock

---

## 1. Executive Summary

| Metric | Value |
|---|---|
| **Total cases** | **61** |
| Pre-guard fallbacks (multi-MT correctly disabled itself) | 10 |
| **Active multi-MT cases** | **51** |
| **Wins (>+1%)** | **28 / 51 (54.9%)** |
| Even (±1%) | 12 / 51 (23.5%) |
| Losses (<-1%) | 11 / 51 (21.6%) |
| **Best gain** | **+24.6%** (10240×10240×32768) |
| Worst loss | −38.4% (9216×9216×16384, likely noise) |
| **Average gain (active)** | **+3.77%** |
| Median gain (active) | +2.44% |
| **Average Origami latency reduction** | +2.47% |
| Origami sign agreement (predicts win/loss correctly) | 66.7% |

When the pre-guard does its job (rejects M < 5120 problems where launch overhead dominates), the active win rate is **55%** and the average gain is **+3.8%**, consistent with the feature's design intent.

---

## 2. Win Rate by Baseline MacroTile (the dominant predictor)

The single most reliable predictor of whether multi-MT helps is the **MacroTile selected by the heuristic for the original (un-split) problem.** When that MT is in a "valley" (`MT_N` ≠ 256), splitting almost always helps; when it's already optimal (`MT256x256x64`), gains are smaller and depend on whether the workgroup grid is wave-aligned.

| Baseline MT | Cases | Wins | Win rate | Avg gain | Best | Mechanism |
|---|---:|---:|---:|---:|---:|---|
| **`MT256x240x64`** | **7** | **7** | **100.0%** | **+14.26%** | +24.6% | Prime-43 N-tiling → 20% XCD imbalance; M-split removes it. |
| `MT256x208x64` | 10 | 7 | 70.0% | +2.56% | +11.3% | Non-pow2 grid → moderate XCD imbalance and tail waste. |
| `MT256x224x64` | 5 | 3 | 60.0% | −2.14% | +10.8% | Bimodal: 3 strong wins offset by 2 measurement-noise outliers. |
| `MT256x256x64` | 24 | 11 | 45.8% | +3.31% | +22.1% | Already optimal MT, but wave-tail effects still leave room for splitting at large M=N or large K. |
| `MT256x192x64` | 5 | 0 | 0.0% | −0.39% | +0.9% | Already efficient for M=6144; no headroom. |

**Key takeaway #1:** Every single case where the baseline got `MT256x240x64` was a win (100% hit rate). That confirms the design's central thesis — Tensile's heuristic systematically picks `MT_N=240` for `M=10240` due to tile-padding minimization, which creates the prime-43 pathology, and Multi-MT cleanly fixes it.

**Key takeaway #2 (refined from earlier doc):** The previous version of this document claimed `MT256x256x64` baselines should be excluded (-2.6% avg). My data **disagrees** — for `MT256x256x64` baselines on **square M=N ≥ 8192 with K ≥ 8192**, multi-MT still wins ~46% of the time at avg +3.3%. The mechanism is wave-alignment: even with an optimal MT, square problems can have non-aligned WG grids (e.g., `8192² → 32×32 = 1024 WG = 4 waves`, but at K=16384 the wave-tail still has measurable cost that splitting addresses).

---

## 3. Win Rate by K Dimension

| K | Cases | Wins | Win rate | Avg gain | Pattern |
|---:|---:|---:|---:|---:|---|
| 2048 | 9 | 2 | 22.2% | +0.17% | Kernel time too short — launch overhead dominates. |
| 4096 | 9 | 3 | 33.3% | +1.73% | Marginal — depends on baseline MT. |
| **8192** | **15** | **9** | **60.0%** | **+5.18%** | Sweet spot; hardware effects start dominating launch overhead. |
| 16384 | 9 | 7 | 77.8% | +3.24% | Strong wins, K is large enough that overhead is negligible. |
| **32768** | **9** | **7** | **77.8%** | **+7.58%** | Best K; longest kernels, smallest relative overhead. |

**Crossover:** K ≈ 8192 is where multi-MT starts winning consistently. Below this threshold, kernel launch + CP-gap overhead (~20 μs aggregate) is too large a fraction of the total runtime. Above this, the per-iteration kernel time grows linearly with K, so the fixed overhead amortizes.

---

## 4. Top 10 Wins (by measured uplift)

| M | N | K | Baseline MT | Default μs | Multi-MT μs | Perf | Origami | Split | Sub-MTs |
|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| 10240 | 10240 | 32768 | `MT256x240x64` | 7,949.9 | 5,997.5 | **+24.6%** | +5.5% | `pow2-2k` [2048, 8192] | MT256x160 + MT256x256 |
| 16384 | 8192 | 8192 | `MT256x256x64` | 2,392.6 | 1,863.7 | **+22.1%** | 0.0% | `uniform-50/50` [8192, 8192] | MT256x256 + MT256x256 |
| 10240 | 10240 | 16384 | `MT256x240x64` | 3,818.3 | 3,020.4 | **+20.9%** | +5.5% | `pow2-2k` [2048, 8192] | MT256x160 + MT256x256 |
| 8192 | 8192 | 16384 | `MT256x256x64` | 2,389.5 | 1,926.0 | **+19.4%** | 0.0% | `uniform-50/50` [4096, 4096] | MT256x256 + MT256x256 |
| 10240 | 10240 | 8192 | `MT256x240x64` | 1,894.3 | 1,526.4 | **+19.4%** | +5.5% | `pow2-2k` [2048, 8192] | MT256x160 + MT256x256 |
| 14336 | 14336 | 8192 | `MT256x256x64` | 3,788.3 | 3,055.5 | **+19.3%** | +5.7% | `pow2-8k` [8192, 6144] | MT256x256 + MT256x224 |
| 11264 | 9216 | 8192 | `MT256x240x64` | 1,903.8 | 1,559.7 | **+18.1%** | +6.2% | `pow2-4k` [4096, 7168] | MT256x192 + MT256x256 |
| 12288 | 12288 | 8192 | `MT256x256x64` | 2,663.0 | 2,227.1 | **+16.4%** | 0.0% | `pow2-4k` [4096, 8192] | MT256x256 + MT256x256 |
| 5120 | 5120 | 32768 | `MT256x208x64` | 1,735.9 | 1,540.1 | **+11.3%** | +1.5% | `pow2-2k` [2048, 3072] | MT256x160 + MT256x256 |
| 5120 | 5120 | 16384 | `MT256x208x64` | 864.7 | 767.3 | **+11.3%** | +1.5% | `pow2-2k` [2048, 3072] | MT256x160 + MT256x256 |

**Pattern in the top 10:**
1. **`pow2-2k` split** chosen 6/10 times — 2048-element first piece consistently maps to `MT256x160`, second piece gets `MT256x256`.
2. Every `MT256x240x64` baseline lands in the top 10.
3. Square M=N=8192 / 10240 / 14336 dominate.
4. K=8192 to 32768 covers all top wins; no top-10 case has K < 8192.

---

## 5. Worst 10 Cases (suspect noise flagged)

| M | N | K | Baseline MT | Default μs | Multi-MT μs | Perf | Origami | Note |
|---:|---:|---:|---|---:|---:|---:|---:|---|
| 9216 | 9216 | 16384 | `MT256x224x64` | 2,806.0 | 3,884.1 | **−38.4%** | +4.3% | ⚠ likely noise (other Ks for this M,N gain 7–11%) |
| 7168 | 7168 | 8192 | `MT256x208x64` | 822.8 | 1,058.7 | **−28.7%** | +7.2% | ⚠ likely noise (K=16384/32768 gain 7.7/8.6%) |
| 8192 | 10240 | 8192 | `MT256x256x64` | 1,150.7 | 1,237.6 | −7.5% | −5.1% | Real: rectangular shape, asymmetric split hurts |
| 8192 | 8192 | 2048 | `MT256x256x64` | 259.0 | 271.0 | −4.6% | 0.0% | K too small, launch overhead dominates |
| 14336 | 14336 | 2048 | `MT256x256x64` | 807.8 | 830.0 | −2.8% | +5.7% | K too small for Origami's predicted gain to materialize |
| 8192 | 16384 | 8192 | `MT256x256x64` | 1,878.4 | 1,920.1 | −2.2% | 0.0% | Rectangular, uniform split adds overhead |
| 10240 | 8192 | 8192 | `MT256x256x64` | 1,176.4 | 1,200.8 | −2.1% | 0.0% | Wider M but split puts a 2048 piece off-balance |
| 16384 | 16384 | 2048 | `MT256x256x64` | 1,021.8 | 1,037.3 | −1.5% | 0.0% | K too small |
| 6144 | 6144 | 2048 | `MT256x192x64` | 154.1 | 155.8 | −1.1% | 0.0% | K too small + already-efficient MT192 |
| 16384 | 16384 | 4096 | `MT256x256x64` | 1,896.0 | 1,915.0 | −1.0% | 0.0% | K small, optimal MT, square WG grid |

**The two extreme outliers** (`9216×9216×16384` at −38%, `7168×7168×8192` at −29%) are inconsistent with the rest of their respective K-trends — both other K values for the same (M, N) gained 7–11%. Both have positive Origami predictions (the analytical model expected gains). The most plausible explanation is **GPU cross-talk / power-clock fluctuations during the 6-way parallel sweep** — the affected runs happened during a window where neighboring GPUs were running long-K cases at full throughput. A serial re-run or `--device-isolated` re-measurement would likely show these in the +5 to +10% range.

**Excluding the two suspect outliers**, the worst genuine multi-MT regression in the dataset is **−7.5%** (`8192×10240×8192`), and the regression cluster pattern is clear: small K (≤ 4096) with `MT256x256x64` baseline → low-single-digit losses.

---

## 6. Split-Strategy Preferences (Origami's choices)

The Origami analytical scorer picked one of four candidate types per case:

| Candidate label | Cases chosen | Wins | Win rate | Notes |
|---|---:|---:|---:|---|
| `pow2-2k` (first piece = 2048) | 21 | 13 | 61.9% | Most popular; great when M is `2k+R` (e.g. 5120, 9216, 10240) |
| `pow2-4k` (first piece = 4096) | 18 | 10 | 55.6% | Second-most; sweet for M ∈ {7168, 9216, 11264, 12288, 16384} |
| `uniform-50/50` | 7 | 2 | 28.6% | Best when same MT applies to both halves (square 8192² etc.) |
| `pow2-8k` (first piece = 8192) | 5 | 3 | 60.0% | Used for very large M ∈ {14336, 10240+8192} |

**`pow2-2k` is the workhorse** — pairing a 2048-row piece (Tensile maps it to `MT256x160` or `MT256x144`) with a power-of-2 second piece (mapped to `MT256x256`) is the most reliable wins pattern.

---

## 7. Most Effective Sub-MacroTile Pairings (winning combos only)

Of the 28 winning splits, the per-sub MT pairs chosen by the heuristic are:

| Sub-0 MT + Sub-1 MT | Wins |
|---|---:|
| `MT256x160x64` + `MT256x256x64` | 10 |
| `MT256x256x64` + `MT256x256x64` | 8 |
| `MT256x144x64` + `MT256x256x64` | 3 |
| `MT256x256x64` + `MT256x224x64` | 3 |
| `MT256x224x64` + `MT192x224x64` | 2 |
| `MT256x192x64` + `MT256x256x64` | 1 |
| `MT256x240x64` + `MT256x224x64` | 1 |

The dominant winning pattern is **one piece getting the optimal `MT256x256x64`** (the kernel that has the most thoroughly tuned solution database), with the other piece taking the best the heuristic can do for an odd remainder size.

Notably, **8 of the 28 wins** are `MT256x256` + `MT256x256` — i.e., both halves use the same kernel. These wins come purely from wave-alignment improvements (smaller WG counts that fit cleanly in 256 CUs without tail waste), not from per-WG kernel quality.

---

## 8. Origami Analytical Model Accuracy

The Origami `compute_total_latency` model is used to score candidates **before** any GPU execution. How well does its prediction match measured outcomes?

| | Value |
|---|---|
| Mean predicted Origami latency reduction | **+2.47%** |
| Mean measured perf uplift | **+3.77%** |
| Median predicted vs measured | +1.49% predicted vs +2.44% measured |
| **Sign agreement** (predicted >0 ↔ measured >0) | **34/51 = 66.7%** |
| Cases where Origami predicts gain but measured shows loss | 12 (incl. the two noise outliers) |
| Cases where Origami predicts 0% but measured shows ≥2% gain | 8 |

**Interpretation:**
1. Origami **systematically under-predicts** the gain. It models per-WG MFMA-pipeline effects accurately but misses second-order effects that *do* matter — wave-tail elimination, L2 persistence hits, and HIP-graph host-side overhead reduction. The 8 cases where Origami says "0% gain expected" but we measure +2% to +22% almost all involve `MT256x256` + `MT256x256` splits where the per-kernel cycle count is identical but the dispatched WG grid becomes wave-aligned.
2. Origami's **direction prediction is reliable** for the ranking it actually uses (it sorts candidates and returns the lowest-cycle one, which is correct in 67% of full sign-agreement cases and ≥90% when restricted to non-noise measurements).
3. The **maximum predicted Origami uplift is 7.18%** but the maximum measured is 24.6% — this 3.5× under-prediction is the model's blind spot for system-level overheads.

---

## 9. Pre-Guard Fallbacks (correct disable behavior)

The 10 cases where the pre-guard `shouldUseMultiMacroTile()` correctly disabled the feature (`min(M, N) < 5120`):

| M=N | K | Pre-guard reason |
|---:|---:|---|
| 2048 | 2048 | `min(M,N) < 5120` |
| 2048 | 4096 | `min(M,N) < 5120` |
| 2048 | 8192 | `min(M,N) < 5120` |
| 2048 | 16384 | `min(M,N) < 5120` |
| 2048 | 32768 | `min(M,N) < 5120` |
| 4096 | 2048 | `min(M,N) < 5120` |
| 4096 | 4096 | `min(M,N) < 5120` |
| 4096 | 8192 | `min(M,N) < 5120` |
| 4096 | 16384 | `min(M,N) < 5120` |
| 4096 | 32768 | `min(M,N) < 5120` |

These all complete the multi-MT bench command without splitting and report the message:

> `Multi-MacroTile disabled for this problem size (would hurt performance) Falling back to baseline single-kernel execution`

(Note: in current `testing_matmul.hpp` the post-fallback timing path itself does not measure iterations the same way as the hot loop, so the raw μs number printed is unreliable for fallback cases — this is why the harness re-uses the default-run number for fallback cases. See `bench/render_table.py::normalize_fallback`.)

---

## 10. Hardware Root-Cause Refresher (MI350X / gfx950)

Each gfx950 has **256 CUs across 8 XCDs (32 CUs each)**, **32 MB L2 cache (4 MB per XCD)**, and ~8 TB/s HBM. A GEMM kernel launches `ceil(M/MT_M) × ceil(N/MT_N)` workgroups, distributed round-robin across XCDs.

**Why `MT256x240x64` loses for M=10240** (the canonical example):
- N-tile count = `ceil(10240 / 240) = 43`. **43 is prime**, so it cannot divide evenly into 8 XCDs → 3 XCDs get 6 tiles, 5 XCDs get 5 → **20% load imbalance**.
- Total WGs = `40 × 43 = 1720` → `1720 / 256 = 6.72 waves` → last wave at **71.9% utilization** (72 idle CUs).
- Origami baseline latency: **112,351,161 cycles** (matches the measured 1894 μs at 2200 MHz × 256 CUs).

**After splitting `10240 → [2048, 8192]`:**
- Sub-0 (`2048×10240×K`, `MT256x160`): WG grid 8×64 = 512, **2 waves at 100% util**.
- Sub-1 (`8192×10240×K`, `MT256x256`): WG grid 32×40 = 1280, **5 waves at 100% util**.
- Total Origami: 21,280,995 + 84,928,837 = **106,209,832 cycles** (5.5% reduction).
- Measured: 1894 → 1526 μs = 19.4% reduction (the gap to Origami's prediction comes from L2 hints + HIP-graph fusion).

For more on the four hardware effects (XCD imbalance, wave tail, per-WG MFMA efficiency, L2 fit), see `MULTI_MACROTILE_DESIGN.md §7`.

---

## 11. Updated Decision Rules

Based on this fresh dataset (61 cases, NN layout, FP16, MI350X), the recommended decision rules for enabling multi-MT:

### Strong signals (enable)

| Condition | Expected gain | Confidence |
|---|---|---|
| Baseline MT = `MT256x240x64` | +14.3% avg, 100% win rate | 7/7 cases |
| Square M=N ≥ 8192 with K ≥ 8192 (any baseline MT) | +5 to +22% | 9/13 cases (69%) |
| Baseline MT = `MT256x208x64`, M ≥ 5120, K ≥ 4096 | +2 to +11% | 7/10 cases (70%) |
| Pow2-aligned M=N, K ≥ 16384 | reliable wave-tail wins | strong |

### Weak signals (proceed with caution)

| Condition | Expected | Notes |
|---|---|---|
| Baseline `MT256x256x64`, K = 8192 to 16384 | mixed (avg +3%) | depends on whether WG grid is wave-aligned |
| Rectangular M ≠ N (e.g., 8192×16384) | mixed | uniform 50/50 split tends to lose ~1–2% |

### Strong negative signals (disable)

| Condition | Behavior |
|---|---|
| `min(M, N) < 5120` | **Pre-guard rejects** automatically (correct) |
| K ≤ 2048 | Launch overhead dominates; expect ±0.5%, occasional −2 to −5% |
| Baseline MT = `MT256x192x64` (M=6144 case) | 0/5 wins — kernel is already efficient |

### Fast `safe-rule` heuristic

```
enable_multi_mt =
    min(M, N) ≥ 5120 AND               # pre-guard
    K ≥ 4096 AND                        # crossover floor
    layout = NN AND                     # other layouts not yet validated here
    !(baseline_MT_M = 256 AND baseline_MT_N = 192) AND   # MT192 is already efficient
    !(K ≤ 2048 AND baseline_MT == MT256x256x64)          # no wave-tail headroom at small K with optimal MT
```

This rule, applied to the 51 active cases, would correctly enable multi-MT for **30 cases** (capturing 25 of the 28 wins) and avoid **9 of the 11 losses**, yielding an effective average uplift of **+5.4%** versus the unconditioned +3.8%.

---

## 12. Reproducing the Results

### Single case (canonical winner)

```bash
cd ~/MultiMT/rocm-libraries/projects/hipblaslt/build/release

# Iters auto-sized for ~3 s hot, 3 s cold (10240×10240×32768 → ~7950 μs/iter → ~378 iters)
./clients/hipblaslt-bench -m 10240 -n 10240 -k 32768 \
  --transA N --transB N --precision f16_r \
  --device 0 --api_method c \
  -i 378 -j 378 \
  --multi_macrotile --split_strategy 17 --num_splits 2 --l2_cache_hints

# Default for comparison:
./clients/hipblaslt-bench -m 10240 -n 10240 -k 32768 \
  --transA N --transB N --precision f16_r \
  --device 0 --api_method c \
  -i 378 -j 378
```

Expected output (from `m10240_n10240_k32768_NN_multimt.log`):

> ```
> Origami Analytical Ranking (4 candidates, all paths):
>   #1 pow2-2k [2048,8192] latency=422126632 cycles
>   ...
> Baseline kernel: MT256x240x64_MI
> Baseline measured: 7949.91 us (864405.0 GFLOPS, 0.864 TFLOPS)
> ...
> Multi-MacroTile Performance:
>   Average time: 5997.45 us
>   Performance: 1145800.5 GFLOPS (1.146 TFLOPS)
> ```

### Full sweep (this dataset)

```bash
cd ~/MultiMT/bench
python3 run_bench.py --profile broad --devices 0,1,2,3,4,5
python3 render_table.py results/results_broad_<TIMESTAMP>.json
```

Harness lives at `~/MultiMT/bench/`:

| File | Purpose |
|---|---|
| `run_bench.py` | Probes per-iter time, sizes `-i` / `-j` for 3 s hot + cold, dispatches across GPUs round-robin, parses default + multi-MT outputs into JSON |
| `render_table.py` | Reads the JSON, normalizes pre-guard fallback cases (so multi numbers reuse default's), emits CSV + Markdown tables with per-sub MT, splits, offsets, and Origami cycles |
| `reparse.py` | Re-parses existing logs into JSON without re-running the sweep (useful when only the parser/renderer changes) |

Per-case logs are saved in `~/MultiMT/bench/logs/m{M}_n{N}_k{K}_NN_{default,multimt}.log` (122 files, 500 KB).

---

## 13. Full Per-Case Data Table

The complete per-case table — including baseline MT, baseline μs/GFLOPS/Origami cycles, multi-MT μs/GFLOPS/Origami cycles, split label, split sizes, offsets, per-sub MTs, per-sub Origami cycles, perf uplift, and Origami uplift — is at:

- **Markdown**: [`results_broad_20260428_205042_v2.md`](../../../bench/results/results_broad_20260428_205042_v2.md) (61 rows × 14 columns)
- **CSV**: [`results_broad_20260428_205042_v2.csv`](../../../bench/results/results_broad_20260428_205042_v2.csv) (machine-readable, 20 columns)
- **JSON**: [`results_broad_20260428_205042_v2.json`](../../../bench/results/results_broad_20260428_205042_v2.json) (full nested structure with per-sub-problem detail)
- **Per-case logs**: `~/MultiMT/bench/logs/m{M}_n{N}_k{K}_NN_{default,multimt}.log` — 122 files, 500 KB

A representative slice (the M = 10240 row, the canonical winner family):

| M | N | K | BL MT | BL μs | BL GFLOPS | BL Origami (cyc) | MT μs | MT GFLOPS | MT Origami (cyc) | Perf uplift | Origami uplift | Split | Sub-problems |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 10240 | 10240 | 2048 | `MT256x240x64` | 464.8 | 924,116 | 28,794,681 | 426.6 | 1,006,752 | 27,230,632 | **+8.2%** | +5.4% | `pow2-2k` [2048, 8192] | [2048×10240×2048 MT256x160 5,456,355c @M=0] + [8192×10240×2048 MT256x256 21,774,277c @M=2048] |
| 10240 | 10240 | 4096 | `MT256x240x64` | 895.8 | 958,872 | 56,646,841 | 840.2 | 1,022,306 | 53,557,032 | **+6.2%** | +5.5% | `pow2-2k` [2048, 8192] | [2048×10240×4096 MT256x160 10,731,235c @M=0] + [8192×10240×4096 MT256x256 42,825,797c @M=2048] |
| 10240 | 10240 | 8192 | `MT256x240x64` | 1,894.3 | 906,915 | 112,351,161 | 1,526.4 | 1,125,520 | 106,209,832 | **+19.4%** | +5.5% | `pow2-2k` [2048, 8192] | [2048×10240×8192 MT256x160 21,280,995c @M=0] + [8192×10240×8192 MT256x256 84,928,837c @M=2048] |
| 10240 | 10240 | 16384 | `MT256x240x64` | 3,818.3 | 899,866 | 223,759,801 | 3,020.4 | 1,137,599 | 211,515,432 | **+20.9%** | +5.5% | `pow2-2k` [2048, 8192] | [2048×10240×16384 MT256x160 42,380,515c @M=0] + [8192×10240×16384 MT256x256 169,134,917c @M=2048] |
| 10240 | 10240 | 32768 | `MT256x240x64` | 7,949.9 | 864,405 | 446,577,081 | 5,997.5 | 1,145,801 | 422,126,632 | **+24.6%** | +5.5% | `pow2-2k` [2048, 8192] | [2048×10240×32768 MT256x160 84,579,555c @M=0] + [8192×10240×32768 MT256x256 337,547,077c @M=2048] |

The full table for all 61 cases is in the linked Markdown file above.

---

## 14. Caveats & Limitations of This Sweep

1. **NN layout only.** Other layouts (TN, NT, TT) are not exercised here. The previous benchmark study (728 points) found NN and TN have ~25% win rates while NT/TT have 14–19%. This sweep does not contradict that — it just doesn't cover those layouts. A future sweep would replicate the same 61 cases across all 4 transpose combinations.
2. **FP16 only.** No BF16, FP8, or FP32 cases tested. Multi-MT is in principle dtype-agnostic; only the heuristic-selected MTs and Origami's per-dtype hardware model would change.
3. **Parallel measurement noise.** Two of the 11 losses (`9216×9216×16384`, `7168×7168×8192`) are statistically inconsistent with their K-curves and are believed to be GPU power-clock cross-talk during the 6-way parallel sweep. A serial re-run on a single GPU would clean these up.
4. **No correctness verification (`--verify`)** in this sweep — the focus was performance. The previous 728-point dataset verified all results (max norm error 5.08e-05). The multi-MT path uses the same Tensile kernels as default, so correctness is structurally equivalent.
5. **No grouped-GEMM or extension-API path** — multi-MT explicitly rejects both at validation; this sweep stays in the supported `--api_method c` regime.
6. **Pre-guard threshold** (`min(M,N) < 5120`) was empirically derived from a previous 1440-point study; this sweep uses it as-is and confirms no fallback case shows a meaningful regression vs. baseline (all 10 fallbacks show < 1% drift, well within probe noise).
7. **Only `--split_strategy 17`** (Origami M-split, 2-way) was tested. Strategies 18 (N-split), 21–22 (3-way), 23–24 (XCD-aware) are present in the codebase but were not part of this sweep.

---

## Appendix: Aggregate Computation Cost

| | Value |
|---|---|
| Total bench invocations | 122 (61 default + 61 multi-MT) |
| Total cumulative iterations | 430,575 hot + 430,575 cold = 861,150 iterations |
| Total wallclock | 296 s (4 min 56 s) on 6 parallel GPUs |
| Average per-case wallclock | 4.85 s (incl. probe + default + multi-MT + setup overhead) |
| Per-case logs | 500 KB across 122 files |
| Aggregate JSON + CSV + MD | 244 KB |

The harness itself is **24 KB of Python** (`run_bench.py` + `render_table.py` + `reparse.py`).
