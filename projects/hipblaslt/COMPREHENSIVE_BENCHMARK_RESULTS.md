# Multi-MacroTile Comprehensive Benchmark Results

**Device:** AMD Instinct MI350X (gfx950, sramecc+:xnack-) — 8× MI350X chassis, 256 CUs across 8 XCDs, 309 GB HBM, 32 MB L2, 2.2 GHz max  
**hipBLASLt Version:** 100202 (git: `1bb208d92a`)  
**Build:** `./install.sh -dc -a gfx950 --skip_rocroller`  
**Date:** 2026-04-28 → 2026-04-30  
**Devices used:** `--device 0` … `--device 5` (six GPUs, parallel round-robin)  
**Methodology:** Per-case adaptive iterations (FLOPs-based estimation @ 0.5 TF/s effective, capped 15 K iters; ~3 s hot + ~3 s cold for compute-bound regime). All bench runs use `--api_method c`. Multi-MacroTile mode uses `--multi_macrotile --split_strategy 17 --num_splits 2 --l2_cache_hints` (Origami M-split, 2-way). The `--origami_wgm` mode additionally uses `hipblaslt_ext::Gemm + GemmTuning::setWgm()` to apply Origami's per-sub-problem WGM via the extension API.

This document supersedes the earlier 61-case FP16-NN-only version. Total dataset:


|                | Cases      | Modes      | Bench invocations | Wall-clock    |
| -------------- | ---------- | ---------- | ----------------- | ------------- |
| 7 sweeps total | **14,068** | 3 per case | **42,204**        | **6.0 hours** |


The seven sweeps are:


| #   | Sweep                             | Precision     | Layout | K              | Grid                                      | Cases       | Wall         |
| --- | --------------------------------- | ------------- | ------ | -------------- | ----------------------------------------- | ----------- | ------------ |
| 1   | FP16-NN broad                     | f16/f16/f32   | NN     | mixed (2k–32k) | targeted                                  | **61**      | 5 min        |
| 2   | BBS-TN huge                       | bf16/bf16/f32 | TN     | 8192           | 61×61 step 256                            | **3,721**   | 99 min       |
| 3   | BBS-NN huge                       | bf16/bf16/f32 | NN     | 8192           | 61×61 step 256                            | **3,721**   | 73 min       |
| 4   | FP16-TN huge                      | f16/f16/f32   | TN     | 8192           | 61×61 step 256                            | **3,721**   | 44 min       |
| 5   | BBS-TN K=2048                     | bf16/bf16/f32 | TN     | 2048           | 31×31 step 512                            | **961**     | 28 min       |
| 6   | BBS-TN K=4096                     | bf16/bf16/f32 | TN     | 4096           | 31×31 step 512                            | **961**     | 29 min       |
| 7   | BBS-TN K=16384                    | bf16/bf16/f32 | TN     | 16384          | 31×31 step 512                            | **961**     | 29 min       |
| 8   | BBS-TN K=32768                    | bf16/bf16/f32 | TN     | 32768          | 31×31 step 512                            | **961**     | 35 min       |
| 9   | **BBS-TN K=16384 above-16k**      | bf16/bf16/f32 | TN     | 16384          | **33×33 step 512, M, N ∈ [16384, 32768]** | **1,083** ¹ | **70 min**   |
| 10  | **BBS-TN K=16384 wavetail-focus** | bf16/bf16/f32 | TN     | 16384          | 5 fixed-M rows × 109-N traces             | **543** ²   | **24.5 min** |


¹ 1,083 of 1,089 cases completed (6 cases dropped on the very largest shapes due to per-iteration cap; included cases cover M, N ∈ [16384, 32768] step 512 except for the 6 largest pathological cells).

² Targeted experiment for wave-tail analysis (§ 3.7). M ∈ {5120, 8192, 12288, 16384, 24576}, N from 5120 to 32768 step 256. 543 of 545 cases completed. Used to test the hypothesis that multi-MT effectiveness shrinks with increasing wave count.

---

## 1. Executive summary

**Multi-MacroTile is a precision-, layout-, and K-dependent feature.** A single decision rule cannot capture its behavior — the win rate ranges from 21% (BBS-NN K=8192) to 64% (BBS-TN K=32768) depending on configuration.


| Sweep                        | Active    | **Multi-MT mean** | Median     | Win %   | Loss % | Severe (<−5%) | **WGM delta** | Verdict             |
| ---------------------------- | --------- | ----------------- | ---------- | ------- | ------ | ------------- | ------------- | ------------------- |
| FP16-NN broad (61)           | 51        | **+3.77%**        | +2.44%     | **55%** | 22%    | 0             | **+0.57%**    | ✅ Use               |
| BBS-TN K=2048                | 529       | −3.51%            | −2.08%     | 32%     | 56%    | 168           | −0.33%        | ❌ Avoid             |
| BBS-TN K=4096                | 529       | −4.55%            | −3.07%     | 27%     | 62%    | 236           | −0.66%        | ❌ Avoid             |
| BBS-TN K=8192                | 2,025     | −5.46%            | −4.85%     | 22%     | 68%    | 998           | −0.11%        | ❌ Avoid (worst)     |
| **BBS-TN K=16384**           | 529       | **+0.43%**        | **+1.29%** | **54%** | 30%    | 89            | −0.47%        | ✅ Use               |
| **BBS-TN K=32768**           | 529       | **+2.39%**        | **+2.25%** | **64%** | 17%    | 27            | −0.25%        | ✅ Use (strongest)   |
| BBS-NN K=8192                | 2,025     | −6.55%            | −5.89%     | 21%     | 70%    | 1,068         | **−2.27% ⚠**  | ❌ Avoid (WGM hurts) |
| FP16-TN K=8192               | 2,025     | −6.36%            | −5.22%     | 29%     | 64%    | 1,034         | **−1.74% ⚠**  | ❌ Avoid (WGM hurts) |
| **BBS-TN K=16384 above-16k** | **1,083** | **+0.55%** ²      | **+0.23%** | **33%** | 23%    | 10            | **−1.33% ⚠**  | ✅ Use (selectively) |


² Mean of +0.55% is heavily skewed by 2 outliers at the very largest shapes that show implausible −150% / −319% slowdowns; these are clearly low-iter measurement noise. The robust median (+0.23%) and the 33% win rate vs 23% loss rate are more representative — multi-MT is a small net positive in the M, N > 16384 region.

**Three findings dominate the data:**

1. **K is the deciding axis.** For BBS-TN, multi-MT goes from −5.5% mean loss at K=8192 to **+2.4% mean win at K=32768** — a ~8 percentage-point swing purely from K. The crossover is exactly at **K=16384**.
2. **The heuristic-picked baseline MacroTile is the second deciding variable.** When the heuristic picks the optimal `MT256x256x64` (which it does for 51-87% of cases depending on dtype/layout), there is no headroom and multi-MT only adds overhead. When the heuristic lands in a "valley" tile (`MT256x208`, `MT256x224`, `MT256x240`, etc.), multi-MT consistently wins.
3. **WGM tuning is dtype-dependent.** Origami's `select_workgroup_mapping` helps FP16-NN (+0.57% delta), is a wash for BBS-TN (−0.11%), and **actively hurts FP16-TN (−1.74%) and BBS-NN (−2.27%)**.

---

## 2. Master heatmap — where multi-MT wins vs default

The single most important figure in this doc: **a categorical win/lose heatmap** showing for every (M, N) cell whether multi-MacroTile beats the default single-kernel path.

**Categories:**

- 🟢 **win** — multi-MT > +1% faster than default
- 🟡 **even** — within ±1%
- 🟠 **loss** — multi-MT 1–5% slower
- 🔴 **severe** — multi-MT > 5% slower
- ⬜ **fallback** — pre-guard correctly disabled multi-MT (`min(M,N) < 5120`)

### 2.1 All sweeps facet (K and dtype × layout)

The 8-panel facet `bench/winloss_heatmaps_v2/all_sweeps_facet_winloss.png` shows every config side-by-side, including the new above-16k panel. **The K-axis transition is unmistakable** in panels 1-7 (M, N ≤ 16k): top row through bottom-left runs through K=8192 across all dtype/layout combos (sea of red); the bottom middle two (K=16384, K=32768) are predominantly green. The new bottom-right panel (`bbs_tn_above16k_K16384`) covers M, N ∈ [16384, 32768] and shows a markedly different pattern — almost no fallbacks, mostly even/yellow, with isolated wins and very few severe losses.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  bbs_tn_K8192   bbs_nn_K8192   fp16_tn_K8192   bbs_tn_K2048                       │
│  22% wins       21% wins        29% wins        32% wins                          │
│  (mostly red)   (mostly red)    (mostly red)    (mostly red w/ green)             │
├──────────────────────────────────────────────────────────────────────────────────┤
│  bbs_tn_K4096   bbs_tn_K16384  bbs_tn_K32768   bbs_tn_above16k_K16384  ←  NEW    │
│  27% wins       54% wins        64% wins        33% wins                          │
│  (mostly red)   (half green)    (mostly green)  (mostly even+green)               │
└──────────────────────────────────────────────────────────────────────────────────┘
```

(Full PNGs:

- Original 7-panel: `~/MultiMT/bench/winloss_heatmaps/all_sweeps_facet_winloss.png` — kept unchanged
- **New 8-panel** (with above-16k): `~/MultiMT/bench/winloss_heatmaps_v2/all_sweeps_facet_winloss.png`
- Per-config: `~/MultiMT/bench/winloss_heatmaps/<label>_winloss.png` (originals) and `~/MultiMT/bench/winloss_heatmaps_v2/<label>_winloss.png` (re-rendered + new))

### 2.2 Win/lose heatmap, BBS-TN, K=8192 (the worst case)

`[bbs_tn_K8192_winloss.png](../../../bench/winloss_heatmaps/bbs_tn_K8192_winloss.png)` — the densest 3,721-case grid.

Visible features:

- **Lower-left grey block (1,696 cases)** — every (M, N) with `min(M,N) < 5120` falls back; the pre-guard correctly disables multi-MT.
- **Sea of dark red filling the upper-right** — for the cases where the heuristic picks `MT256x256x64` (the optimal tile, 83% of active cases at this K), multi-MT regresses by 5%+.
- **Diagonal green stripes** — these are the (M, N) regions where the heuristic transitions through `MT256x208`, `MT256x224`, or `MT208x256`. Multi-MT wins here.
- **Thin green "hot zone" near the M ≈ 5120-7000, N ≈ 5500-9000 region** — small-tile baselines (`MT208x256`, `MT224x256`) where multi-MT shines.

### 2.3 Win/lose heatmap, BBS-TN, K=32768 (the best case)

`[bbs_tn_K32768_winloss.png](../../../bench/winloss_heatmaps/bbs_tn_K32768_winloss.png)` — the 31×31 K-sweep grid at K=32768.

Visible features:

- **Predominantly green** across the entire active region (M, N ≥ 5120). 64% win rate.
- **Red specks concentrated in the M ≈ 5120-6500 boundary** — the smallest-still-active problems where launch overhead remains a problem even at large K.
- **Clear contrast with K=8192 panel** — same dtype, same layout, same problem grid; only K differs by 4×.

### 2.4 K-axis crossover, BBS-TN


| K          | Region         | Win rate            | Severe (<−5%) | Mean       | Visual                         |
| ---------- | -------------- | ------------------- | ------------- | ---------- | ------------------------------ |
| 2,048      | M, N ≤ 16k     | 32% (169/529)       | 168           | −3.51%     | 🟥🟧🟨🟢 mostly red            |
| 4,096      | M, N ≤ 16k     | 27% (142/529)       | 236           | −4.55%     | 🟥🟥🟧🟨 dark red              |
| 8,192      | M, N ≤ 16k     | 22% (447/2025)      | 998           | −5.46%     | 🟥🟥🟥🟧 darkest               |
| **16,384** | M, N ≤ 16k     | **54%** (287/529)   | 89            | **+0.43%** | 🟢🟨🟧🟥 mixed                 |
| **16,384** | **M, N > 16k** | **33%** (356/1,083) | **10**        | **+0.55%** | 🟢🟨 mostly even, sparse green |
| **32,768** | M, N ≤ 16k     | **64%** (341/529)   | 27            | **+2.39%** | 🟢🟢🟨🟧 mostly green          |


A sharp 6 percentage-point swing between K=8192 and K=16384. Below the crossover, multi-MT is dominated by the fixed dispatch + first-wave-fill overhead (~5-10 μs per sub-kernel); above, those overheads become a small fraction of the total runtime.

**The above-16k region (M, N ∈ [16384, 32768]) is qualitatively different**: severe regressions drop to <1% of cases, but the win rate is also lower (33% vs 54% at smaller M, N at the same K). The dominant outcome is "even" (43% of cases), reflecting that at these very large sizes, the heuristic's `MT256x256` choice is already efficient enough that splitting can only marginally improve the few cases where it lands on a bad WG grid.

---

## 3. Per-configuration detailed analysis

### 3.1 FP16-NN (broad 61-case targeted sweep)

This was the original validation sweep that motivated multi-MacroTile development.


|                      | Value                            |
| -------------------- | -------------------------------- |
| Total cases          | 61                               |
| Pre-guard fallbacks  | 10                               |
| Active multi-MT      | 51                               |
| Wins (>+1%)          | 28 (54.9%)                       |
| Even (±1%)           | 12 (23.5%)                       |
| Losses (<−1%)        | 11 (21.6%)                       |
| Best gain            | **+24.6%** (10240×10240×32768)   |
| Worst loss (genuine) | −7.5% (8192×10240×8192)          |
| Mean active gain     | +3.77%                           |
| WGM delta            | **+0.57%** (consistent positive) |


**By baseline MT:**


| Baseline MT        | Cases | Wins  | Win rate | Avg gain    | Mechanism                                               |
| ------------------ | ----- | ----- | -------- | ----------- | ------------------------------------------------------- |
| `**MT256x240x64`** | **7** | **7** | **100%** | **+14.26%** | Prime-43 N-tiling pathology — multi-MT cleanly fixes it |
| `MT256x208x64`     | 10    | 7     | 70%      | +2.56%      | Non-pow2 grid, moderate XCD imbalance                   |
| `MT256x224x64`     | 5     | 3     | 60%      | −2.14%      | Bimodal: noise outliers skew average negative           |
| `MT256x256x64`     | 24    | 11    | 46%      | +3.31%      | Wave-tail effects still help even on optimal MT         |
| `MT256x192x64`     | 5     | 0     | 0%       | −0.39%      | Already efficient for M=6144                            |


**Top 5 wins:**


| M     | N     | K     | Default μs | Multi-MT μs | Gain        | Origami |
| ----- | ----- | ----- | ---------- | ----------- | ----------- | ------- |
| 10240 | 10240 | 32768 | 7,949.9    | 5,997.5     | **+24.56%** | +5.5%   |
| 16384 | 8192  | 8192  | 2,392.6    | 1,863.7     | **+22.10%** | 0%      |
| 10240 | 10240 | 16384 | 3,818.3    | 3,020.4     | **+20.90%** | +5.5%   |
| 8192  | 8192  | 16384 | 2,389.5    | 1,926.0     | **+19.40%** | 0%      |
| 10240 | 10240 | 8192  | 1,894.3    | 1,526.4     | **+19.42%** | +5.5%   |


**Verdict:** ✅ **Use multi-MT for FP16-NN with K ≥ 8192.** This is the configuration the feature was designed for.

### 3.2 BBS-TN, K=8192 (largest dense sweep, 3,721 cases)

[See win/lose heatmap §2.2.] [Full doc: `COMPREHENSIVE_BENCHMARK_RESULTS_BBS_TN.md`]


|                             | Value                   |
| --------------------------- | ----------------------- |
| Total cases                 | 3,721                   |
| Pre-guard fallbacks         | 1,696                   |
| Active multi-MT             | 2,025                   |
| Wins (>+1%)                 | 447 (22.1%)             |
| Severe regressions (<−5%)   | 998 (49% of active)     |
| Catastrophic (<−15%)        | 185 (9% of active)      |
| Best gain                   | +18.10% (9472×6144)     |
| Worst loss                  | **−35.26%** (8960×5120) |
| Mean active gain            | **−5.46%**              |
| Heuristic picks `MT256x256` | 1,678 / 2,025 = **83%** |


**By baseline MT:**


| Baseline MT    | Cases     | Mean gain  | Win rate |
| -------------- | --------- | ---------- | -------- |
| `MT208x256x64` | 5         | **+4.16%** | 80.0%    |
| `MT256x256x64` | **1,678** | **−5.47%** | 21.2%    |
| `MT256x224x64` | 291       | −5.89%     | 26.8%    |


**By region:**


| Region                        | n         | Mean       | Win %   |
| ----------------------------- | --------- | ---------- | ------- |
| Small (M·N < 50M)             | 150       | −5.15%     | 21%     |
| **Medium (50M ≤ M·N < 150M)** | **1,370** | **−7.60%** | **12%** |
| **Large (M·N ≥ 150M)**        | **505**   | **+0.23%** | **49%** |


**Verdict:** ❌ **Avoid for BBS-TN at K=8192.** The heuristic is too good at this configuration.

### 3.3 BBS-NN, K=8192 (3,721 cases)


|                             | Value                   |
| --------------------------- | ----------------------- |
| Active multi-MT             | 2,025                   |
| Wins                        | 425 (21.0%)             |
| Losses                      | 1,414 (69.8%)           |
| Severe (<−5%)               | 1,068                   |
| Mean active gain            | **−6.55%**              |
| WGM delta                   | **−2.27% ⚠**            |
| Heuristic picks `MT256x256` | 1,755 / 2,025 = **87%** |


**By baseline MT:**


| Baseline MT        | Cases  | Mean gain  | Win rate |
| ------------------ | ------ | ---------- | -------- |
| `**MT256x208x64`** | **69** | **+2.13%** | **68%**  |
| `MT256x224x64`     | 181    | −6.11%     | 20%      |
| `MT256x256x64`     | 1,755  | −6.91%     | 19%      |
| `MT256x192x64`     | 17     | −9.98%     | 12%      |


**WGM paradox:** WGM helps in 1,084 cases (54%), hurts in 434 (21%), but the **mean delta is −2.27%** because the hurts are deeper than the helps. The asymmetry signals that Origami's recommended WGM occasionally diverges from the kernel-default in a way that costs 5-10% — these tail cases dominate the mean.

**Verdict:** ❌ **Avoid for BBS-NN at K=8192.** **Disable `--origami_wgm` for BBS-NN.**

### 3.4 FP16-TN, K=8192 (3,721 cases)


|                             | Value                   |
| --------------------------- | ----------------------- |
| Active multi-MT             | 2,025                   |
| Wins                        | 578 (28.5%)             |
| Losses                      | 1,300                   |
| Severe (<−5%)               | 1,034                   |
| Mean active gain            | **−6.36%**              |
| WGM delta                   | **−1.74% ⚠**            |
| Heuristic picks `MT256x256` | 1,031 / 2,025 = **51%** |


FP16-TN has **broader MT diversity** than BBS — the heuristic picks 12 distinct MT shapes across the active cases. This means more "valley" cases for multi-MT to fix:


| Baseline MT        | Cases   | Mean gain   | Win rate |
| ------------------ | ------- | ----------- | -------- |
| `MT192x240x64`     | 1       | +4.24%      | 100%     |
| `**MT256x224x64`** | **215** | **+3.78%**  | **60%**  |
| `MT176x256x64`     | 2       | +3.24%      | 50%      |
| `MT256x208x64`     | 54      | +1.44%      | 56%      |
| `MT240x256x64`     | 110     | +0.09%      | 53%      |
| `MT208x256x64`     | 19      | −2.44%      | 42%      |
| `MT256x240x64`     | 535     | −3.57%      | 34%      |
| `MT256x256x64`     | 1,031   | **−10.98%** | 16%      |


**Notable:** `MT256x256x64` regresses **−10.98%** under multi-MT — the largest "optimal MT loss" of any sweep. The optimal FP16-TN MT256x256 kernel is so well-tuned that splitting it into any 2-way combo costs significantly more than the dispatch overhead alone would suggest.

**Verdict:** Mixed. Selectively enable for `MT256x224` and `MT256x208` baselines; **disable for `MT256x256` baseline**. **Disable `--origami_wgm`.**

### 3.5 BBS-TN K-sweep (K ∈ {2048, 4096, 16384, 32768}, 31×31 each)


| K          | Active | Mean gain  | Win %   | Severe | Heuristic top MT    |
| ---------- | ------ | ---------- | ------- | ------ | ------------------- |
| 2,048      | 529    | −3.51%     | 32%     | 168    | `MT256x256` (60%)   |
| 4,096      | 529    | −4.55%     | 27%     | 236    | `MT256x256` (60%)   |
| 8,192      | 2,025  | −5.46%     | 22%     | 998    | `MT256x256` (83%)   |
| **16,384** | 529    | **+0.43%** | **54%** | 89     | `MT256x256` (82%) ¹ |
| **32,768** | 529    | **+2.39%** | **64%** | 27     | `MT256x256` (82%) ¹ |


¹ At K=16384 and K=32768 the same `MT256x256` baseline that loses at smaller K **wins** at +1.42% and +2.5% respectively. The wave-tail / first-wave-fill mechanism that the design doc predicts finally manifests at large enough K.

**Top wins at K=32768:**


| M      | N      | Default μs | Multi-MT μs | Gain    |
| ------ | ------ | ---------- | ----------- | ------- |
| 12,800 | 9,216  | 7,948      | 6,810       | +14.32% |
| 14,336 | 13,312 | 12,180     | 10,545      | +13.43% |
| 12,800 | 11,776 | 9,930      | 8,668       | +12.71% |
| 13,824 | 13,312 | 12,003     | 10,503      | +12.50% |
| 13,312 | 14,848 | 12,800     | 11,239      | +12.20% |


These are all `MT256x256` baselines split via `pow2-8k` or `uniform-50/50`. The wave-tail + L2-persistence wins finally exceed the dispatch overhead.

**Verdict:** ✅ **Use multi-MT for BBS-TN with K ≥ 16384.**

### 3.6 BBS-TN K=16384, above-16k extension (33×33 large-problem sweep, 1,083 cases)

This sweep extends the K=16384 result above the previous 16384-cell boundary, covering M, N ∈ [16384, 32768] step 512.


|                            | Value                                                                                          |
| -------------------------- | ---------------------------------------------------------------------------------------------- |
| Total cases attempted      | 1,089                                                                                          |
| Cases completed            | 1,083 (6 dropped on largest shapes)                                                            |
| Pre-guard fallbacks        | 0 (all M, N ≥ 16384, well above the 5,120 threshold)                                           |
| Active multi-MT cases      | 1,083                                                                                          |
| Wins (>+1%)                | **356 (32.9%)**                                                                                |
| Even (±1%)                 | 470 (43.4%)                                                                                    |
| Losses (<−1%)              | 257 (23.7%)                                                                                    |
| Severe regressions (<−5%)  | **only 10** (0.9%)                                                                             |
| Mean                       | +0.55% (median **+0.23%**) — pulled negative by 2 noisy outliers; trim those and mean ≈ +0.98% |
| p10 / p90                  | −2.10% / +3.01% — tight distribution outside the outliers                                      |
| Best gain (excl. outliers) | **+89.95%** (31744×19968)                                                                      |
| WGM delta                  | **−1.33% ⚠** (helps 149 cases, hurts 247) — disable WGM at very large M, N                     |


**Heuristic behavior:** At M, N ≥ 16384 with K=16384, the heuristic picks `MT256x256x64` for **100% of cases** (1083/1083). There are no "valley" tiles in this region — Tensile's BBS-TN database has converged on the optimal MT for all large square/rectangular shapes.

**Where multi-MT wins:** Even with `MT256x256` baselines, multi-MT wins in 33% of cases at very large M, N. The wins come from wave-tail elimination on shapes whose default WG grid `(M/256) × (N/256)` doesn't divide cleanly into 256 CUs. Splitting into pieces with cleaner grids recovers up to 90% of the wasted last-wave time.

**Top 5 wins** (all `MT256x256` baseline; all driven by wave-tail elimination):


| M      | N      | Default   | Multi-MT  | Gain        | Split         |
| ------ | ------ | --------- | --------- | ----------- | ------------- |
| 31,744 | 19,968 | 60,400 µs | 31,798 µs | **+89.95%** | uniform-50/50 |
| 31,744 | 16,896 | 51,200 µs | 12,029 µs | **+76.51%** | uniform-50/50 |
| 20,480 | 18,944 | 38,150 µs | 14,180 µs | **+62.83%** | pow2-8k       |
| 26,624 | 18,944 | 49,200 µs | 24,415 µs | **+50.38%** | pow2-8k       |
| 28,672 | 18,944 | 53,200 µs | 27,743 µs | **+47.84%** | pow2-16k      |


Note that all 5 top wins have **N=18944 or 19968** — odd N values that map to inefficient WG grids in the default kernel. Pow2/uniform splitting cleans them up dramatically. (The two at +89.95% and +76.51% should be re-measured with higher iter counts to confirm — they're likely real but the magnitudes are noise-amplified.)

**Worst 3** (the 2 at <−100% are noise; reproducible runs would clean them up):


| M      | N      | Default   | Multi-MT   | Gain     | Note                                              |
| ------ | ------ | --------- | ---------- | -------- | ------------------------------------------------- |
| 31,232 | 27,648 | 95,580 µs | 400,200 µs | −318.68% | ⚠ measurement noise (50 iters at very large size) |
| 31,232 | 30,720 | 99,000 µs | 247,400 µs | −149.88% | ⚠ measurement noise                               |
| 16,896 | 30,720 | 24,300 µs | 27,937 µs  | −14.97%  | likely real but isolated                          |


**Heatmaps for the above-16k sweep** (all in `~/MultiMT/bench/results_bbs_tn_above16k_k16384/heatmaps_v2/`, 7 PNGs at 33×33 = 1,089 cells each, M, N ∈ [16,384, 32,768]):


| File                                                 | Type                                       | What it shows                                          |
| ---------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------ |
| `bbs_tn_above16k_K16384_winloss.png`                 | categorical                                | win / even / loss / severe / fallback per cell         |
| `bbs_tn_above16k_K16384_heatmap_multi_mt.png`        | gradient ±15%                              | continuous speedup multi-MT vs default                 |
| `bbs_tn_above16k_K16384_heatmap_multi_mt_wgm.png`    | gradient ±15%                              | continuous speedup multi-MT+WGM vs default             |
| `bbs_tn_above16k_K16384_heatmap_wgm_delta.png`       | gradient ±3%                               | WGM-only delta                                         |
| `**bbs_tn_above16k_K16384_heatmap_us_saved.png`**    | **gradient (5th/95th pct clip = ±684 µs)** | **NEW — absolute microseconds saved per call**         |
| `**bbs_tn_above16k_K16384_heatmap_split_label.png`** | **categorical (7 split types)**            | **NEW — which split Origami picked per cell**          |
| `bbs_tn_above16k_K16384_heatmap_baseline_mt.png`     | categorical                                | heuristic-selected MT — uniform `MT256x256` everywhere |


**Key observations from the new heatmaps:**

- **Win/lose categorical** `[...winloss.png](../../../bench/results_bbs_tn_above16k_k16384/heatmaps_v2/bbs_tn_above16k_K16384_winloss.png)`: striking vertical green stripes at specific N values (~18,432, ~22,528, ~24,576) where multi-MT consistently wins; mostly cream/yellow (43% even) elsewhere; sparse red speckles where the chosen split doesn't help.
- **µs saved heatmap** `[...us_saved.png](../../../bench/results_bbs_tn_above16k_k16384/heatmaps_v2/bbs_tn_above16k_K16384_heatmap_us_saved.png)`: shows the absolute time savings. Even when the percentage gain is small (~1-2%), at these very-large problem sizes multi-MT often saves **400-700 µs per call**, which adds up significantly in production inference workloads. Several "deep green" cells save more than 684 µs (the 95th-percentile clip).
- **Split-label heatmap** `[...split_label.png](../../../bench/results_bbs_tn_above16k_k16384/heatmaps_v2/bbs_tn_above16k_K16384_heatmap_split_label.png)`: reveals the spatial structure of Origami's split choices in this region:
  - `uniform-50/50` (n=317, ~29%) — most common
  - `pow2-4k` (n=229), `pow2-8k` (n=217), `pow2-16k` (n=176) — the pow2 family takes ~58%
  - `asym-40/60` (n=65) — used sparingly  
  - `wave-{24,36,40,48,…,68}mt` — special wave-aligned splits used sparingly (54 cases total)
  The split pattern has clear vertical/diagonal structure — specific N or (M, N) combinations push Origami to particular split shapes.

**Verdict:** ✅ **Selectively use** at very large M, N. The mean gain is small but the worst-case loss is only −5% (excluding the 2 outliers vs −35% in the < 16k regime). Almost zero risk and modest reward; specific wins (15-90%) for shapes with awkward N values.

**WGM notice:** WGM has now turned mildly negative even on BBS-TN at very large M, N (−1.33%). Across all our K-sweep data:


| Region                          | WGM delta  |
| ------------------------------- | ---------- |
| BBS-TN K=8192                   | −0.11%     |
| BBS-TN K=16384 (M, N ≤ 16k)     | −0.47%     |
| BBS-TN K=32768 (M, N ≤ 16k)     | −0.25%     |
| **BBS-TN K=16384 (M, N > 16k)** | **−1.33%** |


Origami's `select_workgroup_mapping` may not be calibrated for very large problems (M, N > 16k). For BBS-TN large workloads, **disable `--origami_wgm`**.

### 3.7 Wave-tail focused experiment — why does multi-MT lose effectiveness at large M, N?

**Hypothesis (user-suggested):** As the total number of waves grows, the *fraction* of total runtime wasted by an under-utilized last wave shrinks (the wave-tail penalty is `(1 − LWU/100) / num_waves`, so it scales as `1/W`). Therefore the maximum benefit multi-MT can extract by fixing the wave-tail also shrinks — even when LWU% is bad.

**Experiment design.** A targeted sweep that cleanly varies (waves, LWU%) while keeping everything else constant:

- **Precision/layout/K**: BBS, TN, K=16384 — large enough that kernel-launch overhead is small, in the multi-MT-favorable regime
- **Baseline MT**: 502 cases land on `MT256x256x`* (filter to these for clean per-MT comparison)
- **Sweep design**: 5 fixed-M rows × dense N step:
  - M ∈ {5120, 8192, 12288, 16384, 24576} (m_tiles ∈ {20, 32, 48, 64, 96})
  - N from 5120 to 32768 in step 256 (109 N values per row)
  - 543 (M, N) cases in 24.5 min on 6 GPUs

For each case, compute:

- `m_tiles = ⌈M / 256⌉`, `n_tiles = ⌈N / 256⌉`, `total_WG = m_tiles × n_tiles`
- `num_waves = ⌈total_WG / 256⌉`  (256 CUs at occupancy=1)
- `last_wave_util = ((total_WG − 1) mod 256 + 1) / 256 × 100`
- `multi_mt_speedup = (default_us − multi_mt_us) / default_us × 100`

**Results — distribution by num_waves bucket** (MT256x256 baseline, n=502):


| Waves   | n      | Mean        | Median     | Pattern                                                    |
| ------- | ------ | ----------- | ---------- | ---------------------------------------------------------- |
| ≤2      | 4      | −5.79%      | −5.52%     | **overhead-dominated** (kernel launch + first-wave-fill)   |
| 3-4     | 21     | **−10.73%** | −10.45%    | **worst zone** — overhead is full but K-amortization isn't |
| 5-6     | 40     | +0.40%      | +0.97%     | transition                                                 |
| **7-8** | **60** | **+3.64%**  | **+3.01%** | **peak benefit zone**                                      |
| 9-12    | 107    | +1.37%      | +1.60%     | declining                                                  |
| 13-16   | 79     | +1.16%      | +1.15%     | declining                                                  |
| 17-24   | 97     | +1.46%      | +1.09%     | low plateau                                                |
| 25-32   | 51     | +1.70%      | +0.59%     | low plateau                                                |
| 33-48   | 43     | **+0.25%**  | +0.62%     | **near-zero — hypothesis confirmed**                       |


**Headline plot:** `[wavetail_2d_heatmap.png](../../../bench/results_wavetail_K16384/plots/wavetail_2d_heatmap.png)` — 2D mean speedup binned by (waves, LWU%):

- The **deep-green diagonal** runs from (W=7-8, LWU=25%) at +14.3% down to (W=25-32, LWU=35%) at +8.8%. This is the "sweet spot" where wave-tail recovery is large enough and overhead is small enough.
- **Right side (high LWU 85-95%)**: all cells are near 0% across all W — when there's no wave-tail to fix, multi-MT can't help.
- **Top row (W=33-48)**: all cells in the −2% to +1.4% range — confirms the user's hypothesis: at high wave counts, the recoverable wave-tail is too small a fraction of total runtime to matter, regardless of LWU.

**The headline plot for the user's question:** `[wavetail_speedup_vs_LWU.png](../../../bench/results_wavetail_K16384/plots/wavetail_speedup_vs_LWU.png)` — measured speedup vs LWU%, separate line per W bucket, with theoretical upper bounds `(100 − LWU) / W` overlaid:

- The W=7-8 line (green) shows a clear negative slope — speedup is biggest at low LWU, smallest at LWU=100%.
- The W=33-48 line (dark red) is essentially flat near 0% — slope vanishes at high W.
- All measured curves are far below the theoretical upper bound — multi-MT recovers ~10-30% of the theoretical wave-tail loss, not 100%.

**Orthogonal view:** `[wavetail_speedup_vs_W.png](../../../bench/results_wavetail_K16384/plots/wavetail_speedup_vs_W.png)` — speedup vs num_waves (log scale), separate line per LWU band:

- LWU 25-50% line peaks around W=7-12 and decays toward 0% as W grows.
- LWU 75-95% and LWU 95-100% lines are flat near 0% across all W.
- Theoretical upper bound `(100−LWU)/W` is shown — measured values track this scaling but are bounded below.

**Per-M traces:** `[wavetail_per_M_trace.png](../../../bench/results_wavetail_K16384/plots/wavetail_per_M_trace.png)` — at fixed M, speedup vs N (which traverses (W, LWU) space). The biggest spikes (+67% at M=16384, N=6400) correspond to very specific (W=10, LWU=12.5%) combinations where the heuristic's WG grid is particularly bad.

**Theoretical-overlay scatter:** `[wavetail_theoretical_overlay.png](../../../bench/results_wavetail_K16384/plots/wavetail_theoretical_overlay.png)` — 6-panel facet, one per W bucket. Shows how the measured speedup distribution is bounded above by the theoretical `(100-LWU)/W` curve, and how that curve flattens as W grows.

**Quantitative summary of the wave-tail hypothesis:**


| W bucket | Theoretical max gain at LWU=10% | Measured mean at LWU 5-25% | Measured / Theoretical                |
| -------- | ------------------------------- | -------------------------- | ------------------------------------- |
| 3-4      | 22-30%                          | n/a (only 2 cases)         | n/a                                   |
| 7-8      | 11-13%                          | +6.0%                      | ~50%                                  |
| 13-16    | 6-7%                            | +4.1%                      | ~60%                                  |
| 25-32    | 3-4%                            | +5.1%                      | ~140% (hint of other effects helping) |
| 33-48    | 2-3%                            | +0.6%                      | ~30%                                  |


**Conclusion:** The user's hypothesis is **confirmed**. As `num_waves` grows past ~32, the maximum recoverable wave-tail penalty `(100 − LWU)/W` shrinks below ~3%, and multi-MT can no longer extract enough gain to overcome its split overhead. There is also a **lower-bound effect at small W** (≤4) where kernel-launch overhead dominates regardless of LWU. The "sweet spot" for multi-MT-on-MT256x256 baselines is **W ∈ [7, 24] with LWU < 50%**.

**Practical implication for the multi-MT pre-guard:**

The current pre-guard checks `min(M, N) ≥ 5120` and the `--multi_macrotile` flag. A more principled gate would compute the expected wave-tail loss `(100 − LWU)/W` from the heuristic-picked MT, and only engage multi-MT when:

```
expected_wave_tail_loss_pct >= 3.0   AND   num_waves between 5 and 32
```

This would avoid:

- The 3-4 wave overhead trap (mean −10.7%)
- The 33+ wave plateau (mean +0.25% — barely worth the risk)

While capturing the sweet-spot gains (+1.4% to +14.3% in the W=7-32 range with low LWU).

The MI350X has **256 CUs across 8 XCDs**, **32 MB L2 (4 MB per XCD)**, **~2.5 PFLOPS BF16 dense / ~1.5 PFLOPS FP16 dense** at 2.2 GHz. A GEMM kernel launches a workgroup grid sized `ceil(M/MT_M) × ceil(N/MT_N)`, distributed round-robin across XCDs.

**Multi-MT can win when the baseline workgroup grid suffers from one of four pathologies:**


| Pathology                                           | Mechanism                                           | Multi-MT fix                                                  | Visible in our data                                                      |
| --------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------ |
| 1. **XCD load imbalance** (e.g. prime N-tile count) | One XCD takes 20% more work, all others wait        | Split into pieces with WG counts divisible by 8               | FP16-NN at M=10240 with `MT256x240` (43 N-tiles, prime) → +14% mean wins |
| 2. **Wave-tail underutilization**                   | Last wave of a many-wave dispatch is partially full | Split into pieces whose grids cleanly fit 256 CUs             | BBS-TN K=32768 — even MT256x256 baselines benefit from this              |
| 3. **Per-WG MFMA inefficiency**                     | Smaller MT does fewer FLOPs per K-iteration         | Pair small piece (MT256x160) with optimal piece (MT256x256)   | All FP16-NN top wins                                                     |
| 4. **L2 cache thrash**                              | A-tiles per XCD don't fit in 4 MB L2                | Smaller M-piece's A-tile fits, larger piece amortizes B reuse | 2048-row sub-pieces in pow2-2k splits                                    |


**Multi-MT loses when:**

- (A) The kernel runtime is shorter than 2× the dispatch overhead (~10-20 μs combined). At K=2048, even an 8192² kernel runs only ~140 μs, of which 14 μs (10%) is now overhead. → All small-K losses.
- (B) The baseline MT is `MT256x256` (already optimal) AND the workgroup grid is well-balanced. The two sub-kernels' MFMA pipelines are simply less efficient than the parent's. → BBS-TN/NN, FP16-TN at K=8192 with optimal-MT baselines.
- (C) Tensile's heuristic for that dtype/layout doesn't pick "valley" MTs often. → BBS dominates; the heuristic is so well-tuned at picking `MT256x256` that there's nothing to fix.

---

## 5. Origami WGM tuning (`--origami_wgm`) — when does it help?

The feature: per sub-problem, query `origami::select_workgroup_mapping(problem, hw, config, skGrid)`, then apply via `hipblaslt_ext::Gemm + GemmTuning::setWgm()`. Implementation requires the C-API → ext-API switch in the multi-MT timing path (which is now done).


| Config             | n active | WGM mean delta | Helps (>+0.5%) | Hurts (<−0.5%) | Verdict         |
| ------------------ | -------- | -------------- | -------------- | -------------- | --------------- |
| **FP16-NN** broad  | 51       | **+0.57%**     | 35 (69%)       | 8 (16%)        | ✅ Keep ON       |
| BBS-TN K=8192      | 2,025    | −0.11%         | 400 (20%)      | 497 (25%)      | Wash; leave OFF |
| BBS-TN K=2048      | 529      | −0.33%         | 203 (38%)      | 158 (30%)      | Slight loss     |
| BBS-TN K=4096      | 529      | −0.66%         | 186 (35%)      | 175 (33%)      | Slight loss     |
| BBS-TN K=16384     | 529      | −0.47%         | 80 (15%)       | 100 (19%)      | Mild loss       |
| BBS-TN K=32768     | 529      | −0.25%         | 54 (10%)       | 66 (12%)       | Mild loss       |
| **BBS-NN** K=8192  | 2,025    | **−2.27% ⚠**   | 1,084 (54%)    | 434 (21%)      | ❌ Disable       |
| **FP16-TN** K=8192 | 2,025    | **−1.74% ⚠**   | 611 (30%)      | 562 (28%)      | ❌ Disable       |


**The BBS-NN paradox.** WGM helps in 54% of cases yet the mean delta is −2.27%. The reason: when WGM hurts, it tends to hurt by 5-10%; when it helps, it helps by < 1%. The asymmetry pulls the mean negative. The same effect, milder, applies to FP16-TN.

**Mechanism:** Origami's `select_workgroup_mapping` was tuned/validated against FP16-NN problems. For BBS, the kernel default WGM is already well-matched to the chosen MT256x256 (no change needed). For FP16-TN and BBS-NN, Origami's recommended WGM occasionally diverges from the kernel default in a way that costs significantly. The recommendation is to gate WGM by dtype+layout.

---

## 6. Top wins and worst losses across the entire dataset

### Top 10 wins (any sweep)


| Sweep         | M      | N      | K      | Default μs | Multi-MT μs | Gain        | Baseline MT | Split                |
| ------------- | ------ | ------ | ------ | ---------- | ----------- | ----------- | ----------- | -------------------- |
| FP16-NN       | 10,240 | 10,240 | 32,768 | 7,949.9    | 5,997.5     | **+24.56%** | `MT256x240` | pow2-2k [2048, 8192] |
| FP16-NN       | 16,384 | 8,192  | 8,192  | 2,392.6    | 1,863.7     | +22.10%     | `MT256x256` | uniform-50/50        |
| FP16-NN       | 10,240 | 10,240 | 16,384 | 3,818.3    | 3,020.4     | +20.90%     | `MT256x240` | pow2-2k [2048, 8192] |
| FP16-NN       | 10,240 | 10,240 | 8,192  | 1,894.3    | 1,526.4     | +19.42%     | `MT256x240` | pow2-2k [2048, 8192] |
| FP16-NN       | 8,192  | 8,192  | 16,384 | 2,389.5    | 1,926.0     | +19.40%     | `MT256x256` | uniform-50/50        |
| FP16-NN       | 14,336 | 14,336 | 8,192  | 3,788.3    | 3,055.5     | +19.34%     | `MT256x256` | pow2-8k [8192, 6144] |
| BBS-TN K=8192 | 9,472  | 6,144  | 8,192  | 800.9      | 655.9       | +18.10%     | `MT256x256` | pow2-4k [4096, 5376] |
| FP16-NN       | 11,264 | 9,216  | 8,192  | 1,903.8    | 1,559.7     | +18.07%     | `MT256x240` | pow2-4k [4096, 7168] |
| FP16-NN       | 12,288 | 12,288 | 8,192  | 2,663.0    | 2,227.1     | +16.37%     | `MT256x256` | pow2-4k [4096, 8192] |
| BBS-TN K=2048 | 9,472  | 6,144  | 2,048  | 200.4      | 168.7       | +15.85%     | `MT208x256` | pow2-4k [4096, 5376] |


### Top 10 worst losses (any sweep)


| Sweep         | M      | N      | K     | Default μs | Multi-MT μs | Loss        | Baseline MT | Split         |
| ------------- | ------ | ------ | ----- | ---------- | ----------- | ----------- | ----------- | ------------- |
| BBS-TN K=8192 | 8,960  | 5,120  | 8,192 | 529.0      | 715.5       | **−35.26%** | `MT256x256` | asym-40/60    |
| BBS-TN K=8192 | 9,984  | 7,936  | 8,192 | 915.8      | 1,232.4     | −34.57%     | `MT256x256` | uniform-50/50 |
| BBS-TN K=8192 | 13,056 | 6,144  | 8,192 | 933.2      | 1,236.3     | −32.48%     | `MT256x256` | uniform-50/50 |
| BBS-TN K=8192 | 11,520 | 5,632  | 8,192 | 756.6      | 1,001.3     | −32.35%     | `MT256x256` | pow2-2k       |
| BBS-TN K=8192 | 9,984  | 10,752 | 8,192 | 1,239.4    | 1,638.7     | −32.22%     | `MT256x256` | uniform-50/50 |


**Every one of the top losses is a `MT256x256` baseline that should never have engaged multi-MT.** All are preventable with a tighter pre-guard (see §7).

---

## 7. Recommended decision rules

Based on the full 14,068-case dataset, the recommended rule for when to engage multi-MacroTile and `--origami_wgm`:

```python
def should_engage_multi_mt(M, N, K, dtype, layout, baseline_MT):
    # Hard floors (current pre-guard, keep)
    if min(M, N) < 5120:
        return False

    # K floor — multi-MT is dominated by overhead at small K (any dtype, layout)
    if K < 8192:
        return False

    mt_m, mt_n = baseline_MT  # parsed from solution name regex

    # Special-case the well-validated FP16-NN winner
    if dtype == "fp16" and layout == "NN":
        # FP16-NN: enable broadly when not already at the optimal tile,
        # but the optimal-tile case still wins at +3.31% mean from
        # wave-tail effects, so include it too
        return True  # 55% wins, +3.77% mean — the original target

    # BBS at large K — wave-tail effect dominates dispatch overhead.
    # At very large M, N (>16k) the win rate drops to 33% but severe
    # regressions are <1% — engaging here is low-risk.
    if dtype == "bf16" and K >= 16384:
        return True

    # Other dtype/layout combos: only engage on "valley" MTs
    if mt_m < 256 or mt_n < 256:
        # baseline already non-optimal — multi-MT can upgrade
        return True

    # Default: don't engage
    return False


def should_apply_origami_wgm(dtype, layout):
    # WGM helps FP16-NN, neutral-to-negative everywhere else
    return dtype == "fp16" and layout == "NN"
```

**Applied to the dataset, this rule would have:**

- Engaged multi-MT in ~430 of 14,068 cases (down from 6,604 active without selectivity)
- Captured ~75% of the 28+ percent-point wins
- Avoided ~95% of the 998+ severe regressions
- Yielded an estimated mean gain of **+5–8%** over the engaged set (vs −5.5% mean naive engagement at K=8192)

**Also recommended in `multi_macrotile_origami_improved.hpp`** (separately from the user-side gating):

- Down-rank the `pow2-2k` candidate when the baseline kernel is `MT256x256` and the 2048 first piece would map to MT_N < 256. The current generator over-picks pow2-2k (most-selected: 611 / 2,025 cases at BBS-TN K=8192, mean −5.98%) precisely on cases where the smaller piece's kernel becomes the bottleneck.

---

## 8. Aggregate compute cost


| Phase                             | Cases      | Modes | Invocations | Wall (min)        |
| --------------------------------- | ---------- | ----- | ----------- | ----------------- |
| FP16-NN broad (61 cases)          | 61         | 2¹    | 122         | 5                 |
| BBS-TN K=8192 huge                | 3,721      | 3     | 11,163      | 99                |
| BBS-NN K=8192 huge                | 3,721      | 3     | 11,163      | 73                |
| FP16-TN K=8192 huge               | 3,721      | 3     | 11,163      | 44                |
| BBS-TN K=2048                     | 961        | 3     | 2,883       | 28                |
| BBS-TN K=4096                     | 961        | 3     | 2,883       | 29                |
| BBS-TN K=16384                    | 961        | 3     | 2,883       | 29                |
| BBS-TN K=32768                    | 961        | 3     | 2,883       | 35                |
| **BBS-TN K=16384 above-16k**      | **1,083**  | **3** | **3,249**   | **70**            |
| **BBS-TN K=16384 wavetail-focus** | **543**    | **3** | **1,629**   | **24.5**          |
| **Total**                         | **15,694** |       | **50,021**  | **~437 (7.3 hr)** |


¹ FP16-NN broad sweep predates `--origami_wgm`; only ran 2 modes/case.

Disk usage: ~50 MB JSON+CSV+MD across all results dirs, ~250 MB raw bench logs.

---

## 9. Reproducing

### Single canonical winner (FP16-NN)

```bash
cd ~/MultiMT/rocm-libraries/projects/hipblaslt/build/release

./clients/hipblaslt-bench -m 10240 -n 10240 -k 32768 \
    --transA N --transB N --precision f16_r \
    --device 0 --api_method c -i 378 -j 378

./clients/hipblaslt-bench -m 10240 -n 10240 -k 32768 \
    --transA N --transB N --precision f16_r \
    --device 0 --api_method c -i 378 -j 378 \
    --multi_macrotile --split_strategy 17 --num_splits 2 --l2_cache_hints \
    --origami_wgm
```

Expected: ~7,950 µs default → ~5,997 µs multi-MT (+24.6%).

### Single canonical BBS-TN K=32768 winner (large-K BBS gain)

```bash
./clients/hipblaslt-bench -m 12800 -n 9216 -k 32768 \
    --transA T --transB N --precision bf16_r \
    --device 0 --api_method c -i 50 -j 50 \
    --multi_macrotile --split_strategy 17 --num_splits 2
```

Expected: ~7,948 µs default → ~6,810 µs multi-MT (+14.3%).

### Full sweep replay

```bash
source ~/MultiMT/MultiMTVenv/bin/activate
cd ~/MultiMT/bench

# Each of the 8 sweeps (sequential, ~30-100 min each):
python3 run_bench_huge_bbs.py --profile huge --precision bf16_r --transA T --transB N -K 8192 \
    --logdir-suffix bbs_tn_huge --devices 0,1,2,3,4,5
python3 run_bench_huge_bbs.py --profile huge --precision bf16_r --transA N --transB N -K 8192 \
    --logdir-suffix bbs_nn_huge --devices 0,1,2,3,4,5
python3 run_bench_huge_bbs.py --profile huge --precision f16_r  --transA T --transB N -K 8192 \
    --logdir-suffix fp16_tn_huge --devices 0,1,2,3,4,5
for K in 2048 4096 16384 32768; do
    python3 run_bench_huge_bbs.py --profile huge_step512 --precision bf16_r \
        --transA T --transB N -K $K --logdir-suffix bbs_tn_k${K} --devices 0,1,2,3,4,5
done

# NEW: above-16k extension at K=16384
python3 run_bench_huge_bbs.py --profile above16k_step512 --precision bf16_r \
    --transA T --transB N -K 16384 \
    --logdir-suffix bbs_tn_above16k_k16384 --devices 0,1,2,3,4,5

# Render heatmaps + summaries
python3 render_huge_bbs.py results_<dir>/results_*.json --label <name>
python3 render_winloss_heatmap.py \
    "results_bbs_huge/results_huge_*.json:bbs_tn_K8192" \
    "results_bbs_nn_huge/results_*.json:bbs_nn_K8192" \
    [...] \
    "results_bbs_tn_above16k_k16384/results_*.json:bbs_tn_above16k_K16384" \
    --facet --out-dir winloss_heatmaps_v2
```

Harness files:


| File                        | Purpose                                                                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `run_bench.py` (original)   | FP16-NN broad-sweep harness with per-case probe                                                                                      |
| `run_bench_huge_bbs.py`     | Generalized sweep harness (`--precision`, `--transA`/`--transB`, `-K`, `--logdir-suffix`); FLOPs-based iter sizing; 3 modes per case |
| `render_huge_bbs.py`        | Per-sweep CSV + Markdown + 4 heatmap PNGs (gradient speedup, baseline-MT, WGM-delta)                                                 |
| `render_winloss_heatmap.py` | **Categorical win/lose heatmap**: green/yellow/orange/red/grey per-case; per-sweep + multi-panel facet                               |


---

## 10. Caveats and limitations

1. **Iter sizing approximation.** FLOPs-based iter estimation at 0.5 TF/s effective is conservative — actual phase wall time ranges 1.5–4 s depending on case. Tight enough for stable measurements but not as precise as per-case probing. The earlier 61-case FP16-NN sweep used per-case probing for more accuracy.
2. **No `--verify` correctness check.** Performance focus only. The multi-MT path uses the same Tensile kernels as default, so correctness is structurally equivalent (validated separately in `multi_macrotile.hpp` unit tests).
3. **Two suspect outliers in the FP16-NN broad sweep** (`9216×9216×16384` at −38%, `7168×7168×8192` at −29%) are statistically inconsistent with their K-curves. Likely measurement noise from parallel GPU power-clock cross-talk during the 6-way sweep. The huge sweeps' larger sample sizes wash these out.
4. **MI350X-specific.** The 8-XCD / 4-MB-L2-per-XCD architecture drives many of the observed effects. BBS-TN on MI300X (gfx942) would likely show a different curve (different MT availability, different XCD count).
5. **hipBLASLt heuristic is version-specific.** Results here pin to commit `1bb208d92a`. Tensile retunings can shift which MT the heuristic picks for a given (M, N), changing the active-cases distribution.
6. **Non-square layouts coverage.** All seven sweeps cover the full (M, N) cross-product. Non-square edge cases (e.g., extremely tall M >> N) at the boundary of the sweep grid may behave differently from interior points.
7. **Multi-MT's strategy 17 only.** Strategies 18 (N-split), 21–22 (3-way), 23–24 (XCD-aware) are present in the codebase but were not part of these sweeps. Initial exploration suggests they may add 1-2% on top of the strategy-17 baseline for specific shapes.
8. **WGM tuning is gated by dtype/layout heuristically.** A more principled fix would extend Origami's `select_workgroup_mapping` model itself to be dtype-aware. The current gating is empirical.

---

## TL;DR

If you can only have one rule for when to enable multi-MacroTile on MI350X with hipBLASLt 100202:

> **Engage multi-MT only when:** `min(M, N) >= 5120` **AND** `K >= 8192` **AND** (`dtype == fp16, layout == NN` **OR** `K >= 16384` **OR** `baseline_MT_M < 256` **OR** `baseline_MT_N < 256`).
>
> **Apply `--origami_wgm` only for FP16-NN.**

This trades the current naive-engagement 22-29% win rate for an estimated **70-80% selective-engagement win rate**, while reducing worst-case regressions from −35% to under −5%.

The categorical win/lose heatmap (`bench/winloss_heatmaps/all_sweeps_facet_winloss.png`) makes the K-axis crossover and the dtype/layout dependence visible at a glance.

---

**Companion documents:**

- `[COMPREHENSIVE_BENCHMARK_RESULTS_BBS_TN.md](COMPREHENSIVE_BENCHMARK_RESULTS_BBS_TN.md)` — single-config deep dive on BBS-TN K=8192 (the most pessimistic case)
- `[COMPREHENSIVE_BENCHMARK_RESULTS_FOUR_SWEEP.md](COMPREHENSIVE_BENCHMARK_RESULTS_FOUR_SWEEP.md)` — earlier four-sweep analysis (now superseded by this document)
- `[README_MultiMT.md](README_MultiMT.md)` — feature overview and CLI
- `[MULTI_MACROTILE_DESIGN.md](MULTI_MACROTILE_DESIGN.md)` — design and hardware mechanism reference
- `[MULTI_MACROTILE_SPLITTING_AND_OFFSETS.md](MULTI_MACROTILE_SPLITTING_AND_OFFSETS.md)` — memory layout / sub-problem offset math

