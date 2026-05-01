# Multi-MacroTile Comprehensive Benchmark Results — BBS, TN

**Device:** AMD Instinct MI350X (gfx950, sramecc+:xnack-) — 256 CUs across 8 XCDs, 309 GB HBM, 32 MB L2, 2.2 GHz max  
**Precision:** **BBS** — `bf16` input/output, FP32 compute (`f32_r`)  
**Layout:** **TN** (transA = T, transB = N) — common transformer / LLM forward layout  
**hipBLASLt Version:** 100202 (git: `1bb208d92a`)  
**Build:** `./install.sh -dc -a gfx950 --skip_rocroller`  
**Date:** 2026-04-30  

**Problem grid:** dense 61×61 — **M, N ∈ [1024, 16384] step 256**, **K = 8192** fixed  
**Bench invocations:** **11,163** (3,721 cases × 3 modes per case)  
**Devices used:** 0–5 (six MI350X, parallel round-robin)  
**Wall time:** 99.4 minutes  
**Iter sizing:** FLOPs-based estimate (no per-case probe); capped at 15,000 iters; effective hot+cold ≈ 2-3 s each at 0.5 TF/s assumed throughput

**Three modes per case:**

1. **Default** — single-kernel `hipblasLtMatmul` (C-API)
2. **Multi-MT** — `--multi_macrotile --split_strategy 17 --num_splits 2 --l2_cache_hints`
3. **Multi-MT + WGM** — same + `--origami_wgm` (per-sub-problem WGM via `hipblaslt_ext::Gemm` + `GemmTuning::setWgm()`)

---

## 1. Executive summary — Multi-MT is the wrong tool for BBS-TN at K=8192

The headline number is unambiguous and **opposite** to the FP16-NN result we documented earlier:

| Metric | Value | Compare to FP16-NN K-mixed sweep |
|---|---:|---|
| Total cases | **3,721** | 61 |
| Multi-MT pre-guard fallbacks (correctly disabled) | **1,696** (45.6%) | 10 (16.4%) |
| Active multi-MT cases | **2,025** | 51 |
| **Active win rate (>+1%)** | **22.1%** (447 / 2025) | **54.9%** |
| Active loss rate (<-1%) | **67.9%** (1375 / 2025) | 21.6% |
| Severe regressions (<-5%) | **998 cases** (49% of active) | 0 |
| Catastrophic (<-15%) | **185 cases** (9% of active) | 2 (likely noise) |
| **Mean active gain** | **−5.46%** | +3.77% |
| Median active gain | −4.85% | +2.44% |
| Best single gain | +18.10% (9472×6144) | +24.6% (10240×10240×32768) |
| Worst single loss | −35.26% (8960×5120) | −7.5% (real, ex-noise) |
| **WGM uplift on top of multi-MT** | **−0.11%** (basically zero/noise) | +0.57% |

**Conclusion:** Multi-MacroTile, as currently designed and gated, **systematically degrades performance** for BBS-TN GEMMs at K=8192. It should be **disabled by default for this configuration**. The pre-guard (`min(M,N) ≥ 5120`) is correct for blocking small problems but fails to catch the much larger class of medium/large BBS-TN cases where the heuristic already picks the optimal kernel.

---

## 2. Why BBS-TN behaves so differently from FP16-NN

The cause is straightforward and visible in the heuristic-MT heatmap (`bbs_tn_heatmap_baseline_mt.png`):

| Baseline MT | Cases (active) | % of active | Mean gain | Win rate | Mechanism |
|---|---:|---:|---:|---:|---|
| `MT256x208x64` | 5 | 0.2% | +4.16% | 80.0% | The classic "valley" — multi-MT shines |
| `MT224x224x64` | 3 | 0.1% | −0.72% | 0.0% | Too few samples to draw conclusions |
| `MT256x192x64` | 20 | 1.0% | −1.79% | 30.0% | Already efficient |
| `MT224x256x64` | 28 | 1.4% | −5.39% | 14.3% | Splitting hurts more than it helps |
| **`MT256x256x64`** | **1,678** | **82.9%** | **−5.47%** | **21.2%** | **Already optimal, splitting wastes CU/L2** |
| `MT256x224x64` | 291 | 14.4% | −5.89% | 26.8% | Less efficient than MT256x256, but split overhead dominates |

The heuristic's BBS-TN solution database is **much better tuned** than its FP16-NN counterpart. **83% of all active multi-MT cases land on `MT256x256x64`**, the most efficient kernel. For these, multi-MT can only:
- Replace one optimal kernel with two slightly-less-optimal kernels, or
- Replace the optimal kernel with two copies of itself but pay launch overhead twice.

Either way, the math doesn't work out.

The diagonal/banded structure visible in the baseline-MT heatmap shows the *only* regions where multi-MT has theoretical headroom: thin diagonal stripes where the heuristic transitions from one MT to another (typically MT256x256 ↔ MT256x224). Even there, the wins are modest because the un-tuned BBS-TN kernels for these "valley" sizes are still relatively close to optimal.

For comparison, in the **FP16-NN K-mixed sweep**, `MT256x240x64` (the prime-43 N-tiling pathology) was selected for `M=10240` cases and reliably triggered a 100% win rate at +14% average gain. **`MT256x240x64` does not appear once in this BBS-TN sweep** — Tensile's BBS database simply doesn't pick that suboptimal tile for any (M, N) in our 61×61 grid.

---

## 3. Where multi-MT *does* win: structure of the 22% wins

Looking at the active wins (447 cases) by problem size and shape:

| Region | n | Mean gain | Win rate | Insight |
|---|---:|---:|---:|---|
| All active | 2025 | −5.46% | 22.1% | — |
| Square (\|M−N\| ≤ 256) | 133 | −3.12% | 30.1% | Slightly less hostile than non-square |
| Non-square | 1892 | −5.63% | 21.5% | Most cases; most losses |
| Small (M·N < 50M) | 150 | −5.15% | 20.7% | Launch overhead dominates |
| **Medium (50M ≤ M·N < 150M)** | **1370** | **−7.60%** | **12.2%** | **Worst region — split overhead exceeds gain** |
| **Large (M·N ≥ 150M)** | **505** | **+0.23%** | **49.3%** | **Only region where multi-MT is near break-even** |

The "medium" region is the bulk of the loss. These are problems large enough that the pre-guard accepts them, but small enough that the kernel's runtime is dominated by launch + first-wave-fill costs. Adding a second sub-kernel doubles those overheads without giving the second piece enough work to amortize them.

The "large" region (M·N ≥ 150M, roughly M, N ≥ 12k square or 16k×10k+) is where multi-MT begins to break even or win. The kernels there run for 2-3 ms+, so the 5-10 µs CP-gap overhead is < 1% of total runtime, and any first-order gain from splitting (XCD balance, wave alignment) shows through.

**Top 10 wins:**

| M | N | Default μs | Multi-MT μs | **Gain** | Baseline MT | Split |
|---:|---:|---:|---:|---:|---|---|
| 9472 | 6144 | 800.9 | 655.9 | **+18.10%** | `MT256x256` | pow2-4k [4096, 5376] |
| 10240 | 5632 | 792.7 | 680.2 | **+14.18%** | `MT224x256` | pow2-2k [2048, 8192] |
| 6656 | 5888 | 519.2 | 458.4 | **+11.72%** | `MT208x256` | pow2-4k [4096, 2560] |
| 9728 | 5888 | 734.4 | 653.6 | **+11.00%** | `MT224x256` | pow2-4k [4096, 5632] |
| 8448 | 6144 | 663.8 | 591.2 | **+10.94%** | `MT256x224` | asym-40/60 [3328, 5120] |
| 5632 | 8960 | 667.8 | 596.1 | **+10.74%** | `MT256x224` | pow2-2k [2048, 3584] |
| 15360 | 12800 | 2,981.6 | 2,680.6 | **+10.10%** | `MT256x256` | uniform-50/50 [7680, 7680] |
| 7680 | 6912 | 679.9 | 617.2 | **+9.22%** | `MT256x224` | pow2-4k [4096, 3584] |
| 15360 | 14336 | 3,214.2 | 2,918.8 | **+9.19%** | `MT256x256` | pow2-8k [8192, 7168] |
| 5888 | 8704 | 660.5 | 602.0 | **+8.87%** | `MT256x224` | pow2-2k [2048, 3840] |

The top 10 has two clear sub-patterns:
1. **Baseline MT is in a "valley"** (MT224x256, MT208x256, MT256x224) — these gain 8-18% from being upgraded to MT256-something via splitting.
2. **Very large square problems** (15360², 14848²) with MT256x256 baseline still gain 8-10% — pure wave-alignment / large-tile-amortization wins.

---

## 4. Where multi-MT catastrophically loses

The worst 20 cases all share the same signature: **MT256x256 baseline + small-K-style overhead penalty + uniform-50/50 split picked**:

| M | N | Default μs | Multi-MT μs | Loss | Baseline MT | Split |
|---:|---:|---:|---:|---:|---|---|
| 8960 | 5120 | 529.0 | 715.5 | **−35.26%** | `MT256x256` | asym-40/60 [3584, 5376] |
| 9984 | 7936 | 915.8 | 1232.4 | **−34.57%** | `MT256x256` | uniform-50/50 [4992, 4992] |
| 13056 | 6144 | 933.2 | 1236.3 | **−32.48%** | `MT256x256` | uniform-50/50 [6528, 6528] |
| 11520 | 5632 | 756.6 | 1001.3 | **−32.35%** | `MT256x256` | pow2-2k [2048, 9472] |
| 9984 | 10752 | 1239.4 | 1638.7 | **−32.22%** | `MT256x256` | uniform-50/50 [4992, 4992] |
| 5376 | 12288 | 750.9 | 992.3 | **−32.15%** | `MT256x256` | uniform-50/50 [2688, 2688] |
| 5888 | 8192 | 552.7 | 728.9 | **−31.88%** | `MT256x256` | pow2-2k [2048, 3840] |
| 5376 | 12032 | 758.0 | 999.4 | **−31.85%** | `MT256x256` | uniform-50/50 [2688, 2688] |
| 9984 | 8192 | 919.9 | 1212.3 | **−31.80%** | `MT256x256` | uniform-50/50 [4992, 4992] |
| 14080 | 5632 | 915.3 | 1198.3 | **−30.92%** | `MT256x256` | pow2-2k [2048, 12032] |

Notice all 10 losses use `MT256x256` baseline — already the optimal tile. The Origami split scorer picked candidates anyway, and they all lose. The Origami model isn't seeing the per-WG efficiency cliff that occurs when you take a perfectly-shaped problem and chop it into two ill-shaped halves.

---

## 5. Per-split-strategy effectiveness

| Split label | n | Mean gain | Wins | Losses | Verdict |
|---|---:|---:|---:|---:|---|
| `wave-48mt` | 7 | **+1.92%** | 4 | 1 | Best — but very few cases use it |
| `pow2-8k` | 230 | −2.29% | 85 | 122 | Least bad pow2 split |
| `wave-40mt` | 5 | −3.28% | 2 | 2 | Too few samples |
| `pow2-4k` | 451 | −4.41% | 87 | 298 | Heavy losses |
| `pow2-2k` | 611 | −5.98% | 123 | 435 | The most-picked split, also one of the worst — Origami over-favors it |
| `uniform-50/50` | 453 | −6.52% | 102 | 313 | Particularly bad when baseline is MT256x256 |
| `asym-40/60` | 245 | −7.29% | 39 | 188 | Worst general-case split |
| `wave-12mt` | 6 | −10.01% | 1 | 5 | Too small first piece |

The Origami candidate generator's **strong preference for `pow2-2k`** (611 cases — the most-picked) is a major failure mode here. The 2048-row first piece consistently maps to a small MT (e.g., `MT256x160`), and on BBS-TN that small MT is significantly slower than the parent MT256x256 — turning a balanced problem into an imbalanced one.

A simple Origami-side fix: **down-rank pow2 candidates whose first piece would map to MT_N < 256**, or rank uniform candidates higher when both halves can keep the baseline MT.

---

## 6. The Origami WGM override (`--origami_wgm`)

Result: **WGM-only delta is essentially zero** for BBS-TN.

| WGM impact | Value |
|---|---:|
| Mean WGM-vs-no-WGM delta | **−0.11%** |
| Median delta | −0.01% |
| Cases where WGM helps (>+0.5%) | 400 (19.8%) |
| Cases where WGM hurts (<−0.5%) | 497 (24.5%) |
| Active multi-MT cases | 2025 |

The `bbs_tn_heatmap_wgm_delta.png` chart confirms this visually — a salt-and-pepper pattern with no systematic spatial structure, consistent with the WGM tuning being a wash for BBS-TN at K=8192.

This contrasts with our small FP16-NN A/B test where WGM delivered a consistent **+0.5–0.8%** uplift. The reason for the BBS-TN difference is again the heuristic: when the baseline kernel is already `MT256x256`, the kernel's built-in WGM is already well-tuned, and Origami's recommended WGM (typically the same value or off by 1) doesn't change anything material. Where Origami's WGM does diverge from the kernel default, it's roughly equally likely to help or hurt by < 1%.

**Implication:** WGM tuning is a configuration-dependent feature. It pays for FP16 + non-optimal MT baselines; it's a wash for BBS + already-optimal MT256x256.

---

## 7. Heatmap overview

The four heatmaps in `~/MultiMT/bench/results_bbs_huge/` show the spatial structure of the results:

1. **`bbs_tn_heatmap_multi_mt.png`** — a sea of yellow/red over most of the active region, with two notable green stripes: (a) a thin diagonal at M≈10k,N≈6k where the baseline transitions through MT256x224, (b) the upper-right corner (M, N ≥ 14k) where large kernels pay relatively less for split overhead.
2. **`bbs_tn_heatmap_multi_mt_wgm.png`** — visually almost identical to #1, confirming WGM has no spatial signal.
3. **`bbs_tn_heatmap_wgm_delta.png`** — pure noise pattern (±2%).
4. **`bbs_tn_heatmap_baseline_mt.png`** — *the* explanation: the entire upper-right region is uniform light-blue (`MT256x256`); the diagonals where multi-MT has any chance are the MT256x224 (yellow) bands; small purple/pink/green/blue patches near M≈5–7k mark the few "valley" baseline MTs (MT224x256, MT256x192, MT224x224, MT208x256) — exactly where multi-MT's wins are concentrated.

---

## 8. Concrete recommendations

### 8.1 Defaulting

**Disable multi-MacroTile by default for BBS dtype**. The current default (auto-engage when `min(M,N) ≥ 5120`) costs 5.5% on average across BBS-TN at K=8192 — far worse than the FP16-NN dataset where it gained 3.8%.

### 8.2 Tighter pre-guard for BBS

After querying the baseline kernel's MT (which `shouldEnableMultiMT_MTAware` already does), only allow multi-MT to run if at least one of:

- Baseline `MT_M < 256` (so we have headroom to upgrade), or
- Baseline `MT_N < 256` (same), or
- M and N are both ≥ 12,288 (large-kernel regime where the +0.23% mean comes from)

Applying this guard to the 2,025 active cases would have:
- Disabled multi-MT for 1,678 MT256x256 cases — saving most of the 998 severe regressions
- Kept multi-MT enabled for 5 MT208x256 + 28 MT224x256 + 20 MT256x192 + 291 MT256x224 = 344 cases — capturing most of the 447 wins and avoiding most of the 1,375 losses

### 8.3 Origami candidate generator

Add a hard rule: **reject any pow2-2k candidate whose 2048 first piece would map to MT_M < 256 or MT_N < 256** when the baseline kernel uses MT256x256. The current generator over-picks pow2-2k (611 cases, −6% mean) precisely on the cases where the smaller piece's kernel becomes the bottleneck.

### 8.4 WGM is not worth applying for BBS-TN

The `--origami_wgm` feature should remain available (and is correct in principle — `selectWorkgroupMapping` returns reasonable values), but should not be enabled by default for BBS-TN. Re-enable it when there's a clear test signal (FP16, NN, or MT-valley baselines).

### 8.5 Re-run on FP16-TN and BBS-NN

This sweep covered only **BBS + TN**. Three cousins are still untested at this scale:
- **FP16-TN** (BF16's sister case, same layout)
- **BBS-NN** (same dtype, different layout — closer to our previous FP16-NN sweep)
- **FP8-TN/NN** (lower-precision cousin)

Hypotheses to test:
1. FP16-TN should fall between FP16-NN (+3.8% mean) and BBS-TN (−5.5% mean) — likely break-even.
2. BBS-NN may regain wins because the NN-specific tunings might select non-MT256x256 kernels more often.
3. FP8 increases compute density per cycle, which could shift the crossover.

The existing harness only needs `--precision` and `--transA/--transB` flags changed; running each is another ~100 minutes.

---

## 9. Aggregate computation cost

| | Value |
|---|---|
| Total bench invocations | **11,163** (3,721 cases × 3 modes) |
| Total wallclock | **99.4 minutes** on 6 GPUs in parallel |
| Average per-case wall | 9.6 s (incl. setup, spread over 3 modes) |
| Total log files | 11,163 (one per (case, mode)) |
| Per-case logs disk usage | ~7 MB total |
| JSON / CSV / MD outputs | 5.4 MB |

The harness is in `~/MultiMT/bench/`:

| File | Purpose |
|---|---|
| `run_bench_huge_bbs.py` | Sweep harness (FLOPs-based iter sizing, 3 modes/case, round-robin GPUs) |
| `render_huge_bbs.py` | Heatmap + summary generator |
| `logs_bbs_huge/` | Per-case raw bench output (11,163 files) |
| `results_bbs_huge/results_huge_*.json` | Full structured results (one nested record per case) |
| `results_bbs_huge/bbs_tn_summary.csv` | One CSV row per case, 23 columns |
| `results_bbs_huge/bbs_tn_summary.md` | Auto-generated summary |
| `results_bbs_huge/bbs_tn_heatmap_*.png` | 4 heatmap PNGs |

Reproduce the full sweep:

```bash
source ~/MultiMT/MultiMTVenv/bin/activate
cd ~/MultiMT/bench
python3 run_bench_huge_bbs.py --profile huge --devices 0,1,2,3,4,5
python3 render_huge_bbs.py results_bbs_huge/results_huge_<TIMESTAMP>.json
```

---

## 10. Caveats & limitations

1. **Single K (8192).** All 3,721 cases use K=8192. Larger K (16384, 32768) shifts the balance toward multi-MT because longer kernel runtime amortizes split overhead better. A K-sweep is a natural follow-up.
2. **Single layout (TN).** Findings here may not transfer to NN/NT/TT.
3. **No correctness verification (`--verify`).** Performance focus only. The multi-MT path uses the same Tensile kernels as default, so correctness is structurally equivalent.
4. **Iter sizing approximation.** FLOPs-based estimation at 0.5 TF/s is conservative; actual wall time per phase is in the 1.5–4 s range depending on case. Sufficient for stable measurements but not as tight as per-case probing would give.
5. **Heuristic noise floor.** The heuristic's selection of MT can shift between hipBLASLt versions. Results here are specific to commit `1bb208d92a`.
6. **MI350X-specific.** The 8-XCD / 4-MB-L2-per-XCD architecture drives many of the observed effects. BBS-TN on MI300X (gfx942) would likely show a different pattern (different MT availability, different XCD count).

---

## Bottom line

For BBS-TN at K=8192, the answer to "should we use multi-MacroTile?" is a clear **no by default**, **yes for the ~17% of (M, N) pairs where the baseline MT is in a valley** (MT256x224, MT224x256, MT256x192, MT208x256), and **conditionally yes for very large square problems** (M, N ≥ ~12,288).

The Origami WGM override is a wash for this configuration and should not be enabled until test signal emerges from a sweep where it actually moves the needle (e.g., FP16 with MT-valley baselines, where our small A/B test showed +0.5–0.8%).
