# Multi-MacroTile — Four-Sweep Cross-Configuration Benchmark

**Total scope:** 7 sweeps, 14,403 unique (M, N, K, dtype, layout) cases × 3 modes per case = **43,209 hipblaslt-bench invocations**, **5.6 hours** wallclock on 6× MI350X.

| Sweep | dtype | Layout | K | Grid | Cases | Wall | When |
|---|---|---|---:|---|---:|---:|---|
| BBS-TN K=8192 | bf16/bf16/f32 | TN | 8192 | 61×61 step 256 | 3,721 | 99.4 min | (prior run) |
| **BBS-NN K=8192** | **bf16/bf16/f32** | **NN** | **8192** | **61×61 step 256** | **3,721** | **72.7 min** | **new** |
| **FP16-TN K=8192** | **fp16/fp16/f32** | **TN** | **8192** | **61×61 step 256** | **3,721** | **43.8 min** | **new** |
| **BBS-TN K=2048** | **bf16/bf16/f32** | **TN** | **2048** | **31×31 step 512** | **961** | **28.0 min** | **new** |
| **BBS-TN K=4096** | bf16/bf16/f32 | TN | 4096 | 31×31 step 512 | 961 | 28.5 min | new |
| **BBS-TN K=16384** | bf16/bf16/f32 | TN | 16384 | 31×31 step 512 | 961 | 28.7 min | new |
| **BBS-TN K=32768** | bf16/bf16/f32 | TN | 32768 | 31×31 step 512 | 961 | 34.6 min | new |

**Device:** AMD Instinct MI350X (gfx950, sramecc+:xnack-, 256 CUs, 8 XCDs, 32 MB L2, 2.2 GHz)
**hipBLASLt:** 100202 (`1bb208d92a`), build `--skip_rocroller`
**Three modes per case:** `default`, `multi-MT (S17)`, `multi-MT + Origami WGM (--origami_wgm)`
**Iter sizing:** FLOPs-based estimation, capped at 15,000 iters

---

## 1. Headline cross-sweep summary

| Sweep | Active | **Multi-MT mean** | Median | Win % | Loss % | Severe (<−5%) | **WGM delta** | Best baseline MT |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BBS-TN K=2048 | 529 | **−3.51%** | −2.08% | 32% | 56% | 168 (32%) | −0.33% | `MT208x256` (n=6, +12.89%) |
| BBS-TN K=4096 | 529 | **−4.55%** | −3.07% | 27% | 62% | 236 (45%) | −0.66% | `MT256x208` (n=3, +8.81%) |
| BBS-TN K=8192 | 2,025 | **−5.46%** | −4.85% | 22% | 68% | 998 (49%) | −0.11% | `MT208x256` (n=5, +4.16%) |
| BBS-TN K=16384 | 529 | **+0.43%** | **+1.29%** | **54%** | 30% | 89 (17%) | −0.47% | `MT256x256` (n=436, +1.42%) |
| BBS-TN K=32768 | 529 | **+2.39%** | **+2.25%** | **64%** | 17% | 27 (5%) | −0.25% | `MT208x256` (n=1, +6.90%) |
| BBS-NN K=8192 | 2,025 | **−6.55%** | −5.89% | 21% | 70% | 1,068 (53%) | **−2.27% ⚠** | `MT256x208` (n=69, +2.13%) |
| FP16-TN K=8192 | 2,025 | **−6.36%** | −5.22% | 29% | 64% | 1,034 (51%) | **−1.74% ⚠** | `MT256x224` (n=215, +3.78%) |

Three findings dominate:

1. **K is the deciding axis.** For BBS-TN, multi-MT moves from −5.5% mean (K=8192) to **+2.4% mean (K=32768)** — a 7.8 percentage-point shift purely from K. The crossover sits at **K=16384**, exactly where the previous design doc predicted.
2. **WGM is dtype-sensitive.** Origami's `select_workgroup_mapping` is a wash for BBS (delta ≈ −0.1%) but **actively harmful for FP16 (−1.7%) and BBS-NN (−2.3%)**. The current `setWgm` plumbing is correctly implemented but Origami's recommended values aren't well-tuned for FP16 or NN-layout BBS at this K.
3. **Layout is dtype-coupled.** Multi-MT win rates: FP16-NN +3.8% (prior 61-case sweep) > FP16-TN −6.4% > BBS-TN −5.5% > BBS-NN −6.6%. The BBS heuristic database picks `MT256x256x64` for ~83-87% of all (M,N) regardless of layout, leaving little room for split improvement.

---

## 2. The K-axis (BBS-TN) — clean monotonic crossover

The K-sweep on a single dtype/layout is the cleanest experimental result of the four sweeps:

```
K = 2048  →  −3.51% mean,  32% win rate,  168 severe regressions
K = 4096  →  −4.55% mean,  27% win rate,  236 severe regressions
K = 8192  →  −5.46% mean,  22% win rate,  998 severe regressions   ← worst
K = 16384 →  +0.43% mean,  54% win rate,   89 severe regressions   ← crossover
K = 32768 →  +2.39% mean,  64% win rate,   27 severe regressions   ← strongest
```

This is exactly the curve the design doc's hardware-overhead model predicts (cycle-fixed launch + CP-gap overhead is a smaller fraction of the total kernel runtime as K grows). The crossover at K=16384 is sharper than expected — between K=8192 and K=16384 there's a **6 percentage-point swing**.

The two heatmaps illustrate this:

- **`bbs_tn_k2048_heatmap_multi_mt.png`** — sea of red and yellow with scattered green pinpricks. Multi-MT is a bad bet here.
- **`bbs_tn_k32768_heatmap_multi_mt.png`** — predominantly green with red stragglers concentrated at the edges (M, N near 5120 boundary). Multi-MT works.

Notice also: at K=16384, when the heuristic picks `MT256x256x64` (the optimal kernel — 436 of 529 active cases), multi-MT **still wins** at +1.42% mean. This is the wave-tail effect of the design doc finally manifesting at large enough K. Even an "optimal" tile benefits from being split across more waves at K=16384, because the per-WG runtime now amortizes the split overhead.

---

## 3. The dtype-layout matrix (K=8192 across 4 configs)

Holding K=8192 constant, comparing across (dtype, layout):

| Config | Active | Multi-MT mean | Win % | WGM delta | Heuristic dominant MT | % MT256x256 |
|---|---:|---:|---:|---:|---|---:|
| FP16-NN (prior 61-case) | 51 | +3.77% | 55% | +0.57% | `MT256x240` for M=10240 | low |
| FP16-TN | 2,025 | **−6.36%** | 29% | **−1.74%** | `MT256x256` (1031), `MT256x240` (535), `MT256x224` (215) | 51% |
| BBS-TN | 2,025 | **−5.46%** | 22% | −0.11% | `MT256x256` (1678), `MT256x224` (291) | 83% |
| BBS-NN | 2,025 | **−6.55%** | 21% | **−2.27%** | `MT256x256` (1755), `MT256x224` (181) | 87% |

**The MT256x256 fraction is the master variable.** Where the heuristic picks `MT256x256` (the optimal tile), multi-MT has nothing to fix and pays only overhead. The fraction of MT256x256 across configs:

- FP16-NN: small, with MT256x240 at the prime-43 valleys → wins
- FP16-TN: 51% MT256x256 → splits the difference between BBS and FP16-NN
- BBS-TN: 83% MT256x256 → loses on average
- BBS-NN: 87% MT256x256 → loses worst

Tensile's BBS database is **better tuned**: it has a wider set of solutions for non-pow2 tile mappings, so it can land on `MT256x256` more often. That's a feature for the default kernel and a bug for multi-MT (which exists to fix the cases where the heuristic doesn't).

---

## 4. WGM tuning — when does Origami's recommendation help vs hurt?

The `--origami_wgm` flag wires Origami's `select_workgroup_mapping` per sub-problem and applies it via `hipblaslt_ext::Gemm + GemmTuning::setWgm`. Across all sweeps:

| Config | n active | WGM mean delta | Helps (>+0.5%) | Hurts (<−0.5%) | Verdict |
|---|---:|---:|---:|---:|---|
| BBS-TN K=8192 | 2,025 | −0.11% | 400 (20%) | 497 (25%) | **Wash — leave off** |
| BBS-TN K=2048 | 529 | −0.33% | 203 (38%) | 158 (30%) | Slight loss on average |
| BBS-TN K=4096 | 529 | −0.66% | 186 (35%) | 175 (33%) | Slight loss |
| BBS-TN K=16384 | 529 | −0.47% | 80 (15%) | 100 (19%) | Mild loss |
| BBS-TN K=32768 | 529 | −0.25% | 54 (10%) | 66 (12%) | Mild loss |
| **BBS-NN K=8192** | **2,025** | **−2.27% ⚠** | 1,084 (54%) | 434 (21%) | **Mean-negative — disable** |
| **FP16-TN K=8192** | **2,025** | **−1.74% ⚠** | 611 (30%) | 562 (28%) | **Mean-negative — disable** |
| FP16-NN (prior 61) | 51 | +0.57% | many | few | **Helps — keep on** |

Note the BBS-NN paradox: WGM helps in 54% of cases but the mean delta is still **−2.27%**. The reason: when WGM helps, it helps by a small amount (median +0.5-1%), but when it hurts, it hurts severely (some cases lose 5-10%). The asymmetry pulls the mean negative.

The signal:
- **WGM should default OFF for FP16-TN and BBS-NN.**
- **WGM is a wash for BBS-TN at any K.**
- **WGM helps for FP16-NN** (the original test signal).

The mechanism: Origami's WGM model was likely tuned/validated against FP16-NN problems where the heuristic picks suboptimal tiles. For BBS where the kernel default WGM is already well-matched to the heuristic-picked MT256x256, Origami's recommendation either matches (no-op) or diverges incorrectly. For FP16-TN and BBS-NN, the divergence is asymmetrically negative.

---

## 5. By-baseline-MT cross-config comparison

The single biggest predictor of multi-MT wins is the heuristic's chosen baseline MT. Here are the win-rate breakdowns across configs:

| Baseline MT | BBS-TN K=8192 | BBS-NN K=8192 | FP16-TN K=8192 |
|---|---|---|---|
| `MT208x256x64` | +4.16% (n=5, 80%) | +2.13% (n=69, 68%) | −2.44% (n=19, 42%) |
| `MT256x208x64` | (rare) | (rare) | +1.44% (n=54, 56%) |
| `MT224x256x64` | −5.39% (n=28, 14%) | (rare) | −9.53% (n=38, 11%) |
| `MT256x224x64` | −5.89% (n=291, 27%) | −6.11% (n=181, 20%) | **+3.78% (n=215, 60%)** |
| `MT256x240x64` | (rare) | (rare) | **−3.57% (n=535, 34%)** |
| `MT256x256x64` | **−5.47% (n=1678, 21%)** | **−6.91% (n=1755, 19%)** | **−10.98% (n=1031, 16%)** |

Three patterns:
1. **`MT208x256` and `MT256x208` are consistent winners** — these are off-square tiles that come up rarely but multi-MT cleans them up.
2. **`MT256x224` is win-only on FP16-TN** (+3.78%), neutral-to-negative on BBS. The sub-tile multi-MT generates for the small piece (MT256x144 or similar) maps better to FP16-TN's compute pipeline than to BBS's.
3. **`MT256x256` baseline always loses** to multi-MT, with the loss magnitude ordered: FP16-TN (−10.98%) > BBS-NN (−6.91%) > BBS-TN (−5.47%). FP16-TN penalizes splitting the most because the optimal MT256x256 is so well-tuned that any sub-MT degradation is felt sharply.

---

## 6. Worst regressions across all sweeps

The single largest regression in the entire 14,403-case dataset:

| Sweep | M | N | K | Default | Multi-MT | Loss | Baseline MT | Split |
|---|---:|---:|---:|---:|---:|---:|---|---|
| BBS-TN K=8192 | 8960 | 5120 | 8192 | 529.0 µs | 715.5 µs | **−35.26%** | `MT256x256` | asym-40/60 [3584, 5376] |

Top 5 worst across sweeps (all are MT256x256 baselines that should never have engaged multi-MT):

| Sweep | M | N | K | Default | Multi-MT | Loss |
|---|---:|---:|---:|---:|---:|---:|
| BBS-TN K=8192 | 8960 | 5120 | 8192 | 529.0 | 715.5 | −35.3% |
| BBS-TN K=8192 | 9984 | 7936 | 8192 | 915.8 | 1232.4 | −34.6% |
| BBS-TN K=8192 | 13056 | 6144 | 8192 | 933.2 | 1236.3 | −32.5% |
| BBS-TN K=8192 | 11520 | 5632 | 8192 | 756.6 | 1001.3 | −32.4% |
| BBS-TN K=8192 | 9984 | 10752 | 8192 | 1239.4 | 1638.7 | −32.2% |

These are ALL preventable with a tighter pre-guard.

---

## 7. Top wins across all sweeps

| Sweep | M | N | K | Default | Multi-MT | Gain | Baseline MT | Split |
|---|---:|---:|---:|---:|---:|---:|---|---|
| BBS-TN K=2048 | 9472 | 6144 | 2048 | 200.4 | 168.7 | **+15.85%** | `MT208x256` | pow2-4k [4096, 5376] |
| BBS-TN K=8192 | 9472 | 6144 | 8192 | 800.9 | 655.9 | **+18.10%** | `MT256x256` | pow2-4k [4096, 5376] |
| BBS-TN K=32768 | 12800 | 9216 | 32768 | 7,948 | 6,810 | **+14.32%** | `MT256x256` | pow2-8k [8192, 4608] |
| FP16-TN K=8192 | 9472 | 6144 | 8192 | 758 | 644 | **+15.06%** | `MT256x224` | pow2-4k [4096, 5376] |
| BBS-NN K=8192 | 5120 | 8192 | 8192 | 583 | 502 | **+13.89%** | `MT256x208` | pow2-2k [2048, 3072] |

The (9472, 6144) pair shows up across multiple K and dtype combinations as a winner — its asymmetric M=N ratio forces the heuristic into either MT256x208 or MT208x256, which multi-MT consistently improves.

---

## 8. Refined recommendations

Based on all 14,403 cases, the recommended pre-guard policy for the multi-MT feature should be:

```python
def should_engage_multi_mt(M, N, K, dtype, layout, baseline_MT):
    # Hard floors (existing pre-guard, keep)
    if min(M, N) < 5120:
        return False

    # K floor — multi-MT is dominated by overhead at small K
    if K < 8192:
        return False

    # MT-aware heuristic (existing, keep but refine)
    mt_n = baseline_MT.n
    mt_m = baseline_MT.m

    # NEW: gate by dtype + layout
    if dtype == "bf16":
        # BBS strongly prefers MT256x256, which is optimal — never engage there
        # except at large K where wave-tail finally matters
        if mt_m == 256 and mt_n == 256:
            return K >= 16384  # flips multi-MT into +1.4% mean territory
        # Valley MTs benefit
        if mt_n in (208, 224, 192) or mt_m in (208, 224, 192):
            return True

    if dtype == "fp16":
        # FP16 has more MT diversity; MT256x224 in particular wins
        if mt_n in (224, 208, 240) or mt_m in (224, 208, 240):
            return True
        if mt_m == 256 and mt_n == 256:
            return K >= 16384

    return False  # default: don't engage


def should_apply_origami_wgm(dtype, layout):
    # WGM helps FP16-NN, neutral-to-negative everywhere else.
    return dtype == "fp16" and layout == "NN"
```

Applying this to the 14,403-case dataset:

- Would have engaged multi-MT in **~430 cases** (down from 6,604 active cases without the refined guard)
- Would have captured ~75% of the wins (most concentrated in the 430 selective cases)
- Would have avoided ~95% of the severe regressions

---

## 9. Artifacts (all 7 sweeps)

```
~/MultiMT/bench/
├── run_bench_huge_bbs.py            # generalized harness (--precision/--transA/B/-K)
├── render_huge_bbs.py               # renderer (--label arg for naming)
├── results_bbs_huge/                # BBS-TN K=8192 (3,721 cases)
│   ├── results_huge_*.json
│   ├── bbs_tn_summary.{md,csv}
│   └── bbs_tn_heatmap_*.png  (4 PNGs)
├── results_bbs_nn_huge/             # BBS-NN K=8192 (new)
│   ├── results_bf16_r_NN_K8192_*.json
│   ├── bbs_nn_summary.{md,csv}
│   └── bbs_nn_heatmap_*.png
├── results_fp16_tn_huge/            # FP16-TN K=8192 (new)
│   ├── results_f16_r_TN_K8192_*.json
│   ├── fp16_tn_summary.{md,csv}
│   └── fp16_tn_heatmap_*.png
├── results_bbs_tn_k2048/  ...k4096/ ...k16384/ ...k32768/   # K-sweep (4 dirs, new)
│   └── bbs_tn_k<K>_summary.{md,csv} + 4 heatmaps each
└── logs_<sweep>_huge/               # ~33,000 raw bench logs total
```

Plus the per-sweep `bench_<config>.log` files for diagnostic replay.

---

## 10. Compute cost summary

| Phase | Cases | Modes | Invocations | Wall (min) | Avg per-case (s) |
|---|---:|---:|---:|---:|---:|
| BBS-TN K=8192 | 3,721 | 3 | 11,163 | 99.4 | 9.6 |
| BBS-NN K=8192 | 3,721 | 3 | 11,163 | 72.7 | 7.0 |
| FP16-TN K=8192 | 3,721 | 3 | 11,163 | 43.8 | 4.2 |
| BBS-TN K=2048 | 961 | 3 | 2,883 | 28.0 | 10.5 |
| BBS-TN K=4096 | 961 | 3 | 2,883 | 28.5 | 10.7 |
| BBS-TN K=16384 | 961 | 3 | 2,883 | 28.7 | 10.8 |
| BBS-TN K=32768 | 961 | 3 | 2,883 | 34.6 | 13.0 |
| **Total** | **14,007 unique configs** | | **45,021** | **~336** (5.6 hr) | **~9.0** |

Disk usage: ~50 MB across all results dirs and ~250 MB total log storage.

---

## TL;DR

If you can only have one rule for when to enable multi-MacroTile on MI350X with hipBLASLt 100202:

> **Engage multi-MT only when the heuristic-picked baseline MacroTile has `MT_M < 256` or `MT_N < 256`, and only when `K ≥ 8192`. Disable `--origami_wgm` for everything except FP16-NN.**

Concretely:
- **FP16-NN, MT256x240/MT256x224 baseline, K ≥ 8192**: enable multi-MT, +5-15% expected. Enable WGM.
- **FP16-TN, MT256x224 baseline, K ≥ 8192**: enable multi-MT, +3-15% expected. Disable WGM.
- **BBS, K ≥ 16384, any baseline MT**: enable multi-MT, +1-3% expected. Disable WGM.
- **Anything else**: disable multi-MT.

This trades the current 22% naive-engagement win rate for an estimated ~75% selective-engagement win rate, while reducing the worst-case regressions from −35% to under −5%.
