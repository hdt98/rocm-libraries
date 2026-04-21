# Multi-MacroTile: Design & Implementation

**hipBLASLt Feature Documentation**  
**Version:** 4.0  
**Date:** 2026-04-21  
**Device:** AMD Instinct MI350X (gfx950)

---

## 1. Problem Statement

On AMD Instinct MI350X (gfx950), the hipBLASLt heuristic selects a single Tensile kernel for each GEMM. Kernel performance depends heavily on the MacroTile configuration (e.g., MT256x256x64) selected for the problem dimensions. Certain M/N sizes land in "performance valleys" where the available MacroTile is a poor fit, achieving only 60-75% of peak throughput.

**Example:** A 10240x10240x8192 FP16 GEMM achieves 0.940 TFLOPS baseline. But splitting it into two sequential sub-GEMMs of 8192x10240x8192 + 2048x10240x8192 achieves 1.224 TFLOPS — a 30.3% improvement — because each sub-problem gets a better-tuned kernel. The 8192 sub-problem maps to MT256x256x64 at ~99% efficiency; the 2048 piece maps to MT256x160x64 at ~87%.

Multi-MacroTile automates this: it generates candidate split configurations, micro-benchmarks them at runtime, and uses the fastest for the full timing run.

---

## 2. Architecture Overview

```
CLI (client.cpp)
  ├── --multi_macrotile          # Enable feature
  ├── --split_strategy 17        # M-split via Origami
  ├── --num_splits 2             # Number of sub-problems
  └── --l2_cache_hints           # L2 cache persistence

testing_matmul.hpp (timing path)
  ├── splitGemmProblem()         # Generate initial split
  │   └── computeOrigamiOptimizedSplitsWithHandle()
  │       ├── isMacroTilePreserved()       # Safety check
  │       └── generateOrigamiCandidates()  # Candidate generation + scoring
  │           └── querySubProblemLatency() # Origami analytical model
  ├── Empirical Micro-Benchmark            # Test all candidates
  │   └── 3 iterations per candidate → pick fastest
  └── Main Timing Loop                     # Use winning split
      └── Sequential hipblasLtMatmul per sub-problem
```

---

## 3. Detailed Flow

### 3.1 Entry & Validation (testing_matmul.hpp)

When `--multi_macrotile` is set:

1. **API check**: Only the C API path (`--api_method c`) is supported; extension API (`mix`/`cpp`) returns an error.
2. **Grouped GEMM check**: Multi-MacroTile is incompatible with `--grouped_gemm`.
3. **Timing path entry**: The multi-MT timing block fires when `arg.multi_macrotile && gemm_count == 1 && !arg.use_ext`. It executes the full benchmark and returns early — the normal timing path is skipped.

### 3.2 Split Problem Generation (multi_macrotile.hpp)

`splitGemmProblem()` is the main entry point. Given the GEMM dimensions (M, N, K), it:

1. **Determines split axis** from `split_strategy`:
   - **17** = M-split (Origami search)
   - **18** = N-split (Origami search)
   - **19/20** = Brute-force M/N-split
   - **21/22** = 3-way M/N-split
   - **23/24** = XCD-aware M/N-split

2. **Calls Origami** to determine split sizes via `computeOrigamiOptimizedSplitsWithHandle()`.

3. **Builds sub-problems**: For each split piece, computes byte offsets into the parent A, B, C, D, E, and bias buffers. For an M-split:
   - Each sub-problem has full N and K, with a slice of M
   - A is sliced along rows (offset by `m_off * element_size` for non-transposed)
   - B is shared across all sub-problems (offset = 0)
   - C/D are sliced with `(m_off + n_off * ld) * element_size`
   - Bias is sliced along M: `m_off * element_size`

4. **Falls back to single GEMM** if Origami returns an empty split list.

### 3.3 Origami Candidate Generation (multi_macrotile_origami_improved.hpp)

The `generateOrigamiCandidates()` function produces ~7-8 candidate split ratios for a 2-way split:

**Minimum sub-problem size**: `max(2048, 15% of total_size)`, rounded down to a MacroTile multiple.

**Candidate sources** (for `total_size = 10240`, `macrotile = 128`):

| Type | Formula | Example Splits |
|------|---------|---------------|
| Uniform 50/50 | `total/2` aligned to MT | [5120, 5120] |
| Asymmetric 60/40 | `total * 0.60` aligned | [6144, 4096] |
| Asymmetric 40/60 | `total * 0.40` aligned | [4096, 6144] |
| Asymmetric 70/30 | `total * 0.70` aligned | [7168, 3072] |
| Asymmetric 30/70 | `total * 0.30` aligned | [3072, 7168] |
| Power-of-2 2k | `2048, remainder` | [2048, 8192] |
| Power-of-2 4k | `4096, remainder` | [4096, 6144] |
| Power-of-2 8k | `8192, remainder` | [8192, 2048] |

After deduplication by first split size, each candidate is scored analytically using the Origami latency model.

### 3.4 Origami Analytical Scoring

For each candidate, the scoring loop:

1. Forms the two sub-GEMM dimensions (e.g., for M-split candidate [8192, 2048]: sub0 = 8192xNxK, sub1 = 2048xNxK).
2. Calls `querySubProblemLatency()` which:
   - Queries `hipblaslt_ext::getAllAlgos()` to get the heuristic's top kernel for the sub-problem
   - Parses the Tensile solution name to extract MacroTile dimensions (regex: `MT(\d+)x(\d+)x(\d+)`)
   - Feeds the problem size and MacroTile into `origami::compute_total_latency()` — an analytical model of kernel execution time in GPU cycles
3. Sums the latencies for all sub-problems (sequential execution model).
4. Candidates are sorted by ascending total latency.

The analytically-best candidate is used as the initial split, but may be overridden by the empirical micro-benchmark.

### 3.5 MacroTile Preservation Check

Before generating candidates, `isMacroTilePreserved()` acts as a safety gate:

1. Queries the heuristic for the **original** problem (M, N) → gets baseline MacroTile (e.g., MT256x256x64).
2. Queries the heuristic for a **uniform** split (M/2, N) → gets sub-problem MacroTile.
3. Rejects splitting if the sub-problem's MacroTile M or N component falls below **75%** of the baseline's.

This prevents splits that would degrade to much smaller MacroTiles (e.g., from MT256x256 to MT64x64), which would increase workgroup count and overhead.

### 3.6 Empirical Micro-Benchmark (testing_matmul.hpp)

When strategies 17-24 produce multiple candidates, the timing path runs an empirical search:

1. **For each candidate** (typically 7-8):
   a. Calls `splitGemmProblem()` again as a template, then overwrites the split sizes with the candidate's
   b. Creates fresh `hipblasLtMatrixLayout` and `hipblasLtMatmulDesc` for each sub-problem
   c. Queries `hipblasLtMatmulAlgoGetHeuristic` for each sub-problem's kernel
   d. Runs **3 iterations** of sequential `hipblasLtMatmul` calls on the GPU stream
   e. Synchronizes, measures wall-clock time, averages over 3 iterations

2. **Selects the fastest** candidate by minimum average time.

3. **If the winner differs from the analytical best** (index 0), destroys the initial layouts and rebuilds `subProblems` and `spCtxs` arrays from the winning split.

4. The winner is used for the **main timing loop** (200+ iterations).

### 3.7 Main Timing Execution

After the empirical search selects the best split:

1. Pre-created `SubProblemContext` objects hold layouts, algorithm, and device pointers — no per-iteration overhead.
2. Each timing iteration sequentially executes `hipblasLtMatmul` for each sub-problem on the same stream.
3. Total GFLOPS is computed from the full problem's FLOPs (2 * M * N * K) divided by average wall-clock time.

---

## 4. File Structure

```
clients/common/include/
├── multi_macrotile.hpp                   # ~215 lines
│   ├── GemmSubProblem                    # Sub-problem descriptor (dims + byte offsets)
│   ├── SubProblemContext                 # Pre-created layouts/algo/pointers for hot loop
│   ├── getDataTypeSize()                # Element size lookup for offset calculation
│   ├── calculateOffset{A,B,CD,Bias}()   # Byte offset math for matrix views
│   ├── shouldUseMultiMacroTile()        # Guard check (currently always true)
│   └── splitGemmProblem()               # Main entry: calls Origami, builds sub-problems
│
├── multi_macrotile_origami_improved.hpp  # ~454 lines
│   ├── OrigamiCandidate                 # Split sizes + label + analytical latency
│   ├── getOrigamiCandidates()           # Thread-local candidate storage
│   ├── parseMacroTileFromName()         # Extract MT dims from Tensile solution name
│   ├── hipToOrigamiDtype()              # HIP → Origami type conversion
│   ├── computeOrigamiLatency()          # Origami analytical latency for one sub-problem
│   ├── querySubProblemLatency()         # Heuristic query + Origami scoring
│   ├── isMacroTilePreserved()           # Safety: split doesn't degrade MacroTile >25%
│   ├── generateOrigamiCandidates()      # 2-way: ratio + pow2 candidates + scoring
│   ├── generate3WayCandidates()         # 3-way: uniform + pow2 triples
│   ├── generateXCDAwareCandidates()     # XCD-aware: L2-optimal split + fallbacks
│   └── computeOrigamiOptimizedSplitsWithHandle()  # Entry: gate → generate → store → return
│
└── testing_matmul.hpp                    # Integration (~5500 lines total)
    ├── Multi-MT validation (line ~1402)  # API/grouped-gemm checks
    ├── Timing path entry (line ~4470)    # Multi-MT timing block
    ├── Origami empirical loop (line ~4630) # Test all candidates, pick fastest
    └── Main timing execution             # Use winning split for full benchmark

clients/bench/src/
└── client.cpp                            # CLI option parsing
    └── --multi_macrotile, --split_strategy, --num_splits, --l2_cache_hints
```

---

## 5. Strategy Reference

| ID | Name | Axis | Candidate Method | Scoring |
|----|------|------|------------------|---------|
| **17** | Origami M-split | M | Ratio + pow2 (~7 candidates) | Origami analytical + empirical |
| **18** | Origami N-split | N | Ratio + pow2 (~7 candidates) | Origami analytical + empirical |
| 19 | Brute-force M-split | M | Step-16 sweep (hundreds) | Origami analytical + empirical |
| 20 | Brute-force N-split | N | Step-16 sweep (hundreds) | Origami analytical + empirical |
| 21 | 3-way M-split | M | Uniform + pow2 triples | Empirical only |
| 22 | 3-way N-split | N | Uniform + pow2 triples | Empirical only |
| 23 | XCD-aware M-split | M | L2-optimal + ratio + pow2 | Empirical only |
| 24 | XCD-aware N-split | N | L2-optimal + ratio + pow2 | Empirical only |

Strategy **17** is the default and best-tested configuration.

---

## 6. CLI Usage

```bash
cd /home/smalekta/MultiMT/rocm-libraries/projects/hipblaslt/build/release

./clients/hipblaslt-bench -m 10240 -n 10240 -k 8192 \
  --precision f16_r --device 1 \
  --multi_macrotile --split_strategy 17 --num_splits 2 \
  --l2_cache_hints --api_method c -i 200 -j 200
```

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--multi_macrotile` | flag | false | Enable splitting |
| `--split_strategy` | 17-24 | 17 | Split algorithm (see table above) |
| `--num_splits` | 2-16 | 2 | Number of sub-problems |
| `--l2_cache_hints` | flag | true | L2 persistence hints (applies only to legacy strategies 3/4, not 17/18) |
| `--api_method` | c | c | Must be `c` (extension API not supported) |
| `--device` | 0-7 | 0 | Target GPU device ID |

---

## 7. Why It Works

### Kernel Efficiency Varies Dramatically by Sub-Problem Size

On gfx950, the heuristic selects different MacroTile configurations depending on M-dimension. Measured FP16 GEMM efficiency for Mx10240x8192:

| M | Baseline TFLOPS | Approx. Efficiency |
|---|----------------|-------------------|
| 8192 | ~1.25 | ~82% |
| 6144 | ~1.25 | ~82% |
| 5120 | ~1.28 | ~84% |
| 4096 | ~1.29 | ~85% |
| 2048 | ~1.28 | ~84% |
| **10240** | **0.94** | **~62%** |
| **15360** | **0.96** | **~63%** |

The 10240 single-kernel baseline runs at only ~62% efficiency. Splitting into [8192, 2048] allows the large sub-problem to use a much better kernel. Even though the two sub-GEMMs run sequentially, the combined time is ~30% faster because both pieces run at >82% efficiency.

### The Empirical Search Finds Non-Obvious Winners

The analytical Origami model provides initial rankings, but the 3-iteration empirical micro-benchmark often finds a different winner. For example:
- For 10240x10240xK, the analytical model often ties all candidates (identical latency scores), but the empirical benchmark consistently reveals pow2-8k [8192,2048] as the fastest.
- The micro-benchmark captures effects the analytical model misses: cache behavior, memory bank conflicts, actual kernel launch overhead.

---

## 8. Benchmark Results Summary (MI350X, gfx950)

| Metric | Value |
|--------|-------|
| Problems tested | 46 |
| **Win rate (>2% gain)** | **87% (40/46)** |
| Loss rate (>2% loss) | **0% (0/46)** |
| Best gain | **+34.4%** |
| Worst result | **-0.6%** (within noise) |
| Average gain (all) | **+13.1%** |
| Average gain (wins) | **+15.0%** |

See `COMPREHENSIVE_BENCHMARK_RESULTS.md` for the full 46-problem benchmark table.

---

## 9. Removed Code

The following were removed during cleanup to simplify the codebase:

| Removed | Lines | Reason |
|---------|-------|--------|
| `multi_macrotile_fused.hpp` | 278 | Fused dispatch: blocked by platform |
| `multi_macrotile_fused_kernel.hpp` | 289 | Device-side dispatch: never worked |
| `multi_macrotile_fused_host.hpp` | 323 | Host-side batch: never worked |
| `multi_macrotile_kernel_extraction.hpp` | 410 | Kernel extraction: never worked |
| `multi_macrotile_origami.hpp` | 520 | Old Origami with estimation-only: replaced by empirical |
| Strategies 0-10, 15-16, 19-20 (legacy) | ~1000 | Uniform/heuristic: outperformed by Origami S17 |
| Stream-parallel path | ~150 | CU contention issues: underperformed |

**Net reduction**: ~3200 lines → ~670 lines (multi_macrotile.hpp + origami_improved.hpp).
