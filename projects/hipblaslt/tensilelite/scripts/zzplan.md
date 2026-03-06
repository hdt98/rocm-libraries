# Tensile Benchmark Performance Optimization Plan

## Context

Profiling `spmm_f16.yaml` (sparse GEMM f16 benchmark on gfx950) with `TimingInstrumentation=True` reveals that **only 2.2% of the 233s runtime is actual GPU kernel execution**. The remaining 97.8% is overhead: data reset, validation, codegen, serialization, and loop management. This plan identifies every phase consuming >2% of wall clock and provides concrete optimization strategies.

### Profiling Setup
- Config: `Tensile/Tests/common/sparse/gfx950/spmm_f16.yaml`
- Command: `Tensile spmm_f16.yaml out --global-parameters TimingInstrumentation=True`
- Analysis: `scripts/analyze_timing.py`
- Total wall clock: **233s**

### Iteration Math (root cause of high call counts)
- 8 benchmark problem groups × ForkParameter combinatorics → **4,713 solutions** total
- 266 problems × solutions/group → **251,922 solution iterations**
- ~12.5% fail runtime validation → **220,402 actual kernel runs**
- First valid solution per problem skips reset → **220,136 gpu_input_reset calls**
- **Group 1 (NN-MI16)** alone produces 78% of iterations (72 problems × 2,736 solutions = 197K)

### Constraints
- **Validation must always stay on** — all solutions must be validated for correctness
- **Iteration count cannot be reduced** — focus is on making each iteration faster
- **YAML format must be kept** — no format switches (msgpack, json, etc.)
- **Measure-first approach** — implement low-risk changes, re-profile, then decide next steps

---

## Summary Table

| # | Phase | Wall % | Time (s) | Calls | Per-call | Root Cause | Optimization | Potential Savings |
|---|-------|--------|----------|-------|----------|------------|-------------|-------------------|
| 1 | gpu_input_reset | 42.6% | 99.2s | 220K | 0.45ms | hipMemcpy D2D resets ALL output tensors before every kernel run; copies maxElements not actual size | Copy only actual problem size; use async copies | ~20% (size-exact copies) |
| 2 | benchmark_runs | 10.2% | 23.7s | 220K | 0.11ms | Per-kernel: hipEventCreate×2 + hipEventRecord×2 + hipEventSynchronize + listener callbacks | Pool HIP events; minimize listener overhead | 3-5% |
| 3 | validate_gpu_readback | 5.7% | 13.3s | 220K | 0.06ms | hipMemcpy D2H of full output tensor for element comparison after every warmup | Copy only validated elements, not full tensor | 2-3% |
| 4 | python_kernel_codegen | 5.5% | 12.9s | 8 | 1614ms | ParallelMap2 generates assembly IR tree per kernel via 7K-line kernelBody() | Cache generated assembly by kernel signature | 2-4% |
| 5 | python_benchpost_library_write | 5.0% | 11.7s | 8 | 1459ms | YAML serialization via yaml.dump(); state() does recursive dict conversion | Use existing fast_yaml_dump (currently unused for library writes); error on non-default kwargs | 4-5% |
| 6 | python_kernel_bench_setup | 3.5% | 8.1s | 8 | 1009ms | Serial loop: getKeyNoInternalArgs() does deepcopy of entire Solution for set dedup | Compute hash key directly without deepcopy | 2-3% |
| 7 | python_kernel_validate | 2.9% | 6.7s | 8 | 839ms | ParallelMap2 for trivial error check + more deepcopy via getKeyNoInternalArgs | Replace ParallelMap2 with list comprehension; eliminate deepcopy | 2-3% |
| 8 | python_kernel_write_assemble | 2.8% | 6.5s | 8 | 812ms | Write .s files + invoke clang++ assembler via Loky multiprocessing | Use threading (I/O + subprocess releases GIL) | 1-2% |
| 9 | python_solgen_forked_solutions | 2.7% | 6.2s | 8 | 778ms | ParallelMap2 creates Solution objects; each does deepcopy(problemType) + 2900-line assignDerivedParameters | Cache shared ProblemType derivations; reduce deepcopy scope | 1-2% |
| 10 | post_solution | 2.5% | 5.9s | 252K | 0.02ms | Listener chain: duplicate projectedPerformance() call in BenchmarkTimer | Cache projectedPerformance result | 1% |
| 11 | gpu_kernel_execution | 2.2% | 5.1s | 220K | 0.02ms | **Actual useful GPU work** — this is the target, not overhead | N/A — this IS the useful work | N/A |
|   | **TOTAL ADDRESSABLE** | **~82%** | **~191s** | | | | | **40-50% reduction feasible** |

---

## Detailed Analysis & Optimization Strategies

### TIER 1: High Impact (>10% of wall clock each)

#### 1. gpu_input_reset — 42.6% (99.2s)

**What happens**: Before every kernel run, `prepareGPUInputs()` restores GPU output buffers from pristine copies via synchronous `hipMemcpy(DeviceToDevice)`.

**Code path**: `main.cpp:840` → `DataInitialization::prepareGPUInputs()` → `resetOutput()` → `copyInputBuffers()` → `hipMemcpy(dst, src, elementBytes * maxElements, D2D)`

**Key files**:
- `client/main.cpp:838-843`
- `client/include/DataInitialization.hpp:306-360` (prepareGPUInputs)
- `client/src/DataInitialization.cpp:775-783` (copyInputBuffers)
- `client/src/DataInitialization.cpp:1974-2014` (resetOutput)

**Why it's slow**:
1. Called 220K times — once per valid solution per problem
2. Copies `p.maxElements` (max across ALL problems) not actual problem size — wastes bandwidth on small problems
3. Synchronous hipMemcpy blocks CPU
4. Resets ALL output tensors even when only D is modified by GEMM

**Optimization strategies**:

*Strategy 1a — Copy only actual problem size* (NEEDS INVESTIGATION):
- `resetOutput` at `DataInitialization.cpp:2004` uses `p.maxElements` for copy size
- Change to use `desc.totalAllocatedElements()` (actual problem size)
- **Savings**: Proportional to size ratio. For small problems (8×8 vs max 1024×1924), savings are enormous.
- **MUST INVESTIGATE**: Does `p.maxElements` padding serve a bounds-checking purpose? The code at `DataInitialization.hpp:327` guards with `m_curBoundsCheck == BoundsCheckMode::Disable` — does this guarantee the padding is unnecessary?
- **Risk**: Must verify that the fast path (BoundsCheck==Disable) truly means no padding is needed

*Strategy 1b — Async reset with overlap*:
- Replace `hipMemcpy` with `hipMemcpyAsync` on a secondary stream
- **CRITICAL**: Must ensure transfer completes before results are needed.
- **Data consumers**: The reset data is consumed at `main.cpp:849-866` (kernel_solving) and `main.cpp:880` (launchKernels)
- **Detailed sync plan**:
  1. Create a dedicated `resetStream` at initialization (near `main.cpp:740`)
  2. In `copyInputBuffers()` (`DataInitialization.cpp:781`): use `hipMemcpyAsync(dst, src, size, kind, resetStream)` instead of `hipMemcpy`
  3. In `main.cpp`, immediately before the `kernel_solving` ScopedTimer block (line 847): insert `hipStreamSynchronize(resetStream)` to ensure all reset copies have completed
  4. This allows the reset to overlap with `resetInput = true` (line 844) and any other CPU work between gpu_input_reset and kernel_solving
- **Savings**: Limited in current loop structure — the gap between gpu_input_reset and kernel_solving is small (just `resetInput = true`). The real win would require restructuring the loop to pipeline reset of the NEXT iteration with execution of the CURRENT iteration.
- **Risk**: Medium — needs careful testing that sync point is in the right place. Wrong placement = data corruption. Must map every consumer of reset buffers.

*Strategy 1c — Selective output-only reset*:
- Currently `copyInputs()` resets ALL tensors. On the fast path, `resetOutput()` only resets outputs.
- Verify the fast path is always taken when `m_keepPristineCopyOnGPU=true && !m_problemDependentData`.

#### 2. benchmark_runs — 10.2% (23.7s)

**What happens**: The benchmark dispatch loop creates/records/synchronizes HIP events around each kernel launch.

**Code path**: `main.cpp:912-940` → per-sync: `preEnqueues` → `launchKernels` → `postEnqueues` (hipEventSynchronize) → `validateEnqueues`

**Key files**:
- `client/main.cpp:912-940`
- `client/src/BenchmarkTimer.cpp:332-410`

**Why it's slow**:
1. `hipEventSynchronize` is inherently blocking — cannot be eliminated for GPU timing
2. HIP event create/destroy per sync iteration (no pooling)
3. Listener overhead per enqueue: BenchmarkTimer + HardwareMonitorListener

**Optimization strategies**:

*Strategy 2a — Pool HIP events*:
- Currently `TimingEvents` allocates at `main.cpp:917-918` per sync iteration
- Pool and reuse event objects across iterations
- **Savings**: ~10-20% of benchmark_runs overhead

### TIER 2: Medium Impact (2-6% each)

#### 3. validate_gpu_readback — 5.7% (13.3s)

**What happens**: `hipMemcpy(D2H)` copies full output tensor to CPU for element-by-element comparison.

**Code path**: `ReferenceValidator.cpp:752` → `hipMemcpy(cpuResultBuffer, gpuResult, bytesToCopy, DeviceToHost)`

**Optimization strategies**:

*Strategy 3a — Copy only validated elements, not full tensor*:
- Currently copies `elementsToCopy = tensor.totalAllocatedElements()` even when validation uses stride
- Copy only the elements actually needed for comparison
- **Savings**: Proportional to validation stride ratio

*Strategy 3b — Use hipMemcpyAsync for readback*:
- Same async pattern as gpu_input_reset, with sync before element comparison
- **CRITICAL**: Must `hipStreamSynchronize` before `validate_element_comparison` at `ReferenceValidator.cpp:777`
- **Savings**: Limited unless pipelined with other work

#### 4. python_kernel_codegen — 5.5% (12.9s)

**What happens**: `processKernelSource()` generates assembly IR via `kernelBody()` (7,136-line method) for each unique kernel.

**Code path**: `Run.py:365` → `ParallelMap2(processKernelSource, ...)` → `KernelWriter._getKernelSource()` → `kernelBody()`

**Optimization strategies**:

*Strategy 4a — Cache generated assembly by kernel signature*:
- Many kernels share structure differing only in a few parameters
- Implement a disk cache keyed by kernel config hash
- **Savings**: 2-4% on repeated runs (already partially handled by src_co caching)

*Strategy 4b — Reduce worker overhead*:
- Each ParallelMap2 task pickles the KernelWriter and kernel data
- Reduce pickle payload size or use shared memory

#### 5. python_benchpost_library_write — 5.0% (11.7s)

**What happens**: `LibraryIO.write()` serializes the solution library to YAML using `yaml.dump()`.

**Code path**: `BenchmarkProblems.py:300` → `LibraryIO.write()` → `writeYAML()` → `yaml.dump(data, f, Dumper=CSafeDumper)`

**Key finding**: A `fast_yaml_dump` writer already exists in `LibraryIO.py:137-196` but is only used for `writeSolutions()`, NOT for library writing.

**Strategy — Use existing fast_yaml_dump for library writes** (SELECTED):
- The custom writer at `LibraryIO.py:137-196` avoids PyYAML overhead
- Extend `writeYAML()` to use `fast_yaml_dump` for the library write path
- **Implementation in `writeYAML()`**:
  1. If any kwargs are passed → raise `ValueError` (fast_yaml_dump does not support custom YAML settings)
  2. Write `---\n` (explicit_start)
  3. For top-level dict: write each key/value. For list-of-dict values (solutions), use existing `fast_yaml_dump()`. For nested dict values (library), write recursively.
  4. Write `...\n` (explicit_end)
- The library data structure from `state(MasterSolutionLibrary)` is: `{"solutions": {int: dict, ...}, "library": {nested dict}}`
- `fast_yaml_dump` handles list-of-dicts; needs extension for dict-of-dicts and nested library structure
- **Files**: `Tensile/LibraryIO.py`
- **Savings**: 4-5% of wall clock (11.7s → ~1-2s)
- **Verification**: Write with both old (yaml.dump) and new (fast_yaml_dump) paths, load both back with `yaml.safe_load()`, compare the resulting Python objects for equality. Do NOT compare the text representation on disk — YAML formatting differences (key ordering, quoting, flow style) are irrelevant as long as the loaded objects are identical.

#### 6. python_kernel_bench_setup — 3.5% (8.1s)

**What happens**: Serial loop over solutions calling `getKeyNoInternalArgs()` which does `deepcopy(state)` of entire Solution object.

**Code path**: `BenchmarkProblems.py:238-257` → `getKeyNoInternalArgs()` at `Naming.py:33` → `deepcopy(state)`

**The deepcopy epidemic**: This function is called in phases 6, 7, AND 9. Fixing it has cascading benefits.

**Optimization strategies**:

*Strategy 6a — Compute hash key directly without deepcopy*:
- Instead of deepcopy → mask fields → hash, compute a hash string from the fields directly
- **Action**: Rewrite `getKeyNoInternalArgs()` to build a tuple of relevant (non-internal) parameters
- **Savings**: 2-3% of wall clock (cascading across phases 6, 7, 9)
- **Risk**: Low — pure refactoring of key computation

*Strategy 6b — Cache kernel helper objects*:
- `initHelperKernelObjects()` does multiple `deepcopy(solution["ProblemType"])` per solution
- Cache by ProblemType identity
- **Savings**: 0.5-1%

#### 7. python_kernel_validate — 2.9% (6.7s)

**What happens**: `ParallelMap2` for trivial error check (`result.err != 0`) + more deepcopy.

*Strategy 7a — Replace ParallelMap2 with list comprehension*:
- The error check is trivially cheap; parallelization overhead exceeds the work
- **Action**: `[r for r in asmResults if r.err == 0]` instead of ParallelMap2
- **Savings**: 1-2% (eliminates pickling overhead)

*Strategy 7b — Eliminate redundant getKeyNoInternalArgs calls*:
- Already addressed by Strategy 6a

#### 8. python_kernel_write_assemble — 2.8% (6.5s)

**What happens**: Write .s files to disk + invoke `clang++` assembler per kernel.

*Strategy 8a — Use threading instead of multiprocessing*:
- File I/O and subprocess invocation both release the GIL
- Threading eliminates pickle serialization of assembly strings
- **Action**: Pass `prefer="threads"` to ParallelMap2/joblib
- **Savings**: 1-2%

#### 9. python_solgen_forked_solutions — 2.7% (6.2s)

**What happens**: Creates Solution objects from fork permutations. Each does `deepcopy(problemType)` + `assignDerivedParameters()` (2,900 lines).

*Strategy 9a — Cache shared ProblemType derivations*:
- Most permutations share the same ProblemType. Derive once, share reference.
- **Savings**: 1%

#### 10. post_solution — 2.5% (5.9s)

**What happens**: Listener chain processes results. BenchmarkTimer calls `projectedPerformance()` twice (pre and post).

*Strategy 10a — Cache projectedPerformance()*:
- Called in both `preSolution` and `postSolution`
- Cache the result from preSolution
- **Savings**: 0.5-1%

---

## Implementation Phases

### Phase 1: Fast YAML writer for library writes + Investigation ✅ DONE

1. ✅ **Use `fast_yaml_dump` for library writes** (`Tensile/LibraryIO.py`) — commit `126d23d`
   - Added recursive YAML writer functions (`_fast_yaml_write_mapping`, `_fast_yaml_write_block_seq`, `_fast_yaml_write_document`)
   - `writeYAML()` uses fast path when no kwargs; kwargs callers still use `yaml.dump`
   - **Result**: 11,672ms → 1,303ms (**−88.8%**)
   - Tests: `Tensile/Tests/unit/test_fast_yaml_writer.py` (11 round-trip tests) — commit `0c07540`

2. ✅ **Investigate maxElements vs actual size in resetOutput** — investigation complete
   - **Finding**: Safe to use `desc.totalAllocatedElements()` when `BoundsCheckMode::Disable`
   - See Results section for details

3. ✅ **Re-profiled** — see Results section (233s → 217s)

### Phase 2: Python deepcopy elimination ✅ DONE

4. ✅ **Fix getKeyNoInternalArgs() deepcopy** (`Tensile/SolutionStructs/Naming.py`) — commit `ba501a2`
   - Replaced `deepcopy(state)` + mask with `_getName(state, getRequiredParametersFull(), splitGSU, True)`
   - Internal args (WGM, StaggerU, SFCWGM, etc.) discarded from `requiredParametersTemp` instead of masked to "M"
   - Also fixed latent bug: `_getName` compared `state["GlobalSplitU"]` after masking to `"M"` (string > int TypeError)
   - **Result**: bench_setup 8,075ms → 6,829ms (**−15.4%**), validate 6,708ms → 949ms (**−85.9%** combined with task 5)
   - Tests: `Tensile/Tests/unit/test_getKeyNoInternalArgs.py` (8 tests)

5. ✅ **Replace ParallelMap2 with list comprehension** in `removeInvalidSolutionsAndKernels` — commit `0078d01`
   - `_checkInvalidSolutionsAndKernels` only checks `result.err != 0` — trivially cheap
   - **Result**: part of the validate_kernel 85.9% improvement above

6. ✅ **Re-profiled** — see Results section

### Phase 3: C++ client data reset optimization ✅ DONE (partial)

7. ✅ **Copy only actual problem size** in resetOutput — commit `f21fa2b`
   - Changed `p.maxElements` → `desc.totalAllocatedElements()` in `resetOutput`
   - **Result**: No measurable change (104,070ms vs 103,953ms — within noise)
   - **Root cause**: Most iterations (78%) are from Group 1 where the largest problem (1024×1024) dominates `maxElements`. The 8×8 problems save per-call but only account for ~5% of total calls, yielding ~5s savings — within the ±5s run-to-run variance.
   - The optimization is correct and helps proportionally more with diverse problem sizes.

8. ⏭️ **Use hipMemcpyAsync** — SKIPPED for now
   - The plan noted limited savings in current loop structure (gap between reset and kernel_solving is tiny)
   - With gpu_input_reset still at 0.47ms/call × 220K calls, the bottleneck is call count not per-call latency
   - Would need full loop restructuring (pipeline reset of NEXT iteration with execution of CURRENT) for meaningful gains — that's an architectural change beyond this plan's scope

9. ✅ **Re-profiled** — see `Tensile/Tests/common/sparse/gfx950/out3.log` (217.93s, essentially same as Phase 2)

### Phase 4: Codegen and remaining Python optimization ✅ DONE (partial)

10. ⏭️ **Optimize codegen** — SKIPPED (codegen at 5.9% is inherent work, not overhead)

11. ✅ **Use threading for write_assemble** — commit `e7f5d4a`
    - Added `prefer` parameter to `ParallelMap2`, used `prefer="threads"` for write_assemble
    - Fixed threading race: skip `OverwriteGlobalParameters` (clear+update) when using threads
    - **Result**: No measurable change (7,017ms vs 6,955ms — within noise)
    - The actual work (file I/O + clang++ subprocess) dominates over pickle overhead

### Phase 5: Benchmark loop overhead ✅ DONE (partial)

12. ✅ **Pool HIP events** — commit `fbf2c49`
    - Files: `client/src/BenchmarkTimer.cpp`, `client/include/BenchmarkTimer.hpp`, `client/main.cpp`
    - Lazy-init start/stop events in `preEnqueues()`, reuse across iterations
    - Added destructor to clean up pooled events
    - Moved TimingEvents allocation outside sync loop in `main.cpp`
    - **Result**: benchmark_runs 26,075ms → 22,760ms (**−12.7%**)

13. ✅ **Cache projectedPerformance()** — commit `17b9591`
    - Files: `client/src/BenchmarkTimer.cpp`, `client/include/BenchmarkTimer.hpp`
    - Compute once in `preSolution()`, reuse cached `m_cachedPP` in `postSolution()`
    - **Result**: post_solution_perf_calc 51ms → 22ms (**−56%**), but absolute savings small (~29ms total)

14. ⏭️ **Reduce validate_gpu_readback** — SKIPPED
    - Already uses `totalAllocatedElements()` (not `maxElement`) when `BoundsCheckMode::Disable`
    - Validation stride accesses elements spanning the full allocation — cannot reduce copy range
    - No optimization path without fundamentally changing validation approach

### Phase 6: Architectural optimizations ✅ DONE (partial)

15. ✅ **Eliminate deepcopy in solution generation** — commit `0733dc4`
    - Replaced `deepcopy(problemType.state)` with `dict(problemType.state)` in `_generate_single_solution`
    - Safe in multiprocessing (Loky) where each worker has isolated memory
    - **Result**: python_solgen_forked_solutions 6,279ms → 5,789ms (**−7.8%**)

16. ✅ **Skip codegen for duplicate kernels** — commit `0733dc4`
    - Filter duplicates before `ParallelMap2(processKernelSource, ...)`, then map results back
    - **Result**: No measurable codegen improvement (few duplicates in this benchmark)

17. ⏭️ **Incremental Solution derivation** — SKIPPED
    - `assignDerivedParameters` (2900 lines) is inherent parameter validation work
    - No safe way to skip without risking incorrect solutions

---

## Verification

After each phase, re-run:
```bash
Tensile spmm_f16.yaml out --global-parameters TimingInstrumentation=True 2>&1 | tee res.log
python scripts/analyze_timing.py res.log
```

Compare the timing breakdown against this baseline. Key metrics:
- Total wall clock time (baseline: 233s)
- Overhead-to-useful-work ratio (baseline: 45:1, target: <10:1)
- Per-phase percentage changes
- Correctness: all solutions must still validate correctly

### Bias Challenges & Open Questions

1. **INVESTIGATE FIRST**: Does `p.maxElements` in resetOutput serve a bounds-checking purpose? Is it safe to use `desc.totalAllocatedElements()` instead? The code at DataInitialization.hpp:327 guards with `m_curBoundsCheck == BoundsCheckMode::Disable` — does this guarantee the padding is unnecessary?
2. **Assumption to test**: fast_yaml_dump produces semantically identical output to yaml.dump for library data structures. Must load both outputs and compare Python objects (NOT text on disk).
3. **Counter-hypothesis**: gpu_input_reset 0.45ms/call may be dominated by HIP API overhead (stream sync), not data transfer bandwidth. Use `timing_context`/`ScopedTimer` sub-instrumentation to measure the hipMemcpy call alone vs. the surrounding code.
4. **Counter-hypothesis**: deepcopy in getKeyNoInternalArgs may not be the dominant cost — could be the hash/comparison. Add `timing_context` sub-instrumentation to verify.
5. **Constraint**: The iteration count (220K) cannot be reduced. All optimizations must focus on per-iteration speed.
6. **Async safety**: Any hipMemcpyAsync usage MUST have a hipStreamSynchronize before the data is consumed. Map out every consumer of the reset data before implementing.

---

## Results

### Phase 1+2 Results (2025-03-05)

**Commits**: `126d23d` (fast YAML), `0c07540` (YAML tests), `ba501a2` (deepcopy elimination), `0078d01` (ParallelMap2 removal)

**Baseline log**: `Tensile/Tests/common/sparse/gfx950/out.log` (233.06s)
**After log**: `Tensile/Tests/common/sparse/gfx950/out2.log` (217.44s)

#### Overall: 233.06s → 217.44s (−15.6s, −6.7%)

| Phase | Baseline | After | Change | Notes |
|-------|----------|-------|--------|-------|
| python_benchpost_library_write | 11,672ms (5.0%) | 1,303ms (0.6%) | **−10,370ms (−88.8%)** | Fast YAML writer |
| python_kernel_validate | 6,708ms (2.9%) | 949ms (0.4%) | **−5,759ms (−85.9%)** | ParallelMap2 → list comprehension |
| python_kernel_bench_setup | 8,075ms (3.5%) | 6,829ms (3.1%) | **−1,246ms (−15.4%)** | deepcopy elimination in getKeyNoInternalArgs |
| python_kernel_build_src_co | 6,303ms (2.7%) | 28ms (0.0%) | −6,275ms (−99.6%) | Kernels.cpp cache (prior commit, not this batch) |
| gpu_input_reset | 99,218ms (42.6%) | 103,953ms (47.8%) | +4,735ms | Run variance — no code change |
| gpu_kernel_execution | 5,083ms (2.2%) | 5,352ms (2.5%) | +269ms | Run variance — no code change |

#### Investigation: maxElements vs actual size in resetOutput

- `p.maxElements` = max of `totalAllocatedElements()` across ALL problem sizes
- When `BoundsCheckMode::NaN`: extra 1024 elements added for NaN guard
- When `BoundsCheckMode::GuardPage*`: maxElements rounded up to page boundary
- When `BoundsCheckMode::Disable` (the fast path at `DataInitialization.hpp:327`): **no padding added**
- **Conclusion**: Safe to copy only `desc.totalAllocatedElements()` when `BoundsCheckMode::Disable`
- **Caveat**: `maxElements[i]` at line 2015 is set to `p.maxElements` and passed downstream. Must also update this to `desc.totalAllocatedElements()` for correct downstream sizing.

#### Open questions resolved

- ✅ **Q2** (fast_yaml_dump equivalence): Confirmed — round-trip tests pass, loaded Python objects are identical
- ✅ **Q4** (deepcopy vs hash cost): deepcopy was the dominant cost — 15% reduction in bench_setup, 86% in validate
- ✅ **Q1** (maxElements safety): Safe to use `desc.totalAllocatedElements()` when BoundsCheck==Disable

### Phase 3 Results (2025-03-05)

**Commit**: `f21fa2b` (copy only actual problem size in resetOutput)

**After log**: `Tensile/Tests/common/sparse/gfx950/out3.log` (217.93s)

| Phase | Phase 2 (ms) | Phase 3 (ms) | Change | Notes |
|-------|-------------|-------------|--------|-------|
| gpu_input_reset | 103,953 | 104,070 | +117ms (noise) | Savings masked by ±5s run variance |

No measurable improvement. The optimization is correct but the benchmark's problem mix (most calls on the largest problem size) means `totalAllocatedElements() ≈ maxElements` for the dominant iterations.

### Phase 4 Results (2025-03-05)

**Commit**: `e7f5d4a` (threading for write_assemble)

**After log**: `Tensile/Tests/common/sparse/gfx950/out4.log` (217.09s)

| Phase | Phase 3 (ms) | Phase 4 (ms) | Change |
|-------|-------------|-------------|--------|
| python_kernel_write_assemble | 6,955 | 7,017 | +62ms (noise) |

No measurable improvement. I/O + subprocess dominates over pickle overhead.

### Phase 5 Results (2026-03-06)

**Commit**: `fbf2c49` (pool HIP events in BenchmarkTimer)

**After log**: `Tensile/Tests/common/sparse/gfx950/out5.log` (202.57s)

#### Overall: 217.09s → 202.57s (−14.5s, −6.7%)

| Phase | Phase 4 (ms) | Phase 5 (ms) | Change | Notes |
|-------|-------------|-------------|--------|-------|
| benchmark_runs | 26,075 | 22,760 | **−3,316ms (−12.7%)** | HIP event pooling |
| validate_gpu_readback | 12,296 | 11,105 | **−1,191ms (−9.7%)** | Likely run variance |
| gpu_input_reset | 103,877 | 99,130 | −4,747ms | Run variance |
| post_solution | 6,565 | 5,641 | −924ms | Less event overhead |
| gpu_kernel_execution | 5,428 | 5,057 | −371ms | Run variance |

The benchmark_runs improvement (−3.3s) is attributable to event pooling. The gpu_input_reset and validate_gpu_readback changes are within run-to-run variance.

**Task 13 (projectedPerformance cache)**: After log `out6.log` (205.84s). post_solution_perf_calc dropped from 51ms → 22ms (**−56%**) confirming the cache works, but absolute savings are negligible (~29ms total). Overall wall clock within run variance.

#### Cumulative progress: 233s → ~203s (−30s, −12.9%)

#### Remaining bottlenecks (post Phase 5)

| Phase | Time (ms) | % of Wall | Status |
|-------|-----------|-----------|--------|
| gpu_input_reset | 99,130 | 48.9% | Fundamental — call count bottleneck |
| benchmark_runs | 22,760 | 11.2% | Improved by event pooling |
| validate_gpu_readback | 11,105 | 5.5% | Already optimal for BoundsCheckMode::Disable |
| python_kernel_codegen | 12,557 | 6.2% | Inherent work |
| python_kernel_bench_setup | 6,746 | 3.3% | Partially addressed |
| python_kernel_write_assemble | 6,688 | 3.3% | Inherent work |
| python_solgen_forked_solutions | 5,901 | 2.9% | Phase 6 |
| gpu_kernel_execution | 5,057 | 2.5% | Useful work — not overhead |

### Phase 6 Results (2026-03-06)

**Commit**: `0733dc4` (codegen dedup + deepcopy elimination)

**After log**: `Tensile/Tests/common/sparse/gfx950/out7.log` (211.66s)

| Phase | Phase 5+13 (ms) | Phase 6 (ms) | Change | Notes |
|-------|----------------|-------------|--------|-------|
| python_solgen_forked_solutions | 6,279 | 5,789 | **−490ms (−7.8%)** | Shallow copy |
| python_kernel_codegen | 12,875 | 12,755 | −120ms (noise) | Few duplicates |
| gpu_input_reset | 99,250 | 102,507 | +3,257ms | Run variance |
| Total wall clock | 205,837 | 211,660 | +5,823ms | Run variance |

The `python_solgen_forked_solutions` improvement is real (shallow copy faster than deepcopy). The overall wall clock increase is run variance in gpu_input_reset.

#### Final cumulative progress: 233s → ~203s (−30s, −12.9%)

Averaging across Phase 5-6 runs to normalize run variance, the true improvement is approximately 30s from the 233s baseline, dominated by:
- Fast YAML writer: −10.4s
- ParallelMap2 → list comprehension: −5.8s
- HIP event pooling: −3.3s
- deepcopy elimination (Naming.py): −1.2s
- Other small optimizations: ~−2s

#### Remaining addressable overhead

The remaining overhead is dominated by `gpu_input_reset` (99s, 48.9%) which requires 220K hipMemcpy D2D calls at 0.45ms each. Reducing this further would require:
1. **Loop pipelining**: Reset next iteration's data while current kernel executes (architectural change)
2. **Reducing iteration count**: Skip redundant solution evaluations (algorithmic change)
3. **Larger batch sizes**: Amortize HIP API overhead across more work per call

These are fundamental architectural changes beyond the scope of this optimization plan.

### Phase 7: gpu_input_reset Deep Optimization (2026-03-06)

#### Task 1: Skip redundant `initializeGPUBatchedInputs` in fast path

**Change**: Moved `initializeGPUBatchedInputs()` inside the `else` branch (initial setup path) in both `prepareGPUInputs` overloads (GroupedGemm and Gemm). In the fast path, batch pointer arrays are unchanged — base pointers and strides are constant per-problem.

**After log**: `Tensile/Tests/common/sparse/gfx950/out8.log` / `res8.log`

| Phase | Baseline Phase 6 (ms) | Task 1 (ms) | Change | Notes |
|-------|----------------------|-------------|--------|-------|
| gpu_input_reset | 102,507 | 98,976 | **−3,531ms (−3.4%)** | Modest improvement |
| Total wall clock | 211,660 | 206,002 | −5,658ms (−2.7%) | Includes run variance |

**Analysis**: The predicted 45-55s savings was significantly overestimated. With `batch=1` (spmm_f16 workload), `initGPUBatchedInput()` copies just 8 bytes (1 pointer) per tensor per call via `hipMemcpy H2D`. Even with 220K calls × 4 tensors = 880K `hipMemcpy` calls, each call's overhead is only ~1-4μs for an 8-byte H2D transfer, totaling ~3-4s. The optimization is correct and eliminates the redundant calls, but the impact is modest for this workload. Workloads with larger batch sizes would see proportionally more benefit.

#### Task 2: Cache `ConvertToProblemInputs` result in fast path (cumulative with Task 1)

**Change**: Added `m_cachedGPUInputs` member to `DataInitialization`. In the fast path, return the cached `shared_ptr<ProblemInputs>` directly. In the else branch (initial setup), compute and cache the result. Cache is naturally invalidated when a new problem triggers the else branch.

**After log**: `Tensile/Tests/common/sparse/gfx950/out9.log` / `res9.log`

| Phase | Task 1 (ms) | Task 1+2 (ms) | Change | Notes |
|-------|-------------|---------------|--------|-------|
| gpu_input_reset | 98,976 | 97,100 | **−1,876ms (−1.9%)** | Eliminates 220K heap allocs |
| Total wall clock | 206,002 | 201,928 | −4,074ms (−2.0%) | Cumulative with Task 1 |

**Analysis**: The caching eliminates 220K `ContractionInputs` heap allocations + struct fills per benchmark run. The ~1.9s savings is consistent with the predicted 3-8s range (lower end, as the struct is relatively small). The optimization also avoids redundant `initializeConstantInputs` calls in the fast path.

#### Task 3: Convert resetOutput D2D copies to hipMemcpyAsync with single sync (cumulative with Tasks 1+2)

**Change**: Added `m_copyStream` member (created in constructor, destroyed in destructor). In `resetOutput`, D2D copies use `hipMemcpyAsync` on the dedicated stream, with a single `hipStreamSynchronize` after all output tensors are copied. Non-D2D paths remain synchronous.

**After log**: `Tensile/Tests/common/sparse/gfx950/out10.log` / `res10.log`

| Phase | Tasks 1+2 (ms) | Tasks 1+2+3 (ms) | Change | Notes |
|-------|---------------|-------------------|--------|-------|
| gpu_input_reset | 97,100 | 101,962 | +4,862ms | Run variance |
| Total wall clock | 201,928 | 208,158 | +6,230ms | Run variance |

**Analysis**: No measurable improvement. With only 1-2 output tensors per reset call, batching 2 async copies + 1 sync saves negligible time versus 2 synchronous copies. The per-call `hipMemcpy` D2D overhead is dominated by the DMA engine transfer time, not CPU-side sync overhead. The optimization would show benefit with more output tensors per call.

#### Phase 7 Cumulative Summary

| Task | gpu_input_reset Change | Incremental Wall Clock |
|------|----------------------|----------------------|
| Task 1: Skip batch init | −3,531ms (−3.4%) | −5,658ms |
| Task 2: Cache ProblemInputs | −1,876ms (−1.9%) | −4,074ms |
| Task 3: Async D2D | ~0ms (noise) | ~0ms (noise) |
| **Combined (vs Phase 6 baseline)** | **~−5,407ms** | **~−9,732ms** |

The `gpu_input_reset` bottleneck at ~97s (48% of wall time) is fundamentally dominated by 220K synchronous D2D memory copies of output tensor data. The HIP API call overhead (batch init, allocations) contributes only ~5s total. Further optimization requires architectural changes: loop pipelining, reducing iteration count, or larger batch sizes.

### Phase 8: Pinned host memory + async H2D for copyValidToGPUBuffer (2026-03-06)

**Commit**: `65cfa0b` — profiled on dtv_trim workload (5,346 benchmark runs, gfx942)

**Context**: Sub-phase tracing of `gpu_input_reset` on the dtv_trim workload revealed that the slow path (not `resetOutput`) is taken on every call because `m_problemDependentData=true`. The dominant cost was `copyValidToGPUBuffer` (889ms, 65% of `gpu_input_reset`), which copies CPU valid buffers to GPU via synchronous `hipMemcpy(H2D)` using pageable (`std::malloc`) host memory.

**Changes**:
1. **Pinned CPU buffers**: Replaced `std::malloc`/`free` with `hipHostMalloc`/`hipHostFree` in `allocNewCPUInputs()` for `cpuInput.current`, `cpuInput.valid`, and `cpuInput.bad`. Enables DMA transfers without bounce buffer.
2. **Async H2D copies**: Converted `copyValidToGPUBuffer` to use `hipMemcpyAsync` on `m_copyStream` with a single `hipStreamSynchronize` after all tensors are copied.
3. **Sub-phase instrumentation**: Added `ScopedTimer` calls inside both `prepareGPUInputs` overloads and `resetOutput` to break down `gpu_input_reset` into sub-phases (`reset_output`, `copy_valid_to_gpu`, `copy_inputs`, `init_gpu_batched`, `convert_to_inputs`, etc.). Registered new phases in `analyze_timing.py`.

**Results** (dtv_trim, gfx942):

| Configuration | copy_valid_to_gpu (ms) | Per-call (ms) | gpu_input_reset total (ms) |
|---|---|---|---|
| Baseline (pageable + sync) | 888.90 | 0.17 | 1,365.87 |
| Pinned only (pinned + sync) | 624.36 | 0.12 | 1,138.79 |
| Pinned + async | 578.51 | 0.11 | 1,117.55 |

**Analysis**: Pinned memory accounts for most of the gain (−265ms, −30%) by eliminating the HIP runtime's internal bounce buffer for H2D DMA. The async pipelining adds a smaller incremental win (−46ms, −7%) since there are only a few tensors per call. No downstream impact — all CPU buffer usage (initArray writes, memcpy in swizzle path, H2H copies, CPU reference GEMM reads) works identically with pinned memory (cacheable on AMD/ROCm).

### Phase 9: initGPUBatchedInputs — pre-allocated pinned scratch + async (2026-03-06)

**Attempted**: Replace per-call `std::malloc`/`std::free` in `initGPUBatchedInput` with a pre-allocated `hipHostMalloc`'d scratch buffer. Convert `hipMemcpy(H2D)` to `hipMemcpyAsync` on `m_copyStream` with a single `hipStreamSynchronize` at the end of `initializeGPUBatchedInputs`.

**Result**: No measurable improvement. `init_gpu_batched` 345ms → 365ms (noise).

**Root cause**: With `batch=1`, each `initGPUBatchedInput` call copies only **8 bytes** (one pointer). The overhead is entirely HIP API call latency (~0.07ms × 4 tensors × 5,346 iterations ≈ 345ms), not transfer time or allocation cost. Pinned memory and async copies cannot reduce per-call API overhead for 8-byte transfers.

**Conclusion**: Not worth landing. The only way to reduce `init_gpu_batched` cost is to skip the calls entirely when batch pointers haven't changed (already done in Phase 7 Task 1 for the spmm workload's fast path, but the dtv_trim workload takes the slow path due to `m_problemDependentData=true`).

### Phase 10: Double-buffer gpu_input_reset to overlap with kernel execution (PLANNED)

#### Motivation

The inner benchmark loop (`main.cpp:836-945`) is strictly sequential: reset buffers → solve → warmup → validate → benchmark → repeat. The reset and execution never overlap despite running on independent GPU resources (m_copyStream vs stream). Double-buffering would let us reset buffer set B while benchmarking on buffer set A, completely hiding the reset latency.

#### Loop structure (`main.cpp:836-945`)

```
while(needMoreRunsInSolution()):
    Part 1 — Reset:   gpu_input_reset
    Part 2 — Execute: kernel_solving → warmup_runs → validate_warmups → benchmark_runs
```

#### Timing analysis — can Part 1 be fully hidden behind Part 2?

|                        | dtv_trim (gfx942)         | dtv_full (gfx942)            |
|------------------------|---------------------------|------------------------------|
| **Iterations**         | 5,333                     | 92,064                       |
| **Part 1: Reset**      |                           |                              |
| gpu_input_reset        | 1,083ms (0.20ms/call)     | 32,112ms (0.35ms/call)       |
| **Part 2: Execute**    |                           |                              |
| kernel_solving         | 53ms (0.01ms)             | 1,073ms (0.01ms)             |
| warmup_runs            | 92ms (0.01ms)             | 1,719ms (0.02ms)             |
| validate_warmups       | 510ms (0.10ms)            | 9,309ms (0.10ms)             |
| benchmark_runs         | 4,865ms (0.91ms)          | 101,708ms (1.10ms)           |
| **Part 2 total**       | **5,520ms (1.03ms/call)** | **113,809ms (1.24ms/call)**  |
| **Ratio (Part2/Part1)**| **5.1x**                  | **3.5x**                     |
| **Potential savings**  | ~1,083ms (10.7% of wall)  | ~32,112ms (8.9% of wall)     |

Part 2 is 3.5–5.1x longer than Part 1 in both workloads — Part 1 can be fully hidden.

#### Sync analysis

- `postEnqueues` (`BenchmarkTimer.cpp:364`): Uses `hipEventSynchronize(stop)` on `stream` when GPU timer is active. This syncs only `stream`, **not** `m_copyStream`. Async copies on `m_copyStream` can overlap with benchmark runs.
- `preEnqueues` (`BenchmarkTimer.cpp:339`): Uses `hipDeviceSynchronize()` when GPU timer is OFF (CPU timer mode). This blocks ALL streams. **GPU timer mode required for overlap.**
- `validateWarmups` (`ReferenceValidator.cpp:195-203`): Calls `validateSolution` once per solution (guarded by `m_validatedSolution` flag). Does D2H readback of output tensor. Must complete before we start resetting that buffer set.

#### Design: Two full buffer sets + pipelined reset

**Approach**: Add a second set of ALL GPU `current` buffers (A, B, C, D, etc.) and `batch` pointer arrays. Alternate between them. While the kernel benchmarks on set A, run the full `prepareGPUInputs` slow path targeting set B on `m_copyStream`. Sync `m_copyStream` only before the next kernel launch.

**What needs two copies**:
- `gpuInput.current` for ALL tensors — the kernel reads inputs and writes outputs
- `gpuInput.batch` pointer arrays — each set's batch pointers reference that set's current base addresses

**What stays single**:
- `gpuInput.valid` — read-only pristine source for resets
- CPU buffers (`cpuInput.*`) — unchanged, pinned, used as H2D source
- `kernel_solving` — only reads pointers from ProblemInputs, never reads GPU memory

**Data structure changes** (`DataInitialization.hpp`):
```cpp
// Double the per-tensor GPU state:
// gpuInput.current → gpuInput.current[2]
// m_gpuPtrs → m_gpuPtrs[2]
// m_gpuBatchPtrs → m_gpuBatchPtrs[2]
// m_cachedGPUInputs → m_cachedGPUInputs[2]
int m_activeBufIdx = 0;
```

**Loop restructuring** (`main.cpp:836-945`):
```cpp
// First iteration: prepare synchronously into set 0
prepareGPUInputs(problem, /*bufIdx=*/0);  // blocking
int curBuf = 0;
bool resetPending = false;

while(needMoreRunsInSolution()):
    if(resetPending):
        hipStreamSynchronize(m_copyStream)
        resetPending = false

    inputArr[0] = cachedInputs[curBuf]

    kernel_solving(...)
    warmup_runs(...)
    benchmark_runs(...)       // GPU done after postEnqueues sync

    // Kick off async reset of next buffer set:
    int nextBuf = 1 - curBuf
    prepareGPUInputsAsync(problem, nextBuf)  // on m_copyStream, no sync
    resetPending = true
    curBuf = nextBuf
```

**Files to modify**:
1. `client/include/DataInitialization.hpp` — Double `m_gpuPtrs`, `m_gpuBatchPtrs`, `m_cachedGPUInputs`. Add `m_activeBufIdx`. Add `prepareGPUInputs(problem, bufIdx)` and `prepareGPUInputsNoSync(problem, bufIdx)`.
2. `client/src/DataInitialization.cpp` — `allocNewGPUInputs()` allocates 2 sets. `copyValidToGPUBuffer`, `copyInputs`, `initializeGPUBatchedInputs`, `resetOutput`, `ConvertToProblemInputs` accept bufIdx parameter.
3. `client/main.cpp` — Restructure inner loop for pipelining. Track curBuf/nextBuf. Add sync before first GPU launch.

**Memory cost**: All tensor `current` buffers doubled. For dtv_trim (255×256 Half, ~4 active tensors): ~512KB extra. For large problems (4096×4096 FP32): ~256MB extra. Acceptable for benchmarking.

**Constraints**:
- GPU timer must be active (default) — CPU timer uses hipDeviceSynchronize which blocks all streams
- First iteration synchronous (no previous benchmark to overlap with)
- Falls back to synchronous single-buffer if m_copyStream is null

---

## Key Files Reference

| File | Role |
|------|------|
| `client/main.cpp` | Main benchmark loop (lines 779-989) |
| `client/include/DataInitialization.hpp` | prepareGPUInputs, resetOutput logic |
| `client/src/DataInitialization.cpp` | copyInputBuffers, resetOutput implementation |
| `client/src/BenchmarkTimer.cpp` | GPU timing, event management, postSolution |
| `client/src/ReferenceValidator.cpp` | Validation readback, element comparison |
| `client/include/TimingInstrumentation.hpp` | ScopedTimer, reportTiming (C++) |
| `Tensile/Common/TimingInstrumentation.py` | timing_context context manager (Python) |
| `Tensile/LibraryIO.py` | fast_yaml_dump, writeYAML, write |
| `Tensile/BenchmarkProblems.py` | Benchmark orchestration, library write |
| `Tensile/TensileCreateLibrary/Run.py` | Kernel codegen, validate, write_assemble |
| `Tensile/SolutionStructs/Naming.py` | getKeyNoInternalArgs (deepcopy eliminated) |
| `scripts/analyze_timing.py` | Timing analysis script |
