# MIOpen Solver Selection Architecture

A deep analysis of how MIOpen selects, benchmarks, caches, and executes GPU kernel "solvers" for convolution and other operations.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [API Entry Points (Three Generations)](#2-api-entry-points-three-generations)
3. [The Find Pipeline — How a Convolution Request Becomes a Kernel](#3-the-find-pipeline)
4. [Find Mode System](#4-find-mode-system)
5. [Find Enforce / Tuning Policy](#5-find-enforce--tuning-policy)
6. [Database Layer](#6-database-layer)
7. [Solver Framework](#7-solver-framework)
8. [Algorithm-Family Finders](#8-algorithm-family-finders)
9. [Benchmarking and Solution Ranking](#9-benchmarking-and-solution-ranking)
10. [Fallback System](#10-fallback-system)
11. [Non-Convolution Operations](#11-non-convolution-operations)
12. [Environment Variable Reference](#12-environment-variable-reference)
13. [Key Source File Index](#13-key-source-file-index)

---

## 1. High-Level Overview

MIOpen's solver selection can be thought of as a pipeline with four major subsystems, each acting as a black box:

```
User API Call
     |
     v
+------------------+     +------------------+     +-----------------+     +-------------+
| API Translation  | --> | Find / Search    | --> | Database Layer  | --> | Execution   |
| (3 generations)  |     | (mode-dependent) |     | (cache/persist) |     | (invoker)   |
+------------------+     +------------------+     +-----------------+     +-------------+
```

- **API Translation**: Converts the user's C API call into an internal `conv::ProblemDescription` and routes to the appropriate find path.
- **Find / Search**: Depending on the FindMode, either looks up cached results, runs heuristic estimation, or performs exhaustive benchmarking of all applicable solvers.
- **Database Layer**: A two-tier system of read-only system databases (shipped with MIOpen) and read-write user databases (built locally through find operations).
- **Execution**: Compiles GPU kernels, registers invokers with the handle, and dispatches the selected solution.

---

## 2. API Entry Points (Three Generations)

MIOpen exposes three distinct API generations for convolution. All three generations ultimately funnel into the same internal solver selection machinery, but they differ in how the user interacts with the search and execution lifecycle.

### 2.1 Classic Find API (Generation 1)

The original two-step workflow: **find** algorithms, then **execute** with a chosen algorithm.

**Forward Convolution:**

| Step | API Function | Purpose |
|------|-------------|---------|
| 0 | `miopenConvolutionForwardGetWorkSpaceSize(handle, wDesc, xDesc, yDesc, &size)` | Query max workspace needed |
| 1 | `miopenFindConvolutionForwardAlgorithm(handle, xDesc, x, wDesc, w, yDesc, y, reqAlgoCount, &retAlgoCount, perfResults, workSpace, wsSize, exhaustiveSearch)` | Search and benchmark algorithms. Returns `miopenConvAlgoPerf_t[]` with algo, time, memory. |
| 2 | `miopenConvolutionForward(handle, alpha, xDesc, x, wDesc, w, algo, beta, yDesc, y, ws, wsSize)` | Execute with previously found algorithm |

**Backward Data** and **Backward Weights** follow the same pattern with `miopenFindConvolutionBackwardDataAlgorithm` / `miopenConvolutionBackwardData` and `miopenFindConvolutionBackwardWeightsAlgorithm` / `miopenConvolutionBackwardWeights` respectively.

**Internal call chain:**
```
miopenFindConvolutionForwardAlgorithm()          [convolution_api.cpp:555]
  -> ConvolutionDescriptor::FindConvFwdAlgorithm()  [convolutionocl.cpp:571]
    -> FindConvolution()                             [convolutionocl.cpp:427]
       (mode-dependent — see Section 4)
```

**Key characteristic**: The `exhaustiveSearch` parameter on the Find call maps to `ctx.do_search`. When true, it forces the solver framework to benchmark all tunable configs rather than using cached perf-db entries. The Find call both searches AND registers the winning invoker with the handle, so the subsequent Execute call is a simple dispatch by algorithm name.

**Transpose convolution note:** When the convolution descriptor's `mode == miopenTranspose`, the API transparently swaps the direction — a Forward Find calls `FindConvBwdDataAlgorithm` internally, and vice versa.

### 2.2 Immediate Mode API (Generation 2)

Adds explicit solution IDs so users can enumerate, pre-compile, and execute specific solutions without re-running the full find.

**Forward Convolution:**

| Step | API Function | Purpose |
|------|-------------|---------|
| 0 | `miopenConvolutionForwardGetSolutionCount(handle, wDesc, xDesc, yDesc, &count)` | How many solutions are available |
| 1 | `miopenConvolutionForwardGetSolution(handle, wDesc, xDesc, yDesc, maxCount, &count, solutions)` | Get solution list (`miopenConvSolution_t[]` with solution_id, algo, time, workspace_size) |
| 2 | `miopenConvolutionForwardGetSolutionWorkspaceSize(handle, wDesc, xDesc, yDesc, solution_id, &size)` | Workspace for a specific solution |
| 3 | `miopenConvolutionForwardCompileSolution(handle, wDesc, xDesc, yDesc, solution_id)` | Pre-compile kernels (optional, avoids first-run latency) |
| 4 | `miopenConvolutionForwardImmediate(handle, wDesc, w, xDesc, x, yDesc, y, ws, wsSize, solution_id)` | Execute by solution ID |

**Internal call chain (GetSolution):**
```
miopenConvolutionForwardGetSolution()               [convolution_api.cpp:766]
  -> ConvolutionDescriptor::GetSolutions()             [convolutionocl.cpp:1042]
    -> GetSolutions<FindDb>(ctx, problem, maxCount)    [convolutionocl.cpp:87]
       reads from SystemFindDb + UserFindDb (combined)
    -> GetSolutionsFallback() if DB empty              [convolutionocl.cpp:907]
       uses TunaNet AI or WTI heuristic
```

**Internal call chain (Immediate Execute):**
```
miopenConvolutionForwardImmediate()                  [convolution_api.cpp:839]
  -> ConvolutionDescriptor::ConvolutionForwardImmediate() [convolutionocl.cpp:1097]
    -> LoadOrPrepareInvoker()                          [convolutionocl.cpp:215]
       checks handle cache first, then compiles
    -> invoker(handle, invoke_ctx)                     dispatch
```

**Key characteristic**: The GetSolution step reads from the find-db (both system and user) to get pre-computed results. If no DB entries exist, it falls back to heuristic estimation (TunaNet or WTI). No benchmarking occurs — the times returned are either from prior Find operations or heuristic estimates. This makes it much faster than Classic Find at the cost of potentially selecting a non-optimal solver.

### 2.3 V2 / Find2.0 API (Generation 3)

A generalized problem/solution framework that abstracts away the operation type. The same API works for convolution, softmax, MHA, and (in the future) other operations.

**Workflow:**

| Step | API Function | Purpose |
|------|-------------|---------|
| 0 | `miopenCreateConvProblem(&problem, convDesc, direction)` | Create a typed problem |
| 0 | `miopenSetProblemTensorDescriptor(problem, id, desc)` | Register input/weight/output tensors |
| 1 | `miopenCreateFindOptions(&options)` | Create search configuration |
| 1 | `miopenSetFindOptionTuning(options, value)` | Enable exhaustive tuning |
| 1 | `miopenSetFindOptionResultsOrder(options, order)` | Sort by time or workspace |
| 1 | `miopenSetFindOptionWorkspaceLimit(options, limit)` | Cap workspace usage |
| 1 | `miopenSetFindOptionPreallocatedWorkspace(options, ws, size)` | Provide workspace buffer |
| 1 | `miopenSetFindOptionAttachBinaries(options, attach)` | Include kernel binaries in solution for serialization |
| 2 | `miopenFindSolutions(handle, problem, options, maxSolutions, &solutions, &numFound)` | Find applicable solutions |
| 3 | `miopenRunSolution(handle, solution, nInputs, tensors, ws, wsSize)` | Execute a solution |
| - | `miopenSaveSolution(solution, data, &size)` / `miopenLoadSolution(solution, data, size)` | Serialize/deserialize solutions |

**Internal call chain:**
```
miopenFindSolutions()                               [api/find2_0_commons.cpp:251]
  -> Problem::FindSolutions()                        [problem.cpp:157]
    -> std::visit on operator_descriptor:
       ConvolutionDescriptor -> FindSolutionsImpl()   [problem.cpp:459]
         -> FindConvolution()                        [convolutionocl.cpp:427]
            (same pipeline as Classic Find)
       SoftmaxDescriptor -> FindSolutionsImpl()      [problem.cpp:548]
         -> manual solver loop (no find-db)
       MhaDescriptor -> FindSolutionsImpl()          [problem.cpp:600]
         -> manual solver loop (no find-db)
```

**Key characteristic**: For convolution, the V2 API calls into the exact same `FindConvolution()` function as Classic Find. The main benefits are: (1) a unified API across operation types, (2) explicit solution objects that can be serialized/deserialized, (3) fine-grained control via FindOptions, and (4) no implicit algorithm enum — solutions are identified by solver ID.

---

## 3. The Find Pipeline

All three API generations converge on `FindConvolution()` (`src/ocl/convolutionocl.cpp:427`) for convolution. This function implements mode-dependent behavior:

```
FindConvolution(ctx, problem, invoke_ctx, requestAlgoCount, force_attach_binary)
│
├─ if FindMode is Fast or Hybrid:
│   ├─ if TrustVerify:
│   │   ├─ Try UserFindDb first (1 solution)
│   │   └─ If miss: GetSolutions from SystemFindDb (2 solutions)
│   └─ else:
│       └─ GetSolutions from combined FindDb (1 solution)
│
│   └─ If solutions found AND (not Hybrid+fallback OR TrustVerify+AI OR force):
│       ├─ sol = best solution
│       │
│       ├─ if TrustVerify AND from system DB:
│       │   └─ VerifiedFDBSolution(): benchmark, compare, write to user DB
│       │
│       └─ CompileSolution(sol) → return [fast path]
│
├─ [Fallthrough to Normal/exhaustive path]
│   └─ UserFindDbRecord::TryLoad():
│       ├─ If user find-db has valid cached record: return cached solutions
│       └─ If miss:
│           ├─ if DynamicHybrid: set use_dynamic_solutions_only
│           ├─ if TrustVerify fallthrough: set do_search + db_update
│           └─ FindCore():
│               ├─ For each SolverFinder (ImplicitGEMM, GEMM, Winograd, FFT, Direct):
│               │   └─ finder.Find() → SearchForAllSolutions per algorithm family
│               ├─ PrecompileSolutions() — compile all kernels
│               └─ EvaluateInvokers() — benchmark each, remove outliers, pick best
│           └─ Store results in user find-db
│
├─ ShrinkToFind10Results() — keep only best per algorithm
└─ Return sorted results
```

---

## 4. Find Mode System

The FindMode controls which search strategy is used. It is the primary "knob" that controls the tradeoff between first-call latency and solution quality.

**Source:** `src/include/miopen/find_controls.hpp:110-188`, `src/find_controls.cpp:260-268`

### 4.1 Find Mode Values

| Value | Enum | Behavior |
|-------|------|----------|
| 1 | `Normal` | Full exhaustive search. Benchmarks all applicable solvers. Slowest first-call, best solution quality. Results cached in user find-db. |
| 2 | `Fast` | DB-lookup only. Reads from system+user find-db. If no DB entry, falls back to heuristic (TunaNet/WTI). Never benchmarks. Fastest first-call. |
| 3 | `Hybrid` | Tries DB-lookup first. If DB hit, uses it. If DB miss but heuristic fallback succeeds, uses heuristic. If heuristic produces a "fallback" (non-DB) solution, falls through to Normal exhaustive search. |
| 5 | `DynamicHybrid` | **DEFAULT**. Like Hybrid but restricts exhaustive search to dynamic solvers only (those that don't require recompilation for different tensor sizes). |
| 6 | `TrustVerify` | Reads user DB first. If miss, reads system DB. Verifies the system DB entry by benchmarking it on the user's hardware. If the measured time exceeds a tolerance vs. recorded time (controlled by `MIOPEN_VERIFY_TOLERANCE_PCT`), triggers a full exhaustive search. Results stored in user DB so verification only happens once per problem config. |
| 7 | `TrustVerifyFull` | Like TrustVerify but the triggered search is exhaustive (all solvers, not just dynamic). |

### 4.2 Selection Priority

FindMode is determined by the following priority (highest first):

1. **FindEnforce override**: If `MIOPEN_FIND_ENFORCE` is set to anything other than `None`, FindMode is forced to behave as `Normal` regardless of its setting. (`find_controls.hpp:131-138`)
2. **API setter**: `miopenSetConvolutionFindMode(convDesc, mode)` on the convolution descriptor.
3. **Environment variable**: `MIOPEN_FIND_MODE` for convolution, `MIOPEN_FIND_MODE_FUSION` for fusion (defaults to Fast).
4. **Compile-time default**: `DynamicHybrid` (value 5).

### 4.3 Mode Categorization in Code

The FindMode class exposes predicates used by FindConvolution:

- `IsFast(ctx)` — true only for `Fast`
- `IsHybrid(ctx)` — true for `Hybrid`, `DynamicHybrid`, `TrustVerify`, `TrustVerifyFull`
- `IsDynamicHybrid(ctx)` — true for `DynamicHybrid`, `TrustVerify`, `TrustVerifyFull`
- `IsTrustVerify(ctx)` — true for `TrustVerify`, `TrustVerifyFull` (disabled if `MIOPEN_DISABLE_USERDB`)
- `IsExhaustive(ctx)` — true only for `TrustVerifyFull`

All predicates return false if FindEnforce overrides the mode.

---

## 5. Find Enforce / Tuning Policy

FindEnforce provides a mechanism to force specific database and search behaviors, overriding FindMode.

**Source:** `src/include/miopen/find_controls.hpp:49-106`, `src/find_controls.cpp:81-116`

| Value | Name | Behavior |
|-------|------|----------|
| 1 | `None` | No enforcement (default). FindMode controls behavior normally. |
| 2 | `DbUpdate` | Forces results to be written to the perf-db (user database for per-solver tuning parameters). |
| 3 | `Search` | Forces exhaustive search, ignoring any cached DB entries. Does NOT write results to DB. |
| 4 | `SearchDbUpdate` | Forces exhaustive search AND writes results to perf-db. The "full retune" option. |
| 5 | `DbClean` | Removes DB entries for the problem config. |

**Set via:**
- Environment variable `MIOPEN_FIND_ENFORCE` (string: `NONE`, `DB_UPDATE`, `SEARCH`, `SEARCH_DB_UPDATE`, `DB_CLEAN`; or numeric: 1-5)
- Programmatic API: Handle's tuning policy (`miopenSetTuningPolicy`), which takes higher priority than env var

**Important interaction:** When FindEnforce is set to any value 2-5 (i.e., `IsSomethingEnforced()` returns true), the FindMode predicates all return false, effectively forcing Normal find behavior. This is how enforcement overrides mode.

**Disable switch:** `miopen::debug::FindEnforceDisable` can disable enforcement programmatically. Used during MIOpenDriver warm-up phase.

---

## 6. Database Layer

MIOpen uses two distinct database systems that serve different purposes in the solver selection pipeline.

### 6.1 Find Database (Find-DB)

**Purpose:** Caches the final selected solution(s) for a problem configuration. Stores: solver ID, execution time, workspace size, algorithm name.

**Source:** `src/include/miopen/find_db.hpp`, `src/find_db.cpp`

**Two tiers:**
- **System Find-DB** (`SystemFindDb`): Read-only. Shipped with MIOpen, contains pre-benchmarked results for common configurations on each GPU. File format: `<gpu_name>.<suffix>.fdb.txt`. Can be embedded at compile time (`MIOPEN_EMBED_DB`) or loaded from disk.
- **User Find-DB** (`UserFindDb`): Read-write. Built locally by the user's find operations. File format: `<gpu_name>.<suffix>.ufdb.txt`. Located at `GetUserDbPath()` (typically `~/.config/miopen/`).

**Combined access:** `FindDb = MultiFileDb<SystemFindDb, UserFindDb, false>` — reads from both, writes go to user DB.

**Lookup key:** `NetworkConfig` — a string that encodes the full problem description (tensor shapes, strides, data types, convolution parameters, etc.).

**Disable switches:**
- `MIOPEN_DEBUG_DISABLE_FIND_DB` — disables find-db entirely
- `MIOPEN_DISABLE_USERDB` — compile-time flag to disable user DB
- `MIOPEN_DISABLE_SYSDB` — compile-time flag to disable system DB

**Record lifecycle (FindDbRecord_t):**
1. Constructor opens DB and looks up record by problem's NetworkConfig
2. `TryLoad()` — static helper that checks if a cached record exists and is valid (all invokers still compiled). If valid, returns cached solutions. If invalid, calls a regenerator lambda and caches the new results.
3. Destructor writes modified records to disk (unless `dont_store` flag is set due to suboptimal workspace)

**Validation:** `Validate()` checks that all solver invokers in a cached record are still compiled in the handle's invoker cache. If any are missing, the record is considered stale and needs regeneration.

### 6.2 Performance Database (Perf-DB)

**Purpose:** Caches per-solver tuning parameters (performance configs). Used during exhaustive search to avoid re-tuning solvers that have already been tuned for a given problem.

**Separate from Find-DB:** Perf-DB stores one record per (problem, solver) pair containing the solver's `PerformanceConfig`. Find-DB stores the final ranked result across all solvers.

**Flow interaction:** During `FindSolution()` for a tunable solver:
1. Check perf-db for an existing config
2. If found and valid, use it
3. If not found (or forced search), run `solver.Search()` which benchmarks all parameter combinations
4. Store the best config in perf-db

**Access pattern:** `GetDb()` / `MakeConvDbGetter()` returns a database accessor scoped to the primitive type.

### 6.3 DB Path Resolution

The system find-db path is resolved with a GPU-matching algorithm:
1. Try exact match: `<basename>.<suffix>.fdb.txt`
2. If no exact match, iterate all `.fdb.txt` files in the DB directory
3. Match by GPU DB-ID prefix (e.g., `gfx90a`)
4. Among matches, pick the one with the closest compute unit count

---

## 7. Solver Framework

### 7.1 Solver Base Classes

**Source:** `src/include/miopen/solver.hpp`

All solvers inherit from a base class hierarchy:

```
SolverBase
├── SolverInterface<Context, Problem>
│   ├── SolverBaseNonTunable<Context, Problem>   — has GetSolution(ctx, problem)
│   └── SolverBaseTunable<Context, Problem, Cfg> — has Search(), GetDefaultPerformanceConfig()
```

**Key interface methods:**
- `IsApplicable(ctx, problem)` — can this solver handle this problem?
- `IsDynamic()` — does this solver work without recompilation for different sizes?
- `GetWti(ctx, problem)` — Work Thread Intensity estimate (0.0-1.0, used for heuristic ranking)
- `GetWorkspaceSize(ctx, problem)` — how much GPU workspace memory needed
- `GetSolution(ctx, problem[, config])` — produce a `ConvSolution` with kernel info and invoker factory
- `Search(ctx, problem, invoke_ctx)` — (tunable only) exhaustively benchmark all parameter combinations

### 7.2 Solver Registration

Solvers are registered in a global registry indexed by numeric ID and string name. `solver::GetSolversByPrimitive(Primitive::Convolution)` returns all registered convolution solvers.

### 7.3 FindOnlySolver Filter

`MIOPEN_DEBUG_FIND_ONLY_SOLVER` accepts a semicolon-delimited list of solver IDs (numeric or string). When set, only the specified solvers are considered during search. All others are skipped.

---

## 8. Algorithm-Family Finders

During the exhaustive search path (`FindCore()`), solutions are discovered by five algorithm-family finders. Each finder is a subclass of `SolversFinderMixin` and wraps a `SolverContainer` for its algorithm type.

**Source:** `src/conv/solver_finders.cpp`

| Finder | Algorithm | Enable/Disable Env Var | Notes |
|--------|-----------|----------------------|-------|
| `ImplicitGemmSolverFinder` | `miopenConvolutionAlgoImplicitGEMM` | `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM` | Default first in order |
| `GemmSolverFinder` | `miopenConvolutionAlgoGEMM` | `MIOPEN_DEBUG_CONV_GEMM` | Uses rocBLAS |
| `WinogradSolverFinder` | `miopenConvolutionAlgoWinograd` | `MIOPEN_DEBUG_CONV_WINOGRAD` | Forces dynamic-only when `use_winograd_only` |
| `FftSolverFinder` | `miopenConvolutionAlgoFFT` | `MIOPEN_DEBUG_CONV_FFT` | Not available for BackwardWeights |
| `DirectSolverFinder` | `miopenConvolutionAlgoDirect` | `MIOPEN_DEBUG_CONV_DIRECT` | Also gated by `MIOPEN_CONV_DIRECT_MAX_SIZE` |

**Order matters:** The finders are invoked in the order listed above. Within each finder, all applicable solvers in its `SolverContainer` are tested. Each finder returns a list of `ConvSolution` objects. After all finders run, solutions are precompiled together, then benchmarked.

**Disable mechanism:** Each env var (`MIOPEN_DEBUG_CONV_*`) defaults to enabled. Setting it to `0` disables that entire algorithm family. This is an all-or-nothing toggle per family, distinct from `MIOPEN_DEBUG_FIND_ONLY_SOLVER` which filters individual solvers.

---

## 9. Benchmarking and Solution Ranking

### 9.1 EvaluateInvokers

**Source:** `src/conv/solver_finders.cpp:224-371`

After all solutions are compiled, `EvaluateInvokers()` benchmarks each:

1. **Workspace check**: Skip solutions requiring more workspace than provided. If skipped and `MIOPEN_FIND_CONV_INSUFFICIENT_WORKSPACE_ALLOW_FINDDB_UPDATE` is not set, marks result as non-optimal (prevents writing to user find-db).

2. **Search cutoff**: If a previous solver already timed well:
   - Skip any solver with "Naive" in its name if the best so far is < 5ms (`MIOPEN_SEARCH_CUTOFF`)
   - Skip any solver if its first timed run exceeds `best * MIOPEN_FIND_SKIP_PCT / 100` (default 130%)

3. **Benchmarking**: Run up to 8 iterations with a 5-second time limit. First run is warmup (excluded from samples).

4. **Outlier removal**: Use modified Z-score (threshold 2.0) to remove high outliers, then take the mean of remaining samples.

5. **Selection**: Track the best (lowest mean time) across all solutions. Register the winner's invoker with the handle.

### 9.2 Solution Time Comparator

**Source:** `src/ocl/convolutionocl.cpp:66-82`

Solutions are sorted using `SolutionTimeComparator`:
- Negative times = coarse heuristic estimates. More negative = worse.
- Positive times are always preferred over negative.
- Among positive times, lower is better.

### 9.3 ShrinkToFind10Results

After benchmarking, `ShrinkToFind10Results()` keeps only the best solution per algorithm family, matching the cuDNN-like Find API contract that returns one result per algorithm type.

---

## 10. Fallback System

When the find-db has no entry for a problem configuration and Immediate Mode needs to return solutions, the fallback system provides heuristic estimates.

**Source:** `src/ocl/convolutionocl.cpp:907-1035`

### 10.1 TunaNet AI Fallback

If `MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK` is compiled in (and not disabled via `MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK=0`):

1. Calls `ai::immed_mode::PredictSolver(problem, ctx, arch)` — an ML model that predicts the best solver(s) for a given problem.
2. Returns predicted solvers with synthetic times (10ms * rank_index).
3. Only includes dynamic solvers that pass `IsApplicable()`.

### 10.2 WTI Fallback

If TunaNet produces no results:

1. Iterates all registered convolution solvers.
2. Filters to dynamic, applicable solvers.
3. Estimates performance using `GetWti()` (Work Thread Intensity — a static estimate of how well the solver utilizes the GPU).
4. Converts WTI to synthetic time: `time = 10.0 / wti`.

### 10.3 Fallback Path Tracking

An enum `FallbackPath` tracks which fallback was used: `None` (DB hit), `AI` (TunaNet), `WTI`. This is used by TrustVerify mode to decide whether to verify a system-DB entry or an AI prediction.

---

## 11. Non-Convolution Operations

MIOpen's solver/find infrastructure was originally built for convolution. Other operations have adopted varying levels of this infrastructure:

### 11.1 Adoption Summary

| Operation | Solver Framework (SolverContainer) | Find-DB / Perf-DB | V2 Problem API | Fallback System | Notes |
|-----------|-----------------------------------|-------------------|----------------|-----------------|-------|
| **Convolution** | Full (5 finder families) | Both | Yes | TunaNet + WTI | Complete pipeline |
| **Fusion** | Yes (custom FusionSolverFinder) | Yes (uses ConvDbGetter) | No | No | Semi-conv-specific |
| **Softmax** | Yes (SolverContainer + ExecutePrimitive) | Yes (own GetDb()) | Yes | No | Fully modernized |
| **Pooling** | Yes (SolverContainer + ExecutePrimitive) | Yes (own GetDb()) | No | No | Fully modernized |
| **Batch Norm** | No | No | No | No | Legacy Handle.AddKernel() |
| **MHA (Attention)** | Partial (manual solver loop) | No | Yes | No | V2 API but no DB |
| **RNN** | No | No | No | No | Legacy kernel management |

### 11.2 Softmax

**Source:** `src/softmax.cpp`

Softmax uses the modern solver framework with `SolverContainer<solver::softmax::AttnSoftmax, solver::softmax::Softmax>` and calls `ExecutePrimitive()`. It has its own `GetDb()` for perf-db access. Through the V2 API (`miopenFindSolutions` with a softmax problem), it goes through `Problem::FindSolutionsImpl()` which does a manual solver loop with hardcoded time estimates (1.0 for AttnSoftmax, 2.0 for regular — no actual benchmarking in V2 path).

### 11.3 Pooling

**Source:** `src/pooling.cpp`

Similar to softmax — uses `SolverContainer` with multiple pooling solver variants (2D, ND, Naive, Transposed) and calls `ExecutePrimitive()`. Has its own perf-db. Does not currently participate in the V2 Problem API.

### 11.4 MHA (Multi-Head Attention)

**Source:** `src/problem.cpp:600-651`

MHA participates in the V2 Problem/Solution API (`miopenCreateMhaProblem`). Its `FindSolutionsImpl` does a manual loop over `MhaForward` and `MhaBackward` solvers with hardcoded time = 1.0. No perf-db, no find-db, no benchmarking.

### 11.5 Batch Normalization

**Source:** `src/batch_norm.cpp`

Uses the legacy `Handle.AddKernel()` mechanism. Kernel selection is done at the descriptor level with manual parameter computation. Batchnorm has solver classes in `src/solver/batchnorm/` but these are invoked from fusion contexts, not from the main batchnorm API path.

### 11.6 Fusion

**Source:** `src/fusion.cpp`

Fusion uses the solver framework with custom `FusionSolverFinder` instances (one per fusion type, e.g., ConvBiasActivAsm1x1U, ConvBinWinogradRxSFused, BnFwdInferActivationFused). It uses `SearchForAllSolutions()` with `MakeConvDbGetter()` for perf-db access, tying it to the convolution database system.

---

## 12. Environment Variable Reference

### Find Mode & Enforcement

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `MIOPEN_FIND_MODE` | String/Int | `DYNAMIC_HYBRID` (5) | Controls search strategy for convolution |
| `MIOPEN_FIND_MODE_FUSION` | String/Int | `FAST` (2) | Controls search strategy for fusion operations |
| `MIOPEN_FIND_ENFORCE` | String/Int | `NONE` (1) | Forces tuning behavior, overrides FindMode |

### Database Control

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `MIOPEN_DEBUG_DISABLE_FIND_DB` | Bool | 0 | Disable find-db entirely |
| `MIOPEN_DEBUG_FIND_DB_CACHING` | Bool | 0 | Use in-memory caching for find-db |
| `MIOPEN_FIND_CONV_INSUFFICIENT_WORKSPACE_ALLOW_FINDDB_UPDATE` | Bool | 0 | Allow writing suboptimal results to find-db when workspace is insufficient |
| `MIOPEN_CUSTOM_CACHE_DIR` | String | - | Override cache directory location |

### Algorithm Family Toggles

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `MIOPEN_DEBUG_CONV_DIRECT` | Bool | 1 | Enable/disable Direct convolution solvers |
| `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM` | Bool | 1 | Enable/disable Implicit GEMM solvers |
| `MIOPEN_DEBUG_CONV_WINOGRAD` | Bool | 1 | Enable/disable Winograd solvers |
| `MIOPEN_DEBUG_CONV_GEMM` | Bool | 1 | Enable/disable GEMM solvers |
| `MIOPEN_DEBUG_CONV_FFT` | Bool | 1 | Enable/disable FFT solvers |
| `MIOPEN_CONV_DIRECT_MAX_SIZE` | UInt64 | 0 (no limit) | Max problem size for Direct solver |

### Solver Selection

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `MIOPEN_DEBUG_FIND_ONLY_SOLVER` | String | - | Restrict search to specified solver IDs (semicolon-delimited) |

### Search Optimization

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `MIOPEN_SEARCH_CUTOFF` | Bool | 0 | Enable performance-based search cutoff (skip Naive solvers if fast solution found) |
| `MIOPEN_FIND_SKIP_PCT` | UInt64 | 130 | Skip solver if first run exceeds best * this/100 |

### Fallback Control

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `MIOPEN_DEBUG_CONV_IMMED_FALLBACK` | Bool | 1 | Enable/disable immediate mode fallback entirely |
| `MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK` | Bool | 1 | Enable/disable TunaNet AI fallback |
| `MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK` | Bool | 0 | Force immediate mode even in Hybrid when fallback is used |

### TrustVerify

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `MIOPEN_VERIFY_TOLERANCE_PCT` | Float | - | Tolerance for TrustVerify time comparison (as percentage above recorded time) |
| `MIOPEN_WARN_SEARCH` | Bool | 0 | Log search start/end as warnings instead of info |

### Compilation

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `MIOPEN_DEBUG_COMPILE_ONLY` | Bool | 0 | Compile kernels but skip execution (testing/CI use) |

---

## 13. Key Source File Index

| File | Role |
|------|------|
| `include/miopen/miopen.h` | Public C API — all user-facing function declarations and enums |
| `src/convolution_api.cpp` | API function implementations → internal method dispatch |
| `src/api/find2_0_commons.cpp` | V2 Find API (miopenFindSolutions, miopenRunSolution, etc.) |
| `src/ocl/convolutionocl.cpp` | Core convolution logic: FindConvolution(), GetSolutions(), immediate mode execute, Classic Find execute |
| `src/conv/solver_finders.cpp` | Algorithm-family finders, EvaluateInvokers(), FindCore() |
| `src/include/miopen/conv/solver_finders.hpp` | ISolversFinder interface, FindCoreResult, SolversFinderMixin |
| `src/find_controls.cpp` | FindMode and FindEnforce implementation, env var parsing |
| `src/include/miopen/find_controls.hpp` | FindMode and FindEnforce class definitions |
| `src/find_db.cpp` | Find database — path resolution, record validation, storage |
| `src/include/miopen/find_db.hpp` | FindDbRecord_t template, TryLoad() logic |
| `src/problem.cpp` | V2 Problem::FindSolutions() dispatch to operation-specific implementations |
| `src/include/miopen/problem.hpp` | Problem class — variant-based operation descriptor container |
| `src/solution.cpp` | Solution class — solver ID, time, workspace, invoker |
| `src/include/miopen/solver.hpp` | Solver base classes (SolverBase, SolverBaseTunable, SolverBaseNonTunable) |
| `src/solver/` | Individual solver implementations per algorithm family |
| `src/softmax.cpp` | Softmax operation — modern solver framework example |
| `src/pooling.cpp` | Pooling operation — modern solver framework example |
| `src/batch_norm.cpp` | Batch norm — legacy path example |
| `src/fusion.cpp` | Fusion — custom finder integration with conv DB |
| `src/generic_search.cpp` | Shared tuning controls (iteration limits, time limits, thread counts) |
