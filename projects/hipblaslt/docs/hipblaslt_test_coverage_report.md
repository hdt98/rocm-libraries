# hipBLASLt test coverage

This page is the main reference for how **hipBLASLt** is tested: what runs in CI, what the suite covers (and where it doesn’t), and how we’re improving it. Whether you’re new to the library, triaging a failure, or working on quality and testing (including Epic **AIHPBLAS-1003**), the sections below walk through test design, the two CIs, labels, coverage tables, and known gaps—with concrete next steps.

## Coverage at a glance (for stakeholders)

**What is covered:** The test suite exercises the full **matmul (GEMM) API** across all supported precisions (including half, single, double, 8‑bit float, mixed, and **complex** f32_c/f64_c on gfx942 and gfx950) and a wide range of matrix sizes—from 1×1×1 up to large and workload-specific shapes. **Descriptor and helper APIs** (handles, preferences, algorithms, plans) are covered by dedicated auxiliary tests. **Rocroller** integration is covered by one configuration. When a PR changes hipblaslt code, TheRock CI runs the **full** suite (no filter); Math CI runs the full suite on precheckin.

**Scale:** The full suite is on the order of **tens of thousands** of test instances (see §4 for counts by operation group). A fast **smoke** subset (~1.5k tests) runs when only CI/workflow files change, so presubmit stays quick when hipblaslt itself is not modified.

**Limits:** This document describes **test scenarios and CI behavior**—which operations, precisions, and sizes we run—not line- or branch-level code coverage. Known gaps and improvement opportunities are summarized in **§5 Coverage gaps and improvements**. That work is tracked under Epic **AIHPBLAS-1003**; related production issues (e.g. SWDEV-576540, SWDEV-484012, SWDEV-568258) are linked in the Epic.

---

## 1. What the tests are and why they exist

### 1.1 How the suite is built

Tests are **generated** from YAML, not hand-written per case. The flow:

1. **YAML** (`clients/tests/data/`) — Template "tests" with names, categories, and parameter ranges (precision, matrix_size, transA_transB, alpha_beta, etc.).
2. **Generator** (`hipblaslt_gentest.py`) — Expands each template over its ranges (Cartesian product), deduplicates, and writes a binary data file read by the test binary.
3. **Test binary** (`hipblaslt-test`) — One GTest parameterized test per record; selection at run time is by **filter** (e.g. `*smoke*`, `*quick*`) or no filter (run all).

So a single YAML template (e.g. `matmul_small`) becomes many test instances (one per precision × size × trans × alpha_beta × …). Categories (smoke, quick, pre_checkin, nightly) are tags in the YAML; the filter selects which subset runs.

### 1.2 Test flavors (categories and sources)

- **smoke** — Fast sanity. Small sizes (E), one test file (`smoke_gtest.yaml`). Exists so presubmit can run something useful in a few minutes without running the full suite. Covers core matmul, bias, and a spread of precisions (real, f64, 1b, fnuz, int i8, xf32) at small sizes.
- **quick** — Short regression. Only tests tagged `category: quick` in `matmul_gtest.yaml` (e.g. matmul_one, matmul_small, matmul_8, matmul_bf16). Used where the full suite is too heavy (e.g. Windows in TheRock). No auxiliary or rocroller.
- **pre_checkin** / **nightly** — Heavier tests (algo, heuristic, stress, workload shapes, auxiliary, rocroller). Not selected by a separate filter in TheRock or Math CI; they are included when you run **full** (no filter). So "full" = smoke + quick + pre_checkin + nightly (and HMM, etc.) from all four YAML inputs.

The **sources** are four YAML files included by `hipblaslt_gtest.yaml`: **matmul_gtest** (main GEMM/matmul matrix), **auxiliary_gtest** (handle, descriptors, preferences, algorithms, helpers), **smoke_gtest** (smoke variants), **rocroller_gtest** (rocroller predicate test). Full run executes all of them; smoke and quick only run subsets of the matmul + smoke definitions.

### 1.3 Why matmul, auxiliary, and rocroller are separate

- **Matmul** — The main API: GEMM over sizes, precisions, transpositions, bias, activation, grouped gemm, extended API. The tables in §4 are about this surface (operation × precision × size). Most tests and most CI time are here.
- **Auxiliary** — Descriptor and helper API (handle, mat, matmul pref/alg/plan, edge cases). Mostly "does the API accept and return the right things"; only one test runs a real GEMM. So they don't fit the same op × precision × size grid and are summarized in their own small table per config.
- **Rocroller** — Integration with the rocroller path; a single predicate configuration. Again a different kind of coverage, so it's called out separately.

### 1.4 Why size and precision matter

The **size letters** (O, S, M, E, C, G, D, W, F, R) summarize *which* matrix shapes are exercised in each cell. Different kernels and code paths are used for tiny vs large vs skinny matrices; the suite deliberately spans from 1×1×1 to grid_limit and workload-specific shapes so that core, stress, and real-world shapes are all hit.

**Precision** (real, f64, 1b, 1b fnuz, int i8, xf32, mixed, **complex** f32_c/f64_c) matters because the library has different code paths and numerical behavior per type; the tables show which operation groups are tested at which precisions so gaps (e.g. no f64 in bias) are visible. Complex GEMM is supported and tested on **gfx942 and gfx950 only** (see CHANGELOG 1.2.1); complex tests run in the **full** suite (pre_checkin) but are not in smoke or quick.

### 1.5 Batched GEMM coverage

The suite **does** include batched GEMM tests (`batch_count` > 1). When `batch_count` > 0, the generator (`clients/tests/hipblaslt_gentest.py`) sets strides for A, B, C, D, and E from the matrix dimensions and leading dimensions so that batched runs exercise the strided-batch path correctly.

**Default:** In `hipblaslt_common.yaml`, the default is **`batch_count: 1`**; most tests are single-batch.

Tests that use `batch_count` > 1 in `matmul_gtest.yaml`:

| Batch size(s) | Tests |
|---------------|--------|
| **1 and 5**   | `matmul_swizzleA`, `matmul_extapi_swizzleA`, `matmul_swizzleB`, `matmul_extapi_swizzleB` (each has `batch_count: [1, 5]`). |
| **10**        | `matmul_batch_medium`, `matmul_batch_medium_complex`. |
| **16**        | `matmul_fallback_equality_NN_batch16`, `matmul_fallback_equality_TN_batch16`. |
| **17711**     | `matmul_stride_lt_ld` (single test: specific M/N/K, strides, bias; gfx950). |

**Batch-size band coverage:** **Small batch (e.g. 2–128)** is only partially covered — we have 5, 10, and 16; there is no systematic spread (e.g. 2, 32, 64, 128). **Medium batch (129–1024)** and **large batch (1025–8192)** are not covered. **Very large batch (>8192)** has one test (batch_count 17711).

---

## 2. The two CIs and what they run

hipBLASLt is exercised by **two CI systems**. Both drive the same GTest binary (`hipblaslt-test`); they differ in when they run, which filter (if any) they use, and how they shard.

### 2.1 TheRock CI (GitHub Actions, rocm-libraries/.github)

- **Where:** Workflows under `rocm-libraries/.github/workflows/` (therock-ci.yml, therock-ci-linux.yml, therock-test-packages.yml, etc.). Runs on Azure (Linux and Windows).
- **Test levels and when they run:**

| Situation | test_type | hipblaslt-test filter | Shards (Linux) |
|-----------|-----------|------------------------|----------------|
| **PR/push** that changes **project code** (e.g. `projects/hipblaslt/**`) | full | (none — entire suite) | 6 |
| **PR/push** that changes only **.github** (therock workflow/scripts) | smoke | `*smoke*` | 1 |
| **Scheduled (nightly)** | full | (none) | 6 |
| **Windows** (gfx1151) | quick | `*quick*` | 1 |

- **How test_type is chosen:** The setup script (`therock_configure_ci.py`) uses **git diff** to get modified paths and maps them to projects (e.g. `projects/hipblaslt` → hipblaslt). It **defaults to full**; it only sets **smoke** when the changed files are under `.github/workflows/therock*` or `.github/scripts/therock*` (so workflow-only changes do not trigger the full suite). So **if a PR changes files in `projects/hipblaslt`, TheRock CI runs full tests** for the blas group (including hipblaslt). The label **project: hipblaslt** is auto-applied by the labeler when `projects/hipblaslt/**`, `shared/origami/**`, or `shared/rocroller/**` change; that label is for visibility; the CI decides test_type from the actual changed paths, not from the label.
- **Architectures:** Typically gfx94X and gfx950 on Linux, gfx1151 on Windows (from TheRock’s amdgpu_family_matrix). Tests run on a subset of built targets (e.g. gfx94X and gfx1151 as of this writing).
- **Summary:** PRs that touch hipblaslt (or other project code) get **full** tests. Smoke is only used when the only changes are under .github (therock workflow/scripts). On Windows you get **quick**.

### 2.2 Math CI (Jenkins, rocjenkins)

- **Where:** Jenkins at math-ci.amd.com; repo **rocjenkins**. Status appears as “Math CI Summary” on GitHub. Being superseded by TheRock CI.
- **Test levels and when they run:**

| Job | What runs |
|-----|-----------|
| **precheckin** (gating) | **Full** `hipblaslt-test` — **no gtest filter**. Plus one groupedgemm bench and (on non–gfx950 nodes) a set of samples. Timeout 780 min. |
| **codecov** | Subset via filter: `*pre_checkin*` minus two aux tests; for coverage upload. |
| **preliminary** (gating if no rocroller changes) | TensileLite Python only (`tox` in tensilelite), not hipblaslt-test. |
| **performance / perfci** | hipblaslt-perf (PTS), not the GTest suite. |

- **Architectures:** On PRs, 3 node types (rhel9, sles15, ubuntu22) with gfx950, gfx1200;gfx1201, gfx942;gfx90a. **No GTest sharding** — each node runs the full suite once; parallelism is multiple nodes.
- **Summary:** The main regression is **precheckin**, which always runs the **full** suite (no filter). So in practice both TheRock “full” and Math CI precheckin run the same **full** suite; TheRock additionally has **smoke** (presubmit) and **quick** (Windows).

---

## 3. Labels (auto-applied and CI-related)

PRs in rocm-libraries get **auto-applied labels** based on which files changed. These are for **visibility and filtering** (e.g. in the PR list or for triage); they do **not** control which tests run or whether TheRock CI runs full vs smoke — that is decided by `therock_configure_ci.py` from the **git diff**, not from labels.

### 3.1 How labels are applied

The **labeler** workflow (`.github/workflows/labeler.yml`) runs on `pull_request_target` (opened, synchronize, reopened). It uses the config in `.github/labeler.yml`: each label has a set of path globs; if any changed file in the PR matches, that label is added (and labels whose paths no longer match are removed).

### 3.2 Labels relevant to hipBLASLt

| Label | When it is applied |
|-------|--------------------|
| **project: hipblaslt** | Any change under `projects/hipblaslt/**`, `shared/origami/**`, or `shared/rocroller/**`. |
| **project: hipblaslt-provider** | Any change under `dnn-providers/hipblaslt-provider/**`. |
| **project: hipsparselt** | Includes `projects/hipblaslt/tensilelite/**` (TensileLite is shared). |
| **ci:hipsparselt-fast** | Any change under `projects/hipblaslt/tensilelite/**` only. |
| **shared: origami** | Any change under `shared/origami/**`. |
| **shared: rocroller** | Any change under `shared/rocroller/**`. |

So a PR that only touches `projects/hipblaslt/` (e.g. library or test code) will get **project: hipblaslt**. A PR that also touches `shared/origami` or `shared/rocroller` still gets **project: hipblaslt** (and may get **shared: origami** or **shared: rocroller** as well). The repo also has **documentation**, **github actions**, **project: none** (when no project/shared paths change), and org/contributor labels applied by other workflows; the only one that **affects** TheRock CI is **skip-therockci** (if present, CI is skipped).

### 3.3 Labels vs what CI runs

- **project: hipblaslt** tells you "this PR touches hipblaslt (or origami/rocroller)". It does **not** tell the CI to run full tests — the CI already does that because the **paths** in the diff include `projects/hipblaslt/**`. So the label and the CI behavior are aligned but independent: both are driven by the same set of changed files, via different mechanisms (labeler for the label, `therock_configure_ci.py` for test_type and which projects to run).

---

## 4. Coverage tables (full, quick, smoke)

Below are the three coverage views: **full** (no filter), **quick** (`*quick*`), and **smoke** (`*smoke*`). Each cell is the **size-group letters** for array sizes (M×N×K) exercised at that (operation, precision); **Total tests** is the number of test instances in that row from the test generator (run `count_hipblaslt_tests_by_row.py` from `clients/tests` to refresh). **—** = no tests; **·** = N/A (no GEMM sizes). Size letters are defined in the legend after the tables.

### 4.1 Full (no filter)

Used by TheRock CI when test_type=full and by Math CI precheckin. All tests from matmul, auxiliary, smoke, and rocroller YAML. Suite size on the order of tens of thousands of tests.

| Operation group        | Sub-item                    | real (f16/bf16/f32) | f64 | 1b (f8/bf8) | 1b fnuz | int i8 | xf32 | mixed / swizzle | complex (f32_c/f64_c)¹ | Total tests |
|------------------------|-----------------------------|--------------------|-----|-------------|---------|--------|------|------------------|------------------------|-------------|
| **matmul core/sizes**  | core / bad-arg / NaN        | O,S,M,C            | O,S,M,C | O,S,M,C   | O,S,M,C | O,S,M,C | O,S,M,C | O,S,M,C          | M                       | 16,431      |
|                        | sizes / fixed shapes         | F                  | F   | F           | F       | F      | F    | F                | F                       | 4,747       |
|                        | algo / heuristic / MX       | S,M,F              | S,M,F | S,M,F     | S,M,F   | S,M,F  | S,M,F | S,M,F            | —                       | 1,500       |
|                        | stress / nightly shapes     | G,D,W              | G,D,W | D,W        | D,W     | —      | D,W  | —                | —                       | 3,616       |
| **matmul bias/activation** | bias (relu, gelu, swish, …) | S,E            | —   | S,E         | S,E     | —      | —    | —                | —                       | 3,496       |
|                        | gradients (dgelu, drelu, bgradb) | S,E         | —   | S,E         | S,E     | —      | —    | —                | —                       | 1,664       |
|                        | bias_gelu_aux / equality    | S,E                | —   | S,E         | S,E     | —      | —    | —                | —                       | 1,920       |
| **matmul grouped/ext** | grouped gemm                | S,M,F              | —   | S,M,F       | S,M,F   | —      | S,M,F | —                | —                       | 154         |
|                        | extended API (algo, swizzle) | S,M,F             | —   | S,M,F       | S,M,F   | —      | S,M,F | —                | —                       | 6,708       |

¹ *Complex (f32_c/f64_c) is supported and tested on **gfx942 and gfx950 only**. In the full suite there are **~2,300 complex test instances** (from 9 test templates × precisions/trans/sizes); all are pre_checkin and not in smoke or quick. So complex coverage is sparse relative to real/f64/1b.*

**Auxiliary and rocroller (full):**

| Area | What runs |
|------|-----------|
| **aux_*** | All of `auxiliary_gtest.yaml` (category pre_checkin): handle init/destroy/bad-arg; mat descriptor init, set/get attr, copy; matmul descriptor, pref, alg, plan init/set/get and bad workspace size; edge cases (get_sol with null bias, zero alpha); helper coverage (auxiliary_func, float8_func, hipblaslt_ext_op, rocblaslt_utility, status_func, etc.). One test runs a real GEMM: **aux_matmul_bad_ws_size** (size S, hpa_half). |
| **rocroller** | Single test from `rocroller_gtest.yaml`: **rocroller_predicate** — M/N/K 128×128×96, trans T/N, gfx950. |

### 4.2 Quick (`*quick*`)

Only tests with `category: quick` in matmul_gtest.yaml. Used by TheRock on Windows (gfx1151). No aux, no rocroller.

| Operation group        | Sub-item                    | real (f16/bf16/f32) | f64 | 1b (f8/bf8) | 1b fnuz | int i8 | xf32 | mixed / swizzle | complex (f32_c/f64_c) | Total tests |
|------------------------|-----------------------------|--------------------|-----|-------------|---------|--------|------|------------------|------------------------|-------------|
| **matmul core/sizes**  | core / bad-arg / NaN        | O,S                | O,S | O           | —       | O      | —    | —                | —                       | 14,408      |
|                        | sizes / fixed shapes         | F                  | —   | —           | —       | —      | —    | —                | —                       | 144         |
|                        | algo / heuristic / MX       | —                  | —   | —           | —       | —      | —    | —                | —                       | 0           |
|                        | stress / nightly shapes     | —                  | —   | —           | —       | —      | —    | —                | —                       | 0           |
| **matmul bias/activation** | bias (relu, gelu, swish, …) | —            | —   | —           | —       | —      | —    | —                | —                       | 0           |
|                        | gradients (dgelu, drelu, bgradb) | —             | —   | —           | —       | —      | —    | —                | —                       | 0           |
|                        | bias_gelu_aux / equality    | —                | —   | —           | —       | —      | —    | —                | —                       | 0           |
| **matmul grouped/ext** | grouped gemm                | —                  | —   | —           | —       | —      | —    | —                | —                       | 0           |
|                        | extended API (algo, swizzle) | —                 | —   | —           | —       | —      | —    | —                | —                       | 0           |

**Auxiliary and rocroller (quick):**

| Area | What runs |
|------|-----------|
| **aux_*** | None. Quick filter selects only `category: quick` from matmul_gtest; auxiliary tests are pre_checkin only. |
| **rocroller** | None. Rocroller tests are pre_checkin only. |

### 4.3 Smoke (`*smoke*`)

Only tests from smoke_gtest.yaml (`category: smoke`). Used by TheRock presubmit by default. Small sizes (E) or fixed 128; no aux, no rocroller.

| Operation group        | Sub-item                    | real (f16/bf16/f32) | f64 | 1b (f8/bf8) | 1b fnuz | int i8 | xf32 | mixed / swizzle | complex (f32_c/f64_c) | Total tests |
|------------------------|-----------------------------|--------------------|-----|-------------|---------|--------|------|------------------|------------------------|-------------|
| **matmul core/sizes**  | core / bad-arg / NaN        | E                  | E   | E           | E       | E      | E    | —                | —                       | 992         |
|                        | sizes / fixed shapes         | —                  | —   | —           | —       | —      | —    | —                | —                       | 0           |
|                        | algo / heuristic / MX       | —                  | —   | —           | —       | —      | —    | —                | —                       | 0           |
|                        | stress / nightly shapes     | —                  | —   | —           | —       | —      | —    | —                | —                       | 0           |
| **matmul bias/activation** | bias (relu, gelu, swish, …) | E              | —   | —           | —       | —      | —    | —                | —                       | 552         |
|                        | gradients (dgelu, drelu, bgradb) | —             | —   | —           | —       | —      | —    | —                | —                       | 0           |
|                        | bias_gelu_aux / equality    | —                | —   | —           | —       | —      | —    | —                | —                       | 0           |
| **matmul grouped/ext** | grouped gemm                | —                  | —   | —           | —       | —      | —    | —                | —                       | 0           |
|                        | extended API (algo, swizzle) | —                 | —   | —           | —       | —      | —    | —                | —                       | 0           |

**Auxiliary and rocroller (smoke):**

| Area | What runs |
|------|-----------|
| **aux_*** | None. Smoke filter selects only tests from smoke_gtest.yaml (matmul smoke variants); no auxiliary tests. |
| **rocroller** | None. No rocroller tests in smoke. |

### Size-group legend

| Letter | Size group | Description (M×N×K) |
|--------|------------|---------------------|
| **O** | one | 1×1×1 up to 1024×1×512; boundary dims (63–65, 127–129, 255–257) |
| **S** | small | 8×8×8 through 72×72×72 |
| **M** | medium | 8×8×8, 16×16×16, 128×128×64 |
| **E** | smoke | 8×8×8, 16×16×16, 32×32×32 |
| **C** | chunk | 24000×256×256 |
| **G** | grid_limit | Large skinny (e.g. 4096×131072×1) |
| **D** | deepbench | DeepBench irregular shapes |
| **W** | workload | ResNet50, Inception4, ctest shapes |
| **F** | fixed | Single points (matmul_8 … matmul_1024, small2, k0) |
| **R** | rocroller | 128×128×96 |

Transposition is exercised on nearly all matmul tests (N/N, N/T, T/N, T/T).

Together, the test overview (§1), the two CIs (§2), labels (§3), these coverage tables (§4), and coverage gaps (§5) give a complete picture of hipBLASLt test coverage.

---

## 5. Coverage gaps and improvements

This section summarizes known gaps in the current test suite and opportunities to improve coverage. It does not change what is already covered (described in §1–§4) but clarifies what is *not* exercised today. **Tracking:** These gaps and the improvements below are tracked under Epic **AIHPBLAS-1003** (hipBLASLt Testing & Quality Improvements – Prevent Production Crashes). Related production issues (segfaults, data races, multi-GPU failures in JAX/XLA and training workloads) are linked in that Epic; infrastructure work (e.g. sanitizer CI) is tracked in **ROCM-658**; TheRock/CI parity in **AIDEVOPS-73**, **AIDEVOPS-84**, **AIDEVOPS-85**.

### 5.1 Numerical and data-quality gaps

- **NaN:** There is **one** explicit test (**alpha_beta_zero_NaN**) that uses NaN in alpha/beta to confirm the implementation does not propagate NaN inappropriately. No other test fills input matrices or scalars with NaN.
- **Infinity:** No test sets alpha, beta, or input tensors to infinity. Internal validation paths may handle Inf, but the hipblaslt-test regression suite has no dedicated Inf test.
- **Denormals (subnormals):** No test explicitly fills matrices with denormal (subnormal) floating-point values. Behavior in the denormal range is untested by this suite.
- **Zero matrices:** No matmul GTest sets `initialization: zero`. All-zero A, B, or C are not explicitly covered (only alpha=0 or beta=0 in some tests).
- **Ill-posed / order-of-summation–sensitive inputs:** A classic numerical stress case is **one large value and many small values** — the sum can differ with the order of summation (floating point is not associative). GEMM does not guarantee order, so implementations may legitimately differ; testing such inputs would help ensure the library does not overflow or produce NaNs and would document acceptable tolerance. The codebase has a **special** initialization mode (A = large half, B = tiny half, C = small values) designed for this, but **no matmul GTest in the YAML uses it**. Adding a small matmul test with `initialization: special` would close this gap.

These numerical and data-quality gaps are in scope for Epic **AIHPBLAS-1003**; unit-testing and edge-case work is also tracked in **AIMA-22**.

### 5.2 Data initialization (what is used today)

Matrix and vector data are filled by an **initialization** mode (implementation in `clients/common/src/hipblaslt_init_device.cpp`). The RNG is **deterministic** (index-based), so the same test produces the same inputs every run.

| Mode        | Description                         | Used in matmul/smoke GTest?     |
|-------------|-------------------------------------|----------------------------------|
| **hpl**     | Pseudo-random in [-0.5, 0.5]       | Yes (default for almost all)     |
| **trig_float** | sin/cos-based deterministic values | Yes (a few tests)             |
| **rand_int**| Small integers                      | Bench only; not in matmul YAML  |
| **zero**    | All zeros                           | No                              |
| **special** | Large half + tiny half + small C    | No (implemented, not used)      |
| **norm_dist** | Normal distribution              | No (implemented, not used)      |
| **uniform_01** | [0, 1]                           | No (implemented, not used)      |

So today the suite is mostly **deterministic pseudo-random in [-0.5, 0.5]** (hpl). The modes **zero**, **special**, **norm_dist**, and **uniform_01** exist in code but are not exercised by the GTest YAML.

**Complex data types** use the same initialization modes. The device init code has explicit branches for `std::complex<float>` and `std::complex<double>` for **hpl**, **trig_float**, **rand_int**, and **zero**: real and imaginary parts are filled separately (e.g. for **hpl**, each component gets the same deterministic pseudo-random in [-0.5, 0.5] with an offset seed for the imaginary part). The matmul complex tests do not override `initialization` in the YAML, so they use the default **hpl** like most of the suite. Modes **special**, **norm_dist**, and **uniform_01** are not defined in a complex-specific way in the current init code (they target real or half types).

### 5.3 Multithreaded and multi-stream tests

The main **hipblaslt-test** suite does **not** run any tests with multiple CPU threads or multiple HIP streams. The test harness supports it: YAML can set `threads` and `streams`, and the code uses `launch_test_on_threads` / `RUN_TEST_ON_THREADS_STREAMS` to run the same test on multiple threads and streams. In `hipblaslt_common.yaml`, however, the only active combination is **threads: 0, streams: 0** (single-thread, single-stream). All other combinations (e.g. threads: 3, streams: 3) are commented out. So concurrent execution of GEMM or API calls from multiple threads is not exercised in CI.

**Exception:** The **TensileLite** tests (out of scope for the main suite; see §5.6) include one multithreaded test: **ConcurrentFindBestSolutionIsStable**, which runs 16 threads concurrently calling the library-selection API. That stresses thread-safety of the contraction library lookup, not the hipBLASLt GEMM path. *(Epic: AIHPBLAS-1003; production issues: SWDEV-484012 multi-GPU failures, SWDEV-568258 data races / heap corruption.)*

### 5.4 ASAN and TSAN builds

The codebase supports **AddressSanitizer (ASAN)** and **ThreadSanitizer (TSAN)** via CMake options (`HIPBLASLT_ENABLE_ASAN`, `HIPBLASLT_ENABLE_TSAN`) and `install.sh --address-sanitizer`. The test binary can be built and run with these sanitizers enabled for local or one-off use. **CI does not** run the hipblaslt-test suite with ASAN or TSAN builds. So we do not get regular coverage for memory errors (ASAN) or data races (TSAN) from the standard TheRock or Math CI pipelines. *(Epic: AIHPBLAS-1003; sanitizer CI infrastructure: ROCM-658.)*

### 5.5 Complex data type and tier coverage

**Complex (f32_c/f64_c)** GEMM is tested only in the **full** suite (pre_checkin), on **gfx942 and gfx950**. There are no complex tests in the **smoke** or **quick** filters. So presubmit (smoke) and Windows/quick runs do not exercise complex GEMM; only full runs on nodes with gfx942 or gfx950 do. If complex regressions are a concern for a change, run the full suite or a filter that includes the matmul_*_complex tests on an appropriate arch.

### 5.6 Architecture coverage

Tests run on a **limited set of GPU architectures**. TheRock CI typically runs on **gfx94X** and **gfx1151** (and optionally gfx950; the gfx950 test job has been disabled in the past due to capacity). Math CI runs on **gfx950, gfx1200, gfx1201, gfx942, gfx90a** across its nodes. So in practice we cover a small number of CDNA/consumer families. We do **not** run the full test suite on a wide matrix of older or alternate architectures (e.g. other gfx9xx, gfx10xx, or additional gfx11/gfx12 variants). Gaps here mean bugs that are architecture-specific may go undetected until later in the release cycle or in the field. *(Epic: AIHPBLAS-1003; multi-GPU/production issues: SWDEV-484012, SWDEV-574692.)*

### 5.7 Out of scope for this report

- **Code coverage (line/branch %):** This document describes *test* coverage (scenarios, operations, precisions, sizes). It does not report code coverage metrics; those would require a separate instrumentation and reporting setup.
- **TensileLite:** The TensileLite library under `projects/hipblaslt/tensilelite/` has its own C++ GTest, Python tests, and CPU driver tests. They are separate from the main hipblaslt-test suite and are not summarized here.
- **Samples:** The `clients/samples/` programs are demos; they are not part of the automated regression suite described in this report.

### 5.8 Improvement opportunities (summary)

All of the below align with Epic **AIHPBLAS-1003** and its child work; production issues (e.g. SWDEV-576540, SWDEV-484012, SWDEV-568258, SWDEV-553776, SWDEV-565755) are linked in the Epic.

- Add tests that use **initialization: zero** and **initialization: special** (and optionally norm_dist / uniform_01) to exercise data-quality and ill-posed regimes.
- Add dedicated tests for **Inf** in alpha, beta, or inputs, and for **denormal** inputs, if the product requirements call for defined behavior in those cases.
- **Multithreaded / multi-stream:** Uncomment or add YAML entries so that at least a subset of tests run with `threads > 0` and/or `streams > 1` to exercise concurrent handle/stream usage *(AIHPBLAS-1003)*.
- **ASAN / TSAN in CI:** Add periodic or optional CI jobs that build with ASAN (and, where feasible, TSAN) and run the test suite to catch memory and threading bugs early *(AIHPBLAS-1003; infrastructure: ROCM-658)*.
- **Broader architecture matrix:** Where resources allow, run the suite (or a curated subset) on more GPU families to reduce the risk of architecture-specific regressions *(AIHPBLAS-1003; CI parity: AIDEVOPS-73, AIDEVOPS-84, AIDEVOPS-85)*.
- **Batched GEMM bands:** Add or expand tests for medium batch (129–1024) and large batch (1025–8192), and optionally more points in small batch (e.g. 2, 32, 64, 128).
- Consider reporting **code coverage** (e.g. from Math CI codecov or a TheRock run) in a separate dashboard so stakeholders can see line/branch coverage alongside this scenario-based view.