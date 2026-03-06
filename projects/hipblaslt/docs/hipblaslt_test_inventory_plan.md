# hipBLASLt Test Inventory — Investigation & Report Plan

**Purpose:** Define how to produce a report of test categories, counts, run frequency, duration, and coverage for `rocm-libraries/projects/hipblaslt`.

---

## 1. Test Structure Overview

### 1.1 Main regression suite: `hipblaslt-test` (C++ GTest)

| Item | Location | Role |
|------|----------|------|
| **Test data** | Generated at build time | `hipblaslt_gentest.py` reads YAML, expands combinations, writes binary `hipblaslt_gtest.data` |
| **YAML inputs** | `clients/tests/data/` | `hipblaslt_gtest.yaml` → includes `matmul_gtest.yaml`, `auxiliary_gtest.yaml`, `smoke_gtest.yaml`, `rocroller_gtest.yaml` (+ `hipblaslt_common.yaml`, `matmul_common.yaml`, `known_bugs.yaml`) |
| **Test binary** | `clients/hipblaslt-test` | Reads `.data` file, instantiates one GTest parameterized test per record; filtering by category is via `--gtest_filter` (e.g. `*smoke*`, `*quick*`, `*pre_checkin*`, `*nightly*`) |
| **Run scripts** | `rtest.py` | Wraps `hipblaslt-test` with `--gtest_filter=*smoke*` (emulation smoke), `*quick*` (regression), or `*pre_checkin*:*nightly*` (extended) |

Expansion is **Cartesian**: each YAML “test” is a template. The generator expands over:

- `function` (often paired with precision via `function: precision`)
- `matrix_size` (ranges like `small_matrix_size_range`, `medium_matrix_size_range`, etc.)
- `transA_transB` (N/N, N/T, T/N, T/T or conjugate variants)
- `alpha_beta` (e.g. 4 combinations)
- `precision` (a_type, b_type, c_type, d_type, compute_type, scale_type from `*real_precisions`, `*real_precisions_1b`, etc.)
- `batch_count`, `c_equal_d`, `bias_vector`, `activation_type`, `gpu_arch`, etc.

Deduplication: identical `Arguments` byte records are written once. So **total test count** = number of records in `hipblaslt_gtest.data` (or the number of tests listed by `hipblaslt-test --gtest_list_tests`).

### 1.2 TensileLite tests (C++ GTest + Python + CPU driver tests)

| Layer | Location | Description |
|-------|----------|-------------|
| **C++ GTest** | `tensilelite/tests/` | `tensilelite-tests` executable: `test.cpp`, `DataTypes_test.cpp`, `ContractionSelectionLibrary_test.cpp`, `RangeLibrary_test.cpp`, `Predicates_test.cpp`, GEMM kernel tests via `RunGEMMKernelTest.hpp`; `gtest_discover_tests(..., TIMEOUT 60)` |
| **CPU GEMM driver** | Same CMake | `add_test(CPUGemm.*, cpu-gemm-driver ...)` — fixed set of invocations (F32_Sanity, F32_TransA/B/AB, F32_Odd_Sizes, BF16, F16, FastPath_On/Off) |
| **Python** | `tensilelite/Tensile/Tests/` | Unit tests (e.g. `test_Validate*.py`, `test_CustomSchedule*.py`, `test_EstimateQuadCyclesForValidator.py`, etc.) and common/config tests |
| **rocisa** | `tensilelite/rocisa/test/` | Python tests for enum, label, instruction, code, container, cycle, base |

These are **separate** from the main hipBLASLt API regression: they exercise TensileLite library, ISA, and CPU driver, not the `hipblaslt-test` binary.

### 1.3 Samples

`clients/samples/` — many sample programs (e.g. 01_hipblaslt_gemm through 27_hipblaslt_gemm_clamp_bias). These are demos, not an automated test suite for the report unless you explicitly include “samples run” as a category.

### 1.4 TheRock CI (rocm-libraries/.github)

TheRock CI lives under **rocm-libraries/.github**. Relevant workflows and scripts:

| File | Role |
|------|------|
| **workflows/therock-ci.yml** | Main CI: on push/PR to develop or release/therock-*, and workflow_dispatch. Runs a **setup** job (therock_configure_ci.py), then **therock-ci-linux** and **therock-ci-windows** matrices. |
| **workflows/therock-ci-linux.yml** | Build on `azure-linux-scale-rocm` (TheRock build image), then calls **therock-test-packages.yml** with `projects_to_test`, `test_type`, `amdgpu_families`, `test_runs_on`. |
| **workflows/therock-ci-nightly.yml** | Scheduled (cron 7 AM UTC) + workflow_dispatch. Same structure as main CI; uses fixed `AMDGPU_FAMILIES` (gfx94X, gfx950 on Linux; gfx1151 on Windows). |
| **workflows/therock-test-packages.yml** | Runs **configure_test_matrix** on the test runner: calls TheRock's **fetch_test_configurations.py** with `PROJECTS_TO_TEST`, `TEST_TYPE`, `AMDGPU_FAMILIES`. Outputs a **components** matrix (one entry per test job × shard). Then runs **therock-test-component.yml** for each component. |
| **workflows/therock-test-component.yml** | For each component: checks out TheRock, sets up test env (downloads build artifacts), runs **component.test_script** with `TEST_TYPE` and shard env vars (`SHARD_INDEX`, `TOTAL_SHARDS`). |
| **scripts/therock_configure_ci.py** | Determines which projects to build/test from PR labels and changed paths; writes `linux_projects`, `windows_projects`, **test_type**. |

**test_type** is set in **TheRock** by `build_tools/github_actions/configure_ci.py` (used when TheRock is the trigger). For rocm-libraries CI the setup job runs **therock_configure_ci.py** (in rocm-libraries), which uses **therock_matrix.py** and **config_loader**; the **test_type** output is passed into the Linux/Windows workflows and then into **fetch_test_configurations.py**:

- **smoke** — default for presubmit when no submodule/test-label triggers; only one shard per job.
- **full** — scheduled (nightly), or when submodules changed, or when test labels are specified.

**hipblaslt** is wired in **TheRock**'s `build_tools/github_actions/fetch_test_configurations.py`:

- **job_name:** `hipblaslt`
- **test_script:** `test_hipblaslt.py` (in TheRock)
- **timeout_minutes:** 180
- **total_shards_dict:** linux 6, windows 1
- **fetch_artifact_args:** `--blas --tests`

**TheRock**'s `test_hipblaslt.py`:

- Reads `TEST_TYPE` (default `"full"`). On **Windows** with `AMDGPU_FAMILIES=gfx1151` it overrides to **quick** (memory constraints).
- **smoke** → `--gtest_filter=*smoke*`
- **quick** → `--gtest_filter=*quick*`
- **full** → no filter (entire suite)
- Runs `hipblaslt-test` with GTest sharding via `GTEST_SHARD_INDEX` / `GTEST_TOTAL_SHARDS`.

So in CI:

| Trigger / condition | test_type | hipblaslt-test filter | Shards (Linux) |
|----------------------|-----------|------------------------|----------------|
| Presubmit (PR/push), no full-test triggers | smoke | `*smoke*` | 1 |
| Scheduled (nightly), or submodule/test-label | full | (none) | 6 |
| Windows gfx1151 | quick | `*quick*` | 1 |

**Architectures (GPU families):** The setup job calls TheRock's **fetch_package_targets.py** with `AMDGPU_FAMILIES` (default Linux: `gfx94X, gfx950`; default Windows: `gfx1151`). That produces a matrix of target bundles (one per family × platform) from TheRock's **amdgpu_family_matrix.py** (e.g. `gfx94X-dcgpu`, `gfx950-dcgpu`, `gfx1151`). So by default **3** architectures are requested (2 Linux + 1 Windows). As of this writing, **therock-ci-linux.yml** skips the *test* job for `gfx950-dcgpu` (build still runs); only **gfx94X-dcgpu** runs tests on Linux. So hipblaslt tests actually run on **2** architectures: **gfx94X-dcgpu** (Linux) and **gfx1151** (Windows). The full list of supported families (presubmit/postsubmit/nightly) is in TheRock `build_tools/github_actions/amdgpu_family_matrix.py`; workflow_dispatch can pass a different comma-separated list to test more or fewer.

The **rocm-libraries** `rtest.py` (emulation/local) uses `-e smoke | regression | extended` → `*smoke*` \| `*quick*` \| `*pre_checkin*:*nightly*`; that is separate from the above CI (no "pre_checkin"/"nightly" filter in TheRock CI — "full" runs everything).

### 1.5 Math CI (Jenkins, rocjenkins)

**Math CI** is an older Jenkins-based CI for the ROCm math libraries (repo: **rocjenkins**, typically at `rocjenkins` or e.g. `/root/coding/rocjenkins`). It posts status to GitHub as **"Math CI Summary"** and is being superseded by TheRock CI. Below is what it exercises for **hipblaslt**.

**Host / config:** `http://math-ci.amd.com`, GitHub status context `Math CI Summary`. PRs to **ROCm/rocm-libraries** that touch hipblaslt (or labels like `project: hipblaslt`) trigger hipblaslt jobs; a **status-gate** job then aggregates results and sets the Math CI Summary check.

**hipblaslt project definition** (from `resources/com/amd/monorepo/projects.json`):

- **name:** `hipblaslt`
- **basePath:** `projects`
- **sparseCheckoutPaths:** hipblaslt, hipblas-common, rocroller, mxdatagenerator, origami
- **jobTypes:** codeql, codecov, perfci, performance, performance-testonly, precheckin, precheckin-lowresource, precheckin-bringup, precheckin-screening-*, precheckin-testonly, preliminary, preliminary-screening, static-analysis, status-gate
- **gatingJobs:** `precheckin`, `static-analysis`, `preliminary` — these are the jobs that must pass for the Math CI Summary to pass. If the PR also touches **rocroller**, the status gate **ignores** the `preliminary` job for hipblaslt.

**What each hipblaslt job does:**

| Job | What it exercises |
|-----|-------------------|
| **precheckin** | Build via `./install.sh -c -j 32` (split compilation, multi-arch on PRs: gfx950 on rhel9, gfx1200;gfx1201 on sles15, gfx942;gfx90a on ubuntu22). Then: **`/opt/rocm/bin/hipblaslt-test`** (full suite, no gtest filter), **hipblaslt-bench-groupedgemm-fixed-mk** (one bench), and on non-gfx950 nodes a long list of **samples** (e.g. sample_hipblaslt_gemm, sample_hipblaslt_ext_op_amax, …). Test dir `/opt/rocm/bin/hipBLASLt`, timeout test 780 min. |
| **preliminary** | No build/package; runs **TensileLite Python tests** only: `tox` in `tensilelite` for `Tensile/Tests` with `-m common`, plus on develop (or CMS branches) `tox -e unit -- Tensile/Tests/unit`. On branches `hipblaslt_common_cms_dev` / `hipblaslt_common_cms_phase2` runs only `-k custom_mainloop_scheduling`. Runs only if there are changes under `tensilelite` vs `origin/develop`. |
| **static-analysis** | Static analysis stage (scan timeout 60 min). |
| **static_build** | Build with `./install.sh -c --static -j 32`, then same test flow as precheckin (hipblaslt-test, bench, samples) with `TENSILE_CLIENT_STATIC=TRUE`. |
| **nightly** | **No tests.** Runs TensileCreateLibrary (TCL) for gfx942 with 32 and 64 jobs, collects "kernels per second" and memory stats, emails an HTML report. Used for TCL performance tracking, not functional tests. |
| **performance** / **perfci** | Build (with deps: hipBLAS-common); clone ref-repo; run **hipblaslt-perf** (PTS) comparing new vs reference build (e.g. `ci_perf_job` suite or `all`), and/or PTS workload benchmark. Uses datasets from math-ci (e.g. `perfci_bench_iter_1000.yaml`, `pts_hipblaslt_*`). |
| **performance-testonly** | Same as performance but test-only (no build). |
| **codeql** | CodeQL analysis; build with `./install.sh -c`. |
| **codecov** | Build with coverage (`HIPBLASLT_ENABLE_COVERAGE=ON`, gfx950 on ubuntu22), then run tests with **gtest filter** `*pre_checkin*-*aux_auxiliary_func_f16_r*:*aux_rocblaslt_utility_func_f16_r*` and upload coverage. |

**On-demand / debug:** `vars/onDemandHipblaslt.groovy` builds a Docker image for hipblaslt (ubuntu24/22, gfx950/gfx942, ROCm mainline or custom branch/commit) and pushes to Harbor for debug use.

**Downstream:** hipblaslt is in **downstreamTriggers** for **rocblas** — rocblas precheckin (and related jobs) build with libraryDependencies including hipBLAS-common and hipBLASLt when not disabled by PR label `noHipblasLT`.

**Summary for the test inventory:** Math CI’s hipblaslt **precheckin** runs the full `hipblaslt-test` binary (no filter), plus one groupedgemm bench and a fixed set of samples. **preliminary** runs only TensileLite Python (tox) when tensilelite changes. **codecov** runs a subset of tests via gtest filter (pre_checkin minus some aux tests). **performance**/perfci run hipblaslt-perf (PTS), not the gtest suite. So for “how many gtest tests run in Math CI” the answer is: **all of them** in precheckin (and the codecov subset in codecov).

**Architectures and sharding (Math CI):** From **hipblaslt/common.groovy**, on PRs the build is compiled for one GPU target set per node type: **rhel9** → gfx950, **sles15** → gfx1200;gfx1201, **ubuntu22** → gfx942;gfx90a. So there are **3 node types** (OS × runner) and **5 GPU architectures** (gfx950, gfx1200, gfx1201, gfx942, gfx90a). The exact `nodeDetails` (OS/GPU list) passed into the pipeline is set by the Jenkins job that invokes the library (job config or pipeline script), not in rocjenkins; the library only defines what to compile for when `platform.jenkinsLabel` contains that OS. Math CI does **not** use GTest sharding: `hipblaslt-test` is run with no `GTEST_SHARD_INDEX` or `GTEST_TOTAL_SHARDS`. Each node runs the full suite once. So there is **1 run per node** (no shards); parallelism is multiple nodes in parallel, not multiple shards of one run.

### 1.6 Full view: what the full (unfiltered) run covers

In both TheRock CI (test_type=full) and Math CI (precheckin), the main regression run is **hipblaslt-test with no `--gtest_filter`**. That executes every test in the generated suite (all records in `hipblaslt_gtest.data`). Below is a full view of what that covers.

**Definition.** The **full** run = one invocation of `hipblaslt-test` with no filter. Every test from the four YAML inputs runs: matmul, auxiliary, smoke, rocroller. Total test count is the number of expanded records (e.g. ~64k from CI; exact from `hipblaslt-test --gtest_list_tests`).

**Narrative.** The full run exercises the entire regression suite generated from the YAML definitions. It covers the hipBLASLt matmul API (plain GEMM over sizes, transpositions, alpha/beta, and precisions; grouped gemm; extended op with bias, activation, and gradients), the auxiliary handle/mat/matmul descriptor and preference/algorithm APIs, rocroller integration, and bad-argument and edge-case tests. Tests are parameterized over precision, matrix sizes (M/N/K), transposition, alpha/beta, and—where applicable—batch count, bias, activation type, and GPU architecture. All tiers are included: smoke, quick, pre_checkin, nightly, and HMM (and known_bug where not skipped by platform).

---

#### By test definition source (YAML to suite)

| Source | Role | YAML categories | What it exercises |
|--------|------|-----------------|-------------------|
| **matmul_gtest.yaml** | Main GEMM/matmul matrix | quick, pre_checkin, nightly, HMM | Core matmul, sizes, trans, alpha/beta, bad_arg, NaN; extended (bias, activation, gradients, batch, grouped gemm, f8/mixed/int, streamK, swizzle, stride, conv3d, MX); stress/nightly (grid_limit, deepbench, resnet50/inception/ctest, heuristic/algo). |
| **auxiliary_gtest.yaml** | Auxiliary API | pre_checkin | Handle, mat, matmul/pref/alg/plan init and get/set attr; edge cases; helper coverage. |
| **smoke_gtest.yaml** | Fast sanity | smoke | Same areas as matmul but small sizes and fewer combinations; bias/f8/int smoke. |
| **rocroller_gtest.yaml** | Rocroller | pre_checkin | Single rocroller predicate config (M/N/K 128/128/96, trans T/N, gfx950). |

**matmul_gtest.yaml — test names (grouped).** Each name is a template; the generator expands over precision, sizes, trans, alpha/beta, etc.

- **Core / bad-arg / NaN:** matmul_bad_arg, alpha_beta_zero_NaN, matmul_one, matmul_small, matmul_conj_small, matmul_medium, matmul_batch_medium, matmul_medium_HMM, matmul_chunk.
- **Sizes / fixed shapes:** matmul_8, matmul_16, matmul_24, matmul_32_8_128, matmul_48_8_128, matmul_64_8_128, matmul_64_8, matmul_8_64, matmul_96, matmul_128, matmul_128_streamk, matmul_256, matmul_256_8_16, matmul_16_256_8, matmul_8_16_256, matmul_512, matmul_1024, matmul_small2, matmul_k0.
- **Precision / f8 / mixed / int:** matmul_bf16, matmul_bf16_fp32dst, matmul_f8_bf8_dst_* (fp32/bf16/f16), matmul_f8_bf8_fnuz_dst_*, matmul_input_fp16_computeIn_*; gfx12 variants; matmul_real_1b_dst_f8_*, matmul_real_1b_fnuz_dst_*, matmul_one_real_precisions_1b_gfx12, matmul_one_integer_precisions_i8_gfx12; matmul_fallback_*; matmul_gemm_xf32, matmul_gemm_f32_fast_bf16, matmul_gemm_double, matmul_gemm_i8_dst_i32/i8 (94x/1xxx), matmul_gemm_mix_precisions*, matmul_gemm_amaxAB_mix_precisions_fp8_fnuz.
- **Bias / activation / gradient:** matmul_bias_relu, matmul_bias_sigmoid, matmul_bias_relu_streamk, matmul_bias_gelu, matmul_bias_swish, matmul_bias_clamp, matmul_bias_only, matmul_bias_type; matmul_dgelu_*, matmul_drelu_*, matmul_bgradb_*; matmul_bias_gelu_aux_*, matmul_equality_NN_bias_*, matmul_relu_clamp_useE.
- **Grouped gemm / extended API:** matmul_groupedgemm, matmul_groupedgemm_specific_sizes, matmul_groupedgemm_f8_fnuz/bf16/xf32/zero_n, matmul_extapi_groupedgemm, matmul_extapi_algo_method_gemm, matmul_extapi_gemm, matmul_extapi_algo_method_tuning_*, matmul_extapi_swizzleA/B, matmul_swizzleA/B.
- **Algo / heuristic / MX:** matmul_algo, matmul_heuristic_all_solutions*, matmul_mx_datatypes, matmul_mx_large, matmul_mx_all_kernels, matmul_mx_solution_index, matmul_mx_multiple_kernels, matmul_mx_preswizzle_32x8_large.
- **Stress / nightly / shapes:** matmul_grid_limit_* (real, real_1b, double), matmul_deepbench, resnet50_fwd/bwdwrw/bwddata, inception4_*, ctest_*, matmul_large_nt_*, matmul_conv3d_kernels, matmul_stride_lt_ld, matmul_one_bf16_fp32dst_precision_gfx90a.

**auxiliary_gtest.yaml — test names (all).** Each exercises a specific auxiliary API path or edge case.

- **Handle:** aux_handle_init_bad_arg, aux_handle_destroy_bad_arg, aux_handle.
- **Mat descriptor:** aux_mat_init_bad_arg, aux_mat_destroy_bad_arg, aux_mat_set_attr_bad_arg, aux_mat_get_attr_bad_arg, aux_mat_set_get_attr, aux_mat_copy.
- **Matmul descriptor / pref / alg / plan:** aux_matmul_init_bad_arg, aux_matmul_init, aux_matmul_set_attr_bad_arg, aux_matmul_get_attr_bad_arg, aux_matmul_set_get_attr; aux_matmul_pref_get_attr_bad_arg, aux_matmul_pref_get_attr, aux_matmul_pref_init_bad_arg, aux_matmul_pref_init; aux_matmul_alg_init_bad_arg, aux_matmul_alg_init, aux_matmul_alg_get_attr_bad_arg; aux_matmul_alg_null_matmul, aux_matmul_bad_ws_size.
- **Edge / solution:** aux_get_sol_with_null_biasaddr, aux_get_sol_with_zero_alpha_null_a_b, aux_get_sol_with_zero_alpha_null_a_b_ext.
- **Helper coverage:** aux_auxiliary_func, aux_float8_func, aux_hipblaslt_ext_op_func, aux_rocblaslt_utility_func, aux_status_func, aux_hipblaslt_func, aux_tensile_host_func, aux_tuple_helper_equal_func, aux_rocblaslt_rocroller_host_func.

**smoke_gtest.yaml — test names (all).** Smoke versions of matmul and precision variants; small sizes, fewer combinations.

- matmul_smoke; matmul_bias_relu_smoke, matmul_bias_sigmoid_smoke, matmul_bias_gelu_smoke, matmul_bias_only_smoke, matmul_bias_type_smoke.
- matmul_f8_bf8_fnuz_dst_fp16/bf16_smoke; matmul_f8_bf8_dst_*_gfx12_smoke; matmul_real_1b_dst_f8_*_gfx12_smoke; matmul_one_integer_precisions_i8_gfx12_smoke; matmul_real_1b_fnuz_dst_*_smoke; matmul_gemm_xf32_smoke, matmul_gemm_double_smoke, matmul_gemm_i8_dst_i32/i8_*_smoke.

**rocroller_gtest.yaml.** One test template: rocroller_predicate (M/N/K 128/128/96, trans T/N, gpu_arch 950, pre_checkin).

---

#### By library surface (what gets tested)

| Surface | Covered by full run |
|---------|---------------------|
| **Core matmul API** | hipblasLtMatmul: GEMM over all size ranges (one/small/medium/chunk/batch), trans (N/N, N/T, T/N, T/T, conjugate), alpha/beta, precisions (f16, bf16, f32, f64, f8/bf8, mixed, integer). Bad-arg and alpha/beta/NaN tests. |
| **Extended matmul** | Bias (vector, type), activation (relu, gelu, sigmoid, swish, clamp), bias gradients (dgelu, drelu, bgradb), batch, grouped gemm, streamK, fallback paths, swizzle (A/B), stride, conv3d. |
| **Precision / data types** | Real (f16, bf16, f32, f64), 1-byte (f8, bf8, fp8_fnuz), mixed/compute, integer (i8 to i32/i8), xf32; arch-specific (gfx94x, gfx12) where defined. |
| **Auxiliary APIs** | Handle init/destroy; mat init/destroy, set/get attr, copy; matmul init, set/get attr; preference init/get attr; algorithm init/get attr; plan init; workspace-size and null-matmul edge cases. |
| **Helper / internal** | aux_auxiliary_func, aux_float8_func, aux_hipblaslt_ext_op_func, aux_rocblaslt_utility_func, aux_status_func, aux_hipblaslt_func, aux_tensile_host_func, aux_tuple_helper_equal_func, aux_rocblaslt_rocroller_host_func. |
| **Error / edge** | matmul_bad_arg, alpha_beta_zero_NaN; aux_*_bad_arg and aux_get_sol_with_* (null bias, zero alpha null A/B), aux_matmul_bad_ws_size. |
| **Special config** | HMM (unified memory), arch-specific (gfx942, gfx90a, gfx950, gfx12), stress/nightly (grid_limit, deepbench, resnet50, inception, ctest), rocroller predicate. |

**Tiers in the full run.** The full run includes every YAML category: **smoke** (smoke_gtest), **quick** (matmul one/small/conj_small and selected others), **pre_checkin** (most matmul plus all auxiliary plus rocroller), **nightly** (grid_limit, deepbench, resnet50, inception, ctest, heuristic, etc.), **HMM** (matmul_medium_HMM). Tests marked **known_bug** are still in the suite but may be skipped at runtime per known_bug_platforms.

---

#### Parameter space: precisions, sizes, transposition, alpha/beta

When we say tests are "executed over precision and sizes," the following are the concrete values and ranges defined in the YAML. The generator expands over these (and other fields) to produce the test matrix.

**Data types (hipDataType, from hipblaslt_common.yaml).** Base types used in precision tuples: **f16_r**, **f32_r**, **f64_r**, **bf16_r**, **i8_r**, **i32_r**, **f8_r**, **bf8_r**, **f8_fnuz_r**, **bf8_fnuz_r**, **f6_r**, **bf6_r**, **f4_r** (and invalid). **Compute types (hipblasComputeType_t):** c_f16_r, c_f32_r, c_f64_r, c_f32_fast_f16_r, c_f32_fast_bf16_r, c_xf32_r, c_i32_r (and pedantic variants).

**Precision sets (examples; all in hipblaslt_common.yaml).** Each set is a list of tuples `{ a_type, b_type, c_type, d_type, compute_type, scale_type }` (and optionally swizzle_a/b, compute_input_typeA/B):

| Set name | Count | Description |
|----------|-------|-------------|
| **real_precisions** | 3 | f16, bf16, f32 (all with compute c_f32_r). |
| **real_precisions_gemm_only** | 1 | f64 (double) with c_f64_r; often combined with real_precisions. |
| **real_precisions_2b** | 2 | f16, bf16 (2-byte real). |
| **real_precisions_1b** | 16 | f8/bf8 A/B, dst fp32/fp16/bf16/f8/bf8 (4 dst × 4 A/B combos). |
| **real_precisions_1b_fnuz** | 20+ | f8_fnuz/bf8_fnuz A/B, dst fp32/fp16/bf16/f8_fnuz/bf8_fnuz; many combinations. |
| **real_precisions_1b_fnuz_dst_f32/f16/bf16** | 4 each | Subsets by destination type. |
| **integer_precisions_i8** | 2 | i8×i8 → i32, i8×i8 → i8 (compute c_i32_r). |
| **real_precisions_intermeddiate** (xf32) | 1 | f32 with compute c_xf32_r. |
| **real_precisions_intermeddiate_bf16** | 2 | f16/f32 with c_f32_fast_bf16_r. |
| **real_precisions_intermeddiate_f8_fnuz** | 3 | f16 with compute_input_type A/B f8_fnuz/bf8_fnuz. |
| **real_precisions_swizzleA_support** | 4 | f16/bf16/f8_fnuz with swizzle_a: true. |
| **real_precisions_swizzleB_support** | 4 | Same with swizzle_b: true. |
| **real_mix_precisions_fnuz** | 4 | f8_fnuz×f16, f16×f8_fnuz → fp16/fp32. |
| **real_mix_precisions_fp8_fnuz** | 2 | Mixed → f8_fnuz. |
| **real_precisions_mx** | multiple | MX datatypes (scaleA/scaleB 3, f8/bf8/f6/bf6/f4, etc.). |
| **hpa_half_precision** (auxiliary) | 1 | f16×4, c_f32_r; used for aux tests. |

So "precision" in the report means: which of these sets a test uses, and within a set, each tuple is one (a_type, b_type, c_type, d_type, compute_type, scale_type) combination.

**Matrix size ranges (matmul_common.yaml).** Each range is a list of `{ M, N, K }` or `{ M, N, K, lda, ldb, ldc, ldd }`:

| Range name | # entries | Description |
|------------|-----------|-------------|
| **one_matrix_size_range** | 30 | Edge cases: (1,1,1), (1,1,512), (1,1024,512), (1024,1,512); K 63–65, 127–129, 255–257; M/N 63–65, 127–129, 255–257 along each axis. |
| **small_matrix_size_range** | 9 | Square 8×8×8 through 72×72×72 (ld* = size). |
| **medium_matrix_size_range** | 3 | 8×8×8, 16×16×16, 128×128×64. |
| **smoke_matrix_size_range** | 3 | 8×8×8, 16×16×16, 32×32×32. |
| **chunk_matrix_size_range** | 2 | Large: 24000×256×256 with lda/ldc/ldd variants. |
| **grid_limit_matrix_size_real** | 6 | Stress: (2, 2097152, 1), (64, 1048576, 1), (256, 1048576, 1), (1024, 524288, 1), (2048, 262144, 1), (3072/4096, 131072, 1). |
| **grid_limit_matrix_size_double** | 8 | Similar plus (3,2,1) and lda/ldb/ldc/ldd. |
| **deepbench_sizes** | 60+ | DeepBench workload shapes (e.g. 192×64×784, 12544×128×256, many M/N/K combos). |
| **resnet50_fwd/bwdwrw/bwddata_sizes** | 7–8 each | ResNet-50 layer shapes. |
| **inception4_fwd/bwdwrw/bwddata_sizes** | 6–8 each | Inception v4 shapes. |
| **ctest_bwdwrw_sizes** | 80+ | Convolution-test shapes (e.g. M×8×K with various M, K). |

So "sizes" means: which named range a test uses, and each (M, N, K [, lda, ldb, ldc, ldd]) in that range is one test geometry.

**Transposition (transA, transB).** From matmul_gtest.yaml / smoke_gtest.yaml:

- **transA_transB_range:** (N,N), (N,T), (T,N), (T,T) — four combinations.
- **conjugate_transA_transB_range:** (T,C), (C,T) for complex conjugate (used in conjugate tests).
- **deepbench_transA_transB_range** / **ldd_transA_transB_range:** (N,N), (N,T) only.

**Alpha / beta.** Scalar scaling; tests expand over:

- **alpha_beta_range:** (alpha 5, beta 0), (alpha 0, beta 3), (alpha 1, beta 3), (alpha 1, beta 1) — four combinations.
- **alpha_beta_range_small:** (alpha 2, alphai 2, beta -1, betai 2) for complex.
- **deepbench_alpha_beta_range:** (1,0), (1,1).
- Many tests fix alpha=1 and beta=[0, 2] or [0.0, 2.0]; others use alpha/beta lists inline.

**Other dimensions.** Activation: **hipblaslt_activation_type** none, relu, gelu, swish, clamp, sigmoid. Bias: bias_vector (0/1), bias_type (e.g. default, f32_r). **batch_count** (e.g. 10 for matmul_batch_medium; grouped_gemm uses grouped_gemm: [1,5], [1,7], [2]). **gpu_arch** in YAML: e.g. `'942'`, `'950'`, `'90a'`, `'120[0-1]'`, `'9(0a|42)'` (regex). **fortran**: [ false, true ] for bad-arg. **c_equal_d**: [ false, true ]. **initialization**: hpl, rand_int, etc. The generator also expands over **stride_a/b**, **swizzle_a/b**, **algo_method**, **use_ext**, **tensile_solution_selection_method**, and test-specific lists where defined.

---

## 2. Categories Useful for the Report

These are the dimensions that already exist or are easy to derive.

### 2.1 Run frequency / tier (when they run)

- **smoke** — `rtest.py -e smoke` → `--gtest_filter=*smoke*` (fast, emulation/smoke); **TheRock CI presubmit** uses this when test_type=smoke.
- **quick** — `rtest.py -e regression` → `*quick*` (short regression); **TheRock CI** uses this on Windows gfx1151 only.
- **full** — **TheRock CI** uses no gtest filter when test_type=full (nightly, submodule change, or test label); Linux runs 6 shards.
- **pre_checkin** — `rtest.py -e extended` → `*pre_checkin*:*nightly*` (longer); local/rtest only, not in TheRock CI.
- **nightly** — same filter as pre_checkin in rtest (extended); **TheRock nightly workflow** runs full suite (no filter), not this gtest category.
- **HMM** — unified memory tests (category in YAML)
- **known_bug** — tests that are skipped on certain platforms (via `known_bug_platforms`)

So “how often they run” is defined by **YAML `category`** and how CI/local scripts use `--gtest_filter`. (Today `match_test_category` always returns true; selection is by name pattern in filter.)

### 2.2 Operation / function

- **matmul** (and **matmul_bad_arg**), **matmul_* variants** (bias, activation, gradient, etc.)
- **aux_*** (handle, mat, matmul init/set/get, etc.) from `auxiliary_gtest.yaml`
- **rocroller_*** from `rocroller_gtest.yaml`
- Names come from YAML `name` and `function`; the C++ side uses `arg.function` and test name suffix.

### 2.3 Data type / precision

From `Arguments`: `a_type`, `b_type`, `c_type`, `d_type`, `compute_type`, `scale_type` (and optional `compute_input_typeA/B`). The YAML uses named sets, e.g.:

- **Real:** `real_precisions` (f16, bf16, f32), `real_precisions_gemm_only` (+ f64), `real_precisions_1b`, `real_precisions_1b_fnuz`, `real_precisions_2b`, `real_precisions_swizzleA/B_support`, `real_precisions_intermeddiate*`, `real_precisions_mx`, etc.
- **Integer:** `integer_precisions_i8`
- **Complex:** conjugate tests use `conjugate_transA_transB_range`; complex-specific precision sets if any

So “complex vs real” and “precision family” can be derived from the type fields (and from test name suffix, which includes datatype strings).

### 2.4 Size / dimension

- **matrix_size** in YAML: named ranges (e.g. `one_matrix_size_range`, `small_matrix_size_range`, `medium_matrix_size_range`, `smoke_matrix_size_range`, `chunk_matrix_size_range`, `grid_limit_*`, `deepbench_sizes`, `resnet50_*`, `ctest_*`, etc.).
- Each test record has **M, N, K** (and optionally batched); we can bin by (M×N×K) or by “small / medium / large / stress” if we define thresholds.

### 2.5 Architecture

- **gpu_arch** in YAML: e.g. `'942'`, `'950'`, `'90a'`, `'120[0-1]'`, `'9(0a|42)'`, `'1[1-2]\d{2}'`, etc. Some tests are arch-specific; others have no `gpu_arch` (run on all). The test binary can filter at runtime by current GPU; the report can list “tests that declare gpu_arch” vs “generic”.

### 2.6 Other dimensions

- **Batch:** `batch_count > 1`
- **Activation:** `activation_type` (none, relu, gelu, swish, clamp, sigmoid)
- **Bias:** bias_type, bias_source, bias_vector
- **Extended ops / rocroller:** from `rocroller_gtest.yaml` and ext op tests
- **Auxiliary API:** all `aux_*` tests

### 2.7 NaN / Inf tests

- **NaN:** One explicit test in the main GTest suite: **alpha_beta_zero_NaN** (matmul_gtest.yaml, pre_checkin). It uses `alpha: [ .NaN, 2 ]` and `beta: [ .NaN, 2 ]` (2×2 = 4 alpha/beta combinations) with `precision: *real_precisions` (3 precisions) and fixed M=256, N=128, K=64, trans N/N. The YAML comment: "Tests confirm no NaN propagation when alpha = 0, 2 and beta = 0. Value .NaN is converted into zero." So the test checks that when alpha or beta are NaN (or 2), the implementation does not propagate NaN inappropriately (e.g. converts to zero where required). No other YAML test in clients/tests/data uses `.NaN` or similar for alpha/beta or input data.
- **Inf (infinity):** No dedicated GTest in the YAML suite that sets alpha, beta, or input tensors to infinity. TensileLite's ReferenceValidator has a `BoundsCheckMode::NaN` and Reference code handles infinity in comparisons; those are internal validation paths, not a separate "Inf test" in the hipblaslt-test regression list.

### 2.8 How test data (numbers) are filled — and gaps

**How numbers are filled.** Matrix and vector data are filled according to the **initialization** argument (default when omitted: **hpl**). The implementation is in `clients/common/src/hipblaslt_init_device.cpp`. The RNG is **deterministic**: values are a function of element index (and batch), not of run time, so the same test produces the same inputs every run.

| Mode | Description (from code) |
|------|-------------------------|
| **rand_int** | Small integers: A/C 1–10 (float/half) or 1–3 (i8); B same with alternating sign pattern (i^j). Half/bf16: -2..2. |
| **trig_float** | Deterministic trig: A/C = sin(i + j*M + b*M*N), B = cos(...). Non-zero mantissa/exponent; not for i8/i32. |
| **hpl** (default) | HPL-style: pseudo-random in **[-0.5, 0.5]** (double mapped to T). i8: [-128,127] then rounded. |
| **special** | A = 65280 (half), B = small half (~6e-5), C = 1..10 (per-element pseudo-random). Stress case. |
| **zero** | All zeros. |
| **norm_dist** | Box-Muller normal (Xorwow state, seed from host random_device + idx). |
| **uniform_01** | Pseudo-random in **[0, 1]**. |

So the suite is **not** "uniform random" in the sense of different values each run; it is **deterministic pseudo-random** (index-based). Most tests never set `initialization` in YAML, so they use **hpl** ([-0.5, 0.5]). The only explicit overrides in the test YAML are: **hpl** and **trig_float** on a few matmul_mx_* / matmul_one_bf16_* tests; **rand_int** only in the default block for the **hipblaslt-bench** nightly entry in hipblaslt_common.yaml.

**Gaps (initialization and data quality).**

- **Zero matrices:** No matmul GTest sets `initialization: zero`. So there is no dedicated test for "all-zero A or B or C" in the main regression YAML (only alpha=0 or beta=0 in aux tests and in alpha/beta combinations).
- **Special / norm_dist / uniform_01:** None of these are used in any matmul or smoke YAML. So **special** (large half + tiny half + small ints), **norm_dist** (normal distribution), and **uniform_01** ([0,1]) are implemented but not exercised by the GTest YAML suite.
- **Ill-posed / order-of-summation–sensitive inputs:** A classic numerical stress case is **one large value and many small values** — the sum can differ depending on the order of summation (floating point is not associative). GEMM does not guarantee order of summation, so implementations may legitimately differ; the point of testing such inputs would be to check that the library doesn’t overflow or produce NaNs and to document acceptable tolerance. The **special** initialization mode is exactly this: A = 65280 (half, large), B ≈ 6e-5 (half, tiny), C = 1..10. So each inner-product term is large×tiny (order ~1), and the full dot product is a sum over K terms whose result can be sensitive to order. **special** is implemented in `hipblaslt_init_device.cpp` but **no matmul GTest in the YAML uses it**. Adding a test (e.g. a small matmul with `initialization: special`) could be useful to exercise this regime and to decide whether to use relaxed tolerance or norm_check instead of strict unit_check.
- **Denormals / subnormals:** No test explicitly fills with denormal values.
- **Inf:** Already noted in §2.7 — no Inf in inputs.
- **Runtime random for scaling:** Some tests (e.g. scaleC/scaleD) have comments that "will use random value from 0.0 to 1.0" at runtime; that is separate from the initialization enum and may be used for scaling factors in a subset of tests.

**Summary.** The numbers used to fill matrices are mostly **deterministic pseudo-random in [-0.5, 0.5]** (hpl). A few tests use **trig_float** (sin/cos) or **rand_int**; **zero**, **special**, **norm_dist**, and **uniform_01** are available in code but not used in the GTest YAML, and there are no dedicated denormal or Inf input tests.

---

## 3. How to Get Counts, Timing, and “What Runs When”

### 3.1 Test count and list

- **Option A (recommended):** Run the built `hipblaslt-test` with `--gtest_list_tests`. One line per test; parse to count and to infer category from name (e.g. `*smoke*`, `*quick*`, `*pre_checkin*`, `*nightly*` by matching test name to YAML category conventions). Same binary knows the exact expanded set.
- **Option B:** Add a small mode to `hipblaslt_gentest.py` (e.g. `--list` or `--count`) that runs the same expansion and outputs test names or a summary instead of writing binary data. That would require emitting a readable representation of each record (name + category + key params) so you can count and bin without running the C++ binary.
- **Option C:** Parse YAML and replicate the generator’s expansion logic in a script. Possible but fragile (must mirror `generate`/`instantiate` and defaults).

For **TensileLite**, use the test runner (e.g. ctest or `tensilelite-tests --gtest_list_tests`) and/or pytest for Python to get lists and counts.

### 3.2 Duration

- Run `hipblaslt-test` with `--gtest_output=xml` (and optionally `--gtest_print_time=1`); parse the XML for per-test timing. Then bin by the same categories (by test name pattern or by re-running with `--gtest_filter=*smoke*`, `*quick*`, etc.) to get “time per category” or “time per tier”.
- For **TensileLite**, ctest or GTest XML can provide timing.

### 3.3 How often they run

- **Source of truth:** CI/workflow config. See **§1.4 TheRock CI** for rocm-libraries/.github. Summary: presubmit → smoke (1 shard); nightly or triggers → full (6 shards Linux); Windows gfx1151 → quick (1 shard); rtest.py -e extended → pre_checkin+nightly (local only). Original note: which jobs run which `--gtest_filter` or `rtest.py -e smoke|regression|extended`) and any local conventions. Today:
  - `rtest.py -e smoke` → smoke
  - `rtest.py -e regression` → quick
  - `rtest.py -e extended` → pre_checkin + nightly
- So the report can state: “smoke runs in emulation/smoke job”, “quick in short regression”, “pre_checkin and nightly in extended”.

### 3.4 Example: observed from TheRock CI (rocm-libraries PR 5128)

Using `gh pr view 5128 --repo ROCm/rocm-libraries --json statusCheckRollup` and job logs (`gh api repos/ROCm/rocm-libraries/actions/jobs/<job_id>/logs`), the following was observed for a **full** run (test_type=full, 6 shards on Linux gfx94X-dcgpu):

- **Total tests in suite (all shards):** ~64,045 (shard 1: 10,675; shards 2–6: 10,674 each).
- **Test suites (from shard 1):** 8 suites — `_/matmul_test` (~10,423 tests), `_/aux_test` (6), `AllCombinations/MatrixTransformTest` (240), `ExtOpTest` (2), plus others; matmul dominates.
- **Runtime per shard:** ~7–9 min (e.g. 435 s, 530 s, 421 s, 453 s, 403 s, 418 s). Shards run in parallel, so wall-clock for the test phase is the slowest shard (~9 min).
- **Command:** `./build/bin/hipblaslt-test` with no filter; env `GTEST_SHARD_INDEX` / `GTEST_TOTAL_SHARDS` set by TheRock test script.

So for the inventory report you can cite: **~64k gtest tests**, **8 suites**, **matmul_test ~10.4k per shard**, **full run on Linux (6 shards) ~9 min wall-clock**. PR 5128 also had Math CI checks (precheckin, preliminary, static-analysis, codecov) all passing; TheRock and Math CI both ran for this PR.

---

## 4. Suggested Report Bins and Presenting Coverage

### 4.1 Presenting the scope: at-a-glance table and cross-tabs

With many dimensions (tier, operation, precision, size, trans, init, arch, …), a single 2D table can’t show every combination. These formats keep the scope visible without drowning in cells.

**How to read:** “In scope” = at least one test in the suite uses this value or family. “Gaps” = values that exist in code or YAML but no (or almost no) tests use them. The full suite is a **Cartesian expansion** over the YAML; not every (precision × size × trans × …) combination exists — many slices are sparse.

---

#### Scope-at-a-glance (one table, one row per dimension)

| Dimension | In scope (covered) | Gaps / not in scope |
|-----------|---------------------|----------------------|
| **Tier** | smoke, quick, pre_checkin, nightly, HMM, known_bug | — |
| **Operation** | matmul, matmul_bad_arg, matmul_* (bias/activation/gradient/grouped/…), aux_*, rocroller | — |
| **Precision** | real (f16/bf16/f32), f64 (gemm_only), 1b (f8/bf8/fnuz), 2b, integer i8, xf32, mixed, swizzle, MX | (all used in some test) |
| **Size** | one, small, medium, smoke, chunk, grid_limit, deepbench, resnet50, inception4, ctest | — |
| **Transposition** | N/N, N/T, T/N, T/T; conjugate T/C, C/T | — |
| **Alpha/beta** | (5,0), (0,3), (1,3), (1,1); (1,0), (1,1); fixed 1 and [0,2]; .NaN in one test | — |
| **Initialization** | hpl (default), trig_float, rand_int | zero, special, norm_dist, uniform_01 (none in matmul YAML) |
| **Architecture** | all (no gpu_arch), gfx942, gfx90a, gfx950, gfx12 (1200/1201) | (varies by test) |
| **Batch** | 1, >1 (e.g. 10, grouped_gemm counts) | — |
| **Activation** | none, relu, gelu, sigmoid, swish, clamp | — |
| **Bias** | 0/1, type (default, f32_r) | — |
| **Edge / data quality** | alpha/beta NaN (one test) | Inf, denormals, zero matrices, special (large+small) |

Use this as the **first** page of the coverage report: one glance shows what’s in scope and what’s missing.

---

#### Cross-tab: Tier × Source (which tiers get which test definitions)

| Tier | matmul_gtest | auxiliary_gtest | smoke_gtest | rocroller_gtest |
|------|--------------|-----------------|-------------|-----------------|
| **smoke** | — | — | ✓ | — |
| **quick** | ✓ | — | — | — |
| **pre_checkin** | ✓ | ✓ | — | ✓ |
| **nightly** | ✓ | — | — | — |
| **HMM** | ✓ | — | — | — |

So: smoke = smoke_gtest only; quick = matmul only; pre_checkin = matmul + aux + rocroller; nightly = matmul only; HMM = matmul only.

---

#### Cross-tab: Operation group × Precision (which ops run with which precision families)

| Operation group | real (f16/bf16/f32) | f64 | 1b (f8/bf8) | 1b fnuz | int i8 | xf32 | mixed / swizzle |
|-----------------|---------------------|-----|-------------|---------|--------|------|------------------|
| matmul core/sizes | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| matmul bias/activation | ✓ | — | ✓ | ✓ | — | — | — |
| matmul grouped/ext | ✓ | — | ✓ | ✓ | — | ✓ | — |
| aux_* | ✓ (hpa_half) | — | — | — | — | — | — |
| rocroller | ✓ (predicate set) | — | — | — | — | — | — |

Use “✓” when at least one test in that group uses that precision family; “—” when none. (Exact counts can replace ✓ if you have them.)

---

#### Optional: Size band × Tier (which sizes run in which tier)

| Size band | smoke | quick | pre_checkin | nightly |
|-----------|-------|-------|-------------|---------|
| one / edge | — | ✓ | — | — |
| small | ✓ | ✓ | ✓ | — |
| medium | — | — | ✓ | — |
| chunk / large | — | — | ✓ | — |
| grid_limit / stress | — | — | — | ✓ |
| deepbench / resnet / ctest | — | — | — | ✓ |

Helps answer “where do the big/stress sizes run?”

---



#### Time cross-tab: Operation group × Precision (seconds spent)

Same layout as the operation × precision scope table, but **cells = total time (seconds)** spent running tests in that (operation group, precision) bucket. Answers "where does the suite spend its time?"

**How to get the data:**

1. Run the full suite (or a shard) with timing and XML output:
   ```bash
   ./hipblaslt-test --gtest_output=xml:results.xml --gtest_print_time=1
   ```
2. Parse `results.xml`: each `<testcase name="..." time="...">` gives test name and duration in seconds.
3. Map each test name to **(operation group, precision)**:
   - Operation group: from the test name prefix / suite (e.g. `matmul_smoke`, `matmul_bias_relu`, `matmul_groupedgemm`, `aux_handle`, …). Use a small lookup or regex (e.g. name contains `bias` → matmul bias/activation; contains `aux_` → aux_*).
   - Precision: from the test name suffix (the GTest name often includes datatype strings like `f32_r`, `f16_r`, `f8_fnuz`, `i8`). Map to families: real, f64, 1b, 1b fnuz, int i8, xf32, mixed.
4. Aggregate: sum `time` over all tests in each (operation group, precision) cell.

**Example format (numbers TBD from a real timed run):**

| Operation group        | real | f64 | 1b | 1b fnuz | int i8 | xf32 | mixed | **Total (s)** |
|-------------------------|------|-----|-----|---------|--------|------|-------|----------------|
| matmul core/sizes       | 850  | 120 | 200 | 180     | 90     | 45   | 80    | **1565**       |
| matmul bias/activation  | 420  | —   | 95  | 110     | —      | —    | —     | **625**        |
| matmul grouped/ext      | 180  | —   | 50  | 60      | —      | 25   | —     | **315**        |
| aux_*                   | 12   | —   | —   | —       | —      | —    | —     | **12**         |
| rocroller               | 8    | —   | —   | —       | —      | —    | —     | **8**          |
| **Total (s)**           | 1470 | 120 | 345 | 350     | 90     | 70   | 80    | **~2525**      |

Use "—" when no tests in that cell. Totals (row and column) show which operation groups or precisions dominate runtime. You can use **minutes** (e.g. 14.2) if preferred, or show both count and time (e.g. `120 (45s)`).

**Variants:** Same idea for **Tier × time** (how long does smoke vs quick vs pre_checkin take?) or **Size band × time** (time on small vs large vs stress?). Method: timed run → parse XML → map name to dimensions → sum time per cell.


**Summary.** Lead with the **scope-at-a-glance** table (one row per dimension, “in scope” vs “gaps”). Add one or two **2D cross-tabs** for the most important pairs (tier × source, operation × precision, or size × tier). Use ✓/— or small counts; keep each table small enough to fit on a screen. Add a **time cross-tab** (operation × precision or tier × time) to show where the suite spends its time. For deeper detail, link to the full test-name lists (§1.6) and parameter tables (§1.6 parameter space).

---

### 4.2 Report bins (for counts and timing)

Proposed high-level bins for the main `hipblaslt-test` suite:

| Dimension | Bins (examples) |
|-----------|-------------------|
| **Tier / frequency** | smoke, quick, pre_checkin, nightly, HMM, known_bug |
| **Operation** | matmul, matmul_bad_arg, matmul_* (bias/activation/gradient/…), aux_*, rocroller_* |
| **Data type** | real (f16/bf16/f32/f64), 1-byte (f8/bf8/fp8_fnuz/…), mixed, integer i8, complex (if any) |
| **Size band** | tiny (e.g. 1×1×1, small ranges), small, medium, large, stress/grid_limit |
| **Architecture** | all, gfx942, gfx950, gfx90a, gfx1200/1201, other (from gpu_arch in YAML) |
| **Batch** | batch_count=1 vs batch_count>1 |
| **Feature** | plain gemm, bias, activation (relu/gelu/swish/clamp/sigmoid), gradient, extended/rocroller |

You can then build:

- **Counts:** e.g. “N tests in category smoke”, “M tests with f32”, “K tests with batch_count>1”.
- **Timing:** e.g. “Total time for *quick*”, “Average time per test for *nightly*”.
- **Coverage:** e.g. “Which (operation, precision, size) combinations have at least one test in quick vs nightly?”

---

## 5. Practical Next Steps

1. **Count and list hipblaslt-test:**  
   Build hipblaslt, run:
   ```bash
   ./hipblaslt-test --gtest_list_tests
   ```
   Parse output: total count; then filter by substrings (e.g. `smoke`, `quick`, `pre_checkin`, `nightly`) to approximate tier counts (note: test *name* includes `arg.name`; category is in the data file but not necessarily in the name — so either ensure name includes category or add a small `--gtest_list_tests` that also prints category per test).

2. **Check category in test names:**  
   In the current code, the test name is built from `arg.name` + `name_suffix(arg)` (datatypes, activation, bias, etc.). The YAML `name` (e.g. `matmul_smoke`, `matmul_quick`) often embeds the category; if not, we may need to add category to the printed name or to a separate listing mode so the report can bin by tier without re-running with filters.

3. **Timing run:**  
   Run with `--gtest_output=xml:results.xml` and parse `results.xml` for `<testcase>` duration and name; aggregate by tier/operation/type/size.

4. **TensileLite:**  
   Run `tensilelite-tests --gtest_list_tests` and ctest for CPU GEMM tests; run Python tests with pytest (if available) to get list and duration. Report as a separate section (“TensileLite / library tests”) with their own categories (e.g. DataTypes, ContractionSelection, RangeLibrary, CPU GEMM, Python unit).

5. **CI:**  
   Inspect rocm-libraries (and TheRock) CI for where `hipblaslt-test` or `rtest.py` is invoked and document “runs in: smoke job / regression job / nightly” in the report.

6. **Document in report:**  
   One summary table: tier × count × (optional) total time; then sections or tables by operation, data type, size band, arch, and “how often” (which job runs which filter).

---

## 6. Files Reference

| Purpose | Path |
|--------|------|
| Test data generator | `clients/tests/hipblaslt_gentest.py` |
| Main YAML entry | `clients/tests/data/hipblaslt_gtest.yaml` |
| Matmul tests | `clients/tests/data/matmul_gtest.yaml`, `matmul_common.yaml` |
| Auxiliary tests | `clients/tests/data/auxiliary_gtest.yaml` |
| Smoke tests | `clients/tests/data/smoke_gtest.yaml` |
| Rocroller tests | `clients/tests/data/rocroller_gtest.yaml` |
| Datatypes & defaults | `clients/tests/data/hipblaslt_common.yaml` |
| Test binary source | `clients/tests/src/hipblaslt_test.cpp`, `matmul_gtest.cpp`, `auxiliary_gtest.cpp`, `matrix_transform_gtest.cpp`, `hipblaslt_gtest_ext_op.cpp` |
| Run script | `rtest.py` |
| TensileLite C++ tests | `tensilelite/tests/` |
| TensileLite Python tests | `tensilelite/Tensile/Tests/`, `tensilelite/rocisa/test/` |
| TheRock CI (rocm-libraries) | `.github/workflows/therock-ci.yml`, `therock-ci-linux.yml`, `therock-ci-nightly.yml`, `therock-test-packages.yml`, `therock-test-component.yml`; `.github/scripts/therock_configure_ci.py` |
| TheRock test runner (hipblaslt) | TheRock `build_tools/github_actions/fetch_test_configurations.py` (hipblaslt entry), `test_executable_scripts/test_hipblaslt.py` |
| Math CI (Jenkins, rocjenkins) | rocjenkins repo: `src/com/amd/project/hipblaslt/*.groovy` (common, precheckin, preliminary, nightly, performance, static_build, codeql, codecov, etc.), `vars/onDemandHipblaslt.groovy`, `vars/statusGate.groovy`; `resources/com/amd/monorepo/projects.json` (hipblaslt entry, gatingJobs) |

---

*Summary: The main regression is one GTest binary driven by YAML-expanded binary data. Categories for the report are tier (smoke/quick/pre_checkin/nightly), operation, data type, size band, architecture, and feature. Counts and timing are best obtained from the test binary (list_tests + XML output); run frequency from CI and rtest.py. TensileLite tests are a separate set to inventory similarly.*
