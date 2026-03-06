# hipBLASLt test coverage: Operation group × Precision

**Scope:** Main GTest suite (`hipblaslt-test`). Each cell lists **size-group letters** for array sizes (M×N×K) exercised at that (operation, precision) intersection. Three views: **full** (no filter), **quick** (`*quick*`), and **smoke** (`*smoke*`). The main tables cover the three **matmul** operation groups (core/sizes, bias/activation, grouped/ext); **aux_*** and **rocroller** are in separate tables per config, since they exercise API/descriptor and rocroller-integration paths rather than the same op × precision × size grid. **Transposition:** Nearly all matmul tests use `transA_transB_range` (N/N, N/T, T/N, T/T); only a few use a single fixed trans (e.g. workload tests, alpha_beta_zero_NaN, rocroller).

**—** = no tests at that intersection. **·** = N/A (no GEMM array sizes; API-only or fixed config). **Total tests** = number of test instances in that row (all precisions/sizes in that sub-item combined). Computed from the test generator (no C++ binary needed); see *How to get test counts* below.

---

## 1. Full (no filter)

All tests from matmul_gtest, auxiliary_gtest, smoke_gtest, rocroller_gtest. Used by TheRock CI `test_type=full` and Math CI precheckin. Total suite size is on the order of tens of thousands of tests (~64k in typical CI).

| Operation group        | Sub-item                    | real (f16/bf16/f32) | f64 | 1b (f8/bf8) | 1b fnuz | int i8 | xf32 | mixed / swizzle | Total tests |
|------------------------|-----------------------------|--------------------|-----|-------------|---------|--------|------|------------------|-------------|
| **matmul core/sizes**  | core / bad-arg / NaN        | O,S,M,C            | O,S,M,C | O,S,M,C   | O,S,M,C | O,S,M,C | O,S,M,C | O,S,M,C          | 15,843      |
|                        | sizes / fixed shapes         | F                  | F   | F           | F       | F      | F    | F                | 3,057       |
|                        | algo / heuristic / MX       | S,M,F              | S,M,F | S,M,F     | S,M,F   | S,M,F  | S,M,F | S,M,F            | 1,499       |
|                        | stress / nightly shapes     | G,D,W              | G,D,W | D,W        | D,W     | —      | D,W  | —                | 3,616       |
| **matmul bias/activation** | bias (relu, gelu, swish, …) | S,E            | —   | S,E         | S,E     | —      | —    | —                | 3,496       |
|                        | gradients (dgelu, drelu, bgradb) | S,E         | —   | S,E         | S,E     | —      | —    | —                | 1,664       |
|                        | bias_gelu_aux / equality    | S,E                | —   | S,E         | S,E     | —      | —    | —                | 1,920       |
| **matmul grouped/ext** | grouped gemm                | S,M,F              | —   | S,M,F       | S,M,F   | —      | S,M,F | —                | 154         |
|                        | extended API (algo, swizzle) | S,M,F             | —   | S,M,F       | S,M,F   | —      | S,M,F | —                | 6,708       |

**Auxiliary and rocroller (full):**

| Area | What runs |
|------|-----------|
| **aux_*** | All of `auxiliary_gtest.yaml` (category pre_checkin): handle init/destroy/bad-arg; mat descriptor init, set/get attr, copy; matmul descriptor, pref, alg, plan init/set/get and bad workspace size; edge cases (get_sol with null bias, zero alpha); helper coverage (auxiliary_func, float8_func, hipblaslt_ext_op, rocblaslt_utility, status_func, etc.). One test runs a real GEMM: **aux_matmul_bad_ws_size** (size S, hpa_half). |
| **rocroller** | Single test from `rocroller_gtest.yaml`: **rocroller_predicate** — M/N/K 128×128×96, trans T/N, gfx950. |

---

## 2. Quick (`*quick*`)

Only tests with `category: quick` in matmul_gtest.yaml: matmul_one, matmul_small, matmul_conj_small, matmul_8, matmul_64_8, matmul_8_64, matmul_bf16, matmul_bf16_fp32dst, matmul_one_real_precisions_1b_gfx12, matmul_one_integer_precisions_i8_gfx12, matmul_one_bf16_fp32dst_precision_gfx90a. No aux, no rocroller.

| Operation group        | Sub-item                    | real (f16/bf16/f32) | f64 | 1b (f8/bf8) | 1b fnuz | int i8 | xf32 | mixed / swizzle | Total tests |
|------------------------|-----------------------------|--------------------|-----|-------------|---------|--------|------|------------------|-------------|
| **matmul core/sizes**  | core / bad-arg / NaN        | O,S                | O,S | O           | —       | O      | —    | —                | 14,408      |
|                        | sizes / fixed shapes         | F                  | —   | —           | —       | —      | —    | —                | 144         |
|                        | algo / heuristic / MX       | —                  | —   | —           | —       | —      | —    | —                | 0           |
|                        | stress / nightly shapes     | —                  | —   | —           | —       | —      | —    | —                | 0           |
| **matmul bias/activation** | bias (relu, gelu, swish, …) | —            | —   | —           | —       | —      | —    | —                | 0           |
|                        | gradients (dgelu, drelu, bgradb) | —             | —   | —           | —       | —      | —    | —                | 0           |
|                        | bias_gelu_aux / equality    | —                | —   | —           | —       | —      | —    | —                | 0           |
| **matmul grouped/ext** | grouped gemm                | —                  | —   | —           | —       | —      | —    | —                | 0           |
|                        | extended API (algo, swizzle) | —                 | —   | —           | —       | —      | —    | —                | 0           |

**Auxiliary and rocroller (quick):**

| Area | What runs |
|------|-----------|
| **aux_*** | None. Quick filter selects only `category: quick` from matmul_gtest; auxiliary tests are pre_checkin only. |
| **rocroller** | None. Rocroller tests are pre_checkin only. |

---

## 3. Smoke (`*smoke*`)

Only tests from smoke_gtest.yaml (`category: smoke`): matmul_smoke, matmul_bias_*_smoke, matmul_f8_bf8_*_smoke, matmul_gemm_xf32/double/i8_*_smoke. Small sizes (E) or fixed 128; no aux, no rocroller.

| Operation group        | Sub-item                    | real (f16/bf16/f32) | f64 | 1b (f8/bf8) | 1b fnuz | int i8 | xf32 | mixed / swizzle | Total tests |
|------------------------|-----------------------------|--------------------|-----|-------------|---------|--------|------|------------------|-------------|
| **matmul core/sizes**  | core / bad-arg / NaN        | E                  | E   | E           | E       | E      | E    | —                | 992         |
|                        | sizes / fixed shapes         | —                  | —   | —           | —       | —      | —    | —                | 0           |
|                        | algo / heuristic / MX       | —                  | —   | —           | —       | —      | —    | —                | 0           |
|                        | stress / nightly shapes     | —                  | —   | —           | —       | —      | —    | —                | 0           |
| **matmul bias/activation** | bias (relu, gelu, swish, …) | E              | —   | —           | —       | —      | —    | —                | 552         |
|                        | gradients (dgelu, drelu, bgradb) | —             | —   | —           | —       | —      | —    | —                | 0           |
|                        | bias_gelu_aux / equality    | —                | —   | —           | —       | —      | —    | —                | 0           |
| **matmul grouped/ext** | grouped gemm                | —                  | —   | —           | —       | —      | —    | —                | 0           |
|                        | extended API (algo, swizzle) | —                 | —   | —           | —       | —      | —    | —                | 0           |

**Auxiliary and rocroller (smoke):**

| Area | What runs |
|------|-----------|
| **aux_*** | None. Smoke filter selects only tests from smoke_gtest.yaml (matmul smoke variants); no auxiliary tests. |
| **rocroller** | None. No rocroller tests in smoke. |

---

### Size-group legend

| Letter | Size group | Description (M×N×K) |
|--------|------------|---------------------|
| **O** | one | One / edge: 1×1×1 up to 1024×1×512; dims 63–65, 127–129, 255–257 (boundary) |
| **S** | small | Small square: 8×8×8 through 72×72×72 (small_matrix_size_range) |
| **M** | medium | Medium: 8×8×8, 16×16×16, 128×128×64 (medium_matrix_size_range) |
| **E** | smoke | Smoke: 8×8×8, 16×16×16, 32×32×32 (smoke_matrix_size_range) |
| **C** | chunk | Chunk: 24000×256×256 (chunk_matrix_size_range) |
| **G** | grid_limit | Grid limit: large skinny, e.g. 4096×131072×1 (grid_limit_matrix_size_real / _double) |
| **D** | deepbench | DeepBench: irregular workload shapes (deepbench_sizes) |
| **W** | workload | Workload: ResNet50, Inception4, ctest fixed shapes (resnet50_*, inception4_*, ctest_*) |
| **F** | fixed | Fixed single points: matmul_8, matmul_16, …, matmul_1024, small2, k0 (inline M,N,K) |
| **R** | rocroller | Rocroller: 128×128×96 (single predicate config) |

---

### How to get test counts

Counts are computed from the **test generator** (no C++ binary needed). The generator expands the YAML and writes binary records; with `--list` it prints one line per test: `category<tab>name`.

1. **Generator list mode** (in `hipblaslt_gentest.py`):
   ```bash
   python hipblaslt_gentest.py -t data/hipblaslt_common.yaml data/hipblaslt_gtest.yaml --list
   ```
   Output: `category\tname` (e.g. `quick\tmatmul_one`) for each expanded test.

2. **Count script** (`clients/tests/count_hipblaslt_tests_by_row.py`):
   - Run from `clients/tests`; it invokes the generator and maps names to crosstab rows.
   - **Full:** `python count_hipblaslt_tests_by_row.py`
   - **Quick:** `python count_hipblaslt_tests_by_row.py --filter quick`
   - **Smoke:** `python count_hipblaslt_tests_by_row.py --filter smoke`
   - Or pass pre-generated list: `python count_hipblaslt_tests_by_row.py --list-file /tmp/list.txt`

The script uses the same row → test-name mapping as the table above. The "Total tests" numbers in the tables were produced by this script. To refresh after YAML changes, re-run the script from `clients/tests`.

Row → test name mapping: see `ROW_PATTERNS` in `count_hipblaslt_tests_by_row.py`.

---

### Sub-item meanings

- **matmul core/sizes:** *core / bad-arg / NaN* = matmul_bad_arg, alpha_beta_zero_NaN, matmul_one, matmul_small, matmul_conj_small, matmul_medium, matmul_batch_medium, matmul_medium_HMM, matmul_chunk. *sizes / fixed shapes* = matmul_8 … matmul_1024, matmul_small2, matmul_k0. *algo / heuristic / MX* = matmul_algo, matmul_heuristic_all_solutions*, matmul_mx_*. *stress / nightly shapes* = matmul_grid_limit_*, matmul_deepbench, resnet50_*, inception4_*, ctest_*, matmul_large_nt_*, matmul_conv3d_kernels, matmul_stride_lt_ld.
- **matmul bias/activation:** *bias* = matmul_bias_relu, matmul_bias_sigmoid, matmul_bias_gelu, matmul_bias_swish, matmul_bias_clamp, matmul_bias_only, matmul_bias_type. *gradients* = matmul_dgelu_*, matmul_drelu_*, matmul_bgradb_*. *bias_gelu_aux / equality* = matmul_bias_gelu_aux_*, matmul_equality_NN_bias_*, matmul_relu_clamp_useE.
- **matmul grouped/ext:** *grouped gemm* = matmul_groupedgemm, matmul_groupedgemm_specific_sizes, matmul_groupedgemm_f8_fnuz/bf16/xf32/zero_n, matmul_extapi_groupedgemm. *extended API* = matmul_extapi_algo_method_gemm, matmul_extapi_gemm, matmul_extapi_algo_method_tuning_*, matmul_extapi_swizzleA/B, matmul_swizzleA/B.

**Auxiliary and rocroller** are not in the op × precision grid; see the "Auxiliary and rocroller" table under each config (full / quick / smoke) for what runs. Full run: all aux_* (handle, mat, matmul/pref/alg, edge, helper) and rocroller_predicate. Quick and smoke: none.

Source: `clients/tests/data/*.yaml`. See `hipblaslt_test_inventory_plan.md` for full scope and parameter space.
