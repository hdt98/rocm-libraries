# Bug Injection Plan for hipBLASLt

I have created 33 bug patches across 5 categories (A, B, C, D, E) for mutation testing experiments.
The patches are located in `test_symposium_experiments/patches/`.

## Categories

### Category A: Simple Bugs (7)

Basic validation and parameter errors.

| # | Patch File | Description |
|---|------------|-------------|
| 1 | `cat_a_01_hipblaslt_swap_rows_cols.patch` | Swaps rows and cols in `hipblasLtMatrixLayoutCreate`. Should segfault. |
| 2 | `cat_a_02_rocblaslt_mat_remove_type_check.patch` | Removes datatype check for C/D matrices. |
| 3 | `cat_a_03_rocblaslt_transform_swap_lda_ldb.patch` | Swaps ldA and ldB in transform kernel launch. |
| 4 | `cat_a_04_hipblaslt_ext_stride_error.patch` | Calculates strideA using N instead of K. |
| 5 | `cat_a_05_utility_wrong_compute_string.patch` | Returns wrong string for `COMPUTE_32XF`. |
| 6 | `cat_a_06_matrix_layout_ld_bypass.patch` | **NEW** Swaps rows/cols in internal matrix layout storage. |
| 7 | `cat_a_07_tensile_problem_size_swap.patch` | **NEW** Swaps M and N dimensions for C tensor in Tensile problem setup. |

### Category B: Broad Complex Bugs (7)

Logic errors in core components.

| # | Patch File | Description |
|---|------------|-------------|
| 1 | `cat_b_01_tensile_host_memset_off_by_one.patch` | Off-by-one error in memset for heuristic results. |
| 2 | `cat_b_02_rocroller_gemm_trans_mixup.patch` | Uses transA instead of transB for Tensor B. |
| 3 | `cat_b_03_handle_wavefront_size_error.patch` | Incorrect wavefront size calculation (+1). |
| 4 | `cat_b_04_rocblaslt_auxiliary_attr_mixup.patch` | Copies to `op_B` instead of `op_A` when setting `TRANSA`. |
| 5 | `cat_b_05_user_driven_tuning_parser_swap_mn.patch` | Swaps M and N during tuning file parsing. |
| 6 | `cat_b_06_heuristic_algo_count_error.patch` | **NEW** Off-by-one in solution index during heuristic conversion. **Heuristic check failure.** |
| 7 | `cat_b_07_activation_function_error.patch` | **NEW** Rejects RELU epilogue as unsupported. |

### Category C: Deep Advanced Bugs (7)

Subtle logic errors in specific paths.

| # | Patch File | Description |
|---|------------|-------------|
| 1 | `cat_c_01_solution_selection_logic_flip.patch` | Flips condition for analytical type selection. |
| 2 | `cat_c_02_runtime_args_selection_type_mixup.patch` | Uses typeB instead of typeA for element size. |
| 3 | `cat_c_03_hipblaslt_ext_op_softmax_dim_check.patch` | Rejects valid `dim=1` for softmax. |
| 4 | `cat_c_04_rocroller_host_algo_count_logic.patch` | Incorrect check for `returnAlgoCount`. |
| 5 | `cat_c_05_status_swallow_error.patch` | Swallows `hipErrorInvalidDevicePointer` in status conversion. |
| 6 | `cat_c_06_workspace_size_underflow.patch` | **NEW** Reports 25% smaller workspace than required, causing buffer overflow for GSU kernels. |
| 7 | `cat_c_07_alpha_beta_type_confusion.patch` | **NEW** Swaps alpha and beta values, causing D = beta*A*B + alpha*C. |

### Category D: Legendary Bugs (7)

Nuanced bugs with potential for catastrophic/silent failure.

| # | Patch File | Description |
|---|------------|-------------|
| 1 | `cat_d_01_parameter_selection_swizzle_scale.patch` | Incorrectly enables swizzle scale for smaller tiles. |
| 2 | `cat_d_02_rocblaslt_mat_utils_missing_check.patch` | Removes validity check for `HIP_R_16BF` order. |
| 3 | `cat_d_03_tuple_helper_off_by_one.patch` | Off-by-one error in tuple iteration (skips last pair). |
| 4 | `cat_d_04_hipblaslt_ostream_abort_race.patch` | Removes worker stop call during abort, causing potential race. |
| 5 | `cat_d_05_auxiliary_f8_type_mixup.patch` | Returns wrong FP8 type enum from string. |
| 6 | `cat_d_06_batch_stride_corruption.patch` | **NEW** Off-by-one in batch stride for C matrix when batch_count > 1. |
| 7 | `cat_d_07_synchronizer_race_condition.patch` | **NEW** Nullifies synchronizer for grouped batched GEMMs, causing race conditions. |

### Category E: Numerical Precision Bugs (5) - **NEW CATEGORY**

Bugs that induce numerical errors of increasing nuance. These are particularly insidious as they produce "wrong but plausible" results.

| Level | Patch File | Description |
|-------|------------|-------------|
| 1 - Obvious | `cat_e_01_layernorm_epsilon_precision.patch` | Reduces LayerNorm epsilon by 10x, causing numerical instability with small variances. |
| 2 - Moderate | `cat_e_02_scale_factor_truncation.patch` | Swaps scaleB and scaleC pointers, causing incorrect scaling in quantized workflows. |
| 3 - Subtle | `cat_e_03_hpa_precision_loss.patch` | Disables High Precision Accumulate unconditionally, causing precision loss in FP16/BF16 GEMM. |
| 4 - Advanced | `cat_e_04_compute_input_type_mismatch.patch` | Swaps A/B types in mixed-precision compute input type determination. |
| 5 - Expert | `cat_e_05_xf32_math_op_override.patch` | Inverts XFloat32/Float math op logic, using wrong precision for TF32 computations. |

---

## Patch Summary Table

| Category | Count | Primary Files Modified | Detection Difficulty |
|----------|-------|----------------------|---------------------|
| A - Simple | 7 | `hipblaslt.cpp`, `rocblaslt_auxiliary.cpp`, `tensile_host.cpp` | Easy |
| B - Complex | 7 | `tensile_host.cpp`, `rocroller_host.cpp`, `handle.cpp` | Medium |
| C - Advanced | 7 | `tensile_host.cpp`, `hipblaslt-ext-op.cpp`, `status.cpp` | Hard |
| D - Legendary | 7 | `tensile_host.cpp`, `rocblaslt_mat_utils.cpp`, `hipblaslt_ostream.cpp` | Very Hard |
| E - Numerical | 5 | `tensile_host.cpp`, `hipblaslt-ext-op.cpp` | Easy to Expert |

**Total: 33 patches**

---

## Category E Deep Dive: Numerical Precision Errors

Category E patches are specifically designed to test detection of numerical precision issues. These are ordered by increasing difficulty of detection:

### Level 1: Obvious (Patch 23)
- **LayerNorm Epsilon**: Reducing epsilon by 10x causes division by near-zero when variance is small
- **Detection**: NaN/Inf outputs, large relative errors
- **Test coverage needed**: Edge cases with small variance inputs

### Level 2: Moderate (Patch 24)
- **Scale Factor Swap**: Wrong scales applied to matrices B and C
- **Detection**: Requires tests with different scaleB and scaleC values
- **Test coverage needed**: Quantization workflows with non-trivial scaling

### Level 3: Subtle (Patch 25)
- **HPA Disabled**: FP16/BF16 accumulates in low precision instead of FP32
- **Detection**: Accumulated error grows with matrix size
- **Test coverage needed**: Large matrices with numerical verification

### Level 4: Advanced (Patch 26)
- **Type Mismatch**: Only affects mixed-precision scenarios (e.g., FP8×FP16)
- **Detection**: Requires mixed-precision tests with careful result validation
- **Test coverage needed**: Comprehensive mixed-precision matrix combinations

### Level 5: Expert (Patch 27)
- **XF32 Inversion**: Uses TF32 (19-bit mantissa) when FP32 (23-bit) expected
- **Detection**: 4-bit mantissa difference, within typical tolerance for many tests
- **Test coverage needed**: High-precision numerical tests that verify >19-bit accuracy

---

## `tasks.py`

This Python script uses [Invoke](https://www.pyinvoke.org/) to organize and automate the experiments defined in `tasks.py`. It defines the following tasks:

```bash
# Install
pip3 install invoke

# List experiments and other tasks
invoke --list

# See options for a task
invoke --help <task>
```

- **Build the Project:**
  - `invoke build`
- **Patch Management:**
  - Apply a patch: `invoke apply-patch --patch-name=...`
  - Revert all changes: `invoke revert-all`
- **Experiment A – Minimization:**  
  - `invoke experiment-a`
    - Runs all patches against the full test suite.
    - Generates `experiment_a_results.json` (detection vectors).
    - Performs Set Cover Analysis to find the minimal effective test subset.  
    - Saves this subset to `experiment_a_minimal_set.json`.
- **Experiment B – F/T Ratio:**  
  - `invoke experiment-b`
    - Runs a broad patch across various partitions (e.g., f32, f16, NN, TN, etc.).
    - Results are saved to `experiment_b_results.json`.
- **Experiment C – Detection Time:**  
  - `invoke experiment-c`
    - Runs test orders: Default and Randomized.
    - Simulates Prioritized ordering by running "quick" tests first.
    - Results stored in `experiment_c_results.json`.
- **Experiment D – Fractional Testing:**  
  - `invoke experiment-d`
    - Runs tests on shards (100%, 50%, 25%, ... down to 1%) to measure detection.
    - Results saved to `experiment_d_results.json`.

---

## 2. `setup_and_run.sh`

A shell script that automates the experiment workflow:

- Creates a Python virtual environment in `.venv`.
- Installs `invoke`.
- Runs all four experiments in sequence.

**Usage**

To run all experiments in one go:
```bash
./test_symposium_experiments/setup_and_run.sh
```

Or, to run experiments manually:
```bash
cd test_symposium_experiments
source .venv/bin/activate
invoke experiment-a
```

---

> [!NOTE]
> If adding new patches, `git apply` can be too strict and may fail for some patches due to context mismatches. The numbering scheme has patches labeled by category (e.g., `cat_Y_XX_description.patch`) where `Y` is the category letter and `XX` is a sequential number within that category.
