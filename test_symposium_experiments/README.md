# Bug Injection Plan for hipBLASLt

I have created 140 bug patches across 5 categories (A, B, C, D, E) for mutation testing experiments.
The patches are located in `test_symposium_experiments/patches/`.

## Categories

### Category A: Simple Bugs (28)

Basic validation and parameter errors - swaps, off-by-one, wrong constants.

| # | Patch File | Description |
|---|------------|-------------|
| 1 | `cat_a_01_hipblaslt_swap_rows_cols.patch` | Swaps rows and cols in `hipblasLtMatrixLayoutCreate`. |
| 2 | `cat_a_02_rocblaslt_mat_remove_type_check.patch` | Removes datatype check for C/D matrices. |
| 3 | `cat_a_03_rocblaslt_transform_swap_lda_ldb.patch` | Swaps ldA and ldB in transform kernel launch. |
| 4 | `cat_a_04_hipblaslt_ext_stride_error.patch` | Calculates strideA using N instead of K. |
| 5 | `cat_a_05_utility_wrong_compute_string.patch` | Returns wrong string for `COMPUTE_32XF`. |
| 6 | `cat_a_06_matrix_layout_ld_bypass.patch` | Swaps rows/cols in internal matrix layout storage. |
| 7 | `cat_a_07_tensile_problem_size_swap.patch` | Swaps M and N dimensions for C tensor in Tensile problem setup. |
| 8 | `cat_a_08_tensile_host_batch_index_swap.patch` | Swaps batch indices in ContractionProblem setup. |
| 9 | `cat_a_09_tensile_host_stride_swap.patch` | Swaps row_stride_c and col_stride_c for matrix C. |
| 10 | `cat_a_10_tensile_host_m_n_swap.patch` | Swaps m and n dimensions for output matrix D. |
| 11 | `cat_a_11_hipblaslt_ext_op_m_n_swap.patch` | Swaps m and n arguments for LayerNorm operation. |
| 12 | `cat_a_12_contraction_problem_trans_swap.patch` | Swaps k and m in transposed tensor A dimensions. |
| 13 | `cat_a_13_kernel_args_offset_error.patch` | Off-by-one in padding detection for kernel arguments. |
| 14 | `cat_a_14_tensor_descriptor_stride_error.patch` | Wrong dimension check in stride padding calculation. |
| 15 | `cat_a_15_heuristic_result_size_error.patch` | Uses half size for memset in heuristic results. |
| 16 | `cat_a_16_solution_adapter_xnack_off_by_one.patch` | Off-by-one in xnack version string replacement. |
| 17 | `cat_a_17_grouped_gemm_loop_bound_error.patch` | Off-by-one in grouped GEMM problem loop (misses last). |
| 18 | `cat_a_18_solution_index_increment_error.patch` | Increments solution index by 1, causing misalignment. |
| 19 | `cat_a_19_contraction_solution_start_stride_error.patch` | Wrong start stride values for CD and AB. |
| 20 | `cat_a_20_contraction_problem_free_index_swap.patch` | Both free indices marked as not from A. |
| 21 | `cat_a_21_hip_solution_adapter_module_error.patch` | Uses find instead of rfind for path parsing. |
| 22 | `cat_a_22_tensile_host_bias_size_swap.patch` | Inverted condition for bias size dimension selection. |
| 23 | `cat_a_23_hipblaslt_matmul_desc_swap.patch` | Sets op_B instead of op_A for TRANSA attribute. |
| 24 | `cat_a_24_contraction_solution_workgroup_swap.patch` | Swaps wrong dimensions in transposeC01 handling. |
| 25 | `cat_a_25_tensile_host_workspace_offset_error.patch` | Halves workspace size causing buffer overrun. |
| 26 | `cat_a_26_contraction_problem_bound_index_swap.patch` | Swaps bound indices for transposed matrix A. |
| 27 | `cat_a_27_hipblaslt_ext_preference_workspace_error.patch` | Stores half the requested workspace size. |
| 28 | `cat_a_28_tensile_host_return_algo_count_error.patch` | Off-by-one in returned algorithm count. |

### Category B: Broad Complex Bugs (28)

Logic inversions and conditional errors in core components.

| # | Patch File | Description |
|---|------------|-------------|
| 1 | `cat_b_01_tensile_host_memset_off_by_one.patch` | Off-by-one error in memset for heuristic results. |
| 2 | `cat_b_02_rocroller_gemm_trans_mixup.patch` | Uses transA instead of transB for Tensor B. |
| 3 | `cat_b_03_handle_wavefront_size_error.patch` | Incorrect wavefront size calculation (+1). |
| 4 | `cat_b_04_rocblaslt_auxiliary_attr_mixup.patch` | Copies to `op_B` instead of `op_A` when setting `TRANSA`. |
| 5 | `cat_b_05_user_driven_tuning_parser_swap_mn.patch` | Swaps M and N during tuning file parsing. |
| 6 | `cat_b_06_heuristic_algo_count_error.patch` | Off-by-one in solution index during heuristic conversion. |
| 7 | `cat_b_07_activation_function_error.patch` | Rejects RELU epilogue as unsupported. |
| 8 | `cat_b_08_tensile_host_hpa_condition_flip.patch` | Inverts HPA condition (< instead of >). |
| 9 | `cat_b_09_tensile_host_grouped_gemm_flag_inversion.patch` | Inverts grouped GEMM user arguments condition. |
| 10 | `cat_b_10_tensile_host_scale_ab_logic_error.patch` | Inverts AND to OR in scaleAB null check. |
| 11 | `cat_b_11_tensile_host_e_enabled_output_flip.patch` | Inverts E tensor output flag logic. |
| 12 | `cat_b_12_contraction_solution_activation_check_flip.patch` | Inverts activation enabled condition. |
| 13 | `cat_b_13_tensile_host_alpha_restriction_flip.patch` | Inverts k==0 condition for alpha restriction. |
| 14 | `cat_b_14_contraction_problem_trans_b_flip.patch` | Inverts transpose B condition. |
| 15 | `cat_b_15_hip_solution_adapter_lazy_load_flip.patch` | Inverts lazy loading condition (loads when already loaded). |
| 16 | `cat_b_16_tensile_host_cequals_d_flip.patch` | Inverts C equals D condition. |
| 17 | `cat_b_17_contraction_solution_strided_batch_flip.patch` | Inverts strided batch check for input matrices. |
| 18 | `cat_b_18_tensile_host_bias_enabled_flip.patch` | Inverts bias enabled check. |
| 19 | `cat_b_19_contraction_problem_pack_batch_flip.patch` | Wrong bit mask for Y dimension batch packing. |
| 20 | `cat_b_20_tensile_host_gradient_enabled_flip.patch` | Inverts gradient enabled flag. |
| 21 | `cat_b_21_contraction_solution_use_beta_flip.patch` | Inverts useBeta check. |
| 22 | `cat_b_22_tensile_host_amaxd_condition_flip.patch` | Inverts amaxD null check. |
| 23 | `cat_b_23_contraction_problem_transpose_c01_flip.patch` | Inverts transposeC01 check. |
| 24 | `cat_b_24_tensile_host_fallback_condition_flip.patch` | Inverts xfloat32 fallback condition. |
| 25 | `cat_b_25_contraction_solution_use_e_flip.patch` | Inverts useE check. |
| 26 | `cat_b_26_tensile_host_cu_fallback_flip.patch` | Inverts CU fallback detection. |
| 27 | `cat_b_27_contraction_solution_sparse_check_flip.patch` | Inverts sparse check. |
| 28 | `cat_b_28_tensile_host_preload_condition_flip.patch` | Inverts preload check. |

### Category C: Deep Advanced Bugs (28)

Subtle logic errors in specific paths, type confusions, and edge case handling.

| # | Patch File | Description |
|---|------------|-------------|
| 1 | `cat_c_01_solution_selection_logic_flip.patch` | Flips condition for analytical type selection. |
| 2 | `cat_c_02_runtime_args_selection_type_mixup.patch` | Uses typeB instead of typeA for element size. |
| 3 | `cat_c_03_hipblaslt_ext_op_softmax_dim_check.patch` | Rejects valid `dim=1` for softmax. |
| 4 | `cat_c_04_rocroller_host_algo_count_logic.patch` | Incorrect check for `returnAlgoCount`. |
| 5 | `cat_c_05_status_swallow_error.patch` | Swallows `hipErrorInvalidDevicePointer` in status conversion. |
| 6 | `cat_c_06_workspace_size_underflow.patch` | Reports 25% smaller workspace than required. |
| 7 | `cat_c_07_alpha_beta_type_confusion.patch` | Swaps alpha and beta values. |
| 8 | `cat_c_08_tensile_host_compute_input_type_confusion.patch` | Uses b_type twice instead of a_type and b_type. |
| 9 | `cat_c_09_contraction_solution_stream_k_iter_error.patch` | Uses min instead of max for iterations (div by zero). |
| 10 | `cat_c_10_tensile_host_workspace_size_comparison.patch` | Uses > instead of < for workspace comparison. |
| 11 | `cat_c_11_contraction_problem_bound_size_index_error.patch` | Uses wrong array for bound sizes. |
| 12 | `cat_c_12_tensile_host_solution_predicate_short_circuit.patch` | Wrong short-circuit in predicate check. |
| 13 | `cat_c_13_contraction_solution_gsu_calculation_error.patch` | Uses >= instead of > for GSU check. |
| 14 | `cat_c_14_tensile_host_grouped_gemm_size_check.patch` | Inverts grouped GEMM size comparison. |
| 15 | `cat_c_15_contraction_problem_free_size_accumulation.patch` | Uses addition instead of multiplication. |
| 16 | `cat_c_16_tensile_host_activation_type_enum_error.patch` | Uses All instead of Hipblaslt_all enum. |
| 17 | `cat_c_17_contraction_solution_magic_number_shift.patch` | Wrong magic number base (3 instead of 2). |
| 18 | `cat_c_18_tensile_host_f32_math_op_condition.patch` | Inverted f32 math op condition. |
| 19 | `cat_c_19_contraction_problem_beta_restriction_enum.patch` | Hardcoded 1.0 instead of beta value. |
| 20 | `cat_c_20_tensile_host_device_count_boundary.patch` | Off-by-one in device count. |
| 21 | `cat_c_21_contraction_solution_reduction_type_error.patch` | Multiplication instead of division in split calc. |
| 22 | `cat_c_22_tensile_host_algo_data_cast_error.patch` | Casts to short* instead of int*. |
| 23 | `cat_c_23_contraction_problem_dims_accumulation.patch` | Uses min instead of max for dimensions. |
| 24 | `cat_c_24_tensile_host_epilogue_default_check.patch` | Inverts epilogue default check. |
| 25 | `cat_c_25_contraction_solution_sk_tiles_cap.patch` | Uses max instead of min for tiles cap. |
| 26 | `cat_c_26_tensile_host_library_null_check.patch` | Changed null check to truthy check. |
| 27 | `cat_c_27_contraction_problem_eligibility_check.patch` | Inverts PK eligibility check. |
| 28 | `cat_c_28_tensile_host_duplicated_solution_check.patch` | Uses != instead of == for duplicate detection. |

### Category D: Legendary Bugs (28)

Nuanced bugs with potential for catastrophic/silent failure - race conditions, memory issues.

| # | Patch File | Description |
|---|------------|-------------|
| 1 | `cat_d_01_parameter_selection_swizzle_scale.patch` | Incorrectly enables swizzle scale for smaller tiles. |
| 2 | `cat_d_02_rocblaslt_mat_utils_missing_check.patch` | Removes validity check for `HIP_R_16BF` order. |
| 3 | `cat_d_03_tuple_helper_off_by_one.patch` | Off-by-one error in tuple iteration. |
| 4 | `cat_d_04_hipblaslt_ostream_abort_race.patch` | Removes worker stop call causing race. |
| 5 | `cat_d_05_auxiliary_f8_type_mixup.patch` | Returns wrong FP8 type enum from string. |
| 6 | `cat_d_06_batch_stride_corruption.patch` | Off-by-one in batch stride for C matrix. |
| 7 | `cat_d_07_synchronizer_race_condition.patch` | Nullifies synchronizer for grouped GEMMs. |
| 8 | `cat_d_08_tensile_host_mutex_removal.patch` | Removes mutex lock in adapter initialization. |
| 9 | `cat_d_09_hip_solution_adapter_lock_removal.patch` | Removes lock guard in kernel map access. |
| 10 | `cat_d_10_tensile_host_memory_order_relaxed.patch` | Uses relaxed memory order for adapter load. |
| 11 | `cat_d_11_contraction_solution_silent_overflow.patch` | Silent 32-bit overflow in workgroup calculation. |
| 12 | `cat_d_12_tensile_host_ptr_deref_before_null_check.patch` | Dereferences data before null check. |
| 13 | `cat_d_13_contraction_problem_assert_removal.patch` | Removes assertion for null/zero CU count. |
| 14 | `cat_d_14_tensile_host_atomic_store_relaxed.patch` | Uses relaxed store for adapter update. |
| 15 | `cat_d_15_hip_solution_adapter_error_swallow.patch` | Swallows non-NotFound errors silently. |
| 16 | `cat_d_16_tensile_host_exception_swallow.patch` | Returns success on exception. |
| 17 | `cat_d_17_contraction_solution_uninitialized_var.patch` | Uninitialized workgroup variables. |
| 18 | `cat_d_18_tensile_host_shared_ptr_leak.patch` | Memory leak in shared_ptr deleter. |
| 19 | `cat_d_19_contraction_problem_data_race_sizes.patch` | Not clearing vectors before repopulating. |
| 20 | `cat_d_20_tensile_host_double_free_risk.patch` | Double delete of adapter. |
| 21 | `cat_d_21_hip_solution_adapter_use_after_free.patch` | Use after free in module unload. |
| 22 | `cat_d_22_tensile_host_buffer_overread.patch` | Buffer overread in heuristic array. |
| 23 | `cat_d_23_contraction_solution_signed_unsigned_mismatch.patch` | Signed/unsigned mismatch with negative start. |
| 24 | `cat_d_24_tensile_host_stack_buffer_overflow.patch` | Invalid device ID causing overflow. |
| 25 | `cat_d_25_contraction_problem_iterator_invalidation.patch` | Modifying vector while iterating. |
| 26 | `cat_d_26_tensile_host_null_deref_in_logging.patch` | Null dereference in logging path. |
| 27 | `cat_d_27_hip_solution_adapter_module_leak.patch` | Clears modules after insert causing leak. |
| 28 | `cat_d_28_tensile_host_device_id_validation.patch` | Invalid device ID assignment. |

### Category E: Numerical Precision Bugs (28)

Bugs that induce numerical errors - precision loss, type confusion, overflow.

| # | Patch File | Description |
|---|------------|-------------|
| 1 | `cat_e_01_layernorm_epsilon_precision.patch` | Reduces LayerNorm epsilon by 10x. |
| 2 | `cat_e_02_scale_factor_truncation.patch` | Swaps scaleB and scaleC pointers. |
| 3 | `cat_e_03_hpa_precision_loss.patch` | Disables High Precision Accumulate unconditionally. |
| 4 | `cat_e_04_compute_input_type_mismatch.patch` | Swaps A/B types in mixed-precision determination. |
| 5 | `cat_e_05_xf32_math_op_override.patch` | Inverts XFloat32/Float math op logic. |
| 6 | `cat_e_06_tensile_host_alpha_cast_precision_loss.patch` | Casts alpha to float losing double precision. |
| 7 | `cat_e_07_tensile_host_beta_truncation.patch` | Truncates beta to integer. |
| 8 | `cat_e_08_contraction_solution_alpha_half_precision.patch` | Always appends alpha_2 as half. |
| 9 | `cat_e_09_hipblaslt_ext_op_softmax_scale_error.patch` | Scales lda/ldb by wrong factors. |
| 10 | `cat_e_10_contraction_solution_stride_precision.patch` | Casts stride to int16 causing overflow. |
| 11 | `cat_e_11_tensile_host_element_size_comparison.patch` | Uses elementBits with wrong scaling. |
| 12 | `cat_e_12_contraction_solution_activation_arg_cast.patch` | Truncates activation args to int8. |
| 13 | `cat_e_13_hipblaslt_ext_op_amax_scale.patch` | Wrong n and ld values for amax. |
| 14 | `cat_e_14_contraction_problem_granularity_calc.patch` | Integer division loses precision. |
| 15 | `cat_e_15_tensile_host_f16_alpha_conversion.patch` | Double conversion through int truncates f16. |
| 16 | `cat_e_16_contraction_solution_bf16_buffer_handling.patch` | Treats BFloat16 as float. |
| 17 | `cat_e_17_tensile_host_scale_alpha_vec_type.patch` | Uses Int8 for scale vectors. |
| 18 | `cat_e_18_contraction_solution_complex_float_handling.patch` | Only copies real part of complex. |
| 19 | `cat_e_19_hipblaslt_ext_op_layernorm_variance.patch` | Negates inverse variance pointer. |
| 20 | `cat_e_20_tensile_host_workspace_stride_calc.patch` | Integer overflow in workspace stride. |
| 21 | `cat_e_21_tensile_host_pk_scaling_factor.patch` | Wrong order of operations loses precision. |
| 22 | `cat_e_22_contraction_solution_tile_granularity.patch` | Floor divide instead of ceil (missing tiles). |
| 23 | `cat_e_23_tensile_host_depth_u_division.patch` | Floor division causes iteration undercount. |
| 24 | `cat_e_24_contraction_solution_bias_stride_calc.patch` | Off-by-one in bias stride index. |
| 25 | `cat_e_25_tensile_host_scalar_value_enum.patch` | Rounds alpha to nearest integer. |
| 26 | `cat_e_26_contraction_solution_e_stride_precision.patch` | Truncates E stride to uint16. |
| 27 | `cat_e_27_hipblaslt_ext_op_scale_multiply.patch` | Adds spurious scale factor to LayerNorm. |
| 28 | `cat_e_28_tensile_host_batch_stride_overflow.patch` | Multiplies batch stride causing overflow. |

---

## Patch Summary Table

| Category | Count | Primary Files Modified | Detection Difficulty |
|----------|-------|----------------------|---------------------|
| A - Simple | 28 | `hipblaslt.cpp`, `tensile_host.cpp`, `ContractionProblem.cpp` | Easy |
| B - Complex | 28 | `tensile_host.cpp`, `ContractionSolution.cpp`, `ContractionProblem.cpp` | Medium |
| C - Advanced | 28 | `tensile_host.cpp`, `ContractionSolution.cpp`, `ContractionProblem.cpp` | Hard |
| D - Legendary | 28 | `tensile_host.cpp`, `HipSolutionAdapter.cpp`, `ContractionProblem.cpp` | Very Hard |
| E - Numerical | 28 | `tensile_host.cpp`, `ContractionSolution.cpp`, `hipblaslt-ext-op.cpp` | Easy to Expert |

**Total: 140 patches**

---

## TensileLite Integration Coverage

A significant portion of the new patches (40+) specifically target the TensileLite integration layer:

### Files Covered:
- `tensile_host.cpp` - Main integration point (~50 patches)
- `ContractionProblem.cpp` - Problem construction (~15 patches)
- `ContractionSolution.cpp` - Solution execution (~20 patches)
- `HipSolutionAdapter.cpp` - Kernel loading/launching (~8 patches)
- `KernelArguments.cpp` - Argument packing (~2 patches)
- `TensorDescriptor.cpp` - Tensor handling (~2 patches)

### Integration Points Tested:
- Tensor dimension and stride handling
- Batch index management
- Transpose operation logic
- Scale factor handling
- Workspace allocation
- Solution selection and heuristics
- Kernel argument packing
- Memory ordering and synchronization
- Code object loading

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
