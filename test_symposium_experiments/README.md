# Bug Injection Plan for hipBLASLt

I have created 20 bug patches across 4 categories (A, B, C, D) as requested.
The patches are located in `test_symposium_experiments/`.

## Categories

### Category A: Simple Bugs (5)

Basic validation and parameter errors.

1. `01_cat_a_hipblaslt_swap_rows_cols.patch`: Swaps rows and cols in `hipblasLtMatrixLayoutCreate`. Should segfault.
2. `02_cat_a_rocblaslt_mat_remove_type_check.patch`: Removes datatype check for C/D matrices.
3. `03_cat_a_rocblaslt_transform_swap_lda_ldb.patch`: Swaps ldA and ldB in transform kernel launch.
4. `04_cat_a_hipblaslt_ext_stride_error.patch`: Calculates strideA using N instead of K.
5. `05_cat_a_utility_wrong_compute_string.patch`: Returns wrong string for `COMPUTE_32XF`.

### Category B: Broad Complex Bugs (5)

Logic errors in core components.

6. `06_cat_b_tensile_host_memset_off_by_one.patch`: Off-by-one error in memset for heuristic results.
7. `07_cat_b_rocroller_gemm_trans_mixup.patch`: Uses transA instead of transB for Tensor B.
8. `08_cat_b_handle_wavefront_size_error.patch`: Incorrect wavefront size calculation (+1).
9. `09_cat_b_rocblaslt_auxiliary_attr_mixup.patch`: Copies to `op_B` instead of `op_A` when setting `TRANSA`.
10. `10_cat_b_user_driven_tuning_parser_swap_mn.patch`: Swaps M and N during tuning file parsing.

### Category C: Deep Advanced Bugs (5)

Subtle logic errors in specific paths.

11. `11_cat_c_solution_selection_logic_flip.patch`: Flips condition for analytical type selection.
12. `12_cat_c_runtime_args_selection_type_mixup.patch`: Uses typeB instead of typeA for element size.
13. `13_cat_c_hipblaslt_ext_op_softmax_dim_check.patch`: Rejects valid `dim=1` for softmax.
14. `14_cat_c_rocroller_host_algo_count_logic.patch`: Incorrect check for `returnAlgoCount`.
15. `15_cat_c_status_swallow_error.patch`: Swallows `hipErrorInvalidDevicePointer` in status conversion.

### Category D: Legendary Bugs (5)

Nuanced bugs with potential for catastrophic/silent failure.

16. `16_cat_d_parameter_selection_swizzle_scale.patch`: Incorrectly enables swizzle scale for smaller tiles.
17. `17_cat_d_rocblaslt_mat_utils_missing_check.patch`: Removes validity check for `HIP_R_16BF` order.
18. `18_cat_d_tuple_helper_off_by_one.patch`: Off-by-one error in tuple iteration (skips last pair).
19. `19_cat_d_hipblaslt_ostream_abort_race.patch`: Removes worker stop call during abort, causing potential race.
20. `20_cat_d_auxiliary_f8_type_mixup.patch`: Returns wrong FP8 type enum from string.

All patches are generated and ready for use.

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
    - Runs all 20 patches against the full test suite.
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
> If adding new patches, `git apply` can be too strict and may fail for some patches due to context mismatches.
