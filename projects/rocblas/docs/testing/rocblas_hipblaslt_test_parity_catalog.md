# rocBLAS → hipBLASLt client test parity catalog

This document maps each rocBLAS gtest source module listed in
[`clients/gtest/CMakeLists.txt`](../../clients/gtest/CMakeLists.txt) to the closest hipBLASLt client-test
coverage, and records whether **today’s** rocBLAS would **attempt** hipBLASLt on the code path under
discussion.

It is meant for migration and test-strategy conversations (Tensile removal, hipBLASLt-first GEMM, and
long-term parity). It is **not** an exact gtest name-for-name list: rocBLAS expands YAML into very large
parameter spaces via [`clients/common/rocblas_gentest.py`](../../clients/common/rocblas_gentest.py).

## Part A — Legend and routing rules

### Column meanings (suite-level table)

| Column | Meaning |
|--------|---------|
| **rocBLAS module** | Source file linked into `rocblas-test`. |
| **Primary YAML** | Matching `*_gtest.yaml` in [`clients/gtest/`](../../clients/gtest/) when present (some harness sources have no YAML; some `blas_ex/*_ex_gtest.cpp` files share the Level-1 driver YAML). |
| **Category** | BLAS level or infra grouping. |
| **APIs (hint)** | Human hint; generated tests use the rocBLAS C API family under that name. |
| **YAML-backed** | `Y` if the YAML file exists beside the suite (parameterized tests). |
| **Tensile-only build** | `Y` if the object is linked only when `BUILD_WITH_TENSILE` is enabled. |
| **hipBLASLt today** | Whether the **production library** may call hipBLASLt for this family’s math (see below). |
| **Hypothetical LT delegation** | If the long-term assumption is “all GEMM-style contraction work goes through hipBLASLt”, is this row in scope? |
| **hipBLASLt analog** | Closest area under [`hipblaslt/clients/tests/data/`](../../../hipblaslt/clients/tests/data/) or **Gap**. |
| **Notes** | Build flags and caveats. |

### When hipBLASLt is tried today (GEMM / contraction path)

rocBLAS routes many GEMM workloads through `runContractionProblem` in
[`library/src/tensile_host.cpp`](../../library/src/tensile_host.cpp). When `BUILD_WITH_HIPBLASLT` is on and the
selected solution path is not Tensile-forced, the library may call `runContractionProblemHipBlasLT` in
[`library/src/hipblaslt_host.cpp`](../../library/src/hipblaslt_host.cpp) **before** falling back to Tensile.

`useHipBLASLt` (same file) consults `tryHipBLASLt` on the handle. Important behaviors live in
[`library/src/include/handle.hpp`](../../library/src/include/handle.hpp):

- **`ROCBLAS_USE_HIPBLASLT`** — when set to `0` or `1`, forces hipBLASLt off or on; when unset, default
  policy uses `isDefaultHipBLASLtArch()` (gfx **1200**, **1201**, **950**).
- **`ROCBLAS_USE_HIPBLASLT_BATCHED`** — when set to `0`, disables the batched / grouped path that would
  otherwise be considered when hipBLASLt is enabled.
- **Stream capture** — batched path is disabled during capture (`tryHipBLASLt`).
- **gfx950** — `useHipBLASLt` returns false on gfx950 unless hipBLASLt is forced on (see TODO in source).

Rows marked **Yes\*** in the table assume a normal handle, matching architecture policy, and a problem that
actually goes through `gemm_tensile.hpp` / `runContractionProblem`. Always confirm with logging or a
debugger when it matters for a CI configuration.

### hipBLASLt is not full BLAS

hipBLASLt’s primary client tests are **matmul-focused** (`matmul_gtest.yaml`, `auxiliary_gtest.yaml`,
`smoke_gtest.yaml`, `rocroller_gtest.yaml`). Most rocBLAS Level 1–3 tests exercise **non-GEMM** kernels.
Those rows are **out of scope** for “direct hipBLASLt parity” unless rocBLAS is refactored to express work
as hipBLASLt problems.

### Regenerating machine-assisted sections

From the rocBLAS project root:

```bash
python3 scripts/generate_hipblaslt_parity_appendix.py --parity-table-only   # Part B table only
python3 scripts/generate_hipblaslt_parity_appendix.py --json scripts/hipblaslt_parity_appendix_stats.json --no-appendix-stdout
python3 scripts/generate_hipblaslt_parity_appendix.py                      # Part C appendices (stdout)
```

The **suite table (Part B)** is emitted by `--parity-table-only` from
[`scripts/generate_hipblaslt_parity_appendix.py`](../../scripts/generate_hipblaslt_parity_appendix.py) so new
`_*_gtest.cpp` lines in CMake stay tracked—paste or wrap the output when CMake changes.


| rocBLAS module | Primary YAML | Category | APIs (hint) | YAML-backed | Tensile-only build | hipBLASLt today | Hypothetical LT delegation | hipBLASLt analog | Notes |
|---|---|---|---|:---:|:---:|---|---|---|---|
| `multiheaded_gtest.cpp` | `multiheaded_gtest.yaml` | Tensile / GEMM host | Multi-handle / Tensile host init (GEMM-related) | Y | Y | Yes* — hipBLASLt tried in `runContractionProblem` when `BUILD_WITH_HIPBLASLT`, not Tensile-only solution index, `useHipBLASLt` / `tryHipBLASLt` (see legend) | In-scope — GEMM-family / contraction | Gap — multi-handle / atomics + Tensile host | Linked only when `BUILD_WITH_TENSILE` |
| `atomics_mode_gtest.cpp` | `atomics_mode_gtest.yaml` | Tensile / GEMM host | Atomics mode + GEMM path | Y | Y | Yes* — hipBLASLt tried in `runContractionProblem` when `BUILD_WITH_HIPBLASLT`, not Tensile-only solution index, `useHipBLASLt` / `tryHipBLASLt` (see legend) | In-scope — GEMM-family / contraction | Gap — multi-handle / atomics + Tensile host | Linked only when `BUILD_WITH_TENSILE` |
| `get_solutions_gtest.cpp` | `get_solutions_gtest.yaml` | Tensile / GEMM host | `rocblas_get_*_solutions` / solution indices | Y | Y | Yes* — hipBLASLt tried in `runContractionProblem` when `BUILD_WITH_HIPBLASLT`, not Tensile-only solution index, `useHipBLASLt` / `tryHipBLASLt` (see legend) | In-scope — GEMM-family / contraction | `matmul_gtest.yaml` (heuristic / algo) — partial parity | Linked only when `BUILD_WITH_TENSILE` |
| `rocblas_gtest_main.cpp` | — | Client harness / infra | gtest main | — | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `rocblas_test.cpp` | — | Client harness / infra | shared test fixtures | — | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `asan_helpers_gtest.cpp` | — | Client harness / infra | ASan helpers | — | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `general_gtest.cpp` | `general_gtest.yaml` | Client / infra | `rocblas_*general*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `set_get_pointer_mode_gtest.cpp` | `set_get_pointer_mode_gtest.yaml` | Client / infra | `rocblas_*set_get_pointer_mode*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `set_get_atomics_mode_gtest.cpp` | `set_get_atomics_mode_gtest.yaml` | Client / infra | `rocblas_*set_get_atomics_mode*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `logging_mode_gtest.cpp` | `logging_mode_gtest.yaml` | Client / infra | `rocblas_*logging_mode*` (generated) | Y | — | Partial — `logging_mode_internal` targets hipBLASLt backend logging on matching arch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `ostream_threadsafety_gtest.cpp` | `ostream_threadsafety_gtest.yaml` | Client / infra | `rocblas_*ostream_threadsafety*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `set_get_vector_gtest.cpp` | `set_get_vector_gtest.yaml` | Client / infra | `rocblas_*set_get_vector*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `set_get_matrix_gtest.cpp` | `set_get_matrix_gtest.yaml` | Client / infra | `rocblas_*set_get_matrix*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas1/asum_gtest.cpp` | `asum_gtest.yaml` | BLAS 1 | `rocblas_*asum*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas1/axpy_gtest.cpp` | `axpy_gtest.yaml` | BLAS 1 | `rocblas_*axpy*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas1/copy_gtest.cpp` | `copy_gtest.yaml` | BLAS 1 | `rocblas_*copy*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas1/dot_gtest.cpp` | `dot_gtest.yaml` | BLAS 1 | `rocblas_*dot*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas1/iamaxmin_gtest.cpp` | `iamax_iamin_gtest.yaml` | BLAS 1 | `rocblas_*iamaxmin*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas1/nrm2_gtest.cpp` | `nrm2_gtest.yaml` | BLAS 1 | `rocblas_*nrm2*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas1/rot_gtest.cpp` | `rot_gtest.yaml` | BLAS 1 | `rocblas_*rot*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas1/scal_gtest.cpp` | `scal_gtest.yaml` | BLAS 1 | `rocblas_*scal*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas1/swap_gtest.cpp` | `swap_gtest.yaml` | BLAS 1 | `rocblas_*swap*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas_ex/axpy_ex_gtest.cpp` | `axpy_gtest.yaml` | BLAS extension | `rocblas_*axpy_ex*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog | `*_ex` cases live in the Level-1 driver YAML |
| `blas_ex/dot_ex_gtest.cpp` | `dot_gtest.yaml` | BLAS extension | `rocblas_*dot_ex*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog | `*_ex` cases live in the Level-1 driver YAML |
| `blas_ex/nrm2_ex_gtest.cpp` | `nrm2_gtest.yaml` | BLAS extension | `rocblas_*nrm2_ex*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog | `*_ex` cases live in the Level-1 driver YAML |
| `blas_ex/rot_ex_gtest.cpp` | `rot_gtest.yaml` | BLAS extension | `rocblas_*rot_ex*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog | `*_ex` cases live in the Level-1 driver YAML |
| `blas_ex/scal_ex_gtest.cpp` | `scal_gtest.yaml` | BLAS extension | `rocblas_*scal_ex*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog | `*_ex` cases live in the Level-1 driver YAML |
| `blas2/trsv_gtest.cpp` | `trsv_gtest.yaml` | BLAS 2 | `rocblas_*trsv*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/gbmv_gtest.cpp` | `gbmv_gtest.yaml` | BLAS 2 | `rocblas_*gbmv*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/gemv_gtest.cpp` | `gemv_gtest.yaml` | BLAS 2 | `rocblas_*gemv*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/hbmv_gtest.cpp` | `hbmv_gtest.yaml` | BLAS 2 | `rocblas_*hbmv*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/hemv_gtest.cpp` | `hemv_gtest.yaml` | BLAS 2 | `rocblas_*hemv*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/her_gtest.cpp` | `her_gtest.yaml` | BLAS 2 | `rocblas_*her*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/her2_gtest.cpp` | `her2_gtest.yaml` | BLAS 2 | `rocblas_*her2*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/hpmv_gtest.cpp` | `hpmv_gtest.yaml` | BLAS 2 | `rocblas_*hpmv*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/hpr_gtest.cpp` | `hpr_gtest.yaml` | BLAS 2 | `rocblas_*hpr*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/hpr2_gtest.cpp` | `hpr2_gtest.yaml` | BLAS 2 | `rocblas_*hpr2*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/trmv_gtest.cpp` | `trmv_gtest.yaml` | BLAS 2 | `rocblas_*trmv*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/tpmv_gtest.cpp` | `tpmv_gtest.yaml` | BLAS 2 | `rocblas_*tpmv*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/tbmv_gtest.cpp` | `tbmv_gtest.yaml` | BLAS 2 | `rocblas_*tbmv*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/tbsv_gtest.cpp` | `tbsv_gtest.yaml` | BLAS 2 | `rocblas_*tbsv*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/tpsv_gtest.cpp` | `tpsv_gtest.yaml` | BLAS 2 | `rocblas_*tpsv*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/ger_gtest.cpp` | `ger_gtest.yaml` | BLAS 2 | `rocblas_*ger*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/geru_gtest.cpp` | `geruc_gtest.yaml` | BLAS 2 | `rocblas_*geru*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/gerc_gtest.cpp` | `geruc_gtest.yaml` | BLAS 2 | `rocblas_*gerc*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/spr_gtest.cpp` | `spr_gtest.yaml` | BLAS 2 | `rocblas_*spr*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/spr2_gtest.cpp` | `spr2_gtest.yaml` | BLAS 2 | `rocblas_*spr2*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/syr_gtest.cpp` | `syr_gtest.yaml` | BLAS 2 | `rocblas_*syr*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/syr2_gtest.cpp` | `syr2_gtest.yaml` | BLAS 2 | `rocblas_*syr2*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/sbmv_gtest.cpp` | `sbmv_gtest.yaml` | BLAS 2 | `rocblas_*sbmv*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/spmv_gtest.cpp` | `spmv_gtest.yaml` | BLAS 2 | `rocblas_*spmv*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas2/symv_gtest.cpp` | `symv_gtest.yaml` | BLAS 2 | `rocblas_*symv*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas3/gemm_gtest.cpp` | `gemm_gtest.yaml` | BLAS 3 | `rocblas_*gemm*` (generated) | Y | — | Yes* — hipBLASLt tried in `runContractionProblem` when `BUILD_WITH_HIPBLASLT`, not Tensile-only solution index, `useHipBLASLt` / `tryHipBLASLt` (see legend) | In-scope — GEMM-family / contraction | `matmul_gtest.yaml` core matmul; mixed precision & epilogues vary |  |
| `blas_ex/gemm_ex_gtest.cpp` | `gemm_ex_gtest.yaml` | BLAS extension | `rocblas_*gemm_ex*` (generated) | N | — | Yes* — hipBLASLt tried in `runContractionProblem` when `BUILD_WITH_HIPBLASLT`, not Tensile-only solution index, `useHipBLASLt` / `tryHipBLASLt` (see legend) | In-scope — GEMM-family / contraction | `matmul_gtest.yaml` core matmul; mixed precision & epilogues vary | Includes mixed precision / `_ex` paths |
| `blas3/symm_gtest.cpp` | `symm_gtest.yaml` | BLAS 3 | `rocblas_*symm*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas3/hemm_gtest.cpp` | `hemm_gtest.yaml` | BLAS 3 | `rocblas_*hemm*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas3/trsm_gtest.cpp` | `trsm_gtest.yaml` | BLAS 3 | `rocblas_*trsm*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas3/trtri_gtest.cpp` | `trtri_gtest.yaml` | BLAS 3 | `rocblas_*trtri*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas3/trmm_gtest.cpp` | `trmm_gtest.yaml` | BLAS 3 | `rocblas_*trmm*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas3/syrk_gtest.cpp` | `syrk_gtest.yaml` | BLAS 3 | `rocblas_*syrk*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas3/syrkx_gtest.cpp` | `syrkx_gtest.yaml` | BLAS 3 | `rocblas_*syrkx*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas3/syr2k_gtest.cpp` | `syr2k_gtest.yaml` | BLAS 3 | `rocblas_*syr2k*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas3/herk_gtest.cpp` | `herk_gtest.yaml` | BLAS 3 | `rocblas_*herk*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas3/herkx_gtest.cpp` | `herkx_gtest.yaml` | BLAS 3 | `rocblas_*herkx*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas3/her2k_gtest.cpp` | `her2k_gtest.yaml` | BLAS 3 | `rocblas_*her2k*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas3/dgmm_gtest.cpp` | `dgmm_gtest.yaml` | BLAS 3 | `rocblas_*dgmm*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas3/geam_gtest.cpp` | `geam_gtest.yaml` | BLAS 3 | `rocblas_*geam*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas_ex/gemmt_gtest.cpp` | `gemmt_gtest.yaml` | BLAS extension | `rocblas_*gemmt*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas_ex/geam_ex_gtest.cpp` | `geam_ex_gtest.yaml` | BLAS extension | `rocblas_*geam_ex*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas_ex/syrk_ex_gtest.cpp` | `syrk_ex_gtest.yaml` | BLAS extension | `rocblas_*syrk_ex*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |
| `blas_ex/herk_ex_gtest.cpp` | `herk_ex_gtest.yaml` | BLAS extension | `rocblas_*herk_ex*` (generated) | Y | — | No — not `gemm_tensile` / contraction dispatch | Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here | Gap — no direct matmul analog |  |

## Part C — YAML template volume (appendix)

## Appendix: rocBLAS YAML test templates (rocblas_gtest.yaml closure)

Each number counts lines after `Tests:` that start a test template: `- name:` or `- {name: ...` (before `rocblas_gentest.py` expands cross-products). YAML anchor merge is not evaluated; per-file counts are intrinsic to each file.

| YAML file | Test templates |
|-----------|----------------:|
| `asum_gtest.yaml` | 20 |
| `atomics_mode_gtest.yaml` | 1 |
| `aux_gtest.yaml` | 0 |
| `axpy_gtest.yaml` | 25 |
| `blas1_gtest.yaml` | 0 |
| `blas2_gtest.yaml` | 0 |
| `blas3_gtest.yaml` | 0 |
| `copy_gtest.yaml` | 19 |
| `dgmm_gtest.yaml` | 22 |
| `dot_gtest.yaml` | 35 |
| `gbmv_gtest.yaml` | 22 |
| `geam_ex_gtest.yaml` | 8 |
| `geam_gtest.yaml` | 22 |
| `gemm_batched_gtest.yaml` | 37 |
| `gemm_gtest.yaml` | 118 |
| `gemm_strided_batched_gtest.yaml` | 46 |
| `gemmt_gtest.yaml` | 22 |
| `gemv_gtest.yaml` | 49 |
| `general_gtest.yaml` | 3 |
| `ger_gtest.yaml` | 26 |
| `geruc_gtest.yaml` | 20 |
| `get_solutions_gtest.yaml` | 4 |
| `hbmv_gtest.yaml` | 23 |
| `hemm_gtest.yaml` | 21 |
| `hemv_gtest.yaml` | 22 |
| `her2_gtest.yaml` | 23 |
| `her2k_gtest.yaml` | 18 |
| `her_gtest.yaml` | 23 |
| `herk_ex_gtest.yaml` | 8 |
| `herk_gtest.yaml` | 18 |
| `herkx_gtest.yaml` | 18 |
| `hpmv_gtest.yaml` | 23 |
| `hpr2_gtest.yaml` | 21 |
| `hpr_gtest.yaml` | 21 |
| `iamax_iamin_gtest.yaml` | 20 |
| `known_bugs.yaml` | 0 |
| `logging_mode_gtest.yaml` | 2 |
| `multiheaded_gtest.yaml` | 1 |
| `nrm2_gtest.yaml` | 20 |
| `ostream_threadsafety_gtest.yaml` | 1 |
| `rocblas_gtest.yaml` | 0 |
| `rot_gtest.yaml` | 30 |
| `sbmv_gtest.yaml` | 17 |
| `scal_gtest.yaml` | 23 |
| `set_get_atomics_mode_gtest.yaml` | 1 |
| `set_get_matrix_gtest.yaml` | 8 |
| `set_get_pointer_mode_gtest.yaml` | 1 |
| `set_get_vector_gtest.yaml` | 8 |
| `spmv_gtest.yaml` | 22 |
| `spr2_gtest.yaml` | 21 |
| `spr_gtest.yaml` | 21 |
| `swap_gtest.yaml` | 17 |
| `symm_gtest.yaml` | 23 |
| `symv_gtest.yaml` | 27 |
| `syr2_gtest.yaml` | 22 |
| `syr2k_gtest.yaml` | 21 |
| `syr_gtest.yaml` | 22 |
| `syrk_ex_gtest.yaml` | 8 |
| `syrk_gtest.yaml` | 20 |
| `syrkx_gtest.yaml` | 23 |
| `tbmv_gtest.yaml` | 21 |
| `tbsv_gtest.yaml` | 20 |
| `tpmv_gtest.yaml` | 22 |
| `tpsv_gtest.yaml` | 20 |
| `trmm_gtest.yaml` | 45 |
| `trmv_gtest.yaml` | 22 |
| `trsm_gtest.yaml` | 80 |
| `trsv_gtest.yaml` | 22 |
| `trtri_gtest.yaml` | 10 |
| **Sum** | **1379** |


## Appendix: hipBLASLt YAML test templates (hipblaslt_gtest.yaml closure)

Same counting convention as rocBLAS appendix.

| YAML file | Test templates |
|-----------|----------------:|
| `auxiliary_gtest.yaml` | 35 |
| `hipblaslt_common.yaml` | 0 |
| `hipblaslt_gtest.yaml` | 0 |
| `known_bugs.yaml` | 0 |
| `matmul_common.yaml` | 0 |
| `matmul_gtest.yaml` | 149 |
| `rocroller_gtest.yaml` | 1 |
| `smoke_gtest.yaml` | 26 |
| **Sum** | **211** |

