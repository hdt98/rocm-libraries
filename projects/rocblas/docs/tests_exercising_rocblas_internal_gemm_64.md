# Tests Exercising `rocblas_internal_gemm_64`

This document lists and summarizes the rocBLAS client tests that exercise **`rocblas_internal_gemm_64`** — the 64-bit (ILP64) internal GEMM entry point used when migrating to hipBLASlt/TensileLite.

**Scope:** `rocblas_internal_gemm_64` is only on the call path when the **public API is the 64-bit (ILP64) API** (e.g. `rocblas_gemm_64`, `rocblas_syrk_64`). Tests that use `api: C_64` or `api: FORTRAN_64` with the routines listed below will invoke this path when dimensions and leading dimensions fit the 32-bit fast path (otherwise the 64-bit source-kernel path is used).

---

## 1. Library callers of `rocblas_internal_gemm_64`

The following library code paths call `rocblas_internal_gemm_64`:

| Routine / area | File(s) | Notes |
|----------------|--------|--------|
| **GEMM (direct)** | `src64/blas3/rocblas_gemm_kernels_64.cpp` | `rocblas_internal_gemm_template_64` → `rocblas_internal_gemm_64<false>`; `rocblas_internal_gemm_batched_template_64` → `rocblas_internal_gemm_64<true>`. Entry from `rocblas_gemm_64`, `rocblas_gemm_batched_64`, `rocblas_gemm_strided_batched_64`. |
| **SYRK / HERK** | `blas3/rocblas_syrk_herk_kernels.cpp` | Multiple calls for syrk/herk 64-bit paths. |
| **SYR2K / HER2K** | `blas3/rocblas_syr2k_her2k_kernels.cpp` | Multiple calls for syr2k/her2k 64-bit paths. |
| **SYMM / HEMM** | `blas3/rocblas_symm_hemm_kernels.cpp` | Many calls for symm/hemm (left/right, uplo, batched/strided) 64-bit paths. |
| **TRMM** | `blas3/rocblas_trmm_kernels.cpp` | Left/right, batched; uses `rocblas_internal_gemm_64<false>` and `<BATCHED, T>`. |
| **TRSM** | `blas3/rocblas_trsm_kernels.hpp` | Single call for 64-bit trsm path. |
| **GEMMT** | `blas_ex/rocblas_gemmt_kernels.hpp` | Batched and strided-batched gemmt 64-bit paths. |

---

## 2. Client tests by routine (64-bit API only)

Only tests that use **`api: C_64`** or **`api: FORTRAN_64`** for these routines exercise `rocblas_internal_gemm_64`. Tests with `api: C` or `api: FORTRAN` use the 32-bit internal GEMM path instead.

### 2.1 GEMM (direct)

**YAML:** `clients/gtest/gemm_gtest.yaml`

| Test name | Category | Functions | Notes |
|-----------|----------|-----------|--------|
| `gemm_ILP64` | pre_checkin | gemm, gemm_ex | M=N=K=2048, single; api: C_64. |
| `gemm_ILP64` | stress | gemm (half), gemm_ex (hpa half/bf16) | Large/overflow dimensions (e.g. 2147483649); api: C_64. |
| `gemm_ex_ldd_ILP64` | stress | gemm_ex | ldd overflow; api: C_64. |

**Note:** Many tests named `gemm_64` (e.g. `gemm_64`, `gemm_64_9_129`) use default API (C/FORTRAN) and therefore exercise **`rocblas_internal_gemm`** (32-bit), not `rocblas_internal_gemm_64`.

---

### 2.2 GEMM batched

**YAML:** `clients/gtest/gemm_batched_gtest.yaml`

| Test name | Category | Functions | Notes |
|-----------|----------|-----------|--------|
| `gemm_batched_ILP64` | stress | gemm_batched (half), gemm_batched_ex (real, int8) | M=N=K=8, batch_count from `c_grid_yz_require_passes`; api: C_64. |
| `gemm_batched_ex_ILP64` | pre_checkin | gemm_batched_ex | M=N=K=64, batch_count=511; api: C_64. |

---

### 2.3 GEMM strided batched

**YAML:** `clients/gtest/gemm_strided_batched_gtest.yaml`

| Test name | Category | Functions | Notes |
|-----------|----------|-----------|--------|
| `gemm_strided_batched_ILP64` | (see YAML) | gemm_strided_batched | api: C_64. |
| `gemm_strided_batched_ex_ILP64` | (see YAML) | gemm_strided_batched_ex | api: C_64. |

---

### 2.4 SYRK

**YAML:** `clients/gtest/syrk_gtest.yaml`

| Test name | Category | Notes |
|-----------|----------|--------|
| `syrk_ILP64` | stress | matrix_size with N up to 2011/1024, batch_count 2–3; api: C_64. |

---

### 2.5 SYR2K

**YAML:** `clients/gtest/syr2k_gtest.yaml`

| Test name | Category | Notes |
|-----------|----------|--------|
| `syr2k_ILP64` | stress | N up to 2011/1024, batch_count 2–3; api: C_64. |

---

### 2.6 HER2K

**YAML:** `clients/gtest/her2k_gtest.yaml`

| Test name | Category | Notes |
|-----------|----------|--------|
| `her2k_ILP64` | stress | N up to 2011/1024, batch_count 2–3; api: C_64. |

---

### 2.7 SYMM

**YAML:** `clients/gtest/symm_gtest.yaml`

| Test name | Category | Notes |
|-----------|----------|--------|
| `symm_ILP64_L` | stress | side L; api: C_64. |
| `symm_ILP64_R` | stress | side R; api: C_64. |
| `symm_batched_ILP64` | stress | symm_batched, symm_strided_batched; batch_count from `c_grid_yz_require_passes`; api: C_64. |

---

### 2.8 HEMM

**YAML:** `clients/gtest/hemm_gtest.yaml`

| Test name | Category | Notes |
|-----------|----------|--------|
| `hemm_ILP64_L` | stress | api: C_64. |
| `hemm_batched_ILP64` | stress | batch_count from `c_grid_yz_require_passes`; api: C_64. |

---

### 2.9 TRMM

**YAML:** `clients/gtest/trmm_gtest.yaml`

| Test name | Category | Notes |
|-----------|----------|--------|
| `trmm_left_ILP64` | (see YAML) | api: C_64. |
| `trmm_right_ILP64` | (see YAML) | api: C_64. |
| `trmm_batched_ILP64` | (see YAML) | api: C_64. |

---

### 2.10 TRSM

**YAML:** `clients/gtest/trsm_gtest.yaml`

| Test name | Category | Notes |
|-----------|----------|--------|
| `trsm_left_ILP64` | (see YAML) | api: C_64. |
| `trsm_right_ILP64` | (see YAML) | api: C_64. |
| `trsm_batched_ILP64` | (see YAML) | api: C_64. |

---

### 2.11 GEMMT

**YAML:** `clients/gtest/gemmt_gtest.yaml`

| Test name | Category | Notes |
|-----------|----------|--------|
| `gemmt_ILP64` | stress | gemmt_strided_batched; large lda/ldb/ldc; api: C_64. |
| `gemmt_batched_ILP64` | stress | gemmt_batched; batch_count 65536; api: C_64. |
| `gemmt_strided_batched_ILP64` | stress | gemmt_strided_batched; batch_count 65536; api: C_64. |

---

## 3. Tests that do *not* exercise `rocblas_internal_gemm_64`

These are important for hipBLASlt (e.g. batched + offset) but use the **32-bit** internal GEMM path:

| Test name | YAML | Why not _64 |
|-----------|------|-------------|
| `gemm_internal` | gemm_gtest.yaml | `api: INTERNAL`; uses 32-bit API with stride_x/y/d as offsets → `rocblas_internal_gemm`. |
| `gemm_batched_internal` | gemm_batched_gtest.yaml | `api: INTERNAL`; batched with offsets → `rocblas_internal_gemm` (batched). |
| (strided_batched internal if any) | gemm_strided_batched_gtest.yaml | Same idea: INTERNAL → 32-bit path. |

For hipBLASlt requirements, mirror the **problem shapes and offsets** from these tests (e.g. `gemm_internal_matrix_size`, batched + offset) when writing hipBLASlt-side tests, even though they do not call `rocblas_internal_gemm_64`.

---

## 4. Summary table (64-bit tests only)

| Routine | Test names (api: C_64 / FORTRAN_64) | YAML file |
|---------|-------------------------------------|-----------|
| gemm | gemm_ILP64 (×2), gemm_ex_ldd_ILP64 | gemm_gtest.yaml |
| gemm_batched | gemm_batched_ILP64, gemm_batched_ex_ILP64 | gemm_batched_gtest.yaml |
| gemm_strided_batched | gemm_strided_batched_ILP64, gemm_strided_batched_ex_ILP64 | gemm_strided_batched_gtest.yaml |
| syrk | syrk_ILP64 | syrk_gtest.yaml |
| syr2k | syr2k_ILP64 | syr2k_gtest.yaml |
| her2k | her2k_ILP64 | her2k_gtest.yaml |
| symm | symm_ILP64_L, symm_ILP64_R, symm_batched_ILP64 | symm_gtest.yaml |
| hemm | hemm_ILP64_L, hemm_batched_ILP64 | hemm_gtest.yaml |
| trmm | trmm_left_ILP64, trmm_right_ILP64, trmm_batched_ILP64 | trmm_gtest.yaml |
| trsm | trsm_left_ILP64, trsm_right_ILP64, trsm_batched_ILP64 | trsm_gtest.yaml |
| gemmt | gemmt_ILP64, gemmt_batched_ILP64, gemmt_strided_batched_ILP64 | gemmt_gtest.yaml |

---

## 5. How this maps to hipBLASlt tests

- **Direct GEMM path:** When dimensions fit in 32 bits, `rocblas_internal_gemm_64` calls `rocblas_internal_gemm`, which may use hipBLASlt via `runContractionProblem` → `runContractionProblemHipBlasLT`. So the **same contraction problems** that appear in the tests above (for the 32-bit-dimension case) are what rocBLAS will send to hipBLASlt.
- **Indirect paths (syrk, symm, trmm, etc.):** Those routines call `rocblas_internal_gemm_64` with derived (m,n,k, trans, ld, batch, offsets). hipBLASlt tests should cover the resulting **RocblasContractionProblem** shapes (strided batch, batched with/without offsets, trans combinations) that these tests produce.
- **Batched + offset:** Covered today by `gemm_batched_internal` (32-bit path); hipBLASlt should add equivalent tests at the hipBLASlt interface so that when rocBLAS uses hipBLASlt for batched GEMM with offsets, behavior is validated there first.
