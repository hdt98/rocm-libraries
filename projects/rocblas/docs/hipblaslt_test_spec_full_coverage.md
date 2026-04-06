# hipBLASlt Test Spec: Full Coverage for rocBLAS (No Tensile)

This document defines the **complete set of hipBLASlt test cases** needed so that when rocBLAS removes Tensile and uses hipBLASlt for every contraction, every scenario that can occur during the rocBLAS test run is covered by at least one hipBLASlt test.

**Known gaps:** For scenarios we don't think hipBLASlt can currently do (batched + offset, arch 950 float/double/complex, batched in stream capture, etc.), see the section **"Scenarios we don't think hipBLASlt can currently do"** in [hipblaslt_requirements_per_test.md](hipblaslt_requirements_per_test.md). Each test case is specified so it can be implemented in the hipBLASlt repo (or as a rocBLAS-side contract test that calls hipBLASlt directly).

**Scope:** All scenarios that would call `runContractionProblemHipBlasLT` with a `RocblasContractionProblem` when Tensile is removed and all current “don’t try hipBLASlt” constraints are removed (batched, arch 950 float/double/complex, etc.).

---

## 1. Coverage dimensions

Every hipBLASlt test case is a GEMM-like contraction with:

| Dimension | Values | Notes |
|-----------|--------|--------|
| **Type (Ti, To, Tc)** | See §2 | 10 combinations from rocBLAS instantiations. |
| **Batch mode** | single, strided_batch, batched | single = batch_count 1, one base ptr; strided_batch = batch_count ≥ 1, base ptr + strides; batched = array of pointers. |
| **Offset** | zero, non_zero | buffer_offset_a/b/c/d; non_zero from gemm_internal / gemm_batched_internal. |
| **trans_a, trans_b** | N, T (and C for complex) | At least NN, NT, TN, TT for real; add CC, CT, TC, CC for complex. |
| **Sizes (m, n, k)** | See §3 | Representative + all sizes that appear in rocBLAS tests that hit this path. |

---

## 2. Type combinations (all must have at least one test)

| ID | Ti | To | Tc | Description |
|----|-----|-----|-----|-------------|
| H  | rocblas_half | rocblas_half | rocblas_half | Half |
| S  | float | float | float | Single |
| D  | double | double | double | Double |
| C  | rocblas_float_complex | rocblas_float_complex | rocblas_float_complex | Single complex |
| Z  | rocblas_double_complex | rocblas_double_complex | rocblas_double_complex | Double complex |
| HPA_HH | rocblas_half | rocblas_half | float | HPA half→half, float compute |
| HPA_HF | rocblas_half | float | float | HPA half A/B, float C/D |
| HPA_BB | rocblas_bfloat16 | rocblas_bfloat16 | float | BF16, float compute |
| HPA_BF | rocblas_bfloat16 | float | float | BF16 A/B, float C/D |
| I8 | int8_t | int32_t | int32_t | Int8 GEMM |

---

## 3. Size / shape matrix (representative + rocBLAS-triggered)

These (m, n, k) and (lda, ldb, ldc) appear in rocBLAS tests that reach the contraction path. hipBLASlt tests should include at least these.

| Size set | m | n | k | lda | ldb | ldc | Source / notes |
|----------|------|------|------|------|------|------|----------------|
| small | 8 | 8 | 8 | 8 | 8 | 8 | gemm_small, batched small |
| small_rect | 8 | 9 | 10 | 8 | 10 | 8 | small_matrix_size_range |
| algo_cov | 64 | 128 | 4 | 64 | 64 | 64 | algorithm_coverage_matrix_size_range |
| algo_cov_2 | 25 | 26 | 27 | 28 | 29 | 30 | algorithm_coverage_matrix_size_range |
| medium | 128 | 128 | 128 | 128 | 128 | 128 | medium_matrix_size_range, gemm_internal (with offset) |
| large | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | gemm_ILP64 |
| internal_1 | 1000 | 1001 | 101 | 2002 | 1003 | 1004 | gemm_internal_matrix_size (with offset) |
| internal_2 | 32768 | 480 | 1 | 32768 | 480 | 32768 | gemm_internal_matrix_size (with offset) |
| batched_8 | 8 | 8 | 8 | 8 | 8 | 8 | gemm_batched_ILP64 (batch_count 65539 or 511) |
| batched_64 | 64 | 64 | 64 | 64 | 64 | 64 | gemm_batched_ex_ILP64, batch_count 511 |
| syrk_like | 2011 | 2011 | 253 | 2011 | — | 2048 | syrk_ILP64 (derived GEMM) |
| edge_k0 | 8 | 8 | 0 | 8 | 8 | 8 | k=0 (C := beta*C) |
| edge_m0 | 0 | 8 | 8 | 8 | 8 | 8 | m=0 quick return (no call) |
| edge_n0 | 8 | 0 | 8 | 8 | 8 | 8 | n=0 quick return (no call) |

For **offset** tests, use the same sizes as internal_1, internal_2, medium with:

- offset_a = 4294967296 (stride_x in gemm_internal)
- offset_b = 4294967297 (stride_y)
- offset_c = 4294967298 (stride_d)

(Or smaller non-zero offsets if hipBLASlt only supports 32-bit offset; then document the gap.)

---

## 4. Enumerated test cases (checklist)

Each row is one hipBLASlt test case. Implement each in hipBLASlt (call same API as rocBLAS’s `runContractionProblemHipBlasLT` / `ConstructHipBlasLTGemm` / `ConstructHipBlasLTGroupedGemm` with these parameters; verify result vs reference).

### 4.1 Single GEMM (batch_count = 1, no offset)

| # | Type | m | n | k | lda | ldb | ldc | trans_a | trans_b | rocBLAS trigger |
|---|------|------|------|------|------|------|------|---------|---------|------------------|
| 1 | S | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | N | N | gemm_ILP64 |
| 2 | S | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | N | T | gemm_ILP64 |
| 3 | S | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | T | N | gemm_ILP64 |
| 4 | S | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | T | T | gemm_ILP64 |
| 5 | D | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | N | N | gemm_ILP64 |
| 6 | H | 8 | 8 | 8 | 8 | 8 | 8 | N | N | gemm_batched_ILP64 (single batch) |
| 7 | C | 128 | 128 | 128 | 128 | 128 | 128 | N | N | symm/hemm/gemm complex |
| 8 | C | 128 | 128 | 128 | 128 | 128 | 128 | N | T | complex trans |
| 9 | Z | 128 | 128 | 128 | 128 | 128 | 128 | N | N | double complex |
| 10 | HPA_HH | 64 | 64 | 64 | 64 | 64 | 64 | N | N | gemm_ex HPA |
| 11 | HPA_BB | 64 | 64 | 64 | 64 | 64 | 64 | N | N | gemm_ex bf16 |
| 12 | I8 | 64 | 64 | 64 | 64 | 64 | 64 | N | N | gemm_batched_ex_ILP64 int8 |
| 13 | S | 8 | 8 | 0 | 8 | 8 | 8 | N | N | k=0 (beta*C) |
| 14 | S | 1000 | 1001 | 101 | 2002 | 1003 | 1004 | N | N | gemm_internal size (no offset) |

### 4.2 Strided batch (batch_count ≥ 1, single base ptr + strides, no offset)

| # | Type | m | n | k | lda | ldb | ldc | batch_count | stride_a | stride_b | stride_c | trans_a | trans_b | rocBLAS trigger |
|---|------|------|------|------|------|------|------|-------------|----------|----------|----------|---------|---------|------------------|
| 15 | S | 128 | 128 | 128 | 128 | 128 | 128 | 3 | lda*k | ldb*n | ldc*n | N | N | gemm_strided_batched |
| 16 | S | 64 | 64 | 64 | 64 | 64 | 64 | 511 | lda*k | ldb*n | ldc*n | N | T | gemm_strided_batched_ex_ILP64 |
| 17 | H | 8 | 8 | 8 | 8 | 8 | 8 | 3 | lda*k | ldb*n | ldc*n | N | N | gemm_strided_batched |
| 18 | D | 64 | 64 | 64 | 64 | 64 | 64 | 2 | lda*k | ldb*n | ldc*n | T | N | strided batched |
| 19 | C | 32 | 32 | 32 | 32 | 32 | 32 | 3 | lda*k | ldb*n | ldc*n | N | N | complex strided |

### 4.3 Strided batch with non-zero offset

| # | Type | m | n | k | lda | ldb | ldc | batch_count | offset_a | offset_b | offset_c | trans_a | trans_b | rocBLAS trigger |
|---|------|------|------|------|------|------|------|-------------|----------|----------|----------|---------|---------|------------------|
| 20 | S | 1000 | 1001 | 101 | 2002 | 1003 | 1004 | 1 | 4294967296 | 4294967297 | 4294967298 | N | N | gemm_internal |
| 21 | S | 128 | 128 | 128 | 128 | 128 | 128 | 1 | 4294967296 | 4294967297 | 4294967298 | N | N | gemm_internal |
| 22 | S | 32768 | 480 | 1 | 32768 | 480 | 32768 | 1 | 4294967296 | 4294967297 | 4294967298 | N | N | gemm_internal |

(If hipBLASlt only accepts 32-bit offsets, use e.g. 1, 2, 3 or 256, 257, 258 and note that rocBLAS can pass larger offsets; add a separate “large offset” requirement.)

### 4.4 Batched (array of pointers, no offset)

| # | Type | m | n | k | lda | ldb | ldc | batch_count | trans_a | trans_b | rocBLAS trigger |
|---|------|------|------|------|------|------|------|-------------|---------|---------|------------------|
| 23 | S | 8 | 8 | 8 | 8 | 8 | 8 | 3 | N | N | gemm_batched |
| 24 | S | 64 | 64 | 64 | 64 | 64 | 64 | 511 | N | T | gemm_batched_ex_ILP64 |
| 25 | H | 8 | 8 | 8 | 8 | 8 | 8 | 65539 | N | N | gemm_batched_ILP64 (c_grid_yz_require_passes) |
| 26 | D | 32 | 32 | 32 | 32 | 32 | 32 | 4 | T | T | batched |
| 27 | C | 2 | 2 | 2 | 4 | 2 | 4 | 3 | N | N | hemm_batched_ILP64 / symm_batched |
| 28 | I8 | 64 | 64 | 64 | 64 | 64 | 64 | 511 | N | T | gemm_batched_ex_ILP64 int8 |

### 4.5 Batched with non-zero offset (critical gap)

| # | Type | m | n | k | lda | ldb | ldc | batch_count | offset_a | offset_b | offset_c | trans_a | trans_b | rocBLAS trigger |
|---|------|------|------|------|------|------|------|-------------|----------|----------|----------|---------|---------|------------------|
| 29 | S | 1000 | 1001 | 101 | 2002 | 1003 | 1004 | 3 | 4294967296 | 4294967297 | 4294967298 | N | N | gemm_batched_internal |
| 30 | S | 128 | 128 | 128 | 128 | 128 | 128 | 3 | 4294967296 | 4294967297 | 4294967298 | N | N | gemm_batched_internal |
| 31 | S | 32768 | 480 | 1 | 32768 | 480 | 32768 | 3 | 4294967296 | 4294967297 | 4294967298 | N | N | gemm_batched_internal |

(Again, use 32-bit-friendly offsets if needed and document the requirement for larger offsets.)

### 4.6 Types that are currently skipped (arch 950 / float-double-complex)

When we remove “don’t try hipBLASlt on arch 950 for 4-byte types,” these will be attempted on 950. Include at least one test per type so hipBLASlt is validated there:

| # | Type | m | n | k | lda | ldb | ldc | batch_count | trans_a | trans_b | Note |
|---|------|------|------|------|------|------|------|-------------|---------|---------|------|
| 32 | S | 256 | 256 | 256 | 256 | 256 | 256 | 1 | N | N | Arch 950 float |
| 33 | D | 256 | 256 | 256 | 256 | 256 | 256 | 1 | N | N | Arch 950 double |
| 34 | C | 64 | 64 | 64 | 64 | 64 | 64 | 1 | N | N | Arch 950 single complex |
| 35 | Z | 64 | 64 | 64 | 64 | 64 | 64 | 1 | N | N | Arch 950 double complex |

### 4.7 Remaining types (HPA, half) – one each

| # | Type | m | n | k | lda | ldb | ldc | batch_count | trans_a | trans_b |
|---|------|------|------|------|------|------|------|-------------|---------|---------|
| 36 | HPA_HF | 64 | 64 | 64 | 64 | 64 | 64 | 1 | N | N |
| 37 | HPA_BF | 64 | 64 | 64 | 64 | 64 | 64 | 1 | N | N |

### 4.8 Derived GEMM shapes (syrk / symm / trmm / trsm / gemmt)

These routines feed GEMM with specific (m,n,k) and trans. One test per “shape class” is enough if sizes and trans are covered:

| # | Description | m | n | k | lda | ldb | ldc | trans_a | trans_b | rocBLAS trigger |
|---|-------------|------|------|------|------|------|------|---------|---------|------------------|
| 38 | SYRK-like (N, N) | 2011 | 2011 | 253 | 2011 | — | 2048 | N | T | syrk_ILP64 |
| 39 | SYRK-like (T, N) | 2011 | 2011 | 253 | 2011 | — | 2048 | T | N | syrk_ILP64 |
| 40 | SYMM left | 30 | 60 | 30 | 64 | 32 | 32 | N | N | symm_batched_ILP64 |
| 41 | GEMMT-like | 3 | 3 | 469 | 3 | 469 | 3 | N | N | gemmt_ILP64 |

(ldb for syrk depends on trans; use standard BLAS layout. SYMM/HEMM/TRMM/TRSM use specific submatrices; the exact (m,n,k) and lda/ldb/ldc are derived in code—these rows capture representative shapes.)

---

## 5. Summary counts

| Category | # test cases | Notes |
|----------|--------------|--------|
| Single GEMM, no offset | 14 | All types, key sizes, k=0 |
| Strided batch, no offset | 5 | batch_count 2, 3, 511 |
| Strided batch, with offset | 3 | gemm_internal sizes + offsets |
| Batched, no offset | 6 | batch_count 3, 4, 511, 65539 |
| Batched, with offset | 3 | gemm_batched_internal |
| Arch 950 types (S, D, C, Z) | 4 | When constraint removed |
| HPA variants | 2 | HPA_HF, HPA_BF |
| Derived (syrk/symm/gemmt) | 4 | Representative shapes |
| **Total** | **41** | Minimal full-coverage set |

---

## 6. Implementation notes

- **API:** Each test should call the same hipBLASlt API that rocBLAS uses: build the problem (Gemm or GroupedGemm) with the same (m, n, k), types, trans, ld’s, batch_count, strides or pointer arrays, and offsets; run; compare to a reference (e.g. cblas or a simple host GEMM).
- **Pass criteria:** Test passes if hipBLASlt returns success and the result matches reference within tolerance (match rocBLAS test tolerances where applicable).
- **Optional:** Add a “rocBLAS contract” mode: run the corresponding rocBLAS test with Tensile disabled and hipBLASlt forced; assert that the rocBLAS test passed and that no fallback to source occurred (if detectable).
- **Order:** Implement 4.1 and 4.2 first (single + strided batch, no offset), then 4.4 (batched, no offset), then 4.3 and 4.5 (offsets), then 4.6–4.8.

This spec plus the two companion docs ([tests_exercising_rocblas_internal_gemm_64.md](tests_exercising_rocblas_internal_gemm_64.md), [hipblaslt_requirements_per_test.md](hipblaslt_requirements_per_test.md)) define the full set of hipBLASlt tests needed for full coverage of rocBLAS tests when Tensile is removed and hipBLASlt is used for everything.
