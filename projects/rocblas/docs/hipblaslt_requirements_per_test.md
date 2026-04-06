# What hipBLASlt Must Do So rocBLAS Tests Execute on hipBLASlt

This document maps each test scenario (from [tests_exercising_rocblas_internal_gemm_64.md](tests_exercising_rocblas_internal_gemm_64.md)) to the **contraction problem** that rocBLAS would pass to hipBLASlt and what **hipBLASlt must support** so that execution goes through hipBLASlt instead of falling back to Tensile.

**Important:** Execution only reaches hipBLASlt when:

1. The test uses the 64-bit API and the call goes through `rocblas_internal_gemm_64` → `rocblas_internal_gemm` (i.e. dimensions and leading dimensions fit in 32 bits; otherwise the 64-bit source kernel path is used).
2. `useHipBLASLt(prob)` returns true (see “rocBLAS gates” below).
3. `runContractionProblemHipBlasLT(prob, ...)` succeeds: problem construction, heuristic/init, `gemm.initialize()`, and `gemm.run()` all succeed.

So for each scenario we list: the **problem shape** that would be sent to hipBLASlt, and what **hipBLASlt must do** so that run succeeds.

---

## rocBLAS gates (must be true for hipBLASlt to be tried)

- **Build:** `BUILD_WITH_HIPBLASLT` must be defined.
- **Handle:** `handle->tryHipBLASLt(batched)` must be true:
  - Default: `ROCBLAS_USE_HIPBLASLT` unset and default arch supports hipBLASlt, or `ROCBLAS_USE_HIPBLASLT=1`.
  - **Batched only:** `ROCBLAS_USE_HIPBLASLT_BATCHED` must not be `"0"`, and handle must not be in stream capture mode (batched is opt-in).
- **Arch 950:** For element types with `sizeof(Ti) >= 4` (e.g. float, double, complex), rocBLAS skips hipBLASlt on arch 950 unless `handle->isHipBLASLtForcedOn()` is true (i.e. `ROCBLAS_USE_HIPBLASLT=1`).

So for “execution on hipBLASlt” we assume the run is on an arch where rocBLAS tries hipBLASlt and, for batched tests, that batched hipBLASlt is enabled.

---

## hipBLASlt type instantiations (rocBLAS side)

rocBLAS only calls `runContractionProblemHipBlasLT` for these `(Ti, To, Tc)` combinations (from `hipblaslt_host.cpp`):

- `(rocblas_half, rocblas_half, rocblas_half)` — half
- `(float, float, float)` — single
- `(double, double, double)` — double
- `(rocblas_float_complex, rocblas_float_complex, rocblas_float_complex)` — single complex
- `(rocblas_double_complex, rocblas_double_complex, rocblas_double_complex)` — double complex
- HPA: `(rocblas_half, rocblas_half, float)`, `(rocblas_half, float, float)`, `(rocblas_bfloat16, rocblas_bfloat16, float)`, `(rocblas_bfloat16, float, float)`
- Int8: `(int8_t, int32_t, int32_t)`

If a test uses a routine/type that is not in this list, it never goes to hipBLASlt. The requirements below focus on problem shapes and batch/offset behavior for these types.

---

## 1. Direct GEMM tests (64-bit API)

| Test(s) | Problem shape (when dims fit in 32-bit) | What hipBLASlt must do |
|--------|------------------------------------------|-------------------------|
| **gemm_ILP64** (pre_checkin) | Strided batch; `m=n=k=2048`; `batch_count=1`; `transA, transB` in {N,T}; `Ti=To=Tc` = float (and gemm_ex single); no offset. | Support strided GEMM (or single GEMM) for (2048, 2048, 2048), float, trans in {NN, NT, TN, TT}, batch_count=1, no buffer offset. Provide a solution that passes heuristic, init, and run. |
| **gemm_ILP64** (stress) | Same but half, bf16, HPA; some cases have `m` or `n` or `ld` &gt; INT32_MAX → those use 64-bit source path, not hipBLASlt. | For cases with m,n,k,lda,ldb,ldc ≤ INT32_MAX: support half, bf16, and HPA (Ti,To,Tc) as above. For overflow sizes, hipBLASlt is not in the path. |
| **gemm_ex_ldd_ILP64** | Small m,n,k; `ldd` very large (e.g. 2147483649). C/D leading dimension in problem may still be the logical ld. | Support the (Ti,To,Tc) and problem layout used by gemm_ex with the given ldd (if that still results in a 32-bit problem view). |

---

## 2. GEMM batched tests (64-bit API)

| Test(s) | Problem shape | What hipBLASlt must do |
|--------|----------------|-------------------------|
| **gemm_batched_ILP64** | **Batched** (array of pointers); `m=n=k=8`; `batch_count` from `c_grid_yz_require_passes` (large); half and gemm_batched_ex (real, int8); no offset. | Support **batched** GEMM (GroupedGemm path): same (m,n,k), batch_count large, `batch_A != nullptr`. Heuristic/init/run must succeed. rocBLAS only tries hipBLASlt for batched if `ROCBLAS_USE_HIPBLASLT_BATCHED` is not "0". |
| **gemm_batched_ex_ILP64** | Batched; `m=n=k=64`; `batch_count=511`; real and int8 types. | Same: batched GEMM, (64,64,64), batch_count=511. hipBLASlt must support this batch count and type set. |

---

## 3. GEMM strided batched tests (64-bit API)

| Test(s) | Problem shape | What hipBLASlt must do |
|--------|----------------|-------------------------|
| **gemm_strided_batched_ILP64**, **gemm_strided_batched_ex_ILP64** | **Strided batch** (`batch_A == nullptr`, single base ptr + strides); batch_count and sizes per YAML. | Support strided-batch GEMM: one base pointer per matrix, batch_stride_* for each batch. Problem has `strided_batch == true`. Heuristic/init/run must succeed. |

---

## 4. SYRK / SYR2K / HER2K (64-bit API)

These routines call `rocblas_internal_gemm_64` with **derived** (m,n,k) and trans from the syrk/syr2k/her2k arguments. The contraction problem is a GEMM with specific shapes (e.g. N×K, K×N, N×N).

| Test(s) | Problem shape | What hipBLASlt must do |
|--------|----------------|-------------------------|
| **syrk_ILP64**, **syr2k_ILP64**, **her2k_ILP64** | Strided or batched GEMM with N up to ~2011, K to ~1200, batch_count 2–3; types per routine (real/complex). | Support the resulting (m,n,k) and trans combinations and batch layout (strided or batched) that syrk/syr2k/her2k generate. Correct handling of leading dimensions and strides. |

---

## 5. SYMM / HEMM (64-bit API)

Same idea: symm/hemm reduce to one or more GEMM calls with derived dimensions and sides.

| Test(s) | Problem shape | What hipBLASlt must do |
|--------|----------------|-------------------------|
| **symm_ILP64_L/R**, **symm_batched_ILP64** | GEMM problems from symm (left/right, uplo); matrix sizes from YAML; batched test has large batch_count. | Support the (m,n,k), trans, and batch layout produced by symm. |
| **hemm_ILP64_L**, **hemm_batched_ILP64** | GEMM problems from hemm; complex types; batched with large batch_count. | Support complex GEMM shapes and batch layout from hemm. |

---

## 6. TRMM / TRSM (64-bit API)

TRMM/TRSM call `rocblas_internal_gemm_64` for the triangular solve’s GEMM subproblems.

| Test(s) | Problem shape | What hipBLASlt must do |
|--------|----------------|-------------------------|
| **trmm_left_ILP64**, **trmm_right_ILP64**, **trmm_batched_ILP64** | GEMM subproblems from trmm (left/right, diag, trans); sizes from YAML; batched when applicable. | Support the GEMM (m,n,k), trans, and batch/strided layout that trmm generates. |
| **trsm_left_ILP64**, **trsm_right_ILP64**, **trsm_batched_ILP64** | GEMM subproblems from trsm. | Same: support GEMM shapes and layout from trsm. |

---

## 7. GEMMT (64-bit API)

| Test(s) | Problem shape | What hipBLASlt must do |
|--------|----------------|-------------------------|
| **gemmt_ILP64**, **gemmt_batched_ILP64**, **gemmt_strided_batched_ILP64** | GEMM subproblems from gemmt; some with very large lda/ldb/ldc (e.g. 2147483649); batch_count up to 65536. | For cases where the inner GEMM has 32-bit dimensions and leading dims, support the resulting problem. For huge ld or batch_count, rocBLAS may use 64-bit or source path; hipBLASlt must support at least the “normal” sizes and batch counts that gemmt feeds. |

---

## 8. Tests that do *not* use the 64-bit API but matter for hipBLASlt

These use **rocblas_internal_gemm** (32-bit path) with **non-zero offsets** (via `api: INTERNAL`). They are the main “batched + offset” cases.

| Test(s) | Problem shape | What hipBLASlt must do |
|--------|----------------|-------------------------|
| **gemm_internal** | Strided GEMM; `matrix_size`: e.g. (1000,1001,101), (128,128,128), (32768,480,1); **stride_x, stride_y, stride_d** used as **offset_a, offset_b, offset_c** (large values, e.g. 4294967296). | Support **strided GEMM with non-zero buffer_offset_a/b/c/d**. rocBLAS passes pointers already offset (base + offset); hipBLASlt must produce correct results with those inputs and ld’s. |
| **gemm_batched_internal** | **Batched** GEMM; same `gemm_internal_matrix_size` and **offsets** (stride_x/y/d); **batch_count=3**. | Support **batched GEMM with non-zero buffer_offset_a/b/c/d**. Each batch gets `ptr[batch] + offset`; hipBLASlt must handle this correctly (GroupedGemm path with offsets). The reported offset problem may be this path or strided+offset (or both); confirm with team. |

---

## Summary: minimal checklist for hipBLASlt so rocBLAS uses it

1. **Strided GEMM (no offset):** (m,n,k), all supported types, trans, batch_count=1 or strided batch. Heuristic + init + run succeed.
2. **Strided GEMM with offset:** Same but `buffer_offset_a/b/c/d` non-zero; pointers passed as base+offset; correct result.
3. **Batched GEMM (no offset):** Array-of-pointers batched, batch_count ≥ 1; GroupedGemm path; heuristic/init/run succeed; rocBLAS only tries when `ROCBLAS_USE_HIPBLASLT_BATCHED` allows it.
4. **Batched GEMM with offset:** Same as batched but with non-zero offsets applied per batch (as rocBLAS does today). This is the case that needs tests and fixes on the hipBLASlt side.
5. **All (Ti,To,Tc) used by rocBLAS:** half, float, double, complex, HPA, int8 as instantiated in hipblaslt_host.cpp.
6. **Arch/behavior:** Where rocBLAS disables hipBLASlt (e.g. arch 950 for sizeof(Ti)>=4 unless forced), no change needed in hipBLASlt; for other archs, hipBLASlt must support the problem set above so that when rocBLAS tries it, it succeeds instead of falling back to Tensile.

Using this, you can add hipBLASlt-side tests that mirror these problem shapes and offsets; when those tests pass, the corresponding rocBLAS tests should execute on hipBLASlt (subject to the rocBLAS gates above).

---

## Data types: support vs mapping (moving away from Tensile completely)

**Short answer:** rocBLAS already uses the **same** (Ti, To, Tc) type set for both Tensile and hipBLASlt, and already **maps** those types to the hipBLASlt API. To move off Tensile completely, hipBLASlt must **support** (implement correctly) all of those types—no new types or new mappings are required in rocBLAS.

**Details:**

- **Tensile** and **hipBLASlt** in rocBLAS are instantiated for the same `RocblasContractionProblem` type combinations (see `tensile_host.cpp` and `hipblaslt_host.cpp`):
  - Same type: `rocblas_half`, `float`, `double`, `rocblas_float_complex`, `rocblas_double_complex`
  - HPA: `(rocblas_half, rocblas_half, float)`, `(rocblas_half, float, float)`, `(rocblas_bfloat16, rocblas_bfloat16, float)`, `(rocblas_bfloat16, float, float)`
  - Int8: `(int8_t, int32_t, int32_t)`

- **Mapping** from rocBLAS types to hipBLASlt is already in `hipblaslt_host.cpp`: `hipblaslt_datatype<T>` (e.g. float → HIP_R_32F, rocblas_half → HIP_R_16F) and `hipblaslt_compute_type<Tc>` (e.g. float → HIPBLAS_COMPUTE_32F, int32_t → HIPBLAS_COMPUTE_32I). So there is no type-set gap at the interface.

- **What matters for “no Tensile”:** For every one of those (Ti, To, Tc) combinations, hipBLASlt (TensileLite) must return a valid solution and produce correct results. If hipBLASlt does not support a type (e.g. no kernel or wrong result), rocBLAS will keep falling back to Tensile for that case. So the work is on the **hipBLASlt side**: implement and validate support for the same types Tensile already handles. No extra data types need to be added or mapped in rocBLAS to move away from Tensile; hipBLASlt just has to support the existing set.

---

## If Tensile is removed: do we have a complete set of tests that exercise hipBLASlt?

**Short answer:** Yes, with caveats. Every code path that today goes through `runContractionProblem` (and thus could use hipBLASlt) is reachable from the current rocBLAS test suite. So in a future where Tensile is removed and hipBLASlt is called directly, the same tests would exercise hipBLASlt. There is no missing “caller” or “routine” that never gets tested. Gaps are mainly: (1) coverage of *combinations* (e.g. batched + offset) is thin, (2) some paths are only in stress/nightly, and (3) we don’t today have a run that *forces* hipBLASlt and asserts it was used.

**What reaches hipBLASlt today (and in a no-Tensile world):**

The only way to get to `runContractionProblem` (and thus hipBLASlt) is:

1. **`rocblas_internal_gemm`** (in `gemm_templates.cpp`) → `rocblas_call_tensile` → `runContractionProblem`.  
   Called from:
   - Direct GEMM: `rocblas_gemm`, `rocblas_gemm_batched`, `rocblas_gemm_strided_batched` (32-bit API).
   - 64-bit GEMM: `rocblas_internal_gemm_64` when dimensions fit in 32 bits (which then calls `rocblas_internal_gemm`).
   - Level-3 / BLAS Ex that use internal GEMM: syrk, herk, syr2k, her2k, symm, hemm, trmm, trsm, gemmt (all via their 32-bit or 64-bit paths that eventually call `rocblas_internal_gemm` or `rocblas_internal_gemm_64`).

2. **`rocblas_internal_gemm_ex`** (in `rocblas_gemm_ex_kernels.cpp`) → `rocblas_call_tensile` → `runContractionProblem`.  
   Called from: `rocblas_gemm_ex`, `rocblas_gemm_batched_ex`, `rocblas_gemm_strided_batched_ex`.

So every routine that can use hipBLASlt is: GEMM (and _64), GEMM batched/strided batched (and _64), GEMM_ex (all variants), syrk, syr2k, her2k, symm, hemm, trmm, trsm, gemmt.

**Do we have tests for all of that?**

- **Yes.** The rocBLAS client includes (see `clients/gtest/blas3_gtest.yaml` and blas_ex): gemm, gemm_batched, gemm_strided_batched, gemmt, symm, syrk, syrk_ex, syr2k, syrkx, hemm, herk, herk_ex, her2k, herkx, trmm, trsm (and get_solutions, dgmm, geam, trtri, etc.). So every *caller* of the contraction path has tests.
- **Types:** Tests use half, float, double, complex, HPA, and int8 as in the instantiation list, so every (Ti, To, Tc) that can reach hipBLASlt is exercised by some test.
- **Batch modes:** Single, strided batch, and batched (array of pointers) all have tests.
- **Offsets:** Only `gemm_internal` and `gemm_batched_internal` (api: INTERNAL) pass non-zero offsets; that’s a small but existing set.

**Caveats (where “complete” is not perfect):**

1. **Batched + offset** is only exercised by `gemm_batched_internal` (one test configuration). So coverage for that combination is minimal; adding more such tests (or mirroring them in hipBLASlt) would strengthen the claim.
2. **Stress/nightly:** Some tests that hit hipBLASlt (e.g. large sizes, ILP64 stress) are only in stress or nightly. For “complete” in the sense of “every path runs in pre_checkin,” you’d want to ensure critical paths are in quick/pre_checkin.
3. **No “hipBLASlt-only” verification:** Today we don’t have a test run that forces hipBLASlt (e.g. disable Tensile or set env to prefer hipBLASlt) and asserts that the backend was hipBLASlt. So we don’t *prove* that hipBLASlt was used for a given test; we only know the path is covered. To get a “complete set that exercises hipBLASlt” in a strict sense, you’d add such a mode (or run with Tensile removed and confirm all these tests pass on hipBLASlt).
4. **Paths that skip hipBLASlt today** (e.g. batched in stream capture, or arch 950 for 4-byte types unless forced) would in a no-Tensile world need to use hipBLASlt or the source fallback. The same tests still run; they would just need to pass with that new backend choice.

**Conclusion:** For a future with no Tensile and hipBLASlt called directly, the current rocBLAS test suite does give a complete *set* of tests that would exercise hipBLASlt for every routine and type that today can reach it. The main improvement would be: more tests for batched + offset, and (optionally) a dedicated run that forces hipBLASlt and/or removes Tensile to confirm full pass.

---

## Where we don’t try hipBLASlt today (artificial constraints)

rocBLAS only calls `runContractionProblemHipBlasLT` when `useHipBLASLt(prob)` is true. So we **never query hipBLASlt** in these situations:

| Constraint | Where | Effect |
|------------|--------|--------|
| **Build** | `tensile_host.cpp` | `BUILD_WITH_HIPBLASLT` undefined → `useHipBLASLt` always false. |
| **Arch 950 + 4-byte types** | `useHipBLASLt()` | For `sizeof(Ti) >= 4` (float, double, complex), on arch 950 we return false unless `ROCBLAS_USE_HIPBLASLT=1` (force on). So we don’t try hipBLASlt for float/double/complex on 950 by default. |
| **Batched + env** | `tryHipBLASLt(batched)` | When `batched == true`, we require `ROCBLAS_USE_HIPBLASLT_BATCHED` not set to `"0"` and not in stream capture. If user sets that env to 0, we never try hipBLASlt for batched. |
| **Batched + stream capture** | `tryHipBLASLt(batched)` | When handle is in stream capture mode, we don’t try hipBLASlt for batched (sync copies in batched path). |
| **Default arch / env** | `tryHipBLASLt()` | If `ROCBLAS_USE_HIPBLASLT` is set to 0, or default arch doesn’t support hipBLASlt, we never try it. |

So “try hipBLASlt for everything” (with Tensile removed) means: remove or relax these so that every contraction goes to hipBLASlt (or source fallback). That would include **batched** on all archs, **float/double/complex on arch 950**, and **stream-capture batched** (or keep that as source-only if hipBLASlt can’t support it).

---

## Do we have a complete set of *hipBLASlt* tests for all those situations?

**No.** We don’t today have a complete set of **tests that run inside hipBLASlt** (or against the hipBLASlt API) that would cover every situation that would occur when rocBLAS tests run with Tensile removed and the above constraints removed.

What we have today:

- **In rocBLAS:** A full rocBLAS test suite that, in that hypothetical world, would *call* hipBLASlt for all those situations (every type, batched/strided, offsets, arch 950 float/double/complex, batched in stream capture if we tried it, etc.).
- **In our docs:** A *specification* of what problem shapes, types, and scenarios hipBLASlt would see (the two markdown files we created). That’s the list of what “a complete set of hipBLASlt tests” should cover.

What we do **not** have:

- A suite of tests **in the hipBLASlt repo** (or a dedicated “rocBLAS-contract” test suite) that:
  - Calls the hipBLASlt API directly with the same (m, n, k, type, batch, offset, trans) as rocBLAS would,
  - For **every** scenario that would occur when rocBLAS tests run with “hipBLASlt for everything, no Tensile.”

So the answer to “do we now have a complete set of hipBLASlt tests that would exercise all those situations when the rocBLAS tests ran?” is **no**. We have the **spec** for that set (the scenarios and requirements in this doc and the tests-exercising-internal-gemm-64 doc). The actual hipBLASlt tests that mirror those scenarios still need to be written (and would be the “tests that fail until hipBLASlt supports them” you proposed for the TensileLite team).

---

## Scenarios we don't think hipBLASlt can currently do

These are cases where rocBLAS either **does not try** hipBLASlt today (and the code/comments imply a limitation) or where the hipBLASlt team has indicated **support is missing or broken**. For "Tensile removed, hipBLASlt for everything," these would all need to work; until they do, they are explicit gaps.

| # | Scenario | Evidence / reason we think hipBLASlt can't do it today |
|---|----------|--------------------------------------------------------|
| 1 | **Batched GEMM with non-zero buffer offset** | rocBLAS passes `(ptr[batch] + offset)` per matrix (GroupedGemm path). It was reported that "batched calls with an offset" are a problem—**or** the issue may be strided + offset (see #6). We cannot tell from the rocBLAS code which one was meant; confirm with hipBLASlt/TensileLite team. |
| 2 | **Float / double / complex on arch 950 (gfx950)** | rocBLAS skips hipBLASlt for `sizeof(Ti) >= 4` on arch 950 unless `ROCBLAS_USE_HIPBLASLT=1` (tensile_host.cpp: "TODO remove after tuning"). So we assume hipBLASlt is not tuned or not reliable for float/double/complex on 950 yet. |
| 3 | **Batched GEMM when the handle is in stream capture mode** | rocBLAS never tries hipBLASlt for batched in stream capture. Comment in handle.hpp: "hipblaslt batched dispatch does synchronous memory copies," so batched is not graph/capture-safe. hipBLASlt would need an async/capture-safe batched path for this scenario. |
| 4 | **Very large batch counts (e.g. 65539)** | rocBLAS tests use `c_grid_yz_require_passes` (65539) for some batched ILP64 tests. We have not confirmed that hipBLASlt GroupedGemm or batched API supports or performs acceptably at that batch count. |
| 5 | **Large (e.g. 32-bit) buffer offsets** | gemm_internal uses stride_x/y/d as offsets (e.g. 4294967296). If hipBLASlt only supports 32-bit offset in elements (or in bytes), or has bugs with large offsets, this would be a gap. Unconfirmed in code; worth validating. |
| 6 | **Strided GEMM with non-zero offset** | rocBLAS passes `(base + offset)` for A/B/C/D in the strided path (single set of pointers). Same offset values as (5). The reported "offset problem" may be this path rather than (or in addition to) batched + offset; **confirm with hipBLASlt/TensileLite team**. Tests 20-22 in the full-coverage spec use these offsets. |

**Summary:** The reported "offset problem" could be **(1) batched + offset**, **(6) strided + offset**, or both—we cannot tell from the rocBLAS code; **confirm with the hipBLASlt/TensileLite team**. **(2)** and **(3)** are implied by rocBLAS's own skip logic and comments. **(4)** and **(5)** are suspected gaps worth confirming with the team and the test spec.
