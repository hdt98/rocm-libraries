// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GPU GEMM backend for reference implementations.
//
// Public interface uses our stride-based layout (s0, s1 per tensor).
// The BLAS column-major translation is an internal detail.
//
// To replace rocBLAS with WMMA, hipBLAS, or a custom kernel:
// reimplement sgemm() and dgemm() — the callers only see strides.

#pragma once

#include <rocblas/rocblas.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace rocm_ck::test {

namespace detail {

inline void checkRocblas(rocblas_status status, const char* call, const char* file, int line)
{
    if(status != rocblas_status_success)
    {
        std::fprintf(
            stderr, "rocBLAS error %d at %s:%d: %s\n", static_cast<int>(status), file, line, call);
        std::abort();
    }
}

} // namespace detail

#define CHECK_ROCBLAS(call) \
    ::rocm_ck::test::detail::checkRocblas((call), #call, __FILE__, __LINE__)

class GemmBackend
{
    rocblas_handle handle_;

    // -----------------------------------------------------------------
    // BLAS layout translation (private implementation detail)
    // -----------------------------------------------------------------

    /// Parameters for a rocBLAS call, computed from our strides.
    struct BlasParams
    {
        bool transA, transB;
        int M, N, K;
        int lda, ldb, ldc;
        bool swapped; // if true: BLAS A slot = our B ptr, BLAS B slot = our A ptr
    };

    /// Translate stride-based layout to rocBLAS column-major convention.
    ///
    /// BLAS always writes C in column-major.  For row-major C, we use
    /// the identity C = A*B  <=>  C^T = B^T * A^T.  Row-major M×N in
    /// memory IS column-major N×M (the transpose), so we swap A<->B
    /// and M<->N, letting BLAS write C^T directly into our row-major C.
    static BlasParams translate(int M,
                                int N,
                                int K,
                                int64_t a_s0,
                                int64_t a_s1,
                                int64_t b_s0,
                                int64_t b_s1,
                                int64_t c_s0,
                                int64_t c_s1)
    {
        bool a_row = (a_s1 == 1);
        bool b_row = (b_s1 == 1);
        bool c_row = (c_s1 == 1);

        // Leading dimension: the non-unit stride
        auto ld = [](int64_t s0, int64_t s1) {
            return static_cast<int>((s1 == 1) ? s0 : s1);
        };

        int lda = ld(a_s0, a_s1);
        int ldb = ld(b_s0, b_s1);
        int ldc = ld(c_s0, c_s1);

        if(c_row)
        {
            // Swap trick: C^T = B^T * A^T
            // Row-major is already stored as col-major transpose → no BLAS transpose.
            // Col-major needs BLAS transpose to produce the transposed version.
            return {!b_row, !a_row, N, M, K, ldb, lda, ldc, true};
        }
        else
        {
            // Direct: C = A * B
            // Col-major is native BLAS → no transpose.
            // Row-major needs BLAS transpose to undo stored transpose.
            return {a_row, b_row, M, N, K, lda, ldb, ldc, false};
        }
    }

public:
    GemmBackend() { CHECK_ROCBLAS(rocblas_create_handle(&handle_)); }
    ~GemmBackend() { rocblas_destroy_handle(handle_); }

    GemmBackend(const GemmBackend&)            = delete;
    GemmBackend& operator=(const GemmBackend&) = delete;

    /// FP32 GEMM: C = alpha * A * B + beta * C
    /// A is M×K, B is K×N, C is M×N.  Strides (s0, s1) per tensor.
    /// All pointers are device memory.
    void sgemm(int M,
               int N,
               int K,
               float alpha,
               const float* A,
               int64_t a_s0,
               int64_t a_s1,
               const float* B,
               int64_t b_s0,
               int64_t b_s1,
               float beta,
               float* C,
               int64_t c_s0,
               int64_t c_s1)
    {
        auto p           = translate(M, N, K, a_s0, a_s1, b_s0, b_s1, c_s0, c_s1);
        const float* ba  = p.swapped ? B : A;
        const float* bb  = p.swapped ? A : B;
        auto opA         = p.transA ? rocblas_operation_transpose : rocblas_operation_none;
        auto opB         = p.transB ? rocblas_operation_transpose : rocblas_operation_none;
        CHECK_ROCBLAS(rocblas_sgemm(
            handle_, opA, opB, p.M, p.N, p.K, &alpha, ba, p.lda, bb, p.ldb, &beta, C, p.ldc));
    }

    /// FP64 GEMM: C = alpha * A * B + beta * C
    void dgemm(int M,
               int N,
               int K,
               double alpha,
               const double* A,
               int64_t a_s0,
               int64_t a_s1,
               const double* B,
               int64_t b_s0,
               int64_t b_s1,
               double beta,
               double* C,
               int64_t c_s0,
               int64_t c_s1)
    {
        auto p            = translate(M, N, K, a_s0, a_s1, b_s0, b_s1, c_s0, c_s1);
        const double* ba  = p.swapped ? B : A;
        const double* bb  = p.swapped ? A : B;
        auto opA          = p.transA ? rocblas_operation_transpose : rocblas_operation_none;
        auto opB          = p.transB ? rocblas_operation_transpose : rocblas_operation_none;
        CHECK_ROCBLAS(rocblas_dgemm(
            handle_, opA, opB, p.M, p.N, p.K, &alpha, ba, p.lda, bb, p.ldb, &beta, C, p.ldc));
    }
};

#undef CHECK_ROCBLAS

} // namespace rocm_ck::test
