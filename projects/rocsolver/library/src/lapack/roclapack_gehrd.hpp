/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 *
 * and the MAGMA library (version 2.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * ************************************************************************/

#pragma once

#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

ROCSOLVER_BEGIN_NAMESPACE

constexpr rocblas_int GEHRD_GPU_THRESHOLD = 128;

inline rocblas_int get_gehrd_block_size(rocblas_int n)
{
    // Panel factorization uses scalar hblas_* (not optimized BLAS), so a
    // larger nb costs more panel time. Keep nb=32 for moderate sizes where
    // panel cost still matters, and only increase for very large matrices
    // where trailing GEMMs dominate.
    if(n >= 4096)
        return 64;
    return 32;
}

// ============================================================================
// Type-dispatched rocBLAS wrappers for float/double
// ============================================================================

template <typename T>
inline rocblas_status rocblas_gemm_dispatch(rocblas_handle handle,
                                           rocblas_operation transA,
                                           rocblas_operation transB,
                                           rocblas_int m, rocblas_int n, rocblas_int k,
                                           const T* alpha,
                                           const T* A, rocblas_int lda,
                                           const T* B, rocblas_int ldb,
                                           const T* beta,
                                           T* C, rocblas_int ldc);

template <>
inline rocblas_status rocblas_gemm_dispatch<double>(rocblas_handle handle,
                                                   rocblas_operation transA,
                                                   rocblas_operation transB,
                                                   rocblas_int m, rocblas_int n, rocblas_int k,
                                                   const double* alpha,
                                                   const double* A, rocblas_int lda,
                                                   const double* B, rocblas_int ldb,
                                                   const double* beta,
                                                   double* C, rocblas_int ldc)
{
    return rocblas_dgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline rocblas_status rocblas_gemm_dispatch<float>(rocblas_handle handle,
                                                  rocblas_operation transA,
                                                  rocblas_operation transB,
                                                  rocblas_int m, rocblas_int n, rocblas_int k,
                                                  const float* alpha,
                                                  const float* A, rocblas_int lda,
                                                  const float* B, rocblas_int ldb,
                                                  const float* beta,
                                                  float* C, rocblas_int ldc)
{
    return rocblas_sgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <typename T>
inline rocblas_status rocblas_gemv_dispatch(rocblas_handle handle,
                                           rocblas_operation trans,
                                           rocblas_int m, rocblas_int n,
                                           const T* alpha,
                                           const T* A, rocblas_int lda,
                                           const T* x, rocblas_int incx,
                                           const T* beta,
                                           T* y, rocblas_int incy);

template <>
inline rocblas_status rocblas_gemv_dispatch<double>(rocblas_handle handle,
                                                   rocblas_operation trans,
                                                   rocblas_int m, rocblas_int n,
                                                   const double* alpha,
                                                   const double* A, rocblas_int lda,
                                                   const double* x, rocblas_int incx,
                                                   const double* beta,
                                                   double* y, rocblas_int incy)
{
    return rocblas_dgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
inline rocblas_status rocblas_gemv_dispatch<float>(rocblas_handle handle,
                                                  rocblas_operation trans,
                                                  rocblas_int m, rocblas_int n,
                                                  const float* alpha,
                                                  const float* A, rocblas_int lda,
                                                  const float* x, rocblas_int incx,
                                                  const float* beta,
                                                  float* y, rocblas_int incy)
{
    return rocblas_sgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

// ============================================================================
// Host BLAS helpers for panel operations.
// GEMV and TRMV dispatch to Fortran BLAS for SIMD-optimized performance;
// copy/axpy/scal use simple scalar loops (called on very short vectors).
// ============================================================================

extern "C" {
void dgemv_(const char*, const int*, const int*, const double*, const double*, const int*,
            const double*, const int*, const double*, double*, const int*);
void sgemv_(const char*, const int*, const int*, const float*, const float*, const int*,
            const float*, const int*, const float*, float*, const int*);
void dtrmv_(const char*, const char*, const char*, const int*,
            const double*, const int*, double*, const int*);
void strmv_(const char*, const char*, const char*, const int*,
            const float*, const int*, float*, const int*);
void dcopy_(const int*, const double*, const int*, double*, const int*);
void scopy_(const int*, const float*, const int*, float*, const int*);
void daxpy_(const int*, const double*, const double*, const int*, double*, const int*);
void saxpy_(const int*, const float*, const float*, const int*, float*, const int*);
void openblas_set_num_threads(int) __attribute__((weak));
int openblas_get_num_threads(void) __attribute__((weak));
}

struct SingleThreadBLAS {
    int saved;
    SingleThreadBLAS()
    {
        saved = openblas_get_num_threads ? openblas_get_num_threads() : 1;
        if(openblas_set_num_threads)
            openblas_set_num_threads(1);
    }
    ~SingleThreadBLAS()
    {
        if(openblas_set_num_threads)
            openblas_set_num_threads(saved);
    }
};

template <typename T>
static inline void hblas_copy(rocblas_int n, const T* x, rocblas_int incx, T* y, rocblas_int incy);

template <>
inline void hblas_copy<double>(rocblas_int n, const double* x, rocblas_int incx, double* y, rocblas_int incy)
{
    int n_i = n, ix = incx, iy = incy;
    dcopy_(&n_i, x, &ix, y, &iy);
}

template <>
inline void hblas_copy<float>(rocblas_int n, const float* x, rocblas_int incx, float* y, rocblas_int incy)
{
    int n_i = n, ix = incx, iy = incy;
    scopy_(&n_i, x, &ix, y, &iy);
}

template <typename T>
static inline void hblas_axpy(rocblas_int n, T alpha, const T* x, rocblas_int incx, T* y, rocblas_int incy);

template <>
inline void hblas_axpy<double>(rocblas_int n, double alpha, const double* x, rocblas_int incx, double* y, rocblas_int incy)
{
    int n_i = n, ix = incx, iy = incy;
    daxpy_(&n_i, &alpha, x, &ix, y, &iy);
}

template <>
inline void hblas_axpy<float>(rocblas_int n, float alpha, const float* x, rocblas_int incx, float* y, rocblas_int incy)
{
    int n_i = n, ix = incx, iy = incy;
    saxpy_(&n_i, &alpha, x, &ix, y, &iy);
}

template <typename T>
static inline void hblas_scal(rocblas_int n, T alpha, T* x, rocblas_int incx)
{
    for(rocblas_int i = 0; i < n; ++i)
        x[i * incx] *= alpha;
}

// Threshold for dispatching to Fortran BLAS: below this, scalar loops
// are faster due to BLAS function-call and thread-pool overhead.
static constexpr rocblas_int HBLAS_GEMV_THRESHOLD = 128;

template <typename T>
static void hblas_gemv_scalar(char trans, rocblas_int m, rocblas_int n, T alpha,
                              const T* A, rocblas_int lda, const T* x, rocblas_int incx,
                              T beta, T* y, rocblas_int incy)
{
    if(trans == 'N' || trans == 'n')
    {
        for(rocblas_int i = 0; i < m; ++i)
        {
            T sum = T(0);
            for(rocblas_int j = 0; j < n; ++j)
                sum += A[i + j * static_cast<size_t>(lda)] * x[j * incx];
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    }
    else
    {
        for(rocblas_int j = 0; j < n; ++j)
        {
            T sum = T(0);
            for(rocblas_int i = 0; i < m; ++i)
                sum += A[i + j * static_cast<size_t>(lda)] * x[i * incx];
            y[j * incy] = alpha * sum + beta * y[j * incy];
        }
    }
}

template <typename T>
static void hblas_gemv(char trans, rocblas_int m, rocblas_int n, T alpha,
                       const T* A, rocblas_int lda, const T* x, rocblas_int incx,
                       T beta, T* y, rocblas_int incy);

template <>
void hblas_gemv<double>(char trans, rocblas_int m, rocblas_int n, double alpha,
                        const double* A, rocblas_int lda, const double* x, rocblas_int incx,
                        double beta, double* y, rocblas_int incy)
{
    if(m >= HBLAS_GEMV_THRESHOLD || n >= HBLAS_GEMV_THRESHOLD)
    {
        int m_i = m, n_i = n, lda_i = lda, ix = incx, iy = incy;
        dgemv_(&trans, &m_i, &n_i, &alpha, A, &lda_i, x, &ix, &beta, y, &iy);
    }
    else
        hblas_gemv_scalar(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
void hblas_gemv<float>(char trans, rocblas_int m, rocblas_int n, float alpha,
                       const float* A, rocblas_int lda, const float* x, rocblas_int incx,
                       float beta, float* y, rocblas_int incy)
{
    if(m >= HBLAS_GEMV_THRESHOLD || n >= HBLAS_GEMV_THRESHOLD)
    {
        int m_i = m, n_i = n, lda_i = lda, ix = incx, iy = incy;
        sgemv_(&trans, &m_i, &n_i, &alpha, A, &lda_i, x, &ix, &beta, y, &iy);
    }
    else
        hblas_gemv_scalar(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <typename T>
static void hblas_trmv_scalar(char uplo, char trans, char diag, rocblas_int n,
                              const T* A, rocblas_int lda, T* x, rocblas_int incx)
{
    const bool upper = (uplo == 'U' || uplo == 'u');
    const bool notrans = (trans == 'N' || trans == 'n');
    const bool unit = (diag == 'U' || diag == 'u');

    if(upper && notrans)
    {
        for(rocblas_int i = 0; i < n; ++i)
        {
            T sum = unit ? x[i * incx] : A[i + i * static_cast<size_t>(lda)] * x[i * incx];
            for(rocblas_int j = i + 1; j < n; ++j)
                sum += A[i + j * static_cast<size_t>(lda)] * x[j * incx];
            x[i * incx] = sum;
        }
    }
    else if(upper && !notrans)
    {
        for(rocblas_int i = n - 1; i >= 0; --i)
        {
            T sum = unit ? x[i * incx] : A[i + i * static_cast<size_t>(lda)] * x[i * incx];
            for(rocblas_int j = 0; j < i; ++j)
                sum += A[j + i * static_cast<size_t>(lda)] * x[j * incx];
            x[i * incx] = sum;
        }
    }
    else if(!upper && notrans)
    {
        for(rocblas_int i = n - 1; i >= 0; --i)
        {
            T sum = unit ? x[i * incx] : A[i + i * static_cast<size_t>(lda)] * x[i * incx];
            for(rocblas_int j = 0; j < i; ++j)
                sum += A[i + j * static_cast<size_t>(lda)] * x[j * incx];
            x[i * incx] = sum;
        }
    }
    else
    {
        for(rocblas_int i = 0; i < n; ++i)
        {
            T sum = unit ? x[i * incx] : A[i + i * static_cast<size_t>(lda)] * x[i * incx];
            for(rocblas_int j = i + 1; j < n; ++j)
                sum += A[j + i * static_cast<size_t>(lda)] * x[j * incx];
            x[i * incx] = sum;
        }
    }
}

template <typename T>
static void hblas_trmv(char uplo, char trans, char diag, rocblas_int n,
                       const T* A, rocblas_int lda, T* x, rocblas_int incx);

template <>
void hblas_trmv<double>(char uplo, char trans, char diag, rocblas_int n,
                        const double* A, rocblas_int lda, double* x, rocblas_int incx)
{
    if(n >= HBLAS_GEMV_THRESHOLD)
    {
        int n_i = n, lda_i = lda, ix = incx;
        dtrmv_(&uplo, &trans, &diag, &n_i, A, &lda_i, x, &ix);
    }
    else
        hblas_trmv_scalar(uplo, trans, diag, n, A, lda, x, incx);
}

template <>
void hblas_trmv<float>(char uplo, char trans, char diag, rocblas_int n,
                       const float* A, rocblas_int lda, float* x, rocblas_int incx)
{
    if(n >= HBLAS_GEMV_THRESHOLD)
    {
        int n_i = n, lda_i = lda, ix = incx;
        strmv_(&uplo, &trans, &diag, &n_i, A, &lda_i, x, &ix);
    }
    else
        hblas_trmv_scalar(uplo, trans, diag, n, A, lda, x, incx);
}

/** LARFG: generate a Householder reflector H = I - tau * v * v'
 *  such that H * [alpha; x] = [beta; 0]. */
template <typename T>
static void host_larfg(rocblas_int n, T& alpha, T* x, rocblas_int incx, T& tau)
{
    if(n <= 1)
    {
        tau = T(0);
        return;
    }
    T xnorm_sq = T(0);
    for(rocblas_int i = 0; i < n - 1; ++i)
        xnorm_sq += x[i * incx] * x[i * incx];

    if(xnorm_sq == T(0) && alpha >= T(0))
    {
        tau = T(0);
        return;
    }

    T beta = -std::copysign(std::sqrt(alpha * alpha + xnorm_sq), alpha);
    tau = (beta - alpha) / beta;
    T scal = T(1) / (alpha - beta);
    for(rocblas_int i = 0; i < n - 1; ++i)
        x[i * incx] *= scal;
    alpha = beta;
}

// ============================================================================
// Unblocked Hessenberg reduction (LAPACK DGEHD2-style).
// ilo, ihi are 1-based (LAPACK convention).
// ============================================================================

template <typename T, typename I>
void run_gehrd(const I n, const I ilo, const I ihi, T* A, const I lda, T* tau, rocblas_int& info)
{
    info = 0;
    if(n <= 0)
        return;
    for(I k = 0; k < n - 1; ++k)
        tau[k] = T(0);
    if(n <= 1 || ihi - ilo < 2)
        return;

    const I hi = ihi - 1;
    auto a = [&](I i, I j) -> T& { return A[i + j * lda]; };

    for(I i = ilo - 1; i <= ihi - 2; ++i)
    {
        const I nh = hi - i;
        if(nh <= 1)
        {
            tau[i] = T(0);
            continue;
        }

        T xnorm_sq = 0;
        for(I j = i + 2; j <= hi; ++j)
            xnorm_sq += a(j, i) * a(j, i);
        const T alpha = a(i + 1, i);

        if(xnorm_sq == 0 && alpha >= 0)
        {
            tau[i] = T(0);
        }
        else
        {
            const T beta = -std::copysign(std::sqrt(alpha * alpha + xnorm_sq), alpha);
            tau[i] = (beta - alpha) / beta;
            const T scal = T(1) / (alpha - beta);
            for(I j = i + 2; j <= hi; ++j)
                a(j, i) *= scal;
            a(i + 1, i) = beta;
        }

        if(tau[i] != T(0))
        {
            const T saved_beta = a(i + 1, i);
            a(i + 1, i) = T(1);
            for(I j = i + 1; j < n; ++j)
            {
                T dot = 0;
                for(I k = i + 1; k <= hi; ++k)
                    dot += a(k, i) * a(k, j);
                dot *= tau[i];
                for(I k = i + 1; k <= hi; ++k)
                    a(k, j) -= dot * a(k, i);
            }
            for(I row = 0; row < ihi; ++row)
            {
                T dot = 0;
                for(I k = i + 1; k <= hi; ++k)
                    dot += a(row, k) * a(k, i);
                dot *= tau[i];
                for(I k = i + 1; k <= hi; ++k)
                    a(row, k) -= dot * a(k, i);
            }
            a(i + 1, i) = saved_beta;
        }
    }
}

// ============================================================================
// Hybrid LAHR2: Panel factorization following MAGMA's dlahr2.
//
// Reduces NB columns of a general n-by-(n-k+1) matrix to upper Hessenberg
// form. Panel operations (LARFG, column update, T build) on host;
// Y-column computation on GPU via rocBLAS GEMV.
//
// All indexing is 0-based (MAGMA convention — k is already decremented).
//
// Parameters follow MAGMA's dlahr2 after k-=1:
//   n:   number of rows (typically IHI from GEHRD, 1-based)
//   k:   row offset, 0-based (panel start row)
//   nb:  number of columns to reduce
//   dA:  GPU matrix, shifted to panel start column (dA[i + j*ldda])
//   ldda: leading dimension of dA on GPU
//   dV:  GPU reflector storage, n × nb (zeroed before entry)
//   lddv: leading dimension of dV
//   hA:  Host matrix, shifted to panel start column
//   lda: leading dimension of hA on host
//   tau: Host output, nb scalars
//   hT:  Host output, nb × nb upper triangular T
//   ldt: leading dimension of hT
//   hY:  Host workspace, n × nb (Y columns downloaded here)
//   ldy: leading dimension of hY
// ============================================================================

template <typename T>
static rocblas_status hybrid_lahr2(rocblas_handle handle,
                                   hipStream_t stream,
                                   rocblas_int n,
                                   rocblas_int k,
                                   rocblas_int nb,
                                   T* dA,
                                   rocblas_int ldda,
                                   T* dV,
                                   rocblas_int lddv,
                                   T* hA,
                                   rocblas_int lda,
                                   T* tau,
                                   T* hT,
                                   rocblas_int ldt,
                                   T* hY,
                                   rocblas_int ldy)
{
    // Macros matching MAGMA's dlahr2 (0-based after k-=1)
    #define HA(i_, j_) (hA[(i_) + (j_) * static_cast<size_t>(lda)])
    #define HY(i_, j_) (hY[(i_) + (j_) * static_cast<size_t>(ldy)])
    #define HT(i_, j_) (hT[(i_) + (j_) * static_cast<size_t>(ldt)])

    const T c_zero = T(0);
    const T c_one = T(1);
    const T c_neg_one = T(-1);

    T ei = T(0);
    rocblas_int n_k, n_k_i_1;
    const rocblas_int ione = 1;

    if(n <= 1)
        return rocblas_status_success;

    // Panel BLAS operates on nb-wide vectors — force single-threaded BLAS
    // to avoid OpenBLAS thread-pool overhead on small operations.
    SingleThreadBLAS _blas_guard;

    for(rocblas_int i = 0; i < nb; ++i)
    {
        n_k_i_1 = n - k - i - 1;
        n_k = n - k;

        if(i > 0)
        {
            // ---- Update A(k:n-1, i) ----
            // Ref: MAGMA dlahr2.cpp lines 208-278

            // w(0:i-1) = VA(k+i, 0:i-1)' (copy row of A into last col of T as workspace)
            hblas_copy(i, &HA(k + i, 0), lda, &HT(0, nb - 1), ione);

            // w = T(0:i-1, 0:i-1) * w
            hblas_trmv('U', 'N', 'N', i, hT, ldt, &HT(0, nb - 1), ione);

            // A(k:n-1, i) -= Y(k:n-1, 0:i-1) * w
            hblas_gemv('N', n_k, i, c_neg_one, &HY(k, 0), ldy, &HT(0, nb - 1), ione,
                       c_one, &HA(k, i), ione);

            // Apply I - V * T' * V' to column i from the left
            // w := b1 = A(k+1:k+i, i)
            hblas_copy(i, &HA(k + 1, i), ione, &HT(0, nb - 1), ione);

            // w := V1' * b1 (V1 = unit lower tri in A(k+1:k+i, 0:i-1))
            hblas_trmv('L', 'T', 'U', i, &HA(k + 1, 0), lda, &HT(0, nb - 1), ione);

            // w += V2' * b2 (V2 = A(k+i+1:n-1, 0:i-1), b2 = A(k+i+1:n-1, i))
            hblas_gemv('T', n_k_i_1, i, c_one, &HA(k + i + 1, 0), lda,
                       &HA(k + i + 1, i), ione, c_one, &HT(0, nb - 1), ione);

            // w = T' * w
            hblas_trmv('U', 'T', 'N', i, hT, ldt, &HT(0, nb - 1), ione);

            // b2 -= V2 * w
            hblas_gemv('N', n_k_i_1, i, c_neg_one, &HA(k + i + 1, 0), lda,
                       &HT(0, nb - 1), ione, c_one, &HA(k + i + 1, i), ione);

            // b1 -= V1 * w
            hblas_trmv('L', 'N', 'U', i, &HA(k + 1, 0), lda, &HT(0, nb - 1), ione);
            hblas_axpy(i, c_neg_one, &HT(0, nb - 1), ione, &HA(k + 1, i), ione);

            // Restore saved diagonal element from previous column
            HA(k + i, i - 1) = ei;
        }

        // ---- LARFG: generate reflector to annihilate A(k+i+2:n-1, i) ----
        // Ref: MAGMA dlahr2.cpp lines 281-287
        host_larfg(n_k_i_1, HA(k + i + 1, i), &HA(std::min(k + i + 2, n - 1), i), ione, tau[i]);

        // Save diagonal and set to 1 for reflector multiply
        ei = HA(k + i + 1, i);
        HA(k + i + 1, i) = c_one;

        // ---- Upload reflector to dV ----
        // Ref: MAGMA dlahr2.cpp lines 289-292
        // dV(i+1:i+n_k_i_1, i) = A(k+i+1:n-1, i)
        HIP_CHECK(hipMemcpyAsync(dV + (i + 1) + i * static_cast<size_t>(lddv),
                                 &HA(k + i + 1, i), n_k_i_1 * sizeof(T),
                                 hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        // ---- GPU: compute Y(k:n-1, i) = A(k:n-1, i+1:) * V(i+1:, i) ----
        // Ref: MAGMA dlahr2.cpp lines 294-299
        // Y stored in dA(:, i), overwriting panel column on GPU
        if(n_k_i_1 > 0 && tau[i] != T(0))
        {
            rocblas_gemv_dispatch<T>(handle, rocblas_operation_none, n_k, n_k_i_1,
                          &c_one,
                          dA + k + (i + 1) * static_cast<size_t>(ldda), ldda,
                          dV + (i + 1) + i * static_cast<size_t>(lddv), ione,
                          &c_zero,
                          dA + k + i * static_cast<size_t>(ldda), ione);
        }
        else
        {
            HIP_CHECK(hipMemsetAsync(dA + k + i * static_cast<size_t>(ldda), 0,
                                     n_k * sizeof(T), stream));
        }

        // ---- T column: T(0:i-1, i) = -tau * V(i+1:,0:i-1)' * V(i+1:,i) ----
        // (on host; also needed for GPU Y correction below)
        // Ref: MAGMA dlahr2.cpp lines 301-312
        {
            T scale_neg_tau = -tau[i];
            hblas_gemv('T', n_k_i_1, i, scale_neg_tau, &HA(k + i + 1, 0), lda,
                       &HA(k + i + 1, i), ione, c_zero, &HT(0, i), ione);

            hblas_trmv('U', 'N', 'N', i, hT, ldt, &HT(0, i), ione);
            HT(i, i) = tau[i];
        }

        // ---- Download Y column from GPU to host ----
        // Ref: MAGMA dlahr2.cpp lines 315-318
        HIP_CHECK(hipMemcpyAsync(&HY(k, i),
                                 dA + k + i * static_cast<size_t>(ldda),
                                 n_k * sizeof(T), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    // Restore last saved diagonal element
    HA(k + nb, nb - 1) = ei;

    #undef HA
    #undef HY
    #undef HT

    return rocblas_status_success;
}

// ============================================================================
// GPU trailing matrix update following MAGMA's dlahru.
//
// After LAHR2 reduces a panel of nb columns, this routine applies the
// block reflector Q = I - V*T*V' to the trailing submatrix:
//   Right: A := A * Q        (updates Am and Ag)
//   Left:  A := Q' * A       (updates Ag2)
//
// All pointers are shifted to the panel-start column (0-based).
// Dimensions follow MAGMA's dlahru after k adjustment in GEHRD.
//
//   n:     full matrix order
//   ihi:   last active row (1-based LAPACK value, unchanged)
//   k:     panel start row (0-based)
//   nb:    block size
//   hA:    Host matrix shifted to panel column (for Am download)
//   lda:   leading dimension of hA
//   dA:    GPU matrix shifted to panel column
//   ldda:  leading dimension of dA
//   dY:    GPU Y matrix (= panel columns of dA, (ihi-k)×nb starting at dA(k,0))
//   lddy:  leading dimension of dY
//   dV:    GPU reflector matrix, (ihi-k)×nb
//   lddv:  leading dimension of dV
//   dT:    GPU nb×nb T matrix
//   dwork: GPU workspace, at least (ihi-k)×nb
//   ldwork: leading dimension of dwork
// ============================================================================

template <typename T>
static rocblas_status gpu_lahru(rocblas_handle handle,
                                hipStream_t stream,
                                rocblas_int n,
                                rocblas_int ihi,
                                rocblas_int k,
                                rocblas_int nb,
                                T* hA,
                                rocblas_int lda,
                                T* dA,
                                rocblas_int ldda,
                                T* dY,
                                rocblas_int lddy,
                                T* dV,
                                rocblas_int lddv,
                                T* dT,
                                T* dwork,
                                rocblas_int ldwork)
{
    const T c_zero = T(0);
    const T c_one = T(1);
    const T c_neg_one = T(-1);

    // Ym stored in dV below the reflectors (rows ihi-k..n-1 of dV)
    T* dYm = dV + (ihi - k);

    // Ref: MAGMA dlahru.cpp lines 157-162
    // Ym = Am * V = A(0:k-1, 0:ihi-k-1) * V(0:ihi-k-1, 0:nb-1)
    if(k > 0)
    {
        rocblas_gemm_dispatch<T>(handle, rocblas_operation_none, rocblas_operation_none,
                      k, nb, ihi - k,
                      &c_one, dA, ldda, dV, lddv,
                      &c_zero, dYm, ldda);
    }

    // Ref: MAGMA dlahru.cpp lines 166-171
    // W = V * T' (stored in dwork)
    rocblas_gemm_dispatch<T>(handle, rocblas_operation_none, rocblas_operation_transpose,
                  ihi - k, nb, nb,
                  &c_one, dV, lddv, dT, nb,
                  &c_zero, dwork, ldwork);

    // Ref: MAGMA dlahru.cpp lines 173-177
    // Am = Am - Ym * W' = A(0:k-1, 0:ihi-k-1) - Ym * W'
    if(k > 0)
    {
        rocblas_gemm_dispatch<T>(handle, rocblas_operation_none, rocblas_operation_transpose,
                      k, ihi - k, nb,
                      &c_neg_one, dYm, ldda, dwork, ldwork,
                      &c_one, dA, ldda);
    }

    // Ref: MAGMA dlahru.cpp lines 179-180
    // Download first nb columns of Am for host panel operations
    if(k > 0)
    {
        HIP_CHECK(hipMemcpy2DAsync(hA, lda * sizeof(T),
                                   dA, ldda * sizeof(T),
                                   k * sizeof(T), nb,
                                   hipMemcpyDeviceToHost, stream));
    }

    // Ref: MAGMA dlahru.cpp lines 185-189
    // Ag = Ag - Y * W' (right update of active rows, trailing columns only)
    // Ag = A(k:ihi-1, nb:ihi-k-1)
    if(ihi - k - nb > 0)
    {
        rocblas_gemm_dispatch<T>(handle, rocblas_operation_none, rocblas_operation_transpose,
                      ihi - k, ihi - k - nb, nb,
                      &c_neg_one, dY, lddy,
                      dwork + nb, ldwork,
                      &c_one, dA + k + nb * static_cast<size_t>(ldda), ldda);
    }

    // Ref: MAGMA dlahru.cpp lines 197-201
    // Left update: Z = V' * Ag2 (stored in dY, overwriting Y)
    // Ag2 = A(k:ihi-1, nb:n-k-1) — extends to rightmost columns
    if(n - k - nb > 0)
    {
        rocblas_gemm_dispatch<T>(handle, rocblas_operation_transpose, rocblas_operation_none,
                      nb, n - k - nb, ihi - k,
                      &c_one, dV, lddv,
                      dA + k + nb * static_cast<size_t>(ldda), ldda,
                      &c_zero, dY, nb);

        // Ref: MAGMA dlahru.cpp lines 203-207
        // Ag2 = Ag2 - W * Z
        rocblas_gemm_dispatch<T>(handle, rocblas_operation_none, rocblas_operation_none,
                      ihi - k, n - k - nb, nb,
                      &c_neg_one, dwork, ldwork,
                      dY, nb,
                      &c_one, dA + k + nb * static_cast<size_t>(ldda), ldda);
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    return rocblas_status_success;
}

// ============================================================================
// Blocked hybrid Hessenberg reduction for a single matrix.
//
// Follows MAGMA's dgehrd: panel factorization (LAHR2) on host with
// GPU GEMV for Y, and trailing matrix update (LAHRU) on GPU with GEMM.
//
// ilo, ihi: 1-based LAPACK convention.
// dA: n×n matrix on GPU.
// dTau: (n-1) array on GPU.
// ============================================================================

template <typename T>
static rocblas_status blocked_gehrd_gpu(rocblas_handle handle,
                                        hipStream_t stream,
                                        const rocblas_int n,
                                        const rocblas_int ilo_1based,
                                        const rocblas_int ihi_1based,
                                        T* dA,
                                        const rocblas_int lda,
                                        T* dTau)
{
    // Convert to 0-based (MAGMA convention)
    rocblas_int ilo = ilo_1based - 1;
    const rocblas_int ihi = ihi_1based; // ihi stays as LAPACK value

    const rocblas_int nh = ihi - ilo;
    if(nh <= 1)
        return rocblas_status_success;

    const rocblas_int nb = std::min(get_gehrd_block_size(nh), nh);
    const rocblas_int ldda = lda;

    // Host workspace
    const size_t full_elems = static_cast<size_t>(lda) * n;
    std::vector<T> hA(full_elems);
    std::vector<T> hT(nb * nb, T(0));
    std::vector<T> hY(static_cast<size_t>(n) * nb, T(0));
    std::vector<T> hTau(n > 1 ? n - 1 : 1, T(0));

    // GPU workspace: single allocation partitioned into dV, dT, dwork
    const rocblas_int lddv = n;
    const rocblas_int ldwork = n;
    const size_t sz_dV = static_cast<size_t>(n) * nb;
    const size_t sz_dT = static_cast<size_t>(nb) * nb;
    const size_t sz_dwork = static_cast<size_t>(n) * nb;
    const size_t total_gpu_elems = sz_dV + sz_dT + sz_dwork;

    T* dGpuWork = nullptr;
    HIP_CHECK(hipMalloc(&dGpuWork, total_gpu_elems * sizeof(T)));
    T* dV = dGpuWork;
    T* dT_gpu = dV + sz_dV;
    T* dwork = dT_gpu + sz_dT;

    // Ensure handle uses the caller's stream so that rocBLAS GEMM/GEMV ops
    // are ordered with respect to hipMemcpy*Async calls on the same stream.
    // Without this, the batched pipelining path (which passes a per-batch
    // stream different from the handle's default) would race.
    rocblas_set_stream(handle, stream);

    // Set pointer mode to host for BLAS scalars
    rocblas_pointer_mode orig_mode;
    rocblas_get_pointer_mode(handle, &orig_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // Initialize tau to zero outside active range
    for(rocblas_int i = 0; i < ilo; ++i)
        hTau[i] = T(0);
    for(rocblas_int i = std::max(rocblas_int(0), ihi - 1); i < n - 1; ++i)
        hTau[i] = T(0);

    // Ref: MAGMA dgehrd.cpp line 233
    // Zero initial block of dV
    HIP_CHECK(hipMemsetAsync(dV, 0, static_cast<size_t>(n) * nb * sizeof(T), stream));
    HIP_CHECK(hipMemsetAsync(dT_gpu, 0, nb * nb * sizeof(T), stream));

    // Main blocked loop
    // Ref: MAGMA dgehrd.cpp lines 252-278
    rocblas_int i;
    for(i = ilo; i < ihi - 1 - nb; i += nb)
    {
        if(i > ilo)
        {
            HIP_CHECK(hipMemcpy2DAsync(
                hA.data() + i + static_cast<size_t>(i) * lda,
                lda * sizeof(T),
                dA + i + static_cast<size_t>(i) * ldda,
                ldda * sizeof(T),
                static_cast<size_t>(ihi - i) * sizeof(T),
                nb,
                hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
        }
        else
        {
            HIP_CHECK(hipMemcpy2DAsync(
                hA.data() + static_cast<size_t>(ilo) * lda,
                lda * sizeof(T),
                dA + static_cast<size_t>(ilo) * ldda,
                ldda * sizeof(T),
                n * sizeof(T),
                n - ilo,
                hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
        }

        // Reset dV to zero for this panel
        HIP_CHECK(hipMemsetAsync(dV, 0, static_cast<size_t>(n) * nb * sizeof(T), stream));

        // Zero T
        std::fill(hT.begin(), hT.end(), T(0));

        // LAHR2: reduce panel columns i:i+nb-1
        // Ref: MAGMA dgehrd.cpp lines 263-267
        // MAGMA calls: magma_dlahr2(ihi, i+1, nb, dA(0,i-ilo), ldda, dV, ldda,
        //                           A(0,i), lda, tau+i, T, nb, Y, n, queue)
        // After k-=1 inside LAHR2: k = i (0-based)
        //
        // Shifted pointers: dA_shifted = dA + i*ldda, hA_shifted = hA + i*lda
        hybrid_lahr2<T>(handle, stream,
                        ihi,             // n (active row count)
                        i,               // k (0-based, already decremented)
                        nb,              // nb
                        dA + i * static_cast<size_t>(ldda),   // dA shifted to col i
                        ldda,
                        dV, lddv,
                        hA.data() + i * static_cast<size_t>(lda),  // hA shifted to col i
                        lda,
                        hTau.data() + i, // tau[i:i+nb-1]
                        hT.data(), nb,   // T
                        hY.data(), n);   // Y workspace

        // Upload T to GPU
        // Ref: MAGMA dgehrd.cpp line 270
        HIP_CHECK(hipMemcpyAsync(dT_gpu, hT.data(), nb * nb * sizeof(T),
                                 hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        // Trailing matrix update
        // Ref: MAGMA dgehrd.cpp lines 272-278
        // MAGMA calls: magma_dlahru(n, ihi, i, nb, A(0,i), lda,
        //                           dA(0,i-ilo), ldda, dA(i,i-ilo), ldda,
        //                           dV, ldda, dT(0,i-ilo), dwork, queue)
        gpu_lahru<T>(handle, stream,
                     n, ihi, i, nb,
                     hA.data() + i * static_cast<size_t>(lda),  // host shifted
                     lda,
                     dA + i * static_cast<size_t>(ldda),         // dA shifted
                     ldda,
                     dA + i + i * static_cast<size_t>(ldda),     // dY = dA(i, i) shifted
                     ldda,
                     dV, lddv,
                     dT_gpu,
                     dwork, ldwork);
    }

    // Download remainder from GPU to host
    {
        const rocblas_int ncols_rem = n - i;
        if(ncols_rem > 0)
        {
            HIP_CHECK(hipMemcpy2DAsync(
                hA.data() + static_cast<size_t>(i) * lda,
                lda * sizeof(T),
                dA + static_cast<size_t>(i) * ldda,
                ldda * sizeof(T),
                n * sizeof(T),
                ncols_rem,
                hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
        }
    }

    // Unblocked reduction of remaining columns.
    // Use run_gehrd on just the remainder strip by saving/restoring tau.
    {
        const rocblas_int i_1based = i + 1;
        std::vector<T> saved_tau(hTau.begin(), hTau.begin() + i);
        rocblas_int info_rem = 0;
        run_gehrd<T, rocblas_int>(n, i_1based, ihi, hA.data(), lda, hTau.data(), info_rem);
        std::copy(saved_tau.begin(), saved_tau.end(), hTau.begin());
    }

    // Upload result back to GPU using 2D memcpy (only active columns)
    if(n > 0)
    {
        HIP_CHECK(hipMemcpy2DAsync(
            dA, ldda * sizeof(T),
            hA.data(), lda * sizeof(T),
            n * sizeof(T), n,
            hipMemcpyHostToDevice, stream));
    }

    HIP_CHECK(hipMemcpyAsync(dTau, hTau.data(), (n - 1) * sizeof(T),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    rocblas_set_pointer_mode(handle, orig_mode);
    (void)hipFree(dGpuWork);

    return rocblas_status_success;
}

// ============================================================================
// Template infrastructure
// ============================================================================

template <bool BATCHED, bool STRIDED, typename T, typename I>
void rocsolver_gehrd_getMemorySize(const I n, const I batch_count, size_t* size_work)
{
    (void)n;
    (void)batch_count;
    *size_work = 0;
}

template <typename T, typename I, typename U>
rocblas_status rocsolver_gehrd_argCheck(rocblas_handle handle,
                                        const I n,
                                        const I ilo,
                                        const I ihi,
                                        const I lda,
                                        U A,
                                        T* tau,
                                        rocblas_int* info,
                                        const I batch_count = 1)
{
    if(n < 0 || lda < std::max<I>(1, n) || batch_count < 0)
        return rocblas_status_invalid_size;

    if(n > 0 && (ilo < 1 || ilo > std::max<I>(1, n) || ihi < ilo - 1 || ihi > n))
        return rocblas_status_invalid_size;

    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    if(!A || !tau || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_gehrd_template(rocblas_handle handle,
                                        const I n,
                                        const I ilo,
                                        const I ihi,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        T* tau,
                                        const rocblas_stride strideTau,
                                        rocblas_int* info,
                                        const I batch_count)
{
    ROCSOLVER_ENTER("gehrd", "n:", n, "ilo:", ilo, "ihi:", ihi, "lda:", lda, "bc:", batch_count);

    if(!batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    const size_t mat_elems = static_cast<size_t>(n) * static_cast<size_t>(lda);
    const size_t mat_bytes = mat_elems * sizeof(T);
    const size_t tau_elems = n > 0 ? static_cast<size_t>(n - 1) : 0;
    const rocblas_stride effStrideTau
        = strideTau ? strideTau : std::max<rocblas_stride>(1, static_cast<rocblas_stride>(tau_elems));

    if(n == 0)
    {
        const rocblas_int z = 0;
        for(I b = 0; b < batch_count; ++b)
        {
            HIP_CHECK(hipMemcpyAsync(info + b, &z, sizeof(rocblas_int), hipMemcpyHostToDevice, stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        return rocblas_status_success;
    }

    const rocblas_int nh = static_cast<rocblas_int>(ihi) - static_cast<rocblas_int>(ilo);
    const bool use_gpu = !BATCHED && (nh > GEHRD_GPU_THRESHOLD);

    if constexpr(!BATCHED)
    {
        if(use_gpu)
        {
            const rocblas_stride stride_a
                = strideA ? strideA : (rocblas_stride(lda) * rocblas_stride(n));

            for(I b = 0; b < batch_count; ++b)
            {
                T* dA_item = reinterpret_cast<T*>(A) + shiftA + b * stride_a;
                T* dTau_item = tau + static_cast<rocblas_stride>(b) * effStrideTau;

                HIP_CHECK(hipMemsetAsync(dTau_item, 0, tau_elems * sizeof(T), stream));

                blocked_gehrd_gpu<T>(handle, stream,
                                     static_cast<rocblas_int>(n),
                                     static_cast<rocblas_int>(ilo),
                                     static_cast<rocblas_int>(ihi),
                                     dA_item,
                                     static_cast<rocblas_int>(lda),
                                     dTau_item);

                const rocblas_int z = 0;
                HIP_CHECK(hipMemcpyAsync(info + b, &z, sizeof(rocblas_int),
                                         hipMemcpyHostToDevice, stream));
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            return rocblas_status_success;
        }
    }

    // ---- Host unblocked path (for small n or batched) ----
    std::vector<T> hA(mat_elems * static_cast<size_t>(batch_count));
    std::vector<T> hTau(tau_elems * static_cast<size_t>(batch_count));
    std::vector<rocblas_int> hInfo(static_cast<size_t>(batch_count));

    std::vector<T*> hBatchPtrs;
    if constexpr(BATCHED)
    {
        hBatchPtrs.resize(static_cast<size_t>(batch_count));
        HIP_CHECK(hipMemcpyAsync(hBatchPtrs.data(),
                                 reinterpret_cast<T* const*>(A),
                                 sizeof(T*) * static_cast<size_t>(batch_count),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    for(I b = 0; b < batch_count; ++b)
    {
        T* dA_item = nullptr;
        if constexpr(BATCHED)
            dA_item = hBatchPtrs[static_cast<size_t>(b)] + shiftA;
        else if constexpr(STRIDED)
            dA_item = reinterpret_cast<T*>(A) + shiftA + b * strideA;
        else
            dA_item = reinterpret_cast<T*>(A) + shiftA;

        HIP_CHECK(hipMemcpyAsync(hA.data() + static_cast<size_t>(b) * mat_elems,
                                 dA_item, mat_bytes, hipMemcpyDeviceToHost, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    for(I b = 0; b < batch_count; ++b)
    {
        T* pA = hA.data() + static_cast<size_t>(b) * mat_elems;
        T* pTau = hTau.data() + static_cast<size_t>(b) * tau_elems;
        run_gehrd<T, I>(n, ilo, ihi, pA, lda, pTau, hInfo[static_cast<size_t>(b)]);
    }

    if constexpr(BATCHED)
    {
        std::vector<T*> ptrs(static_cast<size_t>(batch_count));
        HIP_CHECK(hipMemcpyAsync(ptrs.data(),
                                 reinterpret_cast<T* const*>(A),
                                 sizeof(T*) * static_cast<size_t>(batch_count),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        for(I b = 0; b < batch_count; ++b)
        {
            HIP_CHECK(hipMemcpyAsync(ptrs[static_cast<size_t>(b)] + shiftA,
                                     hA.data() + static_cast<size_t>(b) * mat_elems,
                                     mat_bytes, hipMemcpyHostToDevice, stream));
        }
    }
    else
    {
        for(I b = 0; b < batch_count; ++b)
        {
            HIP_CHECK(hipMemcpyAsync(reinterpret_cast<T*>(A) + shiftA + b * strideA,
                                     hA.data() + static_cast<size_t>(b) * mat_elems,
                                     mat_bytes, hipMemcpyHostToDevice, stream));
        }
    }

    if(tau_elems > 0)
    {
        const size_t tau_bytes = tau_elems * sizeof(T);
        for(I b = 0; b < batch_count; ++b)
        {
            HIP_CHECK(hipMemcpyAsync(tau + static_cast<rocblas_stride>(b) * effStrideTau,
                                     hTau.data() + static_cast<size_t>(b) * tau_elems,
                                     tau_bytes, hipMemcpyHostToDevice, stream));
        }
    }

    for(I b = 0; b < batch_count; ++b)
    {
        HIP_CHECK(hipMemcpyAsync(info + b, &hInfo[static_cast<size_t>(b)],
                                 sizeof(rocblas_int), hipMemcpyHostToDevice, stream));
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
