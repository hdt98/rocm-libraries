/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * *************************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/** thread-block size for calling the sytrs kernels. **/
#define SYTRS_MAX_THDS 256

/************** Device kernels **************/

/** sytrs_bk_pivot_apply_kernel applies the Bunch-Kaufman pivot row
    interchanges to the right-hand side matrix B.
    direction > 0: forward order (k = 0 .. n-1)
    direction < 0: reverse order (k = n-1 .. 0) **/
template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(SYTRS_MAX_THDS)
    sytrs_bk_pivot_apply_kernel(const rocblas_fill uplo,
                                const rocblas_int n,
                                const rocblas_int nrhs,
                                U BB,
                                const rocblas_stride shiftB,
                                const rocblas_int ldb,
                                const rocblas_stride strideB,
                                const rocblas_int* ipivA,
                                const rocblas_stride strideP,
                                const rocblas_int direction)
{
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int tid = hipThreadIdx_x;

    T* B = load_ptr_batch<T>(BB, bid, shiftB, strideB);
    const rocblas_int* ipiv = ipivA + bid * strideP;

    if(uplo == rocblas_fill_upper)
    {
        // LAPACK dsytrs UPPER pivot application (0-indexed, ipiv is 1-indexed):
        // Forward: k from n-1 down to 0. For 2x2 block, swap row k-1 (not k), skip by 2.
        // Reverse: k from 0 up to n-1. For 2x2 block, swap row k-1, skip by 2.
        if(direction > 0)
        {
            // Forward: k = n-1 down to 0
            rocblas_int k = n - 1;
            while(k >= 0)
            {
                if(ipiv[k] > 0)
                {
                    // 1x1 pivot at position k
                    rocblas_int kp = ipiv[k] - 1; // convert to 0-indexed
                    if(kp != k)
                    {
                        for(rocblas_int j = tid; j < nrhs; j += SYTRS_MAX_THDS)
                        {
                            T tmp = B[k + (size_t)j * ldb];
                            B[k + (size_t)j * ldb] = B[kp + (size_t)j * ldb];
                            B[kp + (size_t)j * ldb] = tmp;
                        }
                    }
                    __syncthreads();
                    k -= 1;
                }
                else
                {
                    // 2x2 pivot at positions k-1 and k
                    rocblas_int kp = -ipiv[k] - 1; // convert to 0-indexed
                    if(kp != k - 1)
                    {
                        for(rocblas_int j = tid; j < nrhs; j += SYTRS_MAX_THDS)
                        {
                            T tmp = B[(k - 1) + (size_t)j * ldb];
                            B[(k - 1) + (size_t)j * ldb] = B[kp + (size_t)j * ldb];
                            B[kp + (size_t)j * ldb] = tmp;
                        }
                    }
                    __syncthreads();
                    k -= 2;
                }
            }
        }
        else
        {
            // Reverse: k = 0 up to n-1
            rocblas_int k = 0;
            while(k < n)
            {
                if(ipiv[k] > 0)
                {
                    rocblas_int kp = ipiv[k] - 1;
                    if(kp != k)
                    {
                        for(rocblas_int j = tid; j < nrhs; j += SYTRS_MAX_THDS)
                        {
                            T tmp = B[k + (size_t)j * ldb];
                            B[k + (size_t)j * ldb] = B[kp + (size_t)j * ldb];
                            B[kp + (size_t)j * ldb] = tmp;
                        }
                    }
                    __syncthreads();
                    k += 1;
                }
                else
                {
                    rocblas_int kp = -ipiv[k] - 1;
                    if(kp != k - 1)
                    {
                        for(rocblas_int j = tid; j < nrhs; j += SYTRS_MAX_THDS)
                        {
                            T tmp = B[(k - 1) + (size_t)j * ldb];
                            B[(k - 1) + (size_t)j * ldb] = B[kp + (size_t)j * ldb];
                            B[kp + (size_t)j * ldb] = tmp;
                        }
                    }
                    __syncthreads();
                    k += 2;
                }
            }
        }
    }
    else
    {
        // For lower: process from top to bottom (forward) or bottom to top (reverse)
        rocblas_int kstart = (direction > 0) ? 0 : n - 1;
        rocblas_int kend = (direction > 0) ? n : -1;
        rocblas_int kinc = (direction > 0) ? 1 : -1;

        for(rocblas_int k = kstart; k != kend; k += kinc)
        {
            if(ipiv[k] > 0)
            {
                // 1x1 pivot: ipiv[k] is 1-indexed
                rocblas_int kp = ipiv[k] - 1;
                if(kp != k)
                {
                    for(rocblas_int j = tid; j < nrhs; j += SYTRS_MAX_THDS)
                    {
                        T tmp = B[k + j * ldb];
                        B[k + j * ldb] = B[kp + j * ldb];
                        B[kp + j * ldb] = tmp;
                    }
                }
            }
            else
            {
                // 2x2 pivot block: ipiv[k] < 0 and ipiv[k+1] < 0
                // Only process once (when we hit the first row of the pair)
                if(direction > 0)
                {
                    // forward: process on first row when ipiv[k+1] < 0
                    if(k < n - 1 && ipiv[k + 1] < 0)
                    {
                        rocblas_int kp = -ipiv[k] - 1;
                        if(kp != k)
                        {
                            for(rocblas_int j = tid; j < nrhs; j += SYTRS_MAX_THDS)
                            {
                                T tmp = B[k + j * ldb];
                                B[k + j * ldb] = B[kp + j * ldb];
                                B[kp + j * ldb] = tmp;
                            }
                        }
                    }
                }
                else
                {
                    // reverse: process on first row when ipiv[k+1] < 0
                    if(k < n - 1 && ipiv[k + 1] < 0)
                    {
                        rocblas_int kp = -ipiv[k] - 1;
                        if(kp != k)
                        {
                            for(rocblas_int j = tid; j < nrhs; j += SYTRS_MAX_THDS)
                            {
                                T tmp = B[k + j * ldb];
                                B[k + j * ldb] = B[kp + j * ldb];
                                B[kp + j * ldb] = tmp;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

/** sytrs_bk_diag_solve_kernel solves D * X = B where D is block diagonal
    with 1x1 and 2x2 blocks, as encoded in the factored matrix A and ipiv
    from sytrf. **/
template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(SYTRS_MAX_THDS)
    sytrs_bk_diag_solve_kernel(const rocblas_fill uplo,
                               const rocblas_int n,
                               const rocblas_int nrhs,
                               U AA,
                               const rocblas_stride shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               U BB,
                               const rocblas_stride shiftB,
                               const rocblas_int ldb,
                               const rocblas_stride strideB,
                               const rocblas_int* ipivA,
                               const rocblas_stride strideP)
{
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int tid = hipThreadIdx_x;

    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch<T>(BB, bid, shiftB, strideB);
    const rocblas_int* ipiv = ipivA + bid * strideP;

    rocblas_int k = 0;

    if(uplo == rocblas_fill_upper)
    {
        // LAPACK dsytrs UPPER diagonal solve: process K from N down to 1
        // For 2x2 block: positions K-1 and K (0-indexed: k-1, k)
        k = n - 1;
        while(k >= 0)
        {
            if(ipiv[k] > 0)
            {
                // 1x1 block at position k
                T akk = A[k + (size_t)k * lda];
                for(rocblas_int j = tid; j < nrhs; j += SYTRS_MAX_THDS)
                    B[k + (size_t)j * ldb] = B[k + (size_t)j * ldb] / akk;

                __syncthreads();
                k -= 1;
            }
            else
            {
                // 2x2 block at positions (k-1, k)
                // D = [ A(k-1,k-1)  A(k-1,k) ]
                //     [ A(k-1,k)    A(k,k)   ]
                T d11 = A[(k - 1) + (size_t)(k - 1) * lda];
                T d12 = A[(k - 1) + (size_t)k * lda];
                T d22 = A[k + (size_t)k * lda];

                // Solve 2x2 system via Cramer's rule
                T denom = d11 * d22 - d12 * d12;

                for(rocblas_int j = tid; j < nrhs; j += SYTRS_MAX_THDS)
                {
                    T b1 = B[(k - 1) + (size_t)j * ldb];
                    T b2 = B[k + (size_t)j * ldb];
                    B[(k - 1) + (size_t)j * ldb] = (d22 * b1 - d12 * b2) / denom;
                    B[k + (size_t)j * ldb] = (d11 * b2 - d12 * b1) / denom;
                }

                __syncthreads();
                k -= 2;
            }
        }
    }
    else
    {
        // Lower: process from 0 to n-1
        k = 0;
        while(k < n)
        {
            if(ipiv[k] > 0)
            {
                // 1x1 block at position k
                T akk = A[k + k * lda];
                for(rocblas_int j = tid; j < nrhs; j += SYTRS_MAX_THDS)
                    B[k + j * ldb] = B[k + j * ldb] / akk;

                k += 1;
            }
            else
            {
                // 2x2 block at positions (k, k+1)
                // In lower factorization with negative ipiv[k] and ipiv[k+1],
                // the 2x2 block D is stored at D(k,k), D(k+1,k), D(k+1,k+1)
                T akk = A[k + k * lda];
                T akp1 = A[(k + 1) + k * lda];
                T ak1k1 = A[(k + 1) + (k + 1) * lda];

                // Solve 2x2 system using Cramer's rule
                T denom = akk * ak1k1 - akp1 * akp1;

                for(rocblas_int j = tid; j < nrhs; j += SYTRS_MAX_THDS)
                {
                    T b1 = B[k + j * ldb];
                    T b2 = B[(k + 1) + j * ldb];
                    B[k + j * ldb] = (ak1k1 * b1 - akp1 * b2) / denom;
                    B[(k + 1) + j * ldb] = (akk * b2 - akp1 * b1) / denom;
                }

                k += 2;
            }
            __syncthreads();
        }
    }
}

/************** Argument checking **************/

template <typename T>
rocblas_status rocsolver_sytrs_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nrhs,
                                        const rocblas_int lda,
                                        const rocblas_int ldb,
                                        T A,
                                        T B,
                                        const rocblas_int* ipiv,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || nrhs < 0 || lda < n || ldb < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !ipiv) || (nrhs && n && !B))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/************** Memory size query **************/

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_sytrs_getMemorySize(const rocblas_fill uplo,
                                   const rocblas_int n,
                                   const rocblas_int nrhs,
                                   const rocblas_int batch_count,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   bool* optim_mem)
{
    // if quick return, no workspace is needed
    if(n == 0 || nrhs == 0 || batch_count == 0)
    {
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *optim_mem = true;
        return;
    }

    // workspace required for calling TRSM
    // need workspace for both transpose and non-transpose calls; take maximum
    size_t size_work1_temp1, size_work1_temp2, size_work2_temp1, size_work2_temp2, size_work3_temp1,
        size_work3_temp2, size_work4_temp1, size_work4_temp2;
    rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, rocblas_operation_none, n, nrhs,
                                            batch_count, &size_work1_temp1, &size_work2_temp1,
                                            &size_work3_temp1, &size_work4_temp1, optim_mem);
    rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, rocblas_operation_transpose, n, nrhs,
                                            batch_count, &size_work1_temp2, &size_work2_temp2,
                                            &size_work3_temp2, &size_work4_temp2, optim_mem);

    *size_work1 = std::max(size_work1_temp1, size_work1_temp2);
    *size_work2 = std::max(size_work2_temp1, size_work2_temp2);
    *size_work3 = std::max(size_work3_temp1, size_work3_temp2);
    *size_work4 = std::max(size_work4_temp1, size_work4_temp2);
}

/************** Template function **************/

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_sytrs_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nrhs,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        const rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        U B,
                                        const rocblas_stride shiftB,
                                        const rocblas_int ldb,
                                        const rocblas_stride strideB,
                                        const rocblas_int batch_count,
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        const bool optim_mem)
{
    ROCSOLVER_ENTER("sytrs", "uplo:", uplo, "n:", n, "nrhs:", nrhs, "shiftA:", shiftA, "lda:", lda,
                    "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    // quick return
    if(n == 0 || nrhs == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    dim3 grid(1, batch_count, 1);
    dim3 threads(SYTRS_MAX_THDS, 1, 1);

    if(uplo == rocblas_fill_upper)
    {
        // A = U * D * U^T
        // Solve A * X = B:
        //   1. Apply forward BK pivots to B
        //   2. Solve U^T * Y = P*B  (unit upper triangular, transposed)
        //   3. Solve D * Z = Y      (block diagonal)
        //   4. Solve U * X = Z      (unit upper triangular)
        //   5. Apply reverse BK pivots

        // Step 1: apply forward BK pivots
        ROCSOLVER_LAUNCH_KERNEL((sytrs_bk_pivot_apply_kernel<T>), grid, threads, 0, stream,
                                uplo, n, nrhs, B, shiftB, ldb, strideB, ipiv, strideP, 1);

        // Step 2: solve U * Z = P*B (unit upper triangular, no transpose)
        rocsolver_trsm_upper<BATCHED, STRIDED, T>(
            handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_unit, n, nrhs,
            A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count, optim_mem, work1, work2,
            work3, work4);

        // Step 3: solve D * Y = Z (block diagonal solve)
        ROCSOLVER_LAUNCH_KERNEL((sytrs_bk_diag_solve_kernel<T>), grid, threads, 0, stream,
                                uplo, n, nrhs, A, shiftA, lda, strideA, B, shiftB, ldb, strideB,
                                ipiv, strideP);

        // Step 4: solve U^T * X = Y (unit upper triangular, transpose)
        rocsolver_trsm_upper<BATCHED, STRIDED, T>(
            handle, rocblas_side_left, rocblas_operation_transpose, rocblas_diagonal_unit, n, nrhs, A,
            shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count, optim_mem, work1, work2,
            work3, work4);

        // Step 5: apply reverse BK pivots
        ROCSOLVER_LAUNCH_KERNEL((sytrs_bk_pivot_apply_kernel<T>), grid, threads, 0, stream,
                                uplo, n, nrhs, B, shiftB, ldb, strideB, ipiv, strideP, -1);
    }
    else
    {
        // A = L * D * L^T
        // Solve A * X = B:
        //   1. Apply forward BK pivots to B
        //   2. Solve L * Y = P*B    (unit lower triangular)
        //   3. Solve D * Z = Y      (block diagonal)
        //   4. Solve L^T * X = Z    (unit lower triangular, transposed)
        //   5. Apply reverse BK pivots

        // Step 1: apply forward BK pivots
        ROCSOLVER_LAUNCH_KERNEL((sytrs_bk_pivot_apply_kernel<T>), grid, threads, 0, stream,
                                uplo, n, nrhs, B, shiftB, ldb, strideB, ipiv, strideP, 1);

        // Step 2: solve L * Y = P*B (unit lower triangular)
        rocsolver_trsm_lower<BATCHED, STRIDED, T>(
            handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_unit, n, nrhs, A,
            shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count, optim_mem, work1, work2,
            work3, work4);

        // Step 3: solve D * Z = Y (block diagonal solve)
        ROCSOLVER_LAUNCH_KERNEL((sytrs_bk_diag_solve_kernel<T>), grid, threads, 0, stream,
                                uplo, n, nrhs, A, shiftA, lda, strideA, B, shiftB, ldb, strideB,
                                ipiv, strideP);

        // Step 4: solve L^T * X = Z (unit lower triangular, transposed)
        rocsolver_trsm_lower<BATCHED, STRIDED, T>(
            handle, rocblas_side_left, rocblas_operation_transpose, rocblas_diagonal_unit, n, nrhs,
            A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count, optim_mem, work1, work2,
            work3, work4);

        // Step 5: apply reverse BK pivots
        ROCSOLVER_LAUNCH_KERNEL((sytrs_bk_pivot_apply_kernel<T>), grid, threads, 0, stream,
                                uplo, n, nrhs, B, shiftB, ldb, strideB, ipiv, strideP, -1);
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
