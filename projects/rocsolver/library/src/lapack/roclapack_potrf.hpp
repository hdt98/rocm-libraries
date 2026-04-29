/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "roclapack_potf2.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// -----------------------------------------------
// Note that the recursive routine passes in a
// row_offset to adjust the info value, so the
// iinfo[] array may not be required.
// -----------------------------------------------

// --------------------------------------------------------
// heuristic to determine whether to use recursive routine
// TODO: need further fine tuning
// --------------------------------------------------------
template <typename T, typename I>
inline static bool use_recursion([[maybe_unused]] rocblas_fill const uplo, [[maybe_unused]] I const n)
{
    // simple heuristic
    // bool const is_use_recursion = (n >= 1024 );
    bool const is_use_recursion = true;

    return is_use_recursion;
};

template <typename I>
static inline I split_n(I const n)
{
    return std::max(I(1), n / 2);
}

template <typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void chk_positive(INFO* iinfo, INFO* info, I j, I batch_count)
{
    I id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if((id < batch_count) && (info[id] == 0) && (iinfo[id] > 0))
        info[id] = iinfo[id] + j;
}

/******************* Host functions for potrf **********************/
/*******************************************************************/

// Method to determine configuration for potrf block size depending on n
// and the type
// TODO: may need fine tuning
template <typename T, typename I, typename INFO = I>
static inline I potrf_get_block_size(I const n)
{
    I const nb = std::max(I(POTF2_MAX_SMALL_SIZE(T)), I(POTRF_POTF2_SWITCHSIZE(T)));
    return std::min(n, nb);
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename INFO = I>
void rocsolver_potrf_getMemorySize(const I n,
                                   const rocblas_fill uplo,
                                   const I batch_count,

                                   size_t* p_size_scalars,
                                   size_t* p_size_work1,
                                   size_t* p_size_work2,
                                   size_t* p_size_work3,
                                   size_t* p_size_work4,
                                   size_t* p_size_pivots,
                                   size_t* p_size_iinfo,
                                   bool* optim_mem)
{
    bool const use_recursion_potrf = use_recursion<T>(uplo, n);
    bool const use_iinfo = !use_recursion_potrf;

    *p_size_scalars = 0;
    *p_size_work1 = 0;
    *p_size_work2 = 0;
    *p_size_work3 = 0;
    *p_size_work4 = 0;
    *p_size_pivots = 0;
    *p_size_iinfo = 0;
    *optim_mem = true;

    // if quick return no need of workspace
    if(n == 0 || batch_count == 0)
    {
        return;
    }

    I const nb = potrf_get_block_size<T>(n);
    if(n <= POTRF_POTF2_SWITCHSIZE(T))
    {
        // requirements for calling a single POTF2
        rocsolver_potf2_getMemorySize<T>(n, batch_count, p_size_scalars, p_size_work1, p_size_pivots);
        *p_size_work2 = 0;
        *p_size_work3 = 0;
        *p_size_work4 = 0;
        *p_size_iinfo = 0;
        *optim_mem = true;
    }
    else
    {
        I const jb = nb;

        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;

        // size to store info about positiveness of each subblock
        if(use_iinfo)
        {
            *p_size_iinfo = sizeof(INFO) * batch_count;
        }

        // requirements for calling POTF2 for the subblocks
        {
            size_t lsize_work1 = 0;

            rocsolver_potf2_getMemorySize<T>(jb, batch_count, p_size_scalars, &lsize_work1,
                                             p_size_pivots);

            size_work1 = std::max(size_work1, lsize_work1);
        }

        // extra requirements for calling TRSM
        if(uplo == rocblas_fill_upper)
        {
            if(use_recursion_potrf)
            {
                I const n1 = split_n(n);
                I const n2 = n - n1;

                // ----------------------------------
                // U12 = U11' \ A12, note U12 has size n1 by n2
                // ----------------------------------

                size_t lsize_work1 = 0;
                size_t lsize_work2 = 0;
                size_t lsize_work3 = 0;
                size_t lsize_work4 = 0;

                if((n1 >= 1) && (n2 >= 1))
                {
                    (void)rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                        rocblas_side_left, rocblas_operation_conjugate_transpose, n1, n2, batch_count,

                        &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4, optim_mem);
                }

                size_work1 = std::max(size_work1, lsize_work1);
                size_work2 = std::max(size_work2, lsize_work2);
                size_work3 = std::max(size_work3, lsize_work3);
                size_work4 = std::max(size_work4, lsize_work4);
            }
            else
            {
                size_t lsize_work1 = 0;
                size_t lsize_work2 = 0;
                size_t lsize_work3 = 0;
                size_t lsize_work4 = 0;

                if(((n - jb) >= 1) && (jb >= 1))
                {
                    (void)rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                        rocblas_side_left, rocblas_operation_conjugate_transpose, jb, n - jb,
                        batch_count,

                        &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4, optim_mem);
                }

                size_work1 = std::max(size_work1, lsize_work1);
                size_work2 = std::max(size_work2, lsize_work2);
                size_work3 = std::max(size_work3, lsize_work3);
                size_work4 = std::max(size_work4, lsize_work4);
            }
        }
        else
        {
            if(use_recursion_potrf)
            {
                I const n1 = split_n(n);
                I const n2 = n - n1;

                // ----------------------------------
                // L21 = A21 / L11',  note L21 has size n2 by n1
                // ----------------------------------

                size_t lsize_work1 = 0;
                size_t lsize_work2 = 0;
                size_t lsize_work3 = 0;
                size_t lsize_work4 = 0;

                if((n1 >= 1) && (n2 >= 1))
                {
                    (void)rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                        rocblas_side_right, rocblas_operation_conjugate_transpose, n2, n1,
                        batch_count,

                        &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4, optim_mem);
                }

                size_work1 = std::max(size_work1, lsize_work1);
                size_work2 = std::max(size_work2, lsize_work2);
                size_work3 = std::max(size_work3, lsize_work3);
                size_work4 = std::max(size_work4, lsize_work4);
            }
            else
            {
                size_t lsize_work1 = 0;
                size_t lsize_work2 = 0;
                size_t lsize_work3 = 0;
                size_t lsize_work4 = 0;

                if(((n - jb) >= 1) && (jb >= 1))
                {
                    (void)rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                        rocblas_side_right, rocblas_operation_conjugate_transpose, n - jb, jb,
                        batch_count,

                        &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4, optim_mem);
                }

                size_work1 = std::max(size_work1, lsize_work1);
                size_work2 = std::max(size_work2, lsize_work2);
                size_work3 = std::max(size_work3, lsize_work3);
                size_work4 = std::max(size_work4, lsize_work4);
            }
        }

        *p_size_work1 = std::max(*p_size_work1, size_work1);
        *p_size_work2 = std::max(*p_size_work2, size_work2);
        *p_size_work3 = std::max(*p_size_work3, size_work3);
        *p_size_work4 = std::max(*p_size_work4, size_work4);
    }
}

// --------------------------------------------------------------
// This is the classic block algorithm for Cholesky factorization
// using a fixed size block size
// --------------------------------------------------------------
template <bool BATCHED, bool STRIDED, typename T, typename I, typename INFO, typename S, typename U>
rocblas_status rocsolver_potrf_template_body(rocblas_handle handle,
                                             const rocblas_fill uplo,
                                             const I n,
                                             U A,
                                             const rocblas_stride shiftA,
                                             const I lda,
                                             const rocblas_stride strideA,
                                             INFO* info,
                                             const I batch_count,
                                             T* scalars,
                                             void* work1,
                                             void* work2,
                                             void* work3,
                                             void* work4,
                                             T* pivots,
                                             INFO* iinfo,
                                             bool optim_mem,
                                             const I row_offset_in = 0,
                                             bool const use_iinfo = true)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // if the matrix is small, use the unblocked (BLAS-levelII) variant of the
    // algorithm
    I nb = potrf_get_block_size<T>(n);
    if(n <= POTRF_POTF2_SWITCHSIZE(T))
    {
        I const row_offset = 0 + row_offset_in;
        auto const istat
            = rocsolver_potf2_template<T>(handle, uplo, n, A, shiftA, lda, strideA, info,
                                          batch_count, scalars, (T*)work1, pivots, row_offset);
        return istat;
    }

    // constants for rocblas functions calls
    S s_one = 1;
    S s_minone = -1;

    // (TODO: When the matrix is detected to be non positive definite, we need to
    //  prevent TRSM and HERK to modify further the input matrix; ideally with no
    //  synchronizations.)

    I j = 0; // note setting j = 0 is important for correctness

    if(uplo == rocblas_fill_upper)
    {
        // Compute the Cholesky factorization A = U'*U.
        while(j < n - POTRF_POTF2_SWITCHSIZE(T))
        {
            // Factor diagonal and subdiagonal blocks
            I const jb = std::min(n - j, nb); // number of columns in the block
            if(use_iinfo)
            {
                ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo,
                                        batch_count, 0);
                auto const istat = rocsolver_potf2_template<T>(
                    handle, uplo, jb, A, shiftA + idx2D(j, j, lda), lda, strideA, iinfo,
                    batch_count, scalars, (T*)work1, pivots);
                if(istat != rocblas_status_success)
                {
                    return istat;
                }

                // test for non-positive-definiteness.
                ROCSOLVER_LAUNCH_KERNEL((chk_positive<I, INFO, U>), gridReset, threads, 0, stream,
                                        iinfo, info, j + row_offset_in, batch_count);
            }
            else
            {
                I const row_offset = j + row_offset_in;
                auto const istat = rocsolver_potf2_template<T>(
                    handle, uplo, jb, A, shiftA + idx2D(j, j, lda), lda, strideA, info, batch_count,
                    scalars, (T*)work1, pivots, row_offset);
                if(istat != rocblas_status_success)
                {
                    return istat;
                }
            }

            if(j + jb < n)
            {
                // update trailing submatrix
                auto const istat_trsm_upper = rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                    handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, jb, (n - j - jb), A, shiftA + idx2D(j, j, lda), lda,
                    strideA, A, shiftA + idx2D(j, j + jb, lda), lda, strideA, batch_count,
                    optim_mem, work1, work2, work3, work4);
                if(istat_trsm_upper != rocblas_status_success)
                {
                    return istat_trsm_upper;
                }

                auto const istat_syrk = rocblasCall_syrk_herk<BATCHED, T>(
                    handle, uplo, rocblas_operation_conjugate_transpose, n - j - jb, jb, &s_minone,
                    A, shiftA + idx2D(j, j + jb, lda), lda, strideA, &s_one, A,
                    shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count);
                if(istat_syrk != rocblas_status_success)
                {
                    return istat_syrk;
                }
            }
            j += nb;
        }
    }
    else
    {
        // Compute the Cholesky factorization A = L*L'.
        while(j < n - POTRF_POTF2_SWITCHSIZE(T))
        {
            // Factor diagonal and subdiagonal blocks
            I const jb = std::min(n - j, nb); // number of columns in the block
            if(use_iinfo)
            {
                ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo,
                                        batch_count, 0);
                auto const istat = rocsolver_potf2_template<T>(
                    handle, uplo, jb, A, shiftA + idx2D(j, j, lda), lda, strideA, iinfo,
                    batch_count, scalars, (T*)work1, pivots);
                if(istat != rocblas_status_success)
                {
                    return istat;
                }

                // test for non-positive-definiteness.
                ROCSOLVER_LAUNCH_KERNEL((chk_positive<I, INFO, U>), gridReset, threads, 0, stream,
                                        iinfo, info, j + row_offset_in, batch_count);
            }
            else
            {
                I const row_offset = j + row_offset_in;
                auto const istat = rocsolver_potf2_template<T>(
                    handle, uplo, jb, A, shiftA + idx2D(j, j, lda), lda, strideA, info, batch_count,
                    scalars, (T*)work1, pivots, row_offset);
                if(istat != rocblas_status_success)
                {
                    return istat;
                }
            }

            if(j + jb < n)
            {
                // update trailing submatrix
                auto const istat_trsm_lower = rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                    handle, rocblas_side_right, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, (n - j - jb), jb, A, shiftA + idx2D(j, j, lda), lda,
                    strideA, A, shiftA + idx2D(j + jb, j, lda), lda, strideA, batch_count,
                    optim_mem, work1, work2, work3, work4);

                if(istat_trsm_lower != rocblas_status_success)
                {
                    return istat_trsm_lower;
                }

                auto const istat_syrk = rocblasCall_syrk_herk<BATCHED, T>(
                    handle, uplo, rocblas_operation_none, n - j - jb, jb, &s_minone, A,
                    shiftA + idx2D(j + jb, j, lda), lda, strideA, &s_one, A,
                    shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count);

                if(istat_syrk != rocblas_status_success)
                {
                    return istat_syrk;
                }
            }
            j += nb;
        }
    }

    // factor last block
    if(j < n)
    {
        if(use_iinfo)
        {
            auto const istat = rocsolver_potf2_template<T>(
                handle, uplo, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA, iinfo, batch_count,
                scalars, (T*)work1, pivots);

            if(istat != rocblas_status_success)
            {
                return istat;
            }

            ROCSOLVER_LAUNCH_KERNEL((chk_positive<I, INFO, U>), gridReset, threads, 0, stream,
                                    iinfo, info, j + row_offset_in, batch_count);
        }
        else
        {
            I const row_offset = j + row_offset_in;
            auto const istat = rocsolver_potf2_template<T>(
                handle, uplo, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA, info, batch_count,
                scalars, (T*)work1, pivots, row_offset);

            if(istat != rocblas_status_success)
            {
                return istat;
            }
        }
    }

    return rocblas_status_success;
}

// ---------------------------------------------------------------
// This is the recursive formulation of the Cholesky factorization
//
// There is early termination of recursion if the matrix is sufficiently small
// and switches to the classic blocked version
// ---------------------------------------------------------------
template <bool BATCHED, bool STRIDED, typename T, typename I, typename INFO, typename S, typename U>
rocblas_status rocsolver_potrf_recursion_template(rocblas_handle handle,
                                                  const rocblas_fill uplo,
                                                  const I n,
                                                  U A,
                                                  const rocblas_stride shiftA,
                                                  const I lda,
                                                  const rocblas_stride strideA,
                                                  INFO* info,
                                                  const I batch_count,
                                                  T* scalars,
                                                  void* work1,
                                                  void* work2,
                                                  void* work3,
                                                  void* work4,
                                                  bool optim_mem,
                                                  const I row_offset_in)
{
    I const NB = POTF2_MAX_SMALL_SIZE(T);
    if(n <= NB)
    {
        // -------------------
        // terminate recursion
        // -------------------

        // ----------------------
        // use specialized kernel
        // ----------------------

        auto const istat = potf2_run_small<T>(handle, uplo, n,

                                              A, shiftA, lda, strideA,

                                              info, batch_count, row_offset_in);

        return istat;
    }

    // ------------------------------------
    // constants for rocblas functions calls
    // ------------------------------------
    S s_one = 1;
    S s_minone = -1;

    bool const use_upper_part = (uplo == rocblas_fill_upper);
    I const n1 = split_n(n);
    I const n2 = n - n1;

    if(use_upper_part)
    {
        // ----------------------------------------------------
        // use upper triangular part
        //
        // [A11   A12 ] =  [ U11'        ] * [ U11   U12 ]
        // [A12'  A22 ]    [ U12'   U22' ]   [       U22 ]
        //
        // step 1:   A11 = U11' * U11,   cholesky factorization
        // step 2:   A12 = U11' * U12, or   U12 = U11' \ A12,  trsm
        // step 3a:  A22 = A22 - U12' * U12,   syrk
        // step 3b:  A22 = U22' * U22,  cholesky factorization
        // ----------------------------------------------------

        // -------------------------------------------------
        // step 1: A11 =  U11' * U11,   cholesky factorization
        // -------------------------------------------------
        {
            // --------------------
            // note A11 is n1 by n1
            // --------------------
            I const row_offset = 0 + row_offset_in;

            auto const istat = rocsolver_potrf_recursion_template<BATCHED, STRIDED, T, I, INFO, S, U>(
                handle, uplo, n1,

                A, shiftA + idx2D(0, 0, lda), lda, strideA, // submatrix A11

                info, batch_count,

                scalars, work1, work2, work3, work4, optim_mem,

                row_offset);
            if(istat != rocblas_status_success)
            {
                return istat;
            };
        }

        // -------------------------------------------------------
        // step 2: A12 =  U11' * U12, or U12 = U11' \ A12
        //
        // note U12 has size   n1 by n2
        // -------------------------------------------------------
        {
            I const mm = n1;
            I const nn = n2;
            auto const istat = rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                rocblas_diagonal_non_unit, mm, nn,

                A, shiftA + idx2D(0, 0, lda), lda, strideA, // submatrix U11

                A, shiftA + idx2D(0, n1, lda), lda, strideA, // submatrix U12

                batch_count, optim_mem, work1, work2, work3, work4);

            if(istat != rocblas_status_success)
            {
                return istat;
            }
        }

        // --------------------------------
        // step 3a:  A22 = A22 - U12' * U12
        // --------------------------------
        {
            I const nn = n2;
            I const kk = n1;
            auto const istat = rocblasCall_syrk_herk<BATCHED, T>(
                handle, uplo, rocblas_operation_conjugate_transpose,

                nn, kk,

                &s_minone,

                A, shiftA + idx2D(0, n1, lda), lda, strideA, // submatrix U12

                &s_one,

                A, shiftA + idx2D(n1, n1, lda), lda, strideA, // submatrix A22

                batch_count);
            if(istat != rocblas_status_success)
            {
                return istat;
            }
        }

        // ---------------------------------------------------
        // step 3b:  A22 = U22' * U22,  cholesky factorization
        // ---------------------------------------------------
        {
            I const row_offset = n1 + row_offset_in;
            auto const istat = rocsolver_potrf_recursion_template<BATCHED, STRIDED, T, I, INFO, S, U>(

                handle, uplo, n2,

                A, shiftA + idx2D(n1, n1, lda), lda, strideA, // submatrix A22

                info, batch_count,

                scalars, work1, work2, work3, work4, optim_mem,

                row_offset);
            if(istat != rocblas_status_success)
            {
                return istat;
            }
        }
    }
    else
    {
        // ------------------------------------------------
        // Use lower triangular part
        //
        // [A11   A21'] = [L11      ] * [L11'   L21' ]
        // [A21   A22 ]   [L21  L22 ]   [       L22' ]
        //
        // step 1:   A11 = L11 * L11',   cholesky factorization
        // step 2:   A21 = L21 * L11', or   L21 = A21 / L11',  trsm
        // step 3a:  A22 <-  A22 - L21 * L21',   syrk
        // step 3b:  A22 = L22 * L22', cholesky factorization
        // ------------------------------------------------

        // ---------------------------------------------------
        // step 1: A11 = L11 * L11',   cholesky factorization
        //
        // note A11 has size n1 by n1
        // ---------------------------------------------------
        {
            I const row_offset = 0 + row_offset_in;
            auto const istat = rocsolver_potrf_recursion_template<BATCHED, STRIDED, T, I, INFO, S, U>(

                handle, uplo, n1,

                A, shiftA + idx2D(0, 0, lda), lda, strideA, // submatrix A11

                info, batch_count,

                scalars, work1, work2, work3, work4, optim_mem,

                row_offset);
            if(istat != rocblas_status_success)
            {
                return istat;
            }
        }

        // --------------------------------------------------------
        // step 2:  A21 = L21 * L11', or   L21 = A21 / L11',  trsm
        //
        // note L21 has size n2 by n1
        // --------------------------------------------------------
        {
            I const mm = n2;
            I const nn = n1;
            auto const istat = rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                handle, rocblas_side_right, rocblas_operation_conjugate_transpose,
                rocblas_diagonal_non_unit, mm, nn,

                A, shiftA + idx2D(0, 0, lda), lda, strideA, // submatrix L11

                A, shiftA + idx2D(n1, 0, lda), lda, strideA, // submatrix L21

                batch_count,

                optim_mem, work1, work2, work3, work4);

            if(istat != rocblas_status_success)
            {
                return istat;
            }
        }

        // ------------------------------------------
        // step 3a:  A22 <-  A22 - L21 * L21',   syrk
        // ------------------------------------------
        {
            I const nn = n2;
            I const kk = n1;
            auto const istat = rocblasCall_syrk_herk<BATCHED, T>(
                handle, uplo, rocblas_operation_none,

                nn, kk,

                &s_minone,

                A, shiftA + idx2D(n1, 0, lda), lda, strideA, // submatrix L21

                &s_one,

                A, shiftA + idx2D(n1, n1, lda), lda, strideA, // submatrix A22

                batch_count);
            if(istat != rocblas_status_success)
            {
                return istat;
            }
        }

        // --------------------------------------------------
        // step 3b:  A22 = L22 * L22', cholesky factorization
        //
        // note A22 has size n2 by n2
        // --------------------------------------------------
        {
            I const row_offset = n1 + row_offset_in;
            auto const istat = rocsolver_potrf_recursion_template<BATCHED, STRIDED, T, I, INFO, S, U>(

                handle, uplo, n2,

                A, shiftA + idx2D(n1, n1, lda), lda, strideA, // submatrix A22

                info, batch_count,

                scalars, work1, work2, work3, work4, optim_mem,

                row_offset);
            if(istat != rocblas_status_success)
            {
                return istat;
            }
        }
    }

    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename INFO, typename S, typename U>
rocblas_status rocsolver_potrf_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        INFO* info,
                                        const I batch_count,
                                        T* scalars,
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        T* pivots,
                                        INFO* iinfo,
                                        bool optim_mem)
{
    ROCSOLVER_ENTER("potrf", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);
    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocksReset = std::max(I(1), (batch_count - 1) / BS1 + 1);
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info=0 (starting with a positive definite matrix)
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 0)
    {
        return rocblas_status_success;
    }

    bool const use_recursion_potrf = use_recursion<T>(uplo, n);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    rocblas_status istat = rocblas_status_success;
    if(use_recursion_potrf)
    {
        I const row_offset = 0;
        istat = rocsolver_potrf_recursion_template<BATCHED, STRIDED, T, I, INFO, S, U>(
            handle, uplo, n,

            A, shiftA, lda, strideA,

            info, batch_count,

            scalars, work1, work2, work3, work4, optim_mem, row_offset);
    }
    else
    {
        istat = rocsolver_potrf_template_body<BATCHED, STRIDED, T, I, INFO, S, U>(
            handle, uplo, n,

            A, shiftA, lda, strideA,

            info, batch_count,

            scalars, work1, work2, work3, work4, pivots, iinfo, optim_mem);
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return istat;
}

ROCSOLVER_END_NAMESPACE
