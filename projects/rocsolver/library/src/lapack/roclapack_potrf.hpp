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

static bool constexpr use_syrk = false;
static bool constexpr use_rocblas_trsm = true;

template <typename T, typename I, typename Istride, typename UA>
__global__ void copy_strictly_triangular_kernel(bool const is_strictly_lower,
                                                bool const is_restore,
                                                I const n,

                                                UA A,
                                                Istride const shiftA,
                                                I const lda,
                                                Istride const strideA,

                                                T* const save_A,
                                                I const batch_count)
{
    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const j_start = threadIdx.y + blockIdx.y * blockDim.y;
    I const j_inc = gridDim.y * blockDim.y;

    I const i_start = threadIdx.x + blockIdx.x + blockDim.x;
    I const i_inc = gridDim.x * blockDim.x;

    auto idx_lower = [=](I const i, I const j) {
        auto const ipos
            = ((static_cast<int64_t>(j) * n - static_cast<int64_t>(j) * (j + 1) / 2) + (i - j - 1));
        return (ipos);
    };

    auto idx_upper = [=](I const i, I const j) {
        auto ipos = (static_cast<int64_t>(j) * (j - 1) / 2 + i);
        return (ipos);
    };

    auto idx2D = [](auto i, auto j, auto lda) { return (i + j * static_cast<int64_t>(lda)); };

    for(I bid = 0 + bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const A_bid = load_ptr_batch<T>(A, bid, shiftA, strideA);

        size_t const size_triangle = size_t(n) * (n - 1) / 2;
        T* const save_A_bid = save_A + size_triangle * bid;

        for(I j = 0 + j_start; j < n; j += j_inc)
        {
            I const row_start = (is_strictly_lower) ? (j + 1) : 0;
            I const row_end = (is_strictly_lower) ? (n - 1) : (j - 1);

            for(I i = row_start + i_start; i <= row_end; i += i_inc)
            {
                auto const ipos = (is_strictly_lower) ? idx_lower(i, j) : idx_upper(i, j);
                auto const ij = idx2D(i, j, lda);

                if(is_restore)
                {
                    A_bid[ij] = save_A_bid[ipos];
                }
                else
                {
                    save_A_bid[ipos] = A_bid[ij];
                }
            } // end for i
        } // end for j
    } // end for bid
}

template <typename T, typename I, typename Istride, typename UA>
static inline void copy_strictly_triangular(bool const is_strictly_lower,
                                            bool const is_restore,
                                            I const n,

                                            UA A,
                                            Istride const shiftA,
                                            I const lda,
                                            Istride const strideA,

                                            T* const save_A,
                                            I const batch_count)
{
    auto ceildiv = [](auto n, auto nx) { return ((n <= 0) ? 0 : (n + nx - 1) / nx); };

    I const nx = 64;
    I const ny = 4;
    I const nz = 1;

    I const max_blocks = 64 * 1024 - 3;
    I const nbz = std::min(max_blocks, batch_count);
    I const nbx = std::min(max_blocks, ceildiv(n, nx));
    I const nby = std::min(max_blocks, ceildiv(n, ny));

    copy_strictly_triangular_kernel<T, I, Istride, UA>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, nz)>>>(is_strictly_lower, is_restore, n,

                                                    A, shiftA, lda, strideA,

                                                    save_A, batch_count);
}

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
static inline bool use_recursion([[maybe_unused]] rocblas_fill const uplo, [[maybe_unused]] I const n)
{
    // simple heuristic
    // bool const is_use_recursion = (n >= 1024 );
    bool const is_use_recursion = true;

    return is_use_recursion;
};

template <typename T, typename I>
static inline I split_n(I const n)
{
    // return std::max(I(1), n / 2);
    I const nb = POTF2_MAX_SMALL_SIZE(T);
    I const n1 = (n <= nb)       ? std::max(I(1), n / 2)
        : (n <= 2 * nb)          ? nb
        : (n <= 4 * nb)          ? 2 * nb
        : (n <= 8 * nb)          ? 4 * nb
        : (n <= 16 * nb)         ? 8 * nb
        : (n <= 32 * nb)         ? 16 * nb
        : (n <= 64 * nb)         ? 32 * nb
        : (n <= 128 * nb)        ? 64 * nb
        : (n <= 256 * nb)        ? 128 * nb
        : (n <= 512 * nb)        ? 256 * nb
        : (n <= 1024 * nb)       ? 512 * nb
        : (n <= 2 * 1024 * nb)   ? 1024 * nb
        : (n <= 4 * 1024 * nb)   ? 2 * 1024 * nb
        : (n <= 8 * 1024 * nb)   ? 4 * 1024 * nb
        : (n <= 16 * 1024 * nb)  ? 8 * 1024 * nb
        : (n <= 32 * 1024 * nb)  ? 16 * 1024 * nb
        : (n <= 64 * 1024 * nb)  ? 32 * 1024 * nb
        : (n <= 128 * 1024 * nb) ? 64 * 1024 * nb
        : (n <= 256 * 1024 * nb) ? 128 * 1024 * nb
                                 : 128 * 1024 * nb;

    return n1;
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
void rocsolver_potrf_recursion_getMemorySize(const I n,
                                             const rocblas_fill uplo,
                                             const I batch_count,

                                             size_t* p_size_work1,
                                             size_t* p_size_work2,
                                             size_t* p_size_work3,
                                             size_t* p_size_work4,
                                             bool* optim_mem)
{
    size_t size_work1 = 0;
    size_t size_work2 = 0;
    size_t size_work3 = 0;
    size_t size_work4 = 0;

    auto is_po2_NB = [](auto const n, auto const NB) -> bool {
        // --------------------------
        // check whether n = 2^k * NB
        // --------------------------

        auto is_power_of_2 = [](auto const n) -> bool {
            // ---------
            // check whether n is an exact power of 2,
            // i.e. n = 2^k   k >= 0
            // ---------
            return (n > 0) && ((n & (n - 1)) == 0);
        };

        return ((n % NB) == 0) && is_power_of_2(n / NB);
    };

    I const NB = POTF2_MAX_SMALL_SIZE(T);

    if(n <= NB)
    {
        // terminate recursion
    }
    else if(is_po2_NB(n, NB))
    {
        // --------------------------
        // special case  n = 2^k * NB
        // --------------------------
        I const n1 = n / 2;
        I const n2 = n - n1;
        if(uplo == rocblas_fill_upper)
        {
            // ----------------------------------
            // U12 = U11' \ A12, note U12 has size n1 by n2
            // ----------------------------------
            size_t lsize_work1 = 0;
            size_t lsize_work2 = 0;
            size_t lsize_work3 = 0;
            size_t lsize_work4 = 0;

            if(use_rocblas_trsm)
            {
                // -------------------------------------------------------
                // TODO: check whether setting ld1 = 1, ld2 = 1 is correct
                // -------------------------------------------------------
                I const ld1 = 1;
                I const ld2 = 1;
                (void)rocblasCall_trsm_mem<BATCHED, T, I>(
                    rocblas_side_left, rocblas_operation_conjugate_transpose,

                    n1, n2, ld1, ld2, batch_count,

                    &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4);
            }
            else
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
            // ----------------------------------
            // L21 = A21 / L11',  note L21 has size n2 by n1
            // ----------------------------------

            size_t lsize_work1 = 0;
            size_t lsize_work2 = 0;
            size_t lsize_work3 = 0;
            size_t lsize_work4 = 0;

            if(use_rocblas_trsm)
            {
                // -------------------------------------------------------
                // TODO: check whether setting ld1 = 1, ld2 = 1 is correct
                // -------------------------------------------------------
                I const ld1 = 1;
                I const ld2 = 1;
                (void)rocblasCall_trsm_mem<BATCHED, T, I>(
                    rocblas_side_right, rocblas_operation_conjugate_transpose, n2, n1, ld1, ld2,
                    batch_count,

                    &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4);
            }
            else
            {
                (void)rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                    rocblas_side_right, rocblas_operation_conjugate_transpose, n2, n1, batch_count,

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
        // ------------
        // general case
        // ------------
        I const n1 = split_n<T>(n);
        I const n2 = n - n1;

        if(uplo == rocblas_fill_upper)
        {
            //  ----------------------------------------------
            //  Cholesky factorization of A11 of size n1 by n1
            //  ----------------------------------------------
            {
                size_t lsize_work1 = 0;
                size_t lsize_work2 = 0;
                size_t lsize_work3 = 0;
                size_t lsize_work4 = 0;

                rocsolver_potrf_recursion_getMemorySize<BATCHED, STRIDED, T, I, INFO>(
                    n1, uplo, batch_count,

                    &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4, optim_mem);

                size_work1 = std::max(size_work1, lsize_work1);
                size_work2 = std::max(size_work2, lsize_work2);
                size_work3 = std::max(size_work3, lsize_work3);
                size_work4 = std::max(size_work4, lsize_work4);
            }

            // ----------------------------------
            // U12 = U11' \ A12, note U12 has size n1 by n2
            // ----------------------------------
            {
                size_t lsize_work1 = 0;
                size_t lsize_work2 = 0;
                size_t lsize_work3 = 0;
                size_t lsize_work4 = 0;

                if(use_rocblas_trsm)
                {
                    // -------------------------------------------------------
                    // TODO: check whether setting ld1 = 1, ld2 = 1 is correct
                    // -------------------------------------------------------
                    I const ld1 = 1;
                    I const ld2 = 1;
                    (void)rocblasCall_trsm_mem<BATCHED, T, I>(
                        rocblas_side_left, rocblas_operation_conjugate_transpose,

                        n1, n2, ld1, ld2, batch_count,

                        &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4);
                }
                else
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

            //  ----------------------------------------------
            //  Cholesky factorization of A22 of size n2 by n2
            //  ----------------------------------------------
            {
                size_t lsize_work1 = 0;
                size_t lsize_work2 = 0;
                size_t lsize_work3 = 0;
                size_t lsize_work4 = 0;

                rocsolver_potrf_recursion_getMemorySize<BATCHED, STRIDED, T, I, INFO>(
                    n2, uplo, batch_count,

                    &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4, optim_mem);

                size_work1 = std::max(size_work1, lsize_work1);
                size_work2 = std::max(size_work2, lsize_work2);
                size_work3 = std::max(size_work3, lsize_work3);
                size_work4 = std::max(size_work4, lsize_work4);
            }
        }
        else
        {
            //  ----------------------------------------------
            //  Cholesky factorization of A11 of size n1 by n1
            //  ----------------------------------------------
            {
                size_t lsize_work1 = 0;
                size_t lsize_work2 = 0;
                size_t lsize_work3 = 0;
                size_t lsize_work4 = 0;

                rocsolver_potrf_recursion_getMemorySize<BATCHED, STRIDED, T, I, INFO>(
                    n1, uplo, batch_count,

                    &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4, optim_mem);

                size_work1 = std::max(size_work1, lsize_work1);
                size_work2 = std::max(size_work2, lsize_work2);
                size_work3 = std::max(size_work3, lsize_work3);
                size_work4 = std::max(size_work4, lsize_work4);
            }

            // ----------------------------------
            // L21 = A21 / L11',  note L21 has size n2 by n1
            // ----------------------------------

            size_t lsize_work1 = 0;
            size_t lsize_work2 = 0;
            size_t lsize_work3 = 0;
            size_t lsize_work4 = 0;

            if(use_rocblas_trsm)
            {
                // -------------------------------------------------------
                // TODO: check whether setting ld1 = 1, ld2 = 1 is correct
                // -------------------------------------------------------
                I const ld1 = 1;
                I const ld2 = 1;
                (void)rocblasCall_trsm_mem<BATCHED, T, I>(
                    rocblas_side_right, rocblas_operation_conjugate_transpose, n2, n1, ld1, ld2,
                    batch_count,

                    &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4);
            }
            else
            {
                (void)rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                    rocblas_side_right, rocblas_operation_conjugate_transpose, n2, n1, batch_count,

                    &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4, optim_mem);
            }

            size_work1 = std::max(size_work1, lsize_work1);
            size_work2 = std::max(size_work2, lsize_work2);
            size_work3 = std::max(size_work3, lsize_work3);
            size_work4 = std::max(size_work4, lsize_work4);

            //  ----------------------------------------------
            //  Cholesky factorization of A22 of size n2 by n2
            //  ----------------------------------------------
            {
                size_t lsize_work1 = 0;
                size_t lsize_work2 = 0;
                size_t lsize_work3 = 0;
                size_t lsize_work4 = 0;

                rocsolver_potrf_recursion_getMemorySize<BATCHED, STRIDED, T, I, INFO>(
                    n2, uplo, batch_count,

                    &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4, optim_mem);

                size_work1 = std::max(size_work1, lsize_work1);
                size_work2 = std::max(size_work2, lsize_work2);
                size_work3 = std::max(size_work3, lsize_work3);
                size_work4 = std::max(size_work4, lsize_work4);
            }
        }
    }

    *p_size_work1 = std::max(*p_size_work1, size_work1);
    *p_size_work2 = std::max(*p_size_work2, size_work2);
    *p_size_work3 = std::max(*p_size_work3, size_work3);
    *p_size_work4 = std::max(*p_size_work4, size_work4);
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

    // ----------------------------------------------------
    // NOTE: set size_scalars even if using host_mode
    // or scalars[] not used since
    // some routines may assume size_scalars to be non-zero
    // ----------------------------------------------------
    size_t size_scalars = sizeof(T) * 3;

    size_t size_work1 = 0;
    size_t size_work2 = 0;
    size_t size_work3 = 0;
    size_t size_work4 = 0;
    size_t size_pivots = 0;
    size_t size_iinfo = sizeof(INFO) * batch_count;

    if(use_recursion_potrf)
    {
        size_t lsize_work1 = 0;
        size_t lsize_work2 = 0;
        size_t lsize_work3 = 0;
        size_t lsize_work4 = 0;

        rocsolver_potrf_recursion_getMemorySize<BATCHED, STRIDED, T, I, INFO>(
            n, uplo, batch_count,

            &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4, optim_mem);

        bool const use_gemm = !use_syrk;
        if(use_gemm)
        {
            // ----------------------------------------------------
            // reuse storage in work1 to
            // save the strictly lower or upper triangular part in work1
            // ----------------------------------------------------
            size_t const size_save_A = sizeof(T) * (size_t(n) * (n - 1) / 2) * batch_count;
            lsize_work1 += size_save_A;
        }

        size_work1 = std::max(size_work1, lsize_work1);
        size_work2 = std::max(size_work2, lsize_work2);
        size_work3 = std::max(size_work3, lsize_work3);
        size_work4 = std::max(size_work4, lsize_work4);

        *p_size_scalars = std::max(*p_size_scalars, size_scalars);

        *p_size_work1 = std::max(*p_size_work1, size_work1);
        *p_size_work2 = std::max(*p_size_work2, size_work2);
        *p_size_work3 = std::max(*p_size_work3, size_work3);
        *p_size_work4 = std::max(*p_size_work4, size_work4);

        *p_size_pivots = std::max(*p_size_pivots, size_pivots);
        *p_size_iinfo = std::max(*p_size_iinfo, size_iinfo);

        return;
    }

    I const nb = potrf_get_block_size<T>(n);
    if(n <= POTRF_POTF2_SWITCHSIZE(T))
    {
        // requirements for calling a single POTF2
        size_t lsize_scalars = 0;
        size_t lsize_work1 = 0;
        size_t lsize_pivots = 0;

        rocsolver_potf2_getMemorySize<T>(n, batch_count, &lsize_scalars, &lsize_work1, &lsize_pivots);

        size_scalars = std::max(size_scalars, lsize_scalars);
        size_work1 = std::max(size_work1, lsize_work1);
        size_pivots = std::max(size_pivots, lsize_pivots);
    }
    else
    {
        I const jb = nb;

        // size to store info about positiveness of each subblock
        if(use_iinfo)
        {
            size_t lsize_iinfo = sizeof(INFO) * batch_count;
            size_iinfo = std::max(size_iinfo, lsize_iinfo);
        }

        // requirements for calling POTF2 for the subblocks
        {
            size_t lsize_scalars = 0;
            size_t lsize_work1 = 0;
            size_t lsize_pivots = 0;

            rocsolver_potf2_getMemorySize<T>(jb, batch_count, &lsize_scalars, &lsize_work1,
                                             &lsize_pivots);

            size_scalars = std::max(size_scalars, lsize_scalars);
            size_work1 = std::max(size_work1, lsize_work1);
            size_pivots = std::max(size_pivots, lsize_pivots);
        }

        // extra requirements for calling TRSM
        if(uplo == rocblas_fill_upper)
        {
            size_t lsize_work1 = 0;
            size_t lsize_work2 = 0;
            size_t lsize_work3 = 0;
            size_t lsize_work4 = 0;

            if(((n - jb) >= 1) && (jb >= 1))
            {
                (void)rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                    rocblas_side_left, rocblas_operation_conjugate_transpose, jb, n - jb, batch_count,

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

    *p_size_scalars = std::max(*p_size_scalars, size_scalars);

    *p_size_work1 = std::max(*p_size_work1, size_work1);
    *p_size_work2 = std::max(*p_size_work2, size_work2);
    *p_size_work3 = std::max(*p_size_work3, size_work3);
    *p_size_work4 = std::max(*p_size_work4, size_work4);

    *p_size_pivots = std::max(*p_size_pivots, size_pivots);
    *p_size_iinfo = std::max(*p_size_iinfo, size_iinfo);
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
    bool const use_upper_part = (uplo == rocblas_fill_upper);
    // ------------------------------------
    // constants for rocblas functions calls
    // ------------------------------------
    S s_one = 1;
    S s_minone = -1;

    auto idx2D = [](I const i, I const j, I const ld) { return (i + j * static_cast<int64_t>(ld)); };

    I const NB = POTF2_MAX_SMALL_SIZE(T);
    if(n <= NB)
    {
        // -------------------
        // terminate recursion
        // -------------------

        // ----------------------
        // use specialized kernel
        // ----------------------

        rocblas_status const istat = potf2_run_small<T>(handle, uplo, n,

                                                        A, shiftA, lda, strideA,

                                                        info, batch_count, row_offset_in);

        return istat;
    }

    auto potrf_syrk
        = [=](rocblas_fill const uplo,

              rocblas_operation const trans,

              I const n,

              I const k,

              S* alpha,

              auto A, rocblas_stride const shiftA, I const lda, rocblas_stride const strideA,

              S* beta,

              auto C, rocblas_stride const shiftC, I const ldc, rocblas_stride const strideC

              ) -> rocblas_status {
        rocblas_status istat = rocblas_status_success;

        if(use_syrk)
        {
            istat = rocblasCall_syrk_herk<BATCHED, T>(handle, uplo, trans,

                                                      n, k,

                                                      alpha,

                                                      A, shiftA, lda, strideA,

                                                      beta,

                                                      C, shiftC, ldc, strideC,

                                                      batch_count);
        }
        else
        {
            I const mm = n;
            I const nn = n;
            I const kk = k;

            T** work = nullptr;

            I const inc1 = 1;
            I const inc2 = 1;
            I const inc3 = 1;

            T lalpha = *alpha;
            T lbeta = *beta;

            // ------------------------------------------------------------------
            // C <- alpha * A * A' + beta * C,   trans == rocblas_operation_none
            // C <- alpha * A' * A + beta * C,   otherwise
            //
            // C is n by n
            // ------------------------------------------------------------------
            bool const is_no_trans = (trans == rocblas_operation_none);

            rocblas_operation trans1
                = (is_no_trans) ? rocblas_operation_none : rocblas_operation_conjugate_transpose;
            rocblas_operation trans2
                = (is_no_trans) ? rocblas_operation_conjugate_transpose : rocblas_operation_none;

            istat = rocsolver_gemm(

                handle, trans1, trans2, mm, nn, kk,

                &lalpha,

                A, shiftA, lda, strideA,

                A, shiftA, lda, strideA,

                &lbeta,

                C, shiftC, ldc, strideC,

                batch_count, work);
        }
        return istat;
    }; // end potrf_syrk

    auto potrf_trsm
        = [=](rocblas_side const side, rocblas_fill const uplo, rocblas_operation const trans,
              rocblas_diagonal const diag,

              I const mm, I const nn,

              auto A, rocblas_stride const shiftA, I const lda, rocblas_stride const strideA,

              auto B, rocblas_stride const shiftB, I const ldb,
              rocblas_stride const strideB) -> rocblas_status {
        rocblas_status istat = rocblas_status_success;

        if(use_rocblas_trsm)
        {
            T alpha = 1;
            istat = rocblasCall_trsm<T, I>(handle, side, uplo, trans, diag,

                                           mm, nn,

                                           &alpha,

                                           A, shiftA, lda, strideA,

                                           B, shiftB, ldb, strideB,

                                           batch_count, optim_mem, work1, work2, work3, work4);
        }
        else
        {
            if(uplo == rocblas_fill_upper)
            {
                istat = rocsolver_trsm_upper<BATCHED, STRIDED, T>(handle, side, trans, diag,

                                                                  mm, nn,

                                                                  A, shiftA, lda, strideA,

                                                                  B, shiftB, ldb, strideB,

                                                                  batch_count, optim_mem, work1,
                                                                  work2, work3, work4);
            }
            else
            {
                istat = rocsolver_trsm_lower<BATCHED, STRIDED, T>(handle, side, trans, diag, mm, nn,

                                                                  A, shiftA, lda, strideA,

                                                                  B, shiftB, ldb, strideB,

                                                                  batch_count, optim_mem, work1,
                                                                  work2, work3, work4);
            }

        } // end if use_rocblas_trsm

        return istat;
    }; // end potrf_trsm

    // -----------------------------
    // check for special sizes for n
    // -----------------------------

    // ----------------------------------------
    // computations for  special case n == 2*NB
    // ----------------------------------------
    auto potrf_2NB = [=](rocblas_stride const shiftA, I const row_offset) -> rocblas_status {
        I const n1 = NB;
        I const n2 = NB;

        rocblas_status istat = rocblas_status_success;

        // --------------------
        // factorization of A11
        // --------------------
        istat = potf2_run_small<T>(handle, uplo, n1,

                                   A, shiftA, lda, strideA,

                                   info, batch_count, row_offset);
        if(istat != rocblas_status_success)
        {
            return istat;
        }

        // -------------------------------------
        // update A12  n1 by n2  if use_upper_part
        // update A21  n2 by n1  otherwise
        //
        // step 2:   A12 = U11' * U12, or   U12 = U11' \ A12,  trsm
        // step 2:   A21 = L21 * L11', or   L21 = A21 / L11',  trsm
        // -------------------------------------

        rocblas_stride const loffset_A12 = idx2D(0, n1, lda);
        rocblas_stride const loffset_A21 = idx2D(n1, 0, lda);

        {
            I const mm = (use_upper_part) ? n1 : n2;
            I const nn = (use_upper_part) ? n2 : n1;

            rocblas_side const side = (use_upper_part) ? rocblas_side_left : rocblas_side_right;

            rocblas_operation const trans = rocblas_operation_conjugate_transpose;
            rocblas_diagonal const diag = rocblas_diagonal_non_unit;

            // ---------------------
            // offset for A12 or A21
            // ---------------------
            rocblas_stride const loffset = (use_upper_part) ? loffset_A12 : loffset_A21;

            istat = potrf_trsm(side, uplo, trans, diag,

                               mm, nn,

                               A, shiftA + idx2D(0, 0, lda), lda, strideA, // submatrix A11

                               A, shiftA + loffset, lda, strideA); // submatrix for A12 or A21
        }
        if(istat != rocblas_status_success)
        {
            return istat;
        }

        // ------------------
        // SYRK to update A22
        //
        // step 3a:  A22 = A22 - U12' * U12,   syrk
        // step 3a:  A22 = A22 - L21 * L21',   syrk
        // ------------------
        {
            I const nn = n2;
            I const kk = n1;
            S alpha = -1;
            S beta = 1;

            // --------------------
            // submatrix U12 or L21
            // --------------------
            rocblas_stride const loffset = (use_upper_part) ? loffset_A12 : loffset_A21;

            rocblas_operation const trans
                = (use_upper_part) ? rocblas_operation_conjugate_transpose : rocblas_operation_none;

            istat = potrf_syrk(

                uplo, trans,

                nn, kk,

                &alpha,

                A, shiftA + loffset, lda, strideA, // submatrix U12 or L21

                &beta,

                A, shiftA + idx2D(n1, n1, lda), lda, strideA // submatrix A22

            );
        }
        if(istat != rocblas_status_success)
        {
            return istat;
        }

        // --------------------
        // factorization of A22
        // --------------------

        istat = potf2_run_small<T>(handle, uplo, n2,

                                   A, shiftA + idx2D(n1, n1, lda), lda, strideA, // submatrix A22

                                   info, batch_count, row_offset + n1);

        return istat;
    }; // end potrf_2NB

    if(n == 2 * NB)
    {
        rocblas_status const istat = potrf_2NB(shiftA, row_offset_in);
        return istat;
    }

    // ----------------------------------------
    // computations for  special case n == 4*NB
    // ----------------------------------------
    auto potrf_4NB = [=](rocblas_stride const shiftA, I const row_offset) -> rocblas_status {
        I const n1 = 2 * NB;
        I const n2 = 2 * NB;

        rocblas_status istat = rocblas_status_success;

        // --------------------
        // factorization of A11
        // --------------------
        istat = potrf_2NB(shiftA, row_offset);

        if(istat != rocblas_status_success)
        {
            return istat;
        }

        // -------------------------------------
        // update A12  n1 by n2  if use_upper_part
        // update A21  n2 by n1  otherwise
        //
        // step 2:   A12 = U11' * U12, or   U12 = U11' \ A12,  trsm
        // step 2:   A21 = L21 * L11', or   L21 = A21 / L11',  trsm
        // -------------------------------------

        rocblas_stride const loffset_A12 = idx2D(0, n1, lda);
        rocblas_stride const loffset_A21 = idx2D(n1, 0, lda);

        {
            I const mm = (use_upper_part) ? n1 : n2;
            I const nn = (use_upper_part) ? n2 : n1;

            rocblas_side const side = (use_upper_part) ? rocblas_side_left : rocblas_side_right;

            rocblas_operation const trans = rocblas_operation_conjugate_transpose;
            rocblas_diagonal const diag = rocblas_diagonal_non_unit;

            // ---------------------
            // offset for A12 or A21
            // ---------------------
            rocblas_stride const loffset = (use_upper_part) ? loffset_A12 : loffset_A21;

            istat = potrf_trsm(side, uplo, trans, diag,

                               mm, nn,

                               A, shiftA + idx2D(0, 0, lda), lda, strideA, // submatrix A11

                               A, shiftA + loffset, lda, strideA); // submatrix for A12 or A21
        }
        if(istat != rocblas_status_success)
        {
            return istat;
        }

        // ------------------
        // SYRK to update A22
        //
        // step 3a:  A22 = A22 - U12' * U12,   syrk
        // step 3a:  A22 = A22 - L21 * L21',   syrk
        // ------------------
        {
            I const nn = n2;
            I const kk = n1;
            S alpha = -1;
            S beta = 1;

            // --------------------
            // submatrix U12 or L21
            // --------------------
            rocblas_stride const loffset = (use_upper_part) ? loffset_A12 : loffset_A21;

            rocblas_operation const trans
                = (use_upper_part) ? rocblas_operation_conjugate_transpose : rocblas_operation_none;

            istat = potrf_syrk(

                uplo, trans,

                nn, kk,

                &alpha,

                A, shiftA + loffset, lda, strideA, // submatrix U12 or L21

                &beta,

                A, shiftA + idx2D(n1, n1, lda), lda, strideA // submatrix A22

            );
        }
        if(istat != rocblas_status_success)
        {
            return istat;
        }

        // --------------------
        // factorization of A22
        // --------------------

        istat = potrf_2NB(shiftA + idx2D(n1, n1, lda), row_offset + n1); // submatrix A22

        return istat;
    }; // end potrf_4NB

    if(n == 4 * NB)
    {
        rocblas_status const istat = potrf_4NB(shiftA, row_offset_in);
        return istat;
    }

    // --------------------------
    // general case for recursion
    // --------------------------

    I const n1 = split_n<T>(n);
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

            rocblas_status const istat
                = potrf_trsm(rocblas_side_left, rocblas_fill_upper,
                             rocblas_operation_conjugate_transpose, rocblas_diagonal_non_unit,

                             mm, nn,

                             A, shiftA + idx2D(0, 0, lda), lda, strideA, // submatrix U11

                             A, shiftA + idx2D(0, n1, lda), lda, strideA // submatrix U12
                );

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
            rocblas_status const istat = potrf_syrk(

                uplo, rocblas_operation_conjugate_transpose,

                nn, kk,

                &s_minone,

                A, shiftA + idx2D(0, n1, lda), lda, strideA, // submatrix U12

                &s_one,

                A, shiftA + idx2D(n1, n1, lda), lda, strideA // submatrix A22

            );
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

            rocblas_status const istat
                = potrf_trsm(rocblas_side_right, rocblas_fill_lower,
                             rocblas_operation_conjugate_transpose, rocblas_diagonal_non_unit,

                             mm, nn,

                             A, shiftA + idx2D(0, 0, lda), lda, strideA, // submatrix L11

                             A, shiftA + idx2D(n1, 0, lda), lda, strideA // submatrix L21
                );

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
            rocblas_status const istat = potrf_syrk(

                uplo, rocblas_operation_none,

                nn, kk,

                &s_minone,

                A, shiftA + idx2D(n1, 0, lda), lda, strideA, // submatrix L21

                &s_one,

                A, shiftA + idx2D(n1, n1, lda), lda, strideA // submatrix A22

            );

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
                                        void* work1_arg,
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
    void* work1 = work1_arg;

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    rocblas_status istat = rocblas_status_success;
    if(use_recursion_potrf)
    {
        T* save_A = nullptr;
        bool const use_gemm = (!use_syrk);
        if(use_gemm)
        {
            // ------------------------------
            // get scratch storage from work1
            // ------------------------------
            std::byte* pfree = reinterpret_cast<std::byte*>(work1_arg);
            size_t const size_save_A = sizeof(T) * (size_t(n) * (n - 1) / 2) * batch_count;

            save_A = reinterpret_cast<T*>(pfree);
            pfree += size_save_A;
            work1 = reinterpret_cast<void*>(pfree);
        }

        if(use_gemm)
        {
            // ----------------------------------------------------------------
            // save a copy of the  strictly lower or upper triangular part of A
            // ----------------------------------------------------------------

            // ---------------------------------------------------------------
            // if factorizing the upper part, then need to save the lower part
            // before it is destroyed in the GEMM operation
            // ---------------------------------------------------------------
            bool const is_copy_strictly_lower = (uplo == rocblas_fill_upper);
            bool const is_restore = false;
            copy_strictly_triangular(is_copy_strictly_lower, is_restore, n,

                                     A, shiftA, lda, strideA,

                                     save_A, batch_count);
        }

        I const row_offset = 0;
        istat = rocsolver_potrf_recursion_template<BATCHED, STRIDED, T, I, INFO, S, U>(
            handle, uplo, n,

            A, shiftA, lda, strideA,

            info, batch_count,

            scalars, work1, work2, work3, work4, optim_mem, row_offset);

        if(use_gemm)
        {
            // --------------------------------------------------------------------
            // restore the copy of the strictly lower or upper triangular part of A
            // --------------------------------------------------------------------
            bool const is_copy_strictly_lower = (uplo == rocblas_fill_upper);
            bool const is_restore = true;

            copy_strictly_triangular(is_copy_strictly_lower, is_restore, n,

                                     A, shiftA, lda, strideA,

                                     save_A, batch_count);
        }
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
