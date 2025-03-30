
/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxiliary/rocauxiliary_lacgv.hpp"
#include "auxiliary/rocauxiliary_larf.hpp"
#include "auxiliary/rocauxiliary_larfb.hpp"
#include "auxiliary/rocauxiliary_larfg.hpp"
#include "auxiliary/rocauxiliary_larft.hpp"
#include "rocblas.hpp"
#include "roclapack_geqr2.hpp"
#include "rocsolver/rocsolver.h"
#include "specialized/roclapack_gemm_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

constexpr bool use_trmm_outofplace = false;

// -------------------------------------------------
// in case GEMM is not fully optimize for small m, n
// consider calling GEMV instead
// -------------------------------------------------
constexpr bool use_gemm_gemv = true;

constexpr bool is_applyQtC_use_larfb = true;

constexpr bool use_geqr2 = true;

#ifndef RGEQR3_BLOCKSIZE
// #define RGEQR3_BLOCKSIZE(T) ((sizeof(T) == 4) ? 256 : (sizeof(T) == 8) ? 128 : (sizeof(T) == 16) ? 64 : 64)
#define RGEQR3_BLOCKSIZE(T) 64
#endif

// Is power of two
static bool rocblas_is_po2(int x)
{
    return (x && !(x & (x - 1)));
}

static uint32_t rocblas_next_po2(uint32_t n)
{
    if(n == 0)
    {
        return 1;
    }

    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// Return previous power of two
static constexpr int rocblas_previous_po2(int x)
{
    if(rocblas_is_po2(x))
    {
        return (x);
    };

    return (rocblas_next_po2(x) / 2);
}

static rocblas_int get_n_small()
{
    return (16);
};

//   -----------------------------------------------------------
//   rocblas gemm may not be optimized for cases with small m,n
//   consider calling  gemv() when feasible for m==1, or n==1
//   consider rocsolver_gemm() when m,n are small
//   otherwise, use rocblas gemm
//   -----------------------------------------------------------

template <typename T, typename I, typename Istride, typename UA, typename UB, typename UC>
static rocblas_status gemm_gemv(rocblas_handle handle,
                                rocblas_operation transA,
                                rocblas_operation transB,
                                I m,
                                I n,
                                I k,

                                T* alpha,

                                UA Amat,
                                Istride shift_Amat,
                                I ldA,
                                Istride stride_Amat,

                                UB Bmat,
                                Istride shift_Bmat,
                                I ldB,
                                Istride stride_Bmat,

                                T* beta,
                                UC Cmat,
                                Istride shift_Cmat,
                                I ldC,
                                Istride stride_Cmat,

                                I batch_count,
                                T** workArr)
{
    //  ------------------------------------
    //  C = beta * C + alpha * op(A) * op(B)
    //  ------------------------------------

    bool const has_work = (m >= 1) && (n >= 1) && (k >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return (rocblas_status_success);
    }

    bool const is_no_transpose_B = (transB == rocblas_operation_none);
    bool const is_transpose_B = (transB == rocblas_operation_transpose);
    bool const is_conj_transpose_B = (transB == rocblas_operation_conjugate_transpose);

    bool const is_no_transpose_A = (transA == rocblas_operation_none);
    bool const is_transpose_A = (transA == rocblas_operation_transpose);
    bool const is_conj_transpose_A = (transA == rocblas_operation_conjugate_transpose);

    I const nrows_A = (is_no_transpose_A) ? m : k;
    I const ncols_A = (is_no_transpose_A) ? k : m;

    I const nrows_B = (is_no_transpose_B) ? k : n;
    I const ncols_B = (is_no_transpose_B) ? n : k;

    if((n == 1) && (!is_conj_transpose_B))
    {
        // -----------------------------------
        // [c1] = beta * [c1] + alpha * [op(A)]  * [b1]
        // [.]           [.]                       [.]
        // [cm]          [cm]                      [bk]
        // -----------------------------------

        auto x = Bmat;
        Istride const offsetx = shift_Bmat;
        I const incx = (is_no_transpose_B) ? 1 : ldB;
        Istride const stridex = stride_Bmat;

        auto y = Cmat;
        Istride const offsety = shift_Cmat;
        I const incy = 1;
        Istride const stridey = stride_Cmat;

        Istride const stride_alpha = 0;
        Istride const stride_beta = 0;
        I const mm = nrows_A;
        I const nn = ncols_A;

        ROCBLAS_CHECK(rocblasCall_gemv(handle, transA, mm, nn, alpha, stride_alpha,

                                       Amat, shift_Amat, ldA, stride_Amat,

                                       x, offsetx, incx, stridex,

                                       beta, stride_beta,

                                       y, offsety, incy, stridey,

                                       batch_count, workArr));
    }
    else if((m == 1) && (!is_conj_transpose_A) && (!is_conj_transpose_B))
    {
        // ---------------------------------------------------------------------------
        // [c1, .., cn] = beta * [c1, ..., cn] + alpha * [a1 ... ak ] * op([B1 |  ... | Bn])
        // ---------------------------------------------------------------------------
        //
        // or take transpose
        //
        // [c1] = beta [c1] + alpha * op2( B ) * [a1]
        // [.]         [.]                       [.]
        // [cn]        [cn]                      [ak]
        // ------------------------------------------------------------------------------

        auto x = Amat;
        Istride offsetx = shift_Amat;
        I incx = (is_no_transpose_A) ? ldA : 1;
        Istride stridex = stride_Amat;

        auto y = Cmat;
        Istride offsety = shift_Cmat;
        I incy = ldC;
        Istride stridey = stride_Cmat;

        Istride const stride_alpha = 0;
        Istride const stride_beta = 0;
        I const mm = nrows_B;
        I const nn = ncols_B;

        rocblas_operation trans_transB
            = (is_no_transpose_B) ? rocblas_operation_transpose : rocblas_operation_none;

        ROCBLAS_CHECK(rocblasCall_gemv(handle, trans_transB, mm, nn, alpha, stride_alpha,

                                       Bmat, shift_Bmat, ldB, stride_Bmat,

                                       x, offsetx, incx, stridex,

                                       beta, stride_beta,

                                       y, offsety, incy, stridey,

                                       batch_count, workArr));
    }
    else
    {
        I const mn_small = 32;
        bool const is_small_mn = (m <= mn_small) && (n <= mn_small);
        bool const use_rocsolver_gemm = true;

        if(use_rocsolver_gemm && is_small_mn)
        {
            ROCBLAS_CHECK(rocsolver_gemm(handle,

                                         transA, transB, m, n, k, alpha,

                                         Amat, shift_Amat, ldA, stride_Amat,

                                         Bmat, shift_Bmat, ldB, stride_Bmat,

                                         beta,

                                         Cmat, shift_Cmat, ldC, stride_Cmat,

                                         batch_count, workArr));
        }
        else
        {
            ROCBLAS_CHECK(rocblasCall_gemm(handle,

                                           transA, transB, m, n, k, alpha,

                                           Amat, shift_Amat, ldA, stride_Amat,

                                           Bmat, shift_Bmat, ldB, stride_Bmat,

                                           beta, Cmat, shift_Cmat, ldC, stride_Cmat,

                                           batch_count, workArr));
        }
    }

    return (rocblas_status_success);
}

template <typename T, typename I>
static void rocblasCall_trmm_mem(rocblas_side const side,
                                 I const mm,
                                 I const nn,
                                 I const batch_count,
                                 size_t* size_trmm_byte)
{
    *size_trmm_byte = 2 * sizeof(T*) * std::max(I(1), batch_count);
}

// -----------------------------------------------
// copy diagonal values
//
// launch as dim(nbx,1,batch_count), dim3(nx,1,1)
// where nbx = ceil( n, nx)
// -----------------------------------------------
template <typename T, typename I, typename Istride>
static __global__ void copy_diagonal_kernel(bool const Tmat2tau,
                                            I const n,
                                            T* const Tmat,
                                            Istride const shift_Tmat,
                                            I const ldT,
                                            Istride const stride_Tmat,

                                            T* const tau_,
                                            Istride const stride_tau,

                                            I const batch_count)

{
    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const i_inc = hipBlockDim_x * hipGridDim_x;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const __restrict__ T_p = load_ptr_batch(Tmat, bid, shift_Tmat, stride_Tmat);
        T* const __restrict__ tau_p = tau_ + bid * stride_tau;

        auto Tp = [=](auto i, auto j) -> T& {
            auto const ij = i + j * static_cast<int64_t>(ldT);
            return (T_p[ij]);
        };

        auto tau = [=](auto i) -> T& { return (tau_p[i]); };

        for(auto i = i_start; i < n; i += i_inc)
        {
            if(Tmat2tau)
            {
                tau(i) = Tp(i, i);
            }
            else
            {
                Tp(i, i) = tau(i);
            }
        }

    } // end for bid
}

// -----------------------------------
// copy diagonal entries from T matrix
// into tau array
// -----------------------------------
template <typename T, typename I, typename Istride>
static void copy_diagonal_template(rocblas_handle handle,
                                   bool const Tmat2tau,
                                   I const nn,
                                   T* const Tmat,
                                   Istride const shift_Tmat,
                                   I const ldT,
                                   Istride const stride_Tmat,
                                   T* const tau_,
                                   Istride const stride_tau,
                                   I const batch_count)
{
    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    I const max_blocks = 1024;
    I const nx = 64;
    I const nbx = std::min(max_blocks, ceil(nn, nx));
    I const nbz = std::min(max_blocks, batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    copy_diagonal_kernel<<<dim3(nbx, 1, nbz), dim3(nx, 1, 1), 0, stream>>>(
        Tmat2tau, nn, Tmat, shift_Tmat, ldT, stride_Tmat, tau_, stride_tau, batch_count);
}

// -----------------------------------------
// geadd() performs matrix addition that is
// similar PxGEADD() in Parallel BLAS library
//
//  C(1:m,1:n) =  beta * C(1:m,1:n) + alpha * op(A)
//
// assume launch with
// dim3(nbx,nby,max_nblocks), dim3(nx,ny,1)
// -----------------------------------------
template <typename T, typename UA, typename UC, typename I, typename Istride>
static __global__ void geadd_kernel(char const trans,
                                    I const m,
                                    I const n,
                                    T const alpha,
                                    UA AA,
                                    I const shiftA,
                                    I const ldA,
                                    Istride const strideA,
                                    T const beta,
                                    UC CC,
                                    I const shiftC,
                                    I const ldC,
                                    Istride const strideC,
                                    I const batch_count)
{
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    };

    auto const nbx = hipGridDim_x;
    auto const nby = hipGridDim_y;
    auto const nx = hipBlockDim_x;
    auto const ny = hipBlockDim_y;

    bool const is_transpose = (trans == 'T') || (trans == 't');
    bool const is_conj_transpose = (trans == 'C') || (trans == 'c');
    bool const is_no_transpose = (!is_transpose) && (!is_conj_transpose);

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * nx;
    I const i_inc = (nbx * nx);
    I const j_start = hipThreadIdx_y + hipBlockIdx_y * ny;
    I const j_inc = (nby * ny);

    auto const bid_inc = hipGridDim_z;
    auto const bid_start = hipBlockIdx_z;

    T const zero = 0;
    bool const is_beta_zero = (beta == zero);
    bool const is_alpha_one = (alpha == 1);

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const A_ = load_ptr_batch(AA, bid, shiftA, strideA);
        T* const C_ = load_ptr_batch(CC, bid, shiftC, strideC);

        auto A = [=](auto i, auto j) -> const T& { return (A_[i + j * static_cast<int64_t>(ldA)]); };

        auto C = [=](auto i, auto j) -> T& { return (C_[i + j * static_cast<int64_t>(ldC)]); };

        if(is_no_transpose)
        {
            // ------------------
            // just copy matrices
            // ------------------
            if(is_beta_zero && is_alpha_one)
            {
                auto const i = i_start;
                auto const j = j_start;
                T const* __restrict__ ap_j = &(A(i, j));
                T* __restrict__ cp_j = &(C(i, j));

                for(auto j = j_start; j < n; j += j_inc)
                {
                    T const* __restrict__ ap = ap_j;
                    T* __restrict__ cp = cp_j;

                    for(auto i = i_start; i < m; i += i_inc)
                    {
                        // ------------------
                        // C(i, j) = A(i, j);
                        // ------------------
                        *cp = *ap;
                        cp += i_inc;
                        ap += i_inc;
                    }

                    ap_j += (ldA * j_inc);
                    cp_j += (ldC * j_inc);
                }
            }
            else
            {
                auto const i = i_start;
                auto const j = j_start;
                T const* __restrict__ ap_j = &(A(i, j));
                T* __restrict__ cp_j = &(C(i, j));

                for(auto j = j_start; j < n; j += j_inc)
                {
                    T const* __restrict__ ap = ap_j;
                    T* __restrict__ cp = cp_j;

                    // auto const beta_cij = (is_beta_zero) ? zero : beta * C(i, j);
                    // C(i, j) = beta_cij + alpha * A(i, j);

                    if(is_beta_zero)
                    {
                        for(auto i = i_start; i < m; i += i_inc)
                        {
                            *cp = alpha * (*ap);
                            cp += i_inc;
                            ap += i_inc;
                        }
                    }
                    else
                    {
                        for(auto i = i_start; i < m; i += i_inc)
                        {
                            *cp = (*cp) * beta + alpha * (*ap);
                            cp += i_inc;
                            ap += i_inc;
                        }
                    }

                    ap_j += (ldA * j_inc);
                    cp_j += (ldC * j_inc);
                }
            }
        }
        else
        {
            I const i = i_start;
            I const j = j_start;

            T const* __restrict__ ap_j = (is_no_transpose) ? &(A(i, j)) : &(A(j, i));
            I const ap_j_inc = (is_no_transpose) ? ldA * j_inc : j_inc;
            I const ap_inc = (is_no_transpose) ? i_inc : ldC * i_inc;

            T* __restrict__ cp_j = &(C(i, j));
            I const cp_j_in = ldC * j_inc;

            for(auto j = j_start; j < n; j += j_inc)
            {
                T const* __restrict__ ap = ap_j;
                T* __restrict__ cp = cp_j;

                if(is_beta_zero)
                {
                    for(auto i = i_start; i < m; i += i_inc)
                    {
                        auto const aij = (is_conj_transpose) ? conj(*ap) : (*ap);
                        *cp = alpha * aij;
                        cp += i_inc;
                        ap += ap_inc;
                    }
                }
                else
                {
                    for(auto i = i_start; i < m; i += i_inc)
                    {
                        auto const aij = (is_conj_transpose) ? conj(*ap) : (*ap);
                        *cp = beta * (*cp) + alpha * aij;
                        cp += i_inc;
                        ap += ap_inc;
                    }
                }
            }
        }
    }
}

// -----------------------------------------
// geadd() performs matrix addition that is
// similar PxGEADD() in Parallel BLAS library
//
//  C(1:m,1:n) =  beta * C(1:m,1:n) + alpha * op(A)
// -----------------------------------------
template <typename T, typename UA, typename UC, typename I, typename Istride>
static void geadd_template(rocblas_handle handle,
                           char const trans,
                           I const m,
                           I const n,
                           T const alpha,
                           UA AA,
                           Istride const shiftA,
                           I const ldA,
                           Istride const strideA,
                           T const beta,
                           UC CC,
                           Istride const shiftC,
                           I const ldC,
                           Istride const strideC,
                           I const batch_count)
{
    auto ceil = [](auto m, auto nb) { return ((m - 1) / nb + 1); };

    I const max_threads = 1024;
    I const max_blocks = 1024;

    I const nx = 32;
    I const ny = max_threads / nx;
    I const nbx = std::min(max_blocks, ceil(m, nx));
    I const nby = std::min(max_blocks, ceil(n, ny));
    I const nbz = std::min(max_blocks, batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    geadd_kernel<T, UA, UC, I, Istride><<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(
        trans, m, n, alpha, AA, shiftA, ldA, strideA, beta, CC, shiftC, ldC, strideC, batch_count);
}

// -------------------------------
//  [W] = formT3( Y1, T1, Y2, T2 )
//
//  compute T3 = W = -T1 * (Y1' * Y2 ) * T2
//
//  Y1 is m by k1
//  Y2 is (m-k1) by k2
//
//
//
//  Let
//  Y1 = [Y11; Y21; Y31];
//
//  Y1 = [ Y11 ]
//       [ Y21 ]
//       [ Y31 ]
//
//
//  Y2 = [   0 ]
//       [ Y12 ]
//       [ Y22 ]
//
//  Merged triangular matrix for block
//  Householder reflectors
//  H1 = (I - Y1 * T1 * Y1')
//  H2 = (I - Y2 * T2 * Y2')
//
//  H3 = H1 * H2 = (I - [Y1 | Y2] * Tmat * [Y1 | Y2]')
//  where
//  Tmat = [T1    T3]
//         [0     T2]
//  --------------------------------
template <typename T, typename I, typename U, typename Istride>
static rocblas_status formT3(rocblas_handle handle,
                             I const m,
                             I const k1,
                             I const k2,

                             U Ymat,
                             Istride const shift_Y1,
                             Istride const shift_Y2,
                             I const ldY,
                             Istride stride_Ymat,

                             T* const Tmat,
                             Istride const shift_T1,
                             Istride const shift_T2,
                             I const ldT,
                             Istride const stride_Tmat,
                             I const batch_count,

                             void* work,
                             I& lwork_bytes)

{
    ROCSOLVER_ENTER("formT3", "m:", m, "k1:", k1, "k2:", k2, "shift_Y1:", shift_Y1, "ldY:", ldY,
                    "bc:", batch_count);

    bool const has_work = (m >= 1) && (k1 >= 1) && (k2 >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return (rocblas_status_success);
    }

    if(work == nullptr)
    {
        return (rocblas_status_invalid_pointer);
    }

    // 1-based Fortran indexing
    auto idx2F
        = [=](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    I total_bytes = 0;
    I remain_bytes = 0;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    std::byte* pfree = reinterpret_cast<std::byte*>(work);

    // W is k1 by k2
    // W  = zeros( size(T1,1), size(T2,2) );

    I const nrows_T1 = k1;
    I const ncols_T1 = k1;

    I const nrows_T2 = k2;
    I const ncols_T2 = k2;

    I const nrows_T3 = nrows_T1;
    I const ncols_T3 = ncols_T2;

    // ---------------------
    // reuse storage T3 be W
    // ---------------------
    I const nrows_W = k1;
    I const ncols_W = k2;

    Istride const shift_T3 = shift_T1 + idx2F(1, k1 + 1, ldT);

    T* const Wmat = Tmat;
    Istride const shift_Wmat = shift_T3;
    I const ldW = ldT;
    Istride const stride_Wmat = stride_Tmat;

    I const nrows_W2 = nrows_W;
    I const ncols_W2 = ncols_W;
    I const ldW2 = nrows_W2;
    Istride const stride_Wmat2 = ldW2 * ncols_W2;
    Istride const shift_Wmat2 = 0;
    size_t const size_Wmat2 = (sizeof(T) * ldW2 * ncols_W2) * batch_count;

    T* Wmat2 = nullptr;
    if(use_trmm_outofplace)
    {
        Wmat2 = reinterpret_cast<T*>(pfree);
        pfree += size_Wmat2;
        total_bytes += size_Wmat2;
    }

    remain_bytes = lwork_bytes - total_bytes;
    if(remain_bytes < 0)
    {
        return (rocblas_status_memory_error);
    }

    // Y1 is m by k1
    // Y2 is (m-k1) by k2

    I const nrows_Y1 = m;
    I const ncols_Y1 = k1;

    I const nrows_Y2 = (m - k1);
    I const ncols_Y2 = k2;

    //
    // m = size(Y1,1);
    // k1 = size(Y1,2);
    // k2 = size(Y2,2);
    // k = k1 + k2;

    I const k = k1 + k2;

    // % -----------------------------------------
    // % Y11 is unit lower triangular size k1 x k1
    // % but Y11 is not used
    // % -----------------------------------------
    // Y11 = Y1(1:k1,1:k1);
    // Y11 = Y11 - triu(Y11) + eye(k1,k1);
    // ---------------------------------------

    Istride const shift_Y11 = shift_Y1 + idx2F(1, 1, ldY);
    I const nrows_Y11 = k1;
    I const ncols_Y11 = k1;

    // % -----------------------------------------
    // % Y12 is unit lower triangular size k2 x k2
    // % -----------------------------------------
    // Y12 = Y2( 1:k2, 1:k2);
    // Y12 = Y12 - triu( Y12 ) + eye( k2,k2);
    // ---------------------------------------
    Istride const shift_Y12 = shift_Y2 + idx2F(1, 1, ldY);
    I const nrows_Y12 = k2;
    I const ncols_Y12 = k2;

    // % -----------------
    // % Y21 is k2 by k1
    // % -----------------
    // Y21 = Y1( (k1+1):(k1 + k2), 1:k1);
    // ----------------------------------
    Istride const shift_Y21 = shift_Y1 + idx2F((k1 + 1), 1, ldY);
    I const nrows_Y21 = k2;
    I const ncols_Y21 = k1;

    // % -----------------
    // % Y31 is (m-k) by k1
    // % -----------------
    // i1 = (k1+k2 + 1);
    // i2 = m;
    // Y31 = Y1( i1:i2, 1:k1);
    // -----------------------
    I i1 = (k1 + k2 + 1);
    I i2 = m;

    Istride const shift_Y31 = shift_Y1 + idx2F(i1, 1, ldY);
    I const nrows_Y31 = (i2 - i1 + 1);
    I const ncols_Y31 = k1;

    // note (nrows_Y31 == (m - k));

    // % ------------------
    // % Y22 is (m-k) by k2
    // % ------------------
    // i2 = size(Y2,1);
    // i1 = (k2+1);
    // Y22 = Y2( i1:i2, 1:k1 );

    i2 = nrows_Y2;
    i1 = (k2 + 1);
    Istride const shift_Y22 = shift_Y2 + idx2F(i1, 1, ldY);
    I const nrows_Y22 = (i2 - i1 + 1);
    I const ncols_Y22 = k2;

    // note (nrows_Y22 == (m - k));

    // % -------------------
    // % (0) first set W =  Y21'
    // % -------------------
    // W = Y21';
    {
        // note (nrows_W == ncols_Y21);
        // note (ncols_W == nrows_Y21);

        char const trans = (rocblas_is_complex<T>) ? 'C' : 'T';
        I const mm = nrows_W;
        I const nn = ncols_W;
        T const alpha = 1;
        T const beta = 0;

        {
            // clang-format off
           geadd_template( handle,
		    trans,
		    mm,
		    nn,
		    alpha,
		    Ymat, shift_Y21, ldY, stride_Ymat,
		    beta,
		    Wmat, shift_Wmat, ldW, stride_Wmat,
		    batch_count );
            // clang-format on
        }
    }

    // % ------------------------
    // % (0) first set W =  Y21'
    // % (1) TRMM   W = (Y21') * Y12
    // % (2) GEMM   W = W + Y31' * Y22
    // % (3) TRMM   W = -T1 * W
    // % (4) TRMM   W = W * T2
    // % ------------------------

    // % --------------------
    // % (1)   W = Y21' * Y12
    // % c_side = 'R';
    // % c_uplo = 'L';
    // % c_trans = 'N';
    // % c_diag = 'U';
    // %  W = trmm( side, uplo, trans, cdiag, mm,nn, alpha, Y12, W );
    // % --------------------
    {
        // note (nrows_Y21 == nrows_Y12);
        // note (ncols_W == ncols_Y12);
        // note (nrows_W == ncols_Y21);

        rocblas_side const side = rocblas_side_right;
        rocblas_fill const uplo = rocblas_fill_lower;
        rocblas_operation const trans = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_unit;

        auto const mm = nrows_W;
        auto const nn = ncols_W;
        T alpha = 1;
        Istride const stride_alpha = 0;

        size_t size_trmm_bytes = 0;
        rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes);

        size_t const size_workArr = size_trmm_bytes;

        T** const workArr = reinterpret_cast<T**>(pfree);
        pfree += size_trmm_bytes;

        total_bytes += size_trmm_bytes;

        remain_bytes = lwork_bytes - total_bytes;

        {
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        if(use_trmm_outofplace)
        {
            // clang-format off
	    ROCBLAS_CHECK(rocblasCall_trmm(handle,
			    side, uplo, trans, diag,
			    mm,nn,
			    &alpha, stride_alpha,
			    Ymat, shift_Y12, ldY, stride_Ymat,
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    Wmat2, shift_Wmat2, ldW2, stride_Wmat2,
			    batch_count, workArr ));
            // clang-format on

            {
                // ------------------
                // copy Wmat2 -> Wmat
                // ------------------
                T const alpha = 1;
                T const beta = 0;
                char const trans = 'N';
                I const mm = nrows_W;
                I const nn = ncols_W;

                // clang-format off
                  geadd_template( handle,
		    trans,
		    mm,
		    nn,
		    alpha,
		    Wmat2, shift_Wmat2, ldW2, stride_Wmat2,
		    beta,
		    Wmat, shift_Wmat, ldW, stride_Wmat,
		    batch_count );
                // clang-format on
            }
        }
        else
        {
            // clang-format off
	    ROCBLAS_CHECK(rocblasCall_trmm(handle,
			    side, uplo, trans, diag,
			    mm,nn,
			    &alpha, stride_alpha,
			    Ymat, shift_Y12, ldY, stride_Ymat,
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    batch_count, workArr ));
            // clang-format on
        }

        total_bytes = total_bytes - size_trmm_bytes;
        pfree = pfree - size_trmm_bytes;
    }

    // % -----------------------------
    // % (2) GEMM   W = W + Y31' * Y22
    // % transA = 'C';
    // % transB = 'N';
    // % mm = size(W,1);
    // % nn = size(W,2);
    // % kk = size(Y22,1);
    // % -----------------------------
    {
        assert(nrows_W == ncols_Y31);
        assert(nrows_Y31 == nrows_Y22);
        assert(ncols_Y22 == ncols_W);

        rocblas_operation const transA = (rocblas_is_complex<T>)
            ? rocblas_operation_conjugate_transpose
            : rocblas_operation_transpose;
        rocblas_operation const transB = rocblas_operation_none;

        auto const mm = nrows_W;
        auto const nn = ncols_W;
        auto const kk = nrows_Y22;

        T alpha = 1;
        T beta = 1;

        size_t size_gemm = 2 * sizeof(T*) * batch_count;
        T** const workArr = reinterpret_cast<T**>(pfree);
        pfree += size_gemm;
        total_bytes += size_gemm;

        remain_bytes = lwork_bytes - total_bytes;

        {
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        // clang-format off
	if (use_gemm_gemv) {
	    ROCBLAS_CHECK( gemm_gemv( handle,
			    transA, transB,
			    mm, nn, kk,
			    &alpha,
			    Ymat, shift_Y31, ldY, stride_Ymat,
			    Ymat, shift_Y22, ldY, stride_Ymat,
			    &beta,
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    batch_count,
			    workArr ));
	}
	else {

            ROCBLAS_CHECK(rocblasCall_gemm(handle,
			    transA, transB,
			    mm, nn, kk,
			    &alpha,
			    Ymat, shift_Y31, ldY, stride_Ymat,
			    Ymat, shift_Y22, ldY, stride_Ymat,
			    &beta,
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    batch_count,
			    workArr ));
	}
        // clang-format on
        total_bytes = total_bytes - size_gemm;
        pfree = pfree - size_gemm;
    }

    // % -----------------------
    // % (3) TRMM    W = -T1 * W
    // % -----------------------
    //
    // side = 'L';
    // uplo = 'U';
    // transA = 'N';
    // cdiag = 'N';
    // mm = size(W,1);
    // nn = size(W,2);
    // alpha = -1;
    //
    // W = trmm( side, uplo, transA, cdiag, mm,nn,alpha, T1, W );

    {
        assert(ncols_T1 == nrows_W);

        rocblas_side const side = rocblas_side_left;
        rocblas_fill const uplo = rocblas_fill_upper;
        rocblas_operation const transA = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_non_unit;

        auto const mm = nrows_W;
        auto const nn = ncols_W;

        T alpha = -1;
        Istride const stride_alpha = 0;

        size_t size_trmm_bytes = 0;
        rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes);
        size_t const size_workArr = size_trmm_bytes;

        T** const workArr = reinterpret_cast<T**>(pfree);
        pfree += size_workArr;

        total_bytes += size_workArr;
        remain_bytes = lwork_bytes - total_bytes;

        {
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        {
            // clang-format off
		    ROCBLAS_CHECK( rocblasCall_trmm( handle,
				    side, uplo, transA, diag,
				    mm, nn,
				    &alpha, stride_alpha,
				    Tmat, shift_T1, ldT, stride_Tmat,
				    Wmat, shift_Wmat,  ldW, stride_Wmat,
				    batch_count,
				    workArr ));
            // clang-format on
        }
        total_bytes = total_bytes - size_workArr;
        pfree = pfree - size_workArr;
    }

    // % ---------------------
    // % (4) TRMM   W = W * T2
    // % ---------------------
    // side = 'R';
    // uplo = 'U';
    // transA = 'N';
    // cdiag = 'N';
    // alpha = 1;
    // W = trmm( side, uplo, transA, cdiag, mm,nn,alpha, T2, W );

    {
        assert(ncols_W == nrows_T2);

        rocblas_side const side = rocblas_side_right;
        rocblas_fill const uplo = rocblas_fill_upper;
        rocblas_operation const transA = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_non_unit;

        T alpha = 1;
        Istride const stride_alpha = 0;

        I const mm = nrows_W;
        I const nn = ncols_W;

        size_t size_trmm_bytes = 0;
        rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes);
        size_t const size_workArr = size_trmm_bytes;

        T** const workArr = reinterpret_cast<T**>(pfree);
        pfree += size_workArr;
        total_bytes += size_workArr;

        remain_bytes = lwork_bytes - total_bytes;

        {
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        {
            // clang-format off
		    ROCBLAS_CHECK( rocblasCall_trmm( handle,
				    side, uplo, transA, diag,
				    mm, nn,
				    &alpha, stride_alpha,
				    Tmat, shift_T2, ldT, stride_Tmat,
				    Wmat, shift_Wmat, ldW, stride_Wmat,
				    batch_count,
				    workArr ));
            // clang-format on
        }

        total_bytes = total_bytes - size_workArr;
        pfree = pfree - size_workArr;
    }

    return (rocblas_status_success);
}

// ------------------------------
//  Perform C = Q' * C,
//  where Q = eye - Y * T * Y'
//  so Q' = eye - Y * T' * Y
//
//  note Y is lower trapezoidal and has unit diagonal
//
//  C is m by n
//  Y is m by k
//  T is k by k
// -------------------------

template <typename T, typename I, typename UY, typename Istride>
static rocblas_status applyQtC_body(rocblas_handle handle,
                                    I const m,
                                    I const n,
                                    I const k,

                                    UY Ymat,
                                    Istride const shift_Ymat,
                                    I const ldY,
                                    Istride const stride_Ymat,

                                    T* const Tmat,
                                    Istride const shift_Tmat,
                                    I const ldT,
                                    Istride const stride_Tmat,

                                    UY Cmat,
                                    Istride const shift_Cmat,
                                    I const ldC,
                                    Istride const stride_Cmat,

                                    I const batch_count,
                                    void* work,
                                    I& lwork_bytes)
{
    ROCSOLVER_ENTER("applyQtC_body", "m:", m, "n:", n, "k:", k, "shift_Ymat:", shift_Ymat,
                    "ldY:", ldY, "bc:", batch_count);
    // 1-based matlab/Fortran indexing
    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    auto swap = [](auto& x, auto& y) {
        auto t = x;
        x = y;
        y = t;
    };

    I total_bytes = 0;
    I remain_bytes = 0;

    {
        bool const has_work = (m >= 1) && (n >= 1) && (k >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        }
    }

    std::byte* pfree = reinterpret_cast<std::byte*>(work);
    {
        if(work == nullptr)
        {
            return (rocblas_status_invalid_pointer);
        }
    }

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // -----------
    // C is m by n
    // Y is m by k
    // T is k by k
    // -----------
    I const nrows_C = m;
    I const ncols_C = n;

    I const nrows_Y = m;
    I const ncols_Y = k;

    I const nrows_T = k;
    I const ncols_T = k;

    //  -------------------
    //  Partition Y and C as
    //   Y = [Y1],    C = [C1]
    //       [Y2]         [C2]
    //
    //
    //   where Y1 is k by k = Y( 1:k,1:k)
    //         Y2 is (m-k) by k = Y( (k+1):m, 1:k)
    //
    // 	C1 is k by n = C(1:k,1:n)
    // 	C2 is (m-k) by n = C( (k+1):m, 1:n )
    //  -------------------

    Istride const shift_C1 = shift_Cmat + idx2F(1, 1, ldC);
    Istride const shift_C2 = shift_Cmat + idx2F((k + 1), 1, ldC);

    Istride const shift_Y1 = shift_Ymat + idx2F(1, 1, ldY);
    Istride const shift_Y2 = shift_Ymat + idx2F((k + 1), 1, ldY);

    I const nrows_C1 = k;
    I const ncols_C1 = n;

    I const nrows_C2 = (m - k);
    I const ncols_C2 = n;

    I const nrows_Y1 = k;
    I const ncols_Y1 = k;

    I const nrows_Y2 = (m - k);
    I const ncols_Y2 = k;

    //   ---------------------------------
    //   [C1] - [Y1] T' * [Y1',  Y2'] * [C1]
    //   [C2]   [Y2]                    [C2]
    //
    //   [C1] - [Y1]  T' * (Y1'*C1 + Y2'*C2)
    //   [C2]   [Y2]
    //
    //   [C1] - [Y1]  T' * W,  where W = Y1'*C1 + Y2'*C2
    //   [C2]   [Y2]
    //
    //   ---------------------------------

    // % --------------------------
    // % (1) W = Y1' * C1, trmm
    // % or
    // % (1a) W = C1,   copy
    // % (1b) W = Y1' * W, trmm
    //
    // % (2) W = W + Y2' * C2, gemm
    // % (3) W = T' * W,   trmm
    // % (4) C2 = C2 - Y2 * W, gemm
    // % (5) W = Y1 * W, trmm
    // % (6) C1 = C1 - W
    // % --------------------------

    // % ------------
    // % (1) W = Y1' * C1;
    // % or
    // % (1a) W = C1,  use copy
    // % (1b) W = (Y1') * W, use trmm
    // % ------------
    //
    // W = C1;
    // side = 'L';
    // transA = 'C';
    // cdiag = 'U';
    // uplo = 'L';
    // alpha = 1;
    // mm = size(C1,1);
    // nn = size(C1,2);
    // W = trmm( side, uplo, transA, cdiag, mm,nn,alpha, Y1, W );

    // allocate storage for Wmat
    //
    I const nrows_W = nrows_C1;
    I const ncols_W = ncols_C1;

    I const ldW = nrows_W;
    Istride const shift_Wmat = 0;
    Istride const stride_Wmat = ldW * ncols_W;

    size_t size_Wmat_bytes = (sizeof(T) * ldW * ncols_W) * batch_count;
    T* Wmat = reinterpret_cast<T*>(pfree);
    pfree += size_Wmat_bytes;
    total_bytes += size_Wmat_bytes;

    T* Wmat2 = nullptr;

    if(use_trmm_outofplace)
    {
        Wmat2 = reinterpret_cast<T*>(pfree);
        pfree += size_Wmat_bytes;
        total_bytes += size_Wmat_bytes;
    }

    remain_bytes = lwork_bytes - total_bytes;

    {
        if(remain_bytes < 0)
        {
            return (rocblas_status_memory_error);
        }
    }

    {
        // ----------
        // step (1a) W = C1
        // ----------
        char const trans = 'N';
        I const mm = nrows_C1;
        I const nn = ncols_C1;
        T const alpha = 1;
        T const beta = 0;
        Istride const stride_alpha = 0;

        {
            // clang-format off
            geadd_template(handle,
			    trans,
			    mm, nn,
			    alpha,
			    Cmat, shift_C1, ldC, stride_Cmat,
			    beta,
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    batch_count);
            // clang-format on
        }
    }

    {
        // --------------------------------
        // step (1b) W = (Y1') * W, use trmm
        // --------------------------------

        rocblas_side const side = rocblas_side_left;
        rocblas_operation const trans = (rocblas_is_complex<T>)
            ? rocblas_operation_conjugate_transpose
            : rocblas_operation_transpose;
        rocblas_diagonal const diag = rocblas_diagonal_unit;
        rocblas_fill const uplo = rocblas_fill_lower;

        T alpha = 1;
        Istride const stride_alpha = 0;

        auto const mm = nrows_W;
        auto const nn = ncols_W;

        assert(nrows_W == ncols_Y1);
        assert(nrows_W == nrows_Y1);

        size_t size_trmm_bytes = 0;
        rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes);
        T** const workArr = reinterpret_cast<T**>(pfree);
        pfree += size_trmm_bytes;

        total_bytes += size_trmm_bytes;

        remain_bytes = lwork_bytes - total_bytes;
        {
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        if(use_trmm_outofplace)
        {
            // clang-format off
	    ROCBLAS_CHECK(rocblasCall_trmm(handle,
			    side, uplo, trans, diag,
			    mm,nn,
			    &alpha, stride_alpha,
			    Ymat, shift_Y1, ldY, stride_Ymat,
			    Wmat,  shift_Wmat, ldW, stride_Wmat,
			    Wmat2, shift_Wmat, ldW, stride_Wmat,
			    batch_count, workArr ));
            // clang-format on

            swap(Wmat, Wmat2);
        }
        else
        {
            // clang-format off
	    ROCBLAS_CHECK(rocblasCall_trmm(handle,
			    side, uplo, trans, diag,
			    mm,nn,
			    &alpha, stride_alpha,
			    Ymat, shift_Y1, ldY, stride_Ymat,
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    batch_count, workArr ));
            // clang-format on
        }

        total_bytes = total_bytes - size_trmm_bytes;
        pfree = pfree - size_trmm_bytes;
    }

    // % ----------------
    // % (2) W = W + Y2' * C2;
    // % ----------------
    //
    // transA = 'C';
    // transB = 'N';
    // mm = size(W,1);
    // nn = size(W,2);
    // kk = size(C2,1);
    // alpha = 1;
    // beta = 1;
    // W = gemm( transA, transB, mm,nn,kk, alpha, Y2, C2, beta, W );

    {
        // -------------------------
        // step (2) W = W + Y2' * C2
        // -------------------------

        rocblas_operation const transA = (rocblas_is_complex<T>)
            ? rocblas_operation_conjugate_transpose
            : rocblas_operation_transpose;
        rocblas_operation const transB = rocblas_operation_none;

        auto const mm = nrows_W;
        auto const nn = ncols_W;
        auto const kk = nrows_C2;

        // note (nrows_W == ncols_Y2);
        // note (ncols_W == ncols_C2);
        // note (nrows_Y2 == nrows_C2);

        T alpha = 1;
        T beta = 1;

        size_t const size_workArr = sizeof(T*) * batch_count;
        T** const workArr = reinterpret_cast<T**>(pfree);
        pfree += size_workArr;
        total_bytes += size_workArr;

        remain_bytes = lwork_bytes - total_bytes;
        {
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        // clang-format off
	if (use_gemm_gemv) {
            ROCBLAS_CHECK( gemm_gemv( handle,
                            transA, transB,
                            mm, nn, kk,
                            &alpha,
                            Ymat, shift_Y2, ldY, stride_Ymat,
                            Cmat, shift_C2, ldC, stride_Cmat,
                            &beta,
                            Wmat, shift_Wmat, ldW, stride_Wmat,
                            batch_count,
                            workArr ));
	}
	else {

            ROCBLAS_CHECK(rocblasCall_gemm(handle,
                            transA, transB,
                            mm, nn, kk,
                            &alpha,
                            Ymat, shift_Y2, ldY, stride_Ymat,
                            Cmat, shift_C2, ldC, stride_Cmat,
                            &beta,
                            Wmat, shift_Wmat, ldW, stride_Wmat,
                            batch_count,
                            workArr ));
	}
        // clang-format on

        pfree = pfree - size_workArr;
        total_bytes = total_bytes - size_workArr;
    }

    // % ----------
    // % (3) W = T' * W;
    // % ----------
    //
    // side = 'L';
    // uplo = 'U';
    // transA = 'C';
    // cdiag = 'N';
    // mm = size(W,1);
    // nn = size(W,2);
    // alpha = 1;
    // W = trmm( side, uplo, transA, cdiag, mm,nn, alpha, T, W );

    {
        // ----------------------
        // step (3)  W = (T') * W
        // ----------------------

        rocblas_side const side = rocblas_side_left;
        rocblas_fill const uplo = rocblas_fill_upper;
        rocblas_operation const transA = (rocblas_is_complex<T>)
            ? rocblas_operation_conjugate_transpose
            : rocblas_operation_transpose;
        rocblas_diagonal const diag = rocblas_diagonal_non_unit;

        auto const mm = nrows_W;
        auto const nn = ncols_W;
        T alpha = 1;
        Istride const stride_alpha = 0;

        // note (nrows_W == ncols_T);
        // note (nrows_W == nrows_T);

        size_t size_trmm_bytes = 0;
        rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes);

        T** const workArr = reinterpret_cast<T**>(pfree);
        pfree += size_trmm_bytes;

        total_bytes += size_trmm_bytes;

        remain_bytes = lwork_bytes - total_bytes;
        {
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        if(use_trmm_outofplace)
        {
            // clang-format off
	    ROCBLAS_CHECK(rocblasCall_trmm<T>(handle,
			    side, uplo, transA, diag,
			    mm,nn,
			    &alpha, stride_alpha,
			    Tmat, shift_Tmat, ldT, stride_Tmat,
			    Wmat,  shift_Wmat, ldW, stride_Wmat,
			    Wmat2, shift_Wmat, ldW, stride_Wmat,
			    batch_count, workArr ));
            // clang-format on

            swap(Wmat, Wmat2);
        }
        else
        {
            // clang-format off
	    ROCBLAS_CHECK(rocblasCall_trmm<T>(handle,
			    side, uplo, transA, diag,
			    mm,nn,
			    &alpha, stride_alpha,
			    Tmat, shift_Tmat, ldT, stride_Tmat,
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    batch_count, workArr ));
            // clang-format on
        }

        pfree = pfree - size_trmm_bytes;
        total_bytes = total_bytes - size_trmm_bytes;
    }

    // % ----------------
    // % (4) C2 = C2 - Y2 * W;
    // % ----------------
    //
    // transA = 'N';
    // transB = 'N';
    // mm = size(C2,1);
    // nn = size(C2,2);
    // kk = size(W,1);
    // alpha = -1;
    // beta = 1;
    // C2 = gemm( transA, transB, mm,nn,kk,  alpha, Y2, W, beta, C2 );

    {
        // ----------------------------------
        // step (4)   C2 = C2 - Y2 * W, using gemm
        // ----------------------------------

        rocblas_operation const transA = rocblas_operation_none;
        rocblas_operation const transB = rocblas_operation_none;
        auto const mm = nrows_C2;
        auto const nn = ncols_C2;
        auto const kk = nrows_W;

        // note (nrows_C2 == nrows_Y2);
        // note (ncols_C2 == ncols_W);
        // note (ncols_Y2 == nrows_W);

        T alpha = -1;
        T beta = 1;

        size_t const size_workArr = sizeof(T*) * batch_count;
        T** const workArr = reinterpret_cast<T**>(pfree);
        pfree += size_workArr;

        total_bytes += size_workArr;

        remain_bytes = lwork_bytes - total_bytes;
        {
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        // clang-format off
	if (use_gemm_gemv) {
            ROCBLAS_CHECK( gemm_gemv( handle,
                            transA, transB,
                            mm, nn, kk,
                            &alpha,
                            Ymat, shift_Y2,   ldY, stride_Ymat,
                            Wmat, shift_Wmat, ldW, stride_Wmat,
                            &beta,
                            Cmat, shift_C2, ldC, stride_Cmat,
                            batch_count,
                            workArr ));
	}
	else {

            ROCBLAS_CHECK( rocblasCall_gemm( handle,
                            transA, transB,
                            mm, nn, kk,
                            &alpha,
                            Ymat, shift_Y2,   ldY, stride_Ymat,
                            Wmat, shift_Wmat, ldW, stride_Wmat,
                            &beta,
                            Cmat, shift_C2, ldC, stride_Cmat,
                            batch_count,
                            workArr ));
	}
        // clang-format on

        pfree = pfree - size_workArr;
        total_bytes = total_bytes - size_workArr;
    }

    // % ----------
    // % (5) W = Y1 * W, use trmm
    // % ----------
    // side = 'L';
    // uplo = 'L';
    // transA = 'N';
    // cdiag = 'U';
    // alpha = 1;
    // mm = size(W,1);
    // nn = size(W,2);
    // W = trmm( side, uplo, transA, cdiag, mm,nn, alpha, Y1, W );
    {
        // ---------------------
        // step (5)  W = Y1 * W, using trmm
        // ---------------------

        rocblas_side const side = rocblas_side_left;
        rocblas_fill const uplo = rocblas_fill_lower;
        rocblas_operation const transA = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_unit;

        T alpha = 1;
        Istride const stride_alpha = 0;

        I const mm = nrows_W;
        I const nn = ncols_W;

        // note (nrows_W == nrows_Y1);
        // note (nrows_W == ncols_Y1);

        size_t size_trmm_bytes = 0;
        rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes);

        size_t const size_workArr = size_trmm_bytes;
        T** const workArr = reinterpret_cast<T**>(pfree);
        pfree += size_trmm_bytes;

        total_bytes += size_trmm_bytes;

        remain_bytes = lwork_bytes - total_bytes;
        {
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        if(use_trmm_outofplace)
        {
            // clang-format off
		    ROCBLAS_CHECK( rocblasCall_trmm<T>( handle,
				    side, uplo, transA, diag,
				    mm, nn,
				    &alpha, stride_alpha,
				    Ymat,  shift_Y1,    ldY, stride_Ymat,
				    Wmat,  shift_Wmat,  ldW, stride_Wmat,
				    Wmat2, shift_Wmat,  ldW, stride_Wmat,
				    batch_count,
				    workArr ));
            // clang-format on

            swap(Wmat, Wmat2);
        }
        else
        {
            // clang-format off
		    ROCBLAS_CHECK( rocblasCall_trmm<T>( handle,
				    side, uplo, transA, diag,
				    mm, nn,
				    &alpha, stride_alpha,
				    Ymat, shift_Y1, ldY, stride_Ymat,
				    Wmat, shift_Wmat,  ldW, stride_Wmat,
				    batch_count,
				    workArr ));
            // clang-format on
        }
        pfree = pfree - size_trmm_bytes;
        total_bytes = total_bytes - size_trmm_bytes;
    }

    //  * -----------
    //  * C1 = C1 - W
    //  * -----------
    {
        char const trans = 'N';
        auto const mm = nrows_W;
        auto const nn = ncols_W;

        // note (nrows_W == nrows_C1);
        // note (ncols_W == ncols_C1);

        T alpha = -1;
        T beta = 1;

        {
            // clang-format off
	      geadd_template( handle,
                    trans,
                    mm,
                    nn,
                    alpha,
                    Wmat, shift_Wmat, ldW, stride_Wmat,
                    beta,
                    Cmat, shift_Cmat, ldC, stride_Cmat,
                    batch_count );
            // clang-format on
        }
    }

    return (rocblas_status_success);
}

template <typename T, typename I, typename UY, typename Istride>
static rocblas_status applyQtC(rocblas_handle handle,
                               I const m,
                               I const n,
                               I const k,

                               UY Ymat,
                               Istride const shift_Ymat,
                               I const ldY,
                               Istride const stride_Ymat,

                               T* const Tmat,
                               Istride const shift_Tmat,
                               I const ldT,
                               Istride const stride_Tmat,

                               UY Cmat,
                               Istride const shift_Cmat,
                               I const ldC,
                               Istride const stride_Cmat,

                               I const batch_count,
                               void* work,
                               I& lwork_bytes)
{
    ROCSOLVER_ENTER("applyQtC_body", "m:", m, "n:", n, "k:", k, "shift_Ymat:", shift_Ymat,
                    "ldY:", ldY, "bc:", batch_count);
    if(is_applyQtC_use_larfb)
    {
        // TODO: check  correct setting for BATCHED and STRIDED
        bool constexpr BATCHED = true;
        bool constexpr STRIDED = true;

        rocblas_side const side = rocblas_side_left;
        rocblas_operation const trans = rocblas_operation_conjugate_transpose;
        rocblas_direct const direct = rocblas_forward_direction;
        rocblas_storev const storev = rocblas_column_wise;

        size_t size_tmptr = 0;
        size_t size_workArr = 0;
        rocsolver_larfb_getMemorySize<BATCHED, T>(side, m, n, k, batch_count, &size_tmptr,
                                                  &size_workArr);

        std::byte* pfree = reinterpret_cast<std::byte*>(work);

        T* const tmptr = reinterpret_cast<T*>(pfree);
        pfree += size_tmptr;

        T** const workArr = reinterpret_cast<T**>(pfree);
        pfree += size_workArr;

        bool const is_mem_ok = ((size_tmptr + size_workArr) <= lwork_bytes);
        if(!is_mem_ok)
        {
            return (rocblas_status_memory_error);
        }

        return (rocsolver_larfb_template<BATCHED, STRIDED, T, UY>(handle, side, trans, direct, storev,

                                                                  m, n, k,

                                                                  Ymat, shift_Ymat, ldY, stride_Ymat,

                                                                  Tmat, shift_Tmat, ldT, stride_Tmat,

                                                                  Cmat, shift_Cmat, ldC, stride_Cmat,

                                                                  batch_count, tmptr, workArr));
    }
    else
    {
        return (applyQtC_body(handle, m, n, k,

                              Ymat, shift_Ymat, ldY, stride_Ymat,

                              Tmat, shift_Tmat, ldT, stride_Tmat,

                              Cmat, shift_Cmat, ldC, stride_Cmat,

                              batch_count, work, lwork_bytes));
    }

    return (rocblas_status_success);
}

template <typename T, typename I>
static void rocsolver_applyQtC_getMemorySize(I const m,
                                             I const n,
                                             I const k,
                                             I const batch_count,
                                             size_t* size_applyQtC)
{
    assert(size_applyQtC != nullptr);
    *size_applyQtC = 0;

    bool const has_work = (m >= 1) && (n >= 1) && (k >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        *size_applyQtC = 0;
        return;
    }

    I const nb = RGEQR3_BLOCKSIZE(T);

    size_t size_trmm_byte = 0;
    {
        rocblas_side const side = rocblas_side_left;
        auto const mm = std::max(k, nb);
        auto const nn = std::max(n, nb);
        rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_byte);
    }

    size_t const size_rocblas_byte = 2 * sizeof(T*) * batch_count;
    size_t const size_Wmat_byte = (sizeof(T) * std::max(k, nb) * std::max(n, nb)) * batch_count;

    *size_applyQtC = size_trmm_byte + size_rocblas_byte + size_Wmat_byte;

    if(use_trmm_outofplace)
    {
        *size_applyQtC += size_Wmat_byte;
    }

    if(is_applyQtC_use_larfb)
    {
        bool constexpr BATCHED = true;
        rocblas_side const side = rocblas_side_left;

        size_t size_tmptr = 0;
        size_t size_workArr = 0;
        rocsolver_larfb_getMemorySize<BATCHED, T>(side, m, n, k, batch_count,

                                                  &size_tmptr, &size_workArr);

        *size_applyQtC = std::max(*size_applyQtC, size_tmptr + size_workArr);
    }
}

// --------------------------------------------------
// Algorithm inspired by the paper
// "Applying recursion to serial and parallel QR factorization"
// by Erik Elmroth and Fred G. Gustavson
// IBM Journal of Research and Development, August 2000
//
//
//  Input A(0:(m-1), 0:(n-1))
//  Output (Y, R, T),  Y and R replaces A, m >= n
//
//  R is n by n upper triangular
//  Y is lower trapezoidal with ones's on the diagonal
//  or Y(i,i) == 1, for i=0:(n-1)
//
//  Note T is n by n  upper triangular matrix
//  The diagonal entries T(i,i) are the "tau" values
//  in lapack GEQRF
// --------------------------------------------------

template <typename T, typename I, typename U, typename Istride>
static rocblas_status rocsolver_rgeqr3_template(rocblas_handle handle,
                                                const I m,
                                                const I n,

                                                U Amat,
                                                const Istride shift_Amat,
                                                const I ldA,
                                                const Istride stride_Amat,

                                                T* const Tmat,
                                                const Istride shift_Tmat,
                                                const I ldT,
                                                const Istride stride_Tmat,

                                                const I batch_count,

                                                void* work,
                                                I& lwork_bytes)
{
    ROCSOLVER_ENTER("rgeqr3", "m:", m, "n:", n, "shift_Amat:", shift_Amat, "lda:", ldA,
                    "bc:", batch_count);

    I total_bytes = 0;
    I remain_bytes = lwork_bytes;

    // quick return
    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return rocblas_status_success;
        }

        if(work == nullptr)
        {
            return (rocblas_status_invalid_pointer);
        }
    }

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    std::byte* pfree = reinterpret_cast<std::byte*>(work);

    auto idx2F
        = [=](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    auto ceil = [](auto m, auto n) { return (1 + (m - 1) / n); };

    I const n_small = get_n_small(); // terminate recursion for small n
    if(n == 1)
    {
        // ------------------------------------------------------
        // generate Householder reflector to work on first column
        // ------------------------------------------------------

        size_t size_work_byte = 0;
        size_t size_norms_byte = 0;
        rocsolver_larfg_getMemorySize<T>(n, batch_count, &size_work_byte, &size_norms_byte);

        // ----------------
        // allocate scratch storage
        // ----------------

        T* const dwork = reinterpret_cast<T*>(pfree);
        pfree += size_work_byte;
        total_bytes += size_work_byte;

        T* const norms = reinterpret_cast<T*>(pfree);
        pfree += size_norms_byte;
        total_bytes += size_work_byte;

        T* tau = Tmat + shift_Tmat;
        Istride const stride_tau = stride_Tmat;
        I const ldtau = 1;
        Istride const shift_tau = 0;

        remain_bytes = lwork_bytes - total_bytes;
        {
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        auto alpha = Amat;
        I const shifta = shift_Amat;
        auto x = Amat;
        I const shiftx = shift_Amat + 1;
        I const incx = 1;
        Istride const stridex = stride_Amat;

        {
            rocsolver_larfg_template(handle, m, alpha, shifta, x, shiftx, incx, stridex, tau,
                                     stride_tau, batch_count, dwork, norms);
        }

        pfree = pfree - size_norms_byte;
        pfree = pfree - size_work_byte;

        total_bytes = total_bytes - (size_norms_byte + size_work_byte);
    }
    else if(use_geqr2 && (n <= n_small))
    {
        // -------------------------------
        // terminate recursion for small n
        // to use GEQR2()
        // -------------------------------

        // ------------------------------------------------------
        // call geqr2 to perform QR factorization of narrow panel
        // of size (m by n), n <= n_small
        // ------------------------------------------------------
        {
            bool constexpr BATCHED = true;
            size_t size_scalars = 0;
            size_t size_work_workArr = 0;
            size_t size_Abyx_norms = 0;
            size_t size_diag = 0;

            auto const min_mn = std::min(m, n);

            rocsolver_geqr2_getMemorySize<BATCHED, T, I>(m, n, batch_count,

                                                         &size_scalars, &size_work_workArr,
                                                         &size_Abyx_norms, &size_diag);

            size_t size_geqr2 = 0;
            size_t const size_tau = min_mn * batch_count;
            Istride const stride_tau = min_mn;

            T* const tau = reinterpret_cast<T*>(pfree);
            pfree += size_tau;
            size_geqr2 += size_tau;

            T* const scalars = reinterpret_cast<T*>(pfree);
            pfree += size_scalars;
            size_geqr2 += size_scalars;

            void* const work_workArr = reinterpret_cast<void*>(pfree);
            pfree += size_work_workArr;
            size_geqr2 += size_work_workArr;

            T* const Abyx_norms = reinterpret_cast<T*>(pfree);
            pfree += size_Abyx_norms;
            size_geqr2 += size_Abyx_norms;

            T* const diag = reinterpret_cast<T*>(diag);
            pfree += size_diag;
            size_geqr2 += size_diag;

            bool const is_mem_ok = (size_geqr2 <= lwork_bytes);
            assert(size_geqr2 <= lwork_bytes);
            if(!is_mem_ok)
            {
                return (rocblas_status_memory_error);
            }

            ROCBLAS_CHECK(rocsolver_geqr2_template(handle, m, n, Amat, shift_Amat, ldA, stride_Amat,

                                                   tau, stride_tau,

                                                   batch_count, scalars, work_workArr, Abyx_norms,
                                                   diag));

            // -----------------------------------
            // copy tau vector to diagonal of Tmat
            // -----------------------------------

            {
                bool const Tmat2tau = false;
                copy_diagonal_template(handle, Tmat2tau, min_mn,

                                       Tmat, shift_Tmat, ldT, stride_Tmat,

                                       tau, stride_tau,

                                       batch_count);
            }

            pfree = pfree - size_geqr2;
        }

        // -----------------------------------
        // call larft to generate the T matrix
        // -----------------------------------

        {
            bool constexpr BATCHED = true;

            size_t size_scalars = 0;
            size_t size_work = 0;
            size_t size_workArr = 0;

            rocsolver_larft_getMemorySize<BATCHED, T>(m, n, batch_count,

                                                      &size_scalars, &size_work, &size_workArr);

            T* const scalars = reinterpret_cast<T*>(pfree);
            pfree += size_scalars;

            T* const work = reinterpret_cast<T*>(pfree);
            pfree += size_work;

            T** const workArr = reinterpret_cast<T**>(pfree);
            pfree += size_workArr;

            size_t const size_tmp_tau = sizeof(T) * n * batch_count;
            Istride const stride_tmp_tau = sizeof(T) * n;

            T* const tmp_tau = reinterpret_cast<T*>(pfree);
            pfree += size_tmp_tau;

            size_t size_larft = size_scalars + size_work + size_workArr;

            assert((size_larft + size_tmp_tau) <= lwork_bytes);

            {
                bool const Tmat2tau = true;
                copy_diagonal_template<T, I, Istride>(handle, Tmat2tau, n,

                                                      Tmat, shift_Tmat, ldT, stride_Tmat,

                                                      tmp_tau, stride_tmp_tau,

                                                      batch_count);
            }

            rocblas_direct const direct = rocblas_forward_direction;
            rocblas_storev const storev = rocblas_column_wise;

            auto const istat_larft = rocsolver_larft_template(handle, direct, storev, m, n,

                                                              Amat, shift_Amat, ldA, stride_Amat,

                                                              tmp_tau, stride_tmp_tau,

                                                              Tmat, ldT, stride_Tmat,

                                                              batch_count, scalars, work, workArr);

            if(istat_larft != rocblas_status_success)
            {
                return (istat_larft);
            }

            pfree = pfree - (size_larft + size_tmp_tau);
        }
    }
    else
    {
        // -----------------
        // perform recursion
        // -----------------

        I const n_small = get_n_small();
        I n1 = (n <= n_small) ? 1 : rocblas_previous_po2(ceil(n, 2));
        I n2 = n - n1;
        // fall back
        if((n1 == 0) || (n2 == 0))
        {
            n1 = n / 2;
            n2 = n - n1;
        }

        assert(n1 >= 1);
        assert(n2 >= 1);

        I const j1 = n1 + 1;
        I const m2 = (m - j1 + 1);

        I const k1 = n1;
        I const k2 = n2;

        // --------------------------------------------
        // [Y1, R1, T1 ] = rgeqr3( A(1:(m), 1:(n1))
        //  where Q1 = eye - Y1 * T1 * Y1'
        // --------------------------------------------

        // ---------------------------------------------
        // Note: Ymat reuses storage in lower trapezoidal
        // part of original Amat
        // ---------------------------------------------
        auto const Ymat = Amat;
        auto const shift_Ymat = shift_Amat;
        auto const stride_Ymat = stride_Amat;
        auto const ldY = ldA;

        auto const shift_Y1 = shift_Ymat + idx2F(1, 1, ldY);
        auto const shift_T1 = shift_Tmat + idx2F(1, 1, ldT);

        auto const nrows_Y1 = m;
        auto const ncols_Y1 = n1;

        auto const nrows_T1 = n1;
        auto const ncols_T1 = n1;

        {
            auto const mm = m;
            auto const nn = n1;

            // clang-format off
              ROCBLAS_CHECK(rocsolver_rgeqr3_template(
                handle,
		mm, nn,
	        Amat, shift_Amat, ldA, stride_Amat,
		Tmat, shift_Tmat, ldT, stride_Tmat,
		batch_count,
		pfree, remain_bytes));
            // clang-format on
        }

        // -----------------------------------------------------
        //
        // compute A(1:m, j1:n) = Q1' * A(1:m, j1:n)
        //
        // where Q1 = eye - Y1 * T1 * Y1',
        // and Y1 is lower trapezoidal with unit diagonal
        //
        // A(1:m,j1:n) = A(1:m,j1:n) - ...
        //   Y1(1:m,1:n1) * (T1(1:n1,1:n1) * (Y1(1:m,1:n1)'*A(1:m,j1:n)));
        // -----------------------------------------------------
        {
            // --------------------------
            // Note: C is alias of A(1:m, j1:n)
            // --------------------------
            auto const Cmat = Amat;
            auto const ldC = ldA;
            auto const shift_Cmat = shift_Amat + idx2F(1, j1, ldA);
            auto const stride_Cmat = stride_Amat;

            auto const mm = m;
            auto const nn = (n - j1 + 1);
            auto const kk = n1;

            {
                // clang-format off
	      ROCBLAS_CHECK( applyQtC( handle,
				    mm, nn, kk,
				    Ymat, shift_Y1, ldY, stride_Ymat,
				    Tmat, shift_T1, ldT, stride_Tmat,
				    Cmat, shift_Cmat, ldC, stride_Cmat,
				    batch_count,
				    pfree,
				    remain_bytes
				    ) );
                // clang-format on
            }
        }

        // -----------------------------------------
        // [Y2, R2, T2 ] = rgeqr3( A( j1:m, j1:n ) )
        // -----------------------------------------

        {
            auto const mm = (m - j1 + 1);
            auto const nn = (n - j1 + 1);
            auto const shift_A2 = shift_Amat + idx2F(j1, j1, ldA);
            auto const shift_T2 = shift_Tmat + idx2F(j1, j1, ldT);

            {
                // clang-format off
                    ROCBLAS_CHECK(rocsolver_rgeqr3_template(
                        handle,
			mm, nn,
			Amat, shift_A2, ldA, stride_Amat,
                        Tmat, shift_T2, ldT, stride_Tmat,
                        batch_count, pfree, remain_bytes));
                // clang-format on
            }
        }

        // % ------------------------------------------
        // % compute T3 = T(1:n1,j1:n) = -T1(Y1' Y2) T2
        // % ------------------------------------------
        //
        // kk = size(Y1,1) - size(Y2,1);
        // % T3 = -T1 * (Y1' * [ zeros(kk,1); Y2(:)]) * T2;
        // T3 = formT3(  Y1, T1, Y2, T2 );
        // %  T(1:n1,j1:n) = T3;
        // % ------------------------------------------

        {
            // -------------------------------------------------------
            // compute T3 = T(1:n1,j1:n) = -T1(Y1' Y2) T2
            //
            // Note that
            // Y1 is m by n1 unit lower trapezoidal,
            // Y2 is (m-n1) by n2 lower trapezoidal
            // ------------------------------------
            auto const Ymat = Amat;
            Istride const shift_Y1 = shift_Ymat + idx2F(1, 1, ldY);
            Istride const shift_Y2 = shift_Ymat + idx2F(j1, j1, ldY);
            I const ldY = ldA;
            Istride const stride_Ymat = stride_Amat;

            Istride const shift_T1 = shift_Tmat + idx2F(1, 1, ldT);
            Istride const shift_T2 = shift_Tmat + idx2F(j1, j1, ldT);
            Istride const shift_T3 = shift_Tmat + idx2F(1, j1, ldT);

            I const kk1 = n1;
            I const kk2 = n2;
            I const mm = m;

            // -------------------
            // Note: reuse Wmat as T3
            // Let T1 be n1 by n1
            //     T2 be n2 by n2
            // then T3 is n1 by n2
            // -------------------

            {
                // clang-format off
		    ROCBLAS_CHECK( formT3( handle,
					    mm,  kk1, kk2,
					    Ymat, shift_Y1, shift_Y2, ldY, stride_Ymat,
					    Tmat, shift_T1, shift_T2, ldT, stride_Tmat,
					    batch_count,
					    pfree, remain_bytes ));
                // clang-format on
            }
        }

        // --------------------------------------------------------------
        // implicitly form Y = [Y1, Y2] where Y is unit lower trapezoidal
        // Note Y over-writes lower part of A
        // --------------------------------------------------------------
        //

        // -----------------------------------
        // R = [ R1     A(0:(n1-1), n1:(n-1)) ]
        //     [ 0      R2                    ]
        //
        // Note R is n by n upper triangular
        // and over-writes matrix A
        // -----------------------------------

        // -----------------------------------
        // T = [ T1     T3 ]
        //     [ 0      T2 ]
        // -----------------------------------
    }

    return (rocblas_status_success);
}

// --------------------------------------
// estimate the amount of scratch memory
// required by rgeqr3()
//
// This is an over-estimation by
// it should require  O(  (nb^2) log(nb)  * batch_count)
// so is still a relatively small amount of storage
// --------------------------------------
template <typename T, typename I>
static void rocsolver_rgeqr3_getMemorySize(I const m, I const n, I const batch_count, size_t* work_size)
{
    *work_size = 0;
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    bool constexpr is_batched = true;
    auto const nb = RGEQR3_BLOCKSIZE(T);

    auto const nlevels = 1 + std::floor(std::log2(static_cast<double>(n)));

    size_t const size_rocblas = (2 * sizeof(T*) * batch_count) * nlevels;
    *work_size += size_rocblas;

    size_t const size_applyQtC = (sizeof(T) * nb * nb) * batch_count * nlevels;
    *work_size += size_applyQtC;

    size_t size_tau = (sizeof(T) * nb) * batch_count;
    *work_size += size_tau;

    if(is_applyQtC_use_larfb)
    {
        {
            size_t size_geqr2 = 0;
            // -----------------
            // scratch space for geqr2
            // -----------------
            size_t size_scalars = 0;
            size_t size_work_workArr = 0;
            size_t size_Abyx_norms = 0;
            size_t size_diag = 0;

            rocsolver_geqr2_getMemorySize<is_batched, T>(
                m, n, batch_count, &size_scalars, &size_work_workArr, &size_Abyx_norms, &size_diag);

            size_geqr2 = (size_scalars + size_work_workArr + size_Abyx_norms + size_diag);
            *work_size += size_geqr2;
        }

        // -----------------------
        // scratch space for larft
        // -----------------------
        {
            size_t size_scalars = 0;
            size_t size_work = 0;
            size_t size_workArr = 0;

            auto const nn = m;
            auto const kk = nb;
            rocsolver_larft_getMemorySize<is_batched, T>(nn, kk, batch_count, &size_scalars,
                                                         &size_work, &size_workArr);

            size_t const size_larft = (size_scalars + size_work + size_workArr);
            *work_size += size_larft;
        }
    }
    size_t size_Wm = (sizeof(T) * nb * nb) * batch_count;
    *work_size += size_Wm;

    if(use_geqr2)
    {
        I const n_small = get_n_small();

        // -----------------------
        // scratch space for geqr2
        // -----------------------
        {
            bool constexpr BATCHED = true;
            I const n_small = get_n_small();

            size_t size_scalars = 0;
            size_t size_work_workArr = 0;
            size_t size_Abyx_norms = 0;
            size_t size_diag = 0;

            rocsolver_geqr2_getMemorySize<BATCHED, T, I>(m, n_small, batch_count, &size_scalars,
                                                         &size_work_workArr, &size_Abyx_norms,
                                                         &size_diag);

            size_t const size_geqr2 = size_scalars + size_work_workArr + size_Abyx_norms + size_diag;
            *work_size += size_geqr2;
        }

        // -----------------------
        // scratch space for larft
        // -----------------------

        {
            bool constexpr BATCHED = true;
            size_t size_scalars = 0;
            size_t size_work = 0;
            size_t size_workArr = 0;

            rocsolver_larft_getMemorySize<BATCHED, T>(m, n_small, batch_count,

                                                      &size_scalars, &size_work, &size_workArr);
            size_t const size_larft = size_scalars + size_work + size_workArr;

            *work_size += size_larft;

            size_t const size_tmp_tau = (sizeof(T) * n_small) * batch_count;
            *work_size += size_tmp_tau;
        }
    }
}

// ----------------------------------------------------------
// perform recursive QR factorization but intended for m >= n
// tall skinny matrix
// ----------------------------------------------------------
template <typename T, typename I, typename UA, typename Istride>
static rocblas_status rocsolver_rgeqrf_template(rocblas_handle handle,
                                                I const m,
                                                I const n,

                                                UA Amat,
                                                Istride const shift_Amat,
                                                I const ldA,
                                                Istride const stride_Amat,

                                                T* tau_,
                                                Istride const stride_tau,

                                                I const batch_count,
                                                void* work,
                                                I& lwork_bytes)
{
    ROCSOLVER_ENTER("rgeqrf", "m:", m, "n:", n, "shift_Amat:", shift_Amat, "lda:", ldA,
                    "bc:", batch_count);

    I total_bytes = 0;
    I remain_bytes = 0;

    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return (rocblas_status_success);
    }

    if(work == nullptr)
    {
        return (rocblas_status_invalid_pointer);
    }

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // 1-based matlab/Fortran indexing
    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    I const nb = RGEQR3_BLOCKSIZE(T);

    std::byte* pfree = reinterpret_cast<std::byte*>(work);

    bool const zero_work_array = false;
    if(zero_work_array)
    {
        HIP_CHECK(hipMemsetAsync(work, 0, lwork_bytes, stream));
    }

    // -------------
    // allocate Wmat
    // -------------
    I const ldW = nb;
    size_t size_Wmat_bytes = (sizeof(T) * ldW * nb) * batch_count;
    Istride const stride_Wmat = ldW * nb;
    Istride const shift_Wmat = 0;
    T* const Wmat = reinterpret_cast<T*>(pfree);
    pfree += size_Wmat_bytes;

    total_bytes += size_Wmat_bytes;
    remain_bytes = lwork_bytes - total_bytes;

    // -------------
    // allocate Tmat
    // -------------
    I const ldT = nb;
    size_t size_Tmat_bytes = (sizeof(T) * ldT * nb) * batch_count;
    Istride const stride_Tmat = ldT * nb;
    Istride const shift_Tmat = 0;

    T* Tmat = reinterpret_cast<T*>(pfree);
    pfree += size_Tmat_bytes;

    total_bytes += size_Tmat_bytes;
    remain_bytes = lwork_bytes - total_bytes;

    if(remain_bytes < 0)
    {
        return (rocblas_status_memory_error);
    }

    for(I j = 1; j <= n; j += nb)
    {
        I const jb = std::min(n - j + 1, nb);
        I const mm = (m - j + 1);
        I const nn = jb;

        // -------------------------------
        // factorize column panel
        //    [Y,R,T] = rgeqr3(  mm,nn,A(j:m, j:(j+jb-1) )  );
        // -------------------------------

        Istride const shift_Aj = shift_Amat + idx2F(j, j, ldA);

        {
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        {
            // clang-format off
                  ROCBLAS_CHECK(rocsolver_rgeqr3_template(
				handle,
				mm, nn,
				Amat, shift_Aj,   ldA, stride_Amat,
				Tmat, shift_Tmat, ldT, stride_Tmat,
				batch_count,
				pfree, remain_bytes));
            // clang-format on
        }

        // ----------------------------------------------------
        // copy diagonal entries from T matrix into "tau" array
        // ----------------------------------------------------

        {
            bool const Tmat2tau = true;
            copy_diagonal_template(handle, Tmat2tau, nn,

                                   Tmat, shift_Tmat, ldT, stride_Tmat,

                                   tau_, stride_tau,

                                   batch_count);
        }

        // -----------------------------------------------------------
        // update A(j:m,(j+jb):n) = applyQtC( Y, T, A(j:m,(j+jb):n ) );
        // -----------------------------------------------------------

        {
            I const mm = (m - j + 1);
            I const nn = n - (j + jb) + 1;
            I const kk = jb;

            auto Ymat = Amat;
            Istride const shift_Y1 = shift_Amat + idx2F(j, j, ldA);
            I const ldY = ldA;
            Istride const stride_Ymat = stride_Amat;

            Istride const shift_T1 = shift_Tmat + idx2F(1, 1, ldT);

            auto Cmat = Amat;
            Istride const shift_Cmat = shift_Amat + idx2F(j, (j + jb), ldA);
            I const ldC = ldA;
            Istride const stride_Cmat = stride_Amat;

            {
                // clang-format off
	                ROCBLAS_CHECK( applyQtC( handle,
				    mm, nn, kk,
				    Ymat, shift_Y1, ldY, stride_Ymat,
				    Tmat, shift_T1, ldT, stride_Tmat,
				    Cmat, shift_Cmat, ldC, stride_Cmat,
				    batch_count,
				    pfree,
				    remain_bytes
				    ) );
                // clang-format on
            }
        }

    } // for j

    return (rocblas_status_success);
}

template <typename T, typename I>
static void
    rocsolver_rgeqrf_getMemorySize(I const m, I const n, I const batch_count, size_t* size_rgeqrf)
{
    assert(size_rgeqrf != nullptr);
    *size_rgeqrf = 0;

    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        *size_rgeqrf = 0;
        return;
    }

    auto const nb = RGEQR3_BLOCKSIZE(T);
    size_t const size_Wmat = (sizeof(T) * nb * std::max(n, nb)) * batch_count;
    size_t const size_Tmat = (sizeof(T) * nb * nb) * batch_count;
    size_t const size_rocblas = (2 * sizeof(T*)) * batch_count;

    size_t size_rgeqr3 = 0;
    rocsolver_rgeqr3_getMemorySize<T>(m, nb, batch_count, &size_rgeqr3);

    *size_rgeqrf = size_Wmat + size_Tmat + size_rocblas + size_rgeqr3;
    if(use_trmm_outofplace)
    {
        *size_rgeqrf += size_Wmat;
    }
}

ROCSOLVER_END_NAMESPACE
