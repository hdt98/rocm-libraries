/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "ideal_sizes.hpp"
#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "rocblas_utility.hpp"
#include "rocsolver_run_specialized_kernels.hpp"
#include <algorithm>

ROCSOLVER_BEGIN_NAMESPACE

/******************************************************************************
 * LACN2: 1-Norm Estimator for Inverse Matrices
 *
 * This is an internal utility routine that estimates ||A⁻¹||₁ using a
 * reverse-communication iterative algorithm based on power iteration.
 *
 * ALGORITHM OVERVIEW:
 * ------------------
 * LACN2 uses a stateful iteration pattern where the caller:
 * 1. Calls LACN2 to get the next operation to perform (via kase parameter)
 * 2. Performs the requested matrix solve (A*x or A'*x)
 * 3. Calls LACN2 again with the result
 * 4. Repeats until kase = 0 (convergence)
 *
 * STATE MACHINE:
 * -------------
 * kase = 0: Converged (d_est contains the final estimate of ||A⁻¹||₁)
 * kase = 1: Caller should solve A*x = x and call LACN2 again
 * kase = 2: Caller should solve A'*x = x and call LACN2 again
 *
 * jump values (internal state transitions):
 * - jump = 1: Initial iteration, compute first estimate
 * - jump = 2: Find maximum component and set up unit vector
 * - jump = 3: Check convergence criteria
 * - jump = 4: Update max index and check stopping conditions
 * - jump = 5: Finalize estimate
 *
 * DESIGN NOTES:
 * ------------
 * - This is an INTERNAL template utility, not a public API
 * - LACN2 only computes the estimate; the caller computes rcond = 1/(||A|| * ||A⁻¹||)
 * - Typically converges in 3-5 iterations
 * - Thread-safe and suitable for batched operations (process one batch at a time)
 *
 * REFERENCE:
 * ---------
 * Based on LAPACK's SLACN2/DLACN2/CLACN2/ZLACN2 routines
 * Higham, N. J. (1988). FORTRAN codes for estimating the one-norm of a
 * real or complex matrix, with applications to condition estimation.
 * ACM Trans. Math. Softw., 14(4), 381-396.
 *****************************************************************************/

#ifndef LACN2_BLOCKSIZE
#define LACN2_BLOCKSIZE 1024
#endif

/**
 * lacn2_init_vector: Initialize a vector to a constant value
 */
template <typename T, typename I>
ROCSOLVER_KERNEL void lacn2_init_vector(T* x, const rocblas_int n, const T value)
{
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    for(I i = tid; i < n; i += hipBlockDim_x * hipGridDim_x)
    {
        if(i < n)
            x[i] = value;
    }
}

/**
 * lacn2_max_index: Device helper for finding max index via warp reduction
 * Reduces within a warp using shuffle operations on both value and index
 */
template <typename T, typename I>
__device__ inline void lacn2_max_index(const I n, T* local_max, I* local_max_index, rocblas_int offset)
{
    T compare_local_max = shift_left(*local_max, offset);
    T compare_local_index = shift_left(*local_max_index, offset);
    if(compare_local_max > *local_max)
    {
        *local_max = compare_local_max;
        *local_max_index = compare_local_index;
    }
}

/**
 * lacn2_jump1_n_equals_one: Special case for n=1
 * Simply stores the absolute value of x[0] as the estimate
 */
template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(LACN2_BLOCKSIZE)
    lacn2_jump1_n_equals_one(const I n, T* x, S* norm, I* isgn)
{
    rocblas_int tid = hipThreadIdx_x;

    if(tid == 0)
    {
        *norm = rocblas_abs(x[0]);
    }
}

/**
 * lacn2_jump1: First iteration - compute initial estimate and set up x
 * Computes sum of absolute values and normalizes x to unit entries
 */
template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(LACN2_BLOCKSIZE)
    lacn2_jump1(const I n, T* x, S* norm, I* isgn)
{
    rocblas_int tid = hipThreadIdx_x;

    __shared__ S sval[LACN2_BLOCKSIZE / WarpSize];

    // sum absolute values
    S sum = 0;
    for(I i = tid; i < n; i += LACN2_BLOCKSIZE)
    {
        sum += rocblas_abs(x[i]);

        if constexpr(rocblas_is_complex<T>)
        {
            S absxi = rocblas_abs(x[i]);
            if(absxi == 0)
            {
                x[i] = T(1);
            }
            else
            {
                x[i] = x[i] / absxi;
            }
        }
        else
        {
            if(x[i] >= 0)
            {
                x[i] = T(1);
                isgn[i] = T(1);
            }
            else
            {
                x[i] = T(-1);
                isgn[i] = T(-1);
            }
        }
    }

    // reduce within Warp
    sum += shift_left(sum, 1);
    sum += shift_left(sum, 2);
    sum += shift_left(sum, 4);
    sum += shift_left(sum, 8);
    sum += shift_left(sum, 16);
    if(WarpSize > 32)
        sum += shift_left(sum, 32);

    if(tid % WarpSize == 0)
        sval[tid / WarpSize] = sum;
    __syncthreads();

    if(tid == 0)
    {
        for(I k = 1; k < LACN2_BLOCKSIZE / WarpSize; k++)
            sum += sval[k];
        *norm = sum;
    }
}

/**
 * lacn2_jump2: Find index of maximum absolute value in vector x
 * Sets all x entries to 0 except the max element which is set to 1
 * Must be called with only a single block
 */
template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(LACN2_BLOCKSIZE) lacn2_jump2(const I n, T* x, I* max_idx)
{
    I tid = threadIdx.x;

    // shared variables
    __shared__ S sval[LACN2_BLOCKSIZE / WarpSize];
    __shared__ S sval_indices[LACN2_BLOCKSIZE / WarpSize];

    // find index of maximum absolute value in vector x
    S local_max = std::numeric_limits<S>::min();
    I local_max_index;
    for(I i = tid; i < n; i += LACN2_BLOCKSIZE)
    {
        if(rocblas_abs(x[i]) > local_max)
        {
            local_max = rocblas_abs(x[i]);
            local_max_index = i;
        }
        x[i] = T(0);
    }

    // reduce within Warp using shuffle on both index and value
    lacn2_max_index<S, I>(n, &local_max, &local_max_index, 1);
    lacn2_max_index<S, I>(n, &local_max, &local_max_index, 2);
    lacn2_max_index<S, I>(n, &local_max, &local_max_index, 4);
    lacn2_max_index<S, I>(n, &local_max, &local_max_index, 8);
    lacn2_max_index<S, I>(n, &local_max, &local_max_index, 16);
    if(WarpSize > 32)
        lacn2_max_index<S, I>(n, &local_max, &local_max_index, 32);

    if(tid % WarpSize == 0)
    {
        sval[tid / WarpSize] = local_max;
        sval_indices[tid / WarpSize] = local_max_index;
    }
    __syncthreads();

    if(tid == 0)
    {
        for(I k = 1; k < LACN2_BLOCKSIZE / WarpSize; k++)
        {
            if(rocblas_abs(sval[k]) > local_max)
            {
                local_max = rocblas_abs(sval[k]);
                local_max_index = sval_indices[k];
            }
        }
        *max_idx = local_max_index;
        x[local_max_index] = T(1);
    }
}

/**
 * lacn2_jump3: Check convergence criteria
 * Must always be launched with one block
 *
 * Three possible outcomes:
 * 1) Repeated signs: write alternating sign pattern, set kase=1, jump=5
 * 2) Not repeated but est <= est_old: same as (1)
 * 3) Not repeated and est > est_old: normalize x, set kase=2, jump=4
 */
template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(LACN2_BLOCKSIZE)
    lacn2_jump3(const I n, T* x, T* v, I* isgn, rocblas_int* kase, rocblas_int* jump, S* est)
{
    rocblas_int tid = hipThreadIdx_x;

    __shared__ S sval[LACN2_BLOCKSIZE / WarpSize];
    __shared__ bool sval_repeated[LACN2_BLOCKSIZE / WarpSize];
    __shared__ S sval_estold; // for broadcasting est_old to all warps

    // Sum absolute values
    S sum = 0;
    bool repeated = (rocblas_is_complex<T>) ? false : true;
    // we iterate over v, since v contains x from previous step (pointers swapped)
    for(I i = tid; i < n; i += LACN2_BLOCKSIZE)
    {
        sum += rocblas_abs(v[i]);
        if constexpr(!rocblas_is_complex<T>)
        {
            if(v[i] >= 0)
            {
                if(isgn[i] <= -1)
                    repeated = false;
            }
            else
            {
                if(isgn[i] >= 1)
                    repeated = false;
            }
        }
    }

    // reduce within Warp
    sum += shift_left(sum, 1);
    repeated = repeated && __shfl_down(repeated, 1);
    sum += shift_left(sum, 2);
    repeated = repeated && __shfl_down(repeated, 2);
    sum += shift_left(sum, 4);
    repeated = repeated && __shfl_down(repeated, 4);
    sum += shift_left(sum, 8);
    repeated = repeated && __shfl_down(repeated, 8);
    sum += shift_left(sum, 16);
    repeated = repeated && __shfl_down(repeated, 16);
    if(WarpSize > 32)
    {
        sum += shift_left(sum, 32);
        repeated = repeated && __shfl_down(repeated, 32);
    }

    if(tid % WarpSize == 0)
    {
        sval[tid / WarpSize] = sum;
        sval_repeated[tid / WarpSize] = repeated;
    }
    __syncthreads();

    if(tid == 0)
    {
        for(I k = 1; k < LACN2_BLOCKSIZE / WarpSize; k++)
        {
            sum += sval[k];
            repeated = repeated && sval_repeated[k];
        }
        sval_estold = *est; // broadcast
        *est = sum;
        sval_repeated[0] = repeated;
    }

    __syncthreads();

    repeated = sval_repeated[0];
    S estold = sval_estold; // read broadcast

    if(repeated || (*est <= estold))
    {
        for(I i = tid; i < n; i += LACN2_BLOCKSIZE)
        {
            // we write over old x
            T sign = (i % 2 == 0) ? T(1) : T(-1);
            x[i] = T(sign * (S(1) + S(i) / S(n - 1)));
        }
        if(tid == 0)
        {
            *kase = 1;
            *jump = 5;
        }

        return;
    }

    // we iterate over v, since v contains old x (pointers swapped)
    if constexpr(rocblas_is_complex<T>)
    {
        for(I i = tid; i < n; i += LACN2_BLOCKSIZE)
        {
            S absxi = rocblas_abs(v[i]);
            if(absxi == 0)
            {
                x[i] = T(1);
            }
            else
            {
                x[i] = v[i] / absxi;
            }
        }
    }
    else
    {
        for(I i = tid; i < n; i += LACN2_BLOCKSIZE)
        {
            if(v[i] >= 0)
            {
                x[i] = T(1);
                isgn[i] = T(1);
            }
            else
            {
                x[i] = T(-1);
                isgn[i] = T(-1);
            }
        }
    }

    if(tid == 0)
    {
        *kase = 2;
        *jump = 4;
    }
}

/**
 * lacn2_jump4: Update max index and check stopping conditions
 * Determines if iteration should continue or finalize
 */
template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(LACN2_BLOCKSIZE) lacn2_jump4(const I n,
                                                                     T* x,
                                                                     I* isgn,
                                                                     rocblas_int* kase,
                                                                     rocblas_int* jump,
                                                                     I* max_idx,
                                                                     const I iters,
                                                                     const I iters_max)
{
    I tid = threadIdx.x;

    int jlast = *max_idx;

    // shared variables
    __shared__ S sval[LACN2_BLOCKSIZE / WarpSize];
    __shared__ S sval_indices[LACN2_BLOCKSIZE / WarpSize];

    // find index of maximum absolute value in vector x
    S local_max = std::numeric_limits<S>::min();
    I local_max_index;
    for(I i = tid; i < n; i += LACN2_BLOCKSIZE)
    {
        if(rocblas_abs(x[i]) > local_max)
        {
            local_max = rocblas_abs(x[i]);
            local_max_index = i;
        }
    }

    // reduce within Warp using shuffle on both index and value
    lacn2_max_index<S, I>(n, &local_max, &local_max_index, 1);
    lacn2_max_index<S, I>(n, &local_max, &local_max_index, 2);
    lacn2_max_index<S, I>(n, &local_max, &local_max_index, 4);
    lacn2_max_index<S, I>(n, &local_max, &local_max_index, 8);
    lacn2_max_index<S, I>(n, &local_max, &local_max_index, 16);
    if(WarpSize > 32)
        lacn2_max_index<S, I>(n, &local_max, &local_max_index, 32);

    if(tid % WarpSize == 0)
    {
        sval[tid / WarpSize] = local_max;
        sval_indices[tid / WarpSize] = local_max_index;
    }
    __syncthreads();

    if(tid == 0)
    {
        for(I k = 1; k < LACN2_BLOCKSIZE / WarpSize; k++)
        {
            if(rocblas_abs(sval[k]) > local_max)
            {
                local_max = rocblas_abs(sval[k]);
                local_max_index = sval_indices[k];
            }
        }
        sval_indices[0] = local_max_index;
        *max_idx = local_max_index;
    }

    __syncthreads();

    I local_max_idx = sval_indices[0];
    S val_new = rocblas_abs(x[local_max_idx]);
    S val_old = rocblas_abs(x[jlast]);

    if(val_new == val_old || iters >= iters_max)
    {
        for(rocblas_int i = tid; i < n; i += LACN2_BLOCKSIZE)
        {
            T sign = (i % 2 == 0) ? T(1) : T(-1);
            x[i] = T(sign * (S(1) + S(i) / S(n - 1)));
        }
        if(tid == 0)
        {
            *kase = 1;
            *jump = 5;
        }

        return;
    }

    for(I i = tid; i < n; i += LACN2_BLOCKSIZE)
    {
        x[i] = T(0);
    }

    // ensure all threads done zeroing before writing to max element
    __syncthreads();

    if(tid == 0)
    {
        x[local_max_idx] = T(1);
        *kase = 1;
        *jump = 3;
    }
}

/**
 * lacn2_jump5: Finalize estimate
 * Computes final estimate and stores in norm
 */
template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(LACN2_BLOCKSIZE)
    lacn2_jump5(const I n, const T* x, S* norm)
{
    rocblas_int tid = hipThreadIdx_x;

    __shared__ S sval[LACN2_BLOCKSIZE / WarpSize];

    // sum absolute values
    S sum = 0;
    for(I i = tid; i < n; i += LACN2_BLOCKSIZE)
        sum += rocblas_abs(x[i]);

    // reduce within Warp
    sum += shift_left(sum, 1);
    sum += shift_left(sum, 2);
    sum += shift_left(sum, 4);
    sum += shift_left(sum, 8);
    sum += shift_left(sum, 16);
    if(WarpSize > 32)
        sum += shift_left(sum, 32);

    if(tid % WarpSize == 0)
        sval[tid / WarpSize] = sum;
    __syncthreads();

    if(tid == 0)
    {
        for(I k = 1; k < LACN2_BLOCKSIZE / WarpSize; k++)
            sum += sval[k];
        sum = 2 * (sum / (3 * n));
        if(sum > *norm)
            *norm = sum;
    }
}

// Main LACN2 host function - drives the reverse-communication iteration
// to estimate ||A^-1||_1. Caller performs matrix solves between iterations.
template <typename T, typename I, typename S>
rocblas_status rocsolver_lacn2_template(rocblas_handle handle,
                                        const I n,
                                        T** x, // passed by reference so swap persists
                                        T** v, // passed by reference so swap persists
                                        I* isgn,
                                        S* d_est,
                                        I* d_max_idx,
                                        rocblas_int* d_kase,
                                        rocblas_int* d_jump,
                                        rocblas_int* h_kase,
                                        rocblas_int* h_jump,
                                        I* h_iters,
                                        const I max_iters,
                                        hipStream_t stream)
{
    if(*h_kase == 0)
    {
        // initialize x = (1/n, ..., 1/n)
        rocblas_int blocks = (n - 1) / LACN2_BLOCKSIZE + 1;
        ROCSOLVER_LAUNCH_KERNEL((lacn2_init_vector<T, I>), dim3(blocks), dim3(LACN2_BLOCKSIZE), 0,
                                stream, *x, n, T(1) / T(n));
        *h_kase = 1;
        *h_jump = 1;
        return rocblas_status_success;
    }

    switch(*h_jump)
    {
    case 1:
        if(n == 1)
        {
            ROCSOLVER_LAUNCH_KERNEL((lacn2_jump1_n_equals_one<T, I, S>), dim3(1),
                                    dim3(LACN2_BLOCKSIZE), 0, stream, n, *x, d_est, isgn);
            *h_kase = 0; // signal to exit
            return rocblas_status_success;
        }

        ROCSOLVER_LAUNCH_KERNEL((lacn2_jump1<T, I, S>), dim3(1), dim3(LACN2_BLOCKSIZE), 0, stream,
                                n, *x, d_est, isgn);

        *h_kase = 2;
        *h_jump = 2;

        return rocblas_status_success;
        break;
    case 2:
        ROCSOLVER_LAUNCH_KERNEL((lacn2_jump2<T, I, S>), dim3(1), dim3(LACN2_BLOCKSIZE), 0, stream,
                                n, *x, d_max_idx);

        *h_kase = 1;
        *h_jump = 3;

        *h_iters = 2;

        return rocblas_status_success;
        break;
    case 3:
        std::swap(*x, *v);
        ROCSOLVER_LAUNCH_KERNEL((lacn2_jump3<T, I, S>), dim3(1), dim3(LACN2_BLOCKSIZE), 0, stream,
                                n, *x, *v, isgn, d_kase, d_jump, d_est);

        HIP_CHECK(hipMemcpyAsync(h_jump, d_jump, sizeof(rocblas_int), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(h_kase, d_kase, sizeof(rocblas_int), hipMemcpyDeviceToHost, stream));

        return rocblas_status_success;
        break;
    case 4:
        ROCSOLVER_LAUNCH_KERNEL((lacn2_jump4<T, I, S>), dim3(1), dim3(LACN2_BLOCKSIZE), 0, stream,
                                n, *x, isgn, d_kase, d_jump, d_max_idx, *h_iters, max_iters);

        HIP_CHECK(hipMemcpyAsync(h_jump, d_jump, sizeof(rocblas_int), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(h_kase, d_kase, sizeof(rocblas_int), hipMemcpyDeviceToHost, stream));

        if(*h_jump == 3)
            *h_iters = *h_iters + 1;

        return rocblas_status_success;
        break;
    case 5:
        ROCSOLVER_LAUNCH_KERNEL((lacn2_jump5<T, I, S>), dim3(1), dim3(LACN2_BLOCKSIZE), 0, stream,
                                n, *x, d_est);
        std::swap(*x, *v);
        *h_kase = 0;

        return rocblas_status_success;
        break;
    default: return rocblas_status_invalid_value; break;
    }
}

// Query workspace sizes needed for LACN2
template <typename T, typename I, typename S>
void rocsolver_lacn2_getMemorySize(const I n,
                                   size_t* size_work_x,
                                   size_t* size_work_v,
                                   size_t* size_work_isgn,
                                   size_t* size_scalar_est,
                                   size_t* size_scalar_max_idx,
                                   size_t* size_scalar_kase,
                                   size_t* size_scalar_jump)
{
    // n elements of type T for x vector
    *size_work_x = sizeof(T) * n;

    // n elements of type T for v vector
    *size_work_v = sizeof(T) * n;

    // n elements of type I for isgn vector (only used by real types, not complex)
    *size_work_isgn = rocblas_is_complex<T> ? 0 : sizeof(I) * n;

    // one S (real) scalar for estimate
    *size_scalar_est = sizeof(S);

    // one I scalar for max index
    *size_scalar_max_idx = sizeof(I);

    // one rocblas_int scalar for kase state
    *size_scalar_kase = sizeof(rocblas_int);

    // one rocblas_int scalar for jump state
    *size_scalar_jump = sizeof(rocblas_int);
}

ROCSOLVER_END_NAMESPACE
