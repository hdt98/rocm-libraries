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
#include "lapack/roclapack_getrs.hpp"
#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "rocblas_utility.hpp"
#include "rocsolver_run_specialized_kernels.hpp"
#include <algorithm>
#include <vector>

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I>
ROCSOLVER_KERNEL void gecon_init_vector(T* x, const rocblas_int n, const T value)
{
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    for(I i = tid; i < n; i += hipBlockDim_x * hipGridDim_x)
    {
        if(i < n)
            x[i] = value;
    }
}

// reduce within Warp using shuffle on both index and value
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

template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(GECON_BLOCKSIZE)
    lacn2_jump1_n_equals_one(const I n, T* x, S* norm, I* isgn, const S* d_anorm)
{
    rocblas_int tid = hipThreadIdx_x;

    if(tid == 0)
    {
        S est = rocblas_abs(x[0]);
        S anorm = *d_anorm;
        if(est != S(0) && anorm != S(0))
            *norm = (S(1) / est) / anorm;
        else
            *norm = S(0);
    }
}

template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(GECON_BLOCKSIZE)
    lacn2_jump1(const I n, T* x, S* norm, I* isgn)
{
    rocblas_int tid = hipThreadIdx_x;

    __shared__ S sval[GECON_BLOCKSIZE / WarpSize];

    // sum absolute values
    S sum = 0;
    for(I i = tid; i < n; i += GECON_BLOCKSIZE)
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
        for(I k = 1; k < GECON_BLOCKSIZE / WarpSize; k++)
            sum += sval[k];
        *norm = sum;
    }
}

// find index of maximum abosolute value in vector x
// to be called with only a single block
template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(GECON_BLOCKSIZE) lacn2_jump2(const I n, T* x, I* max_idx)
{
    I tid = threadIdx.x;

    // shared variables
    __shared__ S sval[GECON_BLOCKSIZE / WarpSize];
    __shared__ S sval_indices[GECON_BLOCKSIZE / WarpSize];

    // find index of maximum abosolute value in vector x
    S local_max = std::numeric_limits<S>::min();
    I local_max_index;
    for(I i = tid; i < n; i += GECON_BLOCKSIZE)
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
        for(I k = 1; k < GECON_BLOCKSIZE / WarpSize; k++)
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

// should always be launched with one block
// three outcomes
// 1) repeated
// has to iterate through entire vector
// write altsgn over x and write kase = 1 and jump = 5
// 2) not repeated, est <= est_old
// has to iterate through entire vector
// has to do reduction on v to compute new est
// write altsgn over x and write kase = 1 and jump = 5
// 3) not repeated, est > est_old
// do all previous steps except for writing altsgn over x and writing kase = 1 and jump = 5
// overwrite x with 1/-1
// overwrite isgn
// write back kase = 2 and jump = 4
template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(GECON_BLOCKSIZE)
    lacn2_jump3(const I n, T* x, T* v, I* isgn, rocblas_int* kase, rocblas_int* jump, S* est)
{
    rocblas_int tid = hipThreadIdx_x;

    __shared__ S sval[GECON_BLOCKSIZE / WarpSize];
    __shared__ bool sval_repeated[GECON_BLOCKSIZE / WarpSize];
    __shared__ S sval_estold; // for broadcasting est_old to all warps

    // Sum absolute values
    S sum = 0;
    bool repeated = (rocblas_is_complex<T>) ? false : true;
    // we iterate over v, since v contains x from previous step (pointers swapped)
    for(I i = tid; i < n; i += GECON_BLOCKSIZE)
    {
        sum += rocblas_abs(v[i]);
        if constexpr(rocblas_is_complex<T>)
            continue;
        else
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
        for(I k = 1; k < GECON_BLOCKSIZE / WarpSize; k++)
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
        for(I i = tid; i < n; i += GECON_BLOCKSIZE)
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
        for(I i = tid; i < n; i += GECON_BLOCKSIZE)
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
        for(I i = tid; i < n; i += GECON_BLOCKSIZE)
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

template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(GECON_BLOCKSIZE) lacn2_jump4(const I n,
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
    __shared__ S sval[GECON_BLOCKSIZE / WarpSize];
    __shared__ S sval_indices[GECON_BLOCKSIZE / WarpSize];

    // find index of maximum abosolute value in vector x
    S local_max = std::numeric_limits<S>::min();
    I local_max_index;
    for(I i = tid; i < n; i += GECON_BLOCKSIZE)
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
        for(I k = 1; k < GECON_BLOCKSIZE / WarpSize; k++)
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
        for(rocblas_int i = tid; i < n; i += GECON_BLOCKSIZE)
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

    for(I i = tid; i < n; i += GECON_BLOCKSIZE)
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

// compute l1 norm of vector v and compare to temporary value
template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(GECON_BLOCKSIZE)
    lacn2_jump5(const I n, const T* x, S* norm, const S* d_anorm)
{
    rocblas_int tid = hipThreadIdx_x;

    __shared__ S sval[GECON_BLOCKSIZE / WarpSize];

    // sum absolute values
    S sum = 0;
    for(I i = tid; i < n; i += GECON_BLOCKSIZE)
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
        for(I k = 1; k < GECON_BLOCKSIZE / WarpSize; k++)
            sum += sval[k];
        sum = 2 * (sum / (3 * n));
        if(sum > *norm)
            *norm = sum;

        S est = *norm;
        S anorm = *d_anorm;
        if(est != S(0) && anorm != S(0))
            *norm = (S(1) / est) / anorm;
        else
            *norm = S(0);
    }
}

// LACN2 host routine
template <typename T, typename I, typename S, typename U>
rocblas_status gecon_lacn2(rocblas_handle handle,
                           const rocsolver_norm_type norm_type,
                           const I n,
                           U A,
                           const rocblas_stride shiftA,
                           const I inca,
                           const I lda,
                           const rocblas_stride strideA,
                           const I* ipiv,
                           const rocblas_stride strideP,
                           hipStream_t stream,
                           T** v, // same type as A, passed by reference so swap persists
                           T** x, // passed by reference so swap persists
                           I* isgn,
                           const I max_iters,
                           I* h_iters,
                           I* d_max_idx,
                           S* d_est, // real_t<T>
                           const S* d_anorm, // real_t<T>
                           rocblas_int* d_jump,
                           rocblas_int* d_kase,
                           rocblas_int* h_jump,
                           rocblas_int* h_kase)
{
    if(*h_kase == 0)
    {
        // initialize x = (1/n, ..., 1/n)
        rocblas_int blocks = (n - 1) / GECON_BLOCKSIZE + 1;
        ROCSOLVER_LAUNCH_KERNEL((gecon_init_vector<T, I>), dim3(blocks), dim3(GECON_BLOCKSIZE), 0,
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
                                    dim3(GECON_BLOCKSIZE), 0, stream, n, *x, d_est, isgn, d_anorm);
            *h_kase = 0; // signal to exit
            return rocblas_status_success;
        }

        ROCSOLVER_LAUNCH_KERNEL((lacn2_jump1<T, I, S>), dim3(1), dim3(GECON_BLOCKSIZE), 0, stream,
                                n, *x, d_est, isgn);

        *h_kase = 2;
        *h_jump = 2;

        return rocblas_status_success;
        break;
    case 2:
        ROCSOLVER_LAUNCH_KERNEL((lacn2_jump2<T, I, S>), dim3(1), dim3(GECON_BLOCKSIZE), 0, stream,
                                n, *x, d_max_idx);

        *h_kase = 1;
        *h_jump = 3;

        *h_iters = 2;

        return rocblas_status_success;
        break;
    case 3:
        std::swap(*x, *v);
        ROCSOLVER_LAUNCH_KERNEL((lacn2_jump3<T, I, S>), dim3(1), dim3(GECON_BLOCKSIZE), 0, stream,
                                n, *x, *v, isgn, d_kase, d_jump, d_est);

        HIP_CHECK(hipMemcpyAsync(h_jump, d_jump, sizeof(rocblas_int), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(h_kase, d_kase, sizeof(rocblas_int), hipMemcpyDeviceToHost, stream));

        return rocblas_status_success;
        break;
    case 4:
        ROCSOLVER_LAUNCH_KERNEL((lacn2_jump4<T, I, S>), dim3(1), dim3(GECON_BLOCKSIZE), 0, stream,
                                n, *x, isgn, d_kase, d_jump, d_max_idx, *h_iters, max_iters);

        HIP_CHECK(hipMemcpyAsync(h_jump, d_jump, sizeof(rocblas_int), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(h_kase, d_kase, sizeof(rocblas_int), hipMemcpyDeviceToHost, stream));

        if(*h_jump == 3)
            *h_iters = *h_iters + 1;

        return rocblas_status_success;
        break;
    case 5:
        ROCSOLVER_LAUNCH_KERNEL((lacn2_jump5<T, I, S>), dim3(1), dim3(GECON_BLOCKSIZE), 0, stream,
                                n, *x, d_est, d_anorm);
        std::swap(*x, *v);
        *h_kase = 0;

        return rocblas_status_success;
        break;
    default: return rocblas_status_invalid_value; break;
    }
}

// main gecon function
template <bool BATCHED, bool STRIDED, typename T, typename I, typename S, typename U>
rocblas_status rocsolver_gecon_template(rocblas_handle handle,
                                        const rocsolver_norm_type norm_type,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I inca,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        const I* ipiv,
                                        const rocblas_stride strideP,
                                        const S* anorm, // S = real_t<T>
                                        S* rcond,
                                        const I batch_count,
                                        T* work_v,
                                        T* work_x,
                                        I* work_isgn,
                                        S* scalar_est,
                                        I* scalar_max_idx,
                                        rocblas_int* scalar_kase,
                                        rocblas_int* scalar_jump,
                                        void* work_getrs_1,
                                        void* work_getrs_2,
                                        void* work_getrs_3,
                                        void* work_getrs_4,
                                        const I max_iter)
{
    ROCSOLVER_ENTER("gecon", "norm_type:", norm_type, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    if(!batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return if no dimensions
    if(n == 0)
    {
        if(rcond)
        {
            rocblas_int blocks = (batch_count - 1) / BS1 + 1;
            dim3 grid(blocks, 1, 1);
            dim3 threads(BS1, 1, 1);
            ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, rcond, batch_count, S(1));
        }
        return rocblas_status_success;
    }

    // iterate over each batch
    std::vector<S> h_anorm(batch_count);
    HIP_CHECK(hipMemcpyAsync(h_anorm.data(), anorm, sizeof(S) * batch_count, hipMemcpyDeviceToHost,
                             stream));
    for(I batch = 0; batch < batch_count; batch++)
    {
        // if anorm is zero for this batch, rcond is zero

        if(h_anorm[batch] == S(0))
        {
            S zero = S(0);
            HIP_CHECK(hipMemcpyAsync(rcond + batch, &zero, sizeof(S), hipMemcpyHostToDevice, stream));
            continue;
        }

        // get workspace pointers for this batch
        T* v = work_v + batch * n;
        T* x = work_x + batch * n;
        I* isgn = work_isgn + batch * n;
        S* d_est = scalar_est; // reuse single scalar for all batches
        const S* d_anorm = anorm + batch;
        I* d_max_idx = scalar_max_idx; // reuse single scalar
        rocblas_int* d_kase = scalar_kase; // reuse single scalar
        rocblas_int* d_jump = scalar_jump; // reuse single scalar

        // initialize lacn2 state
        rocblas_int h_kase = 0;
        rocblas_int h_jump = 0;
        I h_iters = 1;

        // main LACN2 and GETRS iteration loop
        do
        {
            gecon_lacn2<T, I, S>(handle, norm_type, n, A, shiftA + batch * strideA, inca, lda,
                                 strideA, ipiv + batch * strideP, strideP, stream, &v, &x, isgn,
                                 max_iter, &h_iters, d_max_idx, d_est, d_anorm, d_jump, d_kase,
                                 &h_jump, &h_kase);

            if(h_kase == 0)
                break; // converged, d_est contains rcond estimate

            // determine operation based on norm_type and kase
            rocblas_operation opr;
            if(norm_type == rocsolver_norm_type_one)
            {
                opr = (h_kase == 1) ? rocblas_operation_none
                                    : (rocblas_is_complex<T> ? rocblas_operation_conjugate_transpose
                                                             : rocblas_operation_transpose);
            }
            else
            { // infinity norm
                opr = (h_kase == 1) ? (rocblas_is_complex<T> ? rocblas_operation_conjugate_transpose
                                                             : rocblas_operation_transpose)
                                    : rocblas_operation_none;
            }

            rocsolver_getrs_template<false, false, T>(
                handle, opr, n, (I)1, A, shiftA + batch * strideA, inca, lda, strideA,
                ipiv + batch * strideP, strideP, (U)x, 0, (I)1, (I)n, 0, (I)1, work_getrs_1,
                work_getrs_2, work_getrs_3, work_getrs_4, true, true);
        } while(h_kase != 0);

        // copy result from scalar buffer to rcond for this batch
        HIP_CHECK(hipMemcpyAsync(rcond + batch, d_est, sizeof(S), hipMemcpyDeviceToDevice, stream));
    }

    return rocblas_status_success;
}

template <typename T, typename I, typename S>
void rocsolver_gecon_getMemorySize(const I n,
                                   const I lda,
                                   const I batch_count,
                                   size_t* size_work_v,
                                   size_t* size_work_x,
                                   size_t* size_work_isgn,
                                   size_t* size_scalar_est,
                                   size_t* size_scalar_max_idx,
                                   size_t* size_scalar_kase,
                                   size_t* size_scalar_jump,
                                   size_t* size_work_getrs_1,
                                   size_t* size_work_getrs_2,
                                   size_t* size_work_getrs_3,
                                   size_t* size_work_getrs_4)
{
    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
    {
        *size_work_v = 0;
        *size_work_x = 0;
        *size_work_isgn = 0;
        *size_scalar_est = 0;
        *size_scalar_max_idx = 0;
        *size_scalar_kase = 0;
        *size_scalar_jump = 0;
        *size_work_getrs_1 = 0;
        *size_work_getrs_2 = 0;
        *size_work_getrs_3 = 0;
        *size_work_getrs_4 = 0;
        return;
    }

    // need n elements of type T per batch for v vector
    *size_work_v = sizeof(T) * n * batch_count;

    // need n elements of type T per batch for x vector
    *size_work_x = sizeof(T) * n * batch_count;

    // need n elements of type I per batch for isgn vector (only used by real types, not complex)
    *size_work_isgn = rocblas_is_complex<T> ? 0 : sizeof(I) * n * batch_count;

    // Scalars are reused across batches (not allocated per batch)
    // need one S (real) scalar for estimate
    *size_scalar_est = sizeof(S);

    // need one I scalar for max index
    *size_scalar_max_idx = sizeof(I);

    // need one rocblas_int scalar for kase state
    *size_scalar_kase = sizeof(rocblas_int);

    // need one rocblas_int scalar for jump state
    *size_scalar_jump = sizeof(rocblas_int);

    // workspace for getrs
    // optim_mem is an output parameter indicating if optimized memory layout is used
    // by getrs internally; we don't need to act on it here, just provide it to the query
    bool optim_mem;
    rocsolver_getrs_getMemorySize<false, false, T, I>(
        rocblas_operation_none, n, 1, batch_count, size_work_getrs_1, size_work_getrs_2,
        size_work_getrs_3, size_work_getrs_4, &optim_mem, lda, n);
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_gecon_argCheck(rocblas_handle handle,
                                        const rocsolver_norm_type norm_type,
                                        const I n,
                                        const I lda,
                                        T A,
                                        const I* ipiv,
                                        const S* anorm,
                                        S* rcond)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(norm_type != rocsolver_norm_type_one && norm_type != rocsolver_norm_type_infinity
       && norm_type != rocsolver_norm_type_frobenius && norm_type != rocsolver_norm_type_max)
        return rocblas_status_invalid_value;

    // Frobenius and max norms are not supported
    if(norm_type == rocsolver_norm_type_frobenius || norm_type == rocsolver_norm_type_max)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !ipiv) || (n && !anorm) || (n && !rcond))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

ROCSOLVER_END_NAMESPACE
