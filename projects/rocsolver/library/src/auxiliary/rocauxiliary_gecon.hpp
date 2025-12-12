/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "rocblas_utility.hpp"
#include "rocsolver_run_specialized_kernels.hpp"
#include "ideal_sizes.hpp"
#include "lapack/roclapack_getrs.hpp"
#include <algorithm>
#include <vector>
#include <iostream>

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I>
ROCSOLVER_KERNEL void gecon_init_vector(T* x, const rocblas_int n, const T value)
{
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    for (I i = tid; i < n; i += hipBlockDim_x * hipGridDim_x)
    {
        if(i < n)
            x[i] = value;
    }
}

// Compute L1 norm of a vector
// template <int MAX_THDS, typename T, typename S>
// ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS)
//     gecon_compute_l1_norm(const T* v, const rocblas_int n, S* norm)
// {
//     rocblas_int tid = hipThreadIdx_x;

//     __shared__ S sval[MAX_THDS / WarpSize];

//     // Sum absolute values
//     S sum = 0;
//     for(rocblas_int i = tid; i < n; i += MAX_THDS)
//         sum += rocblas_abs(v[i]);

//     // Reduce within Warp
//     sum += shift_left(sum, 1);
//     sum += shift_left(sum, 2);
//     sum += shift_left(sum, 4);
//     sum += shift_left(sum, 8);
//     sum += shift_left(sum, 16);
//     if(WarpSize > 32)
//         sum += shift_left(sum, 32);

//     if(tid % WarpSize == 0)
//         sval[tid / WarpSize] = sum;
//     __syncthreads();

//     if(tid == 0)
//     {
//         for(rocblas_int k = 1; k < MAX_THDS / WarpSize; k++)
//             sum += sval[k];
//         *norm = sum;
//     }
// }

// reduce within Warp using shuffle on both index and value
template <typename T, typename I>
__device__ inline void lacn2_max_index(const I n, T* local_max, I* local_max_index, rocblas_int offset)
{
    T compare_local_max = shift_left(*local_max, offset);
    T compare_local_index = shift_left(*local_max_index, offset);
    if (compare_local_max > *local_max){
        *local_max = compare_local_max;
        *local_max_index = compare_local_index;
    } 
}

// // find index of maximum abosolute value in vector x
// // to be called with only a single block
// template <int MAX_THDS, typename T, typename I>
// ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS) lacn2_max_index_kernel(const I n,
//                                                                    const T* x,
//                                                                    I* max_index)
// {
//     I tid = threadIdx.x;

//     // shared variables
//     __shared__ S sval[MAX_THDS / WarpSize];
//     __shared__ S sval_indices[MAX_THDS / WarpSize];

//     // dot
//     T local_max = std::numeric_limits<T>::min()
//     I local_max_index;
//     for(I i = tid; i < m * n; i += MAX_THDS)
//     {
//         if (rocblas_abs(x[i]) > local_max)
//         {
//             local_max = rocblas_abs(x[i]);
//             local_max_index = i;
//         }
//     }

//     // reduce within Warp using shuffle on both index and value
//     lacn2_max_index<T, I>(n, &local_max, &local_max_index, 1);
//     lacn2_max_index<T, I>(n, &local_max, &local_max_index, 2);
//     lacn2_max_index<T, I>(n, &local_max, &local_max_index, 4);
//     lacn2_max_index<T, I>(n, &local_max, &local_max_index, 8);
//     lacn2_max_index<T, I>(n, &local_max, &local_max_index, 16);
//     if (WarpSize > 32)
//         lacn2_max_index<T, I>(n, &local_max, &local_max_index, 32);

//     if (tid % WarpSize == 0)
//         sval[tid / WarpSize] = local_max;
//         sval_indices[tid / WarpSize] = local_max_index;
//     __syncthreads();


//     if(tid == 0)
//     {
//         for(I k = 1; k < MAX_THDS / WarpSize; k++)
//         {
//             if (rocblas_abs(sval[k]) > local_max)
//             {
//                 local_max = rocblas_abs(sval[k]);
//                 local_max_index = sval_indices[k];
//             }
//         }
//         *max_index = local_max_index;
//     }
// }


// // sets vector to zero except for index of maximum abs element
// template <typename T, typename I>
// ROCSOLVER_KERNEL void lacn2_zero_and_set_max_index_kernel(const I n,
//                                                                    const T* x,
//                                                                    I* max_index)
// {
//     I tid = threadIdx.x + hipBlockIdx_x * hipBlockDim_x;
//     I max_idx = *max_index;

//     for (I i = tid; i < n; i += hipBlockDim_x * hipGridDim_x)
//     {
//         x[i] = T(0);
//     }

//     if (tid == 0)
//         x[max_index] = T(1);

// }

template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(GECON_BLOCKSIZE)
    lacn2_jump1_n_equals_one(const I n, T* x, S* norm, I* isgn, const S* d_anorm)
{
    rocblas_int tid = hipThreadIdx_x;

    if (tid == 0){
        S est = rocblas_abs(x[0]);
        S anorm = *d_anorm;
        if (est != S(0) && anorm != S(0))
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
    for(I i = tid; i < n; i += GECON_BLOCKSIZE){
        sum += rocblas_abs(x[i]);

        if constexpr(rocblas_is_complex<T>){
            S absxi = rocblas_abs(x[i]);
            if (absxi == 0){
                x[i] = T(1);
            } else{
                x[i] = x[i] / absxi;
            }
        } else{
            if(x[i] >= 0){
                x[i] = T(1);
                isgn[i] = T(1);
            } else{
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
ROCSOLVER_KERNEL void __launch_bounds__(GECON_BLOCKSIZE) lacn2_jump2(const I n,
                                                                   T* x,
                                                                   I* max_idx)
{
    I tid = threadIdx.x;

    // shared variables
    __shared__ S sval[GECON_BLOCKSIZE / WarpSize];
    __shared__ S sval_indices[GECON_BLOCKSIZE / WarpSize];

    // dot
    S local_max = std::numeric_limits<S>::min();
    I local_max_index;
    for(I i = tid; i < n; i += GECON_BLOCKSIZE)
    {
        if (rocblas_abs(x[i]) > local_max)
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
    if (WarpSize > 32)
        lacn2_max_index<S, I>(n, &local_max, &local_max_index, 32);

    if (tid % WarpSize == 0){
        sval[tid / WarpSize] = local_max;
        sval_indices[tid / WarpSize] = local_max_index;
    }
    __syncthreads();


    if(tid == 0)
    {
        for(I k = 1; k < GECON_BLOCKSIZE / WarpSize; k++)
        {
            if (rocblas_abs(sval[k]) > local_max)
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
ROCSOLVER_KERNEL void __launch_bounds__(GECON_BLOCKSIZE) lacn2_jump3(const I n,
                                                                   T* x,
                                                                   T* v,
                                                                   I* isgn,
                                                                   rocblas_int* kase,
                                                                   rocblas_int* jump,
                                                                   S* est)
{

    rocblas_int tid = hipThreadIdx_x;

    __shared__ S sval[GECON_BLOCKSIZE / WarpSize];
    __shared__ bool sval_repeated[GECON_BLOCKSIZE / WarpSize];
    __shared__ S sval_estold;  // Shared storage for estold so all threads can access it

    // Sum absolute values
    S sum = 0;
    bool repeated = (rocblas_is_complex<T>) ? false : true;
    // bool repeated = true;
    // we iterate over v, since v contains x from previous step (pointers swapped)
    for(I i = tid; i < n; i += GECON_BLOCKSIZE)
    {
        sum += rocblas_abs(v[i]);
        if constexpr(rocblas_is_complex<T>)
            continue;
        else{
            if(v[i] >= 0){
                if (isgn[i] <= -1)
                    repeated = false;
            }
            else{
                if(isgn[i] >= 1)
                    repeated = false;
            }
        }
    }

    // Reduce within Warp
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
    if(WarpSize > 32){
        sum += shift_left(sum, 32);
        repeated = repeated && __shfl_down(repeated, 32);
    }

    if(tid % WarpSize == 0){
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
        sval_estold = *est;  // Save old estimate to shared memory before updating
        *est = sum;
        sval_repeated[0] = repeated;
    }

    __syncthreads();

    repeated = sval_repeated[0];
    S estold = sval_estold;  // All threads read the same synchronized value

    if (repeated || (*est <= estold)){
        for(I i = tid; i < n; i += GECON_BLOCKSIZE)
        {
            // we write over old x
            T sign = (i % 2 == 0) ? T(1) : T(-1);
            x[i] = T(sign * (S(1) + S(i) / S(n-1)));
            // x[i] = sign * (T(1) + T(i) / T(n-1));
        }
        if(tid == 0){
            *kase = 1;
            *jump = 5; 
        }

        return;
    }

    if constexpr(rocblas_is_complex<T>)
    {
        // we iterate over v, since v contains old x (pointers swapped)
        for(I i = tid; i < n; i += GECON_BLOCKSIZE)
        {
            S absxi = rocblas_abs(v[i]);
            if (absxi == 0){
                x[i] = T(1);
                // isgn[i] = T(1);
            } else{
                x[i] = v[i] / absxi;
                // isgn[i] = T(rocblas_real(x[i]) >= 0 ? 1 : -1);
            }
        }
    } else{
        // we iterate over v, since v contains old x (pointers swapped)
        for(I i = tid; i < n; i += GECON_BLOCKSIZE)
        {
            if (v[i] >= 0)
            {
                x[i] = T(1);
                isgn[i] = T(1);
            }
            else{
                x[i] = T(-1);
                isgn[i] = T(-1);

            }
        }
    }


    if (tid == 0){
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

    // dot
    S local_max = std::numeric_limits<S>::min();
    I local_max_index;
    for(I i = tid; i < n; i += GECON_BLOCKSIZE)
    {
        if (rocblas_abs(x[i]) > local_max)
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
    if (WarpSize > 32)
        lacn2_max_index<S, I>(n, &local_max, &local_max_index, 32);

    if (tid % WarpSize == 0){
        sval[tid / WarpSize] = local_max;
        sval_indices[tid / WarpSize] = local_max_index;
    }
    __syncthreads();


    if(tid == 0)
    {
        for(I k = 1; k < GECON_BLOCKSIZE / WarpSize; k++)
        {
            if (rocblas_abs(sval[k]) > local_max)
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

    if (val_new == val_old || iters == iters_max){
        for(rocblas_int i = tid; i < n; i += GECON_BLOCKSIZE)
        {
            T sign = (i % 2 == 0) ? T(1) : T(-1);
            x[i] = T(sign * (S(1) + S(i) / S(n-1)));
            // x[i] = sign * (T(1) + T(i) / T(n-1));
        }
        if(tid == 0){
            // TODO: increment iters???
            *kase = 1;
            *jump = 5; 
        }

        return;
    }

    for (I i = tid; i < n; i += GECON_BLOCKSIZE)
    {
        x[i] = T(0);
    }

    // Synchronize to ensure all threads finish zeroing before thread 0 sets the max element
    __syncthreads();

    if (tid == 0)
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
        if (sum > *norm)
            *norm = sum;

        S est = *norm;
        S anorm = *d_anorm;
        if (est != S(0) && anorm != S(0))
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
                //  I* isave,       // just preserve elements as individual variables (on host?)
                 const I max_iters,
                 I* h_iters,
                 I* d_max_idx,
                //  const I d_iters, // on host, just maintain iters as they will be incremented within the kernels // if kernel behaviour could change, then maintain device iters as well
                 S* d_est,          // real_t<T>
                 const S* d_anorm,          // real_t<T>
                //  S* d_estold,
                //  I* d_jlast,
                //  I* d_should_break,
                //  I* d_is_equal,
                 rocblas_int* d_jump,
                 rocblas_int* d_kase,
                 rocblas_int* h_jump,
                 rocblas_int* h_kase)
{
    // TODO: consider what return value should be, should we bother with returning success in each case, or just after switch statement?
    // TODO: consider if synchronizations required between kernel laynches or memory transfers :)
    if (*h_kase == 0){
        // initialize x = (1/n, ..., 1/n)
        rocblas_int blocks = (n - 1) / GECON_BLOCKSIZE + 1;
        ROCSOLVER_LAUNCH_KERNEL((gecon_init_vector<T, I>), dim3(blocks), dim3(GECON_BLOCKSIZE), 0,
                                stream, *x, n, T(1) / T(n));
        ROCSOLVER_LAUNCH_KERNEL((gecon_init_vector<T, I>), dim3(blocks), dim3(GECON_BLOCKSIZE), 0,
                                stream, *v, n, T(1) / T(n));
        *h_kase = 1;
        *h_jump = 1;
        return rocblas_status_success;
    }
    
    // Declare debug variables once for reuse in all cases
    std::vector<T> h_x(n), h_v(n);
    std::vector<I> h_isgn_vec(n);
    S h_est;
    I h_max_idx_val;
    // int print_n = std::min(10, (int)n);
    int print_n = (int) n;
    
    switch (*h_jump){
        case 1:
            std::cout << "========================================" << std::endl;
            std::cout << "=== ROCSOLVER Jump " << *h_jump << " ===" << std::endl;
            if (n == 1)
            {
                // hipMemcpyAsync(est, x, sizeof(S), hipMemcpyDeviceToHost, stream);
                // hipStreamSynchronize(stream);
                // *est = rocblas_abs(*est);
                ROCSOLVER_LAUNCH_KERNEL((lacn2_jump1_n_equals_one<T, I, S>), dim3(1), dim3(GECON_BLOCKSIZE), 0,
                                    stream, n, *x, d_est, isgn, d_anorm);
                *h_kase = 0; // signal to exit
                return rocblas_status_success;
            }

            
            ROCSOLVER_LAUNCH_KERNEL((lacn2_jump1<T, I, S>), dim3(1), dim3(GECON_BLOCKSIZE), 0,
                                    stream, n, *x, d_est, isgn);

            // ROCSOLVER_LAUNCH_KERNEL((gecon_compute_l1_norm<GECON_BLOCKSIZE, S>), dim3(1), threads, 0,
            //                         stream, x, n, d_est);

            // ROCSOLVER_LAUNCH_KERNEL(lacn2_init_alternating_vector<S>, blocks, threads, 0, stream, x, isgn, n);

            *h_kase = 2;
            *h_jump = 2;

            // DEBUG: Detailed instrumentation
            hipStreamSynchronize(stream);
            hipMemcpy(h_x.data(), *x, sizeof(T) * n, hipMemcpyDeviceToHost);
            hipMemcpy(h_v.data(), *v, sizeof(T) * n, hipMemcpyDeviceToHost);
            if constexpr(!rocblas_is_complex<T>)
                hipMemcpy(h_isgn_vec.data(), isgn, sizeof(I) * n, hipMemcpyDeviceToHost);
            hipMemcpy(&h_est, d_est, sizeof(S), hipMemcpyDeviceToHost);
            hipMemcpy(&h_max_idx_val, d_max_idx, sizeof(I), hipMemcpyDeviceToHost);

            std::cout << "Iteration: " << *h_iters << std::endl;
            std::cout << "kase: " << *h_kase << std::endl;
            std::cout << "jump: " << *h_jump << std::endl;
            std::cout << "est: " << h_est << std::endl;
            std::cout << "max_idx: " << h_max_idx_val << std::endl;
            
            std::cout << "x[0:" << print_n << "]: ";
            for(int i = 0; i < print_n; i++)
                std::cout << h_x[i] << " ";
            std::cout << std::endl;
            
            std::cout << "v[0:" << print_n << "]: ";
            for(int i = 0; i < print_n; i++)
                std::cout << h_v[i] << " ";
            std::cout << std::endl;
            
            if constexpr(!rocblas_is_complex<T>) {
                std::cout << "isgn[0:" << print_n << "]: ";
                for(int i = 0; i < print_n; i++)
                    std::cout << h_isgn_vec[i] << " ";
                std::cout << std::endl;
            }
            std::cout << "========================================" << std::endl;

            return rocblas_status_success;
            break;
        case 2:
            std::cout << "========================================" << std::endl;
            std::cout << "=== ROCSOLVER Jump " << *h_jump << " ===" << std::endl;
            // ROCSOLVER_LAUNCH_KERNEL(lacn2_max_index_kernel<T, I>, blocks, threads, 0, stream, n, x, d_max_idx);
            // ROCSOLVER_LAUNCH_KERNEL(lacn2_zero_and_set_max_index_kernel<T, I>, blocks, threads, 0, stream, n, x, d_max_idx);
            ROCSOLVER_LAUNCH_KERNEL((lacn2_jump2<T, I, S>), dim3(1), dim3(GECON_BLOCKSIZE), 0, stream, n, *x, d_max_idx);

            // isave(2) = 3?????
            // TODO: how to reflect this

            *h_kase = 1;
            *h_jump = 3;

            *h_iters = 2;

            // DEBUG: Detailed instrumentation
            hipStreamSynchronize(stream);
            hipMemcpy(h_x.data(), *x, sizeof(T) * n, hipMemcpyDeviceToHost);
            hipMemcpy(h_v.data(), *v, sizeof(T) * n, hipMemcpyDeviceToHost);
            if constexpr(!rocblas_is_complex<T>)
                hipMemcpy(h_isgn_vec.data(), isgn, sizeof(I) * n, hipMemcpyDeviceToHost);
            hipMemcpy(&h_est, d_est, sizeof(S), hipMemcpyDeviceToHost);
            hipMemcpy(&h_max_idx_val, d_max_idx, sizeof(I), hipMemcpyDeviceToHost);
            
            std::cout << "Iteration: " << *h_iters << std::endl;
            std::cout << "kase: " << *h_kase << std::endl;
            std::cout << "jump: " << *h_jump << std::endl;
            std::cout << "est: " << h_est << std::endl;
            std::cout << "max_idx: " << h_max_idx_val << std::endl;
            
            std::cout << "x[0:" << print_n << "]: ";
            for(int i = 0; i < print_n; i++)
                std::cout << h_x[i] << " ";
            std::cout << std::endl;
            
            std::cout << "v[0:" << print_n << "]: ";
            for(int i = 0; i < print_n; i++)
                std::cout << h_v[i] << " ";
            std::cout << std::endl;
            
            if constexpr(!rocblas_is_complex<T>) {
                std::cout << "isgn[0:" << print_n << "]: ";
                for(int i = 0; i < print_n; i++)
                    std::cout << h_isgn_vec[i] << " ";
                std::cout << std::endl;
            }
            std::cout << "========================================" << std::endl;

            return rocblas_status_success;
            break;
        case 3:
            std::cout << "========================================" << std::endl;
            std::cout << "=== ROCSOLVER Jump " << *h_jump << " ===" << std::endl;
            hipMemcpy(*v, *x, sizeof(T) * n, hipMemcpyDeviceToDevice);
            std::swap(*x, *v);
            hipStreamSynchronize(stream);
            ROCSOLVER_LAUNCH_KERNEL((lacn2_jump3<T, I, S>), dim3(1), dim3(GECON_BLOCKSIZE), 0,
                                    stream, n, *x, *v, isgn, d_kase, d_jump, d_est);

            hipMemcpyAsync(h_jump, d_jump, sizeof(rocblas_int), hipMemcpyDeviceToHost, stream);
            hipMemcpyAsync(h_kase, d_kase, sizeof(rocblas_int), hipMemcpyDeviceToHost, stream);
            hipStreamSynchronize(stream);
            // ROCSOLVER_LAUNCH_KERNEL((gecon_compute_l1_norm<GECON_BLOCKSIZE, S>), dim3(1), threads, 0,
            //                         stream, x, n, d_est);
            // TODO: transfer updated jump and case from device to host

            // DEBUG: Detailed instrumentation
            hipMemcpy(h_x.data(), *x, sizeof(T) * n, hipMemcpyDeviceToHost);
            hipMemcpy(h_v.data(), *v, sizeof(T) * n, hipMemcpyDeviceToHost);
            if constexpr(!rocblas_is_complex<T>)
                hipMemcpy(h_isgn_vec.data(), isgn, sizeof(I) * n, hipMemcpyDeviceToHost);
            hipMemcpy(&h_est, d_est, sizeof(S), hipMemcpyDeviceToHost);
            hipMemcpy(&h_max_idx_val, d_max_idx, sizeof(I), hipMemcpyDeviceToHost);
            
            std::cout << "Iteration: " << *h_iters << std::endl;
            std::cout << "kase: " << *h_kase << std::endl;
            std::cout << "jump: " << *h_jump << std::endl;
            std::cout << "est: " << h_est << std::endl;
            std::cout << "max_idx: " << h_max_idx_val << std::endl;
            
            std::cout << "x[0:" << print_n << "]: ";
            for(int i = 0; i < print_n; i++)
                std::cout << h_x[i] << " ";
            std::cout << std::endl;
            
            std::cout << "v[0:" << print_n << "]: ";
            for(int i = 0; i < print_n; i++)
                std::cout << h_v[i] << " ";
            std::cout << std::endl;
            
            if constexpr(!rocblas_is_complex<T>) {
                std::cout << "isgn[0:" << print_n << "]: ";
                for(int i = 0; i < print_n; i++)
                    std::cout << h_isgn_vec[i] << " ";
                std::cout << std::endl;
            }
            std::cout << "========================================" << std::endl;
            return rocblas_status_success;
            break;
        case 4:
            std::cout << "========================================" << std::endl;
            std::cout << "=== ROCSOLVER Jump " << *h_jump << " ===" << std::endl;
            // int jlast; 
            // hipMemcpyAsync(jlast, d_max_idx, sizeof(I), hipMemcpyDeviceToHost, stream);
            // hipStreamSynchronize(stream);
            ROCSOLVER_LAUNCH_KERNEL((lacn2_jump4<T, I, S>), dim3(1), dim3(GECON_BLOCKSIZE), 0,
                                    stream, n, *x, isgn, d_kase, d_jump, d_max_idx, *h_iters, max_iters);

            hipMemcpyAsync(h_jump, d_jump, sizeof(rocblas_int), hipMemcpyDeviceToHost, stream);
            hipMemcpyAsync(h_kase, d_kase, sizeof(rocblas_int), hipMemcpyDeviceToHost, stream);
            hipStreamSynchronize(stream);

            if (*h_jump == 5)
                *h_iters = *h_iters+1;

            // DEBUG: Detailed instrumentation
            hipMemcpy(h_x.data(), *x, sizeof(T) * n, hipMemcpyDeviceToHost);
            hipMemcpy(h_v.data(), *v, sizeof(T) * n, hipMemcpyDeviceToHost);
            if constexpr(!rocblas_is_complex<T>)
                hipMemcpy(h_isgn_vec.data(), isgn, sizeof(I) * n, hipMemcpyDeviceToHost);
            hipMemcpy(&h_est, d_est, sizeof(S), hipMemcpyDeviceToHost);
            hipMemcpy(&h_max_idx_val, d_max_idx, sizeof(I), hipMemcpyDeviceToHost);
            
            std::cout << "Iteration: " << *h_iters << std::endl;
            std::cout << "kase: " << *h_kase << std::endl;
            std::cout << "jump: " << *h_jump << std::endl;
            std::cout << "est: " << h_est << std::endl;
            std::cout << "max_idx: " << h_max_idx_val << std::endl;
            
            std::cout << "x[0:" << print_n << "]: ";
            for(int i = 0; i < print_n; i++)
                std::cout << h_x[i] << " ";
            std::cout << std::endl;
            
            std::cout << "v[0:" << print_n << "]: ";
            for(int i = 0; i < print_n; i++)
                std::cout << h_v[i] << " ";
            std::cout << std::endl;
            
            if constexpr(!rocblas_is_complex<T>) {
                std::cout << "isgn[0:" << print_n << "]: ";
                for(int i = 0; i < print_n; i++)
                    std::cout << h_isgn_vec[i] << " ";
                std::cout << std::endl;
            }
            std::cout << "========================================" << std::endl;

            return rocblas_status_success;
            break;
        case 5:
            std::cout << "========================================" << std::endl;
            std::cout << "=== ROCSOLVER Jump " << *h_jump << " ===" << std::endl;
            // afterwards, d_est will contain condition number estimate
            ROCSOLVER_LAUNCH_KERNEL((lacn2_jump5<T, I, S>), dim3(1), dim3(GECON_BLOCKSIZE), 0,
                                    stream, n, *x, d_est, d_anorm);
            // copy x onto v
            std::swap(*x, *v);
            *h_kase = 0;

            // DEBUG: Detailed instrumentation
            hipStreamSynchronize(stream);
            hipMemcpy(h_x.data(), *x, sizeof(T) * n, hipMemcpyDeviceToHost);
            hipMemcpy(h_v.data(), *v, sizeof(T) * n, hipMemcpyDeviceToHost);
            if constexpr(!rocblas_is_complex<T>)
                hipMemcpy(h_isgn_vec.data(), isgn, sizeof(I) * n, hipMemcpyDeviceToHost);
            hipMemcpy(&h_est, d_est, sizeof(S), hipMemcpyDeviceToHost);
            hipMemcpy(&h_max_idx_val, d_max_idx, sizeof(I), hipMemcpyDeviceToHost);
            
            std::cout << "Iteration: " << *h_iters << std::endl;
            std::cout << "kase: " << *h_kase << std::endl;
            std::cout << "jump: " << *h_jump << std::endl;
            std::cout << "est: " << h_est << std::endl;
            std::cout << "max_idx: " << h_max_idx_val << std::endl;
            
            std::cout << "x[0:" << print_n << "]: ";
            for(int i = 0; i < print_n; i++)
                std::cout << h_x[i] << " ";
            std::cout << std::endl;
            
            std::cout << "v[0:" << print_n << "]: ";
            for(int i = 0; i < print_n; i++)
                std::cout << h_v[i] << " ";
            std::cout << std::endl;
            
            if constexpr(!rocblas_is_complex<T>) {
                std::cout << "isgn[0:" << print_n << "]: ";
                for(int i = 0; i < print_n; i++)
                    std::cout << h_isgn_vec[i] << " ";
                std::cout << std::endl;
            }
            std::cout << "========================================" << std::endl;

            return rocblas_status_success;
            break;
        default:
            break;
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
                                        S* scalars_est,
                                        I* scalars_max_idx,
                                        rocblas_int* scalars_kase,
                                        rocblas_int* scalars_jump,
                                        void* work_getrs_1,
                                        void* work_getrs_2,
                                        void* work_getrs_3,
                                        void* work_getrs_4,
                                        const I max_iter)
{
    // TODO: does ROCSOLVER_ENTER need more function argss passed?
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

    // return not implemented for Frobenius and max norms
    if(norm_type == rocsolver_norm_type_frobenius || norm_type == rocsolver_norm_type_max)
        return rocblas_status_not_implemented;

    // iterate over each batch
    std::vector<S> h_anorm(batch_count);
    hipMemcpyAsync(h_anorm.data(), anorm, sizeof(S) * batch_count, hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
    for(I batch = 0; batch < batch_count; batch++)
    {
        // if anorm is zero for this batch, rcond is zero

        if(h_anorm[batch] == S(0))
        {
            S zero = S(0);
            hipMemcpyAsync(rcond + batch, &zero, sizeof(S), hipMemcpyHostToDevice, stream);
            continue;
        }

        // get workspace pointers for this batch
        T* v = work_v + batch * n;
        T* x = work_x + batch * n;
        I* isgn = work_isgn + batch * n;
        S* d_est = rcond + batch;
        const S* d_anorm = anorm + batch;
        I* d_max_idx = scalars_max_idx + batch;
        rocblas_int* d_kase = scalars_kase + batch;
        rocblas_int* d_jump = scalars_jump + batch;


        // initialize lacn2 state
        rocblas_int h_kase = 0;
        rocblas_int h_jump = 0;
        I h_iters = 1;

        std::cout << "\n========================================" << std::endl;
        std::cout << "=== ROCSOLVER GECON START (batch=" << batch << ") ===" << std::endl;
        std::cout << "n=" << n << ", anorm=" << h_anorm[batch] << std::endl;
        std::cout << "========================================\n" << std::endl;

        // main LACN2 and trsv iteration loop
        do
        {
            std::cout << "\n>>> ROCSOLVER: Before LACN2" << std::endl;
            std::cout << "    iter=" << h_iters << ", kase=" << h_kase << ", jump=" << h_jump << std::endl;

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

            std::cout << "\n>>> ROCSOLVER: After LACN2, before triangular solves" << std::endl;
            std::cout << "    opr=" << (opr == rocblas_operation_none ? "none" : 
                       (opr == rocblas_operation_transpose ? "transpose" : "conj_transpose"))
                      << std::endl;

            // copy x to host to print before trsv
            std::vector<T> h_x_before_trsv(n);
            hipMemcpy(h_x_before_trsv.data(), x, sizeof(T) * n, hipMemcpyDeviceToHost);
            int print_n = std::min(10, (int)n);
            std::cout << "    x before GETRS[0:" << print_n << "]: ";
            for(int i = 0; i < print_n; i++)
                std::cout << h_x_before_trsv[i] << " ";
            std::cout << std::endl;

            T* A_ptr = load_ptr_batch<T>(A, batch, shiftA, strideA);
            
            // TRSV requires w_completed_sec pointer
            rocblas_int* w_completed;
            hipMalloc((void**)&w_completed, sizeof(rocblas_int));
            
            std::vector<T> h_x_temp(n);
            
            // if(opr == rocblas_operation_none)
            // {
                // std::cout << "    >>>>> ROCSOLVER: Using TRSV, solving ORIGINAL problem" << std::endl;
                
                // Forward solve: L*y = b, then U*x = y
                // Solve L*x = b (lower triangular, unit diagonal)
                // std::cout << "    >>>>> Solve 1: LOWER triangular, UNIT diagonal" << std::endl;
                // hipMemcpy(h_x_temp.data(), x, sizeof(T) * n, hipMemcpyDeviceToHost);
                // std::cout << "        X before L solve[0:" << print_n << "]: ";
                // for(int i = 0; i < print_n; i++)
                //     std::cout << h_x_temp[i] << " ";
                // std::cout << std::endl;
                
                // rocblasCall_trsv<T>(handle, rocblas_fill_lower, 
                //            rocblas_operation_none, rocblas_diagonal_unit,
                //            (rocblas_int)n, A_ptr, (rocblas_stride)0, (rocblas_int)lda, (rocblas_stride)0,
                //            x, (rocblas_stride)0, (rocblas_int)1, (rocblas_stride)0, (rocblas_int)1, 
                //            w_completed);
                
                // hipStreamSynchronize(stream);
                // hipMemcpy(h_x_temp.data(), x, sizeof(T) * n, hipMemcpyDeviceToHost);
                // std::cout << "        X after L solve[0:" << print_n << "]: ";
                // for(int i = 0; i < print_n; i++)
                //     std::cout << h_x_temp[i] << " ";
                // std::cout << std::endl;
                //
                // Solve U*x = y (upper triangular, non-unit diagonal)
                // std::cout << "    >>>>> Solve 2: UPPER triangular, NON-UNIT diagonal" << std::endl;
                // hipMemcpy(h_x_temp.data(), x, sizeof(T) * n, hipMemcpyDeviceToHost);
                // std::cout << "        X before U solve[0:" << print_n << "]: ";
                // for(int i = 0; i < print_n; i++)
                //     std::cout << h_x_temp[i] << " ";
                // std::cout << std::endl;
                //
                // rocblasCall_trsv<T>(handle, rocblas_fill_upper,
                //            rocblas_operation_none, rocblas_diagonal_non_unit,
                //            (rocblas_int)n, A_ptr, (rocblas_stride)0, (rocblas_int)lda, (rocblas_stride)0,
                //            x, (rocblas_stride)0, (rocblas_int)1, (rocblas_stride)0, (rocblas_int)1,
                //            w_completed);
                
                // hipStreamSynchronize(stream);
                // hipMemcpy(h_x_temp.data(), x, sizeof(T) * n, hipMemcpyDeviceToHost);
                // std::cout << "        X after U solve[0:" << print_n << "]: ";
                // for(int i = 0; i < print_n; i++)
                //     std::cout << h_x_temp[i] << " ";
                // std::cout << std::endl;
            // }
            // else
            // {
                // std::cout << "    >>>>> ROCSOLVER: Using TRSV, solving TRANSPOSE problem" << std::endl;
                
                // Transpose solve: U^T*y = b, then L^T*x = y
                // rocblas_operation trans_op = opr;
                
                // Solve U^T*x = b (upper triangular transposed, non-unit diagonal)
                // std::cout << "    >>>>> Solve 1: UPPER triangular TRANSPOSED, NON-UNIT diagonal" << std::endl;
                // hipMemcpy(h_x_temp.data(), x, sizeof(T) * n, hipMemcpyDeviceToHost);
                // std::cout << "        X before U^T solve[0:" << print_n << "]: ";
                // for(int i = 0; i < print_n; i++)
                //     std::cout << h_x_temp[i] << " ";
                // std::cout << std::endl;
                
                // rocblasCall_trsv<T>(handle, rocblas_fill_upper,
                //            trans_op, rocblas_diagonal_non_unit,
                //            (rocblas_int)n, A_ptr, (rocblas_stride)0, (rocblas_int)lda, (rocblas_stride)0,
                //            x, (rocblas_stride)0, (rocblas_int)1, (rocblas_stride)0, (rocblas_int)1,
                //            w_completed);
                
                // hipStreamSynchronize(stream);
                // hipMemcpy(h_x_temp.data(), x, sizeof(T) * n, hipMemcpyDeviceToHost);
                // std::cout << "        X after U^T solve[0:" << print_n << "]: ";
                // for(int i = 0; i < print_n; i++)
                //     std::cout << h_x_temp[i] << " ";
                // std::cout << std::endl;
                
                // Solve L^T*x = y (lower triangular transposed, unit diagonal)
                // std::cout << "    >>>>> Solve 2: LOWER triangular TRANSPOSED, UNIT diagonal" << std::endl;
                // hipMemcpy(h_x_temp.data(), x, sizeof(T) * n, hipMemcpyDeviceToHost);
                // std::cout << "        X before L^T solve[0:" << print_n << "]: ";
                // for(int i = 0; i < print_n; i++)
                //     std::cout << h_x_temp[i] << " ";
                // std::cout << std::endl;
                
                // rocblasCall_trsv<T>(handle, rocblas_fill_lower,
                //            trans_op, rocblas_diagonal_unit,
                //            (rocblas_int)n, A_ptr, (rocblas_stride)0, (rocblas_int)lda, (rocblas_stride)0,
                //            x, (rocblas_stride)0, (rocblas_int)1, (rocblas_stride)0, (rocblas_int)1,
                //            w_completed);
                
                // hipStreamSynchronize(stream);
                // hipMemcpy(h_x_temp.data(), x, sizeof(T) * n, hipMemcpyDeviceToHost);
                // std::cout << "        X after L^T solve[0:" << print_n << "]: ";
                // for(int i = 0; i < print_n; i++)
                //     std::cout << h_x_temp[i] << " ";
                // std::cout << std::endl;
            // }
            
            // hipFree(w_completed);
                        // Solve system using GETRS (applies pivots from CPU GETRF)
            // Since both CPU and GPU now use the same factorization, this is correct
            rocsolver_getrs_template<false, false, T>(
                handle, opr, n, (I) 1, A, shiftA + batch * strideA, inca, lda, strideA,
                ipiv + batch * strideP, strideP, (U)x, 0, (I) 1, (I) n, 0, (I) 1, work_getrs_1,
                work_getrs_2, work_getrs_3, work_getrs_4, true, true);

            // copy x to host to print before trsv
            // std::vector<T> h_x_before_trsv(n);
            hipMemcpy(h_x_before_trsv.data(), x, sizeof(T) * n, hipMemcpyDeviceToHost);
            // int print_n = std::min(10, (int)n);
            std::cout << "    x after GETRS[0:" << print_n << "]: ";
            for(int i = 0; i < print_n; i++)
                std::cout << h_x_before_trsv[i] << " ";
            std::cout << std::endl;
        } while(h_kase != 0);

        // rcond is computed in jump5 and stored in d_est, which is already rcond[batch]
        // note: d_est and rcond[batch] point to the same location
        
        // S h_final_rcond;
        // hipMemcpy(&h_final_rcond, rcond + batch, sizeof(S), hipMemcpyDeviceToHost);
        // std::cout << "\n========================================" << std::endl;
        // std::cout << "=== ROCSOLVER GECON END (batch=" << batch << ") ===" << std::endl;
        // std::cout << "Final RCOND=" << h_final_rcond << std::endl;
        // std::cout << "Total iterations: " << h_iters << std::endl;
        // std::cout << "========================================\n" << std::endl;
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
                                   size_t* size_scalars_est,
                                   size_t* size_scalars_max_idx,
                                   size_t* size_scalars_kase,
                                   size_t* size_scalars_jump,
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
        *size_scalars_est = 0;
        *size_scalars_max_idx = 0;
        *size_scalars_kase = 0;
        *size_scalars_jump = 0;
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

    // TODO: only needed by real, but how can we do that?
    *size_work_isgn = sizeof(I) * n * batch_count;

    // need S (real) scalar per batch for estimate
    *size_scalars_est = sizeof(S) * batch_count;

    // need I scalar per batch for max index
    *size_scalars_max_idx = sizeof(I) * batch_count;

    // need rocblas_int scalar per batch for kase state
    *size_scalars_kase = sizeof(rocblas_int) * batch_count;

    // need rocblas_int scalar per batch for jump state
    *size_scalars_jump = sizeof(rocblas_int) * batch_count;

    // workspace for getrs
    bool optim_mem;
    rocsolver_getrs_getMemorySize<false, false, T, I>(rocblas_operation_none, n, 1, batch_count,
                                                      size_work_getrs_1, size_work_getrs_2,
                                                      size_work_getrs_3, size_work_getrs_4,
                                                      &optim_mem, lda, n);
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

// /** Main template function **/
// template <bool BATCHED, bool STRIDED, typename T, typename I, typename S, typename U>
// rocblas_status rocsolver_gecon_template(rocblas_handle handle,
//                                         const rocsolver_norm_type norm_type,
//                                         const I n,
//                                         U A,
//                                         const rocblas_stride shiftA,
//                                         const I inca,
//                                         const I lda,
//                                         const rocblas_stride strideA,
//                                         const I* ipiv,
//                                         const rocblas_stride strideP,
//                                         const S anorm,
//                                         S* rcond,
//                                         const I batch_count,
//                                         S* work_v,
//                                         S* work_x,
//                                         I* iwork,
//                                         S* work_scalars_s,
//                                         I* work_scalars_i,
//                                         const I max_iter)
// {
//     ROCSOLVER_ENTER("gecon", "norm_type:", norm_type, "n:", n, "shiftA:", shiftA, "lda:", lda,
//                     "bc:", batch_count);

//     // quick return
//     if(batch_count == 0)
//         return rocblas_status_success;

//     hipStream_t stream;
//     rocblas_get_stream(handle, &stream);

//     // quick return if no dimensions
//     if(n == 0)
//     {
//         rocblas_int blocks = (batch_count - 1) / BS1 + 1;
//         dim3 grid(blocks, 1, 1);
//         dim3 threads(BS1, 1, 1);
//         ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, rcond, batch_count, 0);
//         return rocblas_status_success;
//     }

//     // Return not implemented for Frobenius and max norms
//     if(norm_type == rocsolver_norm_type_frobenius || norm_type == rocsolver_norm_type_max)
//         return rocblas_status_not_implemented;

//     // If anorm is zero, rcond is zero
//     if(anorm == S(0))
//     {
//         rocblas_int blocks = (batch_count - 1) / BS1 + 1;
//         dim3 grid(blocks, 1, 1);
//         dim3 threads(BS1, 1, 1);
//         ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, rcond, batch_count, 0);
//         return rocblas_status_success;
//     }

//     // Main condition estimation loop
//     for(I batch = 0; batch < batch_count; batch++)
//     {
//         // Get workspace pointers for this batch
//         S* v = work_v + batch * n;
//         S* x = work_x + batch * n;
//         I* isave = iwork + batch * n;
        
//         // Get scalar workspace pointers for this batch
//         S* d_est_batch = work_scalars_s + batch * 2;
//         S* d_estold_batch = work_scalars_s + batch * 2 + 1;
        
//         I* d_max_idx_batch = work_scalars_i + batch * 4;
//         I* d_jlast_batch = work_scalars_i + batch * 4 + 1;
//         I* d_should_break_batch = work_scalars_i + batch * 4 + 2;
//         I* d_is_equal_batch = work_scalars_i + batch * 4 + 3;

//         // Call LACN2 algorithm (result stays on device in d_est_batch)
//         gecon_lacn2<T, I, S>(handle, norm_type, n, A, shiftA + batch * strideA, inca, lda, strideA,
//                              ipiv + batch * strideP, strideP, v, x, isave, max_iter, d_est_batch,
//                              d_estold_batch, d_max_idx_batch, d_jlast_batch, d_should_break_batch,
//                              d_is_equal_batch);

//         // Compute reciprocal condition number on device
//         // rcond[batch] = 1 / (d_est_batch * anorm)
//         // This stays on device - rcond is already a device pointer
//         S* rcond_batch = rcond + batch;
        
//         // Launch kernel to compute rcond from estimate
//         ROCSOLVER_LAUNCH_KERNEL(gecon_compute_rcond<S>, dim3(1), dim3(1), 0, stream, d_est_batch,
//                                 anorm, rcond_batch);
//     }

//     return rocblas_status_success;
// }

ROCSOLVER_END_NAMESPACE
