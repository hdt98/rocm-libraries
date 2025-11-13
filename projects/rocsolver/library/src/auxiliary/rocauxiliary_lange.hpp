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

#include "ideal_sizes.hpp"
#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "rocblas_utility.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LANGE_FROBENIUS_BDIM) lange_frobenius_kernel(const rocblas_int m,
                                                                         const rocblas_int n,
                                                                         const U A,
                                                                         const rocblas_int lda,
                                                                         const rocblas_int shiftA,
                                                                         const rocblas_int strideA,
                                                                         S* block_sums)
{
    I bidz = blockIdx.z;
    I bid = blockIdx.x;
    I tid = threadIdx.x;
    I gridSize = gridDim.x * blockDim.x;

    // select batch instance
    rocblas_int blocks = (m * n - 1) / LANGE_FROBENIUS_BDIM + 1;
    T* a = load_ptr_batch<T>(A, bidz, shiftA, strideA);
    S* block_sums_block = load_ptr_batch<S>(block_sums, bidz, 0, blocks);

    // shared variables
    __shared__ S sval[LANGE_FROBENIUS_BDIM / WarpSize];

    // sum absolute values in row bid
    S block_sum = 0;
    for(I i = tid + (bid * blockDim.x); i < n * m; i += gridSize)
    {
        int row = i % m;
        int col = i / m;
        block_sum += std::pow(rocblas_abs(a[row + col * lda]), 2);
    }

    // reduce to get row sum
    block_sum += shift_left(block_sum, 1);
    block_sum += shift_left(block_sum, 2);
    block_sum += shift_left(block_sum, 4);
    block_sum += shift_left(block_sum, 8);
    block_sum += shift_left(block_sum, 16);
    if(warpSize > 32)
        block_sum += shift_left(block_sum, 32);
    if(tid % warpSize == 0)
        sval[tid / warpSize] = block_sum;
    __syncthreads();
    if(tid == 0)
    {
        for(I k = 1; k < LANGE_FROBENIUS_BDIM / warpSize; k++)
            block_sum += sval[k];
        block_sums_block[bid] = block_sum;
    }
}

template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LANGE_FROBENIUS_BDIM)
    lange_frobenius_final_kernel(const rocblas_int m,
                                 const rocblas_int n,
                                 const U A,
                                 const rocblas_int lda,
                                 const rocblas_int shiftA,
                                 const rocblas_int strideA,
                                 S* block_sums,
                                 S* final_norms)
{
    I bid = blockIdx.z;
    I tid = threadIdx.x;

    // select batch instance
    rocblas_int blocks = (m * n - 1) / LANGE_FROBENIUS_BDIM + 1;
    S* block_sum = load_ptr_batch<S>(block_sums, bid, 0, blocks);

    // shared variables
    __shared__ S sval[LANGE_FROBENIUS_BDIM / WarpSize];

    // find maximum of row sums
    S norm_frobenius = 0;
    for(I i = tid; i < blocks; i += LANGE_FROBENIUS_BDIM)
    {
        norm_frobenius += block_sum[i];
    }

    // reduce to find max
    norm_frobenius += shift_left(norm_frobenius, 1);
    norm_frobenius += shift_left(norm_frobenius, 2);
    norm_frobenius += shift_left(norm_frobenius, 4);
    norm_frobenius += shift_left(norm_frobenius, 8);
    norm_frobenius += shift_left(norm_frobenius, 16);
    if(warpSize > 32)
        norm_frobenius += shift_left(norm_frobenius, 32);
    if(tid % warpSize == 0)
        sval[tid / warpSize] = norm_frobenius;
    __syncthreads();
    if(tid == 0)
    {
        for(I k = 1; k < LANGE_FROBENIUS_BDIM / warpSize; k++)
            norm_frobenius += sval[k];
        final_norms[bid] = sqrt(norm_frobenius);
    }
}

template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LANGE_THDS) lange_inf_rows_kernel(const rocblas_int m,
                                                                        const rocblas_int n,
                                                                        const U A,
                                                                        const rocblas_int lda,
                                                                        const rocblas_int shiftA,
                                                                        const rocblas_int strideA,
                                                                        S* row_sums)
{
    I bidz = blockIdx.z;
    I bid = blockIdx.x;
    I tid = threadIdx.x;

    // select batch instance
    T* a = load_ptr_batch<T>(A, bidz, shiftA, strideA);
    S* row_sums_block = load_ptr_batch<S>(row_sums, bidz, 0, m);

    // shared variables
    __shared__ S sval[LANGE_THDS / WarpSize];

    // sum absolute values in row bid
    S row_sum = 0;
    for(I i = tid; i < n; i += LANGE_THDS)
    {
        row_sum += rocblas_abs(a[i * lda + bid]);
    }

    // reduce to get row sum
    row_sum += shift_left(row_sum, 1);
    row_sum += shift_left(row_sum, 2);
    row_sum += shift_left(row_sum, 4);
    row_sum += shift_left(row_sum, 8);
    row_sum += shift_left(row_sum, 16);
    if(warpSize > 32)
        row_sum += shift_left(row_sum, 32);
    if(tid % warpSize == 0)
        sval[tid / warpSize] = row_sum;
    __syncthreads();
    if(tid == 0)
    {
        for(I k = 1; k < LANGE_THDS/ warpSize; k++)
            row_sum += sval[k];
        row_sums_block[bid] = row_sum;
    }
}

template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LANGE_THDS) lange_inf_final_kernel(const rocblas_int m,
                                                                         const rocblas_int n,
                                                                         const U A,
                                                                         const rocblas_int lda,
                                                                         const rocblas_int shiftA,
                                                                         const rocblas_int strideA,
                                                                         S* row_sums,
                                                                         S* final_norms)
{
    I bid = blockIdx.z;
    I tid = threadIdx.x;

    // select batch instance
    S* row_sums_block = load_ptr_batch<S>(row_sums, bid, 0, m);

    // shared variables
    __shared__ S sval[LANGE_THDS / WarpSize];

    // find maximum of row sums
    S norm_one = 0;
    for(I i = tid; i < m; i += LANGE_THDS)
    {
        norm_one = std::max(norm_one, row_sums_block[i]);
    }

    // reduce to find max
    norm_one = std::max(norm_one, shift_left(norm_one, 1));
    norm_one = std::max(norm_one, shift_left(norm_one, 2));
    norm_one = std::max(norm_one, shift_left(norm_one, 4));
    norm_one = std::max(norm_one, shift_left(norm_one, 8));
    norm_one = std::max(norm_one, shift_left(norm_one, 16));
    if(warpSize > 32)
        norm_one = std::max(norm_one, shift_left(norm_one, 32));
    if(tid % warpSize == 0)
        sval[tid / warpSize] = norm_one;
    __syncthreads();
    if(tid == 0)
    {
        for(I k = 1; k < LANGE_THDS / warpSize; k++)
            norm_one = std::max(norm_one, sval[k]);
        final_norms[bid] = norm_one;
    }
}

template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LANGE_THDS) lange_one_columns_kernel(const rocblas_int m,
                                                                           const rocblas_int n,
                                                                           const U A,
                                                                           const rocblas_int lda,
                                                                           const rocblas_int shiftA,
                                                                           const rocblas_int strideA,
                                                                           S* col_sums)
{
    I bidz = blockIdx.z;
    I bid = blockIdx.x;
    I tid = threadIdx.x;

    // select batch instance
    T* a = load_ptr_batch<T>(A, bidz, shiftA, strideA);
    S* col_sums_block = load_ptr_batch<S>(col_sums, bidz, 0, n);

    // shared variables
    __shared__ S sval[LANGE_THDS / WarpSize];

    // sum absolute values in column bid
    S col_sum = 0;
    for(I i = tid; i < m; i += LANGE_THDS)
    {
        col_sum += rocblas_abs(a[i + bid * lda]);
    }

    // reduce to get column sum
    col_sum += shift_left(col_sum, 1);
    col_sum += shift_left(col_sum, 2);
    col_sum += shift_left(col_sum, 4);
    col_sum += shift_left(col_sum, 8);
    col_sum += shift_left(col_sum, 16);
    if(warpSize > 32)
        col_sum += shift_left(col_sum, 32);
    if(tid % warpSize == 0)
        sval[tid / warpSize] = col_sum;
    __syncthreads();
    if(tid == 0)
    {
        for(I k = 1; k < LANGE_THDS / warpSize; k++)
            col_sum += sval[k];
        col_sums_block[bid] = col_sum;
    }
}

template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LANGE_THDS) lange_one_final_kernel(const rocblas_int m,
                                                                         const rocblas_int n,
                                                                         const U A,
                                                                         const rocblas_int lda,
                                                                         const rocblas_int shiftA,
                                                                         const rocblas_int strideA,
                                                                         S* col_sums,
                                                                         S* final_norms)
{
    I bid = blockIdx.z;
    I tid = threadIdx.x;

    // select batch instance
    S* col_sums_block = load_ptr_batch<S>(col_sums, bid, 0, n);

    // shared variables
    __shared__ S sval[LANGE_THDS / WarpSize];

    // find maximum of column sums
    S norm_one = 0;
    for(I i = tid; i < n; i += LANGE_THDS)
    {
        norm_one = std::max(norm_one, col_sums_block[i]);
    }

    // reduce to find max
    norm_one = std::max(norm_one, shift_left(norm_one, 1));
    norm_one = std::max(norm_one, shift_left(norm_one, 2));
    norm_one = std::max(norm_one, shift_left(norm_one, 4));
    norm_one = std::max(norm_one, shift_left(norm_one, 8));
    norm_one = std::max(norm_one, shift_left(norm_one, 16));
    if(warpSize > 32)
        norm_one = std::max(norm_one, shift_left(norm_one, 32));
    if(tid % warpSize == 0)
        sval[tid / warpSize] = norm_one;
    __syncthreads();
    if(tid == 0)
    {
        for(I k = 1; k < LANGE_THDS / warpSize; k++)
            norm_one = std::max(norm_one, sval[k]);
        final_norms[bid] = norm_one;
    }
}

template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LANGE_THDS) lange_max_kernel(const rocblas_int m,
                                                                   const rocblas_int n,
                                                                   const U A,
                                                                   const rocblas_int lda,
                                                                   const rocblas_int shiftA,
                                                                   const rocblas_int strideA,
                                                                   S* final_norms)
{
    I bid = blockIdx.z;
    I tid = threadIdx.x;

    // select batch instance
    T* a = load_ptr_batch<T>(A, bid, shiftA, strideA);

    // shared variables
    __shared__ S sval[LANGE_THDS / WarpSize];

    // dot
    S norm_max = 0;
    for(I i = tid; i < m * n; i += LANGE_THDS)
    {
        int row = i % m;
        int col = i / m;

        norm_max = std::max(norm_max, rocblas_abs(a[row + col * lda]));
    }

    // reduce squared entries to find squared norm of x
    norm_max = std::max(norm_max, shift_left(norm_max, 1));
    norm_max = std::max(norm_max, shift_left(norm_max, 2));
    norm_max = std::max(norm_max, shift_left(norm_max, 4));
    norm_max = std::max(norm_max, shift_left(norm_max, 8));
    norm_max = std::max(norm_max, shift_left(norm_max, 16));
    if(warpSize > 32)
        norm_max = std::max(norm_max, shift_left(norm_max, 32));
    if(tid % warpSize == 0)
        sval[tid / warpSize] = norm_max;
    __syncthreads();
    if(tid == 0)
    {
        for(I k = 1; k < LANGE_THDS / warpSize; k++)
            norm_max = std::max(norm_max, sval[k]);
        final_norms[bid] = norm_max;
    }
}

template <typename T, typename I, typename S>
void rocsolver_lange_getMemorySize(const rocsolver_norm_type norm_type,
                                   const I m,
                                   const I n,
                                   const I batch_count,
                                   size_t* size_work)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || !batch_count)
    {
        *size_work = 0;
        return;
    }

    switch(norm_type)
    {
    case rocsolver_norm_type_max:
    {
        *size_work = 0;
        break;
    }
    case rocsolver_norm_type_one:
    {
        // need space for column sums (one-norm) or row sums (infinity-norm)
        size_t size_per_batch = n;
        *size_work = sizeof(S) * batch_count * size_per_batch;
        break;
    }
    case rocsolver_norm_type_frobenius:
    {
        // need space for row sums
        int blocks = (m * n - 1) / LANGE_FROBENIUS_BDIM + 1;
        size_t size_per_batch = blocks;
        *size_work = sizeof(S) * batch_count * size_per_batch;
        break;
    }
    case rocsolver_norm_type_infinity:
    {
        // need space for row sums
        size_t size_per_batch = m;
        *size_work = sizeof(S) * batch_count * size_per_batch;
        break;
    }
    }
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_lange_argCheck(rocblas_handle handle,
                                        const rocsolver_norm_type norm_type,
                                        const I m,
                                        const I n,
                                        const I lda,
                                        T A,
                                        S* norms)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(norm_type != rocsolver_norm_type_one && norm_type != rocsolver_norm_type_frobenius
       && norm_type != rocsolver_norm_type_infinity && norm_type != rocsolver_norm_type_max)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(m < 0 || n < 0 || lda < m)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m * n && !A) || (m * n && !norms))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename I, typename S, typename U>
rocblas_status rocsolver_lange_template(rocblas_handle handle,
                                        const rocsolver_norm_type norm_type,
                                        const I m,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        const I batch_count,
                                        S* norms,
                                        S* work)
{
    ROCSOLVER_ENTER("lange", "norm_type:", norm_type, "m:", m, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "bc:", batch_count);

    // quick return
    if(m == 0 || n == 0 || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // dispatch to appropriate kernel based on norm type
    switch(norm_type)
    {
    case rocsolver_norm_type_max:
    {
        // Launch max kernel
        ROCSOLVER_LAUNCH_KERNEL((lange_max_kernel<T, I, S>), dim3(1, 1, batch_count),
                                dim3(LANGE_THDS), 0, stream, m, n, A, lda, shiftA, strideA, norms);
        break;
    }
    case rocsolver_norm_type_one:
    {
        // Launch one-norm kernels
        ROCSOLVER_LAUNCH_KERNEL((lange_one_columns_kernel<T, I, S>),
                                dim3(n, 1, batch_count), dim3(LANGE_THDS), 0, stream, m, n, A, lda,
                                shiftA, strideA, work);
        ROCSOLVER_LAUNCH_KERNEL((lange_one_final_kernel<T, I, S>),
                                dim3(1, 1, batch_count), dim3(LANGE_THDS), 0, stream, m, n, A, lda,
                                shiftA, strideA, work, norms);
        break;
    }
    case rocsolver_norm_type_frobenius:
    {
        // Launch Frobenius kernels
        int blocks = (m * n - 1) / LANGE_FROBENIUS_BDIM + 1;
        ROCSOLVER_LAUNCH_KERNEL((lange_frobenius_kernel<T, I, S>),
                                dim3(blocks, 1, batch_count), dim3(LANGE_FROBENIUS_BDIM), 0, stream,
                                m, n, A, lda, shiftA, strideA, work);
        ROCSOLVER_LAUNCH_KERNEL((lange_frobenius_final_kernel<T, I, S>),
                                dim3(1, 1, batch_count), dim3(LANGE_FROBENIUS_BDIM), 0, stream, m,
                                n, A, lda, shiftA, strideA, work, norms);
        break;
    }
    case rocsolver_norm_type_infinity:
    {
        // Launch one-norm kernels
        ROCSOLVER_LAUNCH_KERNEL((lange_inf_rows_kernel<T, I, S>), dim3(m, 1, batch_count),
                                dim3(LANGE_THDS), 0, stream, m, n, A, lda, shiftA, strideA, work);
        ROCSOLVER_LAUNCH_KERNEL((lange_inf_final_kernel<T, I, S>),
                                dim3(1, 1, batch_count), dim3(LANGE_THDS), 0, stream, m, n, A, lda,
                                shiftA, strideA, work, norms);
        break;
    }
    default: return rocblas_status_invalid_value;
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
