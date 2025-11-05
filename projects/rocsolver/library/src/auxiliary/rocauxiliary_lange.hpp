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

ROCSOLVER_BEGIN_NAMESPACE

/*************************************************************
    Templated kernels are instantiated in separate cpp
    files in order to improve compilation times and reduce
    the library size.
*************************************************************/

template <int MAX_THDS, typename T, typename I, typename N, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS) lange_max_kernel(const rocblas_int m,
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
    __shared__ S sval[MAX_THDS / WarpSize];

    // dot
    S norm_max = 0;
    for(I i = tid; i < m * n; i += MAX_THDS)
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
        for(I k = 1; k < MAX_THDS / warpSize; k++)
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

    // size of workspace for column sums (one-norm) or row sums (infinity-norm)
    if(norm_type == rocsolver_norm_type_one)
    {
        // need space for column sums (one-norm) or row sums (infinity-norm)
        size_t size_per_batch =  n;
        *size_work = sizeof(S) * batch_count * size_per_batch;
    } else if (norm_type == rocsolver_norm_type_infinity){
        // need space for row sums 
        size_t size_per_batch = m;
        *size_work = sizeof(S) * batch_count * size_per_batch;
    } else
    {
        // max-norm and Frobenius norm don't need workspace
        *size_work = 0;
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
        constexpr int MAX_THDS = 1024;
        ROCSOLVER_LAUNCH_KERNEL((lange_max_kernel<MAX_THDS, T, I, S>), dim3(1, 1, batch_count),
                                dim3(MAX_THDS), 0, stream, m, n, A, lda, shiftA, strideA, norms);
        break;
    }
    case rocsolver_norm_type_one:
    {
        // Launch one-norm kernels
        constexpr int MAX_THDS = 1024;
        ROCSOLVER_LAUNCH_KERNEL((lange_one_columns_kernel<MAX_THDS, T, I, S>), dim3(n, 1, batch_count),
                                dim3(MAX_THDS), 0, stream, m, n, A, lda, shiftA, strideA, work);
        ROCSOLVER_LAUNCH_KERNEL((lange_one_final_kernel<MAX_THDS, T, I, S>), dim3(1, 1, batch_count),
                                dim3(MAX_THDS), 0, stream, m, n, A, lda, shiftA, strideA, work, norms);
        break;
    }
    case rocsolver_norm_type_frobenius:
        // TODO: Implement Frobenius norm kernel
        return rocblas_status_not_implemented;
    case rocsolver_norm_type_infinity:
        // TODO: Implement infinity-norm kernel
        return rocblas_status_not_implemented;
    default:
        return rocblas_status_invalid_value;
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
