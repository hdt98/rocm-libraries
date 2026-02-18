/* **************************************************************************
 * getf2_small_kernel - Small LU Factorization Kernel
 *
 * Based on rocSOLVER's roclapack_getf2_specialized_kernels.hpp
 *
 * Small kernel algorithm based on:
 * Abdelfattah, A., Haidar, A., Tomov, S., & Dongarra, J. (2017).
 * Factorization and inversion of a million matrices using GPUs: Challenges
 * and countermeasures. Procedia Computer Science, 108, 606-615.
 *
 * MODIFICATION FOR TESTING:
 * The __syncthreads() call that was originally on line 103 (synchronize
 * across waves before overwriting common) has been REMOVED to test race
 * condition detection with CUDA Compute Sanitizer.
 *
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc.
 * *************************************************************************/

#pragma once

#include "../cuda_compat.cuh"
#include "../rocsolver_types.cuh"
#include "../device_helpers.cuh"

ROCSOLVER_BEGIN_NAMESPACE

/** getf2_small_kernel takes care of matrices with m < n
    m <= GETF2_MAX_THDS and n <= GETF2_MAX_COLS **/
template <int DIM, typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(GETF2_SSKER_MAX_M)
    getf2_small_kernel(const I m,
                       U AA,
                       const rocblas_stride shiftA,
                       const I lda,
                       const rocblas_stride strideA,
                       I* ipivA,
                       const rocblas_stride shiftP,
                       const rocblas_stride strideP,
                       INFO* infoA,
                       const I batch_count,
                       const I offset,
                       I* permut_idx,
                       const rocblas_stride stridePI)
{
    using S = decltype(std::real(T{}));

    I myrow = hipThreadIdx_x;
    const I ty = hipThreadIdx_y;
    const I id = hipBlockIdx_y * static_cast<I>(hipBlockDim_y) + ty;

    if(id >= batch_count)
        return;

    // batch instance
    T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);
    I* ipiv = load_ptr_batch<I>(ipivA, id, shiftP, strideP);
    I* permut = (permut_idx != nullptr ? permut_idx + id * stridePI : nullptr);
    INFO* info = infoA + id;

    // shared memory (for communication between threads in group)
    // (SHUFFLES DO NOT IMPROVE PERFORMANCE IN THIS CASE)
    extern __shared__ double lmem[];
    T* common = reinterpret_cast<T*>(lmem);
    common += ty * max(m, static_cast<I>(DIM));

    // local variables
    T pivot_value;
    T test_value;
    I pivot_index;
    I mypiv = myrow + 1; // to build ipiv
    INFO myinfo = 0; // to build info
    T rA[DIM]; // to store this-row values

    // read corresponding row from global memory into local array
#pragma unroll DIM
    for(I j = 0; j < DIM; ++j)
        rA[j] = A[myrow + j * lda];

        // for each pivot (main loop)
#pragma unroll DIM
    for(I k = 0; k < DIM; ++k)
    {
        // share current column
        common[myrow] = rA[k];
        __syncthreads();

        // search pivot index
        pivot_index = k;
        pivot_value = common[k];
        for(I i = k + 1; i < m; ++i)
        {
            test_value = common[i];
            if(aabs<S>(pivot_value) < aabs<S>(test_value))
            {
                pivot_value = test_value;
                pivot_index = i;
            }
        }

        // check singularity and scale value for current column
        if(pivot_value != T(0))
            pivot_value = S(1) / pivot_value;
        else if(myinfo == 0)
            myinfo = k + 1;

        // NOTE: __syncthreads() REMOVED HERE for testing race condition detection
        // Original code had: __syncthreads();
        // This synchronization was to "synchronize across waves before overwriting common"

        // swap rows (lazy swapping)
        if(myrow == pivot_index)
        {
            myrow = k;
            // share pivot row
            for(I j = k + 1; j < DIM; ++j)
                common[j] = rA[j];
        }
        else if(myrow == k)
        {
            myrow = pivot_index;
            mypiv = pivot_index + 1;
            if(permut_idx && pivot_index != k)
                swap(permut[k], permut[pivot_index]);
        }
        __syncthreads();

        // scale current column and update trailing matrix
        if(myrow > k)
        {
            rA[k] *= pivot_value;
            for(I j = k + 1; j < DIM; ++j)
                rA[j] -= rA[k] * common[j];
        }
        __syncthreads();
    }

    // write results to global memory
    if(myrow < DIM)
        ipiv[myrow] = mypiv + offset;
    if(myrow == 0 && *info == 0 && myinfo > 0)
        *info = myinfo + offset;
#pragma unroll DIM
    for(I j = 0; j < DIM; ++j)
        A[myrow + j * lda] = rA[j];
}

/** getf2_npvt_small_kernel (non pivoting version) **/
template <int DIM, typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(GETF2_SSKER_MAX_M)
    getf2_npvt_small_kernel(const I m,
                            U AA,
                            const rocblas_stride shiftA,
                            const I lda,
                            const rocblas_stride strideA,
                            INFO* infoA,
                            const I batch_count,
                            const I offset)
{
    using S = decltype(std::real(T{}));

    I myrow = hipThreadIdx_x;
    const I ty = hipThreadIdx_y;
    const I id = hipBlockIdx_y * static_cast<I>(hipBlockDim_y) + ty;

    if(id >= batch_count)
        return;

    // batch instance
    T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);
    INFO* info = infoA + id;

    // shared memory (for communication between threads in group)
    // (SHUFFLES DO NOT IMPROVE PERFORMANCE IN THIS CASE)
    extern __shared__ double lmem[];
    T* common = reinterpret_cast<T*>(lmem);
    T* val = common + hipBlockDim_y * DIM;
    common += ty * DIM;

    // local variables
    INFO myinfo = 0; // to build info
    T rA[DIM]; // to store this-row values

    // read corresponding row from global memory into local array
#pragma unroll DIM
    for(I j = 0; j < DIM; ++j)
        rA[j] = A[myrow + j * lda];

        // for each pivot (main loop)
#pragma unroll DIM
    for(I k = 0; k < DIM; ++k)
    {
        // share pivot row
        if(myrow == k)
        {
            val[ty] = rA[k];
            for(I j = k + 1; j < DIM; ++j)
                common[j] = rA[j];

            if(val[ty] != T(0))
                val[ty] = S(1) / val[ty];
        }
        __syncthreads();

        // check singularity
        if(val[ty] == 0 && myinfo == 0)
            myinfo = k + 1;

        // scale current column and update trailing matrix
        if(myrow > k)
        {
            rA[k] *= val[ty];
            for(I j = k + 1; j < DIM; ++j)
                rA[j] -= rA[k] * common[j];
        }
        __syncthreads();
    }

    // write results to global memory
    if(myrow == 0 && *info == 0 && myinfo > 0)
        *info = myinfo + offset;
#pragma unroll DIM
    for(I j = 0; j < DIM; ++j)
        A[myrow + j * lda] = rA[j];
}

ROCSOLVER_END_NAMESPACE
