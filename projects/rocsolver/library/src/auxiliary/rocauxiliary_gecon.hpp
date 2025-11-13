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

ROCSOLVER_BEGIN_NAMESPACE

/*************************************************************
    Templated kernels are instantiated in separate cpp
    files in order to improve compilation times and reduce
    the library size.
*************************************************************/

/** LACN2 helper kernels **/

// Initialize vector with constant value
template <typename T, typename I>
ROCSOLVER_KERNEL void gecon_init_vector(T* x, const rocblas_int n, const T value)
{
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    for (I i = tid; i < n; i += hipBlockDim_x * hipGridDim_x)
    {
        if(tid < n)
            x[i] = value;
    }
}

// Compute L1 norm of a vector
template <int MAX_THDS, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS)
    gecon_compute_l1_norm(const S* v, const rocblas_int n, S* norm)
{
    rocblas_int tid = hipThreadIdx_x;

    __shared__ S sval[MAX_THDS / warpSize];

    // Sum absolute values
    S sum = 0;
    for(rocblas_int i = tid; i < n; i += MAX_THDS)
        sum += rocblas_abs(v[i]);

    // Reduce within warp
    sum += shift_left(sum, 1);
    sum += shift_left(sum, 2);
    sum += shift_left(sum, 4);
    sum += shift_left(sum, 8);
    sum += shift_left(sum, 16);
    if(warpSize > 32)
        sum += shift_left(sum, 32);

    if(tid % warpSize == 0)
        sval[tid / warpSize] = sum;
    __syncthreads();

    if(tid == 0)
    {
        for(rocblas_int k = 1; k < MAX_THDS / warpSize; k++)
            sum += sval[k];
        *norm = sum;
    }
}

// reduce within warp using shuffle on both index and value
template <typename T, typename I>
__device__ inline void lacn2_max_index(const I n, T* local_max, I* local_max_index, rocblas_int offset)
{
    T compare_local_max = shift_left(local_max, offset);
    T compare_local_index = shift_left(local_max_index, offset);
    if (compare_local_max > *local_max){
        *local_max = compare_local_max;
        *local_max_index = compare_local_index;
    } 
}

// find index of maximum abosolute value in vector x
// to be called with only a single block
template <int MAX_THDS, typename T, typename I>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS) lacn2_max_index_kernel(const I n,
                                                                   const T* x,
                                                                   I* max_index)
{
    I tid = threadIdx.x;

    // shared variables
    __shared__ S sval[MAX_THDS / WarpSize];
    __shared__ S sval_indices[MAX_THDS / WarpSize];

    // dot
    T local_max = std::numeric_limits<T>::min()
    I local_max_index;
    for(I i = tid; i < m * n; i += MAX_THDS)
    {
        if (rocblas_abs(x[i]) > local_max)
        {
            local_max = rocblas_abs(x[i]);
            local_max_index = i;
        }
    }

    // reduce within warp using shuffle on both index and value
    lacn2_max_index<T, I>(n, &local_max, &local_max_index, 1);
    lacn2_max_index<T, I>(n, &local_max, &local_max_index, 2);
    lacn2_max_index<T, I>(n, &local_max, &local_max_index, 4);
    lacn2_max_index<T, I>(n, &local_max, &local_max_index, 8);
    lacn2_max_index<T, I>(n, &local_max, &local_max_index, 16);
    if (WarpSize > 32)
        lacn2_max_index<T, I>(n, &local_max, &local_max_index, 32);

    if (tid % WarpSize == 0)
        sval[tid / WarpSize] = local_max;
        sval_indices[tid / WarpSize] = local_max_index;
    __syncthreads();


    if(tid == 0)
    {
        for(I k = 1; k < MAX_THDS / warpSize; k++)
        {
            if (rocblas_abs(sval[k]) > local_max)
            {
                local_max = rocblas_abs(sval[k]);
                local_max_index = sval_indices[k];
            }
        }
        *max_index = local_max_index;
    }
}


// sets vector to zero except for index of maximum abs element
template <typename T, typename I>
ROCSOLVER_KERNEL void lacn2_zero_and_set_max_index_kernel(const I n,
                                                                   const T* x,
                                                                   I* max_index)
{
    I tid = threadIdx.x + hipBlockIdx_x * hipBlockDim_x;
    I max_idx = *max_index;

    for (I i = tid; i < n; i += hipBlockDim_x * hipGridDim_x)
    {
        x[i] = T(0);
    }

    if (tid == 0)
        x[max_index] = T(1);

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
template <int MAX_THDS, typename T, typename I>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS) lacn2_jump3(const I n,
                                                                   T* x,
                                                                   T* v,
                                                                   T* isgn,
                                                                   rocblas_int* kase,
                                                                   rocblas_int* jump,
                                                                   const T* est_old,
                                                                   const T* est)
{

    rocblas_int tid = hipThreadIdx_x;

    __shared__ T sval[MAX_THDS / warpSize];
    __shared__ bool sval_repeated[MAX_THDS / warpSize];

    // Sum absolute values
    T sum = 0;
    bool repeated = false;
    for(rocblas_int i = tid; i < n; i += MAX_THDS)
    {
        sum += rocblas_abs(x[i]);
        if(x[i] >= 0){
            if (isgn[i] <= -1)
                repeated = true;
        }
        else{
            if(isgn[i] >= 1)
                repeated = true;

        }

    }

    // Reduce within warp
    sum += shift_left(sum, 1);
    repeated = repeated || __shfl_down(repeated, 1);
    sum += shift_left(sum, 2);
    repeated = repeated || __shfl_down(repeated, 2);
    sum += shift_left(sum, 4);
    repeated = repeated || __shfl_down(repeated, 4);
    sum += shift_left(sum, 8);
    repeated = repeated || __shfl_down(repeated, 8);
    sum += shift_left(sum, 16);
    repeated = repeated || __shfl_down(repeated, 16);
    if(warpSize > 32){
        sum += shift_left(sum, 32);
        repeated = repeated || __shfl_down(repeated, 32);
    }

    if(tid % warpSize == 0)
        sval[tid / warpSize] = sum;
        sval_repeated[tid / warpSize] = repeated;
    __syncthreads();

    if(tid == 0)
    {
        for(rocblas_int k = 1; k < MAX_THDS / warpSize; k++)
        {
            sum += sval[k];
            repeated = repeated || sval_repeated[k];
        }
        *est_old = *est;
        *est = sum;
        sval_repeated[0] = repeated;
    }

    __syncthreads();

    repeated = sval_repeated[0];

    if (repeated || (*est <= *est_old)){
        for(rocblas_int i = tid; i < n; i += MAX_THDS)
        {
            // Alternating pattern that changes each iteration
            T sign = (i % 2 == 0) ? T(1) : T(-1);
            x[i] = sign * (T(1) + S(i) / S(n-1));
        }
        if(tid == 0){
            *kase = 1;
            *jump = 5; 
        }

        return;
    }

    for(rocblas_int i = tid; i < n; i += MAX_THDS)
    {
        if (x[i] >= 0)
        {
            x[i] = T(1);
            isgn[i] = T(1);
        }
        else{
            x[i] = T(-1);
            isgn[i] = T(-1);

        }
        // Alternating pattern that changes each iteration
        T sign = (i % 2 == 0) ? T(1) : T(-1);
        x[i] = sign * (T(1) + S(i) / S(n-1));
    }

    if (tid == 0){
        *kase = 2;
        *jump = 4; 
    }
    
}

template <int MAX_THDS, typename T, typename I>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS) lacn2_jump4(const I n,
                                                                   T* x,
                                                                   T* v,
                                                                   T* isgn,
                                                                   rocblas_int* kase,
                                                                   rocblas_int* jump,
                                                                   const I* d_max_idx,
                                                                   const I* iter,
                                                                   const I* iters_max,
                                                                   const T* est)
{
    I tid = threadIdx.x;

    int jlast = d_max_idx[0];

    // shared variables
    __shared__ S sval[MAX_THDS / WarpSize];
    __shared__ S sval_indices[MAX_THDS / WarpSize];

    // dot
    T local_max = std::numeric_limits<T>::min()
    I local_max_index;
    for(I i = tid; i < m * n; i += MAX_THDS)
    {
        if (rocblas_abs(x[i]) > local_max)
        {
            local_max = rocblas_abs(x[i]);
            local_max_index = i;
        }
    }

    // reduce within warp using shuffle on both index and value
    lacn2_max_index<T, I>(n, &local_max, &local_max_index, 1);
    lacn2_max_index<T, I>(n, &local_max, &local_max_index, 2);
    lacn2_max_index<T, I>(n, &local_max, &local_max_index, 4);
    lacn2_max_index<T, I>(n, &local_max, &local_max_index, 8);
    lacn2_max_index<T, I>(n, &local_max, &local_max_index, 16);
    if (WarpSize > 32)
        lacn2_max_index<T, I>(n, &local_max, &local_max_index, 32);

    if (tid % WarpSize == 0)
        sval[tid / WarpSize] = local_max;
        sval_indices[tid / WarpSize] = local_max_index;
    __syncthreads();


    if(tid == 0)
    {
        for(I k = 1; k < MAX_THDS / warpSize; k++)
        {
            if (rocblas_abs(sval[k]) > local_max)
            {
                local_max = rocblas_abs(sval[k]);
                local_max_index = sval_indices[k];
            }
        }
        sval_indices[0] = local_max_index;
        d_max_idx[0] = local_max_index;
    }

    __syncthreads();

    I max_idx = sval_indices[0];
    if (max_idx == jlast || iters[0] == iters_max[0]){
        for(rocblas_int i = tid; i < n; i += MAX_THDS)
        {
            // Alternating pattern that changes each iteration
            T sign = (i % 2 == 0) ? T(1) : T(-1);
            x[i] = sign * (T(1) + S(i) / S(n-1));
        }
        if(tid == 0){
            *kase = 1;
            *jump = 5; 
        }

        return;
    }

    for (I i = tid; i < n; i += MAX_THDS)
    {
        x[i] = T(0);
    }

    if (tid == 0)
    {
        x[max_idx] = T(1);
        *kase = 1;
        *jump = 3;
    }
}

// compute l1 norm of vector v and compare to temporary value
template <int MAX_THDS, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS)
    lacn2_jump5(const S* v, const rocblas_int n, S* norm)
{
    rocblas_int tid = hipThreadIdx_x;

    __shared__ S sval[MAX_THDS / warpSize];

    // Sum absolute values
    S sum = 0;
    for(rocblas_int i = tid; i < n; i += MAX_THDS)
        sum += rocblas_abs(v[i]);

    // Reduce within warp
    sum += shift_left(sum, 1);
    sum += shift_left(sum, 2);
    sum += shift_left(sum, 4);
    sum += shift_left(sum, 8);
    sum += shift_left(sum, 16);
    if(warpSize > 32)
        sum += shift_left(sum, 32);

    if(tid % warpSize == 0)
        sval[tid / warpSize] = sum;
    __syncthreads();

    if(tid == 0)
    {
        for(rocblas_int k = 1; k < MAX_THDS / warpSize; k++)
            sum += sval[k];
        if (2*(sum/(3*n)) > *norm)
            *norm = sum;
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
                 S* v,
                 S* x,
                 I* isave,
                 S* est,
                 const I max_iter,
                 S* d_est,
                 S* d_estold,
                 I* d_max_idx,
                 I* d_jlast,
                 I* d_should_break,
                 I* d_is_equal,
                 rocblas_int* d_jump,
                 rocblas_int* d_kase,
                 rocblas_int* jump,
                 rocblas_int* kase)
{
    // TODO: consider what return value should be, should we bother with returning success in each case, or just after switch statement?
    // TODO: consider if synchronizations required between kernel laynches or memory transfers :)
    switch (jump){
        case 1:
            if (n == 1)
            {
                hipMemcpyAsync(est, x, sizeof(S), hipMemcpyDeviceToHost, stream);
                hipStreamSynchronize(stream);
                *est = rocblas_abs(*est);
                *kase = 0; // signal to exit
                return rocblas_status_success;
            }

            ROCSOLVER_LAUNCH_KERNEL((gecon_compute_l1_norm<MAX_THDS, S>), dim3(1), threads, 0,
                                    stream, x, n, d_est);

            // one of the AI generated kernels might be fine for this :)
            ROCSOLVER_LAUNCH_KERNEL(lacn2_init_alternating_vector<S>, blocks, threads, 0, stream, x, isgn, n);

            *kase = 2;
            *jump = 2;

            return rocblas_status_success;
            break;
        case 2:
            ROCSOLVER_LAUNCH_KERNEL(lacn2_max_index_kernel<T, I>, blocks, threads, 0, stream, n, x, d_max_idx);
            ROCSOLVER_LAUNCH_KERNEL(lacn2_zero_and_set_max_index_kernel<T, I>, blocks, threads, 0, stream, n, x, d_max_idx);

            // isave(2) = 3?????
            // TODO: how to reflect this

            *kase = 1;
            *jump = 3;

            return rocblas_status_success;
            break;
        case 3:
            ROCSOLVER_LAUNCH_KERNEL((lacn2_jump3<MAX_THDS, S>), dim3(1), threads, 0,
                                    stream, x, n, isgn, d_is_equal);
            // ROCSOLVER_LAUNCH_KERNEL((gecon_compute_l1_norm<MAX_THDS, S>), dim3(1), threads, 0,
            //                         stream, x, n, d_est);
            // TODO: transfer updated jump and case from device to host
            return rocblas_status_success;
            break;
        case 4:
            // int jlast; 
            // hipMemcpyAsync(jlast, d_max_idx, sizeof(I), hipMemcpyDeviceToHost, stream);
            // hipStreamSynchronize(stream);
            ROCSOLVER_LAUNCH_KERNEL((lacn2_jump4<MAX_THDS, S>), dim3(1), threads, 0,
                                    stream, x, n, isgn, d_is_equal);



            break;
        case 5:
            ROCSOLVER_LAUNCH_KERNEL((lacn2_jump5<MAX_THDS, S>), dim3(1), threads, 0,
                                    stream, x, n, d_est);
            *kase = 0;

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
                                        const S anorm,
                                        S* rcond,
                                        const I batch_count,
                                        S* work,
                                        I* iwork,
                                        const I max_iter)
{

    // may make sense to iterate batch by batch since convergence and steps may be different for both?



    ROCSOLVER_LAUNCH_KERNEL(gecon_init_vector<S>, blocks, threads, 0, stream, x, n, S(1) / S(n));


    rocblas_int jump = 1;
    
}

/** Memory size calculation **/
template <typename T, typename I, typename S>
void rocsolver_gecon_getMemorySize(const I n,
                                   const I batch_count,
                                   size_t* size_work_v,
                                   size_t* size_work_x,
                                   size_t* size_iwork,
                                   size_t* size_work_scalars_s,
                                   size_t* size_work_scalars_i)
{
    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
    {
        *size_work_v = 0;
        *size_work_x = 0;
        *size_iwork = 0;
        *size_work_scalars_s = 0;
        *size_work_scalars_i = 0;
        return;
    }

    // Need n reals per batch for v vector
    *size_work_v = sizeof(S) * n * batch_count;

    // Need n reals per batch for x vector  
    *size_work_x = sizeof(S) * n * batch_count;

    // Need n integers per batch (index tracking for LACN2)
    *size_iwork = sizeof(I) * n * batch_count;

    // Need scalar workspace for LACN2 state (per batch): d_est, d_estold
    *size_work_scalars_s = sizeof(S) * 2 * batch_count;

    // Need integer workspace for LACN2 state (per batch): d_max_idx, d_jlast, d_should_break, d_is_equal
    *size_work_scalars_i = sizeof(I) * 4 * batch_count;
}

/** Argument validation **/
template <typename T, typename I, typename S>
rocblas_status rocsolver_gecon_argCheck(rocblas_handle handle,
                                        const rocsolver_norm_type norm_type,
                                        const I n,
                                        const I lda,
                                        T A,
                                        const I* ipiv,
                                        const S anorm,
                                        S* rcond)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(norm_type != rocsolver_norm_type_one && norm_type != rocsolver_norm_type_infinity
       && norm_type != rocsolver_norm_type_frobenius && norm_type != rocsolver_norm_type_max)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || anorm < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !ipiv) || !rcond)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** Main template function **/
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
                                        const S anorm,
                                        S* rcond,
                                        const I batch_count,
                                        S* work,
                                        I* iwork,
                                        const I max_iter)
{
    ROCSOLVER_ENTER("gecon", "norm_type:", norm_type, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return if no dimensions
    if(n == 0)
    {
        rocblas_int blocks = (batch_count - 1) / BS1 + 1;
        dim3 grid(blocks, 1, 1);
        dim3 threads(BS1, 1, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, rcond, batch_count, 0);
        return rocblas_status_success;
    }

    // Return not implemented for Frobenius and max norms
    if(norm_type == rocsolver_norm_type_frobenius || norm_type == rocsolver_norm_type_max)
        return rocblas_status_not_implemented;

    // If anorm is zero, rcond is zero
    if(anorm == S(0))
    {
        rocblas_int blocks = (batch_count - 1) / BS1 + 1;
        dim3 grid(blocks, 1, 1);
        dim3 threads(BS1, 1, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, rcond, batch_count, 0);
        return rocblas_status_success;
    }

    // Main condition estimation loop
    for(I batch = 0; batch < batch_count; batch++)
    {
        // Get workspace pointers for this batch
        S* v = work + batch * n;
        S* x = work + batch * n + n * batch_count;
        I* isave = iwork + batch * n;
        
        // Get scalar workspace pointers for this batch (stored after main work arrays)
        S* scalars_s = work + 2 * n * batch_count + batch * 2;
        S* d_est_batch = scalars_s;
        S* d_estold_batch = scalars_s + 1;
        
        I* scalars_i = iwork + n * batch_count + batch * 4;
        I* d_max_idx_batch = scalars_i;
        I* d_jlast_batch = scalars_i + 1;
        I* d_should_break_batch = scalars_i + 2;
        I* d_is_equal_batch = scalars_i + 3;

        // Call LACN2 algorithm (result stays on device)
        gecon_lacn2<T, I, S>(handle, norm_type, n, A, shiftA + batch * strideA, inca, lda, strideA,
                             ipiv + batch * strideP, strideP, v, x, isave, max_iter, d_est_batch,
                             d_estold_batch, d_max_idx_batch, d_jlast_batch, d_should_break_batch,
                             d_is_equal_batch);
    }

    // NECESSARY TRANSFER: Final result - transfer all estimates back to host at once
    S* d_est_array = work + 2 * n * batch_count;
    S* h_est = new S[batch_count];
    
    // Gather estimates (they're already in device memory at the right locations)
    for(I batch = 0; batch < batch_count; batch++)
    {
        hipMemcpyAsync(&h_est[batch], d_est_array + batch * 2, sizeof(S), 
                      hipMemcpyDeviceToHost, stream);
    }
    hipStreamSynchronize(stream);

    // Compute reciprocal condition numbers on host
    for(I batch = 0; batch < batch_count; batch++)
    {
        S ainvnm = h_est[batch];
        if(ainvnm != S(0))
            rcond[batch] = (S(1) / ainvnm) / anorm;
        else
            rcond[batch] = S(0);
    }

    // Cleanup
    delete[] h_est;

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
