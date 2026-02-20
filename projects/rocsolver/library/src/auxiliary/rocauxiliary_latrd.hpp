/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
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

#include "../auxiliary/rocauxiliary_lacgv.hpp"
#include "../auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#include "rocsolver_device_workspace.hpp"
#include <rocprofiler-sdk-roctx/roctx.h>

static bool print_debug_messages_latrd_forsytrd
    = std::getenv("PRINT_DEBUG") != nullptr ? true : false;

ROCSOLVER_BEGIN_NAMESPACE

template <int MAX_THDS, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS) latrd_dot_scale_axpy(const I n,
                                                                       U AA,
                                                                       const rocblas_stride shiftA,
                                                                       const rocblas_stride strideA,
                                                                       T* pWW,
                                                                       const rocblas_stride shiftW,
                                                                       const rocblas_stride strideW,
                                                                       T* pTauA,
                                                                       const rocblas_stride strideP)
{
    I bid = blockIdx.z;
    I tid = threadIdx.x;

    // select batch instance
    T* pA = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* pW = load_ptr_batch<T>(pWW, bid, shiftW, strideW);
    T* pTau = load_ptr_batch<T>(pTauA, bid, 0, strideP);

    // shared variables
    __shared__ T sval[MAX_THDS / WarpSize];
    __shared__ T sh_A[MAX_THDS];
    __shared__ T sh_W[MAX_THDS];

    // dot
    T norm2 = 0;
    for(I i = tid; i < n; i += MAX_THDS) //todo: too much runtime determined things
    {
        T tempA = pA[i];
        T tempW = pW[i];
        if(i < MAX_THDS)
        {
            sh_A[i] = tempA;
            sh_W[i] = tempW;
        }

        norm2 += tempA * conj(tempW);
    }

    // reduce squared entries to find squared norm of x
    // this can be a single builtin wave reduce add
    norm2 += shift_left(norm2, 1);
    norm2 += shift_left(norm2, 2);
    norm2 += shift_left(norm2, 4);
    norm2 += shift_left(norm2, 8);
    norm2 += shift_left(norm2, 16);
    if(warpSize > 32)
        norm2 += shift_left(norm2, 32);
    if(tid % warpSize == 0)
        sval[tid / warpSize] = norm2;
    __syncthreads();
    if(tid == 0)
    {
        for(I k = 1; k < MAX_THDS / warpSize; k++)
            norm2 += sval[k];
        sval[0] = -0.5 * pTau[0] * norm2;
    }
    __syncthreads();
    auto w_buffer  = __builtin_amdgcn_make_buffer_rsrc(pW, 0,0xffffffff,0x00020000);
    // axpy
    //TODO, excessive traffic to global memory, can we keep W in registers/shared memory?
    //make last write to W l2 passthrough
    const T alpha = sval[0];
    for(I i = tid; i < n; i += MAX_THDS)
    {
        T result;
        if(i < MAX_THDS)
            result = sh_W[i] + alpha * sh_A[i];
        else
            result = pW[i] + alpha * pA[i];

        int out_addr = i * sizeof(T);

        if constexpr(sizeof(T) == 1)
            __builtin_amdgcn_raw_buffer_store_b8(result, w_buffer, out_addr, 0,0x0);
        else if constexpr(sizeof(T) == 2)
            __builtin_amdgcn_raw_buffer_store_b16(result, w_buffer, out_addr, 0,0x0);
        else if constexpr(sizeof(T) == 4)
            __builtin_amdgcn_raw_buffer_store_b32(result, w_buffer, out_addr, 0,0x0);
        else if constexpr(sizeof(T) == 8)
        {
            using uint32x2_t = uint32_t __attribute__((ext_vector_type(2)));
            uint32x2_t tmp;
            memcpy(&tmp, &result, sizeof(T));
            __builtin_amdgcn_raw_buffer_store_b64(tmp, w_buffer, out_addr, 0,0x0);
        }
    }
}

/********************************************************************************/
/******************* Host functions for latrd api *******************************/
/********************************************************************************/
template <typename T, typename S, typename U>
rocblas_status rocsolver_latrd_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        const rocblas_int lda,
                                        const rocblas_int ldw,
                                        T A,
                                        S E,
                                        U tau,
                                        U W,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || k < 0 || k > n || lda < n || ldw < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !E) || (n && !tau) || (n && k && !W))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, typename T>
void rocsolver_latrd_getMemorySize(const rocblas_int n,
                                   const rocblas_int k,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work,
                                   size_t* size_norms,
                                   size_t* size_workArr)
{
    // if quick return no workspace needed
    if(n == 0 || k == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_norms = 0;
        *size_workArr = 0;
        return;
    }

    size_t n1 = 0, n2 = 0;
    size_t w1 = 0, w2 = 0, w3 = 0;

    // size of scalars (constants) for rocblas calls
    *size_scalars = sizeof(T) * 3;

    // size of array of pointers (batched cases)
    if(BATCHED)
        *size_workArr = 2 * sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    // extra requirements for calling larfg
    rocsolver_larfg_getMemorySize<T>(n, batch_count, &w1, &n1);

    // extra requirements for calling symv/hemv
    rocblasCall_symv_hemv_mem<BATCHED, T>(n, batch_count, &w2);

    // size of re-usable workspace
    // TODO: replace with rocBLAS call
    constexpr int ROCBLAS_DOT_NB = 512;
    w3 = n > 2 ? (n - 2) / ROCBLAS_DOT_NB + 2 : 1;
    w3 *= sizeof(T) * batch_count;
    n2 = sizeof(T) * batch_count;

    *size_norms = std::max(n1, n2);
    *size_work = std::max({w1, w2, w3});
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_latrd_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        S* E,
                                        const rocblas_stride strideE,
                                        T* tau,
                                        const rocblas_stride strideP,
                                        T* W,
                                        const rocblas_int shiftW,
                                        const rocblas_int ldw,
                                        const rocblas_stride strideW,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        T* work,
                                        T* norms,
                                        T** workArr)
{
    ROCSOLVER_ENTER("latrd", "uplo:", uplo, "n:", n, "k:", k, "shiftA:", shiftA, "lda:", lda,
                    "shiftW:", shiftW, "ldw:", ldw, "bc:", batch_count);
    roctxRangePush("rocsolver_latrd");

    // quick return
    if(n == 0 || k == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    if(uplo == rocblas_fill_lower)
    {
        // reduce the first k columns of A
        // main loop running forwards (for each column)
        for(rocblas_int j = 0; j < k; ++j)
        {
            // update column j of A with reflector computed in step j-1
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, j, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                                            batch_count);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j, j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(j, 0, lda), lda,
                                strideA, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda), 1,
                                strideA, batch_count, workArr);

            if(COMPLEX)
            {
                rocsolver_lacgv_template<T>(handle, j, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                                            batch_count);
                rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                            batch_count);
            }

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j, j,
                                cast2constType<T>(scalars), 0, W, shiftW + idx2D(j, 0, ldw), ldw,
                                strideW, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda), 1,
                                strideA, batch_count, workArr);

            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                            batch_count);

            // generate Householder reflector to work on column j
            rocsolver_larfg_template(handle, n - j - 1, A, shiftA + idx2D(j + 1, j, lda), E, j,
                                     strideE, A, shiftA + idx2D(std::min(j + 2, n - 1), j, lda), 1,
                                     strideA, (tau + j), strideP, batch_count, work, norms);

            // compute/update column j of W
            rocblasCall_symv_hemv<T>(
                handle, uplo, n - 1 - j, (scalars + 2), 0, A, shiftA + idx2D(j + 1, j + 1, lda),
                lda, strideA, A, shiftA + idx2D(j + 1, j, lda), 1, strideA, (scalars + 1), 0, W,
                shiftW + idx2D(j + 1, j, ldw), 1, strideW, batch_count, work, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, n - j - 1, j,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(j + 1, 0, ldw),
                                ldw, strideW, A, shiftA + idx2D(j + 1, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, W, shiftW + idx2D(0, j, ldw), 1,
                                strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j - 1, j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(j + 1, 0, lda),
                                lda, strideA, W, shiftW + idx2D(0, j, ldw), 1, strideW,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(j + 1, j, ldw),
                                1, strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, n - j - 1, j,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j + 1, 0, lda),
                                lda, strideA, A, shiftA + idx2D(j + 1, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, W, shiftW + idx2D(0, j, ldw), 1,
                                strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j - 1, j,
                                cast2constType<T>(scalars), 0, W, shiftW + idx2D(j + 1, 0, ldw),
                                ldw, strideW, W, shiftW + idx2D(0, j, ldw), 1, strideW,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(j + 1, j, ldw),
                                1, strideW, batch_count, workArr);

            rocblasCall_scal<T>(handle, n - j - 1, (tau + j), strideP, W,
                                shiftW + idx2D(j + 1, j, ldw), 1, strideW, batch_count);

            ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<1024, T>), dim3(1, 1, batch_count),
                                    dim3(1024, 1, 1), 0, stream, n - 1 - j, A,
                                    shiftA + idx2D(j + 1, j, lda), strideA, W,
                                    shiftW + idx2D(j + 1, j, ldw), strideW, tau + j, strideP);
        }
    }

    else
    {
        // reduce the last k columns of A
        // main loop running forwards (for each column)
        rocblas_int jw;
        for(rocblas_int j = n - 1; j >= n - k; --j)
        {
            jw = j - n + k;
            // update column j of A with reflector computed in step j-1
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, n - 1 - j, W, shiftW + idx2D(j, jw + 1, ldw),
                                            ldw, strideW, batch_count);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, j + 1, n - 1 - j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(0, j + 1, lda),
                                lda, strideA, W, shiftW + idx2D(j, jw + 1, ldw), ldw, strideW,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(0, j, lda), 1,
                                strideA, batch_count, workArr);

            if(COMPLEX)
            {
                rocsolver_lacgv_template<T>(handle, n - 1 - j, W, shiftW + idx2D(j, jw + 1, ldw),
                                            ldw, strideW, batch_count);
                rocsolver_lacgv_template<T>(handle, n - 1 - j, A, shiftA + idx2D(j, j + 1, lda),
                                            lda, strideA, batch_count);
            }

            rocblasCall_gemv<T>(handle, rocblas_operation_none, j + 1, n - 1 - j,
                                cast2constType<T>(scalars), 0, W, shiftW + idx2D(0, jw + 1, ldw),
                                ldw, strideW, A, shiftA + idx2D(j, j + 1, lda), lda, strideA,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(0, j, lda), 1,
                                strideA, batch_count, workArr);

            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, n - 1 - j, A, shiftA + idx2D(j, j + 1, lda),
                                            lda, strideA, batch_count);

            // generate Householder reflector to work on column j
            rocsolver_larfg_template(handle, j, A, shiftA + idx2D(j - 1, j, lda), E, j - 1, strideE,
                                     A, shiftA + idx2D(0, j, lda), 1, strideA, (tau + j - 1),
                                     strideP, batch_count, work, norms);

            // compute/update column j of W
            rocblasCall_symv_hemv<T>(handle, uplo, j, (scalars + 2), 0, A, shiftA, lda, strideA, A,
                                     shiftA + idx2D(0, j, lda), 1, strideA, (scalars + 1), 0, W,
                                     shiftW + idx2D(0, jw, ldw), 1, strideW, batch_count, work,
                                     workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, j, n - 1 - j,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(0, jw + 1, ldw),
                                ldw, strideW, A, shiftA + idx2D(0, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, W,
                                shiftW + idx2D(j + 1, jw, ldw), 1, strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, j, n - 1 - j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(0, j + 1, lda),
                                lda, strideA, W, shiftW + idx2D(j + 1, jw, ldw), 1, strideW,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(0, jw, ldw), 1,
                                strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, j, n - 1 - j,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(0, j + 1, lda),
                                lda, strideA, A, shiftA + idx2D(0, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, W,
                                shiftW + idx2D(j + 1, jw, ldw), 1, strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, j, n - 1 - j,
                                cast2constType<T>(scalars), 0, W, shiftW + idx2D(0, jw + 1, ldw),
                                ldw, strideW, W, shiftW + idx2D(j + 1, jw, ldw), 1, strideW,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(0, jw, ldw), 1,
                                strideW, batch_count, workArr);

            rocblasCall_scal<T>(handle, j, (tau + j - 1), strideP, W, shiftW + idx2D(0, jw, ldw), 1,
                                strideW, batch_count);

            ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<1024, T>), dim3(1, 1, batch_count),
                                    dim3(1024, 1, 1), 0, stream, j, A, shiftA + idx2D(0, j, lda),
                                    strideA, W, shiftW + idx2D(0, jw, ldw), strideW, tau + j - 1,
                                    strideP);
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    roctxRangePop();
    return rocblas_status_success;
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
auto rocsolver_latrd_getWorkItems(rocblas_handle handle,
                                  const rocblas_fill uplo,
                                  const rocblas_int n,
                                  const rocblas_int k,
                                  U A,
                                  const rocblas_int shiftA,
                                  const rocblas_int lda,
                                  const rocblas_stride strideA,
                                  S* E,
                                  const rocblas_stride strideE,
                                  T* tau,
                                  const rocblas_stride strideP,
                                  T* W,
                                  const rocblas_int shiftW,
                                  const rocblas_int ldw,
                                  const rocblas_stride strideW,
                                  const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_workArr;
    // extra requirements for calling LARFG
    size_t size_work, size_norms;
    rocsolver_latrd_getMemorySize<false, T>(n, k, batch_count, &size_scalars, &size_work,
                                            &size_norms, &size_workArr);

    auto work_items = create_work_item({"latrd_scalars", size_scalars})
        + create_work_item({"latrd_workArr", size_workArr})
        + create_work_item({"latrd_work", size_work}) + create_work_item({"latrd_norms", size_norms});

    return work_items;
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_latrd_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        S* E,
                                        const rocblas_stride strideE,
                                        T* tau,
                                        const rocblas_stride strideP,
                                        T* W,
                                        const rocblas_int shiftW,
                                        const rocblas_int ldw,
                                        const rocblas_stride strideW,
                                        const rocblas_int batch_count,
                                        rocsolver_device_workspace_ptr_t dwptr)
{
    ROCSOLVER_INIT_DEVICE_WORKSPACE(
        dwptr,
        rocsolver_latrd_getWorkItems(handle, uplo, n, k, A, shiftA, lda, strideA, E, strideE, tau,
                                     strideP, W, shiftW, ldw, strideW, batch_count));

    T* scalars = (T*)dwptr->work("latrd_scalars");
    T* work = (T*)dwptr->work("latrd_work");
    T* norms = (T*)dwptr->work("latrd_norms");
    T** workArr = (T**)dwptr->work("latrd_workArr");

    if(dwptr->size("latrd_scalars") > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_latrd_template<T>(handle, uplo, n, k, A, shiftA, lda, strideA, E, strideE, tau,
                                       strideP, W, shiftW, ldw, strideW, batch_count, (T*)scalars,
                                       (T*)work, (T*)norms, (T**)workArr);
}

/**************************************************************************************/
/***************** Kernels/Device functions *******************************************/
/**************************************************************************************/

/***** Kernel to reduce results inter-groups *****/
/*************************************************/
template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_reduce_kernel(const rocblas_fill uplo,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int c,
                                          T* dacA,
                                          const rocblas_int ldd,
                                          const rocblas_stride strideD,
                                          U yA,
                                          const rocblas_int shiftY,
                                          const rocblas_int ldy,
                                          const rocblas_stride strideY,
                                          T* pWorkA,
                                          const rocblas_stride strideblk)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idr = bidr * threadsr + tidr;
    int idc = bidc * threadsc + tidc;

    // select batch instance
    bool upper = (uplo == rocblas_fill_upper);
    T* y1 = upper ? load_ptr_batch<T>(yA, bid, shiftY, strideY) : pWorkA + bid * strideblk;
    T* y2 = upper ? pWorkA + bid * strideblk : load_ptr_batch<T>(yA, bid, shiftY, strideY);
    T* dac = dacA + bid * strideD;

    // rpgr is the number of rounds a group should run
    // to cover all the rows
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    int i, it;

    // Registers/LDS:
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* tmp = reinterpret_cast<T*>(smem);
    T val;
    T* y;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;
        val = 0;

        it = (i < c) ? i : i - c;
        y = (i < c) ? y1 : y2;

        // read groups results
        if(i < m)
        {
            for(int j = idc; j < n; j += totalthsc)
                val += dac[i + j * ldd];
        }
        tmp[tidr + tidc * threadsr] = val;
        __syncthreads();

        // reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                val += tmp[tidr + (tidc + r) * threadsr];
                tmp[tidr + tidc * threadsr] = val;
            }
            __syncthreads();
        }

        // write results
        if(tidc == 0 && i < m)
            y[it] = val;
    }
}

/***** Kernels to update column of A *****/
/*****************************************/
template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_upper_updateA_kernel(const rocblas_int mm,
                                                 const rocblas_int k,
                                                 const rocblas_int c,
                                                 U AA,
                                                 const rocblas_int shiftA,
                                                 const rocblas_int lda,
                                                 const rocblas_stride strideA,
                                                 T* WA,
                                                 const rocblas_int shiftW,
                                                 const rocblas_int ldw,
                                                 const rocblas_stride strideW)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(WA, bid, shiftW, strideW);

    /* ------------------------
    formulate gemv problem:

        components:
            cw = c - mm + k
            y = A(0:c, c)
            A1 = A(0:c, c+1:mm-1)
            A2 = W(0:c, cw+1:k-1)
            x1 = W(c, cw+1:k-1)
            x2 = A(c, c+1:mm-1)

        operation:
            y = y - A1 * x1' - A2 * x2'
    ------------------------ */
    int n = mm - c - 1;
    int m = c + 1;
    int cw = c - mm + k;
    T* y = A + idx2D(0, c, lda);
    T* A1 = A + idx2D(0, c + 1, lda);
    int lda1 = lda;
    T* A2 = W + idx2D(0, cw + 1, ldw);
    int lda2 = ldw;
    T* x1 = W + idx2D(c, cw + 1, ldw);
    int incx1 = ldw;
    T* x2 = A + idx2D(c, c + 1, lda);
    int incx2 = lda;

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac;
    T sx1, sx2;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;

        // read y
        ac = (idc == 0 && i < m) ? y[i] : 0;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx1 = (j < n) ? conj(x1[j * incx1]) : 0;
            sx2 = (j < n) ? conj(x2[j * incx2]) : 0;

            // operation for all rows
            if(i < m && j < n)
                ac -= A1[i + j * lda1] * sx1 + A2[i + j * lda2] * sx2;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write results
        if(tidc == 0 && i < m)
            y[i] = ac;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_lower_updateA_kernel(const rocblas_int mm,
                                                 const rocblas_int c,
                                                 U AA,
                                                 const rocblas_int shiftA,
                                                 const rocblas_int lda,
                                                 const rocblas_stride strideA,
                                                 T* pWA,
                                                 const rocblas_int shiftW,
                                                 const rocblas_int ldw,
                                                 const rocblas_stride strideW)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(pWA, bid, shiftW, strideW);

    /* ------------------------
    formulate gemv problem:

        components:
            y = A(c:mm-1, c)
            A1 = A(c:mm-1, 0:c-1)
            A2 = W(c:mm-1, 0:c-1)
            x1 = W(c, 0:c-1)
            x2 = A(c, 0:c-1)

        operation:
            y = y - A1 * x1' - A2 * x2'
    ------------------------ */
    int m = mm - c;
    int n = c;
    T* y = A + idx2D(c, c, lda);
    T* A1 = A + idx2D(c, 0, lda);
    int lda1 = lda;
    T* A2 = W + idx2D(c, 0, ldw);
    int lda2 = ldw;
    T* x1 = W + idx2D(c, 0, ldw);
    int incx1 = ldw;
    T* x2 = A + idx2D(c, 0, lda);
    int incx2 = lda;

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac;
    T sx1, sx2;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;

        // read y
        ac = (idc == 0 && i < m) ? y[i] : 0;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx1 = (j < n) ? conj(x1[j * incx1]) : 0;
            sx2 = (j < n) ? conj(x2[j * incx2]) : 0;

            // operation for all rows
            if(i < m && j < n)
                ac -= A1[i + j * lda1] * sx1 + A2[i + j * lda2] * sx2;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write results
        if(tidc == 0 && i < m)
            y[i] = ac;
    }
}

/***** Kernels to compute column of W *****/
/******************************************/
template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_upper_computeW_symv_kernel(const rocblas_int mm,
                                                       const rocblas_int k,
                                                       const rocblas_int c,
                                                       U AA,
                                                       const rocblas_int shiftA,
                                                       const rocblas_int lda,
                                                       const rocblas_stride strideA,
                                                       T* pWA,
                                                       const rocblas_int shiftW,
                                                       const rocblas_int ldw,
                                                       const rocblas_stride strideW,
                                                       T* dacA,
                                                       const rocblas_int ldd,
                                                       const rocblas_stride strideD)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(pWA, bid, shiftW, strideW);
    T* dac = dacA + bid * strideD;

    /* -----------------------------
    formulate gemv problem:

        components:
            cw = c - mm + k
            A1 = A(0:c-1, :)
            A2 = W(0:c-1, cw+1:k-1)
            x = A(0:c-1, c)
            dac = temporary buffer

        operation:
                  [   A1(:, 0:c-1)   ]
                  [        0         ]
            dac = [ A1(:, c+1:mm-1)' ] * x
                  [        A2'       ]

        Notes:
            1. Here A1(:, 0:c-1) is symmetric (data referenced only above diagonal)
            2. dac is further reduced by reduce_kernel; results stored in
                  [ y1 ]
                  [ 0  ]
                  [ y2 ] <- reduce(dac)
                  [ y3 ]
              where
                  [ y1 ]
                  [ 0  ] = W(:, cw)
                  [ y2 ]
                    y3   = work (temp buffer)
    ------------------------------ */
    int n = c;
    int cc = mm - c - 1;
    int m = mm + cc;
    int cw = c - mm + k;
    T* A1 = A;
    T* A2 = W + idx2D(0, cw + 1, ldw);
    int lda1 = lda;
    int lda2 = ldw;
    T* x = A + idx2D(0, c, lda);

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j, it, it2;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac, sx;
    T const* a;
    int ld;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;
        ac = 0;

        it = (i < mm) ? i : i - mm;
        a = (i < mm) ? A1 : A2;
        ld = (i < mm) ? lda1 : lda2;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx = (j < n) ? x[j] : 0;

            // operation for all rows
            if(i < m && j < n && it != c)
                ac += (i > c) ? conj(a[j + it * ld]) * sx
                    : (j < i) ? conj(a[j + i * ld]) * sx
                              : a[i + j * ld] * sx;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        if(tidc == 0 && i < m)
            dac[i + bidc * ldd] = ac;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_lower_computeW_symv_kernel(const rocblas_int mm,
                                                       const rocblas_int c,
                                                       U AA,
                                                       const rocblas_int shiftA,
                                                       const rocblas_int lda,
                                                       const rocblas_stride strideA,
                                                       T* pWA,
                                                       const rocblas_int shiftW,
                                                       const rocblas_int ldw,
                                                       const rocblas_stride strideW,
                                                       T* dacA,
                                                       const rocblas_int ldd,
                                                       const rocblas_stride strideD)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(pWA, bid, shiftW, strideW);
    T* dac = dacA + bid * strideD;

    /* -----------------------------
    formulate gemv problem:

        components:
            A1 = W(c+1:mm-1, 0:c-1)
            A2 = A(c+1:mm-1, :)
            x = A(c+1,mm-1, c)
            dac = temporary buffer

        operation:
                  [       A1'       ]
            dac = [  A2(:, 0:c-1)'  ] * x
                  [        0        ]
                  [ A2(:, c+1:mm-1) ]

        Notes:
            1. Here A2(:, c+1:mm-1) is symmetric (data referenced only below diagonal)
            2. dac is further reduced by reduce_kernel; results stored in
                  [ y1 ]
                  [ y2 ] <- reduce(dac)
                  [ 0  ]
                  [ y3 ]
              where
                    y1   = work (temp buffer)
                  [ y2 ]
                  [ 0  ] = W(:, c)
                  [ y3 ]
    ------------------------------ */
    int n = mm - c - 1;
    int m = mm + c;
    T* A1 = W + idx2D(c + 1, 0, ldw);
    T* A2 = A + idx2D(c + 1, 0, lda);
    int lda1 = ldw;
    int lda2 = lda;
    T* x = A + idx2D(c + 1, c, lda);

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j, it, it2;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac, sx;
    T* a;
    int ld;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;
        ac = 0;

        it = (i < c) ? i : i - c;
        a = (i < c) ? A1 : A2;
        ld = (i < c) ? lda1 : lda2;
        it2 = it - c - 1;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx = (j < n) ? x[j] : 0;

            // operation for all rows
            if(i < m && j < n && it != c)
                ac += (it < c)  ? conj(a[j + it * ld]) * sx
                    : (j > it2) ? conj(a[j + (it2 + c + 1) * ld]) * sx
                                : a[it2 + (j + c + 1) * ld] * sx;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        if(tidc == 0 && i < m)
            dac[i + bidc * ldd] = ac;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_upper_computeW_gemv_kernel(const rocblas_int mm,
                                                       const rocblas_int k,
                                                       const rocblas_int c,
                                                       U AA,
                                                       const rocblas_int shiftA,
                                                       const rocblas_int lda,
                                                       const rocblas_stride strideA,
                                                       T* pWA,
                                                       const rocblas_int shiftW,
                                                       const rocblas_int ldw,
                                                       const rocblas_stride strideW,
                                                       T* dacA,
                                                       const rocblas_int ldd,
                                                       const rocblas_stride strideD)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(pWA, bid, shiftW, strideW);
    T* dac = dacA + bid * strideD;

    /* -----------------------------
    formulate gemv problem:

        components:
            cw = c - mm + k
            A1 = A(0:c-1, :)
            A2 = W(0:c-1, cw+1:k-1)
            x = A(0:c-1, c)
            dac = temporary buffer

        operation:
                  [   A1(:, 0:c-1)   ]
                  [        0         ]
            dac = [ A1(:, c+1:mm-1)' ] * x
                  [        A2'       ]

        Notes:
            1. Here A1(:, 0:c-1) is full/general matrix (data below and above diagonal)
            2. dac is further reduced by reduce_kernel; results stored in
                  [ y1 ]
                  [ 0  ]
                  [ y2 ] <- reduce(dac)
                  [ y3 ]
              where
                  [ y1 ]
                  [ 0  ] = W(:, cw)
                  [ y2 ]
                    y3   = work (temp buffer)
    ------------------------------ */
    int n = c;
    int cc = mm - c - 1;
    int m = mm + cc;
    int cw = c - mm + k;
    T* A1 = A;
    T* A2 = W + idx2D(0, cw + 1, ldw);
    int lda1 = lda;
    int lda2 = ldw;
    T* x = A + idx2D(0, c, lda);

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j, it, it2;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac, sx;
    T const* a;
    int ld;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;
        ac = 0;

        it = (i < mm) ? i : i - mm;
        a = (i < mm) ? A1 : A2;
        ld = (i < mm) ? lda1 : lda2;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx = (j < n) ? x[j] : 0;

            // operation for all rows
            if(i < m && j < n && it != c)
                ac += (i > c) ? conj(a[j + it * ld]) * sx : a[i + j * ld] * sx;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        if(tidc == 0 && i < m)
            dac[i + bidc * ldd] = ac;
    }
}

template <int NB_X, typename T, typename U>
ROCSOLVER_KERNEL void latrd_upper_computeW_gemvt_kernel(const int         blocks,
                                                        const rocblas_int mm,
                                                        const rocblas_int k,
                                                        const rocblas_int c,
                                                        U* pAA,
                                                        const rocblas_int shiftA,
                                                        const rocblas_int lda,
                                                        const rocblas_stride strideA,
                                                        T* pWA, // Householder panel
                                                        const rocblas_int shiftW,
                                                        const rocblas_int ldw,
                                                        const rocblas_stride strideW,
                                                        T* pYA,
                                                        const rocblas_int shiftY,
                                                        const rocblas_int ldy,
                                                        const rocblas_stride strideY,
                                                        T* pWorkA,
                                                        const rocblas_stride strideblk)
{
    rocblas_int bid = blockIdx.z;
    rocblas_int tx  = threadIdx.x;
    T* pA  = load_ptr_batch<T>(pAA, bid, shiftA, strideA);
    T* pW  = load_ptr_batch<T>(pWA, bid, shiftW, strideW);
    T* pY1 = load_ptr_batch<T>(pYA, bid, shiftY, strideY);
    T* pY2 = pWorkA + bid * strideblk;
    int n        = c;
    int cc       = mm - c - 1;
    int cw       = c - mm + k;
    T* pA1       = pA;
    T* pA2       = pW + idx2D(0, cw + 1, ldw);
    int lda1     = lda;
    int lda2     = ldw;
    T* pX        = pA + idx2D(0, c, lda);

    __shared__ T sdata[NB_X];

    // partial sums
    rocblas_int n_full = (n / NB_X) * NB_X;

    for(rocblas_int i = blockIdx.x; i < blocks; i += gridDim.x)
    {
        int it       = (i < mm) ? i    : i - mm;
        T* pAderived = (i < mm) ? pA1  : pA2;
        int ld       = (i < mm) ? lda1 : lda2;
        T* pYderived = (i < mm) ? pY1  : pY2;

        if(tx < n)
        {
            pAderived += tx;
        }

        pAderived += it * size_t(ld);
        T res = 0;


        if(it != c)
        {
            for(rocblas_int j = 0; j < n_full; j += NB_X)
            {
                res += conj(pAderived[j]) * pX[tx + j];
            }

            if(tx + n_full < n)
            {
                res += conj(pAderived[n_full]) * pX[tx + n_full];
            }

            // reduction of partial sums
            res += shift_left(res, 1);
            res += shift_left(res, 2);
            res += shift_left(res, 4);
            res += shift_left(res, 8);
            res += shift_left(res, 16);

            if(warpSize > 32)
            {
                res += shift_left(res, 32);
            }

            if(tx % warpSize == 0)
            {
                sdata[tx / warpSize] = res;
            }

            __syncthreads();
            if(tx == 0)
            {
                for(rocblas_int k = 1; k < NB_X / warpSize; k++)
                    res += sdata[k];
            }
        }

        auto outBuff  = __builtin_amdgcn_make_buffer_rsrc(pYderived, 0,0xffffffff,0x00020000);

        if(tx == 0)
        {

            if constexpr(sizeof(T) == 1)
                __builtin_amdgcn_raw_buffer_store_b8(res, outBuff, it, 0,0x0);
            else if constexpr(sizeof(T) == 2)
                __builtin_amdgcn_raw_buffer_store_b16(res, outBuff, it, 0,0x0);
            else if constexpr(sizeof(T) == 4)
                __builtin_amdgcn_raw_buffer_store_b32(res, outBuff, it, 0,0x0);
            else if constexpr(sizeof(T) == 8)
            {
                using uint32x2_t = uint32_t __attribute__((ext_vector_type(2)));
                uint32x2_t tmp;
                memcpy(&tmp, &res, sizeof(T));
                __builtin_amdgcn_raw_buffer_store_b64(tmp, outBuff, it, 0,0x0);
            }

        }
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_lower_computeW_gemv_kernel(const rocblas_int mm,
                                                       const rocblas_int c,
                                                       U AA,
                                                       const rocblas_int shiftA,
                                                       const rocblas_int lda,
                                                       const rocblas_stride strideA,
                                                       T* pWA,
                                                       const rocblas_int shiftW,
                                                       const rocblas_int ldw,
                                                       const rocblas_stride strideW,
                                                       T* dacA,
                                                       const rocblas_int ldd,
                                                       const rocblas_stride strideD)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(pWA, bid, shiftW, strideW);
    T* dac = dacA + bid * strideD;

    /* -----------------------------
    formulate gemv problem:

        components:
            A1 = W(c+1:mm-1, 0:c-1)
            A2 = A(c+1:mm-1, :)
            x = A(c+1,mm-1, c)
            dac = temporary buffer

        operation:
                  [       A1'       ]
            dac = [  A2(:, 0:c-1)'  ] * x
                  [        0        ]
                  [ A2(:, c+1:mm-1) ]

        Notes:
            1. Here A2(:, c+1:mm-1) is full/general matrix (data below and above diagonal)
            2. dac is further reduced by reduce_kernel; results stored in
                  [ y1 ]
                  [ y2 ] <- reduce(dac)
                  [ 0  ]
                  [ y3 ]
              where
                    y1   = work (temp buffer)
                  [ y2 ]
                  [ 0  ] = W(:, c)
                  [ y3 ]
    ------------------------------ */
    int n = mm - c - 1;
    int m = mm + c;
    T* A1 = W + idx2D(c + 1, 0, ldw);
    T* A2 = A + idx2D(c + 1, 0, lda);
    int lda1 = ldw;
    int lda2 = lda;
    T* x = A + idx2D(c + 1, c, lda);

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j, it, it2;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac, sx;
    T const* a;
    int ld;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;
        ac = 0;

        it = (i < c) ? i : i - c;
        a = (i < c) ? A1 : A2;
        ld = (i < c) ? lda1 : lda2;
        it2 = it - c - 1;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx = (j < n) ? x[j] : 0;

            // operation for all rows
            if(i < m && j < n && it != c)
                ac += (it < c) ? conj(a[j + it * ld]) * sx : a[it2 + (j + c + 1) * ld] * sx;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        if(tidc == 0 && i < m)
            dac[i + bidc * ldd] = ac;
    }
}

template <int NB_X, typename T, typename U>
ROCSOLVER_KERNEL void latrd_lower_computeW_gemvt_kernel(const rocblas_int mm,
                                                        const rocblas_int c,
                                                        U AA,
                                                        const rocblas_int shiftA,
                                                        const rocblas_int lda,
                                                        const rocblas_stride strideA,
                                                        T* pWA,
                                                        const rocblas_int shiftW,
                                                        const rocblas_int ldw,
                                                        const rocblas_stride strideW,
                                                        T* yA,
                                                        const rocblas_int shiftY,
                                                        const rocblas_int ldy,
                                                        const rocblas_stride strideY,
                                                        T* pWorkA,
                                                        const rocblas_stride strideblk)
{
    rocblas_int bid = blockIdx.z;
    rocblas_int tx = threadIdx.x;
    rocblas_int i = blockIdx.x;

    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(pWA, bid, shiftW, strideW);
    T* y1 = pWorkA + bid * strideblk;
    T* y2 = load_ptr_batch<T>(yA, bid, shiftY, strideY);

    int n = mm - c - 1;
    // int m = mm + c;
    T* A1 = W + idx2D(c + 1, 0, ldw);
    T* A2 = A + idx2D(c + 1, 0, lda);
    int lda1 = ldw;
    int lda2 = lda;
    T* x = A + idx2D(c + 1, c, lda);

    int it = (i < c) ? i : i - c;
    T* a = (i < c) ? A1 : A2;
    int ld = (i < c) ? lda1 : lda2;
    int it2 = it - c - 1;
    T* y = (i < c) ? y1 : y2;

    if(tx < n)
        a += tx;

    a += it * size_t(ld);

    T res = 0;

    __shared__ T sdata[NB_X];

    // partial sums
    rocblas_int n_full = (n / NB_X) * NB_X;

    if(it != c)
    {
        for(rocblas_int j = 0; j < n_full; j += NB_X)
            res += conj(a[j]) * x[tx + j];

        if(tx + n_full < n)
            res += conj(a[n_full]) * x[tx + n_full];

        // reduction of partial sums
        res += shift_left(res, 1);
        res += shift_left(res, 2);
        res += shift_left(res, 4);
        res += shift_left(res, 8);
        res += shift_left(res, 16);
        if(warpSize > 32)
            res += shift_left(res, 32);
        if(tx % warpSize == 0)
            sdata[tx / warpSize] = res;
        __syncthreads();
        if(tx == 0)
        {
            for(rocblas_int k = 1; k < NB_X / warpSize; k++)
                res += sdata[k];
        }
    }

    if(tx == 0)
    {
        y[it] = res;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_upper_computeW_kernel(const rocblas_int mm,
                                                  const rocblas_int k,
                                                  const rocblas_int c,
                                                  U AA,
                                                  const rocblas_int shiftA,
                                                  const rocblas_int lda,
                                                  const rocblas_stride strideA,
                                                  T* pWA,
                                                  const rocblas_int shiftW,
                                                  const rocblas_int ldw,
                                                  const rocblas_stride strideW,
                                                  T* dacA,
                                                  const rocblas_int ldd,
                                                  const rocblas_stride strideD)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(pWA, bid, shiftW, strideW);
    T* dac = dacA + bid * strideD;

    /* -----------------------------
    formulate gemv problem:

        components:
            cw = c - mm + k
            A1 = A(0:c-1, c+1:mm-1)
            A2 = W(0:c-1, cw+1:k-1)
            x = A(0:c-1, c)
            dac = temporary buffer

        operation:
                  [ A1' ]
            dac = [ A2' ] * x

        Notes:
            1. dac is further reduced by reduce_kernel; results stored in
                  [ y1 ]
                  [ y2 ] <- reduce(dac)
              where
                  y1 = W(c+1:mm-1, cw)
                  y2 = work (temp buffer)
    ------------------------------ */
    int n = c;
    int cc = mm - c - 1;
    int m = cc + cc;
    int cw = c - mm + k;
    T* A1 = A + idx2D(0, c + 1, lda);
    T* A2 = W + idx2D(0, cw + 1, ldw);
    int lda1 = lda;
    int lda2 = ldw;
    T* x = A + idx2D(0, c, lda);

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j, it;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac, sx;
    T* a;
    int ld;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;
        ac = 0;

        it = (i < cc) ? i : i - cc;
        a = (i < cc) ? A1 : A2;
        ld = (i < cc) ? lda1 : lda2;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx = (j < n) ? x[j] : 0;

            // operation for all rows
            if(i < m && j < n)
                ac += conj(a[j + it * ld]) * sx;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        if(tidc == 0 && i < m)
            dac[i + bidc * ldd] = ac;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_lower_computeW_kernel(const rocblas_int mm,
                                                  const rocblas_int c,
                                                  U AA,
                                                  const rocblas_int shiftA,
                                                  const rocblas_int lda,
                                                  const rocblas_stride strideA,
                                                  T* pWA,
                                                  const rocblas_int shiftW,
                                                  const rocblas_int ldw,
                                                  const rocblas_stride strideW,
                                                  T* dacA,
                                                  const rocblas_int ldd,
                                                  const rocblas_stride strideD)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(pWA, bid, shiftW, strideW);
    T* dac = dacA + bid * strideD;

    /* -----------------------------
    formulate gemv problem:

        components:
            A1 = W(c+1:mm-1, 0:c-1)
            A2 = A(c+1:mm-1, 0:c-1)
            x = A(c+1,mm-1, c)
            dac = temporary buffer

        operation:
                  [ A1' ]
            dac = [ A2' ] * x

        Notes:
            1. dac is further reduced by reduce_kernel; results stored in
                  [ y1 ]
                  [ y2 ] <- reduce(dac)
              where
                  y1 = work (temp buffer)
                  y2 = W(0:c-1, c)
    ------------------------------ */
    int n = mm - c - 1;
    int m = c + c;
    T* A1 = W + idx2D(c + 1, 0, ldw);
    T* A2 = A + idx2D(c + 1, 0, lda);
    int lda1 = ldw;
    int lda2 = lda;
    T* x = A + idx2D(c + 1, c, lda);

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j, it;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac, sx;
    T* a;
    int ld;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;
        ac = 0;

        it = (i < c) ? i : i - c;
        a = (i < c) ? A1 : A2;
        ld = (i < c) ? lda1 : lda2;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx = (j < n) ? x[j] : 0;

            // operation for all rows
            if(i < m && j < n)
                ac += conj(a[j + it * ld]) * sx;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        if(tidc == 0 && i < m)
            dac[i + bidc * ldd] = ac;
    }
}

/***** Kernels to update column of W *****/
/*****************************************/
template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_upper_updateW_kernel(const rocblas_int mm,
                                                 const rocblas_int k,
                                                 const rocblas_int c,
                                                 U AA,
                                                 const rocblas_int shiftA,
                                                 const rocblas_int lda,
                                                 const rocblas_stride strideA,
                                                 T* pWA,
                                                 const rocblas_int shiftW,
                                                 const rocblas_int ldw,
                                                 const rocblas_stride strideW,
                                                 T* pWorkA,
                                                 const rocblas_stride strideblk,
                                                 T* pTauA,
                                                 const rocblas_stride strideP)
{
    int bid       = hipBlockIdx_z;
    int bidr      = hipBlockIdx_x;
    int bidc      = hipBlockIdx_y;
    int tidr      = hipThreadIdx_x;
    int tidc      = hipThreadIdx_y;
    int threadsr  = hipBlockDim_x;
    int threadsc  = hipBlockDim_y;
    int groupsr   = hipGridDim_x;
    int groupsc   = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc       = bidc * threadsc + tidc;
    int idr       = bidr * threadsr + tidr;

    // select batch instance
    T* pA    = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* pW    = load_ptr_batch<T>(pWA, bid, shiftW, strideW);
    T* pWork = pWorkA + bid * strideblk;
    T* pTau  = pTauA + bid * strideP;

    /* ------------------------
    formulate gemv problem:

        components:
            cw = c - mm + k
            y = W(0:c-1, cw)
            A1 = A(0:c-1, c+1:mm-1)
            A2 = W(0:c-1, cw+1:k-1)
            x1 = work (temp buffer)
            x2 = W(c+1:mm-1, cw)
            t = tau(c-1)

        operation:
            y = t * (y - A1 * x1 - A2 * x2)
    ------------------------ */
    int n    = mm - c - 1;
    int m    = c;
    int cw   = c - mm + k;
    T* pY    = pW + idx2D(0, cw, ldw);
    T* pA1   = pA + idx2D(0, c + 1, lda);
    int lda1 = lda;
    T* pA2   = pW + idx2D(0, cw + 1, ldw);
    int lda2 = ldw;
    T* pX1   = pWork;
    T* pX2   = pW + idx2D(c + 1, cw, ldw);
    T* pT     = pTau + c - 1;
    const auto tauVal =  pT[0];
    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp     = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j;

    //define buffer types
    //TODO: bounds should be sized correctly instead of set to max
    auto Y_buffer  = __builtin_amdgcn_make_buffer_rsrc(pY, 0,0xffffffff,0x00020000);
    auto X1_buffer = __builtin_amdgcn_make_buffer_rsrc(pX1,0,0xffffffff,0x00020000);
    auto X2_buffer = __builtin_amdgcn_make_buffer_rsrc(pX2,0,0xffffffff,0x00020000);
    auto A1_buffer = __builtin_amdgcn_make_buffer_rsrc(pA1,0,0xffffffff,0x00020000);
    auto A2_buffer = __builtin_amdgcn_make_buffer_rsrc(pA2,0,0xffffffff,0x00020000);

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs_smem = reinterpret_cast<T*>(smem);
    T ac;
    T sx1, sx2;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;

        // read y
        const auto Y_ld_addr = (idc == 0 && i < m) ? i * sizeof(T) : 0xffffffff;
        if constexpr(sizeof(T) == 1)
        {
            uint8_t tmp1 =  __builtin_amdgcn_raw_buffer_load_b8(Y_buffer, Y_ld_addr,0,0);
            ac = *reinterpret_cast<T*>(&tmp1);
        }
        else if constexpr(sizeof(T) == 2)
        {
            uint16_t tmp1 =__builtin_amdgcn_raw_buffer_load_b16(Y_buffer, Y_ld_addr,0,0);
            ac = *reinterpret_cast<T*>(&tmp1);
        }
        else if constexpr(sizeof(T) == 4)
        {
            uint32_t tmp1 = __builtin_amdgcn_raw_buffer_load_b32(Y_buffer, Y_ld_addr,0,0);
            ac = *reinterpret_cast<T*>(&tmp1);
        }
        else if constexpr(sizeof(T) == 8)
        {
            const auto tmp =  __builtin_amdgcn_raw_buffer_load_b64(Y_buffer, Y_ld_addr,0,0);
            memcpy(&ac, &tmp, sizeof(T));
        }

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;

            const auto x_addr = (j < n) ? j * sizeof(T) : 0xffffffff;
            if constexpr(sizeof(T) == 1)
            {
                uint8_t tmp1 = __builtin_amdgcn_raw_buffer_load_b8(X1_buffer, x_addr, 0, 0);
                uint8_t tmp2 = __builtin_amdgcn_raw_buffer_load_b8(X2_buffer, x_addr, 0, 0);
                sx1 = *reinterpret_cast<T*>(&tmp1);
                sx2 = *reinterpret_cast<T*>(&tmp2);
            }
            else if constexpr(sizeof(T) == 2)
            {
                uint16_t tmp1 = __builtin_amdgcn_raw_buffer_load_b16(X1_buffer, x_addr, 0, 0);
                uint16_t tmp2 = __builtin_amdgcn_raw_buffer_load_b16(X2_buffer, x_addr, 0, 0);
                sx1 = *reinterpret_cast<T*>(&tmp1);
                sx2 = *reinterpret_cast<T*>(&tmp2);
            }
            else if constexpr(sizeof(T) == 4)
            {
                uint32_t tmp1 = __builtin_amdgcn_raw_buffer_load_b32(X1_buffer, x_addr, 0, 0);
                uint32_t tmp2 = __builtin_amdgcn_raw_buffer_load_b32(X2_buffer, x_addr, 0, 0);
                sx1 = *reinterpret_cast<T*>(&tmp1);
                sx2 = *reinterpret_cast<T*>(&tmp2);
            }
            else if constexpr(sizeof(T) == 8)
            {
                const auto tmp1 = __builtin_amdgcn_raw_buffer_load_b64(X1_buffer, x_addr, 0, 0);
                const auto tmp2 = __builtin_amdgcn_raw_buffer_load_b64(X2_buffer, x_addr, 0, 0);
                memcpy(&sx1, &tmp1, sizeof(T));
                memcpy(&sx2, &tmp2, sizeof(T));
            }

            const auto A1_addr = (i < m && j < n) ? (i + j * lda1) * sizeof(T) : 0xffffffff;
            const auto A2_addr = (i < m && j < n) ? (i + j * lda2) * sizeof(T) : 0xffffffff;
            T a1_val, a2_val = 0;
            if constexpr(sizeof(T) == 1)
            {
                uint8_t tmp1 = __builtin_amdgcn_raw_buffer_load_b8(A1_buffer, A1_addr, 0, 0);
                uint8_t tmp2 = __builtin_amdgcn_raw_buffer_load_b8(A2_buffer, A2_addr, 0, 0);
                a1_val = *reinterpret_cast<T*>(&tmp1);
                a2_val = *reinterpret_cast<T*>(&tmp2);
            }
            else if constexpr(sizeof(T) == 2)
            {
                uint16_t tmp1 = __builtin_amdgcn_raw_buffer_load_b16(A1_buffer, A1_addr, 0, 0);
                uint16_t tmp2 = __builtin_amdgcn_raw_buffer_load_b16(A2_buffer, A2_addr, 0, 0);
                a1_val = *reinterpret_cast<T*>(&tmp1);
                a2_val = *reinterpret_cast<T*>(&tmp2);
            }
            else if constexpr(sizeof(T) == 4)
            {
                uint32_t tmp1 = __builtin_amdgcn_raw_buffer_load_b32(A1_buffer, A1_addr, 0, 0);
                uint32_t tmp2 = __builtin_amdgcn_raw_buffer_load_b32(A2_buffer, A2_addr, 0, 0);
                a1_val = *reinterpret_cast<T*>(&tmp1);
                a2_val = *reinterpret_cast<T*>(&tmp2);
            }
            else if constexpr(sizeof(T) == 8)
            {
                const auto tmp1 = __builtin_amdgcn_raw_buffer_load_b64(A1_buffer, A1_addr, 0, 0);
                const auto tmp2 = __builtin_amdgcn_raw_buffer_load_b64(A2_buffer, A2_addr, 0, 0);
                memcpy(&a1_val, &tmp1, sizeof(T));
                memcpy(&a2_val, &tmp2, sizeof(T));
            }
            ac -= a1_val * sx1 + a2_val * sx2;
        }
        acs_smem[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs_smem[tidr + (tidc + r) * threadsr];
                acs_smem[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        const auto result = ac * tauVal;
        const auto Y_out_addr = (i < m) ? (i * sizeof(T)) : 0xffffffff;
        if constexpr(sizeof(T) == 1)
            __builtin_amdgcn_raw_buffer_store_b8(result, Y_buffer, Y_out_addr, 0,0x0);
        else if constexpr(sizeof(T) == 2)
            __builtin_amdgcn_raw_buffer_store_b16(result, Y_buffer, Y_out_addr, 0,0x0);
        else if constexpr(sizeof(T) == 4)
            __builtin_amdgcn_raw_buffer_store_b32(result, Y_buffer, Y_out_addr, 0,0x0);
        else if constexpr(sizeof(T) == 8)
        {
            using uint32x2_t = uint32_t __attribute__((ext_vector_type(2)));
            uint32x2_t tmp;
            memcpy(&tmp, &result, sizeof(T));
            __builtin_amdgcn_raw_buffer_store_b64(tmp, Y_buffer, Y_out_addr, 0,0x0);
        }
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_lower_updateW_kernel(const rocblas_int mm,
                                                 const rocblas_int c,
                                                 U AA,
                                                 const rocblas_int shiftA,
                                                 const rocblas_int lda,
                                                 const rocblas_stride strideA,
                                                 T* pWA,
                                                 const rocblas_int shiftW,
                                                 const rocblas_int ldw,
                                                 const rocblas_stride strideW,
                                                 T* pWorkA,
                                                 const rocblas_stride strideblk,
                                                 T* pTauA,
                                                 const rocblas_stride strideP)
{
    int bid  = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(pWA, bid, shiftW, strideW);
    T* work = pWorkA + bid * strideblk;
    T* tau = pTauA + bid * strideP;

    /* ------------------------
    formulate gemv problem:

        components:
            y = W(c+1:mm-1, c)
            A1 = A(c+1:mm-1, 0:c-1)
            A2 = W(c+1:mm-1, 0:c-1)
            x1 = work (temp buffer)
            x2 = W(0:c-1, c)
            t = tau(c)

        operation:
            y = t * (y - A1 * x1 - A2 * x2)
    ------------------------ */
    int m = mm - c - 1;
    int n = c;
    T* y = W + idx2D(c + 1, c, ldw);
    T* A1 = A + idx2D(c + 1, 0, lda);
    int lda1 = lda;
    T* A2 = W + idx2D(c + 1, 0, ldw);
    int lda2 = ldw;
    T* x1 = work;
    T* x2 = W + idx2D(0, c, ldw);
    T* t = tau + c;

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac;
    T sx1, sx2;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;

        // read y
        ac = (idc == 0 && i < m) ? y[i] : 0;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx1 = (j < n) ? x1[j] : 0;
            sx2 = (j < n) ? x2[j] : 0;

            // operation for all rows
            if(i < m && j < n)
                ac -= A1[i + j * lda1] * sx1 + A2[i + j * lda2] * sx2;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        if(tidc == 0 && i < m)
            y[i] = ac * t[0];
    }
}

/******************* Host functions for latrd aux of sytrd **********************/
/********************************************************************************/

// Method to determine configuration for update kernels depending on n and k
// TODO: fine tuning may be required
template <typename T>
void latrd_get_config_for_updates(const rocblas_int n,
                                  const rocblas_int k,
                                  rocblas_int* dr,
                                  rocblas_int* thr,
                                  rocblas_int* dc,
                                  rocblas_int* thc)
{
    if(n <= 256)
    {
        *thr = 8;
        *thc = 16;
    }
    else if(n <= 3584)
    {
        *thr = 16;
        *thc = 16;
    }
    else if(n <= 7168)
    {
        *thr = 32;
        *thc = 8;
    }
    else
    {
        *thr = 64;
        *thc = 8;
    }

    *dr = 4;
    *dc = 0;
}

template <bool BATCHED, typename T>
void rocsolver_latrd_forsytrd_getMemorySize(const rocblas_int n,
                                            const rocblas_int k,
                                            const rocblas_int batch_count,
                                            size_t* size_scalars,
                                            size_t* size_work,
                                            size_t* size_norms,
                                            size_t* size_workArr)
{
    // if quick return no workspace needed
    if(n == 0 || k == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_norms = 0;
        *size_workArr = 0;
        return;
    }

    size_t n1 = 0, n2 = 0, n3 = 0;
    size_t w1 = 0, w2 = 0, w3 = 0, w4 = 0;

    // size of scalars (constants) for rocblas calls
    *size_scalars = sizeof(T) * 3;

    // size of array of pointers (batched cases)
    if(BATCHED)
        *size_workArr = 2 * sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    // extra requirements for calling larfg
    rocsolver_larfg_getMemorySize<T>(n, batch_count, &w1, &n1);

    // extra requirements for calling symv/hemv
    rocblasCall_symv_hemv_mem<BATCHED, T>(n, batch_count, &w2);

    // extra requirements for calling dotp
    // TODO: replace with rocBLAS call
    constexpr int ROCBLAS_DOT_NB = 512;
    w3 = n > 2 ? (n - 2) / ROCBLAS_DOT_NB + 2 : 1;
    w3 *= sizeof(T) * batch_count;
    n2 = sizeof(T) * batch_count;

    // arrays for temporary values
    // TODO: smaller quotes could be considered if we know the latrd_mode and
    // the configuration of the computeW kernels in advance. For now, taking
    // worst case.
    w4 = sizeof(T) * k * batch_count;
    rocblas_int gr = (n - 1) / 4 + 1;
    n3 = sizeof(T) * (n + k) * gr * batch_count;

    *size_norms = std::max({n1, n2, n3});
    *size_work  = std::max({w1, w2, w3, w4});
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_latrd_forsytrd_template(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n, // matrix dim 1
                                                 const rocblas_int k, // matrix dim 2
                                                 U* A,
                                                 const rocblas_int shiftA,
                                                 const rocblas_int lda,
                                                 const rocblas_stride strideA,
                                                 S* pE,
                                                 const rocblas_stride strideE,
                                                 T* pTau,
                                                 const rocblas_stride strideP,
                                                 T* pW,
                                                 const rocblas_int shiftW,
                                                 const rocblas_int ldw,
                                                 const rocblas_stride strideW,
                                                 const rocblas_int batch_count,
                                                 T* pScalars,
                                                 T* pWork,
                                                 T* pNorms,
                                                 T** ppWorkArr)
{
    ROCSOLVER_ENTER("latrd_forsytrd", "uplo:", uplo, "n:", n, "k:", k, "shiftA:", shiftA,
                    "lda:", lda, "shiftW:", shiftW, "ldw:", ldw, "bc:", batch_count);
    roctxRangePush("rocsolver_latrd_forsytrd");

    // quick return
    if(n == 0 || k == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // configure updateA and updateW kernels:
    rocblas_int dr, dc;
    rocblas_int thr_updates, thc_updates;
    latrd_get_config_for_updates<T>(n, k, &dr, &thr_updates, &dc, &thc_updates);
    size_t lmemsize_updates = sizeof(T) * (thr_updates * thc_updates);
    rocblas_int grr_updates = (n * dr / 4 - 1) / thr_updates + 1;
    rocblas_int grc_updates = (k * dc / 4 - 1) / thc_updates + 1;

    rocblas_stride strideblk = k;

    if(uplo == rocblas_fill_lower)
    {
        // reduce the first k columns of A
        // main loop running forwards (for each column)
        for(rocblas_int j = 0; j < k; ++j)
        {
            // update column j of A with reflector computed in step j-1
            //--------------------------------------------------------------
            dim3 update_grid(grr_updates, grc_updates, batch_count);
            dim3 update_threads(thr_updates, thc_updates, 1);
            ROCSOLVER_LAUNCH_KERNEL(latrd_lower_updateA_kernel<T>,
                                    update_grid,
                                    update_threads, lmemsize_updates, stream, n,
                                    j, A, shiftA, lda, strideA, pW, shiftW, ldw, strideW);
            //--------------------------------------------------------------

            // reduce column j of A with new reflector, then copy off-diagonal element
            // to E(j) and set off-diagonal to 1
            //----------------------------------------------------------
            rocsolver_larfg_template(handle, n - j - 1, A, shiftA + idx2D(j + 1, j, lda), pE, j,
                                     strideE, A, shiftA + idx2D(std::min(j + 2, n - 1), j, lda), 1,
                                     strideA, (pTau + j), strideP, batch_count, pWork, pNorms);
            //--------------------------------------------------------------

            // compute column j of W
            //--------------------------------------------------------------
            static constexpr int NB = 256;
            dim3 gemvt_grid(n + j, 1, batch_count);
            dim3 gemvt_threads(NB);
            ROCSOLVER_LAUNCH_KERNEL((latrd_lower_computeW_gemvt_kernel<NB, T>), gemvt_grid,
                                    gemvt_threads, 0, stream, n, j, A, shiftA, lda, strideA, pW,
                                    shiftW, ldw, strideW, pW, shiftW + idx2D(0, j, ldw), ldw,
                                    strideW, pWork, strideblk);

            // update column j of W
            //--------------------------------------------------------------
            ROCSOLVER_LAUNCH_KERNEL(
                latrd_lower_updateW_kernel<T>, update_grid,
                update_threads, lmemsize_updates, stream, n, j, A, shiftA, lda,
                strideA, pW, shiftW, ldw, strideW, pWork, strideblk, pTau, strideP);

            ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<1024, T>), dim3(1, 1, batch_count),
                                    dim3(1024, 1, 1), 0, stream, n - 1 - j, A,
                                    shiftA + idx2D(j + 1, j, lda), strideA, pW,
                                    shiftW + idx2D(j + 1, j, ldw), strideW, pTau + j, strideP);
            //--------------------------------------------------------------
        }
    }

    else
    {
        // reduce the last k columns of A
        // main loop running forwards (for each column)
        rocblas_int jw;

        // get the hardware compute unit count, take the min of blocksX and 4x cu count to get good occupancy without oversubscribing
        int cuCount;
        hipDeviceProp_t deviceProp;
        hipError_t err = hipGetDeviceProperties(&deviceProp, 0);
        assert(err == hipSuccess);
        cuCount = deviceProp.multiProcessorCount;

        for(rocblas_int j = n - 1; j >= n - k; --j)
        {
            jw = j - n + k;
            dim3 update_grid(grr_updates, grc_updates, batch_count);
            dim3 update_threads(thr_updates, thc_updates, 1);
            // update column j of A with reflector computed in step j-1
            //----------------------------------------------------------
            ROCSOLVER_LAUNCH_KERNEL(latrd_upper_updateA_kernel<T>,
                                    update_grid,
                                    update_threads,
                                    lmemsize_updates,
                                    stream,
                                    n, k, j, A, shiftA,
                                    lda, strideA, pW,
                                    shiftW, ldw, strideW);
            //-------------------------------------------------------------

            // // reduce column j of A with new reflector, then copy off-diagonal element
            // // to E(j) and set off-diagonal to 1
            // //-------------------------------------------------------------
            rocsolver_larfg_template(handle,
                                     j, A, shiftA + idx2D(j - 1, j, lda), pE, j - 1, strideE,
                                     A, shiftA + idx2D(0, j, lda), 1, strideA, (pTau + j - 1),
                                     strideP, batch_count, pWork, pNorms);

            //--------------------------------------------------------------
            // compute column j of W
            //--------------------------------------------------------------
            static constexpr int NB = 256;
            const int blocksX = n + n - j - 1;
            int maxBlocks = cuCount * 8; // max occupancy without oversubscribing
            int gridX = std::min(blocksX, maxBlocks);

            dim3 gemvt_grid(gridX, 1, batch_count);
            dim3 gemvt_threads(NB);
            ROCSOLVER_LAUNCH_KERNEL((latrd_upper_computeW_gemvt_kernel<NB, T>),
                                    gemvt_grid,
                                    gemvt_threads,
                                    0,
                                    stream,
                                    blocksX,
                                    n, k, j, A, shiftA, lda, strideA, pW,
                                    shiftW, ldw, strideW, pW, shiftW + idx2D(0, jw, ldw), ldw,
                                    strideW, pWork, strideblk);

            // update column j of W
            //--------------------------------------------------------------
            ROCSOLVER_LAUNCH_KERNEL(latrd_upper_updateW_kernel<T>,
                                    update_grid,
                                    update_threads,
                                    lmemsize_updates,
                                    stream,
                                    n, k, j, A, shiftA, lda, strideA,
                                    pW, shiftW, ldw, strideW, pWork,
                                    strideblk, pTau, strideP);

            ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<1024, T>),
                                    dim3(1, 1, batch_count),
                                    dim3(1024, 1, 1),
                                    0,
                                    stream,
                                    j, A, shiftA + idx2D(0, j, lda), strideA,
                                    pW, shiftW + idx2D(0, jw, ldw), strideW,
                                    pTau + j - 1, strideP);
        }
    }

    roctxRangePop();
    return rocblas_status_success;
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
auto rocsolver_latrd_forsytrd_getWorkItems(rocblas_handle handle,
                                           const rocblas_fill uplo,
                                           const rocblas_int n,
                                           const rocblas_int k,
                                           U A,
                                           const rocblas_int shiftA,
                                           const rocblas_int lda,
                                           const rocblas_stride strideA,
                                           S* E,
                                           const rocblas_stride strideE,
                                           T* tau,
                                           const rocblas_stride strideP,
                                           T* W,
                                           const rocblas_int shiftW,
                                           const rocblas_int ldw,
                                           const rocblas_stride strideW,
                                           const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_workArr;
    // extra requirements for calling LARFG
    size_t size_work, size_norms;
    rocsolver_latrd_forsytrd_getMemorySize<false, T>(n, k, batch_count, &size_scalars, &size_work,
                                                     &size_norms, &size_workArr);

    auto work_items = create_work_item({"latrd_scalars", size_scalars})
        + create_work_item({"latrd_workArr", size_workArr})
        + create_work_item({"latrd_work", size_work}) + create_work_item({"latrd_norms", size_norms});

    return work_items;
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_latrd_forsytrd_template(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 U A,
                                                 const rocblas_int shiftA,
                                                 const rocblas_int lda,
                                                 const rocblas_stride strideA,
                                                 S* E,
                                                 const rocblas_stride strideE,
                                                 T* tau,
                                                 const rocblas_stride strideP,
                                                 T* W,
                                                 const rocblas_int shiftW,
                                                 const rocblas_int ldw,
                                                 const rocblas_stride strideW,
                                                 const rocblas_int batch_count,
                                                 rocsolver_device_workspace_ptr_t dwptr)
{
    ROCSOLVER_INIT_DEVICE_WORKSPACE(dwptr,
                                    rocsolver_latrd_forsytrd_getWorkItems(
                                        handle, uplo, n, k, A, shiftA, lda, strideA, E, strideE,
                                        tau, strideP, W, shiftW, ldw, strideW, batch_count));

    T* scalars = (T*)dwptr->work("latrd_scalars");
    T* work = (T*)dwptr->work("latrd_work");
    T* norms = (T*)dwptr->work("latrd_norms");
    T** workArr = (T**)dwptr->work("latrd_workArr");

    if(dwptr->size("latrd_scalars") > 0)
        init_scalars(handle, (T*)scalars);

    if(print_debug_messages_latrd_forsytrd)
    {
        std::cout << "Using latrd_forsytrd entry point." << std::endl;
    }

    // execution
    return rocsolver_latrd_forsytrd_template<T>(
        handle, uplo, n, k, A, shiftA, lda, strideA, E, strideE, tau, strideP, W, shiftW, ldw,
        strideW, batch_count, (T*)scalars, (T*)work, (T*)norms, (T**)workArr);
}

ROCSOLVER_END_NAMESPACE
