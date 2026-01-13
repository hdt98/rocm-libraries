/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxiliary/rocauxiliary_larfb.hpp"
#include "auxiliary/rocauxiliary_larft.hpp"
#include "rocblas.hpp"
#include "roclapack_gelq2.hpp"
#include "rocsolver/rocsolver.h"

#include "mem_utils.hpp"

ROCSOLVER_BEGIN_NAMESPACE

#ifndef USE_ORIGINAL
#define USE_ORIGINAL false
#endif

template <bool BATCHED, typename T>
void rocsolver_gelqf_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms_trfact,
                                   size_t* size_diag_tmptr,
                                   size_t* size_workArr)
{
    // if quick return no workspace needed
    *size_scalars = 0;
    *size_work_workArr = 0;
    *size_Abyx_norms_trfact = 0;
    *size_diag_tmptr = 0;
    *size_workArr = 0;
    if(m == 0 || n == 0 || batch_count == 0)
    {
        return;
    }

    if(m <= GExQF_GExQ2_SWITCHSIZE || n <= GExQF_GExQ2_SWITCHSIZE)
    {
        // requirements for a single GELQ2 call
        rocsolver_gelq2_getMemorySize<BATCHED, T>(m, n, batch_count, size_scalars, size_work_workArr,
                                                  size_Abyx_norms_trfact, size_diag_tmptr);
        *size_workArr = 0;
    }
    else
    {
        size_t w1, w2, unused, s1, s2;
        rocblas_int jb = GExQF_BLOCKSIZE;

        // size to store the temporary triangular factor
        *size_Abyx_norms_trfact = sizeof(T) * jb * jb * batch_count;

        // requirements for calling GELQ2 with sub blocks
        rocsolver_gelq2_getMemorySize<BATCHED, T>(jb, n, batch_count, size_scalars, &w1, &s2, &s1);
        *size_Abyx_norms_trfact = std::max(s2, *size_Abyx_norms_trfact);

        // requirements for calling LARFT
        rocsolver_larft_getMemorySize<BATCHED, T>(n, jb, batch_count, &unused, &w2, size_workArr);

        // requirements for calling LARFB
        rocsolver_larfb_getMemorySize<BATCHED, T>(rocblas_side_right, m - jb, n, jb, batch_count,
                                                  &s2, &unused);

        *size_work_workArr = std::max(w1, w2);
        *size_diag_tmptr = std::max(s1, s2);

        // size of workArr is double to accommodate
        // LARFB's TRMM calls in the batched case
        if(BATCHED)
            *size_workArr *= 2;
    }
    adjust_for_alignment(size_scalars);
    adjust_for_alignment(size_work_workArr);
    adjust_for_alignment(size_Abyx_norms_trfact);
    adjust_for_alignment(size_diag_tmptr);
    adjust_for_alignment(size_workArr);
}

template <bool BATCHED, typename T>
void rocsolver_gelqf_getMemorySize_alt(const rocblas_int m,
                                       const rocblas_int n,
                                       const rocblas_int batch_count,

                                       size_t* p_size_gelqf)
{
    size_t size_scalars = 0;
    size_t size_work_workArr = 0;
    size_t size_Abyx_norms_trfact = 0;
    size_t size_diag_tmptr = 0;
    size_t size_workArr = 0;

    rocsolver_gelqf_getMemorySize<BATCHED, T>(m, n, batch_count,

                                              &size_scalars, &size_work_workArr,
                                              &size_Abyx_norms_trfact, &size_diag_tmptr,
                                              &size_workArr);

    size_t size_gelqf = 0;

    size_gelqf += size_scalars;
    size_gelqf += size_work_workArr;
    size_gelqf += size_Abyx_norms_trfact;
    size_gelqf += size_diag_tmptr;
    size_gelqf += size_workArr;

    *p_size_gelqf = size_gelqf;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_gelqf_template(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        void* work_workArr,
                                        T* Abyx_norms_trfact,
                                        T* diag_tmptr,
                                        T** workArr)
{
    ROCSOLVER_ENTER("gelqf", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // if the matrix is small, use the unblocked (BLAS-levelII) variant of the
    // algorithm
    if(m <= GExQF_GExQ2_SWITCHSIZE || n <= GExQF_GExQ2_SWITCHSIZE)
        return rocsolver_gelq2_template<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, strideP,
                                           batch_count, scalars, work_workArr, Abyx_norms_trfact,
                                           diag_tmptr);

    rocblas_int dim = std::min(m, n); // total number of pivots
    rocblas_int jb, j = 0;

    rocblas_int nb = GExQF_BLOCKSIZE;
    rocblas_int ldw = GExQF_BLOCKSIZE;
    rocblas_stride strideW = rocblas_stride(ldw) * ldw;

    while(j < dim - GExQF_GExQ2_SWITCHSIZE)
    {
        // Factor diagonal and subdiagonal blocks
        jb = std::min(dim - j, nb); // number of rows in the block
        rocsolver_gelq2_template<T>(handle, jb, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                    (ipiv + j), strideP, batch_count, scalars, work_workArr,
                                    Abyx_norms_trfact, diag_tmptr);

        // apply transformation to the rest of the matrix
        if(j + jb < m)
        {
            // compute block reflector
            rocsolver_larft_template<T>(handle, rocblas_forward_direction, rocblas_row_wise, n - j,
                                        jb, A, shiftA + idx2D(j, j, lda), lda, strideA, (ipiv + j),
                                        strideP, Abyx_norms_trfact, ldw, strideW, batch_count,
                                        scalars, (T*)work_workArr, workArr);

            // apply the block reflector
            rocsolver_larfb_template<BATCHED, STRIDED, T>(
                handle, rocblas_side_right, rocblas_operation_none, rocblas_forward_direction,
                rocblas_row_wise, m - j - jb, n - j, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
                Abyx_norms_trfact, 0, ldw, strideW, A, shiftA + idx2D(j + jb, j, lda), lda, strideA,
                batch_count, diag_tmptr, workArr);
        }
        j += nb;
    }

    // factor last block
    if(j < dim)
        rocsolver_gelq2_template<T>(handle, m - j, n - j, A, shiftA + idx2D(j, j, lda), lda,
                                    strideA, (ipiv + j), strideP, batch_count, scalars,
                                    work_workArr, Abyx_norms_trfact, diag_tmptr);

    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_gelqf_template_alt(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            U A,
                                            const rocblas_int shiftA,
                                            const rocblas_int lda,
                                            const rocblas_stride strideA,
                                            T* ipiv,
                                            const rocblas_stride strideP,
                                            const rocblas_int batch_count,

                                            void* const work,
                                            size_t const size_work)
{
    std::byte* const pwork = (std::byte*)work;
    std::byte* pfree = pwork;

    size_t size_scalars = 0;
    size_t size_work_workArr = 0;
    size_t size_Abyx_norms_trfact = 0;
    size_t size_diag_tmptr = 0;
    size_t size_workArr = 0;

    rocsolver_gelqf_getMemorySize<BATCHED, T>(m, n, batch_count,

                                              &size_scalars, &size_work_workArr,
                                              &size_Abyx_norms_trfact, &size_diag_tmptr,
                                              &size_workArr);

    T* const scalars = (T*)pfree;
    pfree += size_scalars;
    if(size_scalars > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    void* const work_workArr = (void*)pfree;
    pfree += size_work_workArr;

    T* const Abyx_norms_trfact = (T*)pfree;
    pfree += size_Abyx_norms_trfact;

    T* const diag_tmptr = (T*)pfree;
    pfree += size_diag_tmptr;

    T** const workArr = (T**)pfree;
    pfree += size_workArr;

    MEM_CHECK(pfree);

    rocblas_status const istat = rocsolver_gelqf_template<BATCHED, STRIDED, T, U>(
        handle, m, n, A, shiftA, lda, strideA, ipiv, strideP, batch_count,

        scalars, work_workArr, Abyx_norms_trfact, diag_tmptr, workArr);

    return (istat);
}

ROCSOLVER_END_NAMESPACE
