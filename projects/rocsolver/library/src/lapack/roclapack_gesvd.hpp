/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     April 2012
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxiliary/rocauxiliary_bdsqr.hpp"
#include "auxiliary/rocauxiliary_orgbr_ungbr.hpp"
#include "auxiliary/rocauxiliary_ormbr_unmbr.hpp"
#include "rocblas.hpp"
#include "roclapack_gebrd.hpp"
#include "roclapack_gelqf.hpp"
#include "roclapack_geqrf.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

#include "rocsolver_mem_utils.hpp"

ROCSOLVER_BEGIN_NAMESPACE

#ifndef USE_ORIGINAL
#define USE_ORIGINAL false
#endif

/** wrapper to xxGQR/xxGLQ_TEMPLATE **/
template <bool BATCHED, bool STRIDED, typename T, typename U>
void local_orgqrlq_ungqrlq_template(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    const rocblas_int k,
                                    U A,
                                    const rocblas_int shiftA,
                                    const rocblas_int lda,
                                    const rocblas_stride strideA,
                                    T* ipiv,
                                    const rocblas_stride strideP,
                                    const rocblas_int batch_count,
                                    T* scalars,
                                    T* work,
                                    T* Abyx_tmptr,
                                    T* trfact,
                                    T** workArr,
                                    const bool row)
{
    if(row)
        rocsolver_orgqr_ungqr_template<BATCHED, STRIDED>(handle, m, n, k, A, shiftA, lda, strideA,
                                                         ipiv, strideP, batch_count, scalars, work,
                                                         Abyx_tmptr, trfact, workArr);

    else
        rocsolver_orglq_unglq_template<BATCHED, STRIDED>(handle, m, n, k, A, shiftA, lda, strideA,
                                                         ipiv, strideP, batch_count, scalars, work,
                                                         Abyx_tmptr, trfact, workArr);
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status local_orgqrlq_ungqrlq_template_alt(rocblas_handle handle,
                                                  const rocblas_int m,
                                                  const rocblas_int n,
                                                  const rocblas_int k,
                                                  U A,
                                                  const rocblas_int shiftA,
                                                  const rocblas_int lda,
                                                  const rocblas_stride strideA,
                                                  T* ipiv,
                                                  const rocblas_stride strideP,
                                                  const rocblas_int batch_count,
                                                  const bool row,

                                                  void* const work,
                                                  size_t const size_work)
{
    if(row)
    {
        return (rocsolver_orgqr_ungqr_template_alt<BATCHED, STRIDED>(
            handle, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, batch_count, work, size_work));
    }
    else
    {
        return (rocsolver_orglq_unglq_template_alt<BATCHED, STRIDED>(
            handle, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, batch_count,

            work, size_work));
    }
    return (rocblas_status_success);
}
/** wrapper to GEQRF/GELQF_TEMPLATE **/
template <bool BATCHED, bool STRIDED, typename T, typename U>
void local_geqrlq_template(rocblas_handle handle,
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
                           T** workArr,
                           const bool row)
{
    if(row)
        rocsolver_geqrf_template<BATCHED, STRIDED>(handle, m, n, A, shiftA, lda, strideA, ipiv,
                                                   strideP, batch_count, scalars, work_workArr,
                                                   Abyx_norms_trfact, diag_tmptr, workArr);
    else
        rocsolver_gelqf_template<BATCHED, STRIDED>(handle, m, n, A, shiftA, lda, strideA, ipiv,
                                                   strideP, batch_count, scalars, work_workArr,
                                                   Abyx_norms_trfact, diag_tmptr, workArr);
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status local_geqrlq_template_alt(rocblas_handle handle,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         U A,
                                         const rocblas_int shiftA,
                                         const rocblas_int lda,
                                         const rocblas_stride strideA,
                                         T* ipiv,
                                         const rocblas_stride strideP,
                                         const rocblas_int batch_count,
                                         const bool row,

                                         void* const work,
                                         size_t const size_work)
{
    if(row)
    {
        return (rocsolver_geqrf_template_alt<BATCHED, STRIDED>(handle, m, n, A, shiftA, lda,
                                                               strideA, ipiv, strideP, batch_count,

                                                               work, size_work));
    }
    else
    {
        return (rocsolver_gelqf_template_alt<BATCHED, STRIDED>(handle, m, n, A, shiftA, lda,
                                                               strideA, ipiv, strideP, batch_count,

                                                               work, size_work));
    }
    return (rocblas_status_success);
}

/** Argument checking **/
template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_argCheck(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        W A,
                                        const rocblas_int lda,
                                        TT* S,
                                        T* U,
                                        const rocblas_int ldu,
                                        T* V,
                                        const rocblas_int ldv,
                                        TT* E,
                                        rocblas_int* info,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if((left_svect != rocblas_svect_all && left_svect != rocblas_svect_singular
        && left_svect != rocblas_svect_overwrite && left_svect != rocblas_svect_none)
       || (right_svect != rocblas_svect_all && right_svect != rocblas_svect_singular
           && right_svect != rocblas_svect_overwrite && right_svect != rocblas_svect_none)
       || (left_svect == rocblas_svect_overwrite && right_svect == rocblas_svect_overwrite))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || m < 0 || lda < m || ldu < 1 || ldv < 1 || batch_count < 0)
        return rocblas_status_invalid_size;
    if((left_svect == rocblas_svect_all || left_svect == rocblas_svect_singular) && ldu < m)
        return rocblas_status_invalid_size;
    if((right_svect == rocblas_svect_all && ldv < n)
       || (right_svect == rocblas_svect_singular && ldv < std::min(m, n)))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && m && !A) || (std::min(m, n) > 1 && !E) || (std::min(m, n) && !S)
       || (batch_count && !info))
        return rocblas_status_invalid_pointer;
    if((left_svect == rocblas_svect_all && m && !U)
       || (left_svect == rocblas_svect_singular && std::min(m, n) && !U))
        return rocblas_status_invalid_pointer;
    if((right_svect == rocblas_svect_all || right_svect == rocblas_svect_singular) && n && !V)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename S>
void rocsolver_gesvd_getMemorySize(const rocblas_svect left_svect,
                                   const rocblas_svect right_svect,
                                   const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   const rocblas_workmode fast_alg,

                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms_tmptr_cmplt,
                                   size_t* size_Abyx_norms_trfact_X,
                                   size_t* size_diag_tmptr_Y,
                                   size_t* size_tau_splits,
                                   size_t* size_tempArrayT,
                                   size_t* size_tempArrayC,
                                   size_t* size_workArr)
{
    // if quick return, set workspace to zero
    *size_scalars = 0;
    *size_work_workArr = 0;
    *size_Abyx_norms_tmptr_cmplt = 0;
    *size_Abyx_norms_trfact_X = 0;
    *size_diag_tmptr_Y = 0;
    *size_tau_splits = 0;
    *size_tempArrayT = 0;
    *size_tempArrayC = 0;
    *size_workArr = 0;
    if(n == 0 || m == 0 || batch_count == 0)
    {
        return;
    }

    size_t w[6] = {0, 0, 0, 0, 0, 0};
    size_t a[6] = {0, 0, 0, 0, 0, 0};
    size_t x[6] = {0, 0, 0, 0, 0, 0};
    size_t y[3] = {0, 0, 0};
    size_t unused;

    // booleans used to determine the path that the execution will follow:
    const bool row = (m >= n);
    const bool leftvS = (left_svect == rocblas_svect_singular);
    const bool leftvO = (left_svect == rocblas_svect_overwrite);
    const bool leftvA = (left_svect == rocblas_svect_all);
    const bool leftvN = (left_svect == rocblas_svect_none);
    const bool rightvS = (right_svect == rocblas_svect_singular);
    const bool rightvO = (right_svect == rocblas_svect_overwrite);
    const bool rightvA = (right_svect == rocblas_svect_all);
    const bool rightvN = (right_svect == rocblas_svect_none);
    //const bool leadvS = row ? leftvS : rightvS;
    const bool leadvO = row ? leftvO : rightvO;
    const bool leadvA = row ? leftvA : rightvA;
    const bool leadvN = row ? leftvN : rightvN;
    //const bool othervS = !row ? leftvS : rightvS;
    const bool othervO = !row ? leftvO : rightvO;
    //const bool othervA = !row ? leftvA : rightvA;
    const bool othervN = !row ? leftvN : rightvN;
    const bool thinSVD = (m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m);
    const bool fast_thinSVD = (thinSVD && fast_alg == rocblas_outofplace);

    // auxiliary sizes and variables
    const rocblas_int k = std::min(m, n);
    const rocblas_int kk = std::max(m, n);
    const rocblas_int nu = leftvN ? 0 : ((fast_thinSVD || (thinSVD && leadvN)) ? k : m);
    const rocblas_int nv = rightvN ? 0 : ((fast_thinSVD || (thinSVD && leadvN)) ? k : n);
    const rocblas_storev storev_lead = row ? rocblas_column_wise : rocblas_row_wise;
    const rocblas_storev storev_other = row ? rocblas_row_wise : rocblas_column_wise;
    const rocblas_side side = row ? rocblas_side_right : rocblas_side_left;
    rocblas_int mn;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = 2 * sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    // size of arrays to store temporary copies
    *size_tempArrayT
        = (fast_thinSVD || (thinSVD && leadvO && othervN)) ? sizeof(T) * k * k * batch_count : 0;
    *size_tempArrayC
        = (fast_thinSVD && (othervN || othervO || leadvO)) ? sizeof(T) * m * n * batch_count : 0;

    // workspace required for the bidiagonalization
    if(thinSVD)
        rocsolver_gebrd_getMemorySize<BATCHED, T>(k, k, batch_count, size_scalars, &w[0], &a[0],
                                                  &x[0], &y[0]);
    else
        rocsolver_gebrd_getMemorySize<BATCHED, T>(m, n, batch_count, size_scalars, &w[0], &a[0],
                                                  &x[0], &y[0]);

    // workspace required for the SVD of the bidiagonal form
    rocsolver_bdsqr_getMemorySize<S>(k, nv, nu, 0, batch_count, size_tau_splits, &w[1], &a[1]);

    // size of array tau to store householder scalars on intermediate
    // orthonormal/unitary matrices
    *size_tau_splits = std::max(*size_tau_splits, 2 * sizeof(T) * std::min(m, n) * batch_count);

    // extra requirements for QR/LQ factorization
    if(thinSVD)
    {
        if(row)
            rocsolver_geqrf_getMemorySize<BATCHED, T>(m, n, batch_count, &unused, &w[2], &x[1],
                                                      &y[1], &unused);
        else
            rocsolver_gelqf_getMemorySize<BATCHED, T>(m, n, batch_count, &unused, &w[2], &x[1],
                                                      &y[1], &unused);
    }

    // extra requirements for orthonormal/unitary matrix generation
    // ormbr
    if(thinSVD && !fast_thinSVD && !leadvN)
        rocsolver_ormbr_unmbr_getMemorySize<BATCHED, T>(storev_lead, side, m, n, k, batch_count,
                                                        &unused, &a[2], &y[2], &x[2], &unused);
    // orgbr
    if(thinSVD)
    {
        if(!othervN)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(storev_other, k, k, k, batch_count,
                                                            &unused, &w[3], &a[3], &x[3], &unused);

        if(fast_thinSVD && !leadvN)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(storev_lead, k, k, k, batch_count,
                                                            &unused, &w[4], &a[4], &x[4], &unused);
    }
    else
    {
        mn = (row && leftvS) ? n : m;
        if(leftvS || leftvA)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(
                rocblas_column_wise, m, mn, n, batch_count, &unused, &w[3], &a[3], &x[3], &unused);
        else if(leftvO)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(
                rocblas_column_wise, m, k, n, batch_count, &unused, &w[3], &a[3], &x[3], &unused);

        mn = (!row && rightvS) ? m : n;
        if(rightvS || rightvA)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(rocblas_row_wise, mn, n, m, batch_count,
                                                            &unused, &w[4], &a[4], &x[4], &unused);
        else if(rightvO)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(rocblas_row_wise, k, n, m, batch_count,
                                                            &unused, &w[4], &a[4], &x[4], &unused);
    }
    // orgqr/orglq
    if(thinSVD && !leadvN)
    {
        if(leadvA)
        {
            if(row)
                rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(kk, kk, k, batch_count, &unused,
                                                                &w[5], &a[5], &x[5], &unused);
            else
                rocsolver_orglq_unglq_getMemorySize<BATCHED, T>(kk, kk, k, batch_count, &unused,
                                                                &w[5], &a[5], &x[5], &unused);
        }
        else
        {
            if(row)
                rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(m, n, k, batch_count, &unused,
                                                                &w[5], &a[5], &x[5], &unused);
            else
                rocsolver_orglq_unglq_getMemorySize<BATCHED, T>(m, n, k, batch_count, &unused,
                                                                &w[5], &a[5], &x[5], &unused);
        }
    }

    // get max sizes
    *size_work_workArr = *std::max_element(std::begin(w), std::end(w));
    *size_Abyx_norms_tmptr_cmplt = *std::max_element(std::begin(a), std::end(a));
    *size_Abyx_norms_trfact_X = *std::max_element(std::begin(x), std::end(x));
    *size_diag_tmptr_Y = *std::max_element(std::begin(y), std::end(y));

    adjust_for_alignment(size_scalars);
    adjust_for_alignment(size_work_workArr);
    adjust_for_alignment(size_Abyx_norms_tmptr_cmplt);
    adjust_for_alignment(size_Abyx_norms_trfact_X);
    adjust_for_alignment(size_diag_tmptr_Y);
    adjust_for_alignment(size_tau_splits);
    adjust_for_alignment(size_tempArrayT);
    adjust_for_alignment(size_tempArrayC);
    adjust_for_alignment(size_workArr);
}

template <bool BATCHED, typename T, typename S>
void rocsolver_gesvd_getMemorySize_alt(const rocblas_svect left_svect,
                                       const rocblas_svect right_svect,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       const rocblas_int batch_count,
                                       const rocblas_workmode fast_alg,

                                       size_t* p_size_gesvd)
{
    size_t size_gesvd = 0;
    *p_size_gesvd = size_gesvd;

    // if quick return, set workspace to zero
    size_t size_scalars = 0;
    size_t size_work_workArr = 0;
    size_t size_Abyx_norms_tmptr_cmplt = 0;
    size_t size_Abyx_norms_trfact_X = 0;
    size_t size_diag_tmptr_Y = 0;

    size_t size_tau_splits = 0;
    size_t size_tempArrayT = 0;
    size_t size_tempArrayC = 0;
    size_t size_workArr = 0;

    //       ---------------------------------------------
    // TODO: try to figure out how arrays tempArrayT[], tempArrayC[],
    //       tau_splits[], Abyx_norms_tmptr_trfact_X[],  diag_tmptr_Y[]
    //       are reused
    //       ---------------------------------------------

    {
        rocsolver_gesvd_getMemorySize<BATCHED, T, S>(
            left_svect, right_svect, m, n, batch_count, fast_alg,

            &size_scalars, &size_work_workArr, &size_Abyx_norms_tmptr_cmplt,
            &size_Abyx_norms_trfact_X, &size_diag_tmptr_Y, &size_tau_splits, &size_tempArrayT,
            &size_tempArrayC, &size_workArr);
    }

    if(n == 0 || m == 0 || batch_count == 0)
    {
        return;
    }

#if(0)
    size_t w[6] = {0, 0, 0, 0, 0, 0};
    size_t a[6] = {0, 0, 0, 0, 0, 0};
    size_t x[6] = {0, 0, 0, 0, 0, 0};
    size_t y[3] = {0, 0, 0};
    size_t unused;
#endif

    // booleans used to determine the path that the execution will follow:
    const bool row = (m >= n);
    const bool leftvS = (left_svect == rocblas_svect_singular);
    const bool leftvO = (left_svect == rocblas_svect_overwrite);
    const bool leftvA = (left_svect == rocblas_svect_all);
    const bool leftvN = (left_svect == rocblas_svect_none);
    const bool rightvS = (right_svect == rocblas_svect_singular);
    const bool rightvO = (right_svect == rocblas_svect_overwrite);
    const bool rightvA = (right_svect == rocblas_svect_all);
    const bool rightvN = (right_svect == rocblas_svect_none);
    //const bool leadvS = row ? leftvS : rightvS;
    const bool leadvO = row ? leftvO : rightvO;
    const bool leadvA = row ? leftvA : rightvA;
    const bool leadvN = row ? leftvN : rightvN;
    //const bool othervS = !row ? leftvS : rightvS;
    const bool othervO = !row ? leftvO : rightvO;
    //const bool othervA = !row ? leftvA : rightvA;
    const bool othervN = !row ? leftvN : rightvN;
    const bool thinSVD = (m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m);
    const bool fast_thinSVD = (thinSVD && fast_alg == rocblas_outofplace);

    // auxiliary sizes and variables
    const rocblas_int k = std::min(m, n);
    const rocblas_int kk = std::max(m, n);
    const rocblas_int nu = leftvN ? 0 : ((fast_thinSVD || (thinSVD && leadvN)) ? k : m);
    const rocblas_int nv = rightvN ? 0 : ((fast_thinSVD || (thinSVD && leadvN)) ? k : n);
    const rocblas_storev storev_lead = row ? rocblas_column_wise : rocblas_row_wise;
    const rocblas_storev storev_other = row ? rocblas_row_wise : rocblas_column_wise;
    const rocblas_side side = row ? rocblas_side_right : rocblas_side_left;
    rocblas_int mn;

    // size of array of pointers to workspace
    if(BATCHED)
        size_workArr = 2 * sizeof(T*) * batch_count;
    else
        size_workArr = 0;

    // size of arrays to store temporary copies
    size_tempArrayT
        = (fast_thinSVD || (thinSVD && leadvO && othervN)) ? sizeof(T) * k * k * batch_count : 0;
    size_tempArrayC
        = (fast_thinSVD && (othervN || othervO || leadvO)) ? sizeof(T) * m * n * batch_count : 0;

    // workspace required for the bidiagonalization
    size_t size_gebrd = 0;

    if(thinSVD)
    {
        size_t size_work_temp = 0;

        // --------------------------------------------------------------------
        // assume X, Y are not scratch arrays but are arguments to be passed in
        // --------------------------------------------------------------------
        rocsolver_gebrd_getMemorySize_alt<BATCHED, T>(k, k, batch_count, &size_work_temp);

        size_gebrd = std::max(size_gebrd, size_work_temp);
    }
    else
    {
        size_t size_work_temp = 0;
        rocsolver_gebrd_getMemorySize_alt<BATCHED, T>(m, n, batch_count, &size_work_temp);

        size_gebrd = std::max(size_gebrd, size_work_temp);
    }

    // workspace required for the SVD of the bidiagonal form

    size_t size_bdsqr = 0;

    {
        size_t size_work_temp = 0;
        rocsolver_bdsqr_getMemorySize_alt<S>(k, nv, nu, 0, batch_count, &size_work_temp);
        size_bdsqr = std::max(size_bdsqr, size_work_temp);
    }

    // size of array tau to store householder scalars on intermediate
    // orthonormal/unitary matrices
    size_tau_splits = std::max(size_tau_splits, 2 * sizeof(T) * std::min(m, n) * batch_count);

    // extra requirements for QR/LQ factorization
    size_t size_geqrf = 0;
    size_t size_gelqf = 0;

    if(thinSVD)
    {
        if(row)
        {
            bool constexpr STRIDED = true;

            size_t size_work_temp = 0;
            rocsolver_geqrf_getMemorySize_alt<BATCHED, STRIDED, T>(m, n, batch_count,
                                                                   &size_work_temp);

            size_geqrf = std::max(size_geqrf, size_work_temp);
        }
        else
        {
            size_t size_work_temp = 0;
            rocsolver_gelqf_getMemorySize_alt<BATCHED, T>(m, n, batch_count, &size_work_temp);

            size_gelqf = std::max(size_gelqf, size_work_temp);
        }
    }

    // extra requirements for orthonormal/unitary matrix generation
    // ormbr
    size_t size_ormbr = 0;

    if(thinSVD && !fast_thinSVD && !leadvN)
    {
        size_t size_work_temp = 0;
        rocsolver_ormbr_unmbr_getMemorySize_alt<BATCHED, T>(storev_lead, side, m, n, k, batch_count,
                                                            &size_work_temp);
        size_ormbr = std::max(size_ormbr, size_work_temp);
    }
    // orgbr
    size_t size_orgbr = 0;

    if(thinSVD)
    {
        if(!othervN)
        {
            size_t size_work_temp = 0;
            rocsolver_orgbr_ungbr_getMemorySize_alt<BATCHED, T>(storev_other, k, k, k, batch_count,
                                                                &size_work_temp);
            size_orgbr = std::max(size_orgbr, size_work_temp);
        }

        if(fast_thinSVD && !leadvN)
        {
            size_t size_work_temp = 0;
            rocsolver_orgbr_ungbr_getMemorySize_alt<BATCHED, T>(storev_lead, k, k, k, batch_count,
                                                                &size_work_temp);
            size_orgbr = std::max(size_orgbr, size_work_temp);
        }
    }
    else
    {
        mn = (row && leftvS) ? n : m;
        if(leftvS || leftvA)
        {
            size_t size_work_temp = 0;
            rocsolver_orgbr_ungbr_getMemorySize_alt<BATCHED, T>(rocblas_column_wise, m, mn, n,
                                                                batch_count, &size_work_temp);

            size_orgbr = std::max(size_orgbr, size_work_temp);
        }
        else if(leftvO)
        {
            size_t size_work_temp = 0;
            rocsolver_orgbr_ungbr_getMemorySize_alt<BATCHED, T>(rocblas_column_wise, m, k, n,
                                                                batch_count, &size_work_temp);

            size_orgbr = std::max(size_orgbr, size_work_temp);
        }

        mn = (!row && rightvS) ? m : n;
        if(rightvS || rightvA)
        {
            size_t size_work_temp = 0;
            rocsolver_orgbr_ungbr_getMemorySize_alt<BATCHED, T>(rocblas_row_wise, mn, n, m,
                                                                batch_count, &size_work_temp);
            size_orgbr = std::max(size_orgbr, size_work_temp);
        }
        else if(rightvO)
        {
            size_t size_work_temp = 0;
            rocsolver_orgbr_ungbr_getMemorySize_alt<BATCHED, T>(rocblas_row_wise, k, n, m,
                                                                batch_count, &size_work_temp);
            size_orgbr = std::max(size_orgbr, size_work_temp);
        }
    }
    // orgqr/orglq
    size_t size_orgqr = 0;
    size_t size_orglq = 0;

    if(thinSVD && !leadvN)
    {
        if(leadvA)
        {
            if(row)
            {
                size_t size_work_temp = 0;
                rocsolver_orgqr_ungqr_getMemorySize_alt<BATCHED, T>(kk, kk, k, batch_count,
                                                                    &size_work_temp);
                size_orgqr = std::max(size_orgqr, size_work_temp);
            }
            else
            {
                size_t size_work_temp = 0;
                rocsolver_orglq_unglq_getMemorySize_alt<BATCHED, T>(kk, kk, k, batch_count,
                                                                    &size_work_temp);
                size_orglq = std::max(size_orglq, size_work_temp);
            }
        }
        else
        {
            if(row)
            {
                size_t size_work_temp = 0;
                rocsolver_orgqr_ungqr_getMemorySize_alt<BATCHED, T>(m, n, k, batch_count,
                                                                    &size_work_temp);

                size_orgqr = std::max(size_orgqr, size_work_temp);
            }
            else
            {
                size_t size_work_temp = 0;
                rocsolver_orglq_unglq_getMemorySize_alt<BATCHED, T>(m, n, k, batch_count,
                                                                    &size_work_temp);
                size_orglq = std::max(size_orglq, size_work_temp);
            }
        }
    }

    {
        // TODO: revisit  legacy shared array

        adjust_for_alignment(size_tempArrayT);
        adjust_for_alignment(size_tempArrayC);

        adjust_for_alignment(size_tau_splits);
        adjust_for_alignment(size_Abyx_norms_trfact_X);
        adjust_for_alignment(size_diag_tmptr_Y);

        size_gesvd += size_tempArrayT;
        size_gesvd += size_tempArrayC;

        size_gesvd += size_tau_splits;

        // ----------------------------------------
        // note: no need for temporary arrays
        // Abyx_norms_trfact_X[] and diag_tmptr_Y[]
        // ----------------------------------------
        // size_gesvd += size_diag_tmptr_Y;
        // size_gesvd += size_Abyx_norms_trfact_X;
    }

    {
        // for GEMM

        adjust_for_alignment(size_workArr);
        size_gesvd += size_workArr;
    }

    {
        // TODO: check whether these are still needed
        adjust_for_alignment(size_work_workArr);
        adjust_for_alignment(size_Abyx_norms_tmptr_cmplt);

        size_gesvd += size_work_workArr;
        size_gesvd += size_Abyx_norms_tmptr_cmplt;
    }

    {
        adjust_for_alignment(size_gebrd);
        adjust_for_alignment(size_bdsqr);
        adjust_for_alignment(size_geqrf);
        adjust_for_alignment(size_gelqf);
        adjust_for_alignment(size_ormbr);
        adjust_for_alignment(size_orgbr);
        adjust_for_alignment(size_orgqr);
        adjust_for_alignment(size_orglq);

        // clang-format off
        size_gesvd += std::max( size_gebrd, 
                      std::max( size_bdsqr,
                      std::max( size_geqrf,
                      std::max( size_gelqf,
                      std::max( size_ormbr,
                      std::max( size_orgbr,
                      std::max( size_orgqr,
                      std::max( size_orgqr,
			        size_orglq ) ) ) ) ) ) ) );

        // clang-format on
    }

    adjust_for_alignment(size_gesvd);
    *p_size_gesvd = size_gesvd;
}

template <bool BATCHED, typename T, typename S>
void rocsolver_gesvd_getMemorySize_alt_org(const rocblas_svect left_svect,
                                           const rocblas_svect right_svect,
                                           const rocblas_int m,
                                           const rocblas_int n,
                                           const rocblas_int batch_count,
                                           const rocblas_workmode fast_alg,

                                           size_t* p_size_gesvd)
{
    size_t size_scalars = 0;
    size_t size_work_workArr = 0;
    size_t size_Abyx_norms_tmptr_cmplt = 0;
    size_t size_Abyx_norms_trfact_X = 0;

    size_t size_diag_tmptr_Y = 0;
    size_t size_tau_splits = 0;
    size_t size_tempArrayT = 0;
    size_t size_tempArrayC = 0;
    size_t size_workArr = 0;

    rocsolver_gesvd_getMemorySize<BATCHED, T, S>(
        left_svect, right_svect, m, n, batch_count, fast_alg,

        &size_scalars, &size_work_workArr, &size_Abyx_norms_tmptr_cmplt, &size_Abyx_norms_trfact_X,

        &size_diag_tmptr_Y, &size_tau_splits, &size_tempArrayT, &size_tempArrayC, &size_workArr);

    adjust_for_alignment(size_scalars);
    adjust_for_alignment(size_work_workArr);
    adjust_for_alignment(size_Abyx_norms_tmptr_cmplt);
    adjust_for_alignment(size_Abyx_norms_trfact_X);

    adjust_for_alignment(size_diag_tmptr_Y);
    adjust_for_alignment(size_tau_splits);
    adjust_for_alignment(size_tempArrayT);
    adjust_for_alignment(size_tempArrayC);
    adjust_for_alignment(size_workArr);

    size_t size_gesvd = 0;

    size_gesvd += size_scalars;
    size_gesvd += size_work_workArr;
    size_gesvd += size_Abyx_norms_tmptr_cmplt;
    size_gesvd += size_Abyx_norms_trfact_X;

    size_gesvd += size_diag_tmptr_Y;
    size_gesvd += size_tau_splits;
    size_gesvd += size_tempArrayT;
    size_gesvd += size_tempArrayC;
    size_gesvd += size_workArr;

    adjust_for_alignment(size_gesvd);

    *p_size_gesvd = size_gesvd;
}

template <bool BATCHED, bool STRIDED, typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_template(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        W A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        TT* S,
                                        const rocblas_stride strideS,
                                        T* U,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        T* V,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        TT* E,
                                        const rocblas_stride strideE,
                                        const rocblas_workmode fast_alg,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        void* work_workArr,
                                        T* Abyx_norms_tmptr_cmplt,
                                        T* Abyx_norms_trfact_X,
                                        T* diag_tmptr_Y,
                                        T* tau_splits,
                                        T* tempArrayT,
                                        T* tempArrayC,
                                        T** workArr)
{
    ROCSOLVER_ENTER("gesvd", "leftsv:", left_svect, "rightsv:", right_svect, "m:", m, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "ldu:", ldu, "ldv:", ldv, "mode:", fast_alg,
                    "bc:", batch_count);

    constexpr bool COMPLEX = rocblas_is_complex<T>;

    // quick return
    if(n == 0 || m == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // constants to use when calling rocablas functions
    T one = 1;
    T zero = 0;

    // booleans used to determine the path that the execution will follow:
    const bool row = (m >= n);
    const bool leftvS = (left_svect == rocblas_svect_singular);
    const bool leftvO = (left_svect == rocblas_svect_overwrite);
    const bool leftvA = (left_svect == rocblas_svect_all);
    const bool leftvN = (left_svect == rocblas_svect_none);
    const bool rightvS = (right_svect == rocblas_svect_singular);
    const bool rightvO = (right_svect == rocblas_svect_overwrite);
    const bool rightvA = (right_svect == rocblas_svect_all);
    const bool rightvN = (right_svect == rocblas_svect_none);
    const bool leadvS = row ? leftvS : rightvS;
    const bool leadvO = row ? leftvO : rightvO;
    const bool leadvA = row ? leftvA : rightvA;
    const bool leadvN = row ? leftvN : rightvN;
    const bool othervS = !row ? leftvS : rightvS;
    const bool othervO = !row ? leftvO : rightvO;
    const bool othervA = !row ? leftvA : rightvA;
    const bool othervN = !row ? leftvN : rightvN;
    const bool thinSVD = (m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m);
    const bool fast_thinSVD = (thinSVD && fast_alg == rocblas_outofplace);

    // auxiliary sizes and variables
    const rocblas_int k = std::min(m, n);
    const rocblas_int kk = std::max(m, n);
    const rocblas_int shiftX = 0;
    const rocblas_int shiftY = 0;
    const rocblas_int shiftUV = 0;
    const rocblas_int shiftT = 0;
    const rocblas_int shiftC = 0;
    const rocblas_int shiftU = 0;
    const rocblas_int shiftV = 0;
    const rocblas_int ldx = thinSVD ? k : m;
    const rocblas_int ldy = thinSVD ? k : n;
    const rocblas_stride strideX = ldx * GEBRD_BLOCKSIZE;
    const rocblas_stride strideY = ldy * GEBRD_BLOCKSIZE;
    T* bufferT = tempArrayT;
    rocblas_int ldt = k;
    rocblas_stride strideT = k * k;
    T* bufferC = tempArrayC;
    rocblas_int ldc = m;
    rocblas_stride strideC = m * n;

    T* UV;
    rocblas_int lduv, mn, nu, nv;
    rocblas_int offset_other, offset_lead;
    rocblas_storev storev_other, storev_lead;
    rocblas_stride strideUV;
    rocblas_fill uplo;
    rocblas_side side;
    rocblas_operation trans;

    nu = leftvN ? 0 : m;
    nv = rightvN ? 0 : n;
    if(row)
    {
        UV = U;
        lduv = ldu;
        strideUV = strideU;
        uplo = rocblas_fill_upper;
        storev_other = rocblas_row_wise;
        storev_lead = rocblas_column_wise;
        offset_other = k * batch_count;
        offset_lead = 0;
        if(othervS || othervA)
        {
            bufferC = V;
            ldc = ldv;
            strideC = strideV;
            if(!fast_thinSVD)
            {
                bufferT = V;
                ldt = ldv;
                strideT = strideV;
            }
        }
        side = rocblas_side_right;
        trans = rocblas_operation_none;
    }
    else
    {
        UV = V;
        lduv = ldv;
        strideUV = strideV;
        uplo = rocblas_fill_lower;
        storev_other = rocblas_column_wise;
        storev_lead = rocblas_row_wise;
        offset_other = 0;
        offset_lead = k * batch_count;
        if(othervS || othervA)
        {
            bufferC = U;
            ldc = ldu;
            strideC = strideU;
            if(!fast_thinSVD)
            {
                bufferT = U;
                ldt = ldu;
                strideT = strideU;
            }
        }
        side = rocblas_side_left;
        trans = COMPLEX ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;
    }

    // common block sizes and number of threads for internal kernels
    constexpr rocblas_int thread_count = 32;
    const rocblas_int blocks_m = (m - 1) / thread_count + 1;
    const rocblas_int blocks_n = (n - 1) / thread_count + 1;
    const rocblas_int blocks_k = (k - 1) / thread_count + 1;

    /** A thin SVD could be computed for matrices with sufficiently more rows than
        columns (or columns than rows) by starting with a QR factorization (or LQ
        factorization) and working with the triangular factor afterwards. When
        computing a thin SVD, a fast algorithm could be executed by doing some
        computations out-of-place. **/

    if(thinSVD)
    /*******************************************/
    /********** CASE: CHOOSE THIN-SVD **********/
    /*******************************************/
    {
        if(leadvN)
        /***** SUB-CASE: USE THIN-SVD WITH NO LEAD-DIMENSION VECTORS *****/
        /*****************************************************************/
        {
            nu = leftvN ? 0 : k;
            nv = rightvN ? 0 : k;

            //*** STAGE 1: Row (or column) compression ***//
            local_geqrlq_template<BATCHED, STRIDED>(
                handle, m, n, A, shiftA, lda, strideA, tau_splits, k, batch_count, scalars,
                work_workArr, Abyx_norms_trfact_X, diag_tmptr_Y, workArr, row);

            //*** STAGE 2: generate orthonormal/unitary matrix from row/column compression ***//
            // N/A

            //*** STAGE 3: Bidiagonalization ***//
            // clean triangular factor
            ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                                    dim3(thread_count, thread_count, 1), 0, stream, k, k, A, shiftA,
                                    lda, strideA, uplo);

            rocsolver_gebrd_template<BATCHED, STRIDED>(
                handle, k, k, A, shiftA, lda, strideA, S, strideS, E, strideE, tau_splits, k,
                (tau_splits + k * batch_count), k, Abyx_norms_trfact_X, shiftX, ldx, strideX,
                diag_tmptr_Y, shiftY, ldy, strideY, batch_count, scalars, work_workArr,
                Abyx_norms_tmptr_cmplt);

            //*** STAGE 4: generate orthonormal/unitary matrices from bidiagonalization ***//
            if(!othervN)
                rocsolver_orgbr_ungbr_template<BATCHED, STRIDED>(
                    handle, storev_other, k, k, k, A, shiftA, lda, strideA,
                    (tau_splits + offset_other), k, batch_count, scalars, (T*)work_workArr,
                    Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X, workArr);

            //*** STAGE 5: Compute singular values and vectors from the bidiagonal form ***//
            if(row)
                rocsolver_bdsqr_template<T>(handle, rocblas_fill_upper, k, nv, nu, 0, S, strideS, E,
                                            strideE, A, shiftA, lda, strideA, U, shiftU, ldu,
                                            strideU, (W) nullptr, 0, 1, 1, info, batch_count,
                                            (rocblas_int*)tau_splits, (TT*)work_workArr,
                                            (rocblas_int*)Abyx_norms_tmptr_cmplt);
            else
                rocsolver_bdsqr_template<T>(handle, rocblas_fill_upper, k, nv, nu, 0, S, strideS, E,
                                            strideE, V, shiftV, ldv, strideV, A, shiftA, lda,
                                            strideA, (W) nullptr, 0, 1, 1, info, batch_count,
                                            (rocblas_int*)tau_splits, (TT*)work_workArr,
                                            (rocblas_int*)Abyx_norms_tmptr_cmplt);

            //*** STAGE 6: update vectors with orthonormal/unitary matrices ***//
            if(othervS || othervA)
            {
                mn = row ? n : m;
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, mn, mn, A,
                                        shiftA, lda, strideA, bufferC, shiftC, ldc, strideC);
            }
        }

        else if(fast_thinSVD)
        /***** SUB-CASE: USE FAST (OUT-OF-PLACE) THIN-SVD ALGORITHM *****/
        /****************************************************************/
        {
            nu = leftvN ? 0 : k;
            nv = rightvN ? 0 : k;

            //*** STAGE 1: Row (or column) compression ***//
            local_geqrlq_template<BATCHED, STRIDED>(
                handle, m, n, A, shiftA, lda, strideA, tau_splits, k, batch_count, scalars,
                work_workArr, Abyx_norms_trfact_X, diag_tmptr_Y, workArr, row);

            if(leadvA)
                // copy factorization to U or V when needed
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, m, n, A,
                                        shiftA, lda, strideA, UV, shiftUV, lduv, strideUV);

            // copy the triangular part to be used in the bidiagonalization
            ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                    dim3(thread_count, thread_count, 1), 0, stream, k, k, A, shiftA,
                                    lda, strideA, bufferT, shiftT, ldt, strideT, no_mask{}, uplo);

            //*** STAGE 2: generate orthonormal/unitary matrix from row/column compression ***//
            if(leadvA)
                local_orgqrlq_ungqrlq_template<false, STRIDED>(
                    handle, kk, kk, k, UV, shiftUV, lduv, strideUV, tau_splits, k, batch_count,
                    scalars, (T*)work_workArr, Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X, workArr,
                    row);
            else
                local_orgqrlq_ungqrlq_template<BATCHED, STRIDED>(
                    handle, m, n, k, A, shiftA, lda, strideA, tau_splits, k, batch_count, scalars,
                    (T*)work_workArr, Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X, workArr, row);

            //*** STAGE 3: Bidiagonalization ***//
            // clean triangular factor
            ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                                    dim3(thread_count, thread_count, 1), 0, stream, k, k, bufferT,
                                    shiftT, ldt, strideT, uplo);

            rocsolver_gebrd_template<false, STRIDED>(
                handle, k, k, bufferT, shiftT, ldt, strideT, S, strideS, E, strideE, tau_splits, k,
                (tau_splits + k * batch_count), k, Abyx_norms_trfact_X, shiftX, ldx, strideX,
                diag_tmptr_Y, shiftY, ldy, strideY, batch_count, scalars, work_workArr,
                Abyx_norms_tmptr_cmplt);

            if(!othervN)
                // copy results to generate non-lead vectors if required
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, k, k,
                                        bufferT, shiftT, ldt, strideT, bufferC, shiftC, ldc, strideC);

            //*** STAGE 4: generate orthonormal/unitary matrices from bidiagonalization ***//
            // for lead-dimension vectors
            rocsolver_orgbr_ungbr_template<false, STRIDED>(
                handle, storev_lead, k, k, k, bufferT, shiftT, ldt, strideT,
                (tau_splits + offset_lead), k, batch_count, scalars, (T*)work_workArr,
                Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X, workArr);

            // for the other-side vectors
            if(!othervN)
                rocsolver_orgbr_ungbr_template<false, STRIDED>(
                    handle, storev_other, k, k, k, bufferC, shiftC, ldc, strideC,
                    (tau_splits + offset_other), k, batch_count, scalars, (T*)work_workArr,
                    Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X, workArr);

            //*** STAGE 5: Compute singular values and vectors from the bidiagonal form ***//
            if(row)
                rocsolver_bdsqr_template<T>(handle, rocblas_fill_upper, k, nv, nu, 0, S, strideS, E,
                                            strideE, bufferC, shiftC, ldc, strideC, bufferT, shiftT,
                                            ldt, strideT, (T*)nullptr, 0, 1, 1, info, batch_count,
                                            (rocblas_int*)tau_splits, (TT*)work_workArr,
                                            (rocblas_int*)Abyx_norms_tmptr_cmplt);
            else
                rocsolver_bdsqr_template<T>(handle, rocblas_fill_upper, k, nv, nu, 0, S, strideS, E,
                                            strideE, bufferT, shiftT, ldt, strideT, bufferC, shiftC,
                                            ldc, strideC, (T*)nullptr, 0, 1, 1, info, batch_count,
                                            (rocblas_int*)tau_splits, (TT*)work_workArr,
                                            (rocblas_int*)Abyx_norms_tmptr_cmplt);

            //*** STAGE 6: update vectors with orthonormal/unitary matrices ***//
            if(leadvO)
            {
                bufferC = tempArrayC;
                ldc = m;
                strideC = m * n;

                // update
                if(row)
                    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                                   &one, A, shiftA, lda, strideA, bufferT, shiftT, ldt, strideT,
                                   &zero, bufferC, shiftC, ldc, strideC, batch_count, workArr);
                else
                    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                                   &one, bufferT, shiftT, ldt, strideT, A, shiftA, lda, strideA,
                                   &zero, bufferC, shiftC, ldc, strideC, batch_count, workArr);

                // copy to overwrite A
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, m, n,
                                        bufferC, shiftC, ldc, strideC, A, shiftA, lda, strideA);
            }
            else if(leadvS)
            {
                // update
                if(row)
                    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                                   &one, A, shiftA, lda, strideA, bufferT, shiftT, ldt, strideT,
                                   &zero, UV, shiftUV, lduv, strideUV, batch_count, workArr);
                else
                    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                                   &one, bufferT, shiftT, ldt, strideT, A, shiftA, lda, strideA,
                                   &zero, UV, shiftUV, lduv, strideUV, batch_count, workArr);

                // overwrite A if required
                if(othervO)
                    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                            dim3(thread_count, thread_count, 1), 0, stream, k, k,
                                            bufferC, shiftC, ldc, strideC, A, shiftA, lda, strideA);
            }
            else
            {
                // update
                if(row)
                    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                                   &one, UV, shiftUV, lduv, strideUV, bufferT, shiftT, ldt, strideT,
                                   &zero, A, shiftA, lda, strideA, batch_count, workArr);
                else
                    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                                   &one, bufferT, shiftT, ldt, strideT, UV, shiftUV, lduv, strideUV,
                                   &zero, A, shiftA, lda, strideA, batch_count, workArr);

                // copy back to U/V
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, m, n, A,
                                        shiftA, lda, strideA, UV, shiftUV, lduv, strideUV);

                // overwrite A if required
                if(othervO)
                    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                            dim3(thread_count, thread_count, 1), 0, stream, k, k,
                                            bufferC, shiftC, ldc, strideC, A, shiftA, lda, strideA);
            }
        }

        else
        /************ SUB-CASE: USE IN-PLACE THIN-SVD ALGORITHM *******/
        /**************************************************************/
        {
            /** (Note: A compression is not required when leadvO and othervN. We are
                compressing matrix A here, albeit requiring extra workspace for the purpose of
                testing. -See corresponding unit test for more details-) **/

            //*** STAGE 1: Row (or column) compression ***//
            local_geqrlq_template<BATCHED, STRIDED>(
                handle, m, n, A, shiftA, lda, strideA, tau_splits, k, batch_count, scalars,
                work_workArr, Abyx_norms_trfact_X, diag_tmptr_Y, workArr, row);

            if(!leadvO)
                // copy factorization to U or V when needed
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, m, n, A,
                                        shiftA, lda, strideA, UV, shiftUV, lduv, strideUV);

            if(othervS || othervA || (leadvO && othervN))
                // copy the triangular part
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, k, k, A,
                                        shiftA, lda, strideA, bufferT, shiftT, ldt, strideT,
                                        no_mask{}, uplo);

            //*** STAGE 2: generate orthonormal/unitary matrix from row/column compression ***//
            if(leadvO)
                local_orgqrlq_ungqrlq_template<BATCHED, STRIDED>(
                    handle, m, n, k, A, shiftA, lda, strideA, tau_splits, k, batch_count, scalars,
                    (T*)work_workArr, Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X, workArr, row);
            else if(leadvA)
                local_orgqrlq_ungqrlq_template<false, STRIDED>(
                    handle, kk, kk, k, UV, shiftUV, lduv, strideUV, tau_splits, k, batch_count,
                    scalars, (T*)work_workArr, Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X, workArr,
                    row);
            else
                local_orgqrlq_ungqrlq_template<false, STRIDED>(
                    handle, m, n, k, UV, shiftUV, lduv, strideUV, tau_splits, k, batch_count, scalars,
                    (T*)work_workArr, Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X, workArr, row);

            //*** STAGE 3: Bidiagonalization ***//
            if(othervS || othervA || (leadvO && othervN))
            {
                // clean triangular factor
                ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, k, k,
                                        bufferT, shiftT, ldt, strideT, uplo);

                rocsolver_gebrd_template<false, STRIDED>(
                    handle, k, k, bufferT, shiftT, ldt, strideT, S, strideS, E, strideE, tau_splits,
                    k, (tau_splits + k * batch_count), k, Abyx_norms_trfact_X, shiftX, ldx, strideX,
                    diag_tmptr_Y, shiftY, ldy, strideY, batch_count, scalars, work_workArr,
                    Abyx_norms_tmptr_cmplt);

                uplo = rocblas_fill_upper;
            }
            else
            {
                // clean triangular factor
                ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, k, k, A,
                                        shiftA, lda, strideA, uplo);

                rocsolver_gebrd_template<BATCHED, STRIDED>(
                    handle, k, k, A, shiftA, lda, strideA, S, strideS, E, strideE, tau_splits, k,
                    (tau_splits + k * batch_count), k, Abyx_norms_trfact_X, shiftX, ldx, strideX,
                    diag_tmptr_Y, shiftY, ldy, strideY, batch_count, scalars, work_workArr,
                    Abyx_norms_tmptr_cmplt);

                uplo = rocblas_fill_upper;
            }

            //*** STAGE 4: generate orthonormal/unitary matrices from bidiagonalization ***//
            // for lead-dimension vectors
            if(othervS || othervA || (leadvO && othervN))
            {
                if(leadvO)
                    rocsolver_ormbr_unmbr_template<BATCHED, STRIDED>(
                        handle, storev_lead, side, trans, m, n, k, bufferT, shiftT, ldt, strideT,
                        (tau_splits + offset_lead), k, A, shiftA, lda, strideA, batch_count,
                        scalars, Abyx_norms_tmptr_cmplt, diag_tmptr_Y, Abyx_norms_trfact_X, workArr);
                else
                    rocsolver_ormbr_unmbr_template<false, STRIDED>(
                        handle, storev_lead, side, trans, m, n, k, bufferT, shiftT, ldt, strideT,
                        (tau_splits + offset_lead), k, UV, shiftUV, lduv, strideUV, batch_count,
                        scalars, Abyx_norms_tmptr_cmplt, diag_tmptr_Y, Abyx_norms_trfact_X, workArr);
            }
            else
                rocsolver_ormbr_unmbr_template<BATCHED, STRIDED>(
                    handle, storev_lead, side, trans, m, n, k, A, shiftA, lda, strideA,
                    (tau_splits + offset_lead), k, UV, shiftUV, lduv, strideUV, batch_count,
                    scalars, Abyx_norms_tmptr_cmplt, diag_tmptr_Y, Abyx_norms_trfact_X, workArr);

            // for the other-side vectors
            if(othervS || othervA)
                rocsolver_orgbr_ungbr_template<false, STRIDED>(
                    handle, storev_other, k, k, k, bufferT, shiftT, ldt, strideT,
                    (tau_splits + offset_other), k, batch_count, scalars, (T*)work_workArr,
                    Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X, workArr);
            else if(othervO)
                rocsolver_orgbr_ungbr_template<BATCHED, STRIDED>(
                    handle, storev_other, k, k, k, A, shiftA, lda, strideA,
                    (tau_splits + offset_other), k, batch_count, scalars, (T*)work_workArr,
                    Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X, workArr);

            //*** STAGE 5: Compute singular values and vectors from the bidiagonal form ***//
            uplo = rocblas_fill_upper;
            if(!leftvO && !rightvO)
            {
                rocsolver_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, V,
                                            shiftV, ldv, strideV, U, shiftU, ldu, strideU,
                                            (T*)nullptr, 0, 1, 1, info, batch_count,
                                            (rocblas_int*)tau_splits, (TT*)work_workArr,
                                            (rocblas_int*)Abyx_norms_tmptr_cmplt);
            }
            else if(leftvO && !rightvO)
            {
                rocsolver_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, V,
                                            shiftV, ldv, strideV, A, shiftA, lda, strideA,
                                            (W) nullptr, 0, 1, 1, info, batch_count,
                                            (rocblas_int*)tau_splits, (TT*)work_workArr,
                                            (rocblas_int*)Abyx_norms_tmptr_cmplt);
            }
            else
            {
                rocsolver_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, A,
                                            shiftA, lda, strideA, U, shiftU, ldu, strideU,
                                            (W) nullptr, 0, 1, 1, info, batch_count,
                                            (rocblas_int*)tau_splits, (TT*)work_workArr,
                                            (rocblas_int*)Abyx_norms_tmptr_cmplt);
            }

            //*** STAGE 6: update vectors with orthonormal/unitary matrices ***//
            // N/A
        }
    }

    else
    /*********************************************/
    /********** CASE: CHOOSE NORMAL SVD **********/
    /*********************************************/
    {
        //*** STAGE 1: Row (or column) compression ***//
        // N/A

        //*** STAGE 2: generate orthonormal/unitary matrix from row/column compression ***//
        // N/A

        //*** STAGE 3: Bidiagonalization ***//
        rocsolver_gebrd_template<BATCHED, STRIDED>(
            handle, m, n, A, shiftA, lda, strideA, S, strideS, E, strideE, tau_splits, k,
            (tau_splits + k * batch_count), k, Abyx_norms_trfact_X, shiftX, ldx, strideX,
            diag_tmptr_Y, shiftY, ldy, strideY, batch_count, scalars, work_workArr,
            Abyx_norms_tmptr_cmplt);

        //*** STAGE 4: generate orthonormal/unitary matrices from bidiagonalization ***//
        if(leftvS || leftvA)
        {
            // copy data to matrix U where orthogonal matrix will be generated
            mn = (row && leftvS) ? n : m;
            ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_k, batch_count),
                                    dim3(thread_count, thread_count, 1), 0, stream, m, k, A, shiftA,
                                    lda, strideA, U, shiftU, ldu, strideU);

            rocsolver_orgbr_ungbr_template<false, STRIDED>(
                handle, rocblas_column_wise, m, mn, n, U, shiftU, ldu, strideU, tau_splits, k,
                batch_count, scalars, (T*)work_workArr, Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X,
                workArr);
        }

        if(rightvS || rightvA)
        {
            // copy data to matrix V where othogonal matrix will be generated
            mn = (!row && rightvS) ? m : n;
            ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_n, batch_count),
                                    dim3(thread_count, thread_count, 1), 0, stream, k, n, A, shiftA,
                                    lda, strideA, V, shiftV, ldv, strideV);

            rocsolver_orgbr_ungbr_template<false, STRIDED>(
                handle, rocblas_row_wise, mn, n, m, V, shiftV, ldv, strideV,
                (tau_splits + k * batch_count), k, batch_count, scalars, (T*)work_workArr,
                Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X, workArr);
        }

        if(leftvO)
        {
            rocsolver_orgbr_ungbr_template<BATCHED, STRIDED>(
                handle, rocblas_column_wise, m, k, n, A, shiftA, lda, strideA, tau_splits, k,
                batch_count, scalars, (T*)work_workArr, Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X,
                workArr);
        }

        if(rightvO)
        {
            rocsolver_orgbr_ungbr_template<BATCHED, STRIDED>(
                handle, rocblas_row_wise, k, n, m, A, shiftA, lda, strideA,
                (tau_splits + k * batch_count), k, batch_count, scalars, (T*)work_workArr,
                Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X, workArr);
        }

        //*** STAGE 5: Compute singular values and vectors from the bidiagonal form ***//
        if(!leftvO && !rightvO)
        {
            rocsolver_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, V,
                                        shiftV, ldv, strideV, U, shiftU, ldu, strideU, (T*)nullptr,
                                        0, 1, 1, info, batch_count, (rocblas_int*)tau_splits,
                                        (TT*)work_workArr, (rocblas_int*)Abyx_norms_tmptr_cmplt);
        }

        else if(leftvO && !rightvO)
        {
            rocsolver_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, V,
                                        shiftV, ldv, strideV, A, shiftA, lda, strideA, (W) nullptr,
                                        0, 1, 1, info, batch_count, (rocblas_int*)tau_splits,
                                        (TT*)work_workArr, (rocblas_int*)Abyx_norms_tmptr_cmplt);
        }

        else
        {
            rocsolver_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, A,
                                        shiftA, lda, strideA, U, shiftU, ldu, strideU, (W) nullptr,
                                        0, 1, 1, info, batch_count, (rocblas_int*)tau_splits,
                                        (TT*)work_workArr, (rocblas_int*)Abyx_norms_tmptr_cmplt);
        }

        //*** STAGE 6: update vectors with orthonormal/unitary matrices ***//
        // N/A
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_template_alt(rocblas_handle handle,
                                            const rocblas_svect left_svect,
                                            const rocblas_svect right_svect,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            W A,
                                            const rocblas_int shiftA,
                                            const rocblas_int lda,
                                            const rocblas_stride strideA,
                                            TT* S,
                                            const rocblas_stride strideS,
                                            T* U,
                                            const rocblas_int ldu,
                                            const rocblas_stride strideU,
                                            T* V,
                                            const rocblas_int ldv,
                                            const rocblas_stride strideV,
                                            TT* E,
                                            const rocblas_stride strideE,
                                            const rocblas_workmode fast_alg,
                                            rocblas_int* info,
                                            const rocblas_int batch_count,

                                            void* const work,
                                            size_t const size_work)

{
    ROCSOLVER_ENTER("gesvd", "leftsv:", left_svect, "rightsv:", right_svect, "m:", m, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "ldu:", ldu, "ldv:", ldv, "mode:", fast_alg,
                    "bc:", batch_count);

    constexpr bool COMPLEX = rocblas_is_complex<T>;

    // quick return
    if(n == 0 || m == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // ---------------------------------------------------------------------------------
    // TODO: try to figure out how arrays diag_tmptr_Y, tau_splits, Abyx_norms_trfact_X
    // are shared and reused
    // ---------------------------------------------------------------------------------
    size_t size_scalars = 0;
    size_t size_work_workArr = 0;
    size_t size_Abyx_norms_tmptr_cmplt = 0;
    size_t size_Abyx_norms_trfact_X = 0;
    size_t size_diag_tmptr_Y = 0;
    size_t size_tau_splits = 0;
    size_t size_tempArrayT = 0;
    size_t size_tempArrayC = 0;
    size_t size_workArr = 0;

    rocsolver_gesvd_getMemorySize<BATCHED, T, TT>(
        left_svect, right_svect, m, n, batch_count, fast_alg,

        &size_scalars, &size_work_workArr, &size_Abyx_norms_tmptr_cmplt, &size_Abyx_norms_trfact_X,
        &size_diag_tmptr_Y, &size_tau_splits, &size_tempArrayT, &size_tempArrayC, &size_workArr);

    {
        // -----------------------------------------
        // not need for temporary arrays
        // diag_tmptr_Y[] and Abyx_norms_trfact_X[]
        // -----------------------------------------

        size_Abyx_norms_trfact_X = 0;
        size_diag_tmptr_Y = 0;
    }

    rocblas_status istat = rocblas_status_success;

    std::byte* const pwork = (std::byte*)work;
    std::byte* pfree = pwork;

    T* const tau_splits = (T*)pfree;
    pfree += size_tau_splits;

    T* const Abyx_norms_trfact_X = (T*)pfree;
    pfree += size_Abyx_norms_trfact_X;

    T* const diag_tmptr_Y = (T*)pfree;
    pfree += size_diag_tmptr_Y;

    T* tempArrayC = (T*)pfree;
    pfree += size_tempArrayC;

    T* tempArrayT = (T*)pfree;
    pfree += size_tempArrayT;

    {
        bool const is_mem_ok = (pfree <= (pwork + size_work));
        if(!is_mem_ok)
        {
            istat = rocblas_status_memory_error;
            return (istat);
        }
    }

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // -----------------------------------------
    // use try/throw/catch to reset pointer mode
    // in case of error
    // -----------------------------------------
    try
    {
        // constants to use when calling rocablas functions
        T one = 1;
        T zero = 0;

        // booleans used to determine the path that the execution will follow:
        const bool row = (m >= n);
        const bool leftvS = (left_svect == rocblas_svect_singular);
        const bool leftvO = (left_svect == rocblas_svect_overwrite);
        const bool leftvA = (left_svect == rocblas_svect_all);
        const bool leftvN = (left_svect == rocblas_svect_none);
        const bool rightvS = (right_svect == rocblas_svect_singular);
        const bool rightvO = (right_svect == rocblas_svect_overwrite);
        const bool rightvA = (right_svect == rocblas_svect_all);
        const bool rightvN = (right_svect == rocblas_svect_none);
        const bool leadvS = row ? leftvS : rightvS;
        const bool leadvO = row ? leftvO : rightvO;
        const bool leadvA = row ? leftvA : rightvA;
        const bool leadvN = row ? leftvN : rightvN;
        const bool othervS = !row ? leftvS : rightvS;
        const bool othervO = !row ? leftvO : rightvO;
        const bool othervA = !row ? leftvA : rightvA;
        const bool othervN = !row ? leftvN : rightvN;
        const bool thinSVD = (m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m);
        const bool fast_thinSVD = (thinSVD && fast_alg == rocblas_outofplace);

        // auxiliary sizes and variables
        const rocblas_int k = std::min(m, n);
        const rocblas_int kk = std::max(m, n);
        const rocblas_int shiftX = 0;
        const rocblas_int shiftY = 0;
        const rocblas_int shiftUV = 0;
        const rocblas_int shiftT = 0;
        const rocblas_int shiftC = 0;
        const rocblas_int shiftU = 0;
        const rocblas_int shiftV = 0;
        const rocblas_int ldx = thinSVD ? k : m;
        const rocblas_int ldy = thinSVD ? k : n;
        const rocblas_stride strideX = ldx * GEBRD_BLOCKSIZE;
        const rocblas_stride strideY = ldy * GEBRD_BLOCKSIZE;
        T* bufferT = tempArrayT;
        rocblas_int ldt = k;
        rocblas_stride strideT = k * k;
        T* bufferC = tempArrayC;
        rocblas_int ldc = m;
        rocblas_stride strideC = m * n;

        T* UV;
        rocblas_int lduv, mn, nu, nv;
        rocblas_int offset_other, offset_lead;
        rocblas_storev storev_other, storev_lead;
        rocblas_stride strideUV;
        rocblas_fill uplo;
        rocblas_side side;
        rocblas_operation trans;

        nu = leftvN ? 0 : m;
        nv = rightvN ? 0 : n;
        if(row)
        {
            UV = U;
            lduv = ldu;
            strideUV = strideU;
            uplo = rocblas_fill_upper;
            storev_other = rocblas_row_wise;
            storev_lead = rocblas_column_wise;
            offset_other = k * batch_count;
            offset_lead = 0;
            if(othervS || othervA)
            {
                bufferC = V;
                ldc = ldv;
                strideC = strideV;
                if(!fast_thinSVD)
                {
                    bufferT = V;
                    ldt = ldv;
                    strideT = strideV;
                }
            }
            side = rocblas_side_right;
            trans = rocblas_operation_none;
        }
        else
        {
            UV = V;
            lduv = ldv;
            strideUV = strideV;
            uplo = rocblas_fill_lower;
            storev_other = rocblas_column_wise;
            storev_lead = rocblas_row_wise;
            offset_other = 0;
            offset_lead = k * batch_count;
            if(othervS || othervA)
            {
                bufferC = U;
                ldc = ldu;
                strideC = strideU;
                if(!fast_thinSVD)
                {
                    bufferT = U;
                    ldt = ldu;
                    strideT = strideU;
                }
            }
            side = rocblas_side_left;
            trans = COMPLEX ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;
        }

        // common block sizes and number of threads for internal kernels
        constexpr rocblas_int thread_count = 32;
        const rocblas_int blocks_m = (m - 1) / thread_count + 1;
        const rocblas_int blocks_n = (n - 1) / thread_count + 1;
        const rocblas_int blocks_k = (k - 1) / thread_count + 1;

        /** A thin SVD could be computed for matrices with sufficiently more rows than
        columns (or columns than rows) by starting with a QR factorization (or LQ
        factorization) and working with the triangular factor afterwards. When
        computing a thin SVD, a fast algorithm could be executed by doing some
        computations out-of-place. **/

        if(thinSVD)
        /*******************************************/
        /********** CASE: CHOOSE THIN-SVD **********/
        /*******************************************/
        {
            if(leadvN)
            /***** SUB-CASE: USE THIN-SVD WITH NO LEAD-DIMENSION VECTORS *****/
            /*****************************************************************/
            {
                nu = leftvN ? 0 : k;
                nv = rightvN ? 0 : k;

                //*** STAGE 1: Row (or column) compression ***//
                {
                    size_t const size_geqrlq = (pwork + size_work) - pfree;
                    void* work_geqrlq = (void*)pfree;

                    istat = local_geqrlq_template_alt<BATCHED, STRIDED>(
                        handle, m, n, A, shiftA, lda, strideA, tau_splits, k, batch_count, row,

                        work_geqrlq, size_geqrlq);
                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }

                //*** STAGE 2: generate orthonormal/unitary matrix from row/column compression ***//
                // N/A

                //*** STAGE 3: Bidiagonalization ***//
                // clean triangular factor
                ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, k, k, A,
                                        shiftA, lda, strideA, uplo);

                {
                    size_t const size_work_gebrd = (pwork + size_work) - pfree;
                    void* const work_gebrd = (void*)pfree;

                    // --------------------------------
                    // TODO: are X, Y scratch arrays in gebrd ?
                    // --------------------------------
                    T* const X = (T*)Abyx_norms_trfact_X;
                    T* const Y = (T*)diag_tmptr_Y;

                    istat = rocsolver_gebrd_template_alt<BATCHED, STRIDED>(
                        handle, k, k,

                        A, shiftA, lda, strideA,

                        S, strideS, E, strideE,

                        tau_splits, k, (tau_splits + k * batch_count), k,

                        // X,
                        shiftX, ldx, strideX,

                        // Y,
                        shiftY, ldy, strideY,

                        batch_count,

                        work_gebrd, size_work_gebrd);

                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }

                //*** STAGE 4: generate orthonormal/unitary matrices from bidiagonalization ***//
                if(!othervN)
                {
                    size_t const size_work_orgbr = (pwork + size_work) - pfree;
                    void* const work_orgbr = (void*)pfree;

                    istat = rocsolver_orgbr_ungbr_template_alt<BATCHED, STRIDED>(
                        handle, storev_other, k, k, k, A, shiftA, lda, strideA,
                        (tau_splits + offset_other), k, batch_count,

                        work_orgbr, size_work_orgbr);

                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }

                //*** STAGE 5: Compute singular values and vectors from the bidiagonal form ***//
                if(row)
                {
                    size_t const size_work_bdsqr = (pwork + size_work) - pfree;
                    void* const work_bdsqr = (void*)pfree;

                    istat = rocsolver_bdsqr_template_alt<T>(handle, rocblas_fill_upper, k, nv, nu,
                                                            0, S, strideS, E, strideE, A, shiftA,
                                                            lda, strideA, U, shiftU, ldu, strideU,
                                                            (W) nullptr, 0, 1, 1, info, batch_count,

                                                            work_bdsqr, size_work_bdsqr);

                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }
                else
                {
                    size_t const size_work_bdsqr = (pwork + size_work) - pfree;
                    void* const work_bdsqr = (void*)pfree;

                    istat = rocsolver_bdsqr_template_alt<T>(handle, rocblas_fill_upper, k, nv, nu,
                                                            0, S, strideS, E, strideE, V, shiftV,
                                                            ldv, strideV, A, shiftA, lda, strideA,
                                                            (W) nullptr, 0, 1, 1, info, batch_count,

                                                            work_bdsqr, size_work_bdsqr);
                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }

                //*** STAGE 6: update vectors with orthonormal/unitary matrices ***//
                if(othervS || othervA)
                {
                    mn = row ? n : m;
                    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                                            dim3(thread_count, thread_count, 1), 0, stream, mn, mn,
                                            A, shiftA, lda, strideA, bufferC, shiftC, ldc, strideC);
                }
            }

            else if(fast_thinSVD)
            /***** SUB-CASE: USE FAST (OUT-OF-PLACE) THIN-SVD ALGORITHM *****/
            /****************************************************************/
            {
                nu = leftvN ? 0 : k;
                nv = rightvN ? 0 : k;

                //*** STAGE 1: Row (or column) compression ***//
                {
                    size_t const size_work_geqrlq = (pwork + size_work) - pfree;
                    void* const work_geqrlq = (void*)pfree;

                    istat = local_geqrlq_template_alt<BATCHED, STRIDED>(
                        handle, m, n, A, shiftA, lda, strideA, tau_splits, k, batch_count, row,

                        work_geqrlq, size_work_geqrlq);

                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }

                if(leadvA)
                {
                    // copy factorization to U or V when needed
                    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                                            dim3(thread_count, thread_count, 1), 0, stream, m, n, A,
                                            shiftA, lda, strideA, UV, shiftUV, lduv, strideUV);
                }

                // copy the triangular part to be used in the bidiagonalization
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, k, k, A,
                                        shiftA, lda, strideA, bufferT, shiftT, ldt, strideT,
                                        no_mask{}, uplo);

                //*** STAGE 2: generate orthonormal/unitary matrix from row/column compression ***//
                if(leadvA)
                {
                    size_t const size_work_orgqrlq = (pwork + size_work) - pfree;
                    void* work_orgqrlq = (void*)pfree;

                    istat = local_orgqrlq_ungqrlq_template_alt<false, STRIDED>(
                        handle, kk, kk, k, UV, shiftUV, lduv, strideUV, tau_splits, k, batch_count,
                        row,

                        work_orgqrlq, size_work_orgqrlq);

                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }
                else
                {
                    size_t const size_work_orgqrlq = (pwork + size_work) - pfree;
                    void* work_orgqrlq = (void*)pfree;
                    istat = local_orgqrlq_ungqrlq_template_alt<BATCHED, STRIDED>(
                        handle, m, n, k, A, shiftA, lda, strideA, tau_splits, k, batch_count, row,

                        work_orgqrlq, size_work_orgqrlq);

                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }

                //*** STAGE 3: Bidiagonalization ***//
                // clean triangular factor
                ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, k, k,
                                        bufferT, shiftT, ldt, strideT, uplo);

                {
                    size_t const size_work_gebrd = (pwork + size_work) - pfree;
                    void* const work_gebrd = (void*)pfree;

                    // --------------------------------
                    // TODO: are X, Y scratch arrays in gebrd ?
                    // --------------------------------
                    T* const X = (T*)Abyx_norms_trfact_X;
                    T* const Y = (T*)diag_tmptr_Y;

                    istat = rocsolver_gebrd_template_alt<false, STRIDED>(
                        handle, k, k,

                        bufferT, shiftT, ldt, strideT,

                        S, strideS, E, strideE,

                        tau_splits, k, (tau_splits + k * batch_count), k,

                        // X,
                        shiftX, ldx, strideX,

                        // Y,
                        shiftY, ldy, strideY,

                        batch_count,

                        work_gebrd, size_work_gebrd);

                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }

                if(!othervN)
                {
                    // copy results to generate non-lead vectors if required
                    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                            dim3(thread_count, thread_count, 1), 0, stream, k, k,
                                            bufferT, shiftT, ldt, strideT, bufferC, shiftC, ldc,
                                            strideC);
                }

                //*** STAGE 4: generate orthonormal/unitary matrices from bidiagonalization ***//
                // for lead-dimension vectors
                {
                    size_t const size_work_orgbr = (pfree + size_work) - pfree;
                    void* const work_orgbr = (void*)pfree;

                    istat = rocsolver_orgbr_ungbr_template_alt<false, STRIDED>(
                        handle, storev_lead, k, k, k, bufferT, shiftT, ldt, strideT,
                        (tau_splits + offset_lead), k, batch_count,

                        work_orgbr, size_work_orgbr);

                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }

                // for the other-side vectors
                if(!othervN)
                {
                    size_t const size_work_orgbr = (pfree + size_work) - pfree;
                    void* const work_orgbr = (void*)pfree;

                    istat = rocsolver_orgbr_ungbr_template_alt<false, STRIDED>(
                        handle, storev_other, k, k, k, bufferC, shiftC, ldc, strideC,
                        (tau_splits + offset_other), k, batch_count,

                        work_orgbr, size_work_orgbr);

                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }

                //*** STAGE 5: Compute singular values and vectors from the bidiagonal form ***//
                if(row)
                {
                    size_t const size_work_bdsqr = (pwork + size_work) - pfree;
                    void* const work_bdsqr = (void*)pfree;

                    istat = rocsolver_bdsqr_template_alt<T>(
                        handle, rocblas_fill_upper, k, nv, nu, 0, S, strideS, E, strideE, bufferC,
                        shiftC, ldc, strideC, bufferT, shiftT, ldt, strideT, (T*)nullptr, 0, 1, 1,
                        info, batch_count,

                        work_bdsqr, size_work_bdsqr);

                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }
                else
                {
                    size_t const size_work_bdsqr = (pwork + size_work) - pfree;
                    void* const work_bdsqr = (void*)pfree;

                    istat = rocsolver_bdsqr_template_alt<T>(
                        handle, rocblas_fill_upper, k, nv, nu, 0, S, strideS, E, strideE, bufferT,
                        shiftT, ldt, strideT, bufferC, shiftC, ldc, strideC, (T*)nullptr, 0, 1, 1,
                        info, batch_count,

                        work_bdsqr, size_work_bdsqr);

                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }

                //*** STAGE 6: update vectors with orthonormal/unitary matrices ***//
                if(leadvO)
                {
                    bufferC = tempArrayC;
                    ldc = m;
                    strideC = m * n;

                    // update
                    if(row)
                    {
                        auto const pfree_saved = pfree;
                        size_t const size_workArr = sizeof(T*) * batch_count;

                        T** workArr = (T**)pfree;
                        pfree += size_workArr;
                        MEM_CHECK_THROW(pfree);

                        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n,
                                       k, &one, A, shiftA, lda, strideA, bufferT, shiftT, ldt,
                                       strideT, &zero, bufferC, shiftC, ldc, strideC, batch_count,
                                       workArr);

                        pfree = pfree_saved;
                    }
                    else
                    {
                        auto const pfree_saved = pfree;
                        size_t const size_workArr = sizeof(T*) * batch_count;
                        T** workArr = (T**)pfree;
                        pfree += size_workArr;
                        MEM_CHECK_THROW(pfree);

                        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n,
                                       k, &one, bufferT, shiftT, ldt, strideT, A, shiftA, lda,
                                       strideA, &zero, bufferC, shiftC, ldc, strideC, batch_count,
                                       workArr);

                        pfree = pfree_saved;
                    }

                    // copy to overwrite A
                    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                                            dim3(thread_count, thread_count, 1), 0, stream, m, n,
                                            bufferC, shiftC, ldc, strideC, A, shiftA, lda, strideA);
                }
                else if(leadvS)
                {
                    // update
                    if(row)
                    {
                        auto const pfree_saved = pfree;
                        size_t const size_workArr = sizeof(T*) * batch_count;
                        T** workArr = (T**)pfree;
                        pfree += size_workArr;
                        MEM_CHECK_THROW(pfree);

                        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n,
                                       k, &one, A, shiftA, lda, strideA, bufferT, shiftT, ldt,
                                       strideT, &zero, UV, shiftUV, lduv, strideUV, batch_count,
                                       workArr);

                        pfree = pfree_saved;
                    }
                    else
                    {
                        auto const pfree_saved = pfree;
                        size_t const size_workArr = sizeof(T*) * batch_count;
                        T** workArr = (T**)pfree;
                        pfree += size_workArr;
                        MEM_CHECK_THROW(pfree);

                        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n,
                                       k, &one, bufferT, shiftT, ldt, strideT, A, shiftA, lda,
                                       strideA, &zero, UV, shiftUV, lduv, strideUV, batch_count,
                                       workArr);

                        pfree = pfree_saved;
                    }

                    // overwrite A if required
                    if(othervO)
                        ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                                dim3(thread_count, thread_count, 1), 0, stream, k,
                                                k, bufferC, shiftC, ldc, strideC, A, shiftA, lda,
                                                strideA);
                }
                else
                {
                    // update
                    if(row)
                    {
                        auto const pfree_saved = pfree;
                        size_t const size_workArr = sizeof(T*) * batch_count;
                        T** workArr = (T**)pfree;
                        pfree += size_workArr;
                        MEM_CHECK_THROW(pfree);

                        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n,
                                       k, &one, UV, shiftUV, lduv, strideUV, bufferT, shiftT, ldt,
                                       strideT, &zero, A, shiftA, lda, strideA, batch_count, workArr);
                        pfree = pfree_saved;
                    }
                    else
                    {
                        auto const pfree_saved = pfree;
                        size_t const size_workArr = sizeof(T*) * batch_count;
                        T** workArr = (T**)pfree;
                        pfree += size_workArr;
                        MEM_CHECK_THROW(pfree);

                        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n,
                                       k, &one, bufferT, shiftT, ldt, strideT, UV, shiftUV, lduv,
                                       strideUV, &zero, A, shiftA, lda, strideA, batch_count,
                                       workArr);

                        pfree = pfree_saved;
                    }

                    // copy back to U/V
                    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                                            dim3(thread_count, thread_count, 1), 0, stream, m, n, A,
                                            shiftA, lda, strideA, UV, shiftUV, lduv, strideUV);

                    // overwrite A if required
                    if(othervO)
                        ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                                dim3(thread_count, thread_count, 1), 0, stream, k,
                                                k, bufferC, shiftC, ldc, strideC, A, shiftA, lda,
                                                strideA);
                }
            }

            else
            /************ SUB-CASE: USE IN-PLACE THIN-SVD ALGORITHM *******/
            /**************************************************************/
            {
                /** (Note: A compression is not required when leadvO and othervN. We are
                compressing matrix A here, albeit requiring extra workspace for the purpose of
                testing. -See corresponding unit test for more details-) **/

                //*** STAGE 1: Row (or column) compression ***//
                {
                    size_t const size_work_geqrlq = (pwork + size_work) - pfree;
                    void* const work_geqrlq = pfree;

                    istat = local_geqrlq_template_alt<BATCHED, STRIDED>(
                        handle, m, n, A, shiftA, lda, strideA, tau_splits, k, batch_count, row,

                        work_geqrlq, size_work_geqrlq);

                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }

                if(!leadvO)
                {
                    // copy factorization to U or V when needed
                    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                                            dim3(thread_count, thread_count, 1), 0, stream, m, n, A,
                                            shiftA, lda, strideA, UV, shiftUV, lduv, strideUV);
                }

                if(othervS || othervA || (leadvO && othervN))
                {
                    // copy the triangular part
                    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                            dim3(thread_count, thread_count, 1), 0, stream, k, k, A,
                                            shiftA, lda, strideA, bufferT, shiftT, ldt, strideT,
                                            no_mask{}, uplo);
                }

                //*** STAGE 2: generate orthonormal/unitary matrix from row/column compression ***//
                if(leadvO)
                {
                    size_t const size_work_orgqrlq = (pwork + size_work) - pfree;
                    void* const work_orgqrlq = (void*)pfree;

                    istat = local_orgqrlq_ungqrlq_template_alt<BATCHED, STRIDED>(
                        handle, m, n, k, A, shiftA, lda, strideA, tau_splits, k, batch_count, row,

                        work_orgqrlq, size_work_orgqrlq);
                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }
                else if(leadvA)
                {
                    size_t const size_work_orgqrlq = (pwork + size_work) - pfree;
                    void* const work_orgqrlq = (void*)pfree;

                    istat = local_orgqrlq_ungqrlq_template_alt<false, STRIDED>(
                        handle, kk, kk, k, UV, shiftUV, lduv, strideUV, tau_splits, k, batch_count,
                        row,

                        work_orgqrlq, size_work_orgqrlq);
                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }
                else
                {
                    size_t const size_work_orgqrlq = (pwork + size_work) - pfree;
                    void* const work_orgqrlq = (void*)pfree;

                    istat = local_orgqrlq_ungqrlq_template_alt<false, STRIDED>(
                        handle, m, n, k, UV, shiftUV, lduv, strideUV, tau_splits, k, batch_count, row,

                        work_orgqrlq, size_work_orgqrlq);
                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }

                //*** STAGE 3: Bidiagonalization ***//
                if(othervS || othervA || (leadvO && othervN))
                {
                    // clean triangular factor
                    ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                                            dim3(thread_count, thread_count, 1), 0, stream, k, k,
                                            bufferT, shiftT, ldt, strideT, uplo);

                    {
                        size_t const size_work_gebrd = (pwork + size_work) - pfree;
                        void* const work_gebrd = (void*)pfree;

                        // --------------------------------
                        // TODO: are X, Y scratch arrays in gebrd ?
                        // --------------------------------
                        T* const X = (T*)Abyx_norms_trfact_X;
                        T* const Y = (T*)diag_tmptr_Y;

                        istat = rocsolver_gebrd_template_alt<false, STRIDED>(
                            handle, k, k, bufferT, shiftT, ldt, strideT, S, strideS, E, strideE,

                            tau_splits, k, (tau_splits + k * batch_count), k,

                            // X,
                            shiftX, ldx, strideX,

                            // Y,
                            shiftY, ldy, strideY,

                            batch_count,

                            work_gebrd, size_work_gebrd);
                        if(istat != rocblas_status_success)
                        {
                            throw(istat);
                        }
                    }

                    uplo = rocblas_fill_upper;
                }
                else
                {
                    // clean triangular factor
                    ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                                            dim3(thread_count, thread_count, 1), 0, stream, k, k, A,
                                            shiftA, lda, strideA, uplo);

                    {
                        size_t const size_work_gebrd = (pwork + size_work) - pfree;
                        void* const work_gebrd = (void*)pfree;

                        // --------------------------------
                        // TODO: are X, Y scratch arrays in gebrd ?
                        // --------------------------------
                        T* const X = (T*)Abyx_norms_trfact_X;
                        T* const Y = (T*)diag_tmptr_Y;

                        istat = rocsolver_gebrd_template_alt<BATCHED, STRIDED>(
                            handle, k, k, A, shiftA, lda, strideA, S, strideS, E, strideE,

                            tau_splits, k, (tau_splits + k * batch_count), k,

                            // X,
                            shiftX, ldx, strideX,

                            // Y,
                            shiftY, ldy, strideY,

                            batch_count,

                            work_gebrd, size_work_gebrd);
                        if(istat != rocblas_status_success)
                        {
                            throw(istat);
                        }
                    }

                    uplo = rocblas_fill_upper;
                }

                //*** STAGE 4: generate orthonormal/unitary matrices from bidiagonalization ***//
                // for lead-dimension vectors
                if(othervS || othervA || (leadvO && othervN))
                {
                    if(leadvO)
                    {
                        size_t const size_work_ormbr = (pwork + size_work) - pfree;
                        void* const work_ormbr = (void*)pfree;

                        istat = rocsolver_ormbr_unmbr_template_alt<BATCHED, STRIDED>(
                            handle, storev_lead, side, trans, m, n, k, bufferT, shiftT, ldt, strideT,
                            (tau_splits + offset_lead), k, A, shiftA, lda, strideA, batch_count,

                            work_ormbr, size_work_ormbr);

                        if(istat != rocblas_status_success)
                        {
                            throw(istat);
                        }
                    }
                    else
                    {
                        size_t const size_work_ormbr = (pwork + size_work) - pfree;
                        void* const work_ormbr = (void*)pfree;

                        istat = rocsolver_ormbr_unmbr_template_alt<false, STRIDED>(
                            handle, storev_lead, side, trans, m, n, k, bufferT, shiftT, ldt, strideT,
                            (tau_splits + offset_lead), k, UV, shiftUV, lduv, strideUV, batch_count,

                            work_ormbr, size_work_ormbr);

                        if(istat != rocblas_status_success)
                        {
                            throw(istat);
                        }
                    }
                }
                else
                {
                    size_t const size_work_ormbr = (pwork + size_work) - pfree;
                    void* const work_ormbr = (void*)pfree;

                    istat = rocsolver_ormbr_unmbr_template_alt<BATCHED, STRIDED>(
                        handle, storev_lead, side, trans, m, n, k, A, shiftA, lda, strideA,
                        (tau_splits + offset_lead), k, UV, shiftUV, lduv, strideUV, batch_count,

                        work_ormbr, size_work_ormbr);
                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }

                // for the other-side vectors
                if(othervS || othervA)
                {
                    size_t const size_work_orgbr = (pwork + size_work) - pfree;
                    void* const work_orgbr = (void*)pfree;

                    istat = rocsolver_orgbr_ungbr_template_alt<false, STRIDED>(
                        handle, storev_other, k, k, k, bufferT, shiftT, ldt, strideT,
                        (tau_splits + offset_other), k, batch_count,

                        work_orgbr, size_work_orgbr);
                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }
                else if(othervO)
                {
                    size_t const size_work_orgbr = (pwork + size_work) - pfree;
                    void* const work_orgbr = (void*)pfree;

                    istat = rocsolver_orgbr_ungbr_template_alt<BATCHED, STRIDED>(
                        handle, storev_other, k, k, k, A, shiftA, lda, strideA,
                        (tau_splits + offset_other), k, batch_count,

                        work_orgbr, size_work_orgbr);
                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }

                //*** STAGE 5: Compute singular values and vectors from the bidiagonal form ***//
                uplo = rocblas_fill_upper;
                if(!leftvO && !rightvO)
                {
                    size_t const size_work_bdsqr = (pwork + size_work) - pfree;
                    void* const work_bdsqr = (void*)pfree;

                    istat = rocsolver_bdsqr_template_alt<T>(
                        handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, V, shiftV, ldv, strideV,
                        U, shiftU, ldu, strideU, (T*)nullptr, 0, 1, 1, info, batch_count,

                        work_bdsqr, size_work_bdsqr);

                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }
                else if(leftvO && !rightvO)
                {
                    size_t const size_work_bdsqr = (pwork + size_work) - pfree;
                    void* const work_bdsqr = (void*)pfree;

                    istat = rocsolver_bdsqr_template_alt<T>(
                        handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, V, shiftV, ldv, strideV,
                        A, shiftA, lda, strideA, (W) nullptr, 0, 1, 1, info, batch_count,

                        work_bdsqr, size_work_bdsqr);
                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }
                else
                {
                    size_t const size_work_bdsqr = (pwork + size_work) - pfree;
                    void* const work_bdsqr = (void*)pfree;

                    istat = rocsolver_bdsqr_template_alt<T>(
                        handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, A, shiftA, lda, strideA,
                        U, shiftU, ldu, strideU, (W) nullptr, 0, 1, 1, info, batch_count,

                        work_bdsqr, size_work_bdsqr);

                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }

                //*** STAGE 6: update vectors with orthonormal/unitary matrices ***//
                // N/A
            }
        }

        else
        /*********************************************/
        /********** CASE: CHOOSE NORMAL SVD **********/
        /*********************************************/
        {
            //*** STAGE 1: Row (or column) compression ***//
            // N/A

            //*** STAGE 2: generate orthonormal/unitary matrix from row/column compression ***//
            // N/A

            //*** STAGE 3: Bidiagonalization ***//
            {
                size_t const size_work_gebrd = (pwork + size_work) - pfree;
                void* const work_gebrd = (void*)pfree;

                T* const X = (T*)Abyx_norms_trfact_X;
                T* const Y = (T*)diag_tmptr_Y;

                istat = rocsolver_gebrd_template_alt<BATCHED, STRIDED>(
                    handle, m, n,

                    A, shiftA, lda, strideA,

                    S, strideS, E, strideE, tau_splits,

                    k, (tau_splits + k * batch_count), k,

                    // X,
                    shiftX, ldx, strideX,

                    // Y,
                    shiftY, ldy, strideY,

                    batch_count,

                    work_gebrd, size_work_gebrd);

                if(istat != rocblas_status_success)
                {
                    throw(istat);
                }
            }

            //*** STAGE 4: generate orthonormal/unitary matrices from bidiagonalization ***//
            if(leftvS || leftvA)
            {
                // copy data to matrix U where orthogonal matrix will be generated
                mn = (row && leftvS) ? n : m;
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_k, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, m, k, A,
                                        shiftA, lda, strideA, U, shiftU, ldu, strideU);

                {
                    size_t const size_work_orgbr = (pwork + size_work) - pfree;
                    void* const work_orgbr = (void*)pfree;

                    istat = rocsolver_orgbr_ungbr_template_alt<false, STRIDED>(
                        handle, rocblas_column_wise, m, mn, n, U, shiftU, ldu, strideU, tau_splits,
                        k, batch_count,

                        work_orgbr, size_work_orgbr);

                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }
            }

            if(rightvS || rightvA)
            {
                // copy data to matrix V where othogonal matrix will be generated
                mn = (!row && rightvS) ? m : n;
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_n, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, k, n, A,
                                        shiftA, lda, strideA, V, shiftV, ldv, strideV);

                {
                    size_t const size_work_orgbr = (pwork + size_work) - pfree;
                    void* const work_orgbr = (void*)pfree;

                    istat = rocsolver_orgbr_ungbr_template_alt<false, STRIDED>(
                        handle, rocblas_row_wise, mn, n, m, V, shiftV, ldv, strideV,
                        (tau_splits + k * batch_count), k, batch_count,

                        work_orgbr, size_work_orgbr);
                    if(istat != rocblas_status_success)
                    {
                        throw(istat);
                    }
                }
            }

            if(leftvO)
            {
                size_t const size_work_orgbr = (pwork + size_work) - pfree;
                void* const work_orgbr = (void*)pfree;

                istat = rocsolver_orgbr_ungbr_template_alt<BATCHED, STRIDED>(
                    handle, rocblas_column_wise, m, k, n, A, shiftA, lda, strideA, tau_splits, k,
                    batch_count,

                    work_orgbr, size_work_orgbr);

                if(istat != rocblas_status_success)
                {
                    throw(istat);
                }
            }

            if(rightvO)
            {
                size_t const size_work_orgbr = (pwork + size_work) - pfree;
                void* const work_orgbr = (void*)pfree;

                istat = rocsolver_orgbr_ungbr_template_alt<BATCHED, STRIDED>(
                    handle, rocblas_row_wise, k, n, m, A, shiftA, lda, strideA,
                    (tau_splits + k * batch_count), k, batch_count,

                    work_orgbr, size_work_orgbr);
                if(istat != rocblas_status_success)
                {
                    throw(istat);
                }
            }

            //*** STAGE 5: Compute singular values and vectors from the bidiagonal form ***//
            if(!leftvO && !rightvO)
            {
                size_t const size_work_bdsqr = (pwork + size_work) - pfree;
                void* const work_bdsqr = (void*)pfree;

                istat = rocsolver_bdsqr_template_alt<T>(
                    handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, V, shiftV, ldv, strideV, U,
                    shiftU, ldu, strideU, (T*)nullptr, 0, 1, 1, info, batch_count,

                    work_bdsqr, size_work_bdsqr);
                if(istat != rocblas_status_success)
                {
                    throw(istat);
                }
            }

            else if(leftvO && !rightvO)
            {
                size_t const size_work_bdsqr = (pwork + size_work) - pfree;
                void* const work_bdsqr = (void*)pfree;

                istat = rocsolver_bdsqr_template_alt<T>(
                    handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, V, shiftV, ldv, strideV, A,
                    shiftA, lda, strideA, (W) nullptr, 0, 1, 1, info, batch_count,

                    work_bdsqr, size_work_bdsqr);
                if(istat != rocblas_status_success)
                {
                    throw(istat);
                }
            }

            else
            {
                size_t const size_work_bdsqr = (pwork + size_work) - pfree;
                void* const work_bdsqr = (void*)pfree;

                istat = rocsolver_bdsqr_template_alt<T>(
                    handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, A, shiftA, lda, strideA, U,
                    shiftU, ldu, strideU, (W) nullptr, 0, 1, 1, info, batch_count,

                    work_bdsqr, size_work_bdsqr);
                if(istat != rocblas_status_success)
                {
                    throw(istat);
                }
            }

            //*** STAGE 6: update vectors with orthonormal/unitary matrices ***//
            // N/A
        }

        rocblas_set_pointer_mode(handle, old_mode);
        return rocblas_status_success;
    }
    catch(rocblas_status istat)
    {
        rocblas_set_pointer_mode(handle, old_mode);
        return istat;
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_template_alt_org(rocblas_handle handle,
                                                const rocblas_svect left_svect,
                                                const rocblas_svect right_svect,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                W A,
                                                const rocblas_int shiftA,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                TT* S,
                                                const rocblas_stride strideS,
                                                T* U,
                                                const rocblas_int ldu,
                                                const rocblas_stride strideU,
                                                T* V,
                                                const rocblas_int ldv,
                                                const rocblas_stride strideV,
                                                TT* E,
                                                const rocblas_stride strideE,
                                                const rocblas_workmode fast_alg,
                                                rocblas_int* info,
                                                const rocblas_int batch_count,

                                                void* work,
                                                size_t const size_work)

{
    size_t size_scalars = 0;
    size_t size_work_workArr = 0;
    size_t size_Abyx_norms_tmptr_cmplt = 0;
    size_t size_Abyx_norms_trfact_X = 0;

    size_t size_diag_tmptr_Y = 0;
    size_t size_tau_splits = 0;
    size_t size_tempArrayT = 0;
    size_t size_tempArrayC = 0;
    size_t size_workArr = 0;

    {
        using S = decltype(std::real(T{}));
        rocsolver_gesvd_getMemorySize<BATCHED, T, S>(
            left_svect, right_svect, m, n, batch_count, fast_alg,

            &size_scalars, &size_work_workArr, &size_Abyx_norms_tmptr_cmplt,
            &size_Abyx_norms_trfact_X,

            &size_diag_tmptr_Y, &size_tau_splits, &size_tempArrayT, &size_tempArrayC, &size_workArr);
    }

    std::byte* const pwork = (std::byte*)work;
    std::byte* pfree = pwork;

    T* const scalars = (T*)pfree;
    pfree += size_scalars;
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    void* const work_workArr = (void*)pfree;
    pfree += size_work_workArr;

    T* const Abyx_norms_tmptr_cmplt = (T*)pfree;
    pfree += size_Abyx_norms_tmptr_cmplt;

    T* const Abyx_norms_trfact_X = (T*)pfree;
    pfree += size_Abyx_norms_trfact_X;

    T* const diag_tmptr_Y = (T*)pfree;
    pfree += size_diag_tmptr_Y;

    T* const tau_splits = (T*)pfree;
    pfree += size_tau_splits;

    T* const tempArrayT = (T*)pfree;
    pfree += size_tempArrayT;

    T* const tempArrayC = (T*)pfree;
    pfree += size_tempArrayC;

    T** const workArr = (T**)pfree;
    pfree += size_workArr;

    MEM_CHECK(pfree);

    rocblas_status istat = rocsolver_gesvd_template<BATCHED, STRIDED, T, TT, W>(
        handle, left_svect, right_svect, m, n, A, shiftA, lda, strideA, S, strideS, U, ldu, strideU,
        V, ldv, strideV, E, strideE, fast_alg, info, batch_count,

        scalars, work_workArr, Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X,

        diag_tmptr_Y, tau_splits, tempArrayT, tempArrayC, workArr);

    return (istat);
}

ROCSOLVER_END_NAMESPACE
