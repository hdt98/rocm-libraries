/* **************************************************************************
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

#include "rocsolver_device_workspace.hpp"

ROCSOLVER_BEGIN_NAMESPACE

//
// DRAFT methods
//
// Those are meant to exist on the respective roclapack and rocauxiliary `.hpp`
// files; will be moved when finalized.
//

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
auto rocsolver_geqrf_getWorkItems(rocblas_handle handle,
                                  const I m,
                                  const I n,
                                  U /* A */,
                                  const rocblas_stride shiftA,
                                  const I lda,
                                  const rocblas_stride strideA,
                                  T* /* ipiv */,
                                  const rocblas_stride strideP,
                                  const I batch_count)
{
    //
    // Get sizes using legacy `_getMemorySize` method
    //
    // Size for constants in rocblas calls
    size_t size_scalars;
    // Size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr, size_workArr;
    // Extra requirements for calling GEQR2 and to store temporary triangular factor
    size_t size_Abyx_norms_trfact;
    // Extra requirements for calling GEQR2 and LARFB
    size_t size_diag_tmptr;
    rocsolver_geqrf_getMemorySize<BATCHED, T>(m, n, batch_count, &size_scalars, &size_work_workArr,
                                              &size_Abyx_norms_trfact, &size_diag_tmptr,
                                              &size_workArr);

    //
    // Create a list of work items with previously computed sizes and return it
    //
    auto work_items = create_work_item({"geqrf_scalars", size_scalars})
        + create_work_item({"geqrf_work_workArr", size_work_workArr})
        + create_work_item({"geqrf_workArr", size_workArr})
        + create_work_item({"geqrf_Abyx_norms_trfact", size_Abyx_norms_trfact})
        + create_work_item({"geqrf_diag_tmptr", size_diag_tmptr});

    return work_items;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U, typename DevWorkPtr>
rocblas_status rocsolver_geqrf_template(rocblas_handle handle,
                                        const I m,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const I batch_count,
                                        DevWorkPtr dwptr)
{
    //
    // Initialize workspace if empty (an empty workspace means that this is a top level function call)
    //
    ROCSOLVER_INIT_DEVICE_WORKSPACE(
        dwptr,
        rocsolver_geqrf_getWorkItems<BATCHED, STRIDED, T, I, U>(
            handle, m, n, A, shiftA, lda, strideA, ipiv, strideP, batch_count));

    //
    // Get pointers to buffers in device workspace pointer `dwptr`
    //
    // Note: Work items names must match the names defined in the `_getWorkItems` method
    //
    T* scalars = (T*)dwptr->work("geqrf_scalars");
    void* work_workArr = dwptr->work("geqrf_work_workArr");
    T* Abyx_norms_trfact = (T*)dwptr->work("geqrf_Abyx_norms_trfact");
    T* diag_tmptr = (T*)dwptr->work("geqrf_diag_tmptr");
    T** workArr = (T**)dwptr->work("geqrf_workArr");

    //
    // Initialize memory if required
    //
    if(dwptr->size("geqrf_scalars") > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    //
    // Call legacy method
    //
    return rocsolver_geqrf_template<BATCHED, STRIDED>(handle, m, n, A, shiftA, lda, strideA, ipiv,
                                                      strideP, batch_count, scalars, work_workArr,
                                                      Abyx_norms_trfact, diag_tmptr, workArr);
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
auto rocsolver_gelqf_getWorkItems(rocblas_handle handle,
                                  const rocblas_int m,
                                  const rocblas_int n,
                                  U /* A */,
                                  const rocblas_int shiftA,
                                  const rocblas_int lda,
                                  const rocblas_stride strideA,
                                  T* /* ipiv */,
                                  const rocblas_stride strideP,
                                  const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr, size_workArr;
    // extra requirements for calling GEQL2 and to store temporary triangular factor
    size_t size_Abyx_norms_trfact;
    // extra requirements for calling GEQL2 and LARFB
    size_t size_diag_tmptr;
    rocsolver_gelqf_getMemorySize<BATCHED, T>(m, n, batch_count, &size_scalars, &size_work_workArr,
                                              &size_Abyx_norms_trfact, &size_diag_tmptr,
                                              &size_workArr);

    auto work_items = create_work_item({"gelqf_scalars", size_scalars})
        + create_work_item({"gelqf_work_workArr", size_work_workArr})
        + create_work_item({"gelqf_workArr", size_workArr})
        + create_work_item({"gelqf_Abyx_norms_trfact", size_Abyx_norms_trfact})
        + create_work_item({"gelqf_diag_tmptr", size_diag_tmptr});

    return work_items;
}

template <bool BATCHED, bool STRIDED, typename T, typename U, typename DevWorkPtr>
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
                                        DevWorkPtr dwptr)
{
    if(dwptr == nullptr)
    {
        // This auxiliary method is not meant to be called directly
        std::abort();
    }

    T* scalars = (T*)dwptr->work("gelqf_scalars");
    void* work_workArr = dwptr->work("gelqf_work_workArr");
    T* Abyx_norms_trfact = (T*)dwptr->work("gelqf_Abyx_norms_trfact");
    T* diag_tmptr = (T*)dwptr->work("gelqf_diag_tmptr");
    T** workArr = (T**)dwptr->work("gelqf_workArr");

    if(dwptr->size("gelqf_scalars") > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    return rocsolver_gelqf_template<BATCHED, STRIDED>(handle, m, n, A, shiftA, lda, strideA, ipiv,
                                                      strideP, batch_count, scalars, work_workArr,
                                                      Abyx_norms_trfact, diag_tmptr, workArr);
}

template <typename T, typename S, typename W1, typename W2, typename W3>
auto rocsolver_bdsqr_getWorkItems(rocblas_handle handle,
                                  const rocblas_fill uplo,
                                  const rocblas_int n,
                                  const rocblas_int nv,
                                  const rocblas_int nu,
                                  const rocblas_int nc,
                                  S* /* D */,
                                  const rocblas_stride strideD,
                                  S* /* E */,
                                  const rocblas_stride strideE,
                                  W1 /* V */,
                                  const rocblas_int shiftV,
                                  const rocblas_int ldv,
                                  const rocblas_stride strideV,
                                  W2 /* U */,
                                  const rocblas_int shiftU,
                                  const rocblas_int ldu,
                                  const rocblas_stride strideU,
                                  W3 /* C */,
                                  const rocblas_int shiftC,
                                  const rocblas_int ldc,
                                  const rocblas_stride strideC,
                                  rocblas_int* info,
                                  const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size of re-usable workspace
    size_t size_splits_map, size_work, size_completed;
    rocsolver_bdsqr_getMemorySize<S>(n, nv, nu, nc, batch_count, &size_splits_map, &size_work,
                                     &size_completed);

    auto work_items = create_work_item({"bdsqr_splits_map", size_splits_map})
        + create_work_item({"bdsqr_work", size_work})
        + create_work_item({"bdsqr_completed", size_completed});

    return work_items;
}

template <typename T, typename S, typename W1, typename W2, typename W3, typename DevWorkPtr>
rocblas_status rocsolver_bdsqr_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nv,
                                        const rocblas_int nu,
                                        const rocblas_int nc,
                                        S* D,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_stride strideE,
                                        W1 V,
                                        const rocblas_int shiftV,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        W2 U,
                                        const rocblas_int shiftU,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        W3 C,
                                        const rocblas_int shiftC,
                                        const rocblas_int ldc,
                                        const rocblas_stride strideC,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        DevWorkPtr dwptr)
{
    if(dwptr == nullptr)
    {
        // This auxiliary method is not meant to be called directly
        std::abort();
    }

    void* splits_map = dwptr->work("bdsqr_splits_map");
    void* work = dwptr->work("bdsqr_work");
    void* completed = dwptr->work("bdsqr_completed");

    return rocsolver_bdsqr_template<T>(handle, uplo, n, nv, nu, nc, D, strideD, E, strideE, V,
                                       shiftV, ldv, strideV, U, shiftU, ldu, strideU, C, shiftC,
                                       ldc, strideC, info, batch_count, (rocblas_int*)splits_map,
                                       (S*)work, (rocblas_int*)completed);
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
auto rocsolver_gebrd_getWorkItems(
    rocblas_handle handle,
    const rocblas_int m,
    const rocblas_int n,
    U A,
    const rocblas_int shiftA,
    const rocblas_int lda,
    const rocblas_stride strideA,
    S* D,
    const rocblas_stride strideD,
    S* E,
    const rocblas_stride strideE,
    T* tauq,
    const rocblas_stride strideQ,
    T* taup,
    const rocblas_stride strideP,
    /* T* X, */
    /*                                         const rocblas_int shiftX, */
    /*                                         const rocblas_int ldx, */
    /*                                         const rocblas_stride strideX, */
    /*                                         T* Y, */
    /*                                         const rocblas_int shiftY, */
    /*                                         const rocblas_int ldy, */
    /*                                         const rocblas_stride strideY, */
    const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr;
    // extra requirements for calling GEDB2 and LABRD
    size_t size_Abyx_norms;
    // size for temporary resulting orthogonal matrices when calling LABRD
    size_t size_X;
    size_t size_Y;
    rocsolver_gebrd_getMemorySize<false, T>(m, n, batch_count, &size_scalars, &size_work_workArr,
                                            &size_Abyx_norms, &size_X, &size_Y);

    auto work_items = create_work_item({"gebrd_scalars", size_scalars})
        + create_work_item({"gebrd_work_workArr", size_work_workArr})
        + create_work_item({"gebrd_Abyx_norms", size_Abyx_norms})
        + create_work_item({"gebrd_X", size_X}) + create_work_item({"gebrd_Y", size_Y});

    return work_items;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U, typename DevWorkPtr>
rocblas_status rocsolver_gebrd_template(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        S* D,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_stride strideE,
                                        T* tauq,
                                        const rocblas_stride strideQ,
                                        T* taup,
                                        const rocblas_stride strideP,
                                        /* T* X, */
                                        /* const rocblas_int shiftX, */
                                        /* const rocblas_int ldx, */
                                        /* const rocblas_stride strideX, */
                                        /* T* Y, */
                                        /* const rocblas_int shiftY, */
                                        /* const rocblas_int ldy, */
                                        /* const rocblas_stride strideY, */
                                        const rocblas_int batch_count,
                                        DevWorkPtr dwptr)
{
    ROCSOLVER_INIT_DEVICE_WORKSPACE(dwptr,
                                    rocsolver_gebrd_getWorkItems<BATCHED, STRIDED>(
                                        handle, m, n, A, shiftA, lda, strideA, D, strideD, E,
                                        strideE, tauq, strideQ, taup, strideP, batch_count));

    void* scalars = dwptr->work("gebrd_scalars");
    void* work_workArr = dwptr->work("gebrd_work_workArr");
    void* Abyx_norms = dwptr->work("gebrd_Abyx_norms");
    void* X = dwptr->work("gebrd_X");
    void* Y = dwptr->work("gebrd_Y");

    if(dwptr->size("gebrd_scalars") > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    rocblas_int shiftX{0}, ldx{0}, shiftY{0}, ldy{0};
    rocblas_stride strideX{0}, strideY{0};

    if(BATCHED)
    {
        // working with unshifted arrays
        shiftX = 0;
        shiftY = 0;

        // batched execution
        strideX = m * GEBRD_BLOCKSIZE;
        strideY = n * GEBRD_BLOCKSIZE;
    }
    else if(STRIDED)
    {
        // working with unshifted arrays
        shiftX = 0;
        shiftY = 0;

        // strided_batched execution
        strideX = m * GEBRD_BLOCKSIZE;
        strideY = n * GEBRD_BLOCKSIZE;
    }

    return rocsolver_gebrd_template<BATCHED, STRIDED, T>(
        handle, m, n, A, shiftA, lda, strideA, D, strideD, E, strideE, tauq, strideQ, taup, strideP,
        (T*)X, shiftX, m, strideX, (T*)Y, shiftY, n, strideY, batch_count, (T*)scalars,
        work_workArr, (T*)Abyx_norms);
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
auto rocsolver_orgbr_ungbr_getWorkItems(rocblas_handle handle,
                                        const rocblas_storev storev,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U /* A */,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* /* ipiv */,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    // size of re-usable workspace
    size_t size_work;
    // extra requirements for calling ORG2R/UNG2R and LARFB
    size_t size_Abyx_tmptr;
    // size of temporary array for triangular factor
    size_t size_trfact;
    rocsolver_orgbr_ungbr_getMemorySize<false, T>(storev, m, n, k, batch_count, &size_scalars,
                                                  &size_work, &size_Abyx_tmptr, &size_trfact,
                                                  &size_workArr);

    auto work_items = create_work_item({"orgbr_ungbr_scalars", size_scalars})
        + create_work_item({"orgbr_ungbr_work", size_work})
        + create_work_item({"orgbr_ungbr_Abyx_tmptr", size_Abyx_tmptr})
        + create_work_item({"orgbr_ungbr_trfact", size_trfact})
        + create_work_item({"orgbr_ungbr_workArr", size_workArr});

    return work_items;
}

template <bool BATCHED, bool STRIDED, typename T, typename U, typename DevWorkPtr>
rocblas_status rocsolver_orgbr_ungbr_template(rocblas_handle handle,
                                              const rocblas_storev storev,
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
                                              DevWorkPtr dwptr)
{
    if(dwptr == nullptr)
    {
        // This auxiliary method is not meant to be called directly
        std::abort();
    }

    T* scalars = (T*)dwptr->work("orgbr_ungbr_scalars");
    T* work = (T*)dwptr->work("orgbr_ungbr_work");
    T* Abyx_tmptr = (T*)dwptr->work("orgbr_ungbr_Abyx_tmptr");
    T* trfact = (T*)dwptr->work("orgbr_ungbr_trfact");
    T** workArr = (T**)dwptr->work("orgbr_ungbr_workArr");

    if(dwptr->size("orgbr_ungbr_scalars") > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    return rocsolver_orgbr_ungbr_template<BATCHED, STRIDED>(
        handle, storev, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, batch_count, scalars, work,
        Abyx_tmptr, trfact, workArr);
}

template <bool BATCHED, bool STRIDED, typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
auto rocsolver_ormbr_unmbr_getWorkItems(rocblas_handle handle,
                                        const rocblas_storev storev,
                                        const rocblas_side side,
                                        const rocblas_operation trans,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U /* A */,
                                        /* const rocblas_int shiftA, */
                                        const rocblas_int lda,
                                        /* const rocblas_stride strideA, */
                                        T* /* ipiv */,
                                        /* const rocblas_stride strideP, */
                                        U /* C */,
                                        /* const rocblas_int shiftC, */
                                        const rocblas_int ldc,
                                        /* const rocblas_stride strideC, */
                                        const rocblas_int batch_count)
{
    // memory workspace sizes:
    // requirements for calling ORMQR/UNMQR or ORMLQ/UNMLQ
    size_t size_scalars;
    size_t size_AbyxORwork, size_diagORtmptr;
    size_t size_trfact;
    size_t size_workArr;
    rocsolver_ormbr_unmbr_getMemorySize<false, T>(storev, side, m, n, k, batch_count, &size_scalars,
                                                  &size_AbyxORwork, &size_diagORtmptr, &size_trfact,
                                                  &size_workArr);

    auto work_items = create_work_item({"ormbr_unmbr_scalars", size_scalars})
        + create_work_item({"ormbr_unmbr_AbyxORwork", size_AbyxORwork})
        + create_work_item({"ormbr_unmbr_diagORtmptr", size_diagORtmptr})
        + create_work_item({"ormbr_unmbr_trfact", size_trfact})
        + create_work_item({"ormbr_unmbr_workArr", size_workArr});

    return work_items;
}

template <bool BATCHED,
          bool STRIDED,
          typename T,
          typename U,
          bool COMPLEX = rocblas_is_complex<T>,
          typename DevWorkPtr = rocsolver_device_workspace_ptr_t>
auto rocsolver_ormbr_unmbr_template(rocblas_handle handle,
                                    const rocblas_storev storev,
                                    const rocblas_side side,
                                    const rocblas_operation trans,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    const rocblas_int k,
                                    U A,
                                    rocblas_int shiftA,
                                    const rocblas_int lda,
                                    rocblas_stride strideA,
                                    T* ipiv,
                                    rocblas_stride strideP,
                                    U C,
                                    rocblas_int shiftC,
                                    const rocblas_int ldc,
                                    rocblas_stride strideC,
                                    rocblas_int batch_count,
                                    DevWorkPtr dwptr)
{
    if(dwptr == nullptr)
    {
        // This auxiliary method is not meant to be called directly
        std::abort();
    }

    T* scalars = (T*)dwptr->work("ormbr_unmbr_scalars");
    T* AbyxORwork = (T*)dwptr->work("ormbr_unmbr_AbyxORwork");
    T* diagORtmptr = (T*)dwptr->work("ormbr_unmbr_diagORtmptr");
    T* trfact = (T*)dwptr->work("ormbr_unmbr_trfact");
    T** workArr = (T**)dwptr->work("ormbr_unmbr_workArr");

    if(dwptr->size("ormbr_unmbr_scalars") > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    // working with unshifted arrays
    shiftA = 0;
    shiftC = 0;

    // normal (non-batched non-strided) execution
    strideA = 0;
    strideP = 0;
    strideC = 0;
    batch_count = 1;

    return rocsolver_ormbr_unmbr_template<BATCHED, STRIDED, T>(
        handle, storev, side, trans, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, C, shiftC, ldc,
        strideC, batch_count, (T*)scalars, (T*)AbyxORwork, (T*)diagORtmptr, (T*)trfact, (T**)workArr);
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
auto rocsolver_orgqr_ungqr_getWorkItems(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    // size of re-usable workspace
    size_t size_work;
    // extra requirements for calling ORG2R/UNG2R and LARFB
    size_t size_Abyx_tmptr;
    // size of temporary array for triangular factor
    size_t size_trfact;
    rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(m, n, k, batch_count, &size_scalars, &size_work,
                                                    &size_Abyx_tmptr, &size_trfact, &size_workArr);

    auto work_items = create_work_item({"orgqr_ungqr_scalars", size_scalars})
        + create_work_item({"orgqr_ungqr_workArr", size_workArr})
        + create_work_item({"orgqr_ungqr_work", size_work})
        + create_work_item({"orgqr_ungqr_Abyx_tmptr", size_Abyx_tmptr})
        + create_work_item({"orgqr_ungqr_trfact", size_trfact});

    return work_items;
}

template <bool BATCHED, bool STRIDED, typename T, typename U, typename DevWorkPtr>
rocblas_status rocsolver_orgqr_ungqr_template(rocblas_handle handle,
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
                                              DevWorkPtr dwptr)
{
    if(dwptr == nullptr)
    {
        // This auxiliary method is not meant to be called directly
        std::abort();
    }

    T* scalars = (T*)dwptr->work("orgqr_ungqr_scalars");
    T* work = (T*)dwptr->work("orgqr_ungqr_work");
    T* Abyx_tmptr = (T*)dwptr->work("orgqr_ungqr_Abyx_tmptr");
    T* trfact = (T*)dwptr->work("orgqr_ungqr_trfact");
    T** workArr = (T**)dwptr->work("orgqr_ungqr_workArr");

    if(dwptr->size("orgqr_ungqr_scalars") > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    return rocsolver_orgqr_ungqr_template<BATCHED, STRIDED>(handle, m, n, k, A, shiftA, lda, strideA,
                                                            ipiv, strideP, batch_count, scalars,
                                                            work, Abyx_tmptr, trfact, workArr);
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
auto rocsolver_orglq_unglq_getWorkItems(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    // size of re-usable workspace
    size_t size_work;
    // extra requirements for calling ORGL2/UNGL2 and LARFB
    size_t size_Abyx_tmptr;
    // size of temporary array for triangular factor
    size_t size_trfact;
    rocsolver_orglq_unglq_getMemorySize<BATCHED, T>(m, n, k, batch_count, &size_scalars, &size_work,
                                                    &size_Abyx_tmptr, &size_trfact, &size_workArr);

    auto work_items = create_work_item({"orglq_unglq_scalars", size_scalars})
        + create_work_item({"orglq_unglq_workArr", size_workArr})
        + create_work_item({"orglq_unglq_work", size_work})
        + create_work_item({"orglq_unglq_Abyx_tmptr", size_Abyx_tmptr})
        + create_work_item({"orglq_unglq_trfact", size_trfact});

    return work_items;
}

template <bool BATCHED, bool STRIDED, typename T, typename U, typename DevWorkPtr>
rocblas_status rocsolver_orglq_unglq_template(rocblas_handle handle,
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
                                              DevWorkPtr dwptr)
{
    if(dwptr == nullptr)
    {
        // This auxiliary method is not meant to be called directly
        std::abort();
    }

    T* scalars = (T*)dwptr->work("orglq_unglq_scalars");
    T* work = (T*)dwptr->work("orglq_unglq_work");
    T* Abyx_tmptr = (T*)dwptr->work("orglq_unglq_Abyx_tmptr");
    T* trfact = (T*)dwptr->work("orglq_unglq_trfact");
    T** workArr = (T**)dwptr->work("orglq_unglq_workArr");

    if(dwptr->size("orglq_unglq_scalars") > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    return rocsolver_orglq_unglq_template<BATCHED, STRIDED>(handle, m, n, k, A, shiftA, lda, strideA,
                                                            ipiv, strideP, batch_count, scalars,
                                                            work, Abyx_tmptr, trfact, workArr);
}

// CURRENTLY NOT IN USE
template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
auto rocsolver_stedc_getWorkItems(rocblas_handle handle,
                                  const rocblas_evect evect,
                                  const rocblas_int n,
                                  S* /* D */,
                                  const rocblas_int shiftD,
                                  const rocblas_stride strideD,
                                  S* /* E */,
                                  const rocblas_int shiftE,
                                  const rocblas_stride strideE,
                                  U /* C */,
                                  const rocblas_int shiftC,
                                  const rocblas_int ldc,
                                  const rocblas_stride strideC,
                                  rocblas_int* /* info */,
                                  const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for lasrt stack/stedc workspace
    size_t size_work_stack;
    // size for temporary computations
    size_t size_tempvect, size_tempgemm;
    // size for pointers to workspace (batched case)
    size_t size_workArr;
    // size for vector with positions of split blocks
    size_t size_splits_map;
    // size for temporary diagonal and z vectors.
    size_t size_tmpz;
    rocsolver_stedc_getMemorySize<BATCHED, T, S>(evect, n, batch_count, &size_work_stack,
                                                 &size_tempvect, &size_tempgemm, &size_tmpz,
                                                 &size_splits_map, &size_workArr);

    auto work_items = create_work_item({"stedc_work_stack", size_work_stack})
        + create_work_item({"stedc_tempvect", size_tempvect})
        + create_work_item({"stedc_tempgemm", size_tempgemm})
        + create_work_item({"stedc_workArr", size_workArr})
        + create_work_item({"stedc_splits_map", size_splits_map})
        + create_work_item({"stedc_tmpz", size_tmpz});

    return work_items;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename W>
auto rocsolver_syevd_heevd_getWorkItems(rocblas_handle handle,
                                        const rocblas_evect evect,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        W A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        S* D,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_stride strideE,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspaces
    size_t size_work1;
    size_t size_work2;
    size_t size_work3;
    size_t size_tmptau_W;
    // extra space for call stedc
    size_t size_splits, size_tmpz;
    // size of array of pointers (only for batched case)
    size_t size_workArr;
    // size for temporary householder scalars
    size_t size_tau;

    rocsolver_syevd_heevd_getMemorySize<BATCHED, T, S>(
        handle, evect, uplo, n, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
        &size_tmpz, &size_splits, &size_tmptau_W, &size_tau, &size_workArr);

    auto work_items = create_work_item({"syevd_heevd_scalars", size_scalars})
        + create_work_item({"syevd_heevd_work1", size_work1})
        + create_work_item({"syevd_heevd_work2", size_work2})
        + create_work_item({"syevd_heevd_work3", size_work3})
        + create_work_item({"syevd_heevd_tmptau_W", size_tmptau_W})
        + create_work_item({"syevd_heevd_splits", size_splits})
        + create_work_item({"syevd_heevd_tmpz", size_tmpz})
        + create_work_item({"syevd_heevd_workArr", size_workArr})
        + create_work_item({"syevd_heevd_tau", size_tau});

    return work_items;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename W, typename DevWorkPtr>
rocblas_status rocsolver_syevd_heevd_template(rocblas_handle handle,
                                              const rocblas_evect evect,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              W A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              S* D,
                                              const rocblas_stride strideD,
                                              S* E,
                                              const rocblas_stride strideE,
                                              rocblas_int* info,
                                              const rocblas_int batch_count,
                                              DevWorkPtr dwptr)
{
    if(dwptr == nullptr)
    {
        // This auxiliary method is not meant to be called directly
        std::abort();
    }

    T* scalars = (T*)dwptr->work("syevd_heevd_scalars");
    void* work1 = dwptr->work("syevd_heevd_work1");
    void* work2 = dwptr->work("syevd_heevd_work2");
    void* work3 = dwptr->work("syevd_heevd_work3");
    S* tmpz = (S*)dwptr->work("syevd_heevd_tmpz");
    rocblas_int* splits = (rocblas_int*)dwptr->work("syevd_heevd_splits");
    T* tmptau_W = (T*)dwptr->work("syevd_heevd_tmptau_W");
    T* tau = (T*)dwptr->work("syevd_heevd_tau");
    T** workArr = (T**)dwptr->work("syevd_heevd_workArr");

    if(dwptr->size("syevd_heevd_scalars") > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    return rocsolver_syevd_heevd_template<BATCHED, STRIDED>(
        handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, info, batch_count,
        scalars, work1, work2, work3, tmpz, splits, tmptau_W, tau, workArr);
}

ROCSOLVER_END_NAMESPACE
