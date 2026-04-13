/* **************************************************************************
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "roclapack_geev.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/** Implementation for real types (wr + wi) **/
template <typename T>
rocblas_status rocsolver_geev_impl(rocblas_handle handle,
                                   const rocblas_evect jobvl,
                                   const rocblas_evect jobvr,
                                   const rocblas_int n,
                                   T* A,
                                   const rocblas_int lda,
                                   T* wr,
                                   T* wi,
                                   T* VL,
                                   const rocblas_int ldvl,
                                   T* VR,
                                   const rocblas_int ldvr,
                                   rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("geev", "--jobvl", jobvl, "--jobvr", jobvr, "-n", n, "--lda", lda,
                        "--ldvl", ldvl, "--ldvr", ldvr);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_geev_argCheck(handle, jobvl, jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr, info);
    if(st != rocblas_status_continue)
        return st;

    // this routine does not use device workspace managed by rocblas;
    // all workspace is allocated on the host inside the template
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, 0);

    // execution
    return rocsolver_geev_template<T>(handle, jobvl, jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr,
                                      info);
}

/** Implementation for complex types (w + rwork) **/
template <typename T, typename S>
rocblas_status rocsolver_geev_impl(rocblas_handle handle,
                                   const rocblas_evect jobvl,
                                   const rocblas_evect jobvr,
                                   const rocblas_int n,
                                   T* A,
                                   const rocblas_int lda,
                                   T* w,
                                   T* VL,
                                   const rocblas_int ldvl,
                                   T* VR,
                                   const rocblas_int ldvr,
                                   S* rwork,
                                   rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("geev", "--jobvl", jobvl, "--jobvr", jobvr, "-n", n, "--lda", lda,
                        "--ldvl", ldvl, "--ldvr", ldvr);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_geev_argCheck(handle, jobvl, jobvr, n, A, lda, w, (T*)nullptr, VL, ldvl, VR,
                                  ldvr, info);
    if(st != rocblas_status_continue)
        return st;

    // this routine does not use device workspace managed by rocblas;
    // all workspace is allocated on the host inside the template
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, 0);

    // execution
    return rocsolver_geev_template<T, S>(handle, jobvl, jobvr, n, A, lda, w, VL, ldvl, VR, ldvr,
                                         rwork, info);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgeev(rocblas_handle handle,
                               const rocblas_evect jobvl,
                               const rocblas_evect jobvr,
                               const rocblas_int n,
                               float* A,
                               const rocblas_int lda,
                               float* wr,
                               float* wi,
                               float* VL,
                               const rocblas_int ldvl,
                               float* VR,
                               const rocblas_int ldvr,
                               rocblas_int* info)
{
    return rocsolver::rocsolver_geev_impl<float>(handle, jobvl, jobvr, n, A, lda, wr, wi, VL, ldvl,
                                                 VR, ldvr, info);
}

rocblas_status rocsolver_dgeev(rocblas_handle handle,
                               const rocblas_evect jobvl,
                               const rocblas_evect jobvr,
                               const rocblas_int n,
                               double* A,
                               const rocblas_int lda,
                               double* wr,
                               double* wi,
                               double* VL,
                               const rocblas_int ldvl,
                               double* VR,
                               const rocblas_int ldvr,
                               rocblas_int* info)
{
    return rocsolver::rocsolver_geev_impl<double>(handle, jobvl, jobvr, n, A, lda, wr, wi, VL,
                                                  ldvl, VR, ldvr, info);
}

rocblas_status rocsolver_cgeev(rocblas_handle handle,
                               const rocblas_evect jobvl,
                               const rocblas_evect jobvr,
                               const rocblas_int n,
                               rocblas_float_complex* A,
                               const rocblas_int lda,
                               rocblas_float_complex* w,
                               rocblas_float_complex* VL,
                               const rocblas_int ldvl,
                               rocblas_float_complex* VR,
                               const rocblas_int ldvr,
                               float* rwork,
                               rocblas_int* info)
{
    return rocsolver::rocsolver_geev_impl<rocblas_float_complex, float>(
        handle, jobvl, jobvr, n, A, lda, w, VL, ldvl, VR, ldvr, rwork, info);
}

rocblas_status rocsolver_zgeev(rocblas_handle handle,
                               const rocblas_evect jobvl,
                               const rocblas_evect jobvr,
                               const rocblas_int n,
                               rocblas_double_complex* A,
                               const rocblas_int lda,
                               rocblas_double_complex* w,
                               rocblas_double_complex* VL,
                               const rocblas_int ldvl,
                               rocblas_double_complex* VR,
                               const rocblas_int ldvr,
                               double* rwork,
                               rocblas_int* info)
{
    return rocsolver::rocsolver_geev_impl<rocblas_double_complex, double>(
        handle, jobvl, jobvr, n, A, lda, w, VL, ldvl, VR, ldvr, rwork, info);
}

} // extern C
