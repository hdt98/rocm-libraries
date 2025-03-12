/* **************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "roclapack_gesdd.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename SS, typename W>
rocblas_status rocsolver_gesdd_impl(rocblas_handle handle,
                                    const rocblas_svect left_svect,
                                    const rocblas_svect right_svect,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    W A,
                                    const rocblas_int lda,
                                    SS* S,
                                    T* U,
                                    const rocblas_int ldu,
                                    T* V,
                                    const rocblas_int ldv,
                                    rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("gesdd", "--left_svect", left_svect, "--right_svect", right_svect, "-m", m,
                        "-n", n, "--lda", lda, "--ldu", ldu, "--ldv", ldv);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gesdd_argCheck(handle, left_svect, right_svect, m, n, A, lda, S,
                                                 U, ldu, V, ldv, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideS = 0;
    rocblas_stride strideU = 0;
    rocblas_stride strideV = 0;
    rocblas_int batch_count = 1;

    auto gesdd_work_items = rocsolver_gesdd_getWorkItems<false, false, T, SS, W>(
        handle, left_svect, right_svect, m, n, nullptr /* W A */, lda, strideA, nullptr /* S */,
        strideS, nullptr /* T* U */, ldu, strideU, nullptr /* T* V */, ldv, strideV,
        nullptr /* rocblas_int* info */, batch_count);

    auto [device_work_ptr, status] = device_workspace::Get(handle, gesdd_work_items);
    if(rocblas_is_device_memory_size_query(handle))
    {
        return status;
    }

    if(!device_work_ptr || (status != rocblas_status_success))
    {
        return status;
    }

    // execution
    return rocsolver_gesdd_template<false, false, T>(
        handle, left_svect, right_svect, m, n, A, shiftA, lda, strideA, S, strideS, U, ldu, strideU,
        V, ldv, strideV, info, batch_count, device_work_ptr);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgesdd(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* S,
                                float* U,
                                const rocblas_int ldu,
                                float* V,
                                const rocblas_int ldv,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_gesdd_impl<float>(handle, left_svect, right_svect, m, n, A, lda, S,
                                                  U, ldu, V, ldv, info);
}

rocblas_status rocsolver_dgesdd(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* S,
                                double* U,
                                const rocblas_int ldu,
                                double* V,
                                const rocblas_int ldv,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_gesdd_impl<double>(handle, left_svect, right_svect, m, n, A, lda, S,
                                                   U, ldu, V, ldv, info);
}

rocblas_status rocsolver_cgesdd(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* S,
                                rocblas_float_complex* U,
                                const rocblas_int ldu,
                                rocblas_float_complex* V,
                                const rocblas_int ldv,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_gesdd_impl<rocblas_float_complex>(
        handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, info);
}

rocblas_status rocsolver_zgesdd(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* S,
                                rocblas_double_complex* U,
                                const rocblas_int ldu,
                                rocblas_double_complex* V,
                                const rocblas_int ldv,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_gesdd_impl<rocblas_double_complex>(
        handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, info);
}

} // extern C
