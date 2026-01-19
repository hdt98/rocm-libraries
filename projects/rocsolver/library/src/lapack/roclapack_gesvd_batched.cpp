/* **************************************************************************
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

#include "roclapack_gesvd.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_batched_impl(rocblas_handle handle,
                                            const rocblas_svect left_svect,
                                            const rocblas_svect right_svect,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            W A,
                                            const rocblas_int lda,
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
                                            const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("gesvd_batched", "--left_svect", left_svect, "--right_svect", right_svect,
                        "-m", m, "-n", n, "--lda", lda, "--strideS", strideS, "--ldu", ldu,
                        "--strideU", strideU, "--ldv", ldv, "--strideV", strideV, "--strideE",
                        strideE, "--fast_alg", fast_alg, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gesvd_argCheck(handle, left_svect, right_svect, m, n, A, lda, S,
                                                 U, ldu, V, ldv, E, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // batched execution
    rocblas_stride strideA = 0;

    // execution
    return rocsolver_gesvd_template<true, false, T>(
        handle, left_svect, right_svect, m, n, A, shiftA, lda, strideA, S, strideS, U, ldu, strideU,
        V, ldv, strideV, E, strideE, fast_alg, info, batch_count, nullptr);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgesvd_batched(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        float* const A[],
                                        const rocblas_int lda,
                                        float* S,
                                        const rocblas_stride strideS,
                                        float* U,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        float* V,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        float* E,
                                        const rocblas_stride strideE,
                                        const rocblas_workmode fast_alg,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gesvd_batched_impl<float>(
        handle, left_svect, right_svect, m, n, A, lda, S, strideS, U, ldu, strideU, V, ldv, strideV,
        E, strideE, fast_alg, info, batch_count);
}

rocblas_status rocsolver_dgesvd_batched(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        double* const A[],
                                        const rocblas_int lda,
                                        double* S,
                                        const rocblas_stride strideS,
                                        double* U,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        double* V,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        double* E,
                                        const rocblas_stride strideE,
                                        const rocblas_workmode fast_alg,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gesvd_batched_impl<double>(
        handle, left_svect, right_svect, m, n, A, lda, S, strideS, U, ldu, strideU, V, ldv, strideV,
        E, strideE, fast_alg, info, batch_count);
}

rocblas_status rocsolver_cgesvd_batched(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        rocblas_float_complex* const A[],
                                        const rocblas_int lda,
                                        float* S,
                                        const rocblas_stride strideS,
                                        rocblas_float_complex* U,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        rocblas_float_complex* V,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        float* E,
                                        const rocblas_stride strideE,
                                        const rocblas_workmode fast_alg,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gesvd_batched_impl<rocblas_float_complex>(
        handle, left_svect, right_svect, m, n, A, lda, S, strideS, U, ldu, strideU, V, ldv, strideV,
        E, strideE, fast_alg, info, batch_count);
}

rocblas_status rocsolver_zgesvd_batched(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        rocblas_double_complex* const A[],
                                        const rocblas_int lda,
                                        double* S,
                                        const rocblas_stride strideS,
                                        rocblas_double_complex* U,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        rocblas_double_complex* V,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        double* E,
                                        const rocblas_stride strideE,
                                        const rocblas_workmode fast_alg,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gesvd_batched_impl<rocblas_double_complex>(
        handle, left_svect, right_svect, m, n, A, lda, S, strideS, U, ldu, strideU, V, ldv, strideV,
        E, strideE, fast_alg, info, batch_count);
}

} // extern C
