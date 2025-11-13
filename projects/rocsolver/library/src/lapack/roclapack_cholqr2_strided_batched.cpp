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

#include "roclapack_cholqr.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename Istride = rocblas_stride>
rocblas_status rocsolver_cholqr2_strided_batched_impl(rocblas_handle handle,
                                                      const I m,
                                                      const I n,

                                                      T* const A,
                                                      const I lda,
                                                      const Istride strideA,

                                                      T* const R,
                                                      const I ldr,
                                                      const Istride strideR,

                                                      I* const info,
                                                      const I batch_count)
{
    ROCSOLVER_ENTER_TOP("cholqr2_strided_batched", "-m", m, "-n", n, "--lda", lda, "--strideA",
                        strideA, "--strideR", strideR, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_cholqr2_strided_batched_argCheck<T>(handle, m, n, lda, ldr, A, R, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    Istride const shiftA = 0;
    Istride const shiftR = 0;

    // memory workspace sizes:

    size_t size_lwork = 0;
    rocsolver_cholqr2_getMemorySize<T, I>(m, n, batch_count, &size_lwork);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_lwork);

    rocblas_device_malloc mem(handle, size_lwork);

    if(!mem)
        return rocblas_status_memory_error;

    void* const work = (void*)mem[0];

    // execution

    auto const istat = rocsolver_cholqr2_template<T, I, rocblas_stride>(handle, m, n,

                                                                        A, shiftA, lda, strideA,

                                                                        R, shiftR, ldr, strideR,

                                                                        batch_count,

                                                                        info,

                                                                        work, size_lwork);

    return (istat);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_scholqr2_strided_batched(rocblas_handle handle,
                                                  const rocblas_int m,
                                                  const rocblas_int n,

                                                  float* A,
                                                  const rocblas_int lda,
                                                  const rocblas_stride strideA,

                                                  float* R,
                                                  const rocblas_int ldr,
                                                  const rocblas_stride strideR,

                                                  rocblas_int* info,
                                                  const rocblas_int batch_count)
{
    return rocsolver::rocsolver_cholqr2_strided_batched_impl<float>(handle, m, n,

                                                                    A, lda, strideA,

                                                                    R, ldr, strideR,

                                                                    info,

                                                                    batch_count);
}

rocblas_status rocsolver_dcholqr2_strided_batched(rocblas_handle handle,
                                                  const rocblas_int m,
                                                  const rocblas_int n,

                                                  double* A,
                                                  const rocblas_int lda,
                                                  const rocblas_stride strideA,

                                                  double* R,
                                                  const rocblas_int ldr,
                                                  const rocblas_stride strideR,

                                                  rocblas_int* info,
                                                  const rocblas_int batch_count)
{
    return rocsolver::rocsolver_cholqr2_strided_batched_impl<double>(handle, m, n,

                                                                     A, lda, strideA,

                                                                     R, ldr, strideR,

                                                                     info,

                                                                     batch_count);
}

rocblas_status rocsolver_ccholqr2_strided_batched(rocblas_handle handle,
                                                  const rocblas_int m,
                                                  const rocblas_int n,

                                                  rocblas_float_complex* A,
                                                  const rocblas_int lda,
                                                  const rocblas_stride strideA,

                                                  rocblas_float_complex* R,
                                                  const rocblas_int ldr,
                                                  const rocblas_stride strideR,

                                                  rocblas_int* info,
                                                  const rocblas_int batch_count)
{
    return rocsolver::rocsolver_cholqr2_strided_batched_impl<rocblas_float_complex>(handle, m, n,

                                                                                    A, lda, strideA,

                                                                                    R, ldr, strideR,

                                                                                    info,

                                                                                    batch_count);
}

rocblas_status rocsolver_zcholqr2_strided_batched(rocblas_handle handle,
                                                  const rocblas_int m,
                                                  const rocblas_int n,

                                                  rocblas_double_complex* A,
                                                  const rocblas_int lda,
                                                  const rocblas_stride strideA,

                                                  rocblas_double_complex* R,
                                                  const rocblas_int ldr,
                                                  const rocblas_stride strideR,

                                                  rocblas_int* info,
                                                  const rocblas_int batch_count)
{
    return rocsolver::rocsolver_cholqr2_strided_batched_impl<rocblas_double_complex>(handle, m, n,

                                                                                     A, lda, strideA,

                                                                                     R, ldr, strideR,

                                                                                     info,

                                                                                     batch_count);
}

rocblas_status rocsolver_scholqr2_strided_batched_64(rocblas_handle handle,
                                                     const int64_t m,
                                                     const int64_t n,

                                                     float* A,
                                                     const int64_t lda,
                                                     const rocblas_stride strideA,

                                                     float* R,
                                                     const int64_t ldr,
                                                     const rocblas_stride strideR,

                                                     int64_t* info,

                                                     const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_cholqr2_strided_batched_impl<float>(handle, m, n,

                                                                    A, lda, strideA,

                                                                    R, ldr, strideR,

                                                                    info, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dcholqr2_strided_batched_64(rocblas_handle handle,
                                                     const int64_t m,
                                                     const int64_t n,

                                                     double* A,
                                                     const int64_t lda,
                                                     const rocblas_stride strideA,

                                                     double* R,
                                                     const int64_t ldr,
                                                     const rocblas_stride strideR,

                                                     int64_t* info,

                                                     const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_cholqr2_strided_batched_impl<double>(handle, m, n,

                                                                     A, lda, strideA,

                                                                     R, ldr, strideR,

                                                                     info, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_ccholqr2_strided_batched_64(rocblas_handle handle,
                                                     const int64_t m,
                                                     const int64_t n,

                                                     rocblas_float_complex* A,
                                                     const int64_t lda,
                                                     const rocblas_stride strideA,

                                                     rocblas_float_complex* R,
                                                     const int64_t ldr,
                                                     const rocblas_stride strideR,

                                                     int64_t* info,

                                                     const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_cholqr2_strided_batched_impl<rocblas_float_complex>(handle, m, n,

                                                                                    A, lda, strideA,

                                                                                    R, ldr, strideR,

                                                                                    info,
                                                                                    batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zcholqr2_strided_batched_64(rocblas_handle handle,
                                                     const int64_t m,
                                                     const int64_t n,

                                                     rocblas_double_complex* A,
                                                     const int64_t lda,
                                                     const rocblas_stride strideA,

                                                     rocblas_double_complex* R,
                                                     const int64_t ldr,
                                                     const rocblas_stride strideR,

                                                     int64_t* info,

                                                     const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_cholqr2_strided_batched_impl<rocblas_double_complex>(handle, m, n,

                                                                                     A, lda, strideA,

                                                                                     R, ldr, strideR,

                                                                                     info,
                                                                                     batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
