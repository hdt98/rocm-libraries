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

template <typename T, typename I, typename Istride = rocblas_stride, typename S = decltype(std::real(T{}))>
rocblas_status rocsolver_cholqr_batched_impl(rocblas_handle handle,
                                             const I m,
                                             const I n,

                                             T** const A,
                                             const I lda,

                                             T* const R,
                                             const I ldr,
                                             const Istride strideR,

                                             S* const sigma,
                                             rocsolver_cholqr_algo const algo,

                                             I* const info,
                                             const I batch_count)
{
    ROCSOLVER_ENTER_TOP("cholqr_batched", "-m", m, "-n", n, "--lda", lda, "--strideR", strideR,
                        "--algo", static_cast<I>(algo), "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // working with unshifted arrays
    Istride const shiftA = 0;
    Istride const shiftR = 0;
    Istride const strideA = lda * n;

    // -----------------
    // argument checking
    // -----------------
    {
        rocblas_status st = rocsolver_cholqr_batched_argCheck<T>(handle, m, n,

                                                                 A, lda,

                                                                 R, ldr, strideR,

                                                                 sigma, algo, info, batch_count);

        bool const isok = (st == rocblas_status_continue) || (st == rocblas_status_success);
        if(!isok)
        {
            return st;
        }
    }

    // memory workspace sizes:

    size_t size_work = 0;
    rocsolver_cholqr_getMemorySize<T, I>(m, n, batch_count, algo, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;

    void* const work = (void*)mem[0];

    // execution

    auto const istat = rocsolver_cholqr_template<T, I, rocblas_stride>(handle, m, n,

                                                                       A, shiftA, lda, strideA,

                                                                       R, shiftR, ldr, strideR,

                                                                       sigma, algo,

                                                                       info,

                                                                       batch_count,

                                                                       work, size_work);

    return (istat);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ccholqr_batched(rocblas_handle handle,
                                         rocblas_int const m,
                                         rocblas_int const n,

                                         rocblas_float_complex** const A,
                                         rocblas_int const lda,

                                         rocblas_float_complex* const R,
                                         rocblas_int const ldr,
                                         rocblas_stride const strideR,

                                         float* const sigma,
                                         rocsolver_cholqr_algo const algo,

                                         rocblas_int* const info,
                                         rocblas_int const batch_count)
{
    return rocsolver::rocsolver_cholqr_batched_impl<rocblas_float_complex>(handle, m, n,

                                                                           A, lda,

                                                                           R, ldr, strideR,

                                                                           sigma, algo,

                                                                           info,

                                                                           batch_count);
}

rocblas_status rocsolver_zcholqr_batched(rocblas_handle handle,
                                         rocblas_int const m,
                                         rocblas_int const n,

                                         rocblas_double_complex** const A,
                                         rocblas_int const lda,

                                         rocblas_double_complex* const R,
                                         rocblas_int const ldr,
                                         rocblas_stride const strideR,

                                         double* const sigma,
                                         rocsolver_cholqr_algo const algo,

                                         rocblas_int* const info,
                                         rocblas_int const batch_count)
{
    return rocsolver::rocsolver_cholqr_batched_impl<rocblas_double_complex>(handle, m, n,

                                                                            A, lda,

                                                                            R, ldr, strideR,

                                                                            sigma, algo,

                                                                            info,

                                                                            batch_count);
}

rocblas_status rocsolver_dcholqr_batched(rocblas_handle handle,
                                         rocblas_int const m,
                                         rocblas_int const n,

                                         double** const A,
                                         rocblas_int const lda,

                                         double* const R,
                                         rocblas_int const ldr,
                                         rocblas_stride const strideR,

                                         double* const sigma,
                                         rocsolver_cholqr_algo const algo,

                                         rocblas_int* const info,
                                         rocblas_int const batch_count)
{
    return rocsolver::rocsolver_cholqr_batched_impl<double>(handle, m, n,

                                                            A, lda,

                                                            R, ldr, strideR,

                                                            sigma, algo,

                                                            info,

                                                            batch_count);
}

rocblas_status rocsolver_scholqr_batched(rocblas_handle handle,
                                         rocblas_int const m,
                                         rocblas_int const n,

                                         float** const A,
                                         rocblas_int const lda,

                                         float* const R,
                                         rocblas_int const ldr,
                                         rocblas_stride const strideR,

                                         float* const sigma,
                                         rocsolver_cholqr_algo const algo,

                                         rocblas_int* const info,
                                         rocblas_int const batch_count)
{
    return rocsolver::rocsolver_cholqr_batched_impl<float>(handle, m, n,

                                                           A, lda,

                                                           R, ldr, strideR,

                                                           sigma, algo,

                                                           info,

                                                           batch_count);
}

rocblas_status rocsolver_zcholqr_batched_64(rocblas_handle handle,
                                            int64_t const m,
                                            int64_t const n,

                                            rocblas_double_complex** const A,
                                            int64_t const lda,

                                            rocblas_double_complex* const R,
                                            int64_t const ldr,
                                            rocblas_stride const strideR,

                                            double* const sigma,
                                            rocsolver_cholqr_algo const algo,

                                            int64_t* const info,

                                            int64_t const batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_cholqr_batched_impl<rocblas_double_complex>(handle, m, n,

                                                                            A, lda,

                                                                            R, ldr, strideR,

                                                                            sigma, algo,

                                                                            info, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_ccholqr_batched_64(rocblas_handle handle,
                                            int64_t const m,
                                            int64_t const n,

                                            rocblas_float_complex** const A,
                                            int64_t const lda,

                                            rocblas_float_complex* const R,
                                            int64_t const ldr,
                                            rocblas_stride const strideR,

                                            float* const sigma,
                                            rocsolver_cholqr_algo const algo,

                                            int64_t* const info,

                                            int64_t const batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_cholqr_batched_impl<rocblas_float_complex>(handle, m, n,

                                                                           A, lda,

                                                                           R, ldr, strideR,

                                                                           sigma, algo,

                                                                           info, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dcholqr_batched_64(rocblas_handle handle,
                                            int64_t const m,
                                            int64_t const n,

                                            double** const A,
                                            int64_t const lda,

                                            double* const R,
                                            int64_t const ldr,
                                            rocblas_stride const strideR,

                                            double* const sigma,
                                            rocsolver_cholqr_algo const algo,

                                            int64_t* const info,

                                            int64_t const batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_cholqr_batched_impl<double>(handle, m, n,

                                                            A, lda,

                                                            R, ldr, strideR,

                                                            sigma, algo,

                                                            info, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_scholqr_batched_64(rocblas_handle handle,
                                            int64_t const m,
                                            int64_t const n,

                                            float** const A,
                                            int64_t const lda,

                                            float* const R,
                                            int64_t const ldr,
                                            rocblas_stride const strideR,

                                            float* const sigma,
                                            rocsolver_cholqr_algo const algo,

                                            int64_t* const info,

                                            int64_t const batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_cholqr_batched_impl<float>(handle, m, n,

                                                           A, lda,

                                                           R, ldr, strideR,

                                                           sigma, algo,

                                                           info, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
