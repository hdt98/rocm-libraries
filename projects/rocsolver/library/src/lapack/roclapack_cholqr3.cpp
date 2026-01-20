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

#include "roclapack_cholqr.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I>
rocblas_status rocsolver_cholqr3_impl(rocblas_handle handle,
                                      const I m,
                                      const I n,
                                      T* A,
                                      const I lda,
                                      T* R,
                                      const I ldr,
                                      I* info)
{
    using S = decltype(std::real(T{}));
    // normal (non-batched non-strided) execution
    rocblas_stride const strideA = lda * n;
    rocblas_stride const strideR = ldr * n;
    I const batch_count = 1;

    ROCSOLVER_ENTER_TOP("rocsolver_cholqr3", "-m", m, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    {
        auto const istat = rocsolver_cholqr3_argCheck(handle, m, n, lda, ldr, A, R);
        bool const isok = (istat == rocblas_status_continue) || (istat == rocblas_status_success);
        if(!isok)
        {
            return istat;
        }
    }

    // working with unshifted arrays
    rocblas_stride const shiftA = 0;
    rocblas_stride const shiftR = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_work = 0;
    rocsolver_cholqr3_getMemorySize<T, I>(m, n, batch_count, &size_work);

    size_t size_sigma_array = sizeof(S) * batch_count;

    if(rocblas_is_device_memory_size_query(handle))
    {
        return rocblas_set_optimal_device_memory_size(handle, size_work, size_sigma_array);
    }

    // memory workspace allocation

    rocblas_device_malloc mem(handle, size_work, size_sigma_array);

    if(!mem)
    {
        return rocblas_status_memory_error;
    }

    void* const work = mem[0];
    S* const sigma_array = (S*)mem[1];

    // execution

    rocsolver_cholqr_algo const algo = rocsolver_cholqr_cholqr3_compute;

    auto const istat = rocsolver_cholqr_template<T, I, rocblas_stride>(handle, m, n,

                                                                       A, shiftA, lda, strideA,

                                                                       R, shiftR, ldr, strideR,

                                                                       sigma_array, algo,

                                                                       info, batch_count,

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

rocblas_status rocsolver_scholqr3(rocblas_handle handle,
                                  const rocblas_int m,
                                  const rocblas_int n,

                                  float* A,
                                  const rocblas_int lda,

                                  float* R,
                                  const rocblas_int ldr,
                                  rocblas_int* info)
{
    return (rocsolver::rocsolver_cholqr3_impl<float>(handle, m, n, A, lda, R, ldr, info));
}

rocblas_status rocsolver_dcholqr3(rocblas_handle handle,
                                  const rocblas_int m,
                                  const rocblas_int n,

                                  double* A,
                                  const rocblas_int lda,

                                  double* R,
                                  const rocblas_int ldr,
                                  rocblas_int* info)
{
    return (rocsolver::rocsolver_cholqr3_impl<double>(handle, m, n, A, lda, R, ldr, info));
}

rocblas_status rocsolver_ccholqr3(rocblas_handle handle,
                                  const rocblas_int m,
                                  const rocblas_int n,

                                  rocblas_float_complex* A,
                                  const rocblas_int lda,

                                  rocblas_float_complex* R,
                                  const rocblas_int ldr,
                                  rocblas_int* info)
{
    return (rocsolver::rocsolver_cholqr3_impl<rocblas_float_complex>(handle, m, n, A, lda, R, ldr,
                                                                     info));
}

rocblas_status rocsolver_zcholqr3(rocblas_handle handle,
                                  const rocblas_int m,
                                  const rocblas_int n,

                                  rocblas_double_complex* A,
                                  const rocblas_int lda,

                                  rocblas_double_complex* R,
                                  const rocblas_int ldr,
                                  rocblas_int* info)
{
    return (rocsolver::rocsolver_cholqr3_impl<rocblas_double_complex>(handle, m, n, A, lda, R, ldr,
                                                                      info));
}

rocblas_status rocsolver_scholqr3_64(rocblas_handle handle,
                                     const int64_t m,
                                     const int64_t n,
                                     float* A,
                                     const int64_t lda,
                                     float* R,
                                     const int64_t ldr,
                                     int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_cholqr3_impl<float>(handle, m, n, A, lda, R, ldr, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dcholqr3_64(rocblas_handle handle,
                                     const int64_t m,
                                     const int64_t n,
                                     double* A,
                                     const int64_t lda,
                                     double* R,
                                     const int64_t ldr,
                                     int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_cholqr3_impl<double>(handle, m, n, A, lda, R, ldr, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_ccholqr3_64(rocblas_handle handle,
                                     const int64_t m,
                                     const int64_t n,
                                     rocblas_float_complex* A,
                                     const int64_t lda,
                                     rocblas_float_complex* R,
                                     const int64_t ldr,
                                     int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_cholqr3_impl<rocblas_float_complex>(handle, m, n, A, lda, R, ldr,
                                                                    info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zcholqr3_64(rocblas_handle handle,
                                     const int64_t m,
                                     const int64_t n,
                                     rocblas_double_complex* A,
                                     const int64_t lda,
                                     rocblas_double_complex* R,
                                     const int64_t ldr,
                                     int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_cholqr3_impl<rocblas_double_complex>(handle, m, n, A, lda, R, ldr,
                                                                     info);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
