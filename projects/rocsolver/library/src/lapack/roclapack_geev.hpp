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

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

/*
 * ===========================================================================
 *    LAPACK extern declarations (host LAPACK library)
 *
 *    We declare geev for the n==1 fallback / quick-return paths.
 * ===========================================================================
 */

extern "C" {

/* ---------- geev (fallback for n == 1 or error paths) ---------- */

void sgeev_(char* jobvl,
            char* jobvr,
            int* n,
            float* A,
            int* lda,
            float* wr,
            float* wi,
            float* VL,
            int* ldvl,
            float* VR,
            int* ldvr,
            float* work,
            int* lwork,
            int* info);

void dgeev_(char* jobvl,
            char* jobvr,
            int* n,
            double* A,
            int* lda,
            double* wr,
            double* wi,
            double* VL,
            int* ldvl,
            double* VR,
            int* ldvr,
            double* work,
            int* lwork,
            int* info);

void cgeev_(char* jobvl,
            char* jobvr,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* w,
            rocblas_float_complex* VL,
            int* ldvl,
            rocblas_float_complex* VR,
            int* ldvr,
            rocblas_float_complex* work,
            int* lwork,
            float* rwork,
            int* info);

void zgeev_(char* jobvl,
            char* jobvr,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* w,
            rocblas_double_complex* VL,
            int* ldvl,
            rocblas_double_complex* VR,
            int* ldvr,
            rocblas_double_complex* work,
            int* lwork,
            double* rwork,
            int* info);

} // extern "C" (LAPACK)

/*
 * ===========================================================================
 *    MAGMA extern declarations
 *
 *    MAGMA's geev operates on host data but internally uses GPU for
 *    gehrd (blocked BLAS-3 Hessenberg reduction) and orghr (Q generation).
 *    The Schur decomposition (hseqr) and eigenvector computation run on CPU.
 * ===========================================================================
 */

extern "C" {

int magma_init();

/* Real MAGMA geev */
int magma_sgeev(int jobvl, int jobvr, int n, float* A, int lda,
    float* wr, float* wi, float* VL, int ldvl, float* VR, int ldvr,
    float* work, int lwork, int* info);
int magma_dgeev(int jobvl, int jobvr, int n, double* A, int lda,
    double* wr, double* wi, double* VL, int ldvl, double* VR, int ldvr,
    double* work, int lwork, int* info);

/* Complex MAGMA geev (uses void* for magmaFloatComplex/magmaDoubleComplex compatibility) */
int magma_cgeev(int jobvl, int jobvr, int n, void* A, int lda,
    void* w, void* VL, int ldvl, void* VR, int ldvr,
    void* work, int lwork, float* rwork, int* info);
int magma_zgeev(int jobvl, int jobvr, int n, void* A, int lda,
    void* w, void* VL, int ldvl, void* VR, int ldvr,
    void* work, int lwork, double* rwork, int* info);

} // extern "C" (MAGMA)

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    MAGMA initialization helper
 * ===========================================================================
 */
inline void ensure_magma_init()
{
    static bool initialized = false;
    if(!initialized)
    {
        magma_init();
        initialized = true;
    }
}

/*
 * ===========================================================================
 *    Overloaded wrappers for type-dispatched MAGMA geev calls
 * ===========================================================================
 */

/* --- Real MAGMA geev wrappers --- */
inline int call_magma_geev(int jobvl, int jobvr, int n, float* A, int lda,
                            float* wr, float* wi, float* VL, int ldvl,
                            float* VR, int ldvr, float* work, int lwork, int* info)
{
    return magma_sgeev(jobvl, jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr, work, lwork, info);
}

inline int call_magma_geev(int jobvl, int jobvr, int n, double* A, int lda,
                            double* wr, double* wi, double* VL, int ldvl,
                            double* VR, int ldvr, double* work, int lwork, int* info)
{
    return magma_dgeev(jobvl, jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr, work, lwork, info);
}

/* --- Complex MAGMA geev wrappers --- */
inline int call_magma_geev(int jobvl, int jobvr, int n, rocblas_float_complex* A, int lda,
                            rocblas_float_complex* w, rocblas_float_complex* VL, int ldvl,
                            rocblas_float_complex* VR, int ldvr,
                            rocblas_float_complex* work, int lwork, float* rwork, int* info)
{
    return magma_cgeev(jobvl, jobvr, n, (void*)A, lda, (void*)w,
                       (void*)VL, ldvl, (void*)VR, ldvr,
                       (void*)work, lwork, rwork, info);
}

inline int call_magma_geev(int jobvl, int jobvr, int n, rocblas_double_complex* A, int lda,
                            rocblas_double_complex* w, rocblas_double_complex* VL, int ldvl,
                            rocblas_double_complex* VR, int ldvr,
                            rocblas_double_complex* work, int lwork, double* rwork, int* info)
{
    return magma_zgeev(jobvl, jobvr, n, (void*)A, lda, (void*)w,
                       (void*)VL, ldvl, (void*)VR, ldvr,
                       (void*)work, lwork, rwork, info);
}

/*
 * ===========================================================================
 *    Argument checking
 * ===========================================================================
 */

/** Argument checking for real types (wr + wi) **/
template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_status rocsolver_geev_argCheck(rocblas_handle handle,
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
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if((jobvl != rocblas_evect_original && jobvl != rocblas_evect_none)
       || (jobvr != rocblas_evect_original && jobvr != rocblas_evect_none))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < std::max(1, n))
        return rocblas_status_invalid_size;
    if(jobvl == rocblas_evect_original && ldvl < std::max(1, n))
        return rocblas_status_invalid_size;
    if(jobvr == rocblas_evect_original && ldvr < std::max(1, n))
        return rocblas_status_invalid_size;
    if(jobvl == rocblas_evect_none && ldvl < 1)
        return rocblas_status_invalid_size;
    if(jobvr == rocblas_evect_none && ldvr < 1)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !wr) || (n && !wi) || !info)
        return rocblas_status_invalid_pointer;
    if(jobvl == rocblas_evect_original && n && !VL)
        return rocblas_status_invalid_pointer;
    if(jobvr == rocblas_evect_original && n && !VR)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** Argument checking for complex types (w + rwork) **/
template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
rocblas_status rocsolver_geev_argCheck(rocblas_handle handle,
                                       const rocblas_evect jobvl,
                                       const rocblas_evect jobvr,
                                       const rocblas_int n,
                                       T* A,
                                       const rocblas_int lda,
                                       T* w,
                                       T* /*wi_unused*/,
                                       T* VL,
                                       const rocblas_int ldvl,
                                       T* VR,
                                       const rocblas_int ldvr,
                                       rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if((jobvl != rocblas_evect_original && jobvl != rocblas_evect_none)
       || (jobvr != rocblas_evect_original && jobvr != rocblas_evect_none))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < std::max(1, n))
        return rocblas_status_invalid_size;
    if(jobvl == rocblas_evect_original && ldvl < std::max(1, n))
        return rocblas_status_invalid_size;
    if(jobvr == rocblas_evect_original && ldvr < std::max(1, n))
        return rocblas_status_invalid_size;
    if(jobvl == rocblas_evect_none && ldvl < 1)
        return rocblas_status_invalid_size;
    if(jobvr == rocblas_evect_none && ldvr < 1)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !w) || !info)
        return rocblas_status_invalid_pointer;
    if(jobvl == rocblas_evect_original && n && !VL)
        return rocblas_status_invalid_pointer;
    if(jobvr == rocblas_evect_original && n && !VR)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/*
 * ===========================================================================
 *    Template implementation: MAGMA GPU-accelerated eigenvalue computation
 *
 *    Strategy:
 *      1. Copy device matrix A to host
 *      2. Call MAGMA's geev (which internally uses GPU for gehrd/orghr)
 *      3. Copy eigenvalues back to device
 *      4. Copy eigenvectors back to device (if requested)
 *      5. Copy info back to device
 * ===========================================================================
 */

/** Real type implementation **/
template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_status rocsolver_geev_template(rocblas_handle handle,
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
    using S = decltype(std::real(T{}));

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return if n == 0
    if(n == 0)
    {
        rocblas_int zero = 0;
        HIP_CHECK(hipMemcpyAsync(info, &zero, sizeof(rocblas_int), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        return rocblas_status_success;
    }

    // For n == 1, eigenvalue is simply A[0,0]
    if(n == 1)
    {
        T h_val;
        HIP_CHECK(hipMemcpyAsync(&h_val, A, sizeof(T), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        T zero_val = T(0);
        rocblas_int zero = 0;
        HIP_CHECK(hipMemcpyAsync(wr, &h_val, sizeof(T), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(wi, &zero_val, sizeof(T), hipMemcpyHostToDevice, stream));
        if(jobvl == rocblas_evect_original)
        {
            T one_val = T(1);
            HIP_CHECK(hipMemcpyAsync(VL, &one_val, sizeof(T), hipMemcpyHostToDevice, stream));
        }
        if(jobvr == rocblas_evect_original)
        {
            T one_val = T(1);
            HIP_CHECK(hipMemcpyAsync(VR, &one_val, sizeof(T), hipMemcpyHostToDevice, stream));
        }
        HIP_CHECK(hipMemcpyAsync(info, &zero, sizeof(rocblas_int), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        return rocblas_status_success;
    }

    // -------------------------------------------------------
    // n >= 2: Use MAGMA's GPU-accelerated geev
    // -------------------------------------------------------

    ensure_magma_init();

    bool const wantvl = (jobvl == rocblas_evect_original);
    bool const wantvr = (jobvr == rocblas_evect_original);
    size_t const size_A = size_t(lda) * n;

    // MAGMA enum values: MagmaVec=302, MagmaNoVec=301
    int magma_jobvl = wantvl ? 302 : 301;
    int magma_jobvr = wantvr ? 302 : 301;

    // Step 1: Copy A from device to host
    T* hA = (T*)malloc(sizeof(T) * size_A);
    if(!hA)
        return rocblas_status_memory_error;

    HIP_CHECK(hipMemcpyAsync(hA, A, sizeof(T) * size_A, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Allocate host eigenvalue arrays
    T* hwr = (T*)malloc(sizeof(T) * n);
    T* hwi = (T*)malloc(sizeof(T) * n);
    if(!hwr || !hwi)
    {
        free(hwr);
        free(hwi);
        free(hA);
        return rocblas_status_memory_error;
    }

    // Allocate host eigenvector storage
    int ldvl_host = wantvl ? n : 1;
    int ldvr_host = wantvr ? n : 1;
    T* hVL = wantvl ? (T*)malloc(sizeof(T) * size_t(ldvl_host) * n) : nullptr;
    T* hVR = wantvr ? (T*)malloc(sizeof(T) * size_t(ldvr_host) * n) : nullptr;
    if((wantvl && !hVL) || (wantvr && !hVR))
    {
        free(hVR);
        free(hVL);
        free(hwr);
        free(hwi);
        free(hA);
        return rocblas_status_memory_error;
    }

    // Step 2: Workspace query
    int n_int = n;
    int lda_int = lda;
    int info_int = 0;
    T work_query;
    int lwork = -1;

    call_magma_geev(magma_jobvl, magma_jobvr, n_int, hA, lda_int,
                    hwr, hwi, hVL, ldvl_host, hVR, ldvr_host,
                    &work_query, lwork, &info_int);
    lwork = (int)work_query;
    if(lwork < 1)
        lwork = std::max(1, 4 * n);

    T* work = (T*)malloc(sizeof(T) * lwork);
    if(!work)
    {
        free(hVR);
        free(hVL);
        free(hwr);
        free(hwi);
        free(hA);
        return rocblas_status_memory_error;
    }

    // Step 3: Call MAGMA geev
    info_int = 0;
    call_magma_geev(magma_jobvl, magma_jobvr, n_int, hA, lda_int,
                    hwr, hwi, hVL, ldvl_host, hVR, ldvr_host,
                    work, lwork, &info_int);

    // Step 4: Copy eigenvalues back to device
    HIP_CHECK(hipMemcpyAsync(wr, hwr, sizeof(T) * n, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(wi, hwi, sizeof(T) * n, hipMemcpyHostToDevice, stream));

    // Step 5: Copy eigenvectors back to device (if requested)
    if(wantvl && hVL)
    {
        for(int j = 0; j < n; j++)
        {
            HIP_CHECK(hipMemcpyAsync(VL + size_t(j) * ldvl,
                                      hVL + size_t(j) * ldvl_host,
                                      sizeof(T) * n, hipMemcpyHostToDevice, stream));
        }
    }
    if(wantvr && hVR)
    {
        for(int j = 0; j < n; j++)
        {
            HIP_CHECK(hipMemcpyAsync(VR + size_t(j) * ldvr,
                                      hVR + size_t(j) * ldvr_host,
                                      sizeof(T) * n, hipMemcpyHostToDevice, stream));
        }
    }

    // Step 6: Copy info back to device
    rocblas_int info_result = info_int;
    HIP_CHECK(hipMemcpyAsync(info, &info_result, sizeof(rocblas_int), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Cleanup
    free(work);
    free(hVR);
    free(hVL);
    free(hwr);
    free(hwi);
    free(hA);

    return rocblas_status_success;
}

/** Complex type implementation **/
template <typename T, typename S, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
rocblas_status rocsolver_geev_template(rocblas_handle handle,
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
                                       S* rwork_unused,
                                       rocblas_int* info)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return if n == 0
    if(n == 0)
    {
        rocblas_int zero = 0;
        HIP_CHECK(hipMemcpyAsync(info, &zero, sizeof(rocblas_int), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        return rocblas_status_success;
    }

    // For n == 1, eigenvalue is simply A[0,0]
    if(n == 1)
    {
        T h_val;
        HIP_CHECK(hipMemcpyAsync(&h_val, A, sizeof(T), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        rocblas_int zero = 0;
        HIP_CHECK(hipMemcpyAsync(w, &h_val, sizeof(T), hipMemcpyHostToDevice, stream));
        if(jobvl == rocblas_evect_original)
        {
            T one_val = T(1);
            HIP_CHECK(hipMemcpyAsync(VL, &one_val, sizeof(T), hipMemcpyHostToDevice, stream));
        }
        if(jobvr == rocblas_evect_original)
        {
            T one_val = T(1);
            HIP_CHECK(hipMemcpyAsync(VR, &one_val, sizeof(T), hipMemcpyHostToDevice, stream));
        }
        HIP_CHECK(hipMemcpyAsync(info, &zero, sizeof(rocblas_int), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        return rocblas_status_success;
    }

    // -------------------------------------------------------
    // n >= 2: Use MAGMA's GPU-accelerated geev
    // -------------------------------------------------------

    ensure_magma_init();

    bool const wantvl = (jobvl == rocblas_evect_original);
    bool const wantvr = (jobvr == rocblas_evect_original);
    size_t const size_A = size_t(lda) * n;

    // MAGMA enum values: MagmaVec=302, MagmaNoVec=301
    int magma_jobvl = wantvl ? 302 : 301;
    int magma_jobvr = wantvr ? 302 : 301;

    // Step 1: Copy A from device to host
    T* hA = (T*)malloc(sizeof(T) * size_A);
    if(!hA)
        return rocblas_status_memory_error;

    HIP_CHECK(hipMemcpyAsync(hA, A, sizeof(T) * size_A, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Allocate host eigenvalue array
    T* hW = (T*)malloc(sizeof(T) * n);
    if(!hW)
    {
        free(hA);
        return rocblas_status_memory_error;
    }

    // Allocate host eigenvector storage
    int ldvl_host = wantvl ? n : 1;
    int ldvr_host = wantvr ? n : 1;
    T* hVL = wantvl ? (T*)malloc(sizeof(T) * size_t(ldvl_host) * n) : nullptr;
    T* hVR = wantvr ? (T*)malloc(sizeof(T) * size_t(ldvr_host) * n) : nullptr;
    if((wantvl && !hVL) || (wantvr && !hVR))
    {
        free(hVR);
        free(hVL);
        free(hW);
        free(hA);
        return rocblas_status_memory_error;
    }

    // Allocate rwork for complex geev (MAGMA requires 2*n reals)
    S* rwork = (S*)malloc(sizeof(S) * 2 * n);
    if(!rwork)
    {
        free(hVR);
        free(hVL);
        free(hW);
        free(hA);
        return rocblas_status_memory_error;
    }

    // Step 2: Workspace query
    int n_int = n;
    int lda_int = lda;
    int info_int = 0;
    T work_query;
    int lwork = -1;

    call_magma_geev(magma_jobvl, magma_jobvr, n_int, hA, lda_int,
                    hW, hVL, ldvl_host, hVR, ldvr_host,
                    &work_query, lwork, rwork, &info_int);
    lwork = (int)std::real(work_query);
    if(lwork < 1)
        lwork = std::max(1, 2 * n);

    T* work = (T*)malloc(sizeof(T) * lwork);
    if(!work)
    {
        free(rwork);
        free(hVR);
        free(hVL);
        free(hW);
        free(hA);
        return rocblas_status_memory_error;
    }

    // Need to re-copy A since the workspace query may have modified it
    HIP_CHECK(hipMemcpyAsync(hA, A, sizeof(T) * size_A, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Step 3: Call MAGMA geev
    info_int = 0;
    call_magma_geev(magma_jobvl, magma_jobvr, n_int, hA, lda_int,
                    hW, hVL, ldvl_host, hVR, ldvr_host,
                    work, lwork, rwork, &info_int);

    // Step 4: Copy eigenvalues back to device
    HIP_CHECK(hipMemcpyAsync(w, hW, sizeof(T) * n, hipMemcpyHostToDevice, stream));

    // Step 5: Copy eigenvectors back to device (if requested)
    if(wantvl && hVL)
    {
        for(int j = 0; j < n; j++)
        {
            HIP_CHECK(hipMemcpyAsync(VL + size_t(j) * ldvl,
                                      hVL + size_t(j) * ldvl_host,
                                      sizeof(T) * n, hipMemcpyHostToDevice, stream));
        }
    }
    if(wantvr && hVR)
    {
        for(int j = 0; j < n; j++)
        {
            HIP_CHECK(hipMemcpyAsync(VR + size_t(j) * ldvr,
                                      hVR + size_t(j) * ldvr_host,
                                      sizeof(T) * n, hipMemcpyHostToDevice, stream));
        }
    }

    // Step 6: Copy info back to device
    rocblas_int info_result = info_int;
    HIP_CHECK(hipMemcpyAsync(info, &info_result, sizeof(rocblas_int), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Cleanup
    free(work);
    free(rwork);
    free(hVR);
    free(hVL);
    free(hW);
    free(hA);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
