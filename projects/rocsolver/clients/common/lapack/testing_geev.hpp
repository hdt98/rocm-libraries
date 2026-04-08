/* **************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"

template <bool STRIDED, bool COMPLEX, typename Td, typename Sd, typename Id>
void geev_checkBadArgs(const rocblas_handle handle,
                       const rocblas_evect jobvl,
                       const rocblas_evect jobvr,
                       const rocblas_int n,
                       Td dA,
                       const rocblas_int lda,
                       const rocblas_stride stA,
                       Sd dWr,
                       const rocblas_stride stWr,
                       Sd dWi,
                       const rocblas_stride stWi,
                       Td dVL,
                       const rocblas_int ldvl,
                       const rocblas_stride stVL,
                       Td dVR,
                       const rocblas_int ldvr,
                       const rocblas_stride stVR,
                       Id dinfo,
                       const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_geev(STRIDED, nullptr, jobvl, jobvr, n, dA, lda, stA, dWr, stWr, dWi, stWi, dVL,
                       ldvl, stVL, dVR, ldvr, stVR, dinfo, bc),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(
        rocsolver_geev(STRIDED, handle, rocblas_evect(0), jobvr, n, dA, lda, stA, dWr, stWr, dWi,
                       stWi, dVL, ldvl, stVL, dVR, ldvr, stVR, dinfo, bc),
        rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_geev(STRIDED, handle, jobvl, rocblas_evect(0), n, dA, lda, stA, dWr, stWr, dWi,
                       stWi, dVL, ldvl, stVL, dVR, ldvr, stVR, dinfo, bc),
        rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA, lda, stA, dWr, stWr, dWi, stWi,
                           dVL, ldvl, stVL, dVR, ldvr, stVR, dinfo, -1),
            rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, (Td) nullptr, lda, stA, dWr, stWr, dWi,
                       stWi, dVL, ldvl, stVL, dVR, ldvr, stVR, dinfo, bc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA, lda, stA, (Sd) nullptr, stWr, dWi,
                       stWi, dVL, ldvl, stVL, dVR, ldvr, stVR, dinfo, bc),
        rocblas_status_invalid_pointer);
    if(!COMPLEX)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA, lda, stA, dWr, stWr,
                           (Sd) nullptr, stWi, dVL, ldvl, stVL, dVR, ldvr, stVR, dinfo, bc),
            rocblas_status_invalid_pointer);
    }
    EXPECT_ROCBLAS_STATUS(
        rocsolver_geev(STRIDED, handle, rocblas_evect_original, jobvr, n, dA, lda, stA, dWr, stWr,
                       dWi, stWi, (Td) nullptr, ldvl, stVL, dVR, ldvr, stVR, dinfo, bc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_geev(STRIDED, handle, jobvl, rocblas_evect_original, n, dA, lda, stA, dWr, stWr,
                       dWi, stWi, dVL, ldvl, stVL, (Td) nullptr, ldvr, stVR, dinfo, bc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA, lda, stA, dWr, stWr, dWi, stWi, dVL,
                       ldvl, stVL, dVR, ldvr, stVR, (Id) nullptr, bc),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_geev(STRIDED, handle, jobvl, jobvr, 0, (Td) nullptr, lda, stA, (Sd) nullptr,
                       stWr, (Sd) nullptr, stWi, (Td) nullptr, ldvl, stVL, (Td) nullptr, ldvr,
                       stVR, dinfo, bc),
        rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA, lda, stA, dWr, stWr, dWi, stWi,
                           dVL, ldvl, stVL, dVR, ldvr, stVR, (Id) nullptr, 0),
            rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_geev_bad_arg()
{
    using S = decltype(std::real(T{}));
    constexpr bool COMPLEX = rocblas_is_complex<T>;

    rocblas_local_handle handle;
    rocblas_evect jobvl = rocblas_evect_none;
    rocblas_evect jobvr = rocblas_evect_none;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_int ldvl = 1;
    rocblas_int ldvr = 1;
    rocblas_stride stA = 1;
    rocblas_stride stWr = COMPLEX ? 2 : 1;
    rocblas_stride stWi = COMPLEX ? 0 : 1;
    rocblas_stride stVL = 1;
    rocblas_stride stVR = 1;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<S> dWr(COMPLEX ? 2 : 1, 1, stWr, 1);
        device_strided_batch_vector<S> dWi(1, 1, 1, 1);
        device_batch_vector<T> dVL(1, 1, 1);
        device_batch_vector<T> dVR(1, 1, 1);
        device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dWr.memcheck());
        CHECK_HIP_ERROR(dWi.memcheck());
        CHECK_HIP_ERROR(dVL.memcheck());
        CHECK_HIP_ERROR(dVR.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        geev_checkBadArgs<STRIDED, COMPLEX>(handle, jobvl, jobvr, n, dA.data(), lda, stA,
                                            dWr.data(), stWr, dWi.data(), stWi, dVL.data(), ldvl,
                                            stVL, dVR.data(), ldvr, stVR, dinfo.data(), bc);
    }
    else
    {
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<S> dWr(COMPLEX ? 2 : 1, 1, stWr, 1);
        device_strided_batch_vector<S> dWi(1, 1, 1, 1);
        device_strided_batch_vector<T> dVL(1, 1, 1, 1);
        device_strided_batch_vector<T> dVR(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dWr.memcheck());
        CHECK_HIP_ERROR(dWi.memcheck());
        CHECK_HIP_ERROR(dVL.memcheck());
        CHECK_HIP_ERROR(dVR.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        geev_checkBadArgs<STRIDED, COMPLEX>(handle, jobvl, jobvr, n, dA.data(), lda, stA,
                                            dWr.data(), stWr, dWi.data(), stWi, dVL.data(), ldvl,
                                            stVL, dVR.data(), ldvr, stVR, dinfo.data(), bc);
    }
}

template <bool STRIDED, typename T, typename Td, typename Sd, typename Id, typename Th, typename Sh, typename Ih>
void geev_getError(const rocblas_handle handle,
                   const rocblas_evect jobvl,
                   const rocblas_evect jobvr,
                   const rocblas_int n,
                   Td& dA,
                   const rocblas_int lda,
                   const rocblas_stride stA,
                   Sd& dWr,
                   const rocblas_stride stWr,
                   Sd& dWi,
                   const rocblas_stride stWi,
                   Td& dVL,
                   const rocblas_int ldvl,
                   const rocblas_stride stVL,
                   Td& dVR,
                   const rocblas_int ldvr,
                   const rocblas_stride stVR,
                   Id& dinfo,
                   const rocblas_int bc,
                   Th& hA,
                   Sh& hWr,
                   Sh& hWi,
                   Th& hVR,
                   Ih& hinfo,
                   Ih& hinfoRes,
                   double* max_err)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;
    using S = decltype(std::real(T{}));
    bool wantvr = (jobvr == rocblas_evect_original);

    rocblas_init<T>(hA, true);

    // save copy for residual check
    std::vector<T> Acopy;
    if(wantvr)
    {
        Acopy.resize((size_t)lda * n * bc);
        for(rocblas_int b = 0; b < bc; ++b)
            for(rocblas_int j = 0; j < n; ++j)
                for(rocblas_int i = 0; i < n; ++i)
                    Acopy[b * (size_t)lda * n + i + (size_t)j * lda] = hA[b][i + (size_t)j * lda];
    }

    CHECK_HIP_ERROR(dA.transfer_from(hA));

    // GPU computation
    CHECK_ROCBLAS_ERROR(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA.data(), lda, stA,
                                       dWr.data(), stWr, dWi.data(), stWi, dVL.data(), ldvl, stVL,
                                       dVR.data(), ldvr, stVR, dinfo.data(), bc));

    CHECK_HIP_ERROR(hWr.transfer_from(dWr));
    CHECK_HIP_ERROR(hWi.transfer_from(dWi));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));
    if(wantvr)
        CHECK_HIP_ERROR(hVR.transfer_from(dVR));

    *max_err = 0;

    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hinfoRes[b][0], 0) << "where b = " << b;
        if(hinfoRes[b][0] != 0)
            *max_err += 1;
    }

    if(!wantvr)
        return;

    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(hinfoRes[b][0] != 0)
            continue;

        const T* A = Acopy.data() + b * (size_t)lda * n;

        // ||A||_1
        S anorm = 0;
        for(rocblas_int j = 0; j < n; ++j)
        {
            S s = 0;
            for(rocblas_int i = 0; i < n; ++i)
            {
                if constexpr(COMPLEX)
                    s += std::abs(A[i + (size_t)j * lda].real())
                       + std::abs(A[i + (size_t)j * lda].imag());
                else
                    s += std::abs(A[i + (size_t)j * lda]);
            }
            anorm = std::max(anorm, s);
        }
        if(anorm == S(0))
            anorm = S(1);

        if constexpr(COMPLEX)
        {
            // for complex, eigenvalues are stored as complex in hWr buffer
            // (hWi is unused); reinterpret hWr[b] as T*
            const T* W = reinterpret_cast<const T*>(hWr[b]);

            for(rocblas_int j = 0; j < n; ++j)
            {
                T lambda = W[j];

                // Av = A * vr(:,j)
                std::vector<T> Av(n);
                for(rocblas_int i = 0; i < n; ++i)
                {
                    T sum{0, 0};
                    for(rocblas_int k = 0; k < n; ++k)
                        sum = sum + A[i + (size_t)k * lda] * hVR[b][k + (size_t)j * ldvr];
                    Av[i] = sum;
                }

                // ||vr(:,j)||
                S vn = 0;
                for(rocblas_int i = 0; i < n; ++i)
                {
                    auto v = hVR[b][i + (size_t)j * ldvr];
                    vn += v.real() * v.real() + v.imag() * v.imag();
                }
                vn = std::sqrt(vn);
                if(vn == S(0))
                    vn = S(1);

                // ||Av - lambda*v||
                S r = 0;
                for(rocblas_int i = 0; i < n; ++i)
                {
                    T d = Av[i] - lambda * hVR[b][i + (size_t)j * ldvr];
                    r += d.real() * d.real() + d.imag() * d.imag();
                }
                double err = std::sqrt((double)r) / ((double)anorm * (double)vn);
                *max_err = std::max(*max_err, err);
            }
        }
        else
        {
            rocblas_int j = 0;
            while(j < n)
            {
                if(hWi[b][j] == S(0))
                {
                    S wr_val = hWr[b][j];
                    S vn = 0, r = 0;
                    for(rocblas_int i = 0; i < n; ++i)
                    {
                        S Av_i = 0;
                        for(rocblas_int k = 0; k < n; ++k)
                            Av_i += A[i + (size_t)k * lda] * hVR[b][k + (size_t)j * ldvr];
                        S vi = hVR[b][i + (size_t)j * ldvr];
                        vn += vi * vi;
                        S d = Av_i - wr_val * vi;
                        r += d * d;
                    }
                    vn = std::sqrt(vn);
                    if(vn == S(0))
                        vn = S(1);
                    double err = std::sqrt((double)r) / ((double)anorm * (double)vn);
                    *max_err = std::max(*max_err, err);
                    j++;
                }
                else
                {
                    // complex conjugate pair: skip
                    j += 2;
                }
            }
        }
    }
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_geev(Arguments& argus)
{
    using S = decltype(std::real(T{}));
    constexpr bool COMPLEX = rocblas_is_complex<T>;

    // get arguments
    rocblas_local_handle handle;
    char jvlC = argus.get<char>("jobvl", 'N');
    char jvrC = argus.get<char>("jobvr", 'N');
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldvl = argus.get<rocblas_int>("ldvl", n > 0 ? n : 1);
    rocblas_int ldvr = argus.get<rocblas_int>("ldvr", n > 0 ? n : 1);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", (size_t)lda * n);
    rocblas_stride stVL = argus.get<rocblas_stride>("strideVL", (size_t)ldvl * n);
    rocblas_stride stVR = argus.get<rocblas_stride>("strideVR", (size_t)ldvr * n);

    // for complex: eigenvalues stored as n complex values = 2*n reals in wr buffer
    // for real: wr and wi each hold n reals
    size_t eig_elems = COMPLEX ? 2 * (size_t)n : (size_t)n;
    rocblas_stride stWr = argus.get<rocblas_stride>("strideWr", eig_elems);
    rocblas_stride stWi = argus.get<rocblas_stride>("strideWi", COMPLEX ? 0 : (size_t)n);

    rocblas_evect jobvl = char2rocblas_evect(jvlC);
    rocblas_evect jobvr = char2rocblas_evect(jvrC);
    bool wantvl = (jobvl == rocblas_evect_original);
    bool wantvr = (jobvr == rocblas_evect_original);

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_Wr = eig_elems;
    size_t size_Wi = COMPLEX ? 0 : (size_t)n;
    size_t size_VL = wantvl ? size_t(ldvl) * n : 0;
    size_t size_VR = wantvr ? size_t(ldvr) * n : 0;
    size_t size_VRres = (argus.unit_check || argus.norm_check) ? size_VR : 0;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < std::max(1, n) || bc < 0);
    if(wantvl)
        invalid_size = invalid_size || (ldvl < std::max(1, n));
    if(wantvr)
        invalid_size = invalid_size || (ldvr < std::max(1, n));

    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, (T* const*)nullptr, lda, stA,
                               (S*)nullptr, stWr, (S*)nullptr, stWi, (T* const*)nullptr, ldvl,
                               stVL, (T* const*)nullptr, ldvr, stVR, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(
                rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, (T*)nullptr, lda, stA,
                               (S*)nullptr, stWr, (S*)nullptr, stWi, (T*)nullptr, ldvl, stVL,
                               (T*)nullptr, ldvr, stVR, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
            CHECK_ALLOC_QUERY(
                rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, (T* const*)nullptr, lda, stA,
                               (S*)nullptr, stWr, (S*)nullptr, stWi, (T* const*)nullptr, ldvl,
                               stVL, (T* const*)nullptr, ldvr, stVR, (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(
                rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, (T*)nullptr, lda, stA,
                               (S*)nullptr, stWr, (S*)nullptr, stWi, (T*)nullptr, ldvl, stVL,
                               (T*)nullptr, ldvr, stVR, (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations (all cases)
    host_strided_batch_vector<S> hWr(size_Wr, 1, stWr, bc);
    host_strided_batch_vector<S> hWi(std::max(size_Wi, (size_t)1), 1, std::max(stWi, (rocblas_stride)1), bc);
    host_strided_batch_vector<rocblas_int> hinfo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hinfoRes(1, 1, 1, bc);

    device_strided_batch_vector<S> dWr(size_Wr, 1, stWr, bc);
    device_strided_batch_vector<S> dWi(std::max(size_Wi, (size_t)1), 1, std::max(stWi, (rocblas_stride)1), bc);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, bc);
    if(size_Wr)
        CHECK_HIP_ERROR(dWr.memcheck());
    CHECK_HIP_ERROR(dWi.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    if(BATCHED)
    {
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hVR(std::max(size_VRres, (size_t)1), 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dVL(std::max(size_VL, (size_t)1), 1, bc);
        device_batch_vector<T> dVR(std::max(size_VR, (size_t)1), 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_VL)
            CHECK_HIP_ERROR(dVL.memcheck());
        if(size_VR)
            CHECK_HIP_ERROR(dVR.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(
                rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA.data(), lda, stA, dWr.data(),
                               stWr, dWi.data(), stWi, dVL.data(), ldvl, stVL, dVR.data(), ldvr,
                               stVR, dinfo.data(), bc),
                rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            geev_getError<STRIDED, T>(handle, jobvl, jobvr, n, dA, lda, stA, dWr, stWr, dWi, stWi,
                                      dVL, ldvl, stVL, dVR, ldvr, stVR, dinfo, bc, hA, hWr, hWi,
                                      hVR, hinfo, hinfoRes, &max_error);
        }
    }
    else
    {
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hVR(std::max(size_VRres, (size_t)1), 1, stVR, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dVL(std::max(size_VL, (size_t)1), 1, stVL, bc);
        device_strided_batch_vector<T> dVR(std::max(size_VR, (size_t)1), 1, stVR, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_VL)
            CHECK_HIP_ERROR(dVL.memcheck());
        if(size_VR)
            CHECK_HIP_ERROR(dVR.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(
                rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA.data(), lda, stA, dWr.data(),
                               stWr, dWi.data(), stWi, dVL.data(), ldvl, stVL, dVR.data(), ldvr,
                               stVR, dinfo.data(), bc),
                rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            geev_getError<STRIDED, T>(handle, jobvl, jobvr, n, dA, lda, stA, dWr, stWr, dWi, stWi,
                                      dVL, ldvl, stVL, dVR, ldvr, stVR, dinfo, bc, hA, hWr, hWi,
                                      hVR, hinfo, hinfoRes, &max_error);
        }
    }

    // validate results for rocsolver-test
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("jobvl", "jobvr", "n", "lda", "strideWr", "strideWi",
                                       "ldvl", "ldvr", "batch_c");
                rocsolver_bench_output(jvlC, jvrC, n, lda, stWr, stWi, ldvl, ldvr, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("jobvl", "jobvr", "n", "lda", "strideA", "strideWr",
                                       "strideWi", "ldvl", "strideVL", "ldvr", "strideVR",
                                       "batch_c");
                rocsolver_bench_output(jvlC, jvrC, n, lda, stA, stWr, stWi, ldvl, stVL, ldvr, stVR,
                                       bc);
            }
            else
            {
                rocsolver_bench_output("jobvl", "jobvr", "n", "lda", "ldvl", "ldvr");
                rocsolver_bench_output(jvlC, jvrC, n, lda, ldvl, ldvr);
            }
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_bench_endl();
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_GEEV(...) extern template void testing_geev<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEEV, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
