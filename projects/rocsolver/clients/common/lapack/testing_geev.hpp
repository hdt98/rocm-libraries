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

#include <algorithm>
#include <cmath>
#include <vector>

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

template <bool STRIDED, typename T, typename S, typename U>
void geev_checkBadArgs(const rocblas_handle handle,
                       const rocblas_evect jobvl,
                       const rocblas_evect jobvr,
                       const rocblas_int n,
                       T dA,
                       const rocblas_int lda,
                       const rocblas_stride stA,
                       T dWr,
                       S dWi,
                       const rocblas_stride stW,
                       T dVL,
                       const rocblas_int ldvl,
                       const rocblas_stride stVL,
                       T dVR,
                       const rocblas_int ldvr,
                       const rocblas_stride stVR,
                       U dinfo,
                       const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, nullptr, jobvl, jobvr, n, dA, lda, stA, dWr,
                                         dWi, stW, dVL, ldvl, stVL, dVR, ldvr, stVR, dinfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, rocblas_evect(0), jobvr, n, dA, lda, stA,
                                         dWr, dWi, stW, dVL, ldvl, stVL, dVR, ldvr, stVR, dinfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, rocblas_evect(0), n, dA, lda, stA,
                                         dWr, dWi, stW, dVL, ldvl, stVL, dVR, ldvr, stVR, dinfo, bc),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, (T) nullptr, lda, stA,
                                         dWr, dWi, stW, dVL, ldvl, stVL, dVR, ldvr, stVR, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA, lda, stA,
                                         (T) nullptr, dWi, stW, dVL, ldvl, stVL, dVR, ldvr, stVR, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA, lda, stA,
                                         dWr, dWi, stW, dVL, ldvl, stVL, dVR, ldvr, stVR, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, 0, (T) nullptr, lda, stA,
                                         (T) nullptr, (S) nullptr, stW, (T) nullptr, ldvl, stVL,
                                         (T) nullptr, ldvr, stVR, dinfo, bc),
                          rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_geev_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_evect jobvl = rocblas_evect_none;
    rocblas_evect jobvr = rocblas_evect_none;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_int ldvl = 1;
    rocblas_int ldvr = 1;
    rocblas_stride stA = 1;
    rocblas_stride stW = 1;
    rocblas_stride stVL = 1;
    rocblas_stride stVR = 1;
    rocblas_int bc = 1;

    // memory allocations
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dWr(1, 1, 1, 1);
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

    // check bad arguments
    geev_checkBadArgs<STRIDED>(handle, jobvl, jobvr, n, dA.data(), lda, stA, dWr.data(),
                               dWi.data(), stW, dVL.data(), ldvl, stVL, dVR.data(), ldvr,
                               stVR, dinfo.data(), bc);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void geev_initData(const rocblas_handle handle,
                   const rocblas_int n,
                   Td& dA,
                   const rocblas_int lda,
                   const rocblas_int bc,
                   Th& hA)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities and improve convergence
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool STRIDED, typename T, typename Wd, typename Sd, typename Td, typename Id, typename Wh, typename Sh, typename Th, typename Ih>
void geev_getError(const rocblas_handle handle,
                   const rocblas_evect jobvl,
                   const rocblas_evect jobvr,
                   const rocblas_int n,
                   Td& dA,
                   const rocblas_int lda,
                   const rocblas_stride stA,
                   Wd& dWr,
                   Sd& dWi,
                   const rocblas_stride stW,
                   Td& dVL,
                   const rocblas_int ldvl,
                   const rocblas_stride stVL,
                   Td& dVR,
                   const rocblas_int ldvr,
                   const rocblas_stride stVR,
                   Id& dinfo,
                   const rocblas_int bc,
                   Th& hA,
                   Wh& hWr,
                   Wh& hWrRes,
                   Sh& hWi,
                   Sh& hWiRes,
                   Ih& hinfo,
                   Ih& hinfoRes,
                   double* max_err)
{
    using S = decltype(std::real(T{}));

    // workspace for cpu_geev
    rocblas_int lwork = std::max(1, 4 * n);
    std::vector<T> work(lwork);
    std::vector<S> rwork(std::max(1, 2 * n));

    // we need copies of A for both GPU and CPU
    host_strided_batch_vector<T> hAcopy(size_t(lda) * n, 1, stA, bc);

    // input data initialization
    geev_initData<true, true, T>(handle, n, dA, lda, bc, hA);

    // save a copy for CPU
    for(rocblas_int b = 0; b < bc; ++b)
        for(rocblas_int i = 0; i < lda * n; ++i)
            hAcopy[b][i] = hA[b][i];

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA.data(), lda, stA,
                                       dWr.data(), dWi.data(), stW, dVL.data(), ldvl, stVL,
                                       dVR.data(), ldvr, stVR, dinfo.data(), bc));

    CHECK_HIP_ERROR(hWrRes.transfer_from(dWr));
    CHECK_HIP_ERROR(hWiRes.transfer_from(dWi));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));

    // CPU lapack
    // need temporary arrays for eigenvectors even if not requested
    std::vector<T> cpuVL(ldvl * n);
    std::vector<T> cpuVR(ldvr * n);

    for(rocblas_int b = 0; b < bc; ++b)
    {
        cpu_geev(jobvl, jobvr, n, hAcopy[b], lda, hWr[b], hWi[b], cpuVL.data(), ldvl,
                 cpuVR.data(), ldvr, work.data(), lwork, rwork.data(), hinfo[b]);
    }

    // Check info for non-convergence
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hinfo[b][0], hinfoRes[b][0]) << "where b = " << b;
        if(hinfo[b][0] != hinfoRes[b][0])
            *max_err += 1;
    }

    // compare eigenvalues (sort them first since order is not guaranteed)
    double err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(hinfo[b][0] == 0)
        {
            // Build complex eigenvalue pairs for both CPU and GPU results,
            // sort them, then compare. Eigenvalue ordering is not guaranteed
            // to match between different LAPACK implementations.
            std::vector<std::pair<double, double>> cpu_eigs(n), gpu_eigs(n);

            if constexpr(rocblas_is_complex<T>)
            {
                for(rocblas_int i = 0; i < n; i++)
                {
                    cpu_eigs[i] = {std::real(T(hWr[b][i])), std::imag(T(hWr[b][i]))};
                    gpu_eigs[i] = {std::real(T(hWrRes[b][i])), std::imag(T(hWrRes[b][i]))};
                }
            }
            else
            {
                for(rocblas_int i = 0; i < n; i++)
                {
                    cpu_eigs[i] = {double(hWr[b][i]), double(hWi[b][i])};
                    gpu_eigs[i] = {double(hWrRes[b][i]), double(hWiRes[b][i])};
                }
            }

            auto cmp = [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                if(a.first != b.first)
                    return a.first < b.first;
                return a.second < b.second;
            };
            std::sort(cpu_eigs.begin(), cpu_eigs.end(), cmp);
            std::sort(gpu_eigs.begin(), gpu_eigs.end(), cmp);

            // compute max relative error across all eigenvalues
            double norm = 0, diff = 0;
            for(rocblas_int i = 0; i < n; i++)
            {
                double dr = cpu_eigs[i].first - gpu_eigs[i].first;
                double di = cpu_eigs[i].second - gpu_eigs[i].second;
                double nr = cpu_eigs[i].first;
                double ni = cpu_eigs[i].second;
                diff += dr * dr + di * di;
                norm += nr * nr + ni * ni;
            }
            err = (norm > 0) ? sqrt(diff / norm) : sqrt(diff);
            *max_err = err > *max_err ? err : *max_err;
        }
    }
}

template <bool STRIDED, typename T, typename Wd, typename Sd, typename Td, typename Id, typename Wh, typename Sh, typename Th, typename Ih>
void geev_getPerfData(const rocblas_handle handle,
                      const rocblas_evect jobvl,
                      const rocblas_evect jobvr,
                      const rocblas_int n,
                      Td& dA,
                      const rocblas_int lda,
                      const rocblas_stride stA,
                      Wd& dWr,
                      Sd& dWi,
                      const rocblas_stride stW,
                      Td& dVL,
                      const rocblas_int ldvl,
                      const rocblas_stride stVL,
                      Td& dVR,
                      const rocblas_int ldvr,
                      const rocblas_stride stVR,
                      Id& dinfo,
                      const rocblas_int bc,
                      Th& hA,
                      Wh& hWr,
                      Sh& hWi,
                      Ih& hinfo,
                      double* gpu_time_used,
                      double* cpu_time_used,
                      const rocblas_int hot_calls,
                      const int profile,
                      const bool profile_kernels,
                      const bool perf)
{
    using S = decltype(std::real(T{}));

    rocblas_int lwork = std::max(1, 4 * n);
    std::vector<T> work(lwork);
    std::vector<S> rwork(std::max(1, 2 * n));
    std::vector<T> cpuVL(n * n);
    std::vector<T> cpuVR(n * n);

    if(!perf)
    {
        geev_initData<true, false, T>(handle, n, dA, lda, bc, hA);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            cpu_geev(jobvl, jobvr, n, hA[b], lda, hWr[b], hWi[b], cpuVL.data(), n,
                     cpuVR.data(), n, work.data(), lwork, rwork.data(), hinfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    geev_initData<true, false, T>(handle, n, dA, lda, bc, hA);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        geev_initData<false, true, T>(handle, n, dA, lda, bc, hA);

        CHECK_ROCBLAS_ERROR(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA.data(), lda, stA,
                                           dWr.data(), dWi.data(), stW, dVL.data(), ldvl, stVL,
                                           dVR.data(), ldvr, stVR, dinfo.data(), bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    rocsolver_timer timer;

    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        geev_initData<false, true, T>(handle, n, dA, lda, bc, hA);

        timer.start(stream);
        rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA.data(), lda, stA, dWr.data(),
                       dWi.data(), stW, dVL.data(), ldvl, stVL, dVR.data(), ldvr, stVR,
                       dinfo.data(), bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_geev(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char jobvlC = argus.get<char>("jobvl", 'N');
    char jobvrC = argus.get<char>("jobvr", 'N');
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldvl = argus.get<rocblas_int>("ldvl", (jobvlC == 'V' ? std::max(1, n) : 1));
    rocblas_int ldvr = argus.get<rocblas_int>("ldvr", (jobvrC == 'V' ? std::max(1, n) : 1));
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stW = argus.get<rocblas_stride>("strideW", n);
    rocblas_stride stVL = argus.get<rocblas_stride>("strideVL", ldvl * n);
    rocblas_stride stVR = argus.get<rocblas_stride>("strideVR", ldvr * n);

    rocblas_evect jobvl = char2rocblas_evect(jobvlC);
    rocblas_evect jobvr = char2rocblas_evect(jobvrC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_W = size_t(n);
    size_t size_VL = size_t(ldvl) * n;
    size_t size_VR = size_t(ldvr) * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_WRes = (argus.unit_check || argus.norm_check) ? size_W : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || bc < 0);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, (T*)nullptr, lda, stA,
                           (T*)nullptr, (S*)nullptr, stW, (T*)nullptr, ldvl, stVL,
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
        CHECK_ALLOC_QUERY(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, (T*)nullptr, lda,
                                         stA, (T*)nullptr, (S*)nullptr, stW, (T*)nullptr, ldvl,
                                         stVL, (T*)nullptr, ldvr, stVR, (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
    host_strided_batch_vector<T> hWr(size_W, 1, stW, bc);
    host_strided_batch_vector<T> hWrRes(size_WRes, 1, stW, bc);
    host_strided_batch_vector<S> hWi(size_W, 1, stW, bc);
    host_strided_batch_vector<S> hWiRes(size_WRes, 1, stW, bc);
    host_strided_batch_vector<rocblas_int> hinfo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hinfoRes(1, 1, 1, bc);
    device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
    device_strided_batch_vector<T> dWr(size_W, 1, stW, bc);
    device_strided_batch_vector<S> dWi(size_W, 1, stW, bc);
    device_strided_batch_vector<T> dVL(size_VL, 1, stVL, bc);
    device_strided_batch_vector<T> dVR(size_VR, 1, stVR, bc);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, bc);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_W)
    {
        CHECK_HIP_ERROR(dWr.memcheck());
        CHECK_HIP_ERROR(dWi.memcheck());
    }
    if(size_VL)
        CHECK_HIP_ERROR(dVL.memcheck());
    if(size_VR)
        CHECK_HIP_ERROR(dVR.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    // check quick return
    if(n == 0 || bc == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA.data(), lda, stA,
                           dWr.data(), dWi.data(), stW, dVL.data(), ldvl, stVL,
                           dVR.data(), ldvr, stVR, dinfo.data(), bc),
            rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        geev_getError<STRIDED, T>(handle, jobvl, jobvr, n, dA, lda, stA, dWr, dWi, stW,
                                  dVL, ldvl, stVL, dVR, ldvr, stVR, dinfo, bc, hA, hWr,
                                  hWrRes, hWi, hWiRes, hinfo, hinfoRes, &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        geev_getPerfData<STRIDED, T>(handle, jobvl, jobvr, n, dA, lda, stA, dWr, dWi, stW,
                                     dVL, ldvl, stVL, dVR, ldvr, stVR, dinfo, bc, hA, hWr,
                                     hWi, hinfo, &gpu_time_used, &cpu_time_used, hot_calls,
                                     argus.profile, argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using n * n * machine_precision as tolerance
    // (non-symmetric eigenvalue comparison between two independent implementations
    // with different intermediate-precision strategies requires O(n^2) tolerance)
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("jobvl", "jobvr", "n", "lda");
            rocsolver_bench_output(jobvlC, jobvrC, n, lda);
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

INSTANTIATE(EXTERN_TESTING_GEEV,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
