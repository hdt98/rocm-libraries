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

template <typename T, typename I>
void gecon_checkBadArgs(const rocblas_handle handle,
                        const rocsolver_norm_type norm_type,
                        const I n,
                        T dA,
                        const I lda,
                        I* dipiv,
                        const real_t<T>* danorm,
                        real_t<T>* drcond)
{
    using S = real_t<T>;

    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_gecon(nullptr, norm_type, n, dA, lda, dipiv, danorm, drcond),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gecon(handle, static_cast<rocsolver_norm_type>(0), n, dA, lda, dipiv, danorm, drcond),
        rocblas_status_invalid_value);

    // sizes
    EXPECT_ROCBLAS_STATUS(rocsolver_gecon(handle, norm_type, (I)-1, dA, lda, dipiv, danorm, drcond),
                          rocblas_status_invalid_size);
    EXPECT_ROCBLAS_STATUS(rocsolver_gecon(handle, norm_type, n, dA, (I)-1, dipiv, danorm, drcond),
                          rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gecon(handle, norm_type, n, (T) nullptr, lda, dipiv, danorm, drcond),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gecon(handle, norm_type, n, dA, lda, (I*)nullptr, danorm, drcond),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gecon(handle, norm_type, n, dA, lda, dipiv, (S*)nullptr, drcond),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gecon(handle, norm_type, n, dA, lda, dipiv, danorm, (S*)nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gecon(handle, norm_type, (I)0, (T) nullptr, lda, (I*)nullptr, (S*)nullptr, (S*)nullptr),
                          rocblas_status_success);
}

template <typename T, typename I>
void testing_gecon_bad_arg()
{
    using S = real_t<T>;

    // safe arguments
    rocblas_local_handle handle;
    rocsolver_norm_type norm_type = rocsolver_norm_type_one;
    I n = 1;
    I lda = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<I> dipiv(1, 1, 1, 1);
    device_strided_batch_vector<S> danorm(1, 1, 1, 1);
    device_strided_batch_vector<S> drcond(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dipiv.memcheck());
    CHECK_HIP_ERROR(danorm.memcheck());
    CHECK_HIP_ERROR(drcond.memcheck());

    // check bad arguments
    gecon_checkBadArgs(handle, norm_type, n, dA.data(), lda, dipiv.data(), danorm.data(), drcond.data());
}

template <bool CPU, bool GPU, typename T, typename I, typename S, typename Td, typename Tdi, typename Tda, typename Tdr, typename Th, typename Thi, typename Tha, typename Thr>
void gecon_initData(const rocblas_handle handle,
                    const rocsolver_norm_type norm_type,
                    const I n,
                    Td& dA,
                    const I lda,
                    Tdi& dipiv,
                    Tda& danorm,
                    Tdr& drcond,
                    Th& hA,
                    Th& hA_factored,
                    Thi& hipiv,
                    Tha& hanorm,
                    Thr& hrcond,
                    std::vector<I>& hinfo,
                    const bool singular)
{
    if(CPU)
    {
        using S = real_t<T>;

        rocblas_init<T>(hA, true);

        // make matrix non-singular by adding to diagonal if needed
        if(!singular)
        {
            for(I i = 0; i < n; i++)
                hA[0][i + i * lda] += 400;
        }

        // Copy original matrix for factorization
        for(I i = 0; i < n; i++)
            for(I j = 0; j < n; j++)
                hA_factored[0][i + j * lda] = hA[0][i + j * lda];

        // Compute matrix norm before factorization
        std::vector<S> work(n);
        char norm = rocsolver2char_norm_type(norm_type);
        hanorm[0][0] = cpu_lange<T, S>(norm, n, n, hA[0], lda, work.data());

        // Factor matrix with CPU GETRF
        hinfo.resize(1);
        cpu_getrf(n, n, hA_factored[0], lda, hipiv[0], hinfo.data());
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA_factored));
        CHECK_HIP_ERROR(danorm.transfer_from(hanorm));
    }
}

template <typename T, typename I, typename S, typename Td, typename Tdi, typename Tda, typename Tdr, typename Th, typename Thi, typename Tha, typename Thr>
void gecon_getError(const rocblas_handle handle,
                    const rocsolver_norm_type norm_type,
                    const I n,
                    Td& dA,
                    const I lda,
                    Tdi& dipiv,
                    Tda& danorm,
                    Tdr& drcond,
                    Th& hA,
                    Th& hA_factored,
                    Thi& hipiv,
                    Tha& hanorm,
                    Thr& hrcond,
                    Thr& hrcond_res,
                    double* max_err,
                    const bool singular)
{
    std::vector<I> hinfo;

    // initialize data
    gecon_initData<true, true, T, I, S>(handle, norm_type, n, dA, lda, dipiv, danorm, drcond, hA,
                                        hA_factored, hipiv, hanorm, hrcond, hinfo, singular);

    // On GPU: need to run GETRF first to get factorization and ipiv
    device_strided_batch_vector<I> dinfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dinfo.memcheck());
    CHECK_ROCBLAS_ERROR(
        rocsolver_getrf(handle, n, n, dA.data(), lda, dipiv.data(), dinfo.data()));

    // execute computations: GPU GECON
    CHECK_ROCBLAS_ERROR(
        rocsolver_gecon(handle, norm_type, n, dA.data(), lda, dipiv.data(), danorm.data(), drcond.data()));
    CHECK_HIP_ERROR(hrcond_res.transfer_from(drcond));

    // CPU lapack
    char norm = rocsolver2char_norm_type(norm_type);
    std::vector<S> work(4 * n);
    std::vector<I> iwork(n);
    hrcond[0][0] = cpu_gecon<T, S>(norm, n, hA_factored[0], lda, hanorm[0][0], work.data(), iwork.data());

    // error is ||hrcond - hrcond_res|| / ||hrcond||
    // using absolute value since we only have a single scalar
    *max_err = std::abs(hrcond[0][0] - hrcond_res[0][0]) / std::abs(hrcond[0][0]);
}

template <typename T, typename I, typename S, typename Td, typename Tdi, typename Tda, typename Tdr, typename Th, typename Thi, typename Tha, typename Thr>
void gecon_getPerfData(const rocblas_handle handle,
                       const rocsolver_norm_type norm_type,
                       const I n,
                       Td& dA,
                       const I lda,
                       Tdi& dipiv,
                       Tda& danorm,
                       Tdr& drcond,
                       Th& hA,
                       Th& hA_factored,
                       Thi& hipiv,
                       Tha& hanorm,
                       Thr& hrcond,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    std::vector<I> hinfo;

    if(!perf)
    {
        gecon_initData<true, false, T, I, S>(handle, norm_type, n, dA, lda, dipiv, danorm, drcond,
                                             hA, hA_factored, hipiv, hanorm, hrcond, hinfo, false);

        // cpu-lapack performance (only if not in perf mode)
        char norm = rocsolver2char_norm_type(norm_type);
        std::vector<S> work(4 * n);
        std::vector<I> iwork(n);
        *cpu_time_used = get_time_us_no_sync();
        hrcond[0][0]
            = cpu_gecon<T, S>(norm, n, hA_factored[0], lda, hanorm[0][0], work.data(), iwork.data());
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gecon_initData<true, false, T, I, S>(handle, norm_type, n, dA, lda, dipiv, danorm, drcond, hA,
                                         hA_factored, hipiv, hanorm, hrcond, hinfo, false);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gecon_initData<false, true, T, I, S>(handle, norm_type, n, dA, lda, dipiv, danorm, drcond,
                                             hA, hA_factored, hipiv, hanorm, hrcond, hinfo, false);

        // Need to factor on GPU for each call
        device_strided_batch_vector<I> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dinfo.memcheck());
        CHECK_ROCBLAS_ERROR(
            rocsolver_getrf(handle, n, n, dA.data(), lda, dipiv.data(), dinfo.data()));

        CHECK_ROCBLAS_ERROR(rocsolver_gecon(handle, norm_type, n, dA.data(), lda, dipiv.data(),
                                            danorm.data(), drcond.data()));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double start;

    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(int iter = 0; iter < hot_calls; iter++)
    {
        gecon_initData<false, true, T, I, S>(handle, norm_type, n, dA, lda, dipiv, danorm, drcond,
                                             hA, hA_factored, hipiv, hanorm, hrcond, hinfo, false);

        // Need to factor on GPU for each call
        device_strided_batch_vector<I> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dinfo.memcheck());
        CHECK_ROCBLAS_ERROR(
            rocsolver_getrf(handle, n, n, dA.data(), lda, dipiv.data(), dinfo.data()));

        start = get_time_us_sync(stream);
        rocsolver_gecon(handle, norm_type, n, dA.data(), lda, dipiv.data(), danorm.data(), drcond.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T, typename I>
void testing_gecon(Arguments& argus)
{
    using S = real_t<T>;

    // get arguments
    rocblas_local_handle handle;
    char norm_typeC = argus.get<char>("norm_type");
    I n = argus.get<I>("n");
    I lda = argus.get<I>("lda", n);

    rocsolver_norm_type norm_type = char2rocsolver_norm_type(norm_typeC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(norm_type != rocsolver_norm_type_one && norm_type != rocsolver_norm_type_infinity
       && norm_type != rocsolver_norm_type_frobenius && norm_type != rocsolver_norm_type_max)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_gecon(handle, norm_type, n, (T*)nullptr, lda, (I*)nullptr,
                                              (S*)nullptr, (S*)nullptr),
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // check for not_implemented for Frobenius and max norms
    if(norm_type == rocsolver_norm_type_frobenius || norm_type == rocsolver_norm_type_max)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_gecon(handle, norm_type, n, (T*)nullptr, lda, (I*)nullptr,
                                              (S*)nullptr, (S*)nullptr),
                              rocblas_status_not_implemented);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_ipiv = n;
    size_t size_anorm = 1;
    size_t size_rcond = 1;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_rcond_res = (argus.unit_check || argus.norm_check) ? size_rcond : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_gecon(handle, norm_type, n, (T*)nullptr, lda, (I*)nullptr,
                                              (S*)nullptr, (S*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_gecon(handle, norm_type, n, (T*)nullptr, lda, (I*)nullptr,
                                          (S*)nullptr, (S*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hA_factored(size_A, 1, size_A, 1);
    host_strided_batch_vector<I> hipiv(size_ipiv, 1, size_ipiv, 1);
    host_strided_batch_vector<S> hanorm(size_anorm, 1, size_anorm, 1);
    host_strided_batch_vector<S> hrcond(size_rcond, 1, size_rcond, 1);
    host_strided_batch_vector<S> hrcond_res(size_rcond_res, 1, size_rcond_res, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<I> dipiv(size_ipiv, 1, size_ipiv, 1);
    device_strided_batch_vector<S> danorm(size_anorm, 1, size_anorm, 1);
    device_strided_batch_vector<S> drcond(size_rcond, 1, size_rcond, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_ipiv)
        CHECK_HIP_ERROR(dipiv.memcheck());
    CHECK_HIP_ERROR(danorm.memcheck());
    CHECK_HIP_ERROR(drcond.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_gecon(handle, norm_type, n, dA.data(), lda, dipiv.data(), danorm.data(), drcond.data()),
            rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        gecon_getError<T, I, S>(handle, norm_type, n, dA, lda, dipiv, danorm, drcond, hA,
                                hA_factored, hipiv, hanorm, hrcond, hrcond_res, &max_error, false);

    // collect performance data
    if(argus.timing)
        gecon_getPerfData<T, I, S>(handle, norm_type, n, dA, lda, dipiv, danorm, drcond, hA,
                                   hA_factored, hipiv, hanorm, hrcond, &gpu_time_used, &cpu_time_used,
                                   hot_calls, argus.profile, argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("norm_type", "n", "lda");
            rocsolver_bench_output(norm_typeC, n, lda);

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

#define EXTERN_TESTING_GECON(...) extern template void testing_gecon<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GECON, FOREACH_SCALAR_TYPE, FOREACH_INT_TYPE, APPLY_STAMP)
