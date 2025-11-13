/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef TESTING_DOTCI_HPP
#define TESTING_DOTCI_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_dotci_bad_arg(const Arguments& argus)
{
#if(!defined(CUDART_VERSION))
    int                  nnz       = 100;
    int                  safe_size = 100;
    hipsparseIndexBase_t idx_base  = HIPSPARSE_INDEX_BASE_ZERO;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto dxVal_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dxInd_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dy_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    T*   dxVal = static_cast<T*>(dxVal_managed.get());
    int* dxInd = static_cast<int*>(dxInd_managed.get());
    T*   dy     = static_cast<T*>(dy_managed.get());

    T result;
    verify_hipsparse_status_invalid_pointer(
        hipsparseXdotci(handle, nnz, static_cast<T*>(nullptr), dxInd, dy, &result, idx_base),
        "Error: x_val is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXdotci(handle, nnz, dxVal, static_cast<int*>(nullptr), dy, &result, idx_base),
        "Error: x_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXdotci(handle, nnz, dxVal, dxInd, static_cast<T*>(nullptr), &result, idx_base),
        "Error: y is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXdotci(handle, nnz, dxVal, dxInd, dy, static_cast<T*>(nullptr), idx_base),
        "Error: result is nullptr");
    verify_hipsparse_status_invalid_handle(
        hipsparseXdotci(static_cast<hipsparseHandle_t>(nullptr), nnz, dxVal, dxInd, dy, &result, idx_base));
#endif
}

template <typename T>
hipsparseStatus_t testing_dotci(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    int                  N        = argus.N;
    int                  nnz      = argus.nnz;
    hipsparseIndexBase_t idx_base = argus.baseA;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Grab stream used by handle
    hipStream_t stream;
    CHECK_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Host structures
    std::vector<int> hxInd(nnz);
    std::vector<T>   hxVal(nnz);
    std::vector<T>   hy(N);

    T hresult_1;
    T hresult_2;
    T hresult_gold;

    // Initial Data on CPU
    srand(12345ULL);
    hipsparseInitIndex(hxInd.data(), nnz, 1, N);
    hipsparseInit<T>(hxVal, 1, nnz);
    hipsparseInit<T>(hy, 1, N);

    // allocate memory on device
    auto dxInd_managed    = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dxVal_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dy_managed        = hipsparse_unique_ptr{device_malloc(sizeof(T) * N), device_free};
    auto dresult_2_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    int* dxInd    = static_cast<int*>(dxInd_managed.get());
    T*   dxVal    = static_cast<T*>(dxVal_managed.get());
    T*   dy        = static_cast<T*>(dy_managed.get());
    T*   dresult_2 = static_cast<T*>(dresult_2_managed.get());

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dxInd, hxInd.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dxVal, hxVal.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * N, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        // HIPSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(
            hipsparseXdotci(handle, nnz, dxVal, dxInd, dy, &hresult_1, idx_base));

        // HIPSPARSE pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(
            hipsparseXdotci(handle, nnz, dxVal, dxInd, dy, dresult_2, idx_base));

        // copy output from device to CPU^
        CHECK_HIP_ERROR(hipMemcpy(&hresult_2, dresult_2, sizeof(T), hipMemcpyDeviceToHost));

        // CPU
        hresult_gold = make_DataType<T>(0.0);
        for(int i = 0; i < nnz; ++i)
        {
            hresult_gold
                = hresult_gold + testing_mult(testing_conj(hxVal[i]), hy[hxInd[i] - idx_base]);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        unit_check_general(1, 1, 1, &hresult_gold, &hresult_1);
        unit_check_general(1, 1, 1, &hresult_gold, &hresult_2);
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        double gpu_time_used = benchmark_kernel(
            [&]() { CHECK_HIPSPARSE_ERROR(
                hipsparseXdotci(handle, nnz, dxVal, dxInd, dy, &hresult_1, idx_base));
            CHECK_HIP_ERROR(hipStreamSynchronize(stream)); return HIPSPARSE_STATUS_SUCCESS; },
            number_cold_calls,
            number_hot_calls);

        double gflop_count = doti_gflop_count(nnz);
        double gbyte_count = doti_gbyte_count<T, T>(nnz);

        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);
        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);

        display_timing_info(display_key_t::nnz,
                            nnz,
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_DOTCI_HPP
