// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "../../shared/gpubuf.h"
#include "../../shared/rocfft_complex.h"
#include "jit_callback_helpers.h"
#include "rocfft/rocfft.h"
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>

// verify SPIR-V callback can be loaded
TEST(rocfft_UnitTest, jit_callback_compilation)
{
    if(!rtc_available())
        GTEST_SKIP() << "SPIR-V callback not available (load_callback.spv missing)";

    EXPECT_NO_THROW({
        auto code = load_spirv("load_callback.spv");
        EXPECT_GT(code.size(), 0u) << "SPIR-V file is empty";

        // validate format
        EXPECT_TRUE(is_valid_spirv(code)) << "Invalid SPIR-V format: " << spirv_format_info(code);
    });
}

// single-precision complex FFT with identity load callback
TEST(rocfft_UnitTest, jit_callback_c2c_single_load)
{
    if(!rtc_available())
        GTEST_SKIP() << "SPIR-V callback not available (load_callback.spv missing)";

    const size_t N = 1024;

    // load pre-compiled SPIR-V callback
    auto load_code = load_spirv("load_callback.spv");

    ASSERT_TRUE(is_valid_spirv(load_code)) << "Invalid SPIR-V format - cannot proceed";

    // create plan description
    rocfft_plan_description desc = nullptr;
    ASSERT_EQ(rocfft_plan_description_create(&desc), rocfft_status_success);

    rocfft_status cb_status = rocfft_plan_description_set_load_callback(
        desc, "load_callback", load_code.data(), load_code.size(), nullptr, 0);

    ASSERT_EQ(cb_status, rocfft_status_success) << "Failed to set callback, status: " << cb_status;

    // next, create the FFT plan with the description that has the callback
    rocfft_plan plan = nullptr;

    // catch any exceptions during plan creation
    try
    {
        rocfft_status plan_status = rocfft_plan_create(&plan,
                                                       rocfft_placement_inplace,
                                                       rocfft_transform_type_complex_forward,
                                                       rocfft_precision_single,
                                                       1,
                                                       &N,
                                                       1,
                                                       desc);

        if(plan_status != rocfft_status_success)
        {
            std::cerr << "ERROR: rocfft_plan_create failed with status " << plan_status
                      << std::endl;
            std::cerr << "This may indicate:" << std::endl;
            std::cerr << "  - Invalid SPIR-V format (check extraction)" << std::endl;
            std::cerr << "  - Symbol 'load_callback' not found in SPIR-V" << std::endl;
            std::cerr << "  - Incompatible callback signature" << std::endl;
        }

        ASSERT_EQ(plan_status, rocfft_status_success)
            << "Failed to create plan, status: " << plan_status;
    }
    catch(const std::exception& e)
    {
        std::cerr << "EXCEPTION during plan creation: " << e.what() << std::endl;
        throw;
    }
    catch(...)
    {
        std::cerr << "UNKNOWN EXCEPTION during plan creation" << std::endl;
        throw;
    }

    // get work buffer size
    size_t work_buffer_size = 0;
    ASSERT_EQ(rocfft_plan_get_work_buffer_size(plan, &work_buffer_size), rocfft_status_success);

    if(work_buffer_size > 0)
        std::cout << "  Work buffer size: " << work_buffer_size << " bytes" << std::endl;

    // allocate work buffer if needed
    gpubuf_t<char> work_buffer;
    void*          work_buffer_ptr = nullptr;
    if(work_buffer_size > 0)
    {
        ASSERT_EQ(work_buffer.alloc(work_buffer_size), hipSuccess);
        work_buffer_ptr = work_buffer.data();
    }

    // allocate and initialize data
    gpubuf_t<rocfft_complex<float>> data;
    ASSERT_EQ(data.alloc(N * sizeof(rocfft_complex<float>)), hipSuccess);

    std::vector<rocfft_complex<float>> host_data(N);
    for(size_t i = 0; i < N; ++i)
    {
        host_data[i].x = static_cast<float>(i);
        host_data[i].y = static_cast<float>(i);
    }

    ASSERT_EQ(hipMemcpy(data.data(),
                        host_data.data(),
                        N * sizeof(rocfft_complex<float>),
                        hipMemcpyHostToDevice),
              hipSuccess);

    // execute FFT with callback
    rocfft_execution_info info = nullptr;
    ASSERT_EQ(rocfft_execution_info_create(&info), rocfft_status_success);

    if(work_buffer_size > 0)
    {
        ASSERT_EQ(rocfft_execution_info_set_work_buffer(info, work_buffer_ptr, work_buffer_size),
                  rocfft_status_success);
    }

    void* buffers[] = {data.data()};
    ASSERT_EQ(rocfft_execute(plan, buffers, nullptr, info), rocfft_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // verify the transform actually ran
    std::vector<rocfft_complex<float>> output(N);
    ASSERT_EQ(
        hipMemcpy(
            output.data(), data.data(), N * sizeof(rocfft_complex<float>), hipMemcpyDeviceToHost),
        hipSuccess);

    bool data_changed = false;
    for(size_t i = 0; i < N; ++i)
    {
        if(std::abs(output[i].x - host_data[i].x) > 1e-5f
           || std::abs(output[i].y - host_data[i].y) > 1e-5f)
        {
            data_changed = true;
            break;
        }
    }
    EXPECT_TRUE(data_changed) << "FFT should have transformed the data";

    // cleanup
    rocfft_execution_info_destroy(info);
    rocfft_plan_destroy(plan);
    rocfft_plan_description_destroy(desc);
}

// helper to check if scaling callback is available
static bool scale_callback_available()
{
    static bool checked   = false;
    static bool available = false;
    if(!checked)
    {
        std::ifstream f("load_callback_scale.spv", std::ios::binary);
        available = f.good();
        checked   = true;
    }
    return available;
}

// scaling callback to verify JIT callback is actually applied.
// FFT is linear, so FFT(2*x) = 2*FFT(x). We verify output is scaled by 2.
TEST(rocfft_UnitTest, jit_callback_scaling_verification)
{
    if(!scale_callback_available())
        GTEST_SKIP() << "Scaling callback not available (load_callback_scale.spv missing)";

    const size_t N = 256;

    // use constant input so that we can predict the output
    // input: all ones -> output[0] (sum of inputs) should be N after FFT
    // with 2x scaling callback: output[0] should be 2*N
    std::vector<rocfft_complex<float>> host_input(N, {1.0f, 0.0f});

    // run FFT without scaling callback to get reference result
    std::vector<rocfft_complex<float>> reference_output(N);
    {
        rocfft_plan plan = nullptr;
        ASSERT_EQ(rocfft_plan_create(&plan,
                                     rocfft_placement_inplace,
                                     rocfft_transform_type_complex_forward,
                                     rocfft_precision_single,
                                     1,
                                     &N,
                                     1,
                                     nullptr),
                  rocfft_status_success);

        gpubuf_t<rocfft_complex<float>> data;
        ASSERT_EQ(data.alloc(N * sizeof(rocfft_complex<float>)), hipSuccess);
        ASSERT_EQ(hipMemcpy(data.data(),
                            host_input.data(),
                            N * sizeof(rocfft_complex<float>),
                            hipMemcpyHostToDevice),
                  hipSuccess);

        void* buffers[] = {data.data()};
        ASSERT_EQ(rocfft_execute(plan, buffers, nullptr, nullptr), rocfft_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        ASSERT_EQ(hipMemcpy(reference_output.data(),
                            data.data(),
                            N * sizeof(rocfft_complex<float>),
                            hipMemcpyDeviceToHost),
                  hipSuccess);

        rocfft_plan_destroy(plan);
    }

    // next, run FFT with 2x scaling callback
    std::vector<rocfft_complex<float>> scaled_output(N);
    {
        auto scale_code = load_spirv("load_callback_scale.spv");
        ASSERT_TRUE(is_valid_spirv(scale_code));

        rocfft_plan_description desc = nullptr;
        ASSERT_EQ(rocfft_plan_description_create(&desc), rocfft_status_success);
        ASSERT_EQ(rocfft_plan_description_set_load_callback(
                      desc, "load_callback", scale_code.data(), scale_code.size(), nullptr, 0),
                  rocfft_status_success);

        rocfft_plan plan = nullptr;
        ASSERT_EQ(rocfft_plan_create(&plan,
                                     rocfft_placement_inplace,
                                     rocfft_transform_type_complex_forward,
                                     rocfft_precision_single,
                                     1,
                                     &N,
                                     1,
                                     desc),
                  rocfft_status_success);

        gpubuf_t<rocfft_complex<float>> data;
        ASSERT_EQ(data.alloc(N * sizeof(rocfft_complex<float>)), hipSuccess);
        ASSERT_EQ(hipMemcpy(data.data(),
                            host_input.data(),
                            N * sizeof(rocfft_complex<float>),
                            hipMemcpyHostToDevice),
                  hipSuccess);

        size_t work_buffer_size = 0;
        ASSERT_EQ(rocfft_plan_get_work_buffer_size(plan, &work_buffer_size), rocfft_status_success);

        gpubuf_t<char>        work_buffer;
        void*                 work_buffer_ptr = nullptr;
        rocfft_execution_info info            = nullptr;
        ASSERT_EQ(rocfft_execution_info_create(&info), rocfft_status_success);

        if(work_buffer_size > 0)
        {
            ASSERT_EQ(work_buffer.alloc(work_buffer_size), hipSuccess);
            work_buffer_ptr = work_buffer.data();
            ASSERT_EQ(
                rocfft_execution_info_set_work_buffer(info, work_buffer_ptr, work_buffer_size),
                rocfft_status_success);
        }

        void* buffers[] = {data.data()};
        ASSERT_EQ(rocfft_execute(plan, buffers, nullptr, info), rocfft_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        ASSERT_EQ(hipMemcpy(scaled_output.data(),
                            data.data(),
                            N * sizeof(rocfft_complex<float>),
                            hipMemcpyDeviceToHost),
                  hipSuccess);

        rocfft_execution_info_destroy(info);
        rocfft_plan_destroy(plan);
        rocfft_plan_description_destroy(desc);
    }

    // verify that scaled_output should be approximately 2 * reference_output
    // due to the 2x scaling in the load callback
    const float tolerance        = 1e-4f;
    bool        scaling_verified = true;
    for(size_t i = 0; i < N; ++i)
    {
        float expected_real = 2.0f * reference_output[i].x;
        float expected_imag = 2.0f * reference_output[i].y;
        float actual_real   = scaled_output[i].x;
        float actual_imag   = scaled_output[i].y;

        if(std::abs(actual_real - expected_real) > tolerance * std::abs(expected_real) + tolerance
           || std::abs(actual_imag - expected_imag)
                  > tolerance * std::abs(expected_imag) + tolerance)
        {
            std::cerr << "Mismatch at index " << i << ": expected (" << expected_real << ", "
                      << expected_imag << "), got (" << actual_real << ", " << actual_imag << ")"
                      << std::endl;
            scaling_verified = false;
            break;
        }
    }

    EXPECT_TRUE(scaling_verified)
        << "JIT callback scaling verification failed - output should be 2x reference";

    // check output[0] specifically (sum of all inputs, easiest to verify)
    // for input of all 1s: output[0] = N without scaling, output[0] = 2*N with scaling
    float ref_bin0    = reference_output[0].x;
    float scaled_bin0 = scaled_output[0].x;
    EXPECT_NEAR(scaled_bin0, 2.0f * ref_bin0, tolerance * std::abs(ref_bin0) + tolerance)
        << "output[0] should be 2x with scaling callback. "
        << "Reference: " << ref_bin0 << ", Scaled: " << scaled_bin0;
}

// test with callback_data
TEST(rocfft_UnitTest, jit_callback_with_cbdata)
{
    if(!rtc_available())
        GTEST_SKIP() << "SPIR-V callback not available";

    const size_t N = 256;

    // load SPIR-V
    auto load_code = load_spirv("load_callback.spv");
    ASSERT_TRUE(is_valid_spirv(load_code));

    // allocate callback data on device
    float scale_factor = 2.0f;
    void* cbdata_dev   = nullptr;
    ASSERT_EQ(hipMalloc(&cbdata_dev, sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemcpy(cbdata_dev, &scale_factor, sizeof(float), hipMemcpyHostToDevice),
              hipSuccess);

    // create description with callback
    rocfft_plan_description desc = nullptr;
    ASSERT_EQ(rocfft_plan_description_create(&desc), rocfft_status_success);

    void* cbdata_array[] = {cbdata_dev};
    ASSERT_EQ(rocfft_plan_description_set_load_callback(
                  desc, "load_callback", load_code.data(), load_code.size(), cbdata_array, 0),
              rocfft_status_success);

    // create plan
    rocfft_plan plan = nullptr;
    ASSERT_EQ(rocfft_plan_create(&plan,
                                 rocfft_placement_inplace,
                                 rocfft_transform_type_complex_forward,
                                 rocfft_precision_single,
                                 1,
                                 &N,
                                 1,
                                 desc),
              rocfft_status_success);

    // execute
    gpubuf_t<rocfft_complex<float>> data;
    ASSERT_EQ(data.alloc(N * sizeof(rocfft_complex<float>)), hipSuccess);

    std::vector<rocfft_complex<float>> host_data(N, {1.0f, 0.0f});
    ASSERT_EQ(hipMemcpy(data.data(),
                        host_data.data(),
                        N * sizeof(rocfft_complex<float>),
                        hipMemcpyHostToDevice),
              hipSuccess);

    rocfft_execution_info info = nullptr;
    ASSERT_EQ(rocfft_execution_info_create(&info), rocfft_status_success);

    void* buffers[] = {data.data()};
    ASSERT_EQ(rocfft_execute(plan, buffers, nullptr, info), rocfft_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // cleanup
    rocfft_execution_info_destroy(info);
    rocfft_plan_destroy(plan);
    rocfft_plan_description_destroy(desc);
    hipFree(cbdata_dev);
}