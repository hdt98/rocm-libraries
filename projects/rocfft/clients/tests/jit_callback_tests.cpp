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

#include "rocfft/rocfft.h"

#include "../../shared/gpubuf.h"
#include "../../shared/params_gen.h"
#include "../../shared/rocfft_complex.h"
#include "jit_callback_helpers.h"
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

// basic JIT callback compilation
TEST(rocfft_UnitTest, jit_callback_compilation)
{
    if(!rtc_available())
        GTEST_SKIP() << "RTC not available";

    // should be able to compile all callback variants
    EXPECT_NO_THROW({
        auto code = compile_callback_to_spirv(callback_sources::load_identity_c);
        EXPECT_GT(code.size(), 0u);
    });

    EXPECT_NO_THROW({
        auto code = compile_callback_to_spirv(callback_sources::load_identity_z);
        EXPECT_GT(code.size(), 0u);
    });

    EXPECT_NO_THROW({
        auto code = compile_callback_to_spirv(callback_sources::load_identity_r);
        EXPECT_GT(code.size(), 0u);
    });
}

// single-precision complex FFT with identity load callback
TEST(rocfft_UnitTest, jit_callback_c2c_single_load)
{
    if(!rtc_available())
        GTEST_SKIP() << "RTC not available";

    const size_t N = 1024;

    // compile load callback
    auto load_code = compile_callback_to_spirv(callback_sources::load_identity_c);

    // create plan description with callback
    rocfft_plan_description desc = nullptr;
    ASSERT_EQ(rocfft_plan_description_create(&desc), rocfft_status_success);

    ASSERT_EQ(rocfft_plan_description_set_load_callback(
                  desc, "load_callback", load_code.data(), load_code.size(), nullptr, 0),
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

    // execute FFT
    rocfft_execution_info info = nullptr;
    ASSERT_EQ(rocfft_execution_info_create(&info), rocfft_status_success);

    void* buffers[] = {data.data()};
    ASSERT_EQ(rocfft_execute(plan, buffers, nullptr, info), rocfft_status_success);

    // verify execution completed
    std::vector<rocfft_complex<float>> output(N);
    ASSERT_EQ(
        hipMemcpy(
            output.data(), data.data(), N * sizeof(rocfft_complex<float>), hipMemcpyDeviceToHost),
        hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // at minimum, verify data changed (FFT was applied)
    bool data_changed = false;
    for(size_t i = 0; i < N; ++i)
    {
        if(std::abs(output[i].x - host_data[i].x) > 1e-5
           || std::abs(output[i].y - host_data[i].y) > 1e-5)
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
