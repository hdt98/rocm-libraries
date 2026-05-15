// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// API smoke tests for rocfft_plan_description_set_precision_triple plus
// GPU round-trip tests for the cast and BFP compression backends.

#include "compress_backend.h"
#include "rocfft/rocfft.h"
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <random>
#include <vector>

namespace
{

    class PrecisionTripleTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            ASSERT_EQ(rocfft_setup(), rocfft_status_success);
            ASSERT_EQ(rocfft_plan_description_create(&desc), rocfft_status_success);
        }

        void TearDown() override
        {
            if(desc)
            {
                EXPECT_EQ(rocfft_plan_description_destroy(desc), rocfft_status_success);
                desc = nullptr;
            }
            EXPECT_EQ(rocfft_cleanup(), rocfft_status_success);
        }

        rocfft_plan_description desc = nullptr;
    };

    TEST_F(PrecisionTripleTest, default_is_native)
    {
        ASSERT_EQ(rocfft_plan_description_set_precision_triple(
                      desc, rocfft_compute_precision_native, rocfft_comm_precision_native, 0u),
                  rocfft_status_success);
    }

    TEST_F(PrecisionTripleTest, accepts_cast_fp16)
    {
        ASSERT_EQ(rocfft_plan_description_set_precision_triple(
                      desc, rocfft_compute_precision_native, rocfft_comm_precision_cast_fp16, 0u),
                  rocfft_status_success);
    }

    TEST_F(PrecisionTripleTest, accepts_cast_bf16)
    {
        ASSERT_EQ(rocfft_plan_description_set_precision_triple(
                      desc, rocfft_compute_precision_native, rocfft_comm_precision_cast_bf16, 0u),
                  rocfft_status_success);
    }

    TEST_F(PrecisionTripleTest, accepts_bfp_with_param)
    {
        for(unsigned int bits : {0u, 4u, 8u, 16u, 23u})
        {
            EXPECT_EQ(rocfft_plan_description_set_precision_triple(
                          desc, rocfft_compute_precision_native, rocfft_comm_precision_bfp, bits),
                      rocfft_status_success)
                << "bfp param = " << bits;
        }
    }

    TEST_F(PrecisionTripleTest, rejects_bfp_with_param_too_large)
    {
        EXPECT_EQ(rocfft_plan_description_set_precision_triple(
                      desc, rocfft_compute_precision_native, rocfft_comm_precision_bfp, 24u),
                  rocfft_status_invalid_arg_value);
    }

    TEST_F(PrecisionTripleTest, accepts_zfp_fixed_rate_with_param)
    {
        for(unsigned int bits : {0u, 2u, 8u, 16u, 32u})
        {
            EXPECT_EQ(
                rocfft_plan_description_set_precision_triple(desc,
                                                             rocfft_compute_precision_native,
                                                             rocfft_comm_precision_zfp_fixed_rate,
                                                             bits),
                rocfft_status_success)
                << "zfp param = " << bits;
        }
    }

    TEST_F(PrecisionTripleTest, rejects_zfp_with_param_too_large)
    {
        EXPECT_EQ(
            rocfft_plan_description_set_precision_triple(
                desc, rocfft_compute_precision_native, rocfft_comm_precision_zfp_fixed_rate, 33u),
            rocfft_status_invalid_arg_value);
    }

    TEST_F(PrecisionTripleTest, rejects_unknown_compute_precision)
    {
        constexpr int bogus = 999;
        EXPECT_EQ(rocfft_plan_description_set_precision_triple(
                      desc,
                      static_cast<rocfft_compute_precision>(bogus),
                      rocfft_comm_precision_native,
                      0u),
                  rocfft_status_invalid_arg_value);
    }

    TEST_F(PrecisionTripleTest, rejects_unknown_comm_precision)
    {
        constexpr int bogus = 999;
        EXPECT_EQ(
            rocfft_plan_description_set_precision_triple(desc,
                                                         rocfft_compute_precision_native,
                                                         static_cast<rocfft_comm_precision>(bogus),
                                                         0u),
            rocfft_status_invalid_arg_value);
    }

    TEST_F(PrecisionTripleTest, rejects_null_description)
    {
        EXPECT_EQ(
            rocfft_plan_description_set_precision_triple(
                nullptr, rocfft_compute_precision_native, rocfft_comm_precision_cast_fp16, 0u),
            rocfft_status_invalid_arg_value);
    }

    TEST_F(PrecisionTripleTest, accepts_all_compute_precisions)
    {
        for(auto cp : {rocfft_compute_precision_native,
                       rocfft_compute_precision_fp32_on_fp16,
                       rocfft_compute_precision_fp32_on_bf16,
                       rocfft_compute_precision_fp64_on_fp32})
        {
            EXPECT_EQ(rocfft_plan_description_set_precision_triple(
                          desc, cp, rocfft_comm_precision_native, 0u),
                      rocfft_status_success)
                << "compute precision = " << static_cast<int>(cp);
        }
    }

    // triple is accepted before single-rank plan_create and the plan still builds
    TEST_F(PrecisionTripleTest, integrates_with_plan_create_1d)
    {
        ASSERT_EQ(rocfft_plan_description_set_precision_triple(
                      desc, rocfft_compute_precision_native, rocfft_comm_precision_cast_fp16, 0u),
                  rocfft_status_success);

        rocfft_plan  plan   = nullptr;
        const size_t length = 256;
        auto         rc     = rocfft_plan_create(&plan,
                                     rocfft_placement_inplace,
                                     rocfft_transform_type_complex_forward,
                                     rocfft_precision_single,
                                     1,
                                     &length,
                                     1,
                                     desc);
        ASSERT_EQ(rc, rocfft_status_success);
        ASSERT_NE(plan, nullptr);
        EXPECT_EQ(rocfft_plan_destroy(plan), rocfft_status_success);
    }

    // per-backend GPU round-trip tests
    class CompressRoundTripTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if(hipSetDevice(0) != hipSuccess)
                GTEST_SKIP() << "no HIP device available";
        }

        static void* alloc_zeroed_device(size_t size_bytes)
        {
            void* p = nullptr;
            if(hipMalloc(&p, size_bytes) != hipSuccess)
                return nullptr;
            if(hipMemset(p, 0, size_bytes) != hipSuccess)
            {
                (void)hipFree(p);
                return nullptr;
            }
            return p;
        }

        // copy host input to device, compress, decompress, return decoded host vector
        static std::vector<float> round_trip_fp32(rocfft::compress::CompressBackend& backend,
                                                  const std::vector<float>&          host_input)
        {
            const size_t n           = host_input.size();
            const size_t input_bytes = n * sizeof(float);
            const size_t comp_bytes  = backend.compressed_bytes(rocfft_precision_single, n);

            void*              d_in   = nullptr;
            void*              d_comp = nullptr;
            void*              d_out  = nullptr;
            std::vector<float> host_output;

            if(hipMalloc(&d_in, input_bytes) != hipSuccess)
                return host_output;
            if(hipMalloc(&d_comp, comp_bytes) != hipSuccess)
            {
                (void)hipFree(d_in);
                return host_output;
            }
            if(hipMalloc(&d_out, input_bytes) != hipSuccess)
            {
                (void)hipFree(d_in);
                (void)hipFree(d_comp);
                return host_output;
            }

            if(hipMemcpy(d_in, host_input.data(), input_bytes, hipMemcpyHostToDevice) != hipSuccess)
                goto cleanup;
            if(backend.compress(d_in, rocfft_precision_single, n, d_comp, /*stream=*/0)
               != rocfft_status_success)
                goto cleanup;
            if(hipDeviceSynchronize() != hipSuccess)
                goto cleanup;
            if(backend.decompress(d_comp, d_out, rocfft_precision_single, n, /*stream=*/0)
               != rocfft_status_success)
                goto cleanup;
            if(hipDeviceSynchronize() != hipSuccess)
                goto cleanup;

            host_output.assign(n, 0.0f);
            if(hipMemcpy(host_output.data(), d_out, input_bytes, hipMemcpyDeviceToHost)
               != hipSuccess)
                host_output.clear();

        cleanup:
            (void)hipFree(d_in);
            (void)hipFree(d_comp);
            (void)hipFree(d_out);
            return host_output;
        }

        static std::vector<float> uniform_floats(size_t n, float lo, float hi, unsigned seed = 1)
        {
            std::mt19937                          rng(seed);
            std::uniform_real_distribution<float> dist(lo, hi);
            std::vector<float>                    v(n);
            for(auto& x : v)
                x = dist(rng);
            return v;
        }
    };

    TEST_F(CompressRoundTripTest, cast_fp16_round_trip)
    {
        auto backend = rocfft::compress::make_compress_backend(rocfft_comm_precision_cast_fp16,
                                                               /*param=*/0u);
        ASSERT_NE(backend, nullptr);

        constexpr size_t n     = 1024;
        auto             input = uniform_floats(n, -1.0f, 1.0f);
        auto             out   = round_trip_fp32(*backend, input);
        ASSERT_EQ(out.size(), n);

        // FP16: 11-bit mantissa, rel error ~ 2^(-10) ~= 1e-3
        float max_relative = 0.0f;
        for(size_t i = 0; i < n; ++i)
        {
            const float denom = std::max(std::fabs(input[i]), 1e-6f);
            const float rel   = std::fabs(out[i] - input[i]) / denom;
            max_relative      = std::max(max_relative, rel);
        }
        EXPECT_LT(max_relative, 2e-3f) << "FP16 round-trip relative error too large";
    }

    TEST_F(CompressRoundTripTest, cast_bf16_round_trip)
    {
        auto backend = rocfft::compress::make_compress_backend(rocfft_comm_precision_cast_bf16,
                                                               /*param=*/0u);
        ASSERT_NE(backend, nullptr);

        constexpr size_t n     = 1024;
        auto             input = uniform_floats(n, -1.0f, 1.0f, /*seed=*/2);
        auto             out   = round_trip_fp32(*backend, input);
        ASSERT_EQ(out.size(), n);

        // BF16: 8-bit mantissa, rel error ~ 2^(-7) ~= 8e-3
        float max_relative = 0.0f;
        for(size_t i = 0; i < n; ++i)
        {
            const float denom = std::max(std::fabs(input[i]), 1e-6f);
            const float rel   = std::fabs(out[i] - input[i]) / denom;
            max_relative      = std::max(max_relative, rel);
        }
        EXPECT_LT(max_relative, 1e-2f) << "BF16 round-trip relative error too large";
    }

    TEST_F(CompressRoundTripTest, bfp_m8_round_trip)
    {
        auto backend
            = rocfft::compress::make_compress_backend(rocfft_comm_precision_bfp, /*param=*/8u);
        ASSERT_NE(backend, nullptr);

        constexpr size_t n     = 1024;
        auto             input = uniform_floats(n, -1.0f, 1.0f, /*seed=*/3);
        auto             out   = round_trip_fp32(*backend, input);
        ASSERT_EQ(out.size(), n);

        // BFP m=8: block-relative error ~ 2^(-7) ~= 8e-3
        float max_block_relative = 0.0f;
        for(size_t blk = 0; blk * 32 < n; ++blk)
        {
            float block_maxabs = 0.0f;
            for(size_t k = 0; k < 32 && blk * 32 + k < n; ++k)
                block_maxabs = std::max(block_maxabs, std::fabs(input[blk * 32 + k]));
            const float denom = std::max(block_maxabs, 1e-6f);
            for(size_t k = 0; k < 32 && blk * 32 + k < n; ++k)
            {
                const size_t i     = blk * 32 + k;
                const float  rel   = std::fabs(out[i] - input[i]) / denom;
                max_block_relative = std::max(max_block_relative, rel);
            }
        }
        EXPECT_LT(max_block_relative, 1e-2f) << "BFP m=8 block-relative round-trip error too large";
    }

    TEST_F(CompressRoundTripTest, bfp_m16_round_trip)
    {
        auto backend
            = rocfft::compress::make_compress_backend(rocfft_comm_precision_bfp, /*param=*/16u);
        ASSERT_NE(backend, nullptr);

        constexpr size_t n     = 1024;
        auto             input = uniform_floats(n, -1.0f, 1.0f, /*seed=*/4);
        auto             out   = round_trip_fp32(*backend, input);
        ASSERT_EQ(out.size(), n);

        // BFP m=16: block-relative error ~ 2^(-15) ~= 3e-5
        float max_block_relative = 0.0f;
        for(size_t blk = 0; blk * 32 < n; ++blk)
        {
            float block_maxabs = 0.0f;
            for(size_t k = 0; k < 32 && blk * 32 + k < n; ++k)
                block_maxabs = std::max(block_maxabs, std::fabs(input[blk * 32 + k]));
            const float denom = std::max(block_maxabs, 1e-6f);
            for(size_t k = 0; k < 32 && blk * 32 + k < n; ++k)
            {
                const size_t i     = blk * 32 + k;
                const float  rel   = std::fabs(out[i] - input[i]) / denom;
                max_block_relative = std::max(max_block_relative, rel);
            }
        }
        EXPECT_LT(max_block_relative, 5e-5f)
            << "BFP m=16 block-relative round-trip error too large";
    }

    TEST_F(CompressRoundTripTest, bfp_m4_round_trip)
    {
        auto backend
            = rocfft::compress::make_compress_backend(rocfft_comm_precision_bfp, /*param=*/4u);
        ASSERT_NE(backend, nullptr);

        constexpr size_t n     = 1024;
        auto             input = uniform_floats(n, -1.0f, 1.0f, /*seed=*/5);
        auto             out   = round_trip_fp32(*backend, input);
        ASSERT_EQ(out.size(), n);

        // BFP m=4: block-relative error ~ 2^(-3) ~= 0.125
        float max_block_relative = 0.0f;
        for(size_t blk = 0; blk * 32 < n; ++blk)
        {
            float block_maxabs = 0.0f;
            for(size_t k = 0; k < 32 && blk * 32 + k < n; ++k)
                block_maxabs = std::max(block_maxabs, std::fabs(input[blk * 32 + k]));
            const float denom = std::max(block_maxabs, 1e-6f);
            for(size_t k = 0; k < 32 && blk * 32 + k < n; ++k)
            {
                const size_t i     = blk * 32 + k;
                const float  rel   = std::fabs(out[i] - input[i]) / denom;
                max_block_relative = std::max(max_block_relative, rel);
            }
        }
        EXPECT_LT(max_block_relative, 0.13f) << "BFP m=4 block-relative round-trip error too large";
    }

    TEST_F(CompressRoundTripTest, bfp_handles_partial_trailing_block)
    {
        // n = 1000 leaves a partial trailing BFP block of 8 elements
        auto backend
            = rocfft::compress::make_compress_backend(rocfft_comm_precision_bfp, /*param=*/8u);
        ASSERT_NE(backend, nullptr);

        const size_t n     = 1000;
        auto         input = uniform_floats(n, -10.0f, 10.0f, /*seed=*/6);
        auto         out   = round_trip_fp32(*backend, input);
        ASSERT_EQ(out.size(), n);

        float block_maxabs = 0.0f;
        for(size_t i = (n / 32) * 32; i < n; ++i)
            block_maxabs = std::max(block_maxabs, std::fabs(input[i]));
        const float denom = std::max(block_maxabs, 1e-6f);
        for(size_t i = (n / 32) * 32; i < n; ++i)
        {
            const float rel = std::fabs(out[i] - input[i]) / denom;
            EXPECT_LT(rel, 1e-2f) << "BFP m=8 trailing block error at i=" << i;
        }
    }

    TEST_F(CompressRoundTripTest, bfp_handles_all_zero_block)
    {
        auto backend
            = rocfft::compress::make_compress_backend(rocfft_comm_precision_bfp, /*param=*/8u);
        ASSERT_NE(backend, nullptr);

        constexpr size_t   n = 64;
        std::vector<float> input(n, 0.0f);
        auto               out = round_trip_fp32(*backend, input);
        ASSERT_EQ(out.size(), n);
        for(size_t i = 0; i < n; ++i)
            EXPECT_FLOAT_EQ(out[i], 0.0f) << "BFP m=8 all-zero block did not round-trip at i=" << i;
    }

} // namespace
