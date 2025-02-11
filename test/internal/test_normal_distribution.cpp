// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>
#include <stdio.h>

#include <random>

#include <rng/distribution/normal.hpp>

using namespace rocrand_impl::host;

TEST(normal_distribution_tests, float_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t               size = 4000;
    float                      val[size];
    normal_distribution<float> u(2.0f, 5.0f);

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[2];
        float        output[2];
        input[0] = dis(gen);
        input[1] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += output[0] + output[1];
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0f, mean, 0.4f); // 20%
    EXPECT_NEAR(5.0f, std, 1.0f); // 20%
}

TEST(normal_distribution_tests, double_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t                size = 4000;
    double                      val[size];
    normal_distribution<double> u(2.0, 5.0);

    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[4];
        double       output[2];
        input[0] = dis(gen);
        input[1] = dis(gen);
        input[2] = dis(gen);
        input[3] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += output[0] + output[1];
    }
    mean = mean / size;

    // Calculate stddev
    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0, mean, 0.4); // 20%
    EXPECT_NEAR(5.0, std, 1.0); // 20%
}

TEST(normal_distribution_tests, float_out_uint2_in_test)
{

    struct nd
    {
        const float mean;
        const float stddev;

        nd(float mean, float stddev) : mean(mean), stddev(stddev) {}

        __forceinline__ __host__ __device__
        void operator()(const uint2 &input, float (&output)[2]) const
        {
            float2 v  = rocrand_device::detail::normal_distribution2(input);
            output[0] = mean + v.x * stddev;
            output[1] = mean + v.y * stddev;
        }
    };

    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t size = 4000;
    float        val[size];
    nd           u(2.0, 5.0);

    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i += 2)
    {
        uint2     input;
        float     output[2];
        input.x = dis(gen);
        input.y = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += (output[0] + output[1]);
    }
    mean /= size;

    // Calculate stddev
    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0, mean, 0.4); // 20%
    EXPECT_NEAR(5.0, std, 1.0); // 20%
}

TEST(normal_distribution_tests, float4_out_uint4_in_test)
{

    struct nd
    {
        const float mean;
        const float stddev;

        nd(float mean, float stddev) : mean(mean), stddev(stddev) {}

        __forceinline__ __host__ __device__
        void operator()(const uint4 &input, float4 &output) const
        {
            float4 v  = rocrand_device::detail::normal_distribution4(input);
            output.w = mean + v.w * stddev;
            output.x = mean + v.x * stddev;
            output.y = mean + v.y * stddev;
            output.z = mean + v.z * stddev;
        }
    };

    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t size = 4000;
    float        val[size];
    nd           u(2.0, 5.0);

    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i += 4)
    {
        uint4     input;
        float4     output;
        input.w = dis(gen);
        input.x = dis(gen);
        input.y = dis(gen);
        input.z = dis(gen);
        u(input, output);
        val[i]         = output.w;
        val[i + 1]     = output.x;
        val[i + 2]     = output.y;
        val[i + 3]     = output.z;
        mean += (output.w + output.x + output.y + output.z);
    }
    mean /= size;

    // Calculate stddev
    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0, mean, 0.4); // 20%
    EXPECT_NEAR(5.0, std, 1.0); // 20%
}

TEST(normal_distribution_tests, half_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t              size = 4000;
    half                      val[size];
    normal_distribution<half> u(__float2half(2.0f), __float2half(5.0f));

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[1];
        half         output[2];
        input[0] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += __half2float(output[0]) + __half2float(output[1]);
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(__half2float(val[i]) - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0f, mean, 0.4f); // 20%
    EXPECT_NEAR(5.0f, std, 1.0f); // 20%
}

template<typename mrg, unsigned int m1>
struct mrg_normal_distribution_test_type
{
    typedef mrg                   mrg_type;
    static constexpr unsigned int mrg_m1 = m1;
};

template<typename test_type>
struct mrg_normal_distribution_tests : public ::testing::Test
{
    typedef typename test_type::mrg_type mrg_type;
    static constexpr unsigned int        mrg_m1 = test_type::mrg_m1;
};

typedef ::testing::Types<
    mrg_normal_distribution_test_type<rocrand_state_mrg31k3p, ROCRAND_MRG31K3P_M1>,
    mrg_normal_distribution_test_type<rocrand_state_mrg32k3a, ROCRAND_MRG32K3A_M1>>
    mrg_normal_distribution_test_types;

TYPED_TEST_SUITE(mrg_normal_distribution_tests, mrg_normal_distribution_test_types);

TYPED_TEST(mrg_normal_distribution_tests, float_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, TestFixture::mrg_m1);

    const size_t                                                          size = 4000;
    float                                                                 val[size];
    mrg_engine_normal_distribution<float, typename TestFixture::mrg_type> u(2.0f, 5.0f);

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[2];
        float        output[2];
        input[0] = dis(gen);
        input[1] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += output[0] + output[1];
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0f, mean, 0.4f); // 20%
    EXPECT_NEAR(5.0f, std, 1.0f); // 20%
}

TYPED_TEST(mrg_normal_distribution_tests, double_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, TestFixture::mrg_m1);

    const size_t                                                           size = 4000;
    double                                                                 val[size];
    mrg_engine_normal_distribution<double, typename TestFixture::mrg_type> u(2.0, 5.0);

    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[2];
        double       output[2];
        input[0] = dis(gen);
        input[1] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += output[0] + output[1];
    }
    mean = mean / size;

    // Calculate stddev
    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0, mean, 0.4); // 20%
    EXPECT_NEAR(5.0, std, 1.0); // 20%
}

TYPED_TEST(mrg_normal_distribution_tests, half_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, TestFixture::mrg_m1);

    const size_t                                                         size = 4000;
    half                                                                 val[size];
    mrg_engine_normal_distribution<half, typename TestFixture::mrg_type> u(__float2half(2.0f),
                                                                           __float2half(5.0f));

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[1];
        half         output[2];
        input[0] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += __half2float(output[0]) + __half2float(output[1]);
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(__half2float(val[i]) - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0f, mean, 0.4f); // 20%
    EXPECT_NEAR(5.0f, std, 1.0f); // 20%
}

TEST(sobol_normal_distribution_tests, float_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t                     size = 4000;
    float                            val[size];
    sobol_normal_distribution<float> u(2.0f, 5.0f);

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 1)
    {
        unsigned int input[1];
        float        output[1];
        input[0]  = dis(gen);
        output[0] = u(input[0]);
        val[i]    = output[0];
        mean += output[0];
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0f, mean, 0.4f); // 20%
    EXPECT_NEAR(5.0f, std, 1.0f); // 20%
}

TEST(sobol_normal_distribution_tests, double_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t                      size = 4000;
    double                            val[size];
    sobol_normal_distribution<double> u(2.0, 5.0);

    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i += 1)
    {
        unsigned int input[1];
        double       output[1];
        input[0]  = dis(gen);
        output[0] = u(input[0]);
        val[i]    = output[0];
        mean += output[0];
    }
    mean = mean / size;

    // Calculate stddev
    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0, mean, 0.4); // 20%
    EXPECT_NEAR(5.0, std, 1.0); // 20%
}

TEST(sobol_normal_distribution_tests, half_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t                    size = 4000;
    half                            val[size];
    sobol_normal_distribution<half> u(__float2half(2.0f), __float2half(5.0f));

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 1)
    {
        unsigned int input[1];
        half         output[1];
        input[0]  = dis(gen);
        output[0] = u(input[0]);
        val[i]    = output[0];
        mean += __half2float(output[0]);
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(__half2float(val[i]) - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0f, mean, 0.4f); // 20%
    EXPECT_NEAR(5.0f, std, 1.0f); // 20%
}
