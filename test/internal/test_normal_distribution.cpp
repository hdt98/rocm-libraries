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


TEST(normal_distribution_tests, float2_out_longlong_in_test)
{
    
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t size = 4000;
    float        val[size];
    normal_distribution<float, unsigned long long> u(2, 5);
    
    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned long long input[1];
        float    output[2];
        unsigned long long l = static_cast<unsigned long long>(dis(gen));
        unsigned long long r = static_cast<unsigned long long>(dis(gen));
        input[0] = (l << 32) | r;
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
        std += std::pow(val[i] - mean, 2) / size;
    }
    std = std::sqrt(std);
    
    EXPECT_NEAR(2, mean, 0.4) << "Mean: " << mean << " Expected: " << 2; // 20%
    EXPECT_NEAR(5, std, 1.0) <<  "Stddev: " << std << " Expected: " << 5; // 20%
}

TEST(normal_distribution_tests, float4_outputs){
    struct normal_distribution_float_4_out{
        const float mean;
        const float stddev;

        normal_distribution_float_4_out(float mean, float stddev) : mean(mean), stddev(stddev) {}

        __forceinline__ __host__ __device__
        void uint4_in(const uint4 &input, float (&output)[4]) const
        {
            float4 v  = rocrand_device::detail::normal_distribution4(input);
            output[0] = mean + v.w * stddev; 
            output[1] = mean + v.x * stddev;
            output[2] = mean + v.y * stddev;
            output[3] = mean + v.z * stddev;
        }

        __forceinline__ __host__ __device__
        void longlong2_in(const longlong2 &input, float (&output)[4]) const
        {
            float4 v  = rocrand_device::detail::normal_distribution4(input);
            output[0] = mean + v.w * stddev; 
            output[1] = mean + v.x * stddev;
            output[2] = mean + v.y * stddev;
            output[3] = mean + v.z * stddev;
        }

        __forceinline__ __host__ __device__
        void ull_2_in(const unsigned long long (&input)[2], float (&output)[4]) const
        {
            float4 v  = rocrand_device::detail::normal_distribution4(input[0], input[1]);
            output[0] = mean + v.w * stddev; 
            output[1] = mean + v.x * stddev;
            output[2] = mean + v.y * stddev;
            output[3] = mean + v.z * stddev;
        }
    };
    
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t size = 4000;
    float        vull2[size], vll2[size], vui4[size];
    normal_distribution_float_4_out u(2, 5);

    // Calculate mean
    double mui4 = 0, mll2 = 0, mull2 = 0;
    for(size_t i = 0; i < size; i += 4)
    {
        unsigned long long ull2[2];
        longlong2          ll2;
        uint4              ui4;
        float     output[4];
        unsigned long long l = static_cast<unsigned long long>(dis(gen));
        unsigned long long r = static_cast<unsigned long long>(dis(gen));
        
        unsigned long long in1 = (l << 32) | r;
    
        ull2[0] = in1;
        ll2.x = in1;
        ui4.w = l; ui4.x = r;

        l = static_cast<unsigned long long>(dis(gen));
        r = static_cast<unsigned long long>(dis(gen));
        unsigned long long in2 = (l << 32) | r;

        ull2[1] = in2;
        ll2.y = in2;
        ui4.y = l; ui4.z = r;

        u.uint4_in(ui4, output);
        vui4[i] = output[0]; 
        vui4[i + 1] = output[1]; 
        vui4[i + 2] = output[2]; 
        vui4[i + 3] = output[3];
        mui4 += (output[0] + output[1] + output[2] + output[3]) / size;

        u.longlong2_in(ll2, output);
        vll2[i] = output[0]; 
        vll2[i + 1] = output[1]; 
        vll2[i + 2] = output[2]; 
        vll2[i + 3] = output[3];
        mll2 += (output[0] + output[1] + output[2] + output[3]) / size;

        u.ull_2_in(ull2, output);
        vull2[i] = output[0]; 
        vull2[i + 1] = output[1]; 
        vull2[i + 2] = output[2]; 
        vull2[i + 3] = output[3];
        mull2 += (output[0] + output[1] + output[2] + output[3]) / size;
    }

    double sui4 = 0, sll2 = 0, sull2 = 0;
    for(size_t i = 0; i < size; i++)
    {
        sui4 += std::pow(vui4[i] - mui4, 2);
        sll2 += std::pow(vll2[i] - mll2, 2);
        sull2 += std::pow(vull2[i] - mull2, 2);
    }

    sui4 = sqrt(sui4 / size);
    sll2 = sqrt(sll2 / size);
    sull2 = sqrt(sull2 / size);

    EXPECT_NEAR(2, mui4, 0.4) << "Mean: " << mui4 << " Expected: " << 2; // 20%
    EXPECT_NEAR(5, sui4, 1.0) <<  "Stddev: " << sui4 << " Expected: " << 5; // 20%

    EXPECT_NEAR(2, mll2, 0.4) << "Mean: " << mll2 << " Expected: " << 2; // 20%
    EXPECT_NEAR(5, sll2, 1.0) <<  "Stddev: " << sll2 << " Expected: " << 5; // 20%

    EXPECT_NEAR(2, mull2, 0.4) << "Mean: " << mull2 << " Expected: " << 2; // 20%
    EXPECT_NEAR(5, sull2, 1.0) <<  "Stddev: " << sull2 << " Expected: " << 5; // 20%
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
