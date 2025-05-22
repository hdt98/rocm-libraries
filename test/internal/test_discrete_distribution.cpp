// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <vector>
#include <rocrand/rocrand_discrete.h>

#define HIP_CHECK(cmd)                                                                          \
    do                                                                                          \
    {                                                                                           \
        auto error = (cmd);                                                                     \
        if(error != hipSuccess)                                                                 \
        {                                                                                       \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error) << ") at line "  \
                      << __LINE__ << " in file " << __FILE__ << "\n";                           \
            exit(-1);                                                                           \
        }                                                                                       \
    } while(0)                                                                                  \

#define ROCRAND_CHECK(cmd)                                                                      \
    do                                                                                          \
    {                                                                                           \
        auto status = cmd;                                                                      \
        if(status != 0){                                                                        \
            std::cerr << "Encountered ROCRAND error: " << status << "at line"                   \
            << __LINE__ << " in file " << __FILE__ << "\n";                                     \
            exit(-1);                                                                           \
        }                                                                                       \
    } while(0)                                                                                  \



using DiscreteDataType = ::testing::Types<double, unsigned int, unsigned long, unsigned long long int>;

template <typename DT>
class DiscreteDistributionTests : public ::testing::Test{
    public:
        using T = DT;
};

TYPED_TEST_SUITE(DiscreteDistributionTests, DiscreteDataType);

template<typename T, size_t items_per_thread, size_t block_size, class DiscreteFunc>
__global__ void discrete_kernel(T * device_input, unsigned int * device_output, rocrand_discrete_distribution_st &dis, const DiscreteFunc & f){
    constexpr size_t items_per_block = items_per_thread * block_size;
    const size_t offset = (items_per_block * blockIdx.x) + (items_per_thread * threadIdx.x);

    for(size_t i = 0; i < items_per_thread; i++){
        device_output[offset + i] = f(device_input[offset + i], dis);
    }
}


template <typename T, class DiscreteFunc>
void run_internal_discrete_tests(const DiscreteFunc & f){
    constexpr size_t items_per_thread = 256;
    constexpr size_t block_size = 32;
    constexpr size_t items_per_block = items_per_thread * block_size;
    constexpr size_t grid_size = 1234;
    constexpr size_t size = grid_size * items_per_block;

    std::vector<std::vector<double>> all_distributions = {
        {10, 10, 10, 10},
        {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1},
        {1234, 1677, 1519, 1032, 561, 254, 98, 33, 10, 2},
        {1, 2, 8, 4, 3, 2, 1}
    };

    std::random_device rd;
    std::mt19937 gen(rd());
    

    T * host_input = new T[size];
    unsigned int * host_output = new unsigned int[size];

    if constexpr (std::is_same_v<T, double>){
        std::uniform_real_distribution<double> dis(0, 1);
        for(size_t i = 0; i < size; i++) host_input[i] = dis(gen);
    }
    else if constexpr(std::is_same_v<T, unsigned long> || std::is_same_v<T, unsigned int>){
        std::uniform_int_distribution<T> dis(0, std::numeric_limits<unsigned int>::max());
        for(size_t i = 0; i < size; i++) host_input[i] = dis(gen);
    }
    else{
        std::uniform_int_distribution<T> dis(0, std::numeric_limits<T>::max());
        for(size_t i = 0; i < size; i++) host_input[i] = dis(gen);
    }

    T * device_input;
    unsigned int * device_output;

    HIP_CHECK(hipMalloc(&device_input, sizeof(T) * size));
    HIP_CHECK(hipMalloc(&device_output, sizeof(unsigned int) * size));

    HIP_CHECK(hipMemcpy(device_input, host_input, sizeof(T) * size, hipMemcpyHostToDevice));

    for(std::vector<double> distribution : all_distributions){
        double sum = std::accumulate(distribution.begin(), distribution.end(), 0);
        std::string error_dis;

        for(const auto & x : distribution) error_dis += std::to_string(x) + " ";
        
        SCOPED_TRACE(testing::Message() << "With current distribution: \n" << error_dis << "\n");
        
        std::vector<double> expected_prob(distribution.size());
        for(size_t i = 0; i < distribution.size(); i++)
            expected_prob[i] = distribution[i] / sum;
        
        rocrand_discrete_distribution discrete_distribution;
        ROCRAND_CHECK(rocrand_create_discrete_distribution(expected_prob.data(), expected_prob.size(), 0, &discrete_distribution));

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(discrete_kernel<T, items_per_thread, block_size>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_output, *discrete_distribution, f
        );


        HIP_CHECK(hipMemcpy(host_output, device_output, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));

        std::vector<double> histogram(distribution.size());

        for(size_t i = 0; i < size; i++)
            histogram[host_output[i]]++;

        std::vector<double> actual_prob(distribution.size());
        for(size_t i = 0; i < actual_prob.size(); i++)
            actual_prob[i] = histogram[i] / static_cast<double>(size);
        

        for(size_t i = 0; i < expected_prob.size(); i++){
            double eps = expected_prob[i] > 0.05 ? expected_prob[i] * 0.01 : 0.01;
            ASSERT_NEAR(expected_prob[i], actual_prob[i], eps); 
        }

        ROCRAND_CHECK(rocrand_destroy_discrete_distribution(discrete_distribution));
    }

    delete [] host_input;
    delete [] host_output;

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}

TYPED_TEST(DiscreteDistributionTests, InternalDiscreteAliasTest){
    using T = TestFixture::T;
    run_internal_discrete_tests<T>(
        [=] __device__(T val, rocrand_discrete_distribution_st & dis)
            {return rocrand_device::detail::discrete_alias(val, dis);}
    );
}

TYPED_TEST(DiscreteDistributionTests, InternalDiscreteCDFTest){
    using T = TestFixture::T;
    run_internal_discrete_tests<T>(
        [=] __device__(T val, rocrand_discrete_distribution_st & dis)
            {return rocrand_device::detail::discrete_cdf(val, dis);}
    );
}
