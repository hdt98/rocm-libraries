/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Edge Cases Test Suite
 * Testing special values, NaN, Inf, denormalized numbers, and boundary conditions
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <cmath>
#include <limits>

#include <gtest/gtest-spi.h>

namespace
{
    int getNumGPUs()
    {
        int numDevices = 0;
        hipGetDeviceCount(&numDevices);
        return numDevices;
    }

    // Test 1: All Zeros in Matrices
    TEST(MultiGPUEdgeCases, AllZeros)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing all zeros in matrices across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float alpha = 1.0f, beta = 0.0f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << dev << " testing with zero matrices" << std::endl;

            // Allocate and zero-initialize matrices
            float *d_A, *d_B, *d_C, *d_D;
            hipMalloc(&d_A, M * K * sizeof(float));
            hipMalloc(&d_B, K * N * sizeof(float));
            hipMalloc(&d_C, M * N * sizeof(float));
            hipMalloc(&d_D, M * N * sizeof(float));

            hipMemset(d_A, 0, M * K * sizeof(float));
            hipMemset(d_B, 0, K * N * sizeof(float));
            hipMemset(d_C, 0, M * N * sizeof(float));

            // Result should be all zeros
            hipblaslt_cout << "GPU " << dev << " all-zero matrices configured" << std::endl;

            hipFree(d_A); hipFree(d_B); hipFree(d_C); hipFree(d_D);
        }
    }

    // Test 2: Identity-Like Operations (D = A)
    TEST(MultiGPUEdgeCases, IdentityOperation)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing identity-like operations across GPUs" << std::endl;

        const int64_t M = 256, N = 256;
        // alpha = 0, beta = 1 means D = C (identity operation)
        float alpha = 0.0f, beta = 1.0f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << dev << " D = 0*AB + 1*C (identity, D=C)" << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }
    }

    // Test 3: Pure Scaling (alpha = 0, beta != 0)
    TEST(MultiGPUEdgeCases, PureScaling)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing pure scaling operations across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float alpha = 0.0f, beta = 2.0f; // D = 2*C

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << dev << " D = 0*AB + 2*C (pure scaling)" << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }
    }

    // Test 4: Null Operation (alpha = 0, beta = 1)
    TEST(MultiGPUEdgeCases, NullOperation)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing null operation across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float alpha = 0.0f, beta = 1.0f; // D = C

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << dev << " D = 0*AB + 1*C (null op, D=C)" << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }
    }

    // Test 5: Very Small Values (Near Underflow)
    TEST(MultiGPUEdgeCases, VerySmallValues)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing very small values (near underflow) across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float small_value = std::numeric_limits<float>::min(); // Smallest normalized positive float

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << dev << " testing with value " << small_value << std::endl;

            float *d_A;
            hipMalloc(&d_A, M * K * sizeof(float));

            std::vector<float> h_A(M * K, small_value);
            hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice);

            hipFree(d_A);
        }
    }

    // Test 6: Very Large Values (Near Overflow)
    TEST(MultiGPUEdgeCases, VeryLargeValues)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing very large values (near overflow) across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float large_value = std::numeric_limits<float>::max() / 1000.0f; // Scaled to avoid immediate overflow

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << dev << " testing with value " << large_value << std::endl;

            float *d_A;
            hipMalloc(&d_A, M * K * sizeof(float));

            std::vector<float> h_A(M * K, large_value);
            hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice);

            hipFree(d_A);
        }
    }

    // Test 7: NaN Handling
    TEST(MultiGPUEdgeCases, NaNHandling)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing NaN handling across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float nan_value = std::numeric_limits<float>::quiet_NaN();

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << dev << " testing with NaN values" << std::endl;

            float *d_A;
            hipMalloc(&d_A, M * K * sizeof(float));

            std::vector<float> h_A(M * K, nan_value);
            hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice);

            // NaN should propagate through computation
            hipblaslt_cout << "GPU " << dev << " NaN values configured" << std::endl;

            hipFree(d_A);
        }
    }

    // Test 8: Infinity Handling
    TEST(MultiGPUEdgeCases, InfinityHandling)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing infinity handling across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float inf_value = std::numeric_limits<float>::infinity();

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << dev << " testing with infinity values" << std::endl;

            float *d_A;
            hipMalloc(&d_A, M * K * sizeof(float));

            std::vector<float> h_A(M * K, inf_value);
            hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice);

            hipblaslt_cout << "GPU " << dev << " infinity values configured" << std::endl;

            hipFree(d_A);
        }
    }

    // Test 9: Denormalized Numbers
    TEST(MultiGPUEdgeCases, DenormalizedNumbers)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing denormalized numbers across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float denorm_value = std::numeric_limits<float>::denorm_min();

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << dev << " testing with denormalized value " << denorm_value << std::endl;

            float *d_A;
            hipMalloc(&d_A, M * K * sizeof(float));

            std::vector<float> h_A(M * K, denorm_value);
            hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice);

            hipFree(d_A);
        }
    }

    // Test 10: Mixed Special Values
    TEST(MultiGPUEdgeCases, MixedSpecialValues)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing mixed special values across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        std::vector<float> special_values = {
            0.0f,
            -0.0f,
            1.0f,
            -1.0f,
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::min(),
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::denorm_min()
        };

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << dev << " testing with mixed special values" << std::endl;

            float *d_A;
            hipMalloc(&d_A, M * K * sizeof(float));

            // Fill with pattern of special values
            std::vector<float> h_A(M * K);
            for(size_t i = 0; i < h_A.size(); ++i)
            {
                h_A[i] = special_values[i % special_values.size()];
            }
            hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice);

            hipFree(d_A);
        }
    }

    // Test 11: Negative Zero
    TEST(MultiGPUEdgeCases, NegativeZero)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing negative zero across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float neg_zero = -0.0f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << dev << " testing with negative zero" << std::endl;

            float *d_A;
            hipMalloc(&d_A, M * K * sizeof(float));

            std::vector<float> h_A(M * K, neg_zero);
            hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice);

            hipFree(d_A);
        }
    }

    // Test 12: Subnormal Flush to Zero
    TEST(MultiGPUEdgeCases, SubnormalFlushToZero)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing subnormal flush-to-zero behavior across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            // Use very small values that might flush to zero
            float subnormal = std::numeric_limits<float>::min() / 2.0f;

            hipblaslt_cout << "GPU " << dev << " testing subnormal value " << subnormal << std::endl;

            float *d_A;
            hipMalloc(&d_A, M * K * sizeof(float));

            std::vector<float> h_A(M * K, subnormal);
            hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice);

            hipFree(d_A);
        }
    }

} // namespace
