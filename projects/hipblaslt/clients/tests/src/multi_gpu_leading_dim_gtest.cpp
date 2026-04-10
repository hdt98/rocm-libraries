/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Leading Dimension Test Suite
 * Testing leading dimension variations, padding, and alignment
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>

#include <gtest/gtest-spi.h>

namespace
{
    int getNumGPUs()
    {
        int numDevices = 0;
        hipGetDeviceCount(&numDevices);
        return numDevices;
    }

    // Test 1: ld = dim (Contiguous, no padding)
    TEST(MultiGPULeadingDim, Contiguous_NoPadding)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing ld = dim (contiguous, no padding) across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // ld = dim (contiguous)
            int64_t ldA = M, ldB = K, ldD = M;

            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, ldA);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, ldB);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, ldD);

            hipblaslt_cout << "GPU " << dev << " ldA=" << ldA << " (M=" << M << "), "
                           << "ldB=" << ldB << " (K=" << K << "), "
                           << "ldD=" << ldD << " (M=" << M << ")" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // Test 2: ld > dim (Padding +16)
    TEST(MultiGPULeadingDim, Padding_Plus16)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing ld > dim (padding +16) across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // ld = dim + 16 (padding)
            int64_t ldA = M + 16, ldB = K + 16, ldD = M + 16;

            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, ldA);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, ldB);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, ldD);

            hipblaslt_cout << "GPU " << dev << " ldA=" << ldA << " (M=" << M << "+16), "
                           << "ldB=" << ldB << " (K=" << K << "+16), "
                           << "ldD=" << ldD << " (M=" << M << "+16)" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // Test 3: ld > dim (Padding +32)
    TEST(MultiGPULeadingDim, Padding_Plus32)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing ld > dim (padding +32) across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // ld = dim + 32 (padding)
            int64_t ldA = M + 32, ldB = K + 32, ldD = M + 32;

            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, ldA);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, ldB);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, ldD);

            hipblaslt_cout << "GPU " << dev << " ldA=" << ldA << " (M=" << M << "+32), "
                           << "ldB=" << ldB << " (K=" << K << "+32), "
                           << "ldD=" << ldD << " (M=" << M << "+32)" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // Test 4: ld > dim (Padding +64)
    TEST(MultiGPULeadingDim, Padding_Plus64)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing ld > dim (padding +64) across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // ld = dim + 64 (padding)
            int64_t ldA = M + 64, ldB = K + 64, ldD = M + 64;

            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, ldA);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, ldB);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, ldD);

            hipblaslt_cout << "GPU " << dev << " ldA=" << ldA << " (M=" << M << "+64), "
                           << "ldB=" << ldB << " (K=" << K << "+64), "
                           << "ldD=" << ldD << " (M=" << M << "+64)" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // Test 5: Non-Aligned Padding
    TEST(MultiGPULeadingDim, NonAlignedPadding)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing non-aligned padding across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // ld with non-aligned padding
            int64_t ldA = M + 7, ldB = K + 13, ldD = M + 23;

            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, ldA);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, ldB);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, ldD);

            hipblaslt_cout << "GPU " << dev << " ldA=" << ldA << " (M=" << M << "+7), "
                           << "ldB=" << ldB << " (K=" << K << "+13), "
                           << "ldD=" << ldD << " (M=" << M << "+23)" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // Test 6: Different ld Values for A, B, C, D
    TEST(MultiGPULeadingDim, DifferentLDValues)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing different ld values for A, B, D across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Different padding for each matrix
            int64_t ldA = M + 16;
            int64_t ldB = K + 32;
            int64_t ldD = M + 64;

            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, ldA);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, ldB);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, ldD);

            hipblaslt_cout << "GPU " << dev << " different padding: "
                           << "ldA=" << ldA << ", ldB=" << ldB << ", ldD=" << ldD << std::endl;

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // Test 7: ld Variations Across GPUs
    TEST(MultiGPULeadingDim, VariationsAcrossGPUs)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing different ld variations per GPU" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        std::vector<int64_t> padding_values = {0, 16, 32, 64};

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            int64_t padding = padding_values[dev % padding_values.size()];
            int64_t ldA = M + padding;

            hipblasLtMatrixLayout_t matA;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, ldA);

            hipblaslt_cout << "GPU " << dev << " padding=" << padding
                           << ", ldA=" << ldA << " (M=" << M << ")" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtDestroy(handle);
        }
    }

    // Test 8: Maximum ld Values
    TEST(MultiGPULeadingDim, MaximumLDValues)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing maximum ld values across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Large padding (but reasonable)
            int64_t ldA = M + 1024;
            int64_t ldB = K + 1024;
            int64_t ldD = M + 1024;

            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, ldA);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, ldB);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, ldD);

            hipblaslt_cout << "GPU " << dev << " large padding: "
                           << "ldA=" << ldA << ", ldB=" << ldB << ", ldD=" << ldD << std::endl;

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // Test 9: ld with Different Matrix Sizes
    TEST(MultiGPULeadingDim, LDWithDifferentSizes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing ld with different matrix sizes across GPUs" << std::endl;

        struct SizeConfig { int64_t M, N, K, padding; };
        std::vector<SizeConfig> configs = {
            {64, 64, 64, 16},
            {256, 256, 256, 32},
            {1024, 1024, 1024, 64}
        };

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            for(const auto& cfg : configs)
            {
                int64_t ldA = cfg.M + cfg.padding;

                hipblasLtMatrixLayout_t matA;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, cfg.M, cfg.K, ldA);

                hipblaslt_cout << "GPU " << dev << " M=" << cfg.M << ", K=" << cfg.K
                               << ", ldA=" << ldA << " (padding=" << cfg.padding << ")" << std::endl;

                hipblasLtMatrixLayoutDestroy(matA);
            }

            hipblasLtDestroy(handle);
        }
    }

    // Test 10: ld Aligned to 32-byte Boundaries
    TEST(MultiGPULeadingDim, Aligned32ByteBoundaries)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing ld aligned to 32-byte boundaries across GPUs" << std::endl;

        const int64_t M = 250, N = 250, K = 250; // Not naturally aligned

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Align to 32-byte boundary (8 floats for FP32)
            int64_t ldA = ((M + 7) / 8) * 8; // Round up to multiple of 8

            hipblasLtMatrixLayout_t matA;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, ldA);

            hipblaslt_cout << "GPU " << dev << " M=" << M << ", ldA=" << ldA
                           << " (aligned to 32-byte boundary)" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtDestroy(handle);
        }
    }

} // namespace
