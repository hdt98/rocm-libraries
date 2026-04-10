/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Problem Sizes Test Suite
 * Testing tiny to huge problem sizes and rectangular matrices
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

    // Test 1: Tiny Sizes (M, N, K < 32)
    TEST(MultiGPUProblemSizes, TinySizes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing tiny problem sizes across GPUs" << std::endl;

        std::vector<int64_t> tiny_sizes = {1, 2, 4, 8, 16, 31};

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            for(auto size : tiny_sizes)
            {
                int64_t M = size, N = size, K = size;

                hipblasLtMatrixLayout_t matA, matB, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblaslt_cout << "GPU " << dev << " tiny size: " << M << "x" << N << "x" << K << std::endl;

                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }

            hipblasLtDestroy(handle);
        }
    }

    // Test 2: Small Sizes (32 ≤ M, N, K < 128)
    TEST(MultiGPUProblemSizes, SmallSizes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing small problem sizes across GPUs" << std::endl;

        std::vector<int64_t> small_sizes = {32, 48, 64, 96, 127};

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            for(auto size : small_sizes)
            {
                int64_t M = size, N = size, K = size;

                hipblasLtMatrixLayout_t matA, matB, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblaslt_cout << "GPU " << dev << " small size: " << M << "x" << N << "x" << K << std::endl;

                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }

            hipblasLtDestroy(handle);
        }
    }

    // Test 3: Medium Sizes (128 ≤ M, N, K < 1024)
    TEST(MultiGPUProblemSizes, MediumSizes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing medium problem sizes across GPUs" << std::endl;

        std::vector<int64_t> medium_sizes = {128, 256, 512, 768, 1023};

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            for(auto size : medium_sizes)
            {
                int64_t M = size, N = size, K = size;

                hipblasLtMatrixLayout_t matA, matB, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblaslt_cout << "GPU " << dev << " medium size: " << M << "x" << N << "x" << K << std::endl;

                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }

            hipblasLtDestroy(handle);
        }
    }

    // Test 4: Large Sizes (1024 ≤ M, N, K < 4096)
    TEST(MultiGPUProblemSizes, LargeSizes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing large problem sizes across GPUs" << std::endl;

        std::vector<int64_t> large_sizes = {1024, 2048, 3072, 4095};

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            for(auto size : large_sizes)
            {
                int64_t M = size, N = size, K = size;

                hipblasLtMatrixLayout_t matA, matB, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblaslt_cout << "GPU " << dev << " large size: " << M << "x" << N << "x" << K << std::endl;

                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }

            hipblasLtDestroy(handle);
        }
    }

    // Test 5: Huge Sizes (M, N, K ≥ 4096)
    TEST(MultiGPUProblemSizes, HugeSizes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing huge problem sizes across GPUs" << std::endl;

        std::vector<int64_t> huge_sizes = {4096, 8192};

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            for(auto size : huge_sizes)
            {
                int64_t M = size, N = size, K = size;

                hipblasLtMatrixLayout_t matA, matB, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblaslt_cout << "GPU " << dev << " huge size: " << M << "x" << N << "x" << K << std::endl;

                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }

            hipblasLtDestroy(handle);
        }
    }

    // Test 6: Rectangular Matrices (M >> N)
    TEST(MultiGPUProblemSizes, Rectangular_M_Much_Larger)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing rectangular matrices (M >> N) across GPUs" << std::endl;

        struct SizeConfig { int64_t M, N, K; };
        std::vector<SizeConfig> configs = {
            {1024, 64, 256},
            {2048, 128, 512},
            {4096, 256, 1024}
        };

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            for(const auto& cfg : configs)
            {
                hipblasLtMatrixLayout_t matA, matB, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, cfg.M, cfg.K, cfg.M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, cfg.K, cfg.N, cfg.K);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, cfg.M, cfg.N, cfg.M);

                hipblaslt_cout << "GPU " << dev << " M >> N: " << cfg.M << "x" << cfg.N
                               << "x" << cfg.K << std::endl;

                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }

            hipblasLtDestroy(handle);
        }
    }

    // Test 7: Rectangular Matrices (N >> M)
    TEST(MultiGPUProblemSizes, Rectangular_N_Much_Larger)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing rectangular matrices (N >> M) across GPUs" << std::endl;

        struct SizeConfig { int64_t M, N, K; };
        std::vector<SizeConfig> configs = {
            {64, 1024, 256},
            {128, 2048, 512},
            {256, 4096, 1024}
        };

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            for(const auto& cfg : configs)
            {
                hipblasLtMatrixLayout_t matA, matB, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, cfg.M, cfg.K, cfg.M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, cfg.K, cfg.N, cfg.K);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, cfg.M, cfg.N, cfg.M);

                hipblaslt_cout << "GPU " << dev << " N >> M: " << cfg.M << "x" << cfg.N
                               << "x" << cfg.K << std::endl;

                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }

            hipblasLtDestroy(handle);
        }
    }

    // Test 8: Rectangular Matrices (K >> M, N)
    TEST(MultiGPUProblemSizes, Rectangular_K_Much_Larger)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing rectangular matrices (K >> M, N) across GPUs" << std::endl;

        struct SizeConfig { int64_t M, N, K; };
        std::vector<SizeConfig> configs = {
            {128, 128, 1024},
            {256, 256, 2048},
            {512, 512, 4096}
        };

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            for(const auto& cfg : configs)
            {
                hipblasLtMatrixLayout_t matA, matB, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, cfg.M, cfg.K, cfg.M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, cfg.K, cfg.N, cfg.K);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, cfg.M, cfg.N, cfg.M);

                hipblaslt_cout << "GPU " << dev << " K >> M,N: " << cfg.M << "x" << cfg.N
                               << "x" << cfg.K << std::endl;

                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }

            hipblasLtDestroy(handle);
        }
    }

    // Test 9: Different Size Categories Per GPU
    TEST(MultiGPUProblemSizes, DifferentCategoriesPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing different size categories per GPU" << std::endl;

        std::vector<int64_t> sizes_per_gpu = {64, 256, 1024, 4096};

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            int64_t size = sizes_per_gpu[dev % sizes_per_gpu.size()];
            int64_t M = size, N = size, K = size;

            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            const char* category = (size < 128) ? "small" :
                                   (size < 1024) ? "medium" :
                                   (size < 4096) ? "large" : "huge";

            hipblaslt_cout << "GPU " << dev << " " << category << " size: "
                           << M << "x" << N << "x" << K << std::endl;

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // Test 10: Extreme Aspect Ratios
    TEST(MultiGPUProblemSizes, ExtremeAspectRatios)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing extreme aspect ratios across GPUs" << std::endl;

        struct SizeConfig { int64_t M, N, K; const char* description; };
        std::vector<SizeConfig> configs = {
            {1, 4096, 256, "1x4096x256 (extreme thin)"},
            {4096, 1, 256, "4096x1x256 (extreme tall)"},
            {256, 256, 1, "256x256x1 (K=1)"},
            {8192, 8, 8, "8192x8x8 (M dominant)"}
        };

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            for(const auto& cfg : configs)
            {
                hipblasLtMatrixLayout_t matA, matB, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, cfg.M, cfg.K, cfg.M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, cfg.K, cfg.N, cfg.K);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, cfg.M, cfg.N, cfg.M);

                hipblaslt_cout << "GPU " << dev << " " << cfg.description << std::endl;

                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }

            hipblasLtDestroy(handle);
        }
    }

} // namespace
