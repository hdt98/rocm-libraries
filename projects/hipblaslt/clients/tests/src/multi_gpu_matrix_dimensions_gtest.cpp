/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Matrix Dimensions Test Suite
 * Comprehensive variations of M, N, K, and leading dimensions
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

    // Helper to run GEMM with specific dimensions
    bool runGEMM(int deviceId, int64_t M, int64_t N, int64_t K,
                 int64_t lda, int64_t ldb, int64_t ldc, int64_t ldd)
    {
        auto hipErr = hipSetDevice(deviceId);
        if(hipErr != hipSuccess) return false;

        hipblasLtHandle_t handle;
        auto status = hipblasLtCreate(&handle);
        if(status != HIPBLAS_STATUS_SUCCESS) return false;

        hipblasLtMatrixLayout_t matA, matB, matC, matD;
        status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, lda);
        if(status != HIPBLAS_STATUS_SUCCESS) { hipblasLtDestroy(handle); return false; }
        status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, ldb);
        if(status != HIPBLAS_STATUS_SUCCESS) { hipblasLtMatrixLayoutDestroy(matA); hipblasLtDestroy(handle); return false; }
        status = hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, ldc);
        if(status != HIPBLAS_STATUS_SUCCESS) { hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB); hipblasLtDestroy(handle); return false; }
        status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, ldd);
        if(status != HIPBLAS_STATUS_SUCCESS) {
            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtDestroy(handle); return false;
        }

        hipblasLtMatmulDesc_t matmul;
        status = hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        if(status != HIPBLAS_STATUS_SUCCESS) {
            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle); return false;
        }

        hipblasOperation_t opA = HIPBLAS_OP_N;
        hipblasOperation_t opB = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
        hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

        // Allocate
        float *d_a, *d_b, *d_c, *d_d;
        hipMalloc(&d_a, lda * K * sizeof(float));
        hipMalloc(&d_b, ldb * N * sizeof(float));
        hipMalloc(&d_c, ldc * N * sizeof(float));
        hipMalloc(&d_d, ldd * N * sizeof(float));

        std::vector<float> h_a(lda * K, 1.0f);
        std::vector<float> h_b(ldb * N, 2.0f);
        std::vector<float> h_c(ldc * N, 0.0f);

        hipMemcpy(d_a, h_a.data(), lda * K * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_b, h_b.data(), ldb * N * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_c, h_c.data(), ldc * N * sizeof(float), hipMemcpyHostToDevice);

        hipblasLtMatmulPreference_t pref;
        hipblasLtMatmulPreferenceCreate(&pref);
        size_t max_workspace = 32 * 1024 * 1024;
        hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace, sizeof(max_workspace));

        hipblasLtMatmulHeuristicResult_t heuristicResult[1];
        int returnedAlgoCount = 0;
        status = hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matC, matD,
                                                 pref, 1, heuristicResult, &returnedAlgoCount);

        bool success = false;
        if(status == HIPBLAS_STATUS_SUCCESS && returnedAlgoCount > 0)
        {
            void* d_workspace = nullptr;
            if(heuristicResult[0].workspaceSize > 0)
                hipMalloc(&d_workspace, heuristicResult[0].workspaceSize);

            float alpha = 1.0f, beta = 0.0f;
            status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                     &beta, d_c, matC, d_d, matD,
                                     &heuristicResult[0].algo, d_workspace,
                                     heuristicResult[0].workspaceSize, 0);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipDeviceSynchronize();
                std::vector<float> h_d(M * N);
                hipMemcpy(h_d.data(), d_d, M * N * sizeof(float), hipMemcpyDeviceToHost);
                float expected = static_cast<float>(K) * 1.0f * 2.0f;
                success = (std::abs(h_d[0] - expected) < 0.01f);
            }

            if(d_workspace) hipFree(d_workspace);
        }

        hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_d);
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
        hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
        hipblasLtMatmulDescDestroy(matmul);
        hipblasLtDestroy(handle);

        return success;
    }

    // ----------------------------------------------------------------------------
    // Test 1: Small Dimensions (1-64)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDimensions, SmallSizes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing small dimensions across GPUs" << std::endl;

        std::vector<int64_t> small_sizes = {1, 2, 4, 8, 16, 32, 64};

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            for(auto size : small_sizes)
            {
                hipblaslt_cout << "GPU " << dev << " testing " << size << "x" << size << "x" << size << std::endl;
                bool result = runGEMM(dev, size, size, size, size, size, size, size);
                EXPECT_TRUE(result) << "Failed on GPU " << dev << " with size " << size;
            }
        }

        hipblaslt_cout << "Small sizes test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 2: Medium Dimensions (128-1024)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDimensions, MediumSizes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing medium dimensions across GPUs" << std::endl;

        std::vector<int64_t> medium_sizes = {128, 256, 512, 1024};

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            for(auto size : medium_sizes)
            {
                hipblaslt_cout << "GPU " << dev << " testing " << size << "x" << size << "x" << size << std::endl;
                bool result = runGEMM(dev, size, size, size, size, size, size, size);
                EXPECT_TRUE(result) << "Failed on GPU " << dev << " with size " << size;
            }
        }

        hipblaslt_cout << "Medium sizes test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 3: Large Dimensions (2048-8192)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDimensions, LargeSizes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing large dimensions across GPUs" << std::endl;

        std::vector<int64_t> large_sizes = {2048, 4096};

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            for(auto size : large_sizes)
            {
                hipblaslt_cout << "GPU " << dev << " testing " << size << "x" << size << "x" << size << std::endl;
                bool result = runGEMM(dev, size, size, size, size, size, size, size);
                EXPECT_TRUE(result) << "Failed on GPU " << dev << " with size " << size;
            }
        }

        hipblaslt_cout << "Large sizes test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 4: Prime Number Dimensions
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDimensions, PrimeNumberSizes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing prime number dimensions across GPUs" << std::endl;

        std::vector<int64_t> prime_sizes = {127, 251, 509};

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            for(auto size : prime_sizes)
            {
                hipblaslt_cout << "GPU " << dev << " testing prime " << size << "x" << size << "x" << size << std::endl;
                bool result = runGEMM(dev, size, size, size, size, size, size, size);
                EXPECT_TRUE(result) << "Failed on GPU " << dev << " with prime size " << size;
            }
        }

        hipblaslt_cout << "Prime number sizes test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 5: Non-Square Matrices (M != N)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDimensions, NonSquareMatrices)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing non-square matrices across GPUs" << std::endl;

        struct DimConfig { int64_t M, N, K; const char* desc; };
        std::vector<DimConfig> configs = {
            {64, 128, 256, "64x128x256"},
            {256, 64, 128, "256x64x128"},
            {128, 256, 64, "128x256x64"},
            {1024, 512, 256, "1024x512x256"},
            {512, 1024, 512, "512x1024x512"}
        };

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            for(const auto& cfg : configs)
            {
                hipblaslt_cout << "GPU " << dev << " testing " << cfg.desc << std::endl;
                bool result = runGEMM(dev, cfg.M, cfg.N, cfg.K, cfg.M, cfg.K, cfg.M, cfg.M);
                EXPECT_TRUE(result) << "Failed on GPU " << dev << " with " << cfg.desc;
            }
        }

        hipblaslt_cout << "Non-square matrices test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 6: Edge Case Dimensions (M=1, N=1, K=1)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDimensions, EdgeCaseDimensions)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing edge case dimensions across GPUs" << std::endl;

        struct DimConfig { int64_t M, N, K; const char* desc; };
        std::vector<DimConfig> edge_cases = {
            {1, 1, 1, "1x1x1"},
            {1, 64, 64, "1x64x64"},
            {64, 1, 64, "64x1x64"},
            {64, 64, 1, "64x64x1"},
            {1, 1024, 1024, "1x1024x1024"},
            {1024, 1, 1024, "1024x1x1024"}
        };

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            for(const auto& cfg : edge_cases)
            {
                hipblaslt_cout << "GPU " << dev << " testing edge case " << cfg.desc << std::endl;
                bool result = runGEMM(dev, cfg.M, cfg.N, cfg.K, cfg.M, cfg.K, cfg.M, cfg.M);
                EXPECT_TRUE(result) << "Failed on GPU " << dev << " with " << cfg.desc;
            }
        }

        hipblaslt_cout << "Edge case dimensions test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 7: Non-Aligned Dimensions (not multiples of 16/32)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDimensions, NonAlignedDimensions)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing non-aligned dimensions across GPUs" << std::endl;

        std::vector<int64_t> non_aligned = {17, 33, 65, 129, 257};

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            for(auto size : non_aligned)
            {
                hipblaslt_cout << "GPU " << dev << " testing non-aligned " << size << "x" << size << "x" << size << std::endl;
                bool result = runGEMM(dev, size, size, size, size, size, size, size);
                EXPECT_TRUE(result) << "Failed on GPU " << dev << " with non-aligned size " << size;
            }
        }

        hipblaslt_cout << "Non-aligned dimensions test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 8: Leading Dimension with Padding (ld > dim)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDimensions, LeadingDimensionPadding)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing leading dimension padding across GPUs" << std::endl;

        struct LDConfig { int64_t M, N, K, lda, ldb, ldc, ldd; const char* desc; };
        std::vector<LDConfig> configs = {
            {64, 64, 64, 80, 80, 80, 80, "64x64x64 with ld=80"},
            {128, 128, 128, 160, 160, 160, 160, "128x128x128 with ld=160"},
            {256, 256, 256, 288, 288, 288, 288, "256x256x256 with ld=288"}
        };

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            for(const auto& cfg : configs)
            {
                hipblaslt_cout << "GPU " << dev << " testing " << cfg.desc << std::endl;
                bool result = runGEMM(dev, cfg.M, cfg.N, cfg.K, cfg.lda, cfg.ldb, cfg.ldc, cfg.ldd);
                EXPECT_TRUE(result) << "Failed on GPU " << dev << " with " << cfg.desc;
            }
        }

        hipblaslt_cout << "Leading dimension padding test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 9: Extreme Rectangular Matrices
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDimensions, ExtremeRectangular)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing extreme rectangular matrices across GPUs" << std::endl;

        struct DimConfig { int64_t M, N, K; const char* desc; };
        std::vector<DimConfig> extreme = {
            {1, 4096, 64, "Very wide: 1x4096x64"},
            {4096, 1, 64, "Very tall: 4096x1x64"},
            {64, 64, 4096, "Deep K: 64x64x4096"},
            {8, 2048, 128, "Wide: 8x2048x128"},
            {2048, 8, 128, "Tall: 2048x8x128"}
        };

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            for(const auto& cfg : extreme)
            {
                hipblaslt_cout << "GPU " << dev << " testing " << cfg.desc << std::endl;
                bool result = runGEMM(dev, cfg.M, cfg.N, cfg.K, cfg.M, cfg.K, cfg.M, cfg.M);
                EXPECT_TRUE(result) << "Failed on GPU " << dev << " with " << cfg.desc;
            }
        }

        hipblaslt_cout << "Extreme rectangular test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 10: Powers of 2 Dimensions
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDimensions, PowersOfTwo)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing powers of 2 dimensions across GPUs" << std::endl;

        std::vector<int64_t> powers = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            for(auto size : powers)
            {
                hipblaslt_cout << "GPU " << dev << " testing power of 2: " << size << "x" << size << "x" << size << std::endl;
                bool result = runGEMM(dev, size, size, size, size, size, size, size);
                EXPECT_TRUE(result) << "Failed on GPU " << dev << " with power-of-2 size " << size;
            }
        }

        hipblaslt_cout << "Powers of 2 test passed" << std::endl;
    }

} // namespace
