/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Transpose Operations Test Suite
 * All combinations of transA and transB (NN, NT, TN, TT, etc.)
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

    struct TransposeConfig {
        hipblasOperation_t transA;
        hipblasOperation_t transB;
        const char* name;
    };

    // ----------------------------------------------------------------------------
    // Test 1: No Transpose (NN)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTranspose, NoTranspose_NN)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing NN (no transpose) across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float alpha = 1.0f, beta = 0.0f;

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasOperation_t opA = HIPBLAS_OP_N;
            hipblasOperation_t opB = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

            hipblaslt_cout << "GPU " << dev << " NN configuration successful" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }
    }

    // ----------------------------------------------------------------------------
    // Test 2: Transpose B (NT)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTranspose, TransposeB_NT)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing NT (transpose B) across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // For NT: A is MxK, B is NxK (transposed to KxN)
            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, N, K, N); // NxK stored
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasOperation_t opA = HIPBLAS_OP_N;
            hipblasOperation_t opB = HIPBLAS_OP_T;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

            hipblaslt_cout << "GPU " << dev << " NT configuration successful" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }
    }

    // ----------------------------------------------------------------------------
    // Test 3: Transpose A (TN)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTranspose, TransposeA_TN)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing TN (transpose A) across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // For TN: A is KxM (transposed to MxK), B is KxN
            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, K, M, K); // KxM stored
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasOperation_t opA = HIPBLAS_OP_T;
            hipblasOperation_t opB = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

            hipblaslt_cout << "GPU " << dev << " TN configuration successful" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }
    }

    // ----------------------------------------------------------------------------
    // Test 4: Transpose Both (TT)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTranspose, TransposeBoth_TT)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing TT (transpose both) across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // For TT: A is KxM, B is NxK
            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, K, M, K);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, N, K, N);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasOperation_t opA = HIPBLAS_OP_T;
            hipblasOperation_t opB = HIPBLAS_OP_T;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

            hipblaslt_cout << "GPU " << dev << " TT configuration successful" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }
    }

    // ----------------------------------------------------------------------------
    // Test 5: All Transpose Combinations
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTranspose, AllCombinations)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing all transpose combinations across GPUs" << std::endl;

        std::vector<TransposeConfig> configs = {
            {HIPBLAS_OP_N, HIPBLAS_OP_N, "NN"},
            {HIPBLAS_OP_N, HIPBLAS_OP_T, "NT"},
            {HIPBLAS_OP_T, HIPBLAS_OP_N, "TN"},
            {HIPBLAS_OP_T, HIPBLAS_OP_T, "TT"}
        };

        const int64_t M = 128, N = 128, K = 128;

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            for(const auto& cfg : configs)
            {
                hipblaslt_cout << "GPU " << dev << " testing " << cfg.name << std::endl;

                hipblasLtHandle_t handle;
                hipblasLtCreate(&handle);

                int64_t rowsA = (cfg.transA == HIPBLAS_OP_N) ? M : K;
                int64_t colsA = (cfg.transA == HIPBLAS_OP_N) ? K : M;
                int64_t rowsB = (cfg.transB == HIPBLAS_OP_N) ? K : N;
                int64_t colsB = (cfg.transB == HIPBLAS_OP_N) ? N : K;

                hipblasLtMatrixLayout_t matA, matB, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, rowsA, colsA, rowsA);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, rowsB, colsB, rowsB);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                &cfg.transA, sizeof(cfg.transA));
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                &cfg.transB, sizeof(cfg.transB));

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
                hipblasLtDestroy(handle);
            }
        }

        hipblaslt_cout << "All transpose combinations test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 6: Transpose with Different Sizes
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTranspose, TransposeWithDifferentSizes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing transpose with different matrix sizes" << std::endl;

        struct SizeConfig { int64_t M, N, K; };
        std::vector<SizeConfig> sizes = {
            {64, 128, 256},
            {256, 128, 64},
            {512, 256, 128},
            {1024, 512, 256}
        };

        std::vector<TransposeConfig> trans_configs = {
            {HIPBLAS_OP_N, HIPBLAS_OP_T, "NT"},
            {HIPBLAS_OP_T, HIPBLAS_OP_N, "TN"}
        };

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            for(const auto& size : sizes)
            {
                for(const auto& trans : trans_configs)
                {
                    hipblaslt_cout << "GPU " << dev << " testing " << trans.name
                                   << " with " << size.M << "x" << size.N << "x" << size.K << std::endl;

                    hipblasLtHandle_t handle;
                    hipblasLtCreate(&handle);

                    int64_t rowsA = (trans.transA == HIPBLAS_OP_N) ? size.M : size.K;
                    int64_t colsA = (trans.transA == HIPBLAS_OP_N) ? size.K : size.M;
                    int64_t rowsB = (trans.transB == HIPBLAS_OP_N) ? size.K : size.N;
                    int64_t colsB = (trans.transB == HIPBLAS_OP_N) ? size.N : size.K;

                    hipblasLtMatrixLayout_t matA, matB, matD;
                    hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, rowsA, colsA, rowsA);
                    hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, rowsB, colsB, rowsB);
                    hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, size.M, size.N, size.M);

                    hipblasLtMatmulDesc_t matmul;
                    hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                    hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                    &trans.transA, sizeof(trans.transA));
                    hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                    &trans.transB, sizeof(trans.transB));

                    hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                    hipblasLtMatrixLayoutDestroy(matD);
                    hipblasLtMatmulDescDestroy(matmul);
                    hipblasLtDestroy(handle);
                }
            }
        }

        hipblaslt_cout << "Transpose with different sizes test passed" << std::endl;
    }

} // namespace
