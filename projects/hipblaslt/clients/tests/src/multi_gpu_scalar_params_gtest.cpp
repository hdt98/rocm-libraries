/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Scalar Parameters Test Suite
 * Comprehensive variations of alpha and beta scalars
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
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

    // Helper to run GEMM with specific alpha/beta
    bool runGEMMWithScalars(int deviceId, float alpha, float beta, float& result)
    {
        auto hipErr = hipSetDevice(deviceId);
        if(hipErr != hipSuccess) return false;

        const int64_t M = 64, N = 64, K = 64;

        hipblasLtHandle_t handle;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS) return false;

        hipblasLtMatrixLayout_t matA, matB, matC, matD;
        hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
        hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
        hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
        hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

        hipblasLtMatmulDesc_t matmul;
        hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

        hipblasOperation_t op = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
        hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));

        float *d_a, *d_b, *d_c, *d_d;
        hipMalloc(&d_a, M * K * sizeof(float));
        hipMalloc(&d_b, K * N * sizeof(float));
        hipMalloc(&d_c, M * N * sizeof(float));
        hipMalloc(&d_d, M * N * sizeof(float));

        std::vector<float> h_a(M * K, 1.0f);
        std::vector<float> h_b(K * N, 2.0f);
        std::vector<float> h_c(M * N, 3.0f);

        hipMemcpy(d_a, h_a.data(), M * K * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_b, h_b.data(), K * N * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_c, h_c.data(), M * N * sizeof(float), hipMemcpyHostToDevice);

        hipblasLtMatmulPreference_t pref;
        hipblasLtMatmulPreferenceCreate(&pref);
        size_t workspace_size = 32 * 1024 * 1024;
        hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &workspace_size, sizeof(workspace_size));

        hipblasLtMatmulHeuristicResult_t heuristicResult[1];
        int algoCount = 0;
        auto status = hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matC, matD,
                                                      pref, 1, heuristicResult, &algoCount);

        bool success = false;
        if(status == HIPBLAS_STATUS_SUCCESS && algoCount > 0)
        {
            void* workspace = nullptr;
            if(heuristicResult[0].workspaceSize > 0)
                hipMalloc(&workspace, heuristicResult[0].workspaceSize);

            status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                     &beta, d_c, matC, d_d, matD,
                                     &heuristicResult[0].algo, workspace,
                                     heuristicResult[0].workspaceSize, 0);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipDeviceSynchronize();
                std::vector<float> h_d(M * N);
                hipMemcpy(h_d.data(), d_d, M * N * sizeof(float), hipMemcpyDeviceToHost);
                result = h_d[0];
                success = true;
            }

            if(workspace) hipFree(workspace);
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
    // Test 1: Standard GEMM (alpha=1, beta=0)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUScalarParams, Standard_Alpha1_Beta0)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing standard alpha=1, beta=0 across GPUs" << std::endl;

        float alpha = 1.0f, beta = 0.0f;
        const int64_t K = 64;
        float expected = K * 1.0f * 2.0f; // alpha * K * A[i] * B[j]

        for(int dev = 0; dev < numDevices; ++dev)
        {
            float result;
            bool success = runGEMMWithScalars(dev, alpha, beta, result);
            ASSERT_TRUE(success) << "GPU " << dev;
            EXPECT_NEAR(result, expected, 0.01f) << "GPU " << dev;
            hipblaslt_cout << "GPU " << dev << " result: " << result << " (expected: " << expected << ")" << std::endl;
        }
    }

    // ----------------------------------------------------------------------------
    // Test 2: Accumulate (alpha=1, beta=1)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUScalarParams, Accumulate_Alpha1_Beta1)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing accumulate alpha=1, beta=1 across GPUs" << std::endl;

        float alpha = 1.0f, beta = 1.0f;
        const int64_t K = 64;
        float expected = K * 1.0f * 2.0f + 3.0f; // alpha*A*B + beta*C

        for(int dev = 0; dev < numDevices; ++dev)
        {
            float result;
            bool success = runGEMMWithScalars(dev, alpha, beta, result);
            ASSERT_TRUE(success) << "GPU " << dev;
            EXPECT_NEAR(result, expected, 0.01f) << "GPU " << dev;
            hipblaslt_cout << "GPU " << dev << " accumulate result: " << result << std::endl;
        }
    }

    // ----------------------------------------------------------------------------
    // Test 3: Scale Only (alpha=0, beta=1) - D = C
    // ----------------------------------------------------------------------------
    TEST(MultiGPUScalarParams, ScaleOnly_Alpha0_Beta1)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing scale only alpha=0, beta=1 across GPUs" << std::endl;

        float alpha = 0.0f, beta = 1.0f;
        float expected = 3.0f; // D = beta*C = 1*3 = 3

        for(int dev = 0; dev < numDevices; ++dev)
        {
            float result;
            bool success = runGEMMWithScalars(dev, alpha, beta, result);
            ASSERT_TRUE(success) << "GPU " << dev;
            EXPECT_NEAR(result, expected, 0.01f) << "GPU " << dev;
            hipblaslt_cout << "GPU " << dev << " scale-only result: " << result << std::endl;
        }
    }

    // ----------------------------------------------------------------------------
    // Test 4: Zero Output (alpha=0, beta=0)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUScalarParams, ZeroOutput_Alpha0_Beta0)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing zero output alpha=0, beta=0 across GPUs" << std::endl;

        float alpha = 0.0f, beta = 0.0f;
        float expected = 0.0f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            float result;
            bool success = runGEMMWithScalars(dev, alpha, beta, result);
            ASSERT_TRUE(success) << "GPU " << dev;
            EXPECT_NEAR(result, expected, 0.01f) << "GPU " << dev;
            hipblaslt_cout << "GPU " << dev << " zero output result: " << result << std::endl;
        }
    }

    // ----------------------------------------------------------------------------
    // Test 5: Scaled GEMM (alpha=2.0, beta=0)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUScalarParams, Scaled_Alpha2_Beta0)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing scaled alpha=2.0, beta=0 across GPUs" << std::endl;

        float alpha = 2.0f, beta = 0.0f;
        const int64_t K = 64;
        float expected = 2.0f * K * 1.0f * 2.0f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            float result;
            bool success = runGEMMWithScalars(dev, alpha, beta, result);
            ASSERT_TRUE(success) << "GPU " << dev;
            EXPECT_NEAR(result, expected, 0.01f) << "GPU " << dev;
            hipblaslt_cout << "GPU " << dev << " scaled result: " << result << std::endl;
        }
    }

    // ----------------------------------------------------------------------------
    // Test 6: Fractional Scaling (alpha=0.5, beta=0.5)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUScalarParams, Fractional_AlphaHalf_BetaHalf)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing fractional alpha=0.5, beta=0.5 across GPUs" << std::endl;

        float alpha = 0.5f, beta = 0.5f;
        const int64_t K = 64;
        float expected = 0.5f * K * 1.0f * 2.0f + 0.5f * 3.0f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            float result;
            bool success = runGEMMWithScalars(dev, alpha, beta, result);
            ASSERT_TRUE(success) << "GPU " << dev;
            EXPECT_NEAR(result, expected, 0.01f) << "GPU " << dev;
            hipblaslt_cout << "GPU " << dev << " fractional result: " << result << std::endl;
        }
    }

    // ----------------------------------------------------------------------------
    // Test 7: Negative Alpha (alpha=-1, beta=0)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUScalarParams, Negative_AlphaNeg1_Beta0)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing negative alpha=-1, beta=0 across GPUs" << std::endl;

        float alpha = -1.0f, beta = 0.0f;
        const int64_t K = 64;
        float expected = -1.0f * K * 1.0f * 2.0f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            float result;
            bool success = runGEMMWithScalars(dev, alpha, beta, result);
            ASSERT_TRUE(success) << "GPU " << dev;
            EXPECT_NEAR(result, expected, 0.01f) << "GPU " << dev;
            hipblaslt_cout << "GPU " << dev << " negative alpha result: " << result << std::endl;
        }
    }

    // ----------------------------------------------------------------------------
    // Test 8: Negative Beta (alpha=1, beta=-1)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUScalarParams, Negative_Alpha1_BetaNeg1)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing negative beta=-1 across GPUs" << std::endl;

        float alpha = 1.0f, beta = -1.0f;
        const int64_t K = 64;
        float expected = K * 1.0f * 2.0f - 3.0f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            float result;
            bool success = runGEMMWithScalars(dev, alpha, beta, result);
            ASSERT_TRUE(success) << "GPU " << dev;
            EXPECT_NEAR(result, expected, 0.01f) << "GPU " << dev;
            hipblaslt_cout << "GPU " << dev << " negative beta result: " << result << std::endl;
        }
    }

    // ----------------------------------------------------------------------------
    // Test 9: Very Small Alpha (near underflow)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUScalarParams, VerySmall_AlphaTiny_Beta0)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing very small alpha across GPUs" << std::endl;

        float alpha = 1e-6f, beta = 0.0f;
        const int64_t K = 64;
        float expected = 1e-6f * K * 1.0f * 2.0f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            float result;
            bool success = runGEMMWithScalars(dev, alpha, beta, result);
            ASSERT_TRUE(success) << "GPU " << dev;
            EXPECT_NEAR(result, expected, 1e-5f) << "GPU " << dev;
            hipblaslt_cout << "GPU " << dev << " very small alpha result: " << result << std::endl;
        }
    }

    // ----------------------------------------------------------------------------
    // Test 10: Very Large Alpha
    // ----------------------------------------------------------------------------
    TEST(MultiGPUScalarParams, VeryLarge_AlphaLarge_Beta0)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing very large alpha across GPUs" << std::endl;

        float alpha = 1e6f, beta = 0.0f;
        const int64_t K = 64;
        float expected = 1e6f * K * 1.0f * 2.0f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            float result;
            bool success = runGEMMWithScalars(dev, alpha, beta, result);
            ASSERT_TRUE(success) << "GPU " << dev;
            EXPECT_NEAR(result, expected, expected * 0.001f) << "GPU " << dev;
            hipblaslt_cout << "GPU " << dev << " very large alpha result: " << result << std::endl;
        }
    }

    // ----------------------------------------------------------------------------
    // Test 11: Different Scalars Per GPU
    // ----------------------------------------------------------------------------
    TEST(MultiGPUScalarParams, DifferentScalarsPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing different scalars per GPU" << std::endl;

        std::vector<float> alphas = {1.0f, 2.0f, 0.5f, 3.0f};
        std::vector<float> betas = {0.0f, 0.5f, 1.0f, 0.25f};
        const int64_t K = 64;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            // Cycle through alpha/beta values for all GPUs
            float alpha = alphas[dev % alphas.size()];
            float beta = betas[dev % betas.size()];
            float expected = alpha * K * 1.0f * 2.0f + beta * 3.0f;

            float result;
            bool success = runGEMMWithScalars(dev, alpha, beta, result);
            ASSERT_TRUE(success) << "GPU " << dev;
            EXPECT_NEAR(result, expected, 0.01f) << "GPU " << dev << " alpha=" << alpha << " beta=" << beta;
            hipblaslt_cout << "GPU " << dev << " alpha=" << alpha << " beta=" << beta
                           << " result: " << result << std::endl;
        }
    }

} // namespace
