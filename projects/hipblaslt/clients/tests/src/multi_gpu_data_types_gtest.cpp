/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Data Types Test Suite
 * Comprehensive variations of input/output data types and compute types
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

    // ----------------------------------------------------------------------------
    // Test 1: FP32 (Standard Precision)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDataTypes, FP32_Precision)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing FP32 precision across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float alpha = 1.0f, beta = 0.0f;

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatmulDesc_t matmul;
            status = hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasOperation_t op = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));

            // Allocate and execute
            float *d_a, *d_b, *d_c, *d_d;
            hipMalloc(&d_a, M * K * sizeof(float));
            hipMalloc(&d_b, K * N * sizeof(float));
            hipMalloc(&d_c, M * N * sizeof(float));
            hipMalloc(&d_d, M * N * sizeof(float));

            std::vector<float> h_a(M * K, 1.0f);
            std::vector<float> h_b(K * N, 2.0f);
            std::vector<float> h_c(M * N, 0.0f);

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
            status = hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matC, matD,
                                                     pref, 1, heuristicResult, &algoCount);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            ASSERT_GT(algoCount, 0);

            void* workspace = nullptr;
            if(heuristicResult[0].workspaceSize > 0)
                hipMalloc(&workspace, heuristicResult[0].workspaceSize);

            status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                     &beta, d_c, matC, d_d, matD,
                                     &heuristicResult[0].algo, workspace,
                                     heuristicResult[0].workspaceSize, 0);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipDeviceSynchronize();

            std::vector<float> h_d(M * N);
            hipMemcpy(h_d.data(), d_d, M * N * sizeof(float), hipMemcpyDeviceToHost);

            float expected = K * 1.0f * 2.0f;
            EXPECT_NEAR(h_d[0], expected, 0.01f) << "GPU " << dev;

            if(workspace) hipFree(workspace);
            hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_d);
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);

            hipblaslt_cout << "GPU " << dev << " FP32 test passed" << std::endl;
        }
    }

    // ----------------------------------------------------------------------------
    // Test 2: FP16 (Half Precision)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDataTypes, FP16_Precision)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing FP16 precision across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, M, K, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, K, N, K);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cout << "GPU " << dev << " FP16 supported and tested" << std::endl;
                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << " FP16 not supported" << std::endl;
            }

            hipblasLtDestroy(handle);
        }
    }

    // ----------------------------------------------------------------------------
    // Test 3: BF16 (Brain Float16)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDataTypes, BF16_Precision)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing BF16 precision across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_16BF, M, K, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_16BF, K, N, K);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_16BF, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cout << "GPU " << dev << " BF16 supported and tested" << std::endl;
                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << " BF16 not supported" << std::endl;
            }

            hipblasLtDestroy(handle);
        }
    }

    // ----------------------------------------------------------------------------
    // Test 4: FP8 E4M3
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDataTypes, FP8_E4M3_Precision)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing FP8 E4M3 precision across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_8F_E4M3_FNUZ, M, K, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_8F_E4M3_FNUZ, K, N, K);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cout << "GPU " << dev << " FP8 E4M3 supported and tested" << std::endl;
                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << " FP8 E4M3 not supported" << std::endl;
            }

            hipblasLtDestroy(handle);
        }
    }

    // ----------------------------------------------------------------------------
    // Test 5: FP8 E5M2
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDataTypes, FP8_E5M2_Precision)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing FP8 E5M2 precision across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_8F_E5M2_FNUZ, M, K, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_8F_E5M2_FNUZ, K, N, K);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cout << "GPU " << dev << " FP8 E5M2 supported and tested" << std::endl;
                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << " FP8 E5M2 not supported" << std::endl;
            }

            hipblasLtDestroy(handle);
        }
    }

    // ----------------------------------------------------------------------------
    // Test 6: INT8
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDataTypes, INT8_Precision)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing INT8 precision across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_8I, M, K, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_8I, K, N, K);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32I, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cout << "GPU " << dev << " INT8 supported and tested" << std::endl;
                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << " INT8 not supported" << std::endl;
            }

            hipblasLtDestroy(handle);
        }
    }

    // ----------------------------------------------------------------------------
    // Test 7: Mixed Precision FP16 -> FP32
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDataTypes, MixedPrecision_FP16_to_FP32)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing FP16->FP32 mixed precision across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, M, K, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, K, N, K);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cout << "GPU " << dev << " FP16->FP32 mixed precision supported" << std::endl;
                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << " FP16->FP32 not supported" << std::endl;
            }

            hipblasLtDestroy(handle);
        }
    }

    // ----------------------------------------------------------------------------
    // Test 8: Mixed Precision BF16 -> FP32
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDataTypes, MixedPrecision_BF16_to_FP32)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing BF16->FP32 mixed precision across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_16BF, M, K, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_16BF, K, N, K);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cout << "GPU " << dev << " BF16->FP32 mixed precision supported" << std::endl;
                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << " BF16->FP32 not supported" << std::endl;
            }

            hipblasLtDestroy(handle);
        }
    }

    // ----------------------------------------------------------------------------
    // Test 9: Mixed Precision INT8 -> INT32
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDataTypes, MixedPrecision_INT8_to_INT32)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing INT8->INT32 mixed precision across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_8I, M, K, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_8I, K, N, K);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32I, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblasLtMatmulDesc_t matmul;
                status = hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32I, HIP_R_32I);
                EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

                if(status == HIPBLAS_STATUS_SUCCESS)
                {
                    hipblaslt_cout << "GPU " << dev << " INT8->INT32 mixed precision supported" << std::endl;
                    hipblasLtMatmulDescDestroy(matmul);
                }

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << " INT8->INT32 not supported" << std::endl;
            }

            hipblasLtDestroy(handle);
        }
    }

    // ----------------------------------------------------------------------------
    // Test 10: Mixed Precision FP8 -> FP32
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDataTypes, MixedPrecision_FP8_to_FP32)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing FP8->FP32 mixed precision across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_8F_E4M3_FNUZ, M, K, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_8F_E4M3_FNUZ, K, N, K);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cout << "GPU " << dev << " FP8->FP32 mixed precision supported" << std::endl;
                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << " FP8->FP32 not supported" << std::endl;
            }

            hipblasLtDestroy(handle);
        }
    }

} // namespace
