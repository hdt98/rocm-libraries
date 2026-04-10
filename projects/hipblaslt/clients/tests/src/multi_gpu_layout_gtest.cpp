/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Layout/Format Features Test Suite
 * Tests: Swizzle Operations, Different Layouts, Strided Batching
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
    // Test 1: Swizzle Operations across GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPULayout, SwizzleOperations)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing swizzle operations across " << numDevices << " GPUs" << std::endl;

        const int64_t M = 64;
        const int64_t N = 64;
        const int64_t K = 64;

        // Test swizzle-compatible data types
        std::vector<hipDataType> swizzle_types = {HIP_R_16F, HIP_R_16BF};
        std::vector<hipblasLtOrder_t> swizzle_orders = {
            HIPBLASLT_ORDER_COL16_4R8,
            HIPBLASLT_ORDER_COL16_4R16
        };

        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipDataType dtype = swizzle_types[deviceId % swizzle_types.size()];
            hipblasLtOrder_t order = swizzle_orders[deviceId % swizzle_orders.size()];

            hipblaslt_cout << "GPU " << deviceId << " testing swizzle with "
                           << hip_datatype_to_string(dtype) << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, dtype, M, K, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, dtype, K, N, K);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, dtype, M, N, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Set swizzle order
            status = hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                       &order, sizeof(order));
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            status = hipblasLtMatrixLayoutDestroy(matA);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutDestroy(matB);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutDestroy(matD);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtDestroy(handle);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        }

        hipblaslt_cout << "Swizzle operations test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 2: Different Memory Layouts across GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPULayout, DifferentMemoryLayouts)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing different memory layouts across " << numDevices << " GPUs" << std::endl;

        const int64_t M = 64;
        const int64_t N = 64;
        const int64_t K = 64;

        std::vector<hipblasLtOrder_t> layouts = {
            HIPBLASLT_ORDER_COL,  // Column-major
            HIPBLASLT_ORDER_ROW   // Row-major
        };

        const char* layout_names[] = {"Column-major", "Row-major"};

        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtOrder_t layout = layouts[deviceId % layouts.size()];
            hipblaslt_cout << "GPU " << deviceId << " testing " << layout_names[deviceId % 2] << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matC, matD;

            // For row-major, leading dimension is different
            int64_t ldA = (layout == HIPBLASLT_ORDER_ROW) ? K : M;
            int64_t ldB = (layout == HIPBLASLT_ORDER_ROW) ? N : K;
            int64_t ldC = (layout == HIPBLASLT_ORDER_ROW) ? N : M;
            int64_t ldD = (layout == HIPBLASLT_ORDER_ROW) ? N : M;

            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, ldA);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, ldB);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, ldC);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, ldD);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Set layout order
            status = hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                       &layout, sizeof(layout));
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                       &layout, sizeof(layout));
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                       &layout, sizeof(layout));
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                       &layout, sizeof(layout));
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatmulDesc_t matmul;
            status = hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasOperation_t opA = HIPBLAS_OP_N;
            hipblasOperation_t opB = HIPBLAS_OP_N;
            status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            float *d_a, *d_b, *d_c, *d_d;
            hipErr = hipMalloc(&d_a, M * K * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_b, K * N * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_c, M * N * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_d, M * N * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);

            std::vector<float> h_a(M * K, 1.0f);
            std::vector<float> h_b(K * N, 2.0f);
            std::vector<float> h_c(M * N, 0.0f);

            hipErr = hipMemcpy(d_a, h_a.data(), M * K * sizeof(float), hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMemcpy(d_b, h_b.data(), K * N * sizeof(float), hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMemcpy(d_c, h_c.data(), M * N * sizeof(float), hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtMatmulPreference_t pref;
            status = hipblasLtMatmulPreferenceCreate(&pref);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            size_t max_workspace_size = 32 * 1024 * 1024;
            status = hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size, sizeof(max_workspace_size));

            hipblasLtMatmulHeuristicResult_t heuristicResult[1];
            int returnedAlgoCount = 0;
            status = hipblasLtMatmulAlgoGetHeuristic(
                handle, matmul, matA, matB, matC, matD, pref, 1, heuristicResult, &returnedAlgoCount);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(returnedAlgoCount > 0)
            {
                void* d_workspace = nullptr;
                if(heuristicResult[0].workspaceSize > 0)
                {
                    hipErr = hipMalloc(&d_workspace, heuristicResult[0].workspaceSize);
                    ASSERT_EQ(hipErr, hipSuccess);
                }

                float alpha = 1.0f, beta = 0.0f;
                status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                         &beta, d_c, matC, d_d, matD,
                                         &heuristicResult[0].algo, d_workspace, heuristicResult[0].workspaceSize, 0);
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

                hipErr = hipDeviceSynchronize();
                ASSERT_EQ(hipErr, hipSuccess);

                std::vector<float> h_d(M * N);
                hipErr = hipMemcpy(h_d.data(), d_d, M * N * sizeof(float), hipMemcpyDeviceToHost);
                ASSERT_EQ(hipErr, hipSuccess);

                float expected = static_cast<float>(K) * 1.0f * 2.0f;
                EXPECT_NEAR(h_d[0], expected, 0.01f) << "GPU " << deviceId;

                if(d_workspace)
                {
                    hipErr = hipFree(d_workspace);
                    EXPECT_EQ(hipErr, hipSuccess);
                }
            }

            hipErr = hipFree(d_a);
            EXPECT_EQ(hipErr, hipSuccess);
            hipErr = hipFree(d_b);
            EXPECT_EQ(hipErr, hipSuccess);
            hipErr = hipFree(d_c);
            EXPECT_EQ(hipErr, hipSuccess);
            hipErr = hipFree(d_d);
            EXPECT_EQ(hipErr, hipSuccess);

            status = hipblasLtMatmulPreferenceDestroy(pref);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutDestroy(matA);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutDestroy(matB);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutDestroy(matC);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutDestroy(matD);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatmulDescDestroy(matmul);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtDestroy(handle);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        }

        hipblaslt_cout << "Different memory layouts test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 3: Strided Batching Patterns across GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPULayout, StridedBatchingPatterns)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing strided batching patterns across " << numDevices << " GPUs" << std::endl;

        const int64_t M = 64;
        const int64_t N = 64;
        const int64_t K = 64;
        const int64_t batch_count = 4;

        // Different stride patterns per GPU
        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            // Different stride multipliers
            int64_t stride_multiplier = (deviceId == 0) ? 1 : 2;
            int64_t strideA = M * K * stride_multiplier;
            int64_t strideB = K * N * stride_multiplier;
            int64_t strideC = M * N * stride_multiplier;
            int64_t strideD = M * N * stride_multiplier;

            hipblaslt_cout << "GPU " << deviceId << " using stride multiplier: "
                           << stride_multiplier << std::endl;

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

            // Set batch count
            status = hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                       &batch_count, sizeof(batch_count));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                       &batch_count, sizeof(batch_count));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                       &batch_count, sizeof(batch_count));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                       &batch_count, sizeof(batch_count));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Set strides
            status = hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                       &strideA, sizeof(strideA));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                       &strideB, sizeof(strideB));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                       &strideC, sizeof(strideC));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                       &strideD, sizeof(strideD));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            status = hipblasLtMatrixLayoutDestroy(matA);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutDestroy(matB);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutDestroy(matC);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutDestroy(matD);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtDestroy(handle);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        }

        hipblaslt_cout << "Strided batching patterns test passed" << std::endl;
    }

} // namespace
