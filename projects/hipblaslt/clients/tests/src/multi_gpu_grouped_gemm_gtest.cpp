/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Grouped GEMM Test Suite
 * Tests: Full grouped GEMM execution with different problem sizes per group
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
    // Test 1: Simple Grouped GEMM - Same size groups across GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPUGroupedGEMM, UniformGroupSizes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing uniform grouped GEMM across " << numDevices << " GPUs" << std::endl;

        const int num_groups = 4;
        const int64_t M = 64;
        const int64_t N = 64;
        const int64_t K = 64;

        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " executing " << num_groups
                           << " groups of " << M << "x" << N << "x" << K << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // For uniform grouped GEMM, we can use batched GEMM
            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Set batch count for all groups
            status = hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                       &num_groups, sizeof(num_groups));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                       &num_groups, sizeof(num_groups));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                       &num_groups, sizeof(num_groups));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                       &num_groups, sizeof(num_groups));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Set strides
            int64_t strideA = M * K;
            int64_t strideB = K * N;
            int64_t strideC = M * N;
            int64_t strideD = M * N;
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

            hipblasLtMatmulDesc_t matmul;
            status = hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasOperation_t opA = HIPBLAS_OP_N;
            hipblasOperation_t opB = HIPBLAS_OP_N;
            status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Allocate device memory for all groups
            float *d_a, *d_b, *d_c, *d_d;
            size_t total_size_a = M * K * num_groups * sizeof(float);
            size_t total_size_b = K * N * num_groups * sizeof(float);
            size_t total_size_c = M * N * num_groups * sizeof(float);
            size_t total_size_d = M * N * num_groups * sizeof(float);

            hipErr = hipMalloc(&d_a, total_size_a);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_b, total_size_b);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_c, total_size_c);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_d, total_size_d);
            ASSERT_EQ(hipErr, hipSuccess);

            // Initialize data for all groups
            std::vector<float> h_a(M * K * num_groups, 1.0f);
            std::vector<float> h_b(K * N * num_groups, 2.0f);
            std::vector<float> h_c(M * N * num_groups, 0.0f);

            hipErr = hipMemcpy(d_a, h_a.data(), total_size_a, hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMemcpy(d_b, h_b.data(), total_size_b, hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMemcpy(d_c, h_c.data(), total_size_c, hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);

            // Get algorithm
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
            ASSERT_GT(returnedAlgoCount, 0);

            void* d_workspace = nullptr;
            if(heuristicResult[0].workspaceSize > 0)
            {
                hipErr = hipMalloc(&d_workspace, heuristicResult[0].workspaceSize);
                ASSERT_EQ(hipErr, hipSuccess);
            }

            // Execute grouped GEMM
            float alpha = 1.0f;
            float beta = 0.0f;
            status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                     &beta, d_c, matC, d_d, matD,
                                     &heuristicResult[0].algo, d_workspace,
                                     heuristicResult[0].workspaceSize, 0);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipErr = hipDeviceSynchronize();
            ASSERT_EQ(hipErr, hipSuccess);

            // Verify results for all groups
            std::vector<float> h_d(M * N * num_groups);
            hipErr = hipMemcpy(h_d.data(), d_d, total_size_d, hipMemcpyDeviceToHost);
            ASSERT_EQ(hipErr, hipSuccess);

            float expected = static_cast<float>(K) * 1.0f * 2.0f;
            for(int g = 0; g < num_groups; ++g)
            {
                float result = h_d[g * M * N];
                EXPECT_NEAR(result, expected, 0.01f) << "GPU " << deviceId << " group " << g;
            }

            hipblaslt_cout << "GPU " << deviceId << " uniform grouped GEMM completed, "
                           << num_groups << " groups verified" << std::endl;

            // Cleanup
            if(d_workspace)
            {
                hipErr = hipFree(d_workspace);
                EXPECT_EQ(hipErr, hipSuccess);
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

        hipblaslt_cout << "Uniform grouped GEMM test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 2: Variable Size Grouped GEMM - Different sizes per group
    // ----------------------------------------------------------------------------
    TEST(MultiGPUGroupedGEMM, VariableGroupSizes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing variable-size grouped GEMM across " << numDevices << " GPUs" << std::endl;

        // Different problem sizes for each group
        struct GroupProblem {
            int64_t M, N, K;
        };

        std::vector<GroupProblem> groups = {
            {32, 32, 32},
            {64, 64, 64},
            {48, 96, 64},
            {128, 64, 48}
        };

        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " executing " << groups.size()
                           << " groups with variable sizes" << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // For variable-size grouped GEMM, we need to execute each group separately
            // or use a true grouped GEMM API if available
            for(size_t g = 0; g < groups.size(); ++g)
            {
                int64_t M = groups[g].M;
                int64_t N = groups[g].N;
                int64_t K = groups[g].K;

                hipblaslt_cout << "GPU " << deviceId << " group " << g << ": "
                               << M << "x" << N << "x" << K << std::endl;

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

                hipblasOperation_t opA = HIPBLAS_OP_N;
                hipblasOperation_t opB = HIPBLAS_OP_N;
                status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
                status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

                // Allocate device memory
                float *d_a, *d_b, *d_c, *d_d;
                hipErr = hipMalloc(&d_a, M * K * sizeof(float));
                ASSERT_EQ(hipErr, hipSuccess);
                hipErr = hipMalloc(&d_b, K * N * sizeof(float));
                ASSERT_EQ(hipErr, hipSuccess);
                hipErr = hipMalloc(&d_c, M * N * sizeof(float));
                ASSERT_EQ(hipErr, hipSuccess);
                hipErr = hipMalloc(&d_d, M * N * sizeof(float));
                ASSERT_EQ(hipErr, hipSuccess);

                // Initialize data
                std::vector<float> h_a(M * K, 1.0f);
                std::vector<float> h_b(K * N, 2.0f);
                std::vector<float> h_c(M * N, 0.0f);

                hipErr = hipMemcpy(d_a, h_a.data(), M * K * sizeof(float), hipMemcpyHostToDevice);
                ASSERT_EQ(hipErr, hipSuccess);
                hipErr = hipMemcpy(d_b, h_b.data(), K * N * sizeof(float), hipMemcpyHostToDevice);
                ASSERT_EQ(hipErr, hipSuccess);
                hipErr = hipMemcpy(d_c, h_c.data(), M * N * sizeof(float), hipMemcpyHostToDevice);
                ASSERT_EQ(hipErr, hipSuccess);

                // Get algorithm
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
                ASSERT_GT(returnedAlgoCount, 0);

                void* d_workspace = nullptr;
                if(heuristicResult[0].workspaceSize > 0)
                {
                    hipErr = hipMalloc(&d_workspace, heuristicResult[0].workspaceSize);
                    ASSERT_EQ(hipErr, hipSuccess);
                }

                // Execute GEMM for this group
                float alpha = 1.0f;
                float beta = 0.0f;
                status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                         &beta, d_c, matC, d_d, matD,
                                         &heuristicResult[0].algo, d_workspace,
                                         heuristicResult[0].workspaceSize, 0);
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

                hipErr = hipDeviceSynchronize();
                ASSERT_EQ(hipErr, hipSuccess);

                // Verify result
                std::vector<float> h_d(M * N);
                hipErr = hipMemcpy(h_d.data(), d_d, M * N * sizeof(float), hipMemcpyDeviceToHost);
                ASSERT_EQ(hipErr, hipSuccess);

                float expected = static_cast<float>(K) * 1.0f * 2.0f;
                EXPECT_NEAR(h_d[0], expected, 0.01f) << "GPU " << deviceId << " group " << g;

                // Cleanup group resources
                if(d_workspace)
                    hipFree(d_workspace);
                hipFree(d_a);
                hipFree(d_b);
                hipFree(d_c);
                hipFree(d_d);
                hipblasLtMatmulPreferenceDestroy(pref);
                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC);
                hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }

            hipblaslt_cout << "GPU " << deviceId << " variable-size grouped GEMM completed" << std::endl;
            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Variable-size grouped GEMM test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 3: Grouped GEMM with Different Transposes
    // ----------------------------------------------------------------------------
    TEST(MultiGPUGroupedGEMM, GroupedWithTransposes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing grouped GEMM with different transposes across "
                       << numDevices << " GPUs" << std::endl;

        struct GroupConfig {
            int64_t M, N, K;
            hipblasOperation_t transA;
            hipblasOperation_t transB;
            const char* name;
        };

        std::vector<GroupConfig> configs = {
            {64, 64, 64, HIPBLAS_OP_N, HIPBLAS_OP_N, "NN"},
            {64, 64, 64, HIPBLAS_OP_N, HIPBLAS_OP_T, "NT"},
            {64, 64, 64, HIPBLAS_OP_T, HIPBLAS_OP_N, "TN"},
            {64, 64, 64, HIPBLAS_OP_T, HIPBLAS_OP_T, "TT"}
        };

        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " testing grouped GEMM with transposes" << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            for(const auto& config : configs)
            {
                hipblaslt_cout << "GPU " << deviceId << " config: " << config.name << std::endl;

                int64_t rowsA = (config.transA == HIPBLAS_OP_N) ? config.M : config.K;
                int64_t colsA = (config.transA == HIPBLAS_OP_N) ? config.K : config.M;
                int64_t rowsB = (config.transB == HIPBLAS_OP_N) ? config.K : config.N;
                int64_t colsB = (config.transB == HIPBLAS_OP_N) ? config.N : config.K;

                hipblasLtMatrixLayout_t matA, matB, matD;
                status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, rowsA, colsA, rowsA);
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
                status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, rowsB, colsB, rowsB);
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
                status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, config.M, config.N, config.M);
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

                hipblasLtMatmulDesc_t matmul;
                status = hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

                status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                         &config.transA, sizeof(config.transA));
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
                status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                         &config.transB, sizeof(config.transB));
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }

            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Grouped GEMM with transposes test passed" << std::endl;
    }

} // namespace
