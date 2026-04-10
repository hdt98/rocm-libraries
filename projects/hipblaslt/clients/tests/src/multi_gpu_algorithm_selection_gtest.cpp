/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Algorithm Selection Test Suite
 * Tests: Different algorithms per GPU, solution indices, algorithm preferences
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
    // Test 1: Different Algorithms on Different GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPUAlgorithmSelection, DifferentAlgorithmsPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing different algorithms on different GPUs" << std::endl;

        const int64_t M = 256;
        const int64_t N = 256;
        const int64_t K = 256;
        float alpha = 1.0f;
        float beta = 0.0f;

        for(int deviceId = 0; deviceId < numDevices; ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " querying available algorithms" << std::endl;

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

            hipblasOperation_t opA = HIPBLAS_OP_N;
            hipblasOperation_t opB = HIPBLAS_OP_N;
            status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatmulPreference_t pref;
            status = hipblasLtMatmulPreferenceCreate(&pref);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            size_t max_workspace_size = 32 * 1024 * 1024;
            status = hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size, sizeof(max_workspace_size));

            // Get multiple algorithms
            const int max_algos = 10;
            hipblasLtMatmulHeuristicResult_t heuristicResults[max_algos];
            int returnedAlgoCount = 0;
            status = hipblasLtMatmulAlgoGetHeuristic(
                handle, matmul, matA, matB, matC, matD, pref, max_algos, heuristicResults, &returnedAlgoCount);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblaslt_cout << "GPU " << deviceId << " found " << returnedAlgoCount << " algorithms" << std::endl;

            // Use different algorithm index on each GPU
            int algo_index = deviceId % returnedAlgoCount;

            if(returnedAlgoCount > 0)
            {
                hipblaslt_cout << "GPU " << deviceId << " using algorithm index " << algo_index
                               << ", workspace: " << heuristicResults[algo_index].workspaceSize << " bytes" << std::endl;

                // Allocate matrices
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

                void* d_workspace = nullptr;
                if(heuristicResults[algo_index].workspaceSize > 0)
                {
                    hipErr = hipMalloc(&d_workspace, heuristicResults[algo_index].workspaceSize);
                    ASSERT_EQ(hipErr, hipSuccess);
                }

                // Execute with selected algorithm
                status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                         &beta, d_c, matC, d_d, matD,
                                         &heuristicResults[algo_index].algo, d_workspace,
                                         heuristicResults[algo_index].workspaceSize, 0);
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

                hipErr = hipDeviceSynchronize();
                ASSERT_EQ(hipErr, hipSuccess);

                // Verify result
                std::vector<float> h_d(M * N);
                hipErr = hipMemcpy(h_d.data(), d_d, M * N * sizeof(float), hipMemcpyDeviceToHost);
                ASSERT_EQ(hipErr, hipSuccess);

                float expected = static_cast<float>(K) * 1.0f * 2.0f;
                EXPECT_NEAR(h_d[0], expected, 0.01f) << "GPU " << deviceId << " algo " << algo_index;

                hipblaslt_cout << "GPU " << deviceId << " algo " << algo_index
                               << " result: " << h_d[0] << std::endl;

                // Cleanup
                if(d_workspace)
                    hipFree(d_workspace);
                hipFree(d_a);
                hipFree(d_b);
                hipFree(d_c);
                hipFree(d_d);
            }

            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Different algorithms per GPU test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 2: Workspace Size Constraints per GPU
    // ----------------------------------------------------------------------------
    TEST(MultiGPUAlgorithmSelection, WorkspaceSizeConstraints)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing workspace size constraints across GPUs" << std::endl;

        const int64_t M = 512;
        const int64_t N = 512;
        const int64_t K = 512;

        // Different workspace limits for each GPU
        std::vector<size_t> workspace_limits = {
            0,              // GPU 0: No workspace
            4 * 1024 * 1024,    // GPU 1: 4MB
            16 * 1024 * 1024,   // GPU 2: 16MB
            64 * 1024 * 1024    // GPU 3: 64MB
        };

        for(int deviceId = 0; deviceId < numDevices; ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            size_t max_workspace = workspace_limits[deviceId];
            hipblaslt_cout << "GPU " << deviceId << " workspace limit: "
                           << (max_workspace / 1024 / 1024) << " MB" << std::endl;

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

            hipblasOperation_t opA = HIPBLAS_OP_N;
            hipblasOperation_t opB = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

            hipblasLtMatmulPreference_t pref;
            status = hipblasLtMatmulPreferenceCreate(&pref);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Set workspace limit for this GPU
            status = hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, sizeof(max_workspace));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Get algorithms that fit within workspace constraint
            const int max_algos = 10;
            hipblasLtMatmulHeuristicResult_t heuristicResults[max_algos];
            int returnedAlgoCount = 0;
            status = hipblasLtMatmulAlgoGetHeuristic(
                handle, matmul, matA, matB, matC, matD, pref, max_algos, heuristicResults, &returnedAlgoCount);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblaslt_cout << "GPU " << deviceId << " found " << returnedAlgoCount
                           << " algorithms within workspace limit" << std::endl;

            // Verify all returned algorithms fit within workspace limit
            for(int i = 0; i < returnedAlgoCount; ++i)
            {
                EXPECT_LE(heuristicResults[i].workspaceSize, max_workspace)
                    << "GPU " << deviceId << " algo " << i << " exceeds workspace limit";

                if(i < 3) // Log first few
                {
                    hipblaslt_cout << "  Algo " << i << " workspace: "
                                   << heuristicResults[i].workspaceSize << " bytes" << std::endl;
                }
            }

            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Workspace size constraints test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 3: Solution Index Selection Across GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPUAlgorithmSelection, SolutionIndexSelection)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing solution index selection across GPUs" << std::endl;

        const int64_t M = 128;
        const int64_t N = 128;
        const int64_t K = 128;

        for(int deviceId = 0; deviceId < numDevices; ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " testing solution indices" << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatmulDesc_t matmul;
            status = hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasOperation_t opA = HIPBLAS_OP_N;
            hipblasOperation_t opB = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

            hipblasLtMatmulPreference_t pref;
            status = hipblasLtMatmulPreferenceCreate(&pref);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            size_t max_workspace_size = 32 * 1024 * 1024;
            hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size, sizeof(max_workspace_size));

            // Request multiple solutions
            const int num_requested = 5;
            hipblasLtMatmulHeuristicResult_t heuristicResults[num_requested];
            int returnedAlgoCount = 0;
            status = hipblasLtMatmulAlgoGetHeuristic(
                handle, matmul, matA, matB, matB, matD, pref, num_requested, heuristicResults, &returnedAlgoCount);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblaslt_cout << "GPU " << deviceId << " received " << returnedAlgoCount << " solutions" << std::endl;

            // Log solution properties
            for(int i = 0; i < std::min(returnedAlgoCount, 3); ++i)
            {
                hipblaslt_cout << "  Solution " << i << ": workspace="
                               << heuristicResults[i].workspaceSize << " bytes, state="
                               << heuristicResults[i].state << std::endl;
            }

            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Solution index selection test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 4: Algorithm Performance Comparison
    // ----------------------------------------------------------------------------
    TEST(MultiGPUAlgorithmSelection, AlgorithmPerformanceComparison)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing algorithm performance comparison across GPUs" << std::endl;

        const int64_t M = 1024;
        const int64_t N = 1024;
        const int64_t K = 1024;
        const int num_iterations = 5;
        float alpha = 1.0f;
        float beta = 0.0f;

        for(int deviceId = 0; deviceId < numDevices; ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " comparing algorithm performance" << std::endl;

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

            hipblasOperation_t opA = HIPBLAS_OP_N;
            hipblasOperation_t opB = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

            // Allocate matrices
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

            hipMemcpy(d_a, h_a.data(), M * K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_b, h_b.data(), K * N * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_c, h_c.data(), M * N * sizeof(float), hipMemcpyHostToDevice);

            hipblasLtMatmulPreference_t pref;
            status = hipblasLtMatmulPreferenceCreate(&pref);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            size_t max_workspace_size = 64 * 1024 * 1024;
            hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size, sizeof(max_workspace_size));

            // Get algorithms
            const int max_algos = 3;
            hipblasLtMatmulHeuristicResult_t heuristicResults[max_algos];
            int returnedAlgoCount = 0;
            status = hipblasLtMatmulAlgoGetHeuristic(
                handle, matmul, matA, matB, matC, matD, pref, max_algos, heuristicResults, &returnedAlgoCount);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Test each algorithm
            for(int algo_idx = 0; algo_idx < std::min(returnedAlgoCount, max_algos); ++algo_idx)
            {
                void* d_workspace = nullptr;
                if(heuristicResults[algo_idx].workspaceSize > 0)
                {
                    hipErr = hipMalloc(&d_workspace, heuristicResults[algo_idx].workspaceSize);
                    ASSERT_EQ(hipErr, hipSuccess);
                }

                // Warmup
                status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                         &beta, d_c, matC, d_d, matD,
                                         &heuristicResults[algo_idx].algo, d_workspace,
                                         heuristicResults[algo_idx].workspaceSize, 0);

                hipDeviceSynchronize();

                // Time multiple iterations
                hipEvent_t start, stop;
                hipEventCreate(&start);
                hipEventCreate(&stop);

                hipEventRecord(start, 0);
                for(int iter = 0; iter < num_iterations; ++iter)
                {
                    status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                             &beta, d_c, matC, d_d, matD,
                                             &heuristicResults[algo_idx].algo, d_workspace,
                                             heuristicResults[algo_idx].workspaceSize, 0);
                }
                hipEventRecord(stop, 0);
                hipEventSynchronize(stop);

                float elapsed_ms = 0.0f;
                hipEventElapsedTime(&elapsed_ms, start, stop);
                float avg_time_ms = elapsed_ms / num_iterations;

                hipblaslt_cout << "  GPU " << deviceId << " Algo " << algo_idx
                               << ": avg time = " << avg_time_ms << " ms" << std::endl;

                hipEventDestroy(start);
                hipEventDestroy(stop);
                if(d_workspace)
                    hipFree(d_workspace);
            }

            // Cleanup
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
            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Algorithm performance comparison test passed" << std::endl;
    }

} // namespace
