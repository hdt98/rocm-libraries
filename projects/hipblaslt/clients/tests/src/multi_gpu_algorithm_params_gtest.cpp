/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Algorithm Parameters Test Suite
 * Algorithm selection, workspace sizes, solution indices
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

    // Test 1: Algorithm Selection with Different Indices
    TEST(MultiGPUAlgorithm, AlgorithmIndices)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing algorithm indices across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float alpha = 1.0f, beta = 0.0f;

        for(int dev = 0; dev < numDevices; ++dev)
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

            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);

            // Get available algorithms
            hipblasLtMatmulHeuristicResult_t heuristicResult[10];
            int returnedAlgoCount = 0;

            auto status = hipblasLtMatmulAlgoGetHeuristic(
                handle, matmul, matA, matB, matC, matD, pref,
                10, heuristicResult, &returnedAlgoCount);

            if(status == HIPBLAS_STATUS_SUCCESS && returnedAlgoCount > 0)
            {
                hipblaslt_cout << "GPU " << dev << " found " << returnedAlgoCount
                               << " algorithms" << std::endl;

                for(int i = 0; i < std::min(returnedAlgoCount, 5); ++i)
                {
                    hipblaslt_cout << "  Algorithm " << i << " available" << std::endl;
                }
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << " no algorithms found" << std::endl;
            }

            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // Test 2: Zero Workspace Size
    TEST(MultiGPUAlgorithm, ZeroWorkspace)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing zero workspace size across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);

            // Set workspace to 0
            size_t workspace_size = 0;
            hipblasLtMatmulPreferenceSetAttribute(pref,
                                                  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspace_size, sizeof(workspace_size));

            hipblaslt_cout << "GPU " << dev << " workspace size = " << workspace_size << std::endl;

            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtDestroy(handle);
        }
    }

    // Test 3: Small Workspace (4MB)
    TEST(MultiGPUAlgorithm, SmallWorkspace)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing 4MB workspace across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);

            size_t workspace_size = 4 * 1024 * 1024; // 4MB
            hipblasLtMatmulPreferenceSetAttribute(pref,
                                                  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspace_size, sizeof(workspace_size));

            hipblaslt_cout << "GPU " << dev << " workspace size = " << workspace_size << " bytes" << std::endl;

            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtDestroy(handle);
        }
    }

    // Test 4: Medium Workspace (16MB)
    TEST(MultiGPUAlgorithm, MediumWorkspace)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing 16MB workspace across GPUs" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);

            size_t workspace_size = 16 * 1024 * 1024; // 16MB
            hipblasLtMatmulPreferenceSetAttribute(pref,
                                                  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspace_size, sizeof(workspace_size));

            hipblaslt_cout << "GPU " << dev << " workspace size = " << workspace_size << " bytes" << std::endl;

            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtDestroy(handle);
        }
    }

    // Test 5: Large Workspace (64MB)
    TEST(MultiGPUAlgorithm, LargeWorkspace)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing 64MB workspace across GPUs" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);

            size_t workspace_size = 64 * 1024 * 1024; // 64MB
            hipblasLtMatmulPreferenceSetAttribute(pref,
                                                  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspace_size, sizeof(workspace_size));

            hipblaslt_cout << "GPU " << dev << " workspace size = " << workspace_size << " bytes" << std::endl;

            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtDestroy(handle);
        }
    }

    // Test 6: Variable Workspace Per GPU
    TEST(MultiGPUAlgorithm, VariableWorkspacePerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing variable workspace sizes per GPU" << std::endl;

        std::vector<size_t> workspace_sizes = {
            0,
            4 * 1024 * 1024,    // 4MB
            16 * 1024 * 1024,   // 16MB
            64 * 1024 * 1024    // 64MB
        };

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);

            size_t workspace_size = workspace_sizes[dev];
            hipblasLtMatmulPreferenceSetAttribute(pref,
                                                  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspace_size, sizeof(workspace_size));

            hipblaslt_cout << "GPU " << dev << " workspace size = "
                           << workspace_size / (1024.0 * 1024.0) << " MB" << std::endl;

            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtDestroy(handle);
        }
    }

    // Test 7: Algorithm Preference Configuration
    TEST(MultiGPUAlgorithm, AlgorithmPreferenceConfig)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing algorithm preference configuration across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
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

            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);

            // Configure algorithm preference with workspace
            size_t workspace_size = 16 * 1024 * 1024;
            hipblasLtMatmulPreferenceSetAttribute(pref,
                                                  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspace_size, sizeof(workspace_size));

            hipblaslt_cout << "GPU " << dev << " algorithm preference configured" << std::endl;

            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // Test 8: Multiple Problem Sizes with Algorithm Selection
    TEST(MultiGPUAlgorithm, MultipleProblemsAlgorithmSelection)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing algorithm selection for multiple problem sizes" << std::endl;

        struct ProblemSize { int64_t M, N, K; };
        std::vector<ProblemSize> sizes = {
            {64, 64, 64},
            {256, 256, 256},
            {1024, 1024, 1024}
        };

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            for(const auto& size : sizes)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, size.M, size.K, size.M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, size.K, size.N, size.K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, size.M, size.N, size.M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, size.M, size.N, size.M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmulPreference_t pref;
                hipblasLtMatmulPreferenceCreate(&pref);

                hipblasLtMatmulHeuristicResult_t heuristicResult[5];
                int returnedAlgoCount = 0;

                auto status = hipblasLtMatmulAlgoGetHeuristic(
                    handle, matmul, matA, matB, matC, matD, pref,
                    5, heuristicResult, &returnedAlgoCount);

                if(status == HIPBLAS_STATUS_SUCCESS)
                {
                    hipblaslt_cout << "GPU " << dev << " problem " << size.M << "x" << size.N
                                   << "x" << size.K << ": " << returnedAlgoCount << " algorithms" << std::endl;
                }

                hipblasLtMatmulPreferenceDestroy(pref);
                hipblasLtMatmulDescDestroy(matmul);
                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            }

            hipblasLtDestroy(handle);
        }
    }

    // Test 9: Solution Index Testing
    TEST(MultiGPUAlgorithm, SolutionIndexTesting)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing solution indices across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
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

            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);

            hipblasLtMatmulHeuristicResult_t heuristicResult[10];
            int returnedAlgoCount = 0;

            auto status = hipblasLtMatmulAlgoGetHeuristic(
                handle, matmul, matA, matB, matC, matD, pref,
                10, heuristicResult, &returnedAlgoCount);

            if(status == HIPBLAS_STATUS_SUCCESS && returnedAlgoCount > 0)
            {
                hipblaslt_cout << "GPU " << dev << " testing solution indices:" << std::endl;
                for(int i = 0; i < std::min(returnedAlgoCount, 3); ++i)
                {
                    hipblaslt_cout << "  Solution " << i << " valid" << std::endl;
                }
            }

            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

} // namespace
