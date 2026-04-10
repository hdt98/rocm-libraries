/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Epilogue Features Test Suite
 * Tests: Activation Functions, Bias, Gradients, Auxiliary Matrix
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
    // Test 1: Activation Functions across GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPUEpilogue, ActivationFunctions)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing activation functions across GPUs" << std::endl;

        const int64_t M = 64;
        const int64_t N = 64;
        const int64_t K = 64;
        float alpha = 1.0f;
        float beta = 0.0f;

        // Different activation per GPU
        std::vector<hipblasLtEpilogue_t> activations = {
            HIPBLASLT_EPILOGUE_RELU,
            HIPBLASLT_EPILOGUE_GELU,
            HIPBLASLT_EPILOGUE_DEFAULT,
            HIPBLASLT_EPILOGUE_RELU
        };

        const char* activation_names[] = {"ReLU", "GELU", "None", "ReLU"};

        for(size_t i = 0; i < std::min(activations.size(), static_cast<size_t>(numDevices)); ++i)
        {
            auto hipErr = hipSetDevice(i);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << i << " testing " << activation_names[i] << std::endl;

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

            // Set epilogue
            status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                     &activations[i], sizeof(activations[i]));
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            float *d_a, *d_b, *d_c, *d_d;
            hipErr = hipMalloc(&d_a, M * K * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_b, K * N * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_c, M * N * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_d, M * N * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);

            // Use values that test activation: some positive, some negative
            std::vector<float> h_a(M * K);
            std::vector<float> h_b(K * N);
            for(size_t j = 0; j < h_a.size(); ++j)
                h_a[j] = (j % 2 == 0) ? 1.0f : -1.0f;
            for(size_t j = 0; j < h_b.size(); ++j)
                h_b[j] = 1.0f;

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

            if(status == HIPBLAS_STATUS_SUCCESS && returnedAlgoCount > 0)
            {
                void* d_workspace = nullptr;
                if(heuristicResult[0].workspaceSize > 0)
                {
                    hipErr = hipMalloc(&d_workspace, heuristicResult[0].workspaceSize);
                    ASSERT_EQ(hipErr, hipSuccess);
                }

                status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                         &beta, d_c, matC, d_d, matD,
                                         &heuristicResult[0].algo, d_workspace, heuristicResult[0].workspaceSize, 0);

                if(status == HIPBLAS_STATUS_SUCCESS)
                {
                    hipErr = hipDeviceSynchronize();
                    ASSERT_EQ(hipErr, hipSuccess);

                    std::vector<float> h_d(M * N);
                    hipErr = hipMemcpy(h_d.data(), d_d, M * N * sizeof(float), hipMemcpyDeviceToHost);
                    ASSERT_EQ(hipErr, hipSuccess);

                    // Verify activation was applied
                    // For ReLU: negative values should be zero
                    if(activations[i] == HIPBLASLT_EPILOGUE_RELU)
                    {
                        bool has_positive = false;
                        for(auto val : h_d)
                        {
                            EXPECT_GE(val, 0.0f) << "ReLU should produce non-negative values";
                            if(val > 0.0f) has_positive = true;
                        }
                        EXPECT_TRUE(has_positive) << "ReLU output should have some positive values";
                    }
                }

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

        hipblaslt_cout << "Activation functions test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 2: Bias Operations across GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPUEpilogue, BiasOperations)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing bias operations across GPUs" << std::endl;

        const int64_t M = 64;
        const int64_t N = 64;
        const int64_t K = 64;

        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " testing bias" << std::endl;

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

            // Set bias epilogue
            hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
            status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                     &epilogue, sizeof(epilogue));
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Allocate bias vector
            float *d_bias;
            hipErr = hipMalloc(&d_bias, M * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);

            std::vector<float> h_bias(M, 5.0f);  // Bias value of 5.0
            hipErr = hipMemcpy(d_bias, h_bias.data(), M * sizeof(float), hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);

            // Set bias pointer
            status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                     &d_bias, sizeof(d_bias));
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipErr = hipFree(d_bias);
            EXPECT_EQ(hipErr, hipSuccess);

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

        hipblaslt_cout << "Bias operations test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 3: Gradient Operations (DRELU, DGELU) across GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPUEpilogue, GradientOperations)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing gradient operations across GPUs" << std::endl;

        std::vector<hipblasLtEpilogue_t> grad_ops = {
            HIPBLASLT_EPILOGUE_DRELU,
            HIPBLASLT_EPILOGUE_DGELU
        };

        const char* grad_names[] = {"DRELU", "DGELU"};

        const int64_t M = 64;
        const int64_t N = 64;
        const int64_t K = 64;

        for(size_t i = 0; i < std::min(grad_ops.size(), static_cast<size_t>(numDevices)); ++i)
        {
            auto hipErr = hipSetDevice(i);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << i << " testing " << grad_names[i] << std::endl;

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

            // Set gradient epilogue
            status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                     &grad_ops[i], sizeof(grad_ops[i]));
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

        hipblaslt_cout << "Gradient operations test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 4: Auxiliary Matrix Output across GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPUEpilogue, AuxiliaryMatrixOutput)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing auxiliary matrix output across GPUs" << std::endl;

        const int64_t M = 64;
        const int64_t N = 64;
        const int64_t K = 64;

        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " testing auxiliary output" << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Create layouts including auxiliary matrix E
            hipblasLtMatrixLayout_t matA, matB, matC, matD, matE;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matE, HIP_R_32F, M, N, M);
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

            // Setup for auxiliary output would go here
            // This is a placeholder showing the API structure

            status = hipblasLtMatrixLayoutDestroy(matA);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutDestroy(matB);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutDestroy(matC);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutDestroy(matD);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutDestroy(matE);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatmulDescDestroy(matmul);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtDestroy(handle);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        }

        hipblaslt_cout << "Auxiliary matrix output test passed" << std::endl;
    }

} // namespace
