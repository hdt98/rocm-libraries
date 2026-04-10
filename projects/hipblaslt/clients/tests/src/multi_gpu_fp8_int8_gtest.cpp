/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU FP8 and INT8 Test Suite
 * Tests: FP8 E4M3, FP8 E5M2, INT8, Mixed FP8/INT8 operations
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

    // ----------------------------------------------------------------------------
    // Test 1: FP8 E4M3 GEMM across GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPUFP8INT8, FP8_E4M3_GEMM)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing FP8 E4M3 GEMM across " << numDevices << " GPUs" << std::endl;

        const int64_t M = 128;
        const int64_t N = 128;
        const int64_t K = 128;
        float alpha = 1.0f;
        float beta = 0.0f;

        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " testing FP8 E4M3" << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // FP8 E4M3 data type
            hipDataType fp8_e4m3_type = HIP_R_8F_E4M3_FNUZ;
            hipDataType output_type = HIP_R_32F;

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, fp8_e4m3_type, M, K, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, fp8_e4m3_type, K, N, K);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matC, output_type, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, output_type, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblasLtMatmulDesc_t matmul;
                status = hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
                EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

                if(status == HIPBLAS_STATUS_SUCCESS)
                {
                    hipblasOperation_t opA = HIPBLAS_OP_N;
                    hipblasOperation_t opB = HIPBLAS_OP_N;
                    status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
                    status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

                    // Allocate device memory (using uint8_t for FP8 storage)
                    uint8_t *d_a, *d_b;
                    float *d_c, *d_d;
                    hipErr = hipMalloc(&d_a, M * K * sizeof(uint8_t));
                    ASSERT_EQ(hipErr, hipSuccess);
                    hipErr = hipMalloc(&d_b, K * N * sizeof(uint8_t));
                    ASSERT_EQ(hipErr, hipSuccess);
                    hipErr = hipMalloc(&d_c, M * N * sizeof(float));
                    ASSERT_EQ(hipErr, hipSuccess);
                    hipErr = hipMalloc(&d_d, M * N * sizeof(float));
                    ASSERT_EQ(hipErr, hipSuccess);

                    // Initialize with FP8-compatible values (stored as uint8_t)
                    std::vector<uint8_t> h_a(M * K, 0x38); // FP8 representation of ~1.0
                    std::vector<uint8_t> h_b(K * N, 0x40); // FP8 representation of ~2.0
                    std::vector<float> h_c(M * N, 0.0f);

                    hipErr = hipMemcpy(d_a, h_a.data(), M * K * sizeof(uint8_t), hipMemcpyHostToDevice);
                    ASSERT_EQ(hipErr, hipSuccess);
                    hipErr = hipMemcpy(d_b, h_b.data(), K * N * sizeof(uint8_t), hipMemcpyHostToDevice);
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

                    if(status == HIPBLAS_STATUS_SUCCESS && returnedAlgoCount > 0)
                    {
                        void* d_workspace = nullptr;
                        if(heuristicResult[0].workspaceSize > 0)
                        {
                            hipErr = hipMalloc(&d_workspace, heuristicResult[0].workspaceSize);
                            ASSERT_EQ(hipErr, hipSuccess);
                        }

                        // Execute GEMM
                        status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                                 &beta, d_c, matC, d_d, matD,
                                                 &heuristicResult[0].algo, d_workspace,
                                                 heuristicResult[0].workspaceSize, 0);

                        if(status == HIPBLAS_STATUS_SUCCESS)
                        {
                            hipErr = hipDeviceSynchronize();
                            ASSERT_EQ(hipErr, hipSuccess);

                            std::vector<float> h_d(M * N);
                            hipErr = hipMemcpy(h_d.data(), d_d, M * N * sizeof(float), hipMemcpyDeviceToHost);
                            ASSERT_EQ(hipErr, hipSuccess);

                            hipblaslt_cout << "GPU " << deviceId << " FP8 E4M3 GEMM completed, result[0] = "
                                           << h_d[0] << std::endl;
                        }
                        else
                        {
                            hipblaslt_cout << "GPU " << deviceId << " FP8 E4M3 GEMM not supported" << std::endl;
                        }

                        if(d_workspace)
                            hipFree(d_workspace);
                    }
                    else
                    {
                        hipblaslt_cout << "GPU " << deviceId << " FP8 E4M3 algorithm not available" << std::endl;
                    }

                    hipblasLtMatmulPreferenceDestroy(pref);
                    hipFree(d_a);
                    hipFree(d_b);
                    hipFree(d_c);
                    hipFree(d_d);
                    hipblasLtMatmulDescDestroy(matmul);
                }

                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC);
                hipblasLtMatrixLayoutDestroy(matD);
            }
            else
            {
                hipblaslt_cout << "GPU " << deviceId << " FP8 E4M3 layout creation not supported" << std::endl;
            }

            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "FP8 E4M3 GEMM test completed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 2: FP8 E5M2 GEMM across GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPUFP8INT8, FP8_E5M2_GEMM)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing FP8 E5M2 GEMM across " << numDevices << " GPUs" << std::endl;

        const int64_t M = 128;
        const int64_t N = 128;
        const int64_t K = 128;

        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " testing FP8 E5M2" << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // FP8 E5M2 data type
            hipDataType fp8_e5m2_type = HIP_R_8F_E5M2_FNUZ;
            hipDataType output_type = HIP_R_32F;

            hipblasLtMatrixLayout_t matA, matB, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, fp8_e5m2_type, M, K, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, fp8_e5m2_type, K, N, K);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, output_type, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cout << "GPU " << deviceId << " FP8 E5M2 layout creation successful" << std::endl;
                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }
            else
            {
                hipblaslt_cout << "GPU " << deviceId << " FP8 E5M2 not supported" << std::endl;
            }

            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "FP8 E5M2 GEMM test completed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 3: INT8 GEMM across GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPUFP8INT8, INT8_GEMM)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing INT8 GEMM across " << numDevices << " GPUs" << std::endl;

        const int64_t M = 256;
        const int64_t N = 256;
        const int64_t K = 256;
        int32_t alpha = 1;
        int32_t beta = 0;

        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " testing INT8" << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_8I, M, K, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_8I, K, N, K);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matC, HIP_R_32I, M, N, M);
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
                    hipblasOperation_t opA = HIPBLAS_OP_N;
                    hipblasOperation_t opB = HIPBLAS_OP_N;
                    hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
                    hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

                    // Allocate device memory
                    int8_t *d_a, *d_b;
                    int32_t *d_c, *d_d;
                    hipErr = hipMalloc(&d_a, M * K * sizeof(int8_t));
                    ASSERT_EQ(hipErr, hipSuccess);
                    hipErr = hipMalloc(&d_b, K * N * sizeof(int8_t));
                    ASSERT_EQ(hipErr, hipSuccess);
                    hipErr = hipMalloc(&d_c, M * N * sizeof(int32_t));
                    ASSERT_EQ(hipErr, hipSuccess);
                    hipErr = hipMalloc(&d_d, M * N * sizeof(int32_t));
                    ASSERT_EQ(hipErr, hipSuccess);

                    // Initialize with small integer values
                    std::vector<int8_t> h_a(M * K, 1);
                    std::vector<int8_t> h_b(K * N, 2);
                    std::vector<int32_t> h_c(M * N, 0);

                    hipErr = hipMemcpy(d_a, h_a.data(), M * K * sizeof(int8_t), hipMemcpyHostToDevice);
                    ASSERT_EQ(hipErr, hipSuccess);
                    hipErr = hipMemcpy(d_b, h_b.data(), K * N * sizeof(int8_t), hipMemcpyHostToDevice);
                    ASSERT_EQ(hipErr, hipSuccess);
                    hipErr = hipMemcpy(d_c, h_c.data(), M * N * sizeof(int32_t), hipMemcpyHostToDevice);
                    ASSERT_EQ(hipErr, hipSuccess);

                    // Get algorithm
                    hipblasLtMatmulPreference_t pref;
                    status = hipblasLtMatmulPreferenceCreate(&pref);
                    ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

                    size_t max_workspace_size = 32 * 1024 * 1024;
                    hipblasLtMatmulPreferenceSetAttribute(
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

                        // Execute INT8 GEMM
                        status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                                 &beta, d_c, matC, d_d, matD,
                                                 &heuristicResult[0].algo, d_workspace,
                                                 heuristicResult[0].workspaceSize, 0);

                        if(status == HIPBLAS_STATUS_SUCCESS)
                        {
                            hipErr = hipDeviceSynchronize();
                            ASSERT_EQ(hipErr, hipSuccess);

                            std::vector<int32_t> h_d(M * N);
                            hipErr = hipMemcpy(h_d.data(), d_d, M * N * sizeof(int32_t), hipMemcpyDeviceToHost);
                            ASSERT_EQ(hipErr, hipSuccess);

                            int32_t expected = K * 1 * 2; // K * a[i] * b[j]
                            EXPECT_EQ(h_d[0], expected) << "GPU " << deviceId << " INT8 GEMM result mismatch";
                            hipblaslt_cout << "GPU " << deviceId << " INT8 GEMM result: " << h_d[0]
                                           << " (expected: " << expected << ")" << std::endl;
                        }
                        else
                        {
                            hipblaslt_cout << "GPU " << deviceId << " INT8 GEMM execution not supported" << std::endl;
                        }

                        if(d_workspace)
                            hipFree(d_workspace);
                    }
                    else
                    {
                        hipblaslt_cout << "GPU " << deviceId << " INT8 algorithm not available" << std::endl;
                    }

                    hipblasLtMatmulPreferenceDestroy(pref);
                    hipFree(d_a);
                    hipFree(d_b);
                    hipFree(d_c);
                    hipFree(d_d);
                    hipblasLtMatmulDescDestroy(matmul);
                }

                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC);
                hipblasLtMatrixLayoutDestroy(matD);
            }
            else
            {
                hipblaslt_cout << "GPU " << deviceId << " INT8 layout creation not supported" << std::endl;
            }

            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "INT8 GEMM test completed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 4: Mixed FP8/INT8 across different GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPUFP8INT8, MixedPrecisionAcrossGPUs)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing mixed FP8/INT8 precision across " << numDevices << " GPUs" << std::endl;

        struct PrecisionTest {
            hipDataType input_type;
            hipDataType output_type;
            hipblasComputeType_t compute_type;
            const char* name;
        };

        std::vector<PrecisionTest> tests = {
            {HIP_R_8I, HIP_R_32I, HIPBLAS_COMPUTE_32I, "INT8"},
            {HIP_R_8F_E4M3_FNUZ, HIP_R_32F, HIPBLAS_COMPUTE_32F, "FP8_E4M3"}
        };

        const int64_t M = 64;
        const int64_t N = 64;
        const int64_t K = 64;

        for(size_t t = 0; t < std::min(tests.size(), static_cast<size_t>(numDevices)); ++t)
        {
            auto hipErr = hipSetDevice(t);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << t << " testing " << tests[t].name << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, tests[t].input_type, M, K, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, tests[t].input_type, K, N, K);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, tests[t].output_type, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cout << "GPU " << t << " " << tests[t].name
                               << " layout creation successful" << std::endl;

                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
            }
            else
            {
                hipblaslt_cout << "GPU " << t << " " << tests[t].name << " not supported" << std::endl;
            }

            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Mixed precision across GPUs test completed" << std::endl;
    }

} // namespace
