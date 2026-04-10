/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <thread>
#include <future>

#include <gtest/gtest-spi.h>

namespace
{
    // Helper function to get number of available GPUs
    int getNumGPUs()
    {
        int numDevices = 0;
        hipGetDeviceCount(&numDevices);
        return numDevices;
    }

    // ----------------------------------------------------------------------------
    // Test 1: Data Parallelism - Same GEMM on each GPU independently
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTest, DataParallelism_IndependentGEMM)
    {
        int numDevices = getNumGPUs();

        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs, only " << numDevices << " available";
        }

        hipblaslt_cout << "Testing data parallelism on " << numDevices << " GPUs" << std::endl;

        const int64_t M = 128;
        const int64_t N = 128;
        const int64_t K = 128;

        float alpha = 1.0f;
        float beta = 0.0f;

        auto a_type = HIP_R_32F;
        auto b_type = HIP_R_32F;
        auto c_type = HIP_R_32F;
        auto d_type = HIP_R_32F;
        auto compute_type = HIPBLAS_COMPUTE_32F;

        // Run same GEMM on all GPUs
        for (int deviceId = 0; deviceId < numDevices; ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Create matrix descriptors
            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, a_type, M, K, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, b_type, K, N, K);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matC, c_type, M, N, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, d_type, M, N, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatmulDesc_t matmul;
            status = hipblasLtMatmulDescCreate(&matmul, compute_type, HIP_R_32F);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasOperation_t opA = HIPBLAS_OP_N;
            hipblasOperation_t opB = HIPBLAS_OP_N;
            status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            float *d_a, *d_b, *d_c, *d_d;
            size_t size_a = M * K * sizeof(float);
            size_t size_b = K * N * sizeof(float);
            size_t size_c = M * N * sizeof(float);
            size_t size_d = M * N * sizeof(float);

            hipErr = hipMalloc(&d_a, size_a);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_b, size_b);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_c, size_c);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_d, size_d);
            ASSERT_EQ(hipErr, hipSuccess);

            std::vector<float> h_a(M * K, 1.0f);
            std::vector<float> h_b(K * N, 2.0f);
            std::vector<float> h_c(M * N, 0.0f);

            hipErr = hipMemcpy(d_a, h_a.data(), size_a, hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMemcpy(d_b, h_b.data(), size_b, hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMemcpy(d_c, h_c.data(), size_c, hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtMatmulPreference_t pref;
            status = hipblasLtMatmulPreferenceCreate(&pref);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            size_t max_workspace_size = 32 * 1024 * 1024;
            status = hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size, sizeof(max_workspace_size));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatmulHeuristicResult_t heuristicResult[1];
            int returnedAlgoCount = 0;
            status = hipblasLtMatmulAlgoGetHeuristic(
                handle, matmul, matA, matB, matC, matD, pref, 1, heuristicResult, &returnedAlgoCount);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            ASSERT_GT(returnedAlgoCount, 0);

            void* d_workspace = nullptr;
            size_t workspace_size = heuristicResult[0].workspaceSize;
            if(workspace_size > 0)
            {
                hipErr = hipMalloc(&d_workspace, workspace_size);
                ASSERT_EQ(hipErr, hipSuccess);
            }

            status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                     &beta, d_c, matC, d_d, matD,
                                     &heuristicResult[0].algo, d_workspace, workspace_size, 0);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipErr = hipDeviceSynchronize();
            ASSERT_EQ(hipErr, hipSuccess);

            std::vector<float> h_d(M * N);
            hipErr = hipMemcpy(h_d.data(), d_d, size_d, hipMemcpyDeviceToHost);
            ASSERT_EQ(hipErr, hipSuccess);

            float expected = static_cast<float>(K) * 1.0f * 2.0f;
            EXPECT_NEAR(h_d[0], expected, 0.01f) << "GPU " << deviceId << " result mismatch";

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

        hipblaslt_cout << "Data parallelism test passed on " << numDevices << " GPUs" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 2: Row-Partitioned GEMM - Split matrix rows across GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTest, ModelParallelism_RowPartitioned)
    {
        int numDevices = getNumGPUs();

        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs, only " << numDevices << " available";
        }

        hipblaslt_cout << "Testing row-partitioned GEMM across " << numDevices << " GPUs" << std::endl;

        // Total problem: D = A * B where A is M×K, B is K×N, D is M×N
        // Split A and D by rows across GPUs
        const int64_t M_total = 256;  // Total rows
        const int64_t N = 128;
        const int64_t K = 128;

        // Each GPU gets M_total/numDevices rows
        const int64_t M_per_gpu = M_total / numDevices;

        float alpha = 1.0f;
        float beta = 0.0f;

        // Initialize full matrices on host
        std::vector<float> h_a_full(M_total * K, 1.0f);
        std::vector<float> h_b_full(K * N, 2.0f);  // Same B on all GPUs
        std::vector<float> h_d_parts(numDevices * M_per_gpu * N);

        struct GPUData {
            float *d_a, *d_b, *d_d;
            hipblasLtHandle_t handle;
            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatmulDesc_t matmul;
            void* d_workspace;
        };

        std::vector<GPUData> gpu_data(numDevices);

        // Setup and execute on each GPU
        for (int deviceId = 0; deviceId < numDevices; ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            auto& data = gpu_data[deviceId];

            // Create handle
            auto status = hipblasLtCreate(&data.handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Create matrix layouts for this GPU's partition
            status = hipblasLtMatrixLayoutCreate(&data.matA, HIP_R_32F, M_per_gpu, K, M_per_gpu);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&data.matB, HIP_R_32F, K, N, K);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&data.matD, HIP_R_32F, M_per_gpu, N, M_per_gpu);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            status = hipblasLtMatmulDescCreate(&data.matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasOperation_t opA = HIPBLAS_OP_N;
            hipblasOperation_t opB = HIPBLAS_OP_N;
            status = hipblasLtMatmulDescSetAttribute(data.matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatmulDescSetAttribute(data.matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Allocate device memory
            hipErr = hipMalloc(&data.d_a, M_per_gpu * K * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&data.d_b, K * N * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&data.d_d, M_per_gpu * N * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);

            // Copy this GPU's partition of A (rows deviceId*M_per_gpu to (deviceId+1)*M_per_gpu)
            std::vector<float> h_a_partition(M_per_gpu * K);
            for(int64_t i = 0; i < M_per_gpu; ++i)
            {
                for(int64_t j = 0; j < K; ++j)
                {
                    h_a_partition[i * K + j] = h_a_full[(deviceId * M_per_gpu + i) * K + j];
                }
            }

            hipErr = hipMemcpy(data.d_a, h_a_partition.data(), M_per_gpu * K * sizeof(float), hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMemcpy(data.d_b, h_b_full.data(), K * N * sizeof(float), hipMemcpyHostToDevice);
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
                data.handle, data.matmul, data.matA, data.matB, data.matB, data.matD,
                pref, 1, heuristicResult, &returnedAlgoCount);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            ASSERT_GT(returnedAlgoCount, 0);

            data.d_workspace = nullptr;
            size_t workspace_size = heuristicResult[0].workspaceSize;
            if(workspace_size > 0)
            {
                hipErr = hipMalloc(&data.d_workspace, workspace_size);
                ASSERT_EQ(hipErr, hipSuccess);
            }

            // Execute GEMM
            status = hipblasLtMatmul(data.handle, data.matmul, &alpha,
                                     data.d_a, data.matA, data.d_b, data.matB,
                                     &beta, data.d_d, data.matB, data.d_d, data.matD,
                                     &heuristicResult[0].algo, data.d_workspace, workspace_size, 0);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatmulPreferenceDestroy(pref);
        }

        // Synchronize all devices and gather results
        for (int deviceId = 0; deviceId < numDevices; ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipErr = hipDeviceSynchronize();
            ASSERT_EQ(hipErr, hipSuccess);

            // Copy result back
            std::vector<float> h_d_partition(M_per_gpu * N);
            hipErr = hipMemcpy(h_d_partition.data(), gpu_data[deviceId].d_d,
                              M_per_gpu * N * sizeof(float), hipMemcpyDeviceToHost);
            ASSERT_EQ(hipErr, hipSuccess);

            // Store in combined result
            for(int64_t i = 0; i < M_per_gpu * N; ++i)
            {
                h_d_parts[deviceId * M_per_gpu * N + i] = h_d_partition[i];
            }
        }

        // Verify results
        float expected = static_cast<float>(K) * 1.0f * 2.0f;
        for(size_t i = 0; i < h_d_parts.size(); ++i)
        {
            EXPECT_NEAR(h_d_parts[i], expected, 0.01f) << "Mismatch at index " << i;
            if(h_d_parts[i] != expected)
                break; // Only report first error
        }

        // Cleanup
        for (int deviceId = 0; deviceId < numDevices; ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            auto& data = gpu_data[deviceId];

            if(data.d_workspace)
            {
                hipErr = hipFree(data.d_workspace);
                EXPECT_EQ(hipErr, hipSuccess);
            }
            hipErr = hipFree(data.d_a);
            EXPECT_EQ(hipErr, hipSuccess);
            hipErr = hipFree(data.d_b);
            EXPECT_EQ(hipErr, hipSuccess);
            hipErr = hipFree(data.d_d);
            EXPECT_EQ(hipErr, hipSuccess);

            auto status = hipblasLtMatrixLayoutDestroy(data.matA);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutDestroy(data.matB);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutDestroy(data.matD);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatmulDescDestroy(data.matmul);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtDestroy(data.handle);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        }

        hipblaslt_cout << "Row-partitioned GEMM test passed across " << numDevices << " GPUs" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 3: Batched GEMM across GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTest, BatchedGEMM_AcrossGPUs)
    {
        int numDevices = getNumGPUs();

        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs, only " << numDevices << " available";
        }

        hipblaslt_cout << "Testing batched GEMM across " << numDevices << " GPUs" << std::endl;

        // Each GPU handles different batches
        const int64_t M = 64;
        const int64_t N = 64;
        const int64_t K = 64;
        const int64_t batches_per_gpu = 4;
        const int64_t total_batches = numDevices * batches_per_gpu;

        float alpha = 1.0f;
        float beta = 0.0f;

        for (int deviceId = 0; deviceId < numDevices; ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Create matrix layouts with batch
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
            status = hipblasLtMatrixLayoutSetAttribute(
                matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches_per_gpu, sizeof(batches_per_gpu));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(
                matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches_per_gpu, sizeof(batches_per_gpu));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(
                matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches_per_gpu, sizeof(batches_per_gpu));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(
                matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches_per_gpu, sizeof(batches_per_gpu));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Set strides
            int64_t strideA = M * K;
            int64_t strideB = K * N;
            int64_t strideC = M * N;
            int64_t strideD = M * N;
            status = hipblasLtMatrixLayoutSetAttribute(
                matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(
                matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(
                matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC));
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutSetAttribute(
                matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideD, sizeof(strideD));
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

            // Allocate batched data
            float *d_a, *d_b, *d_c, *d_d;
            size_t size_a = M * K * batches_per_gpu * sizeof(float);
            size_t size_b = K * N * batches_per_gpu * sizeof(float);
            size_t size_c = M * N * batches_per_gpu * sizeof(float);
            size_t size_d = M * N * batches_per_gpu * sizeof(float);

            hipErr = hipMalloc(&d_a, size_a);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_b, size_b);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_c, size_c);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_d, size_d);
            ASSERT_EQ(hipErr, hipSuccess);

            std::vector<float> h_a(M * K * batches_per_gpu, 1.0f);
            std::vector<float> h_b(K * N * batches_per_gpu, 2.0f);
            std::vector<float> h_c(M * N * batches_per_gpu, 0.0f);

            hipErr = hipMemcpy(d_a, h_a.data(), size_a, hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMemcpy(d_b, h_b.data(), size_b, hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMemcpy(d_c, h_c.data(), size_c, hipMemcpyHostToDevice);
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
            ASSERT_GT(returnedAlgoCount, 0);

            void* d_workspace = nullptr;
            size_t workspace_size = heuristicResult[0].workspaceSize;
            if(workspace_size > 0)
            {
                hipErr = hipMalloc(&d_workspace, workspace_size);
                ASSERT_EQ(hipErr, hipSuccess);
            }

            status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                     &beta, d_c, matC, d_d, matD,
                                     &heuristicResult[0].algo, d_workspace, workspace_size, 0);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipErr = hipDeviceSynchronize();
            ASSERT_EQ(hipErr, hipSuccess);

            std::vector<float> h_d(M * N * batches_per_gpu);
            hipErr = hipMemcpy(h_d.data(), d_d, size_d, hipMemcpyDeviceToHost);
            ASSERT_EQ(hipErr, hipSuccess);

            float expected = static_cast<float>(K) * 1.0f * 2.0f;
            for(size_t i = 0; i < h_d.size(); ++i)
            {
                EXPECT_NEAR(h_d[i], expected, 0.01f) << "GPU " << deviceId << " batch element " << i;
                if(h_d[i] != expected)
                    break;
            }

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

        hipblaslt_cout << "Batched GEMM test passed: " << total_batches
                       << " total batches across " << numDevices << " GPUs" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 4: Concurrent Execution on Multiple GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTest, ConcurrentExecution)
    {
        int numDevices = getNumGPUs();

        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs, only " << numDevices << " available";
        }

        hipblaslt_cout << "Testing concurrent execution on " << numDevices << " GPUs" << std::endl;

        const int64_t M = 256;
        const int64_t N = 256;
        const int64_t K = 256;

        // Lambda to run GEMM on a specific GPU
        auto runGEMMOnDevice = [&](int deviceId) -> bool {
            auto hipErr = hipSetDevice(deviceId);
            if(hipErr != hipSuccess)
                return false;

            hipStream_t stream;
            hipErr = hipStreamCreate(&stream);
            if(hipErr != hipSuccess)
                return false;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return false;

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return false;
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return false;
            status = hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return false;
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return false;

            hipblasLtMatmulDesc_t matmul;
            status = hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return false;

            hipblasOperation_t opA = HIPBLAS_OP_N;
            hipblasOperation_t opB = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

            float *d_a, *d_b, *d_c, *d_d;
            hipMalloc(&d_a, M * K * sizeof(float));
            hipMalloc(&d_b, K * N * sizeof(float));
            hipMalloc(&d_c, M * N * sizeof(float));
            hipMalloc(&d_d, M * N * sizeof(float));

            std::vector<float> h_a(M * K, 1.0f);
            std::vector<float> h_b(K * N, 2.0f);
            std::vector<float> h_c(M * N, 0.0f);

            hipMemcpyAsync(d_a, h_a.data(), M * K * sizeof(float), hipMemcpyHostToDevice, stream);
            hipMemcpyAsync(d_b, h_b.data(), K * N * sizeof(float), hipMemcpyHostToDevice, stream);
            hipMemcpyAsync(d_c, h_c.data(), M * N * sizeof(float), hipMemcpyHostToDevice, stream);

            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);
            size_t max_workspace_size = 32 * 1024 * 1024;
            hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size, sizeof(max_workspace_size));

            hipblasLtMatmulHeuristicResult_t heuristicResult[1];
            int returnedAlgoCount = 0;
            hipblasLtMatmulAlgoGetHeuristic(
                handle, matmul, matA, matB, matC, matD, pref, 1, heuristicResult, &returnedAlgoCount);

            void* d_workspace = nullptr;
            if(heuristicResult[0].workspaceSize > 0)
                hipMalloc(&d_workspace, heuristicResult[0].workspaceSize);

            float alpha = 1.0f, beta = 0.0f;
            status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                     &beta, d_c, matC, d_d, matD,
                                     &heuristicResult[0].algo, d_workspace,
                                     heuristicResult[0].workspaceSize, stream);

            hipStreamSynchronize(stream);

            std::vector<float> h_d(M * N);
            hipMemcpy(h_d.data(), d_d, M * N * sizeof(float), hipMemcpyDeviceToHost);

            bool success = true;
            float expected = static_cast<float>(K) * 1.0f * 2.0f;
            if(std::abs(h_d[0] - expected) > 0.01f)
                success = false;

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
            hipblasLtDestroy(handle);
            hipStreamDestroy(stream);

            return success;
        };

        // Launch concurrent operations on all GPUs using threads
        std::vector<std::future<bool>> futures;
        for(int i = 0; i < numDevices; ++i)
        {
            futures.push_back(std::async(std::launch::async, runGEMMOnDevice, i));
        }

        // Wait for all to complete and check results
        for(int i = 0; i < numDevices; ++i)
        {
            bool result = futures[i].get();
            EXPECT_TRUE(result) << "GPU " << i << " failed in concurrent execution";
        }

        hipblaslt_cout << "Concurrent execution test passed on " << numDevices << " GPUs" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 5: Different Data Types on Different GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTest, DifferentDataTypes)
    {
        int numDevices = getNumGPUs();

        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs, only " << numDevices << " available";
        }

        hipblaslt_cout << "Testing different data types across GPUs" << std::endl;

        const int64_t M = 64;
        const int64_t N = 64;
        const int64_t K = 64;

        // GPU 0: FP32, GPU 1: FP16, GPU 2: BF16, etc.
        std::vector<hipDataType> types = {HIP_R_32F, HIP_R_16F, HIP_R_16BF};

        for(int deviceId = 0; deviceId < std::min(numDevices, 3); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipDataType dtype = types[deviceId % types.size()];

            hipblaslt_cout << "GPU " << deviceId << " using datatype: "
                           << hip_datatype_to_string(dtype) << std::endl;

            // For simplicity, just test that handle creation and basic operations work
            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA;
            status = hipblasLtMatrixLayoutCreate(&matA, dtype, M, K, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                status = hipblasLtMatrixLayoutDestroy(matA);
                EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            }

            status = hipblasLtDestroy(handle);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        }

        hipblaslt_cout << "Different data types test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 6: Multi-GPU with Different Matrix Sizes
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTest, DifferentMatrixSizes)
    {
        int numDevices = getNumGPUs();

        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs, only " << numDevices << " available";
        }

        hipblaslt_cout << "Testing different matrix sizes on different GPUs" << std::endl;

        // Each GPU gets progressively larger matrices
        for(int deviceId = 0; deviceId < numDevices; ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            int64_t size_multiplier = (deviceId + 1);
            int64_t M = 32 * size_multiplier;
            int64_t N = 32 * size_multiplier;
            int64_t K = 32 * size_multiplier;

            hipblaslt_cout << "GPU " << deviceId << " running " << M << "x" << N << "x" << K << " GEMM" << std::endl;

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

            float *d_a, *d_b, *d_d;
            hipMalloc(&d_a, M * K * sizeof(float));
            hipMalloc(&d_b, K * N * sizeof(float));
            hipMalloc(&d_d, M * N * sizeof(float));

            std::vector<float> h_a(M * K, 1.0f);
            std::vector<float> h_b(K * N, 2.0f);

            hipMemcpy(d_a, h_a.data(), M * K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_b, h_b.data(), K * N * sizeof(float), hipMemcpyHostToDevice);

            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);
            size_t max_workspace_size = 32 * 1024 * 1024;
            hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size, sizeof(max_workspace_size));

            hipblasLtMatmulHeuristicResult_t heuristicResult[1];
            int returnedAlgoCount = 0;
            status = hipblasLtMatmulAlgoGetHeuristic(
                handle, matmul, matA, matB, matB, matD, pref, 1, heuristicResult, &returnedAlgoCount);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            ASSERT_GT(returnedAlgoCount, 0);

            void* d_workspace = nullptr;
            if(heuristicResult[0].workspaceSize > 0)
                hipMalloc(&d_workspace, heuristicResult[0].workspaceSize);

            float alpha = 1.0f, beta = 0.0f;
            status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                     &beta, d_d, matB, d_d, matD,
                                     &heuristicResult[0].algo, d_workspace, heuristicResult[0].workspaceSize, 0);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipDeviceSynchronize();

            std::vector<float> h_d(M * N);
            hipMemcpy(h_d.data(), d_d, M * N * sizeof(float), hipMemcpyDeviceToHost);

            float expected = static_cast<float>(K) * 1.0f * 2.0f;
            EXPECT_NEAR(h_d[0], expected, 0.01f) << "GPU " << deviceId;

            if(d_workspace)
                hipFree(d_workspace);
            hipFree(d_a);
            hipFree(d_b);
            hipFree(d_d);
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Different matrix sizes test passed" << std::endl;
    }

} // namespace
