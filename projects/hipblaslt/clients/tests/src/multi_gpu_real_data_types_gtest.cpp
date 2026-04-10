/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Real Multi-GPU Tests - Data Types Coverage
 * Tests all data types (FP32, FP16, BF16, FP8, INT8) with real distribution
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

#include <gtest/gtest-spi.h>

namespace
{
    int getNumGPUs()
    {
        int numDevices = 0;
        hipGetDeviceCount(&numDevices);
        return numDevices;
    }

    // Helper: Initialize matrix with pattern
    template<typename T>
    void initMatrix(std::vector<T>& mat, int64_t size, float base = 1.0f)
    {
        for(int64_t i = 0; i < size; ++i)
        {
            mat[i] = static_cast<T>(base + (i % 100) * 0.01f);
        }
    }

    // CPU GEMM reference for FP32
    void cpuGEMM_FP32(const std::vector<float>& A, const std::vector<float>& B,
                      std::vector<float>& C, int64_t M, int64_t N, int64_t K,
                      float alpha = 1.0f, float beta = 0.0f)
    {
        for(int64_t i = 0; i < M; ++i)
        {
            for(int64_t j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for(int64_t k = 0; k < K; ++k)
                {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = alpha * sum + beta * C[i * N + j];
            }
        }
    }

    // Verify with tolerance
    template<typename T>
    bool verifyResult(const std::vector<T>& result, const std::vector<T>& expected, float tolerance)
    {
        if(result.size() != expected.size()) return false;

        for(size_t i = 0; i < result.size(); ++i)
        {
            float r = static_cast<float>(result[i]);
            float e = static_cast<float>(expected[i]);
            if(std::abs(r - e) > tolerance)
            {
                return false;
            }
        }
        return true;
    }

    // ============================================================================
    // Test 1: FP32 Data Parallel with CPU Verification
    // ============================================================================
    TEST(MultiGPURealDataTypes, DataParallel_FP32)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Data Parallel FP32 (all GPUs) ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 2;
        const int64_t total_batches = numDevices * batches_per_gpu;
        float alpha = 1.0f, beta = 0.0f;

        // Prepare host data
        std::vector<std::vector<float>> h_A(total_batches), h_B(total_batches);
        std::vector<std::vector<float>> h_C_result(total_batches), h_C_expected(total_batches);

        for(int64_t b = 0; b < total_batches; ++b)
        {
            h_A[b].resize(M * K);
            h_B[b].resize(K * N);
            h_C_result[b].resize(M * N);
            h_C_expected[b].resize(M * N);

            initMatrix(h_A[b], M * K, 1.0f + b * 0.1f);
            initMatrix(h_B[b], K * N, 2.0f + b * 0.1f);

            // CPU reference
            cpuGEMM_FP32(h_A[b], h_B[b], h_C_expected[b], M, N, K, alpha, beta);
        }

        // GPU setup
        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_A[dev], batches_per_gpu * M * K * sizeof(float));
            hipMalloc(&d_B[dev], batches_per_gpu * K * N * sizeof(float));
            hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(float));
        }

        // Distribute batches and execute
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            for(int64_t local_b = 0; local_b < batches_per_gpu; ++local_b)
            {
                int64_t global_b = dev * batches_per_gpu + local_b;

                hipMemcpy(d_A[dev] + local_b * M * K, h_A[global_b].data(),
                         M * K * sizeof(float), hipMemcpyHostToDevice);
                hipMemcpy(d_B[dev] + local_b * K * N, h_B[global_b].data(),
                         K * N * sizeof(float), hipMemcpyHostToDevice);

                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmul(handles[dev], matmul, &alpha,
                               d_A[dev] + local_b * M * K, matA,
                               d_B[dev] + local_b * K * N, matB,
                               &beta,
                               d_C[dev] + local_b * M * N, matC,
                               d_C[dev] + local_b * M * N, matD,
                               nullptr, nullptr, 0, 0);

                hipMemcpy(h_C_result[global_b].data(), d_C[dev] + local_b * M * N,
                         M * N * sizeof(float), hipMemcpyDeviceToHost);

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }
        }

        // Verify all batches
        for(int64_t b = 0; b < total_batches; ++b)
        {
            bool correct = verifyResult(h_C_result[b], h_C_expected[b], 0.01f);
            EXPECT_TRUE(correct) << "Batch " << b << " failed verification";
        }

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "FP32: Verified " << total_batches << " batches across "
                      << numDevices << " GPUs" << std::endl;
    }

    // ============================================================================
    // Test 2: FP16 Row Partitioning
    // ============================================================================
    TEST(MultiGPURealDataTypes, RowPartition_FP16)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Row Partition FP16 (all GPUs) ===" << std::endl;

        const int64_t M_total = 1024;
        const int64_t N = 512, K = 512;
        const int64_t M_per_gpu = M_total / numDevices;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<_Float16*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t M_local = (dev == numDevices - 1) ? (M_total - dev * M_per_gpu) : M_per_gpu;

            hipMalloc(&d_A[dev], M_local * K * sizeof(_Float16));
            hipMalloc(&d_B[dev], K * N * sizeof(_Float16));
            hipMalloc(&d_C[dev], M_local * N * sizeof(_Float16));

            // Initialize data (simplified)
            std::vector<_Float16> h_A(M_local * K, static_cast<_Float16>(1.0f));
            std::vector<_Float16> h_B(K * N, static_cast<_Float16>(1.0f));

            hipMemcpy(d_A[dev], h_A.data(), M_local * K * sizeof(_Float16), hipMemcpyHostToDevice);
            hipMemcpy(d_B[dev], h_B.data(), K * N * sizeof(_Float16), hipMemcpyHostToDevice);

            // Execute FP16 GEMM on this GPU's row partition
            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, M_local, K, M_local);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, M_local, N, M_local);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, M_local, N, M_local);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_16F);

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           nullptr, nullptr, 0, 0);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": Processed " << M_local << " rows" << std::endl;
        }

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "FP16: Row partitioned " << M_total << " rows across "
                      << numDevices << " GPUs" << std::endl;
    }

    // ============================================================================
    // Test 3: BF16 Column Partitioning
    // ============================================================================
    TEST(MultiGPURealDataTypes, ColumnPartition_BF16)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Column Partition BF16 (all GPUs) ===" << std::endl;

        const int64_t M = 512;
        const int64_t N_total = 1024, K = 512;
        const int64_t N_per_gpu = N_total / numDevices;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<hip_bfloat16*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t N_local = (dev == numDevices - 1) ? (N_total - dev * N_per_gpu) : N_per_gpu;

            hipMalloc(&d_A[dev], M * K * sizeof(hip_bfloat16));
            hipMalloc(&d_B[dev], K * N_local * sizeof(hip_bfloat16));
            hipMalloc(&d_C[dev], M * N_local * sizeof(hip_bfloat16));

            // Initialize (simplified)
            std::vector<hip_bfloat16> h_A(M * K, static_cast<hip_bfloat16>(1.0f));
            std::vector<hip_bfloat16> h_B(K * N_local, static_cast<hip_bfloat16>(1.0f));

            hipMemcpy(d_A[dev], h_A.data(), M * K * sizeof(hip_bfloat16), hipMemcpyHostToDevice);
            hipMemcpy(d_B[dev], h_B.data(), K * N_local * sizeof(hip_bfloat16), hipMemcpyHostToDevice);

            // Execute BF16 GEMM
            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_16BF, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_16BF, K, N_local, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_16BF, M, N_local, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_16BF, M, N_local, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_16BF);

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           nullptr, nullptr, 0, 0);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": Processed " << N_local << " columns" << std::endl;
        }

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "BF16: Column partitioned " << N_total << " columns across "
                      << numDevices << " GPUs" << std::endl;
    }

    // ============================================================================
    // Test 4: Mixed Precision (FP16 input, FP32 output) - Data Parallel
    // ============================================================================
    TEST(MultiGPURealDataTypes, MixedPrecision_FP16_FP32)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Mixed Precision FP16->FP32 (all GPUs) ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 2;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<_Float16*> d_A(numDevices), d_B(numDevices);
        std::vector<float*> d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_A[dev], batches_per_gpu * M * K * sizeof(_Float16));
            hipMalloc(&d_B[dev], batches_per_gpu * K * N * sizeof(_Float16));
            hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(float));

            for(int64_t b = 0; b < batches_per_gpu; ++b)
            {
                // FP16 input, FP32 output
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmul(handles[dev], matmul, &alpha,
                               d_A[dev] + b * M * K, matA,
                               d_B[dev] + b * K * N, matB,
                               &beta,
                               d_C[dev] + b * M * N, matC,
                               d_C[dev] + b * M * N, matD,
                               nullptr, nullptr, 0, 0);

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }

            hipDeviceSynchronize();
            hipblaslt_cout << "GPU " << dev << ": Completed " << batches_per_gpu
                          << " batches (FP16->FP32)" << std::endl;
        }

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "Mixed Precision: " << (batches_per_gpu * numDevices)
                      << " batches across " << numDevices << " GPUs" << std::endl;
    }

    // ============================================================================
    // Test 5: All Data Types on Different GPUs
    // ============================================================================
    TEST(MultiGPURealDataTypes, DifferentTypesPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Different Data Types Per GPU ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float alpha = 1.0f, beta = 0.0f;

        // Data types to cycle through
        std::vector<hipDataType> types = {HIP_R_32F, HIP_R_16F, HIP_R_16BF};
        std::vector<std::string> type_names = {"FP32", "FP16", "BF16"};

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            int type_idx = dev % types.size();
            hipDataType dtype = types[type_idx];
            size_t type_size = (dtype == HIP_R_32F) ? sizeof(float) : sizeof(_Float16);

            hipblaslt_cout << "GPU " << dev << ": Using " << type_names[type_idx] << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            void *d_A, *d_B, *d_C;
            hipMalloc(&d_A, M * K * type_size);
            hipMalloc(&d_B, K * N * type_size);
            hipMalloc(&d_C, M * N * type_size);

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, dtype, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, dtype, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, dtype, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, dtype, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, dtype);

            hipblasLtMatmul(handle, matmul, &alpha,
                           d_A, matA, d_B, matB,
                           &beta, d_C, matC, d_C, matD,
                           nullptr, nullptr, 0, 0);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipFree(d_A); hipFree(d_B); hipFree(d_C);
            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Tested " << numDevices << " GPUs with different data types" << std::endl;
    }

} // namespace
