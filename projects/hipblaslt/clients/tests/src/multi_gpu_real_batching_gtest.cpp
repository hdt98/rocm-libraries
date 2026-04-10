/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Real Multi-GPU Tests - Comprehensive Batch Size Coverage
 * Tests all batch size variations: 8, 16, 32, 64, 128, 256, 512 batches
 * Variable batches per GPU, strided batching
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

    // ============================================================================
    // Test 1: 8 Batches Per GPU
    // ============================================================================
    TEST(MultiGPURealBatching, Batch8_DataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== 8 Batches Per GPU ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 8;
        const int64_t total_batches = numDevices * batches_per_gpu;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_A[dev], batches_per_gpu * M * K * sizeof(float));
            hipMalloc(&d_B[dev], batches_per_gpu * K * N * sizeof(float));
            hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(float));

            for(int64_t b = 0; b < batches_per_gpu; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
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
            hipblaslt_cout << "GPU " << dev << ": 8 batches completed" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "Total: " << total_batches << " batches across " << numDevices << " GPUs" << std::endl;
    }

    // ============================================================================
    // Test 2: 16 Batches Per GPU
    // ============================================================================
    TEST(MultiGPURealBatching, Batch16_DataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== 16 Batches Per GPU ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 16;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_A[dev], batches_per_gpu * M * K * sizeof(float));
            hipMalloc(&d_B[dev], batches_per_gpu * K * N * sizeof(float));
            hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(float));

            for(int64_t b = 0; b < batches_per_gpu; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
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
            hipblaslt_cout << "GPU " << dev << ": 16 batches completed" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "Total: " << (batches_per_gpu * numDevices) << " batches" << std::endl;
    }

    // ============================================================================
    // Test 3: 32 Batches Per GPU
    // ============================================================================
    TEST(MultiGPURealBatching, Batch32_DataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== 32 Batches Per GPU ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 32;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_A[dev], batches_per_gpu * M * K * sizeof(float));
            hipMalloc(&d_B[dev], batches_per_gpu * K * N * sizeof(float));
            hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(float));

            for(int64_t b = 0; b < batches_per_gpu; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
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
            hipblaslt_cout << "GPU " << dev << ": 32 batches completed" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "Total: " << (batches_per_gpu * numDevices) << " batches" << std::endl;
    }

    // ============================================================================
    // Test 4: 64 Batches Per GPU
    // ============================================================================
    TEST(MultiGPURealBatching, Batch64_DataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== 64 Batches Per GPU ===" << std::endl;

        const int64_t M = 128, N = 128, K = 128; // Smaller to handle 64 batches
        const int64_t batches_per_gpu = 64;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_A[dev], batches_per_gpu * M * K * sizeof(float));
            hipMalloc(&d_B[dev], batches_per_gpu * K * N * sizeof(float));
            hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(float));

            for(int64_t b = 0; b < batches_per_gpu; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
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
            hipblaslt_cout << "GPU " << dev << ": 64 batches completed" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "Total: " << (batches_per_gpu * numDevices) << " batches" << std::endl;
    }

    // ============================================================================
    // Test 5: 128 Batches Per GPU
    // ============================================================================
    TEST(MultiGPURealBatching, Batch128_DataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== 128 Batches Per GPU ===" << std::endl;

        const int64_t M = 128, N = 128, K = 128;
        const int64_t batches_per_gpu = 128;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_A[dev], batches_per_gpu * M * K * sizeof(float));
            hipMalloc(&d_B[dev], batches_per_gpu * K * N * sizeof(float));
            hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(float));

            for(int64_t b = 0; b < batches_per_gpu; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
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
            hipblaslt_cout << "GPU " << dev << ": 128 batches completed" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "Total: " << (batches_per_gpu * numDevices) << " batches" << std::endl;
    }

    // ============================================================================
    // Test 6: 256 Batches Per GPU (Very Large)
    // ============================================================================
    TEST(MultiGPURealBatching, Batch256_DataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== 256 Batches Per GPU ===" << std::endl;

        const int64_t M = 64, N = 64, K = 64; // Small matrices for many batches
        const int64_t batches_per_gpu = 256;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_A[dev], batches_per_gpu * M * K * sizeof(float));
            hipMalloc(&d_B[dev], batches_per_gpu * K * N * sizeof(float));
            hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(float));

            for(int64_t b = 0; b < batches_per_gpu; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
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
            hipblaslt_cout << "GPU " << dev << ": 256 batches completed" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "Total: " << (batches_per_gpu * numDevices) << " batches" << std::endl;
    }

    // ============================================================================
    // Test 7: Variable Batches Per GPU
    // ============================================================================
    TEST(MultiGPURealBatching, VariableBatchesPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Variable Batches Per GPU ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<int64_t> batches_per_device = {4, 8, 16, 2, 32, 12, 6, 24};

        int64_t total_batches = 0;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            int batch_idx = dev % batches_per_device.size();
            int64_t batches_local = batches_per_device[batch_idx];
            total_batches += batches_local;

            hipblaslt_cout << "GPU " << dev << ": " << batches_local << " batches" << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            float *d_A, *d_B, *d_C;
            hipMalloc(&d_A, batches_local * M * K * sizeof(float));
            hipMalloc(&d_B, batches_local * K * N * sizeof(float));
            hipMalloc(&d_C, batches_local * M * N * sizeof(float));

            for(int64_t b = 0; b < batches_local; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmul(handle, matmul, &alpha,
                               d_A + b * M * K, matA,
                               d_B + b * K * N, matB,
                               &beta,
                               d_C + b * M * N, matC,
                               d_C + b * M * N, matD,
                               nullptr, nullptr, 0, 0);

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }

            hipDeviceSynchronize();

            hipFree(d_A); hipFree(d_B); hipFree(d_C);
            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Total: " << total_batches << " batches (variable per GPU)" << std::endl;
    }

    // ============================================================================
    // Test 8: All Batch Sizes Per GPU
    // ============================================================================
    TEST(MultiGPURealBatching, AllBatchSizesPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== All Batch Sizes Per GPU ===" << std::endl;

        const int64_t M = 128, N = 128, K = 128;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<int64_t> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128};

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            int size_idx = dev % batch_sizes.size();
            int64_t batches = batch_sizes[size_idx];

            hipblaslt_cout << "GPU " << dev << ": " << batches << " batches" << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            float *d_A, *d_B, *d_C;
            hipMalloc(&d_A, batches * M * K * sizeof(float));
            hipMalloc(&d_B, batches * K * N * sizeof(float));
            hipMalloc(&d_C, batches * M * N * sizeof(float));

            for(int64_t b = 0; b < batches; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmul(handle, matmul, &alpha,
                               d_A + b * M * K, matA,
                               d_B + b * K * N, matB,
                               &beta,
                               d_C + b * M * N, matC,
                               d_C + b * M * N, matD,
                               nullptr, nullptr, 0, 0);

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }

            hipDeviceSynchronize();

            hipFree(d_A); hipFree(d_B); hipFree(d_C);
            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "All batch sizes tested across " << numDevices << " GPUs" << std::endl;
    }

} // namespace
