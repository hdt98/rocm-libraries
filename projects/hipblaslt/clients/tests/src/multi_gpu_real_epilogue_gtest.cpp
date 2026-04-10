/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Real Multi-GPU Tests - Epilogue Operations Coverage
 * Tests all epilogue operations (RELU, GELU, BIAS, etc.) with real distribution
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <cmath>

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
    // Test 1: DEFAULT Epilogue - Data Parallel
    // ============================================================================
    TEST(MultiGPURealEpilogue, DataParallel_DEFAULT)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Epilogue DEFAULT - Data Parallel (all GPUs) ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 2;
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

                hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                               &epilogue, sizeof(epilogue));

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
            hipblaslt_cout << "GPU " << dev << ": DEFAULT epilogue completed" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "DEFAULT epilogue: " << (batches_per_gpu * numDevices) << " batches" << std::endl;
    }

    // ============================================================================
    // Test 2: RELU Epilogue - Row Partition
    // ============================================================================
    TEST(MultiGPURealEpilogue, RowPartition_RELU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Epilogue RELU - Row Partition (all GPUs) ===" << std::endl;

        const int64_t M_total = 1024, N = 512, K = 512;
        const int64_t M_per_gpu = M_total / numDevices;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t M_local = (dev == numDevices - 1) ? (M_total - dev * M_per_gpu) : M_per_gpu;

            hipMalloc(&d_A[dev], M_local * K * sizeof(float));
            hipMalloc(&d_B[dev], K * N * sizeof(float));
            hipMalloc(&d_C[dev], M_local * N * sizeof(float));

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M_local, K, M_local);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M_local, N, M_local);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M_local, N, M_local);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_RELU;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                           &epilogue, sizeof(epilogue));

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           nullptr, nullptr, 0, 0);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": RELU epilogue on " << M_local << " rows" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "RELU epilogue: Row partitioned across " << numDevices << " GPUs" << std::endl;
    }

    // ============================================================================
    // Test 3: GELU Epilogue - Column Partition
    // ============================================================================
    TEST(MultiGPURealEpilogue, ColumnPartition_GELU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Epilogue GELU - Column Partition (all GPUs) ===" << std::endl;

        const int64_t M = 512, N_total = 1024, K = 512;
        const int64_t N_per_gpu = N_total / numDevices;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t N_local = (dev == numDevices - 1) ? (N_total - dev * N_per_gpu) : N_per_gpu;

            hipMalloc(&d_A[dev], M * K * sizeof(float));
            hipMalloc(&d_B[dev], K * N_local * sizeof(float));
            hipMalloc(&d_C[dev], M * N_local * sizeof(float));

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N_local, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N_local, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N_local, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_GELU;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                           &epilogue, sizeof(epilogue));

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           nullptr, nullptr, 0, 0);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": GELU epilogue on " << N_local << " columns" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "GELU epilogue: Column partitioned across " << numDevices << " GPUs" << std::endl;
    }

    // ============================================================================
    // Test 4: BIAS Epilogue - Data Parallel
    // ============================================================================
    TEST(MultiGPURealEpilogue, DataParallel_BIAS)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Epilogue BIAS - Data Parallel (all GPUs) ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 2;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices), d_bias(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_A[dev], batches_per_gpu * M * K * sizeof(float));
            hipMalloc(&d_B[dev], batches_per_gpu * K * N * sizeof(float));
            hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(float));
            hipMalloc(&d_bias[dev], M * sizeof(float)); // Bias vector

            std::vector<float> h_bias(M, 0.5f);
            hipMemcpy(d_bias[dev], h_bias.data(), M * sizeof(float), hipMemcpyHostToDevice);

            for(int64_t b = 0; b < batches_per_gpu; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                               &epilogue, sizeof(epilogue));

                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                               &d_bias[dev], sizeof(d_bias[dev]));

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
            hipblaslt_cout << "GPU " << dev << ": BIAS epilogue completed" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]); hipFree(d_bias[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "BIAS epilogue: " << (batches_per_gpu * numDevices) << " batches" << std::endl;
    }

    // ============================================================================
    // Test 5: BIAS + RELU Combined Epilogue
    // ============================================================================
    TEST(MultiGPURealEpilogue, DataParallel_BIAS_RELU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Epilogue BIAS+RELU - Data Parallel (all GPUs) ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 2;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices), d_bias(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_A[dev], batches_per_gpu * M * K * sizeof(float));
            hipMalloc(&d_B[dev], batches_per_gpu * K * N * sizeof(float));
            hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(float));
            hipMalloc(&d_bias[dev], M * sizeof(float));

            std::vector<float> h_bias(M, 0.1f);
            hipMemcpy(d_bias[dev], h_bias.data(), M * sizeof(float), hipMemcpyHostToDevice);

            for(int64_t b = 0; b < batches_per_gpu; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_RELU_BIAS;
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                               &epilogue, sizeof(epilogue));

                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                               &d_bias[dev], sizeof(d_bias[dev]));

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
            hipblaslt_cout << "GPU " << dev << ": BIAS+RELU epilogue completed" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]); hipFree(d_bias[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "BIAS+RELU epilogue: " << (batches_per_gpu * numDevices) << " batches" << std::endl;
    }

    // ============================================================================
    // Test 6: Different Epilogues Per GPU
    // ============================================================================
    TEST(MultiGPURealEpilogue, DifferentEpiloguesPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Different Epilogues Per GPU ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtEpilogue_t> epilogues = {
            HIPBLASLT_EPILOGUE_DEFAULT,
            HIPBLASLT_EPILOGUE_RELU,
            HIPBLASLT_EPILOGUE_GELU,
            HIPBLASLT_EPILOGUE_BIAS,
            HIPBLASLT_EPILOGUE_RELU_BIAS,
            HIPBLASLT_EPILOGUE_GELU_BIAS,
            HIPBLASLT_EPILOGUE_DGELU,
            HIPBLASLT_EPILOGUE_GELU_AUX
        };

        std::vector<std::string> epilogue_names = {
            "DEFAULT", "RELU", "GELU", "BIAS", "RELU_BIAS", "GELU_BIAS", "DGELU", "GELU_AUX"
        };

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            int epilogue_idx = dev % epilogues.size();
            hipblasLtEpilogue_t epilogue = epilogues[epilogue_idx];

            hipblaslt_cout << "GPU " << dev << ": Using " << epilogue_names[epilogue_idx]
                          << " epilogue" << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            float *d_A, *d_B, *d_C, *d_bias = nullptr;
            hipMalloc(&d_A, M * K * sizeof(float));
            hipMalloc(&d_B, K * N * sizeof(float));
            hipMalloc(&d_C, M * N * sizeof(float));

            // Allocate bias if needed
            if(epilogue == HIPBLASLT_EPILOGUE_BIAS ||
               epilogue == HIPBLASLT_EPILOGUE_RELU_BIAS ||
               epilogue == HIPBLASLT_EPILOGUE_GELU_BIAS)
            {
                hipMalloc(&d_bias, M * sizeof(float));
            }

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                           &epilogue, sizeof(epilogue));

            if(d_bias != nullptr)
            {
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                               &d_bias, sizeof(d_bias));
            }

            hipblasLtMatmul(handle, matmul, &alpha,
                           d_A, matA, d_B, matB,
                           &beta, d_C, matC, d_C, matD,
                           nullptr, nullptr, 0, 0);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipFree(d_A); hipFree(d_B); hipFree(d_C);
            if(d_bias) hipFree(d_bias);
            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Tested different epilogues across " << numDevices << " GPUs" << std::endl;
    }

} // namespace
