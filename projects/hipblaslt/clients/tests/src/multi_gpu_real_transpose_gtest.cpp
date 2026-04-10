/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Real Multi-GPU Tests - Transpose Operations Coverage
 * Tests all transpose combinations (NN, NT, TN, TT) with real distribution
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

    // CPU GEMM with transpose support
    void cpuGEMM_Transpose(const std::vector<float>& A, const std::vector<float>& B,
                           std::vector<float>& C, int64_t M, int64_t N, int64_t K,
                           bool transA, bool transB, float alpha = 1.0f, float beta = 0.0f)
    {
        for(int64_t i = 0; i < M; ++i)
        {
            for(int64_t j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for(int64_t k = 0; k < K; ++k)
                {
                    float a_val = transA ? A[k * M + i] : A[i * K + k];
                    float b_val = transB ? B[j * K + k] : B[k * N + j];
                    sum += a_val * b_val;
                }
                C[i * N + j] = alpha * sum + beta * C[i * N + j];
            }
        }
    }

    bool verifyResult(const std::vector<float>& result, const std::vector<float>& expected, float tol = 0.01f)
    {
        if(result.size() != expected.size()) return false;
        for(size_t i = 0; i < result.size(); ++i)
        {
            if(std::abs(result[i] - expected[i]) > tol) return false;
        }
        return true;
    }

    // ============================================================================
    // Test 1: NN (No Transpose) - Data Parallel
    // ============================================================================
    TEST(MultiGPURealTranspose, DataParallel_NN)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Transpose NN - Data Parallel (all GPUs) ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 2;
        const int64_t total_batches = numDevices * batches_per_gpu;
        float alpha = 1.0f, beta = 0.0f;

        // Prepare host data with CPU verification
        std::vector<std::vector<float>> h_A(total_batches), h_B(total_batches);
        std::vector<std::vector<float>> h_C_result(total_batches), h_C_expected(total_batches);

        for(int64_t b = 0; b < total_batches; ++b)
        {
            h_A[b].resize(M * K, 1.0f + b * 0.1f);
            h_B[b].resize(K * N, 2.0f + b * 0.1f);
            h_C_result[b].resize(M * N);
            h_C_expected[b].resize(M * N);

            cpuGEMM_Transpose(h_A[b], h_B[b], h_C_expected[b], M, N, K, false, false, alpha, beta);
        }

        // GPU execution
        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_A[dev], batches_per_gpu * M * K * sizeof(float));
            hipMalloc(&d_B[dev], batches_per_gpu * K * N * sizeof(float));
            hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(float));

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

                hipblasOperation_t opA = HIPBLAS_OP_N, opB = HIPBLAS_OP_N;
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                               &opA, sizeof(opA));
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                               &opB, sizeof(opB));

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

        // Verify
        for(int64_t b = 0; b < total_batches; ++b)
        {
            EXPECT_TRUE(verifyResult(h_C_result[b], h_C_expected[b], 0.1f))
                << "Batch " << b << " verification failed";
        }

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "NN: Verified " << total_batches << " batches" << std::endl;
    }

    // ============================================================================
    // Test 2: NT (No Transpose A, Transpose B) - Row Partition
    // ============================================================================
    TEST(MultiGPURealTranspose, RowPartition_NT)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Transpose NT - Row Partition (all GPUs) ===" << std::endl;

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
            hipMalloc(&d_B[dev], N * K * sizeof(float)); // Note: B is transposed, so N×K
            hipMalloc(&d_C[dev], M_local * N * sizeof(float));

            std::vector<float> h_A(M_local * K, 1.0f);
            std::vector<float> h_B(N * K, 1.0f);

            hipMemcpy(d_A[dev], h_A.data(), M_local * K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B[dev], h_B.data(), N * K * sizeof(float), hipMemcpyHostToDevice);

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M_local, K, M_local);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, N); // Transposed layout
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M_local, N, M_local);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M_local, N, M_local);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasOperation_t opA = HIPBLAS_OP_N, opB = HIPBLAS_OP_T;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                           &opA, sizeof(opA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                           &opB, sizeof(opB));

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           nullptr, nullptr, 0, 0);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": NT operation on " << M_local << " rows" << std::endl;
        }

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "NT: Row partitioned across " << numDevices << " GPUs" << std::endl;
    }

    // ============================================================================
    // Test 3: TN (Transpose A, No Transpose B) - Column Partition
    // ============================================================================
    TEST(MultiGPURealTranspose, ColumnPartition_TN)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Transpose TN - Column Partition (all GPUs) ===" << std::endl;

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

            hipMalloc(&d_A[dev], K * M * sizeof(float)); // A is transposed, so K×M
            hipMalloc(&d_B[dev], K * N_local * sizeof(float));
            hipMalloc(&d_C[dev], M * N_local * sizeof(float));

            std::vector<float> h_A(K * M, 1.0f);
            std::vector<float> h_B(K * N_local, 1.0f);

            hipMemcpy(d_A[dev], h_A.data(), K * M * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B[dev], h_B.data(), K * N_local * sizeof(float), hipMemcpyHostToDevice);

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, K); // Transposed layout
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N_local, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N_local, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N_local, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasOperation_t opA = HIPBLAS_OP_T, opB = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                           &opA, sizeof(opA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                           &opB, sizeof(opB));

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           nullptr, nullptr, 0, 0);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": TN operation on " << N_local << " columns" << std::endl;
        }

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "TN: Column partitioned across " << numDevices << " GPUs" << std::endl;
    }

    // ============================================================================
    // Test 4: TT (Transpose both A and B) - Data Parallel
    // ============================================================================
    TEST(MultiGPURealTranspose, DataParallel_TT)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Transpose TT - Data Parallel (all GPUs) ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 2;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_A[dev], batches_per_gpu * K * M * sizeof(float));
            hipMalloc(&d_B[dev], batches_per_gpu * N * K * sizeof(float));
            hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(float));

            for(int64_t b = 0; b < batches_per_gpu; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, K);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, N);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasOperation_t opA = HIPBLAS_OP_T, opB = HIPBLAS_OP_T;
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                               &opA, sizeof(opA));
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                               &opB, sizeof(opB));

                hipblasLtMatmul(handles[dev], matmul, &alpha,
                               d_A[dev] + b * K * M, matA,
                               d_B[dev] + b * N * K, matB,
                               &beta,
                               d_C[dev] + b * M * N, matC,
                               d_C[dev] + b * M * N, matD,
                               nullptr, nullptr, 0, 0);

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }

            hipDeviceSynchronize();
            hipblaslt_cout << "GPU " << dev << ": TT operation on " << batches_per_gpu << " batches" << std::endl;
        }

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "TT: " << (batches_per_gpu * numDevices) << " batches across "
                      << numDevices << " GPUs" << std::endl;
    }

    // ============================================================================
    // Test 5: All Transpose Combinations on Different GPUs
    // ============================================================================
    TEST(MultiGPURealTranspose, DifferentTransposesPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 4) GTEST_SKIP() << "Requires 4+ GPUs for all combinations";

        hipblaslt_cout << "=== Different Transpose Per GPU ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float alpha = 1.0f, beta = 0.0f;

        struct TransposeConfig {
            hipblasOperation_t opA, opB;
            int64_t dimA_rows, dimA_cols, dimB_rows, dimB_cols;
            std::string name;
        };

        std::vector<TransposeConfig> configs = {
            {HIPBLAS_OP_N, HIPBLAS_OP_N, M, K, K, N, "NN"},
            {HIPBLAS_OP_N, HIPBLAS_OP_T, M, K, N, K, "NT"},
            {HIPBLAS_OP_T, HIPBLAS_OP_N, K, M, K, N, "TN"},
            {HIPBLAS_OP_T, HIPBLAS_OP_T, K, M, N, K, "TT"}
        };

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            int config_idx = dev % configs.size();
            auto& cfg = configs[config_idx];

            hipblaslt_cout << "GPU " << dev << ": Using transpose " << cfg.name << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            float *d_A, *d_B, *d_C;
            hipMalloc(&d_A, cfg.dimA_rows * cfg.dimA_cols * sizeof(float));
            hipMalloc(&d_B, cfg.dimB_rows * cfg.dimB_cols * sizeof(float));
            hipMalloc(&d_C, M * N * sizeof(float));

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, cfg.dimA_rows, cfg.dimA_cols,
                                       (cfg.opA == HIPBLAS_OP_N) ? cfg.dimA_rows : cfg.dimA_cols);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, cfg.dimB_rows, cfg.dimB_cols,
                                       (cfg.opB == HIPBLAS_OP_N) ? cfg.dimB_rows : cfg.dimB_cols);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                           &cfg.opA, sizeof(cfg.opA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                           &cfg.opB, sizeof(cfg.opB));

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

        hipblaslt_cout << "Tested all transpose combinations across " << numDevices << " GPUs" << std::endl;
    }

} // namespace
