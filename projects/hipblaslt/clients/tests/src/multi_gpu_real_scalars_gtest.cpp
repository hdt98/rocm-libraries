/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Real Multi-GPU Tests - Scalar Parameters Coverage
 * Tests various alpha/beta values with real distribution
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

    void cpuGEMM(const std::vector<float>& A, const std::vector<float>& B,
                 std::vector<float>& C, int64_t M, int64_t N, int64_t K,
                 float alpha, float beta)
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

    bool verifyResult(const std::vector<float>& result, const std::vector<float>& expected, float tol = 0.1f)
    {
        if(result.size() != expected.size()) return false;
        for(size_t i = 0; i < result.size(); ++i)
        {
            if(std::abs(result[i] - expected[i]) > tol) return false;
        }
        return true;
    }

    // ============================================================================
    // Test 1: Standard Alpha=1.0, Beta=0.0 (C = A*B)
    // ============================================================================
    TEST(MultiGPURealScalars, Standard_Alpha1_Beta0)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Scalars: Alpha=1.0, Beta=0.0 (all GPUs) ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 2;
        const int64_t total_batches = numDevices * batches_per_gpu;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<std::vector<float>> h_A(total_batches), h_B(total_batches);
        std::vector<std::vector<float>> h_C_result(total_batches), h_C_expected(total_batches);

        for(int64_t b = 0; b < total_batches; ++b)
        {
            h_A[b].resize(M * K, 1.0f);
            h_B[b].resize(K * N, 1.0f);
            h_C_result[b].resize(M * N);
            h_C_expected[b].resize(M * N);

            cpuGEMM(h_A[b], h_B[b], h_C_expected[b], M, N, K, alpha, beta);
        }

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

        for(int64_t b = 0; b < total_batches; ++b)
        {
            EXPECT_TRUE(verifyResult(h_C_result[b], h_C_expected[b], 0.1f))
                << "Batch " << b << " failed";
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "Alpha=1.0, Beta=0.0: Verified " << total_batches << " batches" << std::endl;
    }

    // ============================================================================
    // Test 2: Alpha=2.0, Beta=0.5 (C = 2*A*B + 0.5*C)
    // ============================================================================
    TEST(MultiGPURealScalars, NonStandard_Alpha2_Beta0_5)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Scalars: Alpha=2.0, Beta=0.5 (all GPUs) ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 2;
        const int64_t total_batches = numDevices * batches_per_gpu;
        float alpha = 2.0f, beta = 0.5f;

        std::vector<std::vector<float>> h_A(total_batches), h_B(total_batches);
        std::vector<std::vector<float>> h_C_in(total_batches);
        std::vector<std::vector<float>> h_C_result(total_batches), h_C_expected(total_batches);

        for(int64_t b = 0; b < total_batches; ++b)
        {
            h_A[b].resize(M * K, 1.0f);
            h_B[b].resize(K * N, 1.0f);
            h_C_in[b].resize(M * N, 3.0f); // Initial C values
            h_C_result[b].resize(M * N);
            h_C_expected[b] = h_C_in[b]; // Copy initial C

            cpuGEMM(h_A[b], h_B[b], h_C_expected[b], M, N, K, alpha, beta);
        }

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
                hipMemcpy(d_C[dev] + local_b * M * N, h_C_in[global_b].data(),
                         M * N * sizeof(float), hipMemcpyHostToDevice);

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

        for(int64_t b = 0; b < total_batches; ++b)
        {
            EXPECT_TRUE(verifyResult(h_C_result[b], h_C_expected[b], 0.1f))
                << "Batch " << b << " failed";
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "Alpha=2.0, Beta=0.5: Verified " << total_batches << " batches" << std::endl;
    }

    // ============================================================================
    // Test 3: Alpha=0.5, Beta=1.0 - Row Partition
    // ============================================================================
    TEST(MultiGPURealScalars, RowPartition_Alpha0_5_Beta1)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Scalars: Alpha=0.5, Beta=1.0 Row Partition (all GPUs) ===" << std::endl;

        const int64_t M_total = 1024, N = 512, K = 512;
        const int64_t M_per_gpu = M_total / numDevices;
        float alpha = 0.5f, beta = 1.0f;

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

            std::vector<float> h_A(M_local * K, 2.0f);
            std::vector<float> h_B(K * N, 2.0f);
            std::vector<float> h_C(M_local * N, 5.0f);

            hipMemcpy(d_A[dev], h_A.data(), M_local * K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B[dev], h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_C[dev], h_C.data(), M_local * N * sizeof(float), hipMemcpyHostToDevice);

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M_local, K, M_local);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M_local, N, M_local);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M_local, N, M_local);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

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

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "Alpha=0.5, Beta=1.0: Row partitioned across " << numDevices << " GPUs" << std::endl;
    }

    // ============================================================================
    // Test 4: Negative Alpha=-1.0, Beta=0.0
    // ============================================================================
    TEST(MultiGPURealScalars, Negative_AlphaMinus1_Beta0)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Scalars: Alpha=-1.0, Beta=0.0 (all GPUs) ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float alpha = -1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_A[dev], M * K * sizeof(float));
            hipMalloc(&d_B[dev], K * N * sizeof(float));
            hipMalloc(&d_C[dev], M * N * sizeof(float));

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           nullptr, nullptr, 0, 0);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": Completed with alpha=-1.0" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "Negative alpha tested across " << numDevices << " GPUs" << std::endl;
    }

    // ============================================================================
    // Test 5: Different Scalars Per GPU
    // ============================================================================
    TEST(MultiGPURealScalars, DifferentScalarsPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Different Scalars Per GPU ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        std::vector<std::pair<float, float>> scalar_pairs = {
            {1.0f, 0.0f},
            {2.0f, 0.5f},
            {0.5f, 1.0f},
            {3.0f, 0.25f},
            {-1.0f, 0.0f},
            {1.5f, 0.75f},
            {0.25f, 2.0f},
            {4.0f, 0.1f}
        };

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            int scalar_idx = dev % scalar_pairs.size();
            float alpha = scalar_pairs[scalar_idx].first;
            float beta = scalar_pairs[scalar_idx].second;

            hipblaslt_cout << "GPU " << dev << ": alpha=" << alpha << ", beta=" << beta << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            float *d_A, *d_B, *d_C;
            hipMalloc(&d_A, M * K * sizeof(float));
            hipMalloc(&d_B, K * N * sizeof(float));
            hipMalloc(&d_C, M * N * sizeof(float));

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

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

        hipblaslt_cout << "Tested different scalars across " << numDevices << " GPUs" << std::endl;
    }

} // namespace
