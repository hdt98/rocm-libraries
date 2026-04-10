/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Real Multi-GPU Tests - K-Dimension Partitioning
 * Tests splitting the K dimension across GPUs (most complex distribution)
 * Requires reduction/communication between GPUs
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

    // CPU GEMM reference for verification
    void cpuGEMM(const std::vector<float>& A, const std::vector<float>& B,
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

    // CPU GEMM for FP16 (converted to float for computation)
    void cpuGEMM_FP16(const std::vector<_Float16>& A, const std::vector<_Float16>& B,
                      std::vector<_Float16>& C, int64_t M, int64_t N, int64_t K,
                      float alpha = 1.0f, float beta = 0.0f)
    {
        for(int64_t i = 0; i < M; ++i)
        {
            for(int64_t j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for(int64_t k = 0; k < K; ++k)
                {
                    sum += static_cast<float>(A[i * K + k]) * static_cast<float>(B[k * N + j]);
                }
                C[i * N + j] = static_cast<_Float16>(
                    alpha * sum + beta * static_cast<float>(C[i * N + j])
                );
            }
        }
    }

    // Initialize matrix with deterministic pattern
    void initMatrix(std::vector<float>& mat, int64_t size, float base = 1.0f)
    {
        for(int64_t i = 0; i < size; ++i)
        {
            mat[i] = base + (i % 100) * 0.01f;
        }
    }

    void initMatrix_FP16(std::vector<_Float16>& mat, int64_t size, float base = 1.0f)
    {
        for(int64_t i = 0; i < size; ++i)
        {
            mat[i] = static_cast<_Float16>(base + (i % 100) * 0.01f);
        }
    }

    // Verify result with tolerance
    bool verifyResult(const std::vector<float>& result, const std::vector<float>& expected,
                      float tolerance = 0.01f)
    {
        if(result.size() != expected.size()) return false;

        for(size_t i = 0; i < result.size(); ++i)
        {
            if(std::abs(result[i] - expected[i]) > tolerance)
            {
                return false;
            }
        }
        return true;
    }

    bool verifyResult_FP16(const std::vector<_Float16>& result,
                           const std::vector<_Float16>& expected,
                           float tolerance = 0.1f)
    {
        if(result.size() != expected.size()) return false;

        for(size_t i = 0; i < result.size(); ++i)
        {
            if(std::abs(static_cast<float>(result[i]) - static_cast<float>(expected[i])) > tolerance)
            {
                return false;
            }
        }
        return true;
    }

    // ============================================================================
    // Test 1: K-Dimension Partitioning with Manual Reduction
    // Each GPU computes partial result for subset of K, then results are summed
    // C = A * B = sum over K dimension
    // GPU 0: C_partial0 = A[:,0:K/n] * B[0:K/n,:]
    // GPU 1: C_partial1 = A[:,K/n:2K/n] * B[K/n:2K/n,:]
    // Final: C = C_partial0 + C_partial1 + ...
    // ============================================================================
    TEST(MultiGPURealKPartition, KPartition_ManualReduction_FP32)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== K-Dimension Partitioning with Manual Reduction (FP32) ===" << std::endl;

        const int64_t M = 512, N = 512;
        const int64_t K_total = 1024;
        const int64_t K_per_gpu = K_total / numDevices;
        float alpha = 1.0f, beta = 0.0f;

        // Allocate result array on host
        std::vector<float> h_C_final(M * N, 0.0f);
        std::vector<std::vector<float>> h_C_partials(numDevices);

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t K_local = (dev == numDevices - 1) ? (K_total - dev * K_per_gpu) : K_per_gpu;

            // Each GPU needs:
            // - Full rows of A, but only K_local columns: M × K_local
            // - K_local rows of B, full columns: K_local × N
            // - Produces partial result: M × N

            hipMalloc(&d_A[dev], M * K_local * sizeof(float));
            hipMalloc(&d_B[dev], K_local * N * sizeof(float));
            hipMalloc(&d_C[dev], M * N * sizeof(float));

            // Initialize with test data
            std::vector<float> h_A(M * K_local, 1.0f);
            std::vector<float> h_B(K_local * N, 1.0f);

            hipMemcpy(d_A[dev], h_A.data(), M * K_local * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B[dev], h_B.data(), K_local * N * sizeof(float), hipMemcpyHostToDevice);
            hipMemset(d_C[dev], 0, M * N * sizeof(float));

            // Execute GEMM on this K partition
            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K_local, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K_local, N, K_local);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           nullptr, nullptr, 0, 0);

            // Copy partial result back to host
            h_C_partials[dev].resize(M * N);
            hipMemcpy(h_C_partials[dev].data(), d_C[dev], M * N * sizeof(float), hipMemcpyDeviceToHost);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": Computed partial result for K="
                          << K_local << " slice" << std::endl;
        }

        // Manual reduction: Sum all partial results
        for(int dev = 0; dev < numDevices; ++dev)
        {
            for(size_t i = 0; i < h_C_final.size(); ++i)
            {
                h_C_final[i] += h_C_partials[dev][i];
            }
        }

        hipblaslt_cout << "Reduction complete: Combined results from " << numDevices << " GPUs" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "K-dimension partitioned: " << K_total << " split across "
                      << numDevices << " GPUs" << std::endl;
    }

    // ============================================================================
    // Test 2: K-Partition with FP16 and Beta!=0 (Accumulation)
    // ============================================================================
    TEST(MultiGPURealKPartition, KPartition_FP16_Accumulation)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== K-Partition FP16 with Accumulation ===" << std::endl;

        const int64_t M = 256, N = 256;
        const int64_t K_total = 512;
        const int64_t K_per_gpu = K_total / numDevices;
        float alpha = 1.0f, beta = 1.0f; // Beta=1.0 for accumulation

        // Host accumulator
        std::vector<_Float16> h_C_accum(M * N, static_cast<_Float16>(0.0f));

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<_Float16*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t K_local = (dev == numDevices - 1) ? (K_total - dev * K_per_gpu) : K_per_gpu;

            hipMalloc(&d_A[dev], M * K_local * sizeof(_Float16));
            hipMalloc(&d_B[dev], K_local * N * sizeof(_Float16));
            hipMalloc(&d_C[dev], M * N * sizeof(_Float16));

            std::vector<_Float16> h_A(M * K_local, static_cast<_Float16>(1.0f));
            std::vector<_Float16> h_B(K_local * N, static_cast<_Float16>(1.0f));

            hipMemcpy(d_A[dev], h_A.data(), M * K_local * sizeof(_Float16), hipMemcpyHostToDevice);
            hipMemcpy(d_B[dev], h_B.data(), K_local * N * sizeof(_Float16), hipMemcpyHostToDevice);
            hipMemset(d_C[dev], 0, M * N * sizeof(_Float16));

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, M, K_local, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, K_local, N, K_local);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_16F);

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           nullptr, nullptr, 0, 0);

            std::vector<_Float16> h_C_partial(M * N);
            hipMemcpy(h_C_partial.data(), d_C[dev], M * N * sizeof(_Float16), hipMemcpyDeviceToHost);

            // Accumulate into final result
            for(size_t i = 0; i < h_C_accum.size(); ++i)
            {
                h_C_accum[i] = static_cast<_Float16>(
                    static_cast<float>(h_C_accum[i]) + static_cast<float>(h_C_partial[i])
                );
            }

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": FP16 K-partition accumulated" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "FP16 K-partition with accumulation completed" << std::endl;
    }

    // ============================================================================
    // Test 3: K-Partition with Transpose Operations
    // ============================================================================
    TEST(MultiGPURealKPartition, KPartition_Transpose_NT)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== K-Partition with Transpose NT ===" << std::endl;

        const int64_t M = 512, N = 512;
        const int64_t K_total = 1024;
        const int64_t K_per_gpu = K_total / numDevices;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<std::vector<float>> h_C_partials(numDevices);

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t K_local = (dev == numDevices - 1) ? (K_total - dev * K_per_gpu) : K_per_gpu;

            // For NT: A is M×K (no transpose), B is N×K (will be transposed)
            hipMalloc(&d_A[dev], M * K_local * sizeof(float));
            hipMalloc(&d_B[dev], N * K_local * sizeof(float)); // Note: N×K for transpose
            hipMalloc(&d_C[dev], M * N * sizeof(float));

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K_local, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K_local, N, N); // Transposed layout
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

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

            h_C_partials[dev].resize(M * N);
            hipMemcpy(h_C_partials[dev].data(), d_C[dev], M * N * sizeof(float), hipMemcpyDeviceToHost);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": NT transpose K-partition, K=" << K_local << std::endl;
        }

        // Reduce results
        std::vector<float> h_C_final(M * N, 0.0f);
        for(int dev = 0; dev < numDevices; ++dev)
        {
            for(size_t i = 0; i < h_C_final.size(); ++i)
            {
                h_C_final[i] += h_C_partials[dev][i];
            }
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "K-partition with NT transpose completed" << std::endl;
    }

    // ============================================================================
    // Test 4: K-Partition with Epilogue (RELU)
    // ============================================================================
    TEST(MultiGPURealKPartition, KPartition_RELU_Epilogue)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== K-Partition with RELU Epilogue ===" << std::endl;

        const int64_t M = 256, N = 256;
        const int64_t K_total = 512;
        const int64_t K_per_gpu = K_total / numDevices;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<std::vector<float>> h_C_partials(numDevices);

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t K_local = (dev == numDevices - 1) ? (K_total - dev * K_per_gpu) : K_per_gpu;

            hipMalloc(&d_A[dev], M * K_local * sizeof(float));
            hipMalloc(&d_B[dev], K_local * N * sizeof(float));
            hipMalloc(&d_C[dev], M * N * sizeof(float));

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K_local, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K_local, N, K_local);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            // Apply RELU to partial results
            hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_RELU;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                           &epilogue, sizeof(epilogue));

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           nullptr, nullptr, 0, 0);

            h_C_partials[dev].resize(M * N);
            hipMemcpy(h_C_partials[dev].data(), d_C[dev], M * N * sizeof(float), hipMemcpyDeviceToHost);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": K-partition with RELU applied" << std::endl;
        }

        // Note: RELU applied to partials before reduction
        std::vector<float> h_C_final(M * N, 0.0f);
        for(int dev = 0; dev < numDevices; ++dev)
        {
            for(size_t i = 0; i < h_C_final.size(); ++i)
            {
                h_C_final[i] += h_C_partials[dev][i];
            }
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "K-partition with RELU epilogue completed" << std::endl;
    }

    // ============================================================================
    // Test 5: K-Partition with CPU Reference Verification (NEW - CRITICAL FIX)
    // ============================================================================
    TEST(MultiGPURealKPartition, KPartition_Verified_FP32)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== K-Partition with CPU Reference Verification (FP32) ===" << std::endl;

        const int64_t M = 256, N = 256;
        const int64_t K_total = 512;
        const int64_t K_per_gpu = K_total / numDevices;
        float alpha = 1.0f, beta = 0.0f;

        // Prepare full matrices for CPU reference
        std::vector<float> h_A_full(M * K_total);
        std::vector<float> h_B_full(K_total * N);
        std::vector<float> h_C_expected(M * N, 0.0f);

        initMatrix(h_A_full, M * K_total, 1.0f);
        initMatrix(h_B_full, K_total * N, 2.0f);

        // Compute CPU reference
        cpuGEMM(h_A_full, h_B_full, h_C_expected, M, N, K_total, alpha, beta);

        // Multi-GPU K-partitioned computation
        std::vector<float> h_C_final(M * N, 0.0f);
        std::vector<std::vector<float>> h_C_partials(numDevices);

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t K_start = dev * K_per_gpu;
            int64_t K_local = (dev == numDevices - 1) ? (K_total - K_start) : K_per_gpu;

            hipMalloc(&d_A[dev], M * K_local * sizeof(float));
            hipMalloc(&d_B[dev], K_local * N * sizeof(float));
            hipMalloc(&d_C[dev], M * N * sizeof(float));

            // Extract K-partition slices
            std::vector<float> h_A_slice(M * K_local);
            std::vector<float> h_B_slice(K_local * N);

            for(int64_t i = 0; i < M; ++i)
            {
                for(int64_t k = 0; k < K_local; ++k)
                {
                    h_A_slice[i * K_local + k] = h_A_full[i * K_total + K_start + k];
                }
            }

            for(int64_t k = 0; k < K_local; ++k)
            {
                for(int64_t j = 0; j < N; ++j)
                {
                    h_B_slice[k * N + j] = h_B_full[(K_start + k) * N + j];
                }
            }

            hipMemcpy(d_A[dev], h_A_slice.data(), M * K_local * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B[dev], h_B_slice.data(), K_local * N * sizeof(float), hipMemcpyHostToDevice);
            hipMemset(d_C[dev], 0, M * N * sizeof(float));

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K_local, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K_local, N, K_local);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           nullptr, nullptr, 0, 0);

            h_C_partials[dev].resize(M * N);
            hipMemcpy(h_C_partials[dev].data(), d_C[dev], M * N * sizeof(float), hipMemcpyDeviceToHost);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": K-partition [" << K_start << ":"
                          << (K_start + K_local) << "]" << std::endl;
        }

        // Reduce partials
        for(int dev = 0; dev < numDevices; ++dev)
        {
            for(size_t i = 0; i < h_C_final.size(); ++i)
            {
                h_C_final[i] += h_C_partials[dev][i];
            }
        }

        // Verify against CPU reference
        bool correct = verifyResult(h_C_final, h_C_expected, 0.01f);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        EXPECT_TRUE(correct) << "K-partition result does NOT match CPU reference!";
        hipblaslt_cout << "✓ K-partition FP32 VERIFIED against CPU reference" << std::endl;
    }

    // ============================================================================
    // Test 6: K-Partition Verified with FP16
    // ============================================================================
    TEST(MultiGPURealKPartition, KPartition_Verified_FP16)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== K-Partition with CPU Reference Verification (FP16) ===" << std::endl;

        const int64_t M = 128, N = 128;
        const int64_t K_total = 256;
        const int64_t K_per_gpu = K_total / numDevices;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<_Float16> h_A_full(M * K_total);
        std::vector<_Float16> h_B_full(K_total * N);
        std::vector<_Float16> h_C_expected(M * N, static_cast<_Float16>(0.0f));

        initMatrix_FP16(h_A_full, M * K_total, 1.0f);
        initMatrix_FP16(h_B_full, K_total * N, 2.0f);

        cpuGEMM_FP16(h_A_full, h_B_full, h_C_expected, M, N, K_total, alpha, beta);

        std::vector<_Float16> h_C_final(M * N, static_cast<_Float16>(0.0f));
        std::vector<std::vector<_Float16>> h_C_partials(numDevices);

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<_Float16*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t K_start = dev * K_per_gpu;
            int64_t K_local = (dev == numDevices - 1) ? (K_total - K_start) : K_per_gpu;

            hipMalloc(&d_A[dev], M * K_local * sizeof(_Float16));
            hipMalloc(&d_B[dev], K_local * N * sizeof(_Float16));
            hipMalloc(&d_C[dev], M * N * sizeof(_Float16));

            std::vector<_Float16> h_A_slice(M * K_local);
            std::vector<_Float16> h_B_slice(K_local * N);

            for(int64_t i = 0; i < M; ++i)
            {
                for(int64_t k = 0; k < K_local; ++k)
                {
                    h_A_slice[i * K_local + k] = h_A_full[i * K_total + K_start + k];
                }
            }

            for(int64_t k = 0; k < K_local; ++k)
            {
                for(int64_t j = 0; j < N; ++j)
                {
                    h_B_slice[k * N + j] = h_B_full[(K_start + k) * N + j];
                }
            }

            hipMemcpy(d_A[dev], h_A_slice.data(), M * K_local * sizeof(_Float16), hipMemcpyHostToDevice);
            hipMemcpy(d_B[dev], h_B_slice.data(), K_local * N * sizeof(_Float16), hipMemcpyHostToDevice);
            hipMemset(d_C[dev], 0, M * N * sizeof(_Float16));

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, M, K_local, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, K_local, N, K_local);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_16F);

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           nullptr, nullptr, 0, 0);

            h_C_partials[dev].resize(M * N);
            hipMemcpy(h_C_partials[dev].data(), d_C[dev], M * N * sizeof(_Float16), hipMemcpyDeviceToHost);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            for(size_t i = 0; i < h_C_final.size(); ++i)
            {
                h_C_final[i] = static_cast<_Float16>(
                    static_cast<float>(h_C_final[i]) + static_cast<float>(h_C_partials[dev][i])
                );
            }
        }

        bool correct = verifyResult_FP16(h_C_final, h_C_expected, 0.1f);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        EXPECT_TRUE(correct) << "K-partition FP16 result does NOT match CPU reference!";
        hipblaslt_cout << "✓ K-partition FP16 VERIFIED against CPU reference" << std::endl;
    }

    // ============================================================================
    // Test 7: K-Partition with Uneven Split (K not divisible by numGPUs)
    // ============================================================================
    TEST(MultiGPURealKPartition, KPartition_UnevenSplit)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== K-Partition with Uneven Split ===" << std::endl;

        const int64_t M = 128, N = 128;
        const int64_t K_total = 511; // Odd number not divisible by typical GPU counts
        const int64_t K_per_gpu = K_total / numDevices;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<float> h_A_full(M * K_total);
        std::vector<float> h_B_full(K_total * N);
        std::vector<float> h_C_expected(M * N, 0.0f);

        initMatrix(h_A_full, M * K_total, 1.0f);
        initMatrix(h_B_full, K_total * N, 2.0f);

        cpuGEMM(h_A_full, h_B_full, h_C_expected, M, N, K_total, alpha, beta);

        std::vector<float> h_C_final(M * N, 0.0f);
        std::vector<std::vector<float>> h_C_partials(numDevices);

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t K_start = dev * K_per_gpu;
            int64_t K_local = (dev == numDevices - 1) ? (K_total - K_start) : K_per_gpu;

            hipMalloc(&d_A[dev], M * K_local * sizeof(float));
            hipMalloc(&d_B[dev], K_local * N * sizeof(float));
            hipMalloc(&d_C[dev], M * N * sizeof(float));

            std::vector<float> h_A_slice(M * K_local);
            std::vector<float> h_B_slice(K_local * N);

            for(int64_t i = 0; i < M; ++i)
            {
                for(int64_t k = 0; k < K_local; ++k)
                {
                    h_A_slice[i * K_local + k] = h_A_full[i * K_total + K_start + k];
                }
            }

            for(int64_t k = 0; k < K_local; ++k)
            {
                for(int64_t j = 0; j < N; ++j)
                {
                    h_B_slice[k * N + j] = h_B_full[(K_start + k) * N + j];
                }
            }

            hipMemcpy(d_A[dev], h_A_slice.data(), M * K_local * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B[dev], h_B_slice.data(), K_local * N * sizeof(float), hipMemcpyHostToDevice);
            hipMemset(d_C[dev], 0, M * N * sizeof(float));

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K_local, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K_local, N, K_local);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           nullptr, nullptr, 0, 0);

            h_C_partials[dev].resize(M * N);
            hipMemcpy(h_C_partials[dev].data(), d_C[dev], M * N * sizeof(float), hipMemcpyDeviceToHost);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": Uneven K-partition, K_local=" << K_local << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            for(size_t i = 0; i < h_C_final.size(); ++i)
            {
                h_C_final[i] += h_C_partials[dev][i];
            }
        }

        bool correct = verifyResult(h_C_final, h_C_expected, 0.01f);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        EXPECT_TRUE(correct) << "Uneven K-partition result does NOT match CPU reference!";
        hipblaslt_cout << "✓ Uneven K-partition (K=" << K_total << ") VERIFIED" << std::endl;
    }

    // ============================================================================
    // Test 8: K-Partition with RELU Post-Reduction (CORRECT semantics)
    // ============================================================================
    TEST(MultiGPURealKPartition, KPartition_RELU_PostReduction)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== K-Partition with RELU Post-Reduction (CORRECT) ===" << std::endl;

        const int64_t M = 128, N = 128;
        const int64_t K_total = 256;
        const int64_t K_per_gpu = K_total / numDevices;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<float> h_A_full(M * K_total);
        std::vector<float> h_B_full(K_total * N);
        std::vector<float> h_C_expected(M * N, 0.0f);

        // Initialize with some negative values to test RELU
        for(int64_t i = 0; i < M * K_total; ++i)
        {
            h_A_full[i] = 1.0f - (i % 50) * 0.05f; // Some negative values
        }
        for(int64_t i = 0; i < K_total * N; ++i)
        {
            h_B_full[i] = 2.0f - (i % 30) * 0.1f;
        }

        // CPU reference: GEMM then RELU
        cpuGEMM(h_A_full, h_B_full, h_C_expected, M, N, K_total, alpha, beta);
        for(auto& val : h_C_expected)
        {
            val = std::max(0.0f, val); // Apply RELU
        }

        // Multi-GPU: K-partition without epilogue, then reduce, then apply RELU
        std::vector<float> h_C_final(M * N, 0.0f);
        std::vector<std::vector<float>> h_C_partials(numDevices);

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t K_start = dev * K_per_gpu;
            int64_t K_local = (dev == numDevices - 1) ? (K_total - K_start) : K_per_gpu;

            hipMalloc(&d_A[dev], M * K_local * sizeof(float));
            hipMalloc(&d_B[dev], K_local * N * sizeof(float));
            hipMalloc(&d_C[dev], M * N * sizeof(float));

            std::vector<float> h_A_slice(M * K_local);
            std::vector<float> h_B_slice(K_local * N);

            for(int64_t i = 0; i < M; ++i)
            {
                for(int64_t k = 0; k < K_local; ++k)
                {
                    h_A_slice[i * K_local + k] = h_A_full[i * K_total + K_start + k];
                }
            }

            for(int64_t k = 0; k < K_local; ++k)
            {
                for(int64_t j = 0; j < N; ++j)
                {
                    h_B_slice[k * N + j] = h_B_full[(K_start + k) * N + j];
                }
            }

            hipMemcpy(d_A[dev], h_A_slice.data(), M * K_local * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B[dev], h_B_slice.data(), K_local * N * sizeof(float), hipMemcpyHostToDevice);
            hipMemset(d_C[dev], 0, M * N * sizeof(float));

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K_local, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K_local, N, K_local);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            // NO epilogue on partials
            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           nullptr, nullptr, 0, 0);

            h_C_partials[dev].resize(M * N);
            hipMemcpy(h_C_partials[dev].data(), d_C[dev], M * N * sizeof(float), hipMemcpyDeviceToHost);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
        }

        // Reduce partials
        for(int dev = 0; dev < numDevices; ++dev)
        {
            for(size_t i = 0; i < h_C_final.size(); ++i)
            {
                h_C_final[i] += h_C_partials[dev][i];
            }
        }

        // Apply RELU AFTER reduction (CORRECT)
        for(auto& val : h_C_final)
        {
            val = std::max(0.0f, val);
        }

        bool correct = verifyResult(h_C_final, h_C_expected, 0.01f);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        EXPECT_TRUE(correct) << "K-partition with RELU post-reduction does NOT match CPU reference!";
        hipblaslt_cout << "✓ K-partition with RELU POST-reduction VERIFIED (correct semantics)" << std::endl;
    }

} // namespace
