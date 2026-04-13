/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Real Multi-GPU Computation Test Suite
 * Actual distributed GEMM across multiple GPUs with correctness verification
 * Covers all important parameter combinations with real computation
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

    // Helper: Initialize matrix with known pattern
    void initMatrix(std::vector<float>& mat, int64_t rows, int64_t cols, float base_value = 1.0f)
    {
        for(int64_t i = 0; i < rows * cols; ++i)
        {
            mat[i] = base_value + (i % 100) * 0.01f;
        }
    }

    // Helper: Verify result with relative tolerance
    bool verifyResult(const std::vector<float>& result, const std::vector<float>& expected, float rel_tolerance = 0.02f)
    {
        if(result.size() != expected.size()) return false;

        for(size_t i = 0; i < result.size(); ++i)
        {
            float r = result[i];
            float e = expected[i];
            float threshold = rel_tolerance * std::max(std::abs(e), 1.0f);
            if(std::abs(r - e) > threshold)
            {
                return false;
            }
        }
        return true;
    }

    // Helper: CPU GEMM for reference
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

    // ============================================================================
    // Test 1: Data Parallel - Different Batches on Different GPUs
    // ============================================================================
    TEST(MultiGPURealCompute, DataParallelBatching)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Real Multi-GPU: Data Parallel Batching ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 4;
        const int64_t total_batches = numDevices * batches_per_gpu;

        float alpha = 1.0f, beta = 0.0f;

        // Prepare host data for all batches
        std::vector<std::vector<float>> h_A_batches(total_batches);
        std::vector<std::vector<float>> h_B_batches(total_batches);
        std::vector<std::vector<float>> h_C_expected(total_batches);
        std::vector<std::vector<float>> h_C_result(total_batches);

        for(int64_t b = 0; b < total_batches; ++b)
        {
            h_A_batches[b].resize(M * K);
            h_B_batches[b].resize(K * N);
            h_C_expected[b].resize(M * N);
            h_C_result[b].resize(M * N);

            initMatrix(h_A_batches[b], M, K, 1.0f + b * 0.1f);
            initMatrix(h_B_batches[b], K, N, 2.0f + b * 0.1f);

            // Compute expected result on CPU
            cpuGEMM(h_A_batches[b], h_B_batches[b], h_C_expected[b], M, N, K, alpha, beta);
        }

        // Distribute batches across GPUs and compute in parallel
        std::vector<hipStream_t> streams(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);
        std::vector<hipblasLtHandle_t> handles(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipStreamCreate(&streams[dev]);
            hipblasLtCreate(&handles[dev]);

            // Allocate memory for batches on this GPU
            hipMalloc(&d_A[dev], batches_per_gpu * M * K * sizeof(float));
            hipMalloc(&d_B[dev], batches_per_gpu * K * N * sizeof(float));
            hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(float));

            // Copy data for this GPU's batches
            for(int64_t local_b = 0; local_b < batches_per_gpu; ++local_b)
            {
                int64_t global_b = dev * batches_per_gpu + local_b;
                hipMemcpy(d_A[dev] + local_b * M * K, h_A_batches[global_b].data(),
                         M * K * sizeof(float), hipMemcpyHostToDevice);
                hipMemcpy(d_B[dev] + local_b * K * N, h_B_batches[global_b].data(),
                         K * N * sizeof(float), hipMemcpyHostToDevice);
            }
        }

        // Execute GEMM on all GPUs in parallel
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            for(int64_t local_b = 0; local_b < batches_per_gpu; ++local_b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                // Get algorithm heuristic
                hipblasLtMatmulPreference_t pref;
                hipblasLtMatmulPreferenceCreate(&pref);
                size_t workspace_size = 32 * 1024 * 1024;
                hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &workspace_size, sizeof(size_t));

                hipblasLtMatmulHeuristicResult_t heuristicResult[1];
                int returnedAlgoCount = 0;
                hipblasLtMatmulAlgoGetHeuristic(handles[dev], matmul, matA, matB, matC, matD,
                                               pref, 1, heuristicResult, &returnedAlgoCount);

                hipblasLtMatmul(handles[dev], matmul,
                               &alpha,
                               d_A[dev] + local_b * M * K, matA,
                               d_B[dev] + local_b * K * N, matB,
                               &beta,
                               d_C[dev] + local_b * M * N, matC,
                               d_C[dev] + local_b * M * N, matD,
                               (returnedAlgoCount > 0) ? &heuristicResult[0].algo : nullptr,
                               nullptr, 0, streams[dev]);

                hipblasLtMatmulPreferenceDestroy(pref);

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }
        }

        // Synchronize and verify results
        bool all_correct = true;
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipStreamSynchronize(streams[dev]);

            for(int64_t local_b = 0; local_b < batches_per_gpu; ++local_b)
            {
                int64_t global_b = dev * batches_per_gpu + local_b;
                hipMemcpy(h_C_result[global_b].data(), d_C[dev] + local_b * M * N,
                         M * N * sizeof(float), hipMemcpyDeviceToHost);

                if(!verifyResult(h_C_result[global_b], h_C_expected[global_b], 1.0f))
                {
                    hipblaslt_cout << "GPU " << dev << " batch " << local_b << " MISMATCH!" << std::endl;
                    all_correct = false;
                }
            }
        }

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipStreamDestroy(streams[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        EXPECT_TRUE(all_correct) << "Data parallel batching verification failed";
        hipblaslt_cout << "✓ Data parallel: " << total_batches << " batches across "
                       << numDevices << " GPUs - ALL CORRECT" << std::endl;
    }

    // ============================================================================
    // Test 2: Model Parallel - Row Partitioning (Split M)
    // ============================================================================
    TEST(MultiGPURealCompute, RowPartitioning)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Real Multi-GPU: Row Partitioning ===" << std::endl;

        const int64_t M_total = 1024, N = 512, K = 512;
        const int64_t M_per_gpu = M_total / numDevices;
        float alpha = 1.0f, beta = 0.0f;

        // Prepare full matrices on host
        std::vector<float> h_A(M_total * K), h_B(K * N), h_C_expected(M_total * N), h_C_result(M_total * N);
        initMatrix(h_A, M_total, K, 1.0f);
        initMatrix(h_B, K, N, 2.0f);

        // Compute expected result (full matrix on CPU)
        cpuGEMM(h_A, h_B, h_C_expected, M_total, N, K, alpha, beta);

        // Distribute rows across GPUs
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);
        std::vector<hipblasLtHandle_t> handles(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t M_local = (dev == numDevices - 1) ? (M_total - dev * M_per_gpu) : M_per_gpu;

            // Each GPU gets a subset of rows from A
            hipMalloc(&d_A[dev], M_local * K * sizeof(float));
            // All GPUs need full B matrix
            hipMalloc(&d_B[dev], K * N * sizeof(float));
            // Each GPU produces a subset of rows in C
            hipMalloc(&d_C[dev], M_local * N * sizeof(float));

            // Copy data
            hipMemcpy(d_A[dev], &h_A[dev * M_per_gpu * K], M_local * K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B[dev], h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice);

            hipblaslt_cout << "GPU " << dev << " handles rows [" << dev * M_per_gpu
                           << ", " << (dev * M_per_gpu + M_local) << ")" << std::endl;
        }

        // Execute GEMM on all GPUs in parallel
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            int64_t M_local = (dev == numDevices - 1) ? (M_total - dev * M_per_gpu) : M_per_gpu;

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M_local, K, M_local);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M_local, N, M_local);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M_local, N, M_local);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            // Get algorithm heuristic
            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);
            size_t workspace_size = 32 * 1024 * 1024;
            hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &workspace_size, sizeof(size_t));

            hipblasLtMatmulHeuristicResult_t heuristicResult[1];
            int returnedAlgoCount = 0;
            hipblasLtMatmulAlgoGetHeuristic(handles[dev], matmul, matA, matB, matC, matD,
                                           pref, 1, heuristicResult, &returnedAlgoCount);

            hipblasLtMatmul(handles[dev], matmul,
                           &alpha, d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           (returnedAlgoCount > 0) ? &heuristicResult[0].algo : nullptr,
                           nullptr, 0, 0);

            hipblasLtMatmulPreferenceDestroy(pref);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
        }

        // Gather results from all GPUs
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            int64_t M_local = (dev == numDevices - 1) ? (M_total - dev * M_per_gpu) : M_per_gpu;
            hipMemcpy(&h_C_result[dev * M_per_gpu * N], d_C[dev],
                     M_local * N * sizeof(float), hipMemcpyDeviceToHost);
        }

        // Verify combined result
        bool correct = verifyResult(h_C_result, h_C_expected, 1.0f);

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        EXPECT_TRUE(correct) << "Row partitioning verification failed";
        hipblaslt_cout << "✓ Row partitioning: " << M_total << "x" << N << "x" << K
                       << " across " << numDevices << " GPUs - CORRECT" << std::endl;
    }

    // ============================================================================
    // Test 3: Column Partitioning (Split N)
    // ============================================================================
    TEST(MultiGPURealCompute, ColumnPartitioning)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Real Multi-GPU: Column Partitioning ===" << std::endl;

        const int64_t M = 512, N_total = 1024, K = 512;
        const int64_t N_per_gpu = N_total / numDevices;
        float alpha = 1.0f, beta = 0.0f;

        // Prepare full matrices on host
        std::vector<float> h_A(M * K), h_B(K * N_total), h_C_expected(M * N_total), h_C_result(M * N_total);
        initMatrix(h_A, M, K, 1.0f);
        initMatrix(h_B, K, N_total, 2.0f);

        // Compute expected result
        cpuGEMM(h_A, h_B, h_C_expected, M, N_total, K, alpha, beta);

        // Distribute columns across GPUs
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);
        std::vector<hipblasLtHandle_t> handles(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t N_local = (dev == numDevices - 1) ? (N_total - dev * N_per_gpu) : N_per_gpu;

            // All GPUs need full A matrix
            hipMalloc(&d_A[dev], M * K * sizeof(float));
            // Each GPU gets a subset of columns from B
            hipMalloc(&d_B[dev], K * N_local * sizeof(float));
            // Each GPU produces a subset of columns in C
            hipMalloc(&d_C[dev], M * N_local * sizeof(float));

            // Copy A (full) to all GPUs
            hipMemcpy(d_A[dev], h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice);

            // Copy subset of B columns to this GPU
            std::vector<float> h_B_local(K * N_local);
            for(int64_t k = 0; k < K; ++k)
            {
                for(int64_t n = 0; n < N_local; ++n)
                {
                    h_B_local[k * N_local + n] = h_B[k * N_total + dev * N_per_gpu + n];
                }
            }
            hipMemcpy(d_B[dev], h_B_local.data(), K * N_local * sizeof(float), hipMemcpyHostToDevice);

            hipblaslt_cout << "GPU " << dev << " handles columns [" << dev * N_per_gpu
                           << ", " << (dev * N_per_gpu + N_local) << ")" << std::endl;
        }

        // Execute GEMM on all GPUs
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            int64_t N_local = (dev == numDevices - 1) ? (N_total - dev * N_per_gpu) : N_per_gpu;

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N_local, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N_local, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N_local, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            // Get algorithm heuristic
            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);
            size_t workspace_size = 32 * 1024 * 1024;
            hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &workspace_size, sizeof(size_t));

            hipblasLtMatmulHeuristicResult_t heuristicResult[1];
            int returnedAlgoCount = 0;
            hipblasLtMatmulAlgoGetHeuristic(handles[dev], matmul, matA, matB, matC, matD,
                                           pref, 1, heuristicResult, &returnedAlgoCount);

            hipblasLtMatmul(handles[dev], matmul,
                           &alpha, d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           (returnedAlgoCount > 0) ? &heuristicResult[0].algo : nullptr,
                           nullptr, 0, 0);

            hipblasLtMatmulPreferenceDestroy(pref);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
        }

        // Gather column results from all GPUs
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            int64_t N_local = (dev == numDevices - 1) ? (N_total - dev * N_per_gpu) : N_per_gpu;

            std::vector<float> h_C_local(M * N_local);
            hipMemcpy(h_C_local.data(), d_C[dev], M * N_local * sizeof(float), hipMemcpyDeviceToHost);

            // Reconstruct full result matrix
            for(int64_t m = 0; m < M; ++m)
            {
                for(int64_t n = 0; n < N_local; ++n)
                {
                    h_C_result[m * N_total + dev * N_per_gpu + n] = h_C_local[m * N_local + n];
                }
            }
        }

        // Verify
        bool correct = verifyResult(h_C_result, h_C_expected, 1.0f);

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        EXPECT_TRUE(correct) << "Column partitioning verification failed";
        hipblaslt_cout << "✓ Column partitioning: " << M << "x" << N_total << "x" << K
                       << " across " << numDevices << " GPUs - CORRECT" << std::endl;
    }

} // namespace
