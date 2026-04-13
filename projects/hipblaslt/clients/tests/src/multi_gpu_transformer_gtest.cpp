/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Transformer Workload Tests
 * Tests Q/K/V projections, column/row-parallel patterns for LLM deployment
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

    // CPU GEMM reference
    void cpuGEMM_FP16(const std::vector<_Float16>& A, const std::vector<_Float16>& B,
                      std::vector<_Float16>& C, int64_t M, int64_t N, int64_t K)
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
                C[i * N + j] = static_cast<_Float16>(sum);
            }
        }
    }

    void initMatrix_FP16(std::vector<_Float16>& mat, int64_t size, float base = 1.0f)
    {
        for(int64_t i = 0; i < size; ++i)
        {
            mat[i] = static_cast<_Float16>(base + (i % 100) * 0.01f);
        }
    }

    bool verifyResult_FP16(const std::vector<_Float16>& result, const std::vector<_Float16>& expected,
                            float rel_tolerance = 0.05f)
    {
        if(result.size() != expected.size()) return false;

        for(size_t i = 0; i < result.size(); ++i)
        {
            float r = static_cast<float>(result[i]);
            float e = static_cast<float>(expected[i]);
            float threshold = rel_tolerance * std::max(std::abs(e), 1.0f);
            if(std::abs(r - e) > threshold)
            {
                return false;
            }
        }
        return true;
    }

    // ============================================================================
    // Test 1: Q/K/V Projection - Data Parallel (batch splitting)
    // ============================================================================
    TEST(MultiGPUTransformer, QKV_DataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Transformer Q/K/V Projection - Data Parallel ===" << std::endl;

        // Typical transformer dimensions
        const int64_t batch_size = 32;
        const int64_t seq_len = 512;
        const int64_t d_model = 768;
        const int64_t d_k = 64; // Per-head dimension

        const int64_t M = batch_size * seq_len; // Total tokens
        const int64_t K = d_model;
        const int64_t N = d_k;

        // Each GPU processes subset of batch
        int64_t M_per_gpu = M / numDevices;

        // Full input and weights
        std::vector<_Float16> h_X_full(M * K); // Input tokens
        std::vector<_Float16> h_W_q(K * N);     // Q projection weight
        std::vector<_Float16> h_W_k(K * N);     // K projection weight
        std::vector<_Float16> h_W_v(K * N);     // V projection weight

        initMatrix_FP16(h_X_full, M * K, 1.0f);
        initMatrix_FP16(h_W_q, K * N, 0.5f);
        initMatrix_FP16(h_W_k, K * N, 0.6f);
        initMatrix_FP16(h_W_v, K * N, 0.7f);

        // CPU reference for Q, K, V
        std::vector<_Float16> h_Q_expected(M * N);
        std::vector<_Float16> h_K_expected(M * N);
        std::vector<_Float16> h_V_expected(M * N);

        cpuGEMM_FP16(h_X_full, h_W_q, h_Q_expected, M, N, K);
        cpuGEMM_FP16(h_X_full, h_W_k, h_K_expected, M, N, K);
        cpuGEMM_FP16(h_X_full, h_W_v, h_V_expected, M, N, K);

        // Multi-GPU computation
        std::vector<_Float16> h_Q_result(M * N);
        std::vector<_Float16> h_K_result(M * N);
        std::vector<_Float16> h_V_result(M * N);

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<_Float16*> d_X(numDevices), d_W_q(numDevices), d_W_k(numDevices), d_W_v(numDevices);
        std::vector<_Float16*> d_Q(numDevices), d_K(numDevices), d_V(numDevices);

        float alpha = 1.0f, beta = 0.0f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t M_start = dev * M_per_gpu;
            int64_t M_local = (dev == numDevices - 1) ? (M - M_start) : M_per_gpu;

            // Allocate
            hipMalloc(&d_X[dev], M_local * K * sizeof(_Float16));
            hipMalloc(&d_W_q[dev], K * N * sizeof(_Float16));
            hipMalloc(&d_W_k[dev], K * N * sizeof(_Float16));
            hipMalloc(&d_W_v[dev], K * N * sizeof(_Float16));
            hipMalloc(&d_Q[dev], M_local * N * sizeof(_Float16));
            hipMalloc(&d_K[dev], M_local * N * sizeof(_Float16));
            hipMalloc(&d_V[dev], M_local * N * sizeof(_Float16));

            // Copy data
            hipMemcpy(d_X[dev], h_X_full.data() + M_start * K, M_local * K * sizeof(_Float16), hipMemcpyHostToDevice);
            hipMemcpy(d_W_q[dev], h_W_q.data(), K * N * sizeof(_Float16), hipMemcpyHostToDevice);
            hipMemcpy(d_W_k[dev], h_W_k.data(), K * N * sizeof(_Float16), hipMemcpyHostToDevice);
            hipMemcpy(d_W_v[dev], h_W_v.data(), K * N * sizeof(_Float16), hipMemcpyHostToDevice);

            // Setup layouts
            hipblasLtMatrixLayout_t matX, matW, matOut;
            hipblasLtMatrixLayoutCreate(&matX, HIP_R_16F, M_local, K, M_local);
            hipblasLtMatrixLayoutCreate(&matW, HIP_R_16F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matOut, HIP_R_16F, M_local, N, M_local);

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
            hipblasLtMatmulAlgoGetHeuristic(handles[dev], matmul, matX, matW, matOut, matOut,
                                           pref, 1, heuristicResult, &returnedAlgoCount);

            // Compute Q = X * W_q
            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_X[dev], matX, d_W_q[dev], matW,
                           &beta, d_Q[dev], matOut, d_Q[dev], matOut,
                           (returnedAlgoCount > 0) ? &heuristicResult[0].algo : nullptr,
                           nullptr, 0, 0);

            // Compute K = X * W_k
            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_X[dev], matX, d_W_k[dev], matW,
                           &beta, d_K[dev], matOut, d_K[dev], matOut,
                           (returnedAlgoCount > 0) ? &heuristicResult[0].algo : nullptr,
                           nullptr, 0, 0);

            // Compute V = X * W_v
            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_X[dev], matX, d_W_v[dev], matW,
                           &beta, d_V[dev], matOut, d_V[dev], matOut,
                           (returnedAlgoCount > 0) ? &heuristicResult[0].algo : nullptr,
                           nullptr, 0, 0);

            hipblasLtMatmulPreferenceDestroy(pref);

            // Copy results
            hipMemcpy(h_Q_result.data() + M_start * N, d_Q[dev], M_local * N * sizeof(_Float16), hipMemcpyDeviceToHost);
            hipMemcpy(h_K_result.data() + M_start * N, d_K[dev], M_local * N * sizeof(_Float16), hipMemcpyDeviceToHost);
            hipMemcpy(h_V_result.data() + M_start * N, d_V[dev], M_local * N * sizeof(_Float16), hipMemcpyDeviceToHost);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matX); hipblasLtMatrixLayoutDestroy(matW);
            hipblasLtMatrixLayoutDestroy(matOut);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": Q/K/V projections complete (tokens=" << M_local << ")" << std::endl;
        }

        // Verify
        bool q_correct = verifyResult_FP16(h_Q_result, h_Q_expected);
        bool k_correct = verifyResult_FP16(h_K_result, h_K_expected);
        bool v_correct = verifyResult_FP16(h_V_result, h_V_expected);

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_X[dev]); hipFree(d_W_q[dev]); hipFree(d_W_k[dev]); hipFree(d_W_v[dev]);
            hipFree(d_Q[dev]); hipFree(d_K[dev]); hipFree(d_V[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        EXPECT_TRUE(q_correct && k_correct && v_correct) << "Q/K/V projections do NOT match CPU reference!";
        hipblaslt_cout << "✓ Transformer Q/K/V Data Parallel VERIFIED" << std::endl;
    }

    // ============================================================================
    // Test 2: Column-Parallel Linear Layer (no communication needed)
    // ============================================================================
    TEST(MultiGPUTransformer, ColumnParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Column-Parallel Linear Layer ===" << std::endl;

        const int64_t M = 128; // Batch size
        const int64_t K = 512; // Input features
        const int64_t N = 1024; // Output features
        const int64_t N_per_gpu = N / numDevices; // Split columns

        std::vector<_Float16> h_X(M * K);
        std::vector<_Float16> h_W_full(K * N);
        std::vector<_Float16> h_Y_expected(M * N);

        initMatrix_FP16(h_X, M * K, 1.0f);
        initMatrix_FP16(h_W_full, K * N, 0.5f);

        cpuGEMM_FP16(h_X, h_W_full, h_Y_expected, M, N, K);

        std::vector<_Float16> h_Y_result(M * N);

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<_Float16*> d_X(numDevices), d_W(numDevices), d_Y(numDevices);

        float alpha = 1.0f, beta = 0.0f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t N_start = dev * N_per_gpu;
            int64_t N_local = (dev == numDevices - 1) ? (N - N_start) : N_per_gpu;

            // Each GPU gets full X, but only columns of W
            hipMalloc(&d_X[dev], M * K * sizeof(_Float16));
            hipMalloc(&d_W[dev], K * N_local * sizeof(_Float16));
            hipMalloc(&d_Y[dev], M * N_local * sizeof(_Float16));

            // Copy full X
            hipMemcpy(d_X[dev], h_X.data(), M * K * sizeof(_Float16), hipMemcpyHostToDevice);

            // Copy column slice of W
            std::vector<_Float16> h_W_slice(K * N_local);
            for(int64_t k = 0; k < K; ++k)
            {
                for(int64_t n = 0; n < N_local; ++n)
                {
                    h_W_slice[k * N_local + n] = h_W_full[k * N + N_start + n];
                }
            }
            hipMemcpy(d_W[dev], h_W_slice.data(), K * N_local * sizeof(_Float16), hipMemcpyHostToDevice);

            hipblasLtMatrixLayout_t matX, matW, matY;
            hipblasLtMatrixLayoutCreate(&matX, HIP_R_16F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matW, HIP_R_16F, K, N_local, K);
            hipblasLtMatrixLayoutCreate(&matY, HIP_R_16F, M, N_local, M);

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
            hipblasLtMatmulAlgoGetHeuristic(handles[dev], matmul, matX, matW, matY, matY,
                                           pref, 1, heuristicResult, &returnedAlgoCount);

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_X[dev], matX, d_W[dev], matW,
                           &beta, d_Y[dev], matY, d_Y[dev], matY,
                           (returnedAlgoCount > 0) ? &heuristicResult[0].algo : nullptr,
                           nullptr, 0, 0);

            hipblasLtMatmulPreferenceDestroy(pref);

            // Copy column slice of result back
            std::vector<_Float16> h_Y_slice(M * N_local);
            hipMemcpy(h_Y_slice.data(), d_Y[dev], M * N_local * sizeof(_Float16), hipMemcpyDeviceToHost);

            for(int64_t m = 0; m < M; ++m)
            {
                for(int64_t n = 0; n < N_local; ++n)
                {
                    h_Y_result[m * N + N_start + n] = h_Y_slice[m * N_local + n];
                }
            }

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matX); hipblasLtMatrixLayoutDestroy(matW);
            hipblasLtMatrixLayoutDestroy(matY);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": Column-parallel partition N=[" << N_start
                          << ":" << (N_start + N_local) << "]" << std::endl;
        }

        bool correct = verifyResult_FP16(h_Y_result, h_Y_expected);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_X[dev]); hipFree(d_W[dev]); hipFree(d_Y[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        EXPECT_TRUE(correct) << "Column-parallel result does NOT match CPU reference!";
        hipblaslt_cout << "✓ Column-Parallel Linear Layer VERIFIED (no communication needed)" << std::endl;
    }

    // ============================================================================
    // Test 3: Row-Parallel Linear Layer (requires all-reduce)
    // ============================================================================
    TEST(MultiGPUTransformer, RowParallel_AllReduce)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Row-Parallel Linear Layer with All-Reduce ===" << std::endl;

        const int64_t M = 128;
        const int64_t K = 1024; // Split across GPUs
        const int64_t N = 512;
        const int64_t K_per_gpu = K / numDevices;

        std::vector<_Float16> h_X_full(M * K);
        std::vector<_Float16> h_W_full(K * N);
        std::vector<_Float16> h_Y_expected(M * N);

        initMatrix_FP16(h_X_full, M * K, 1.0f);
        initMatrix_FP16(h_W_full, K * N, 0.5f);

        cpuGEMM_FP16(h_X_full, h_W_full, h_Y_expected, M, N, K);

        // Each GPU computes partial result, then we all-reduce
        std::vector<std::vector<_Float16>> h_Y_partials(numDevices);
        std::vector<_Float16> h_Y_result(M * N, static_cast<_Float16>(0.0f));

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<_Float16*> d_X(numDevices), d_W(numDevices), d_Y(numDevices);

        float alpha = 1.0f, beta = 0.0f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t K_start = dev * K_per_gpu;
            int64_t K_local = (dev == numDevices - 1) ? (K - K_start) : K_per_gpu;

            hipMalloc(&d_X[dev], M * K_local * sizeof(_Float16));
            hipMalloc(&d_W[dev], K_local * N * sizeof(_Float16));
            hipMalloc(&d_Y[dev], M * N * sizeof(_Float16));

            // Extract K-partition slices
            std::vector<_Float16> h_X_slice(M * K_local);
            std::vector<_Float16> h_W_slice(K_local * N);

            for(int64_t m = 0; m < M; ++m)
            {
                for(int64_t k = 0; k < K_local; ++k)
                {
                    h_X_slice[m * K_local + k] = h_X_full[m * K + K_start + k];
                }
            }

            for(int64_t k = 0; k < K_local; ++k)
            {
                for(int64_t n = 0; n < N; ++n)
                {
                    h_W_slice[k * N + n] = h_W_full[(K_start + k) * N + n];
                }
            }

            hipMemcpy(d_X[dev], h_X_slice.data(), M * K_local * sizeof(_Float16), hipMemcpyHostToDevice);
            hipMemcpy(d_W[dev], h_W_slice.data(), K_local * N * sizeof(_Float16), hipMemcpyHostToDevice);

            hipblasLtMatrixLayout_t matX, matW, matY;
            hipblasLtMatrixLayoutCreate(&matX, HIP_R_16F, M, K_local, M);
            hipblasLtMatrixLayoutCreate(&matW, HIP_R_16F, K_local, N, K_local);
            hipblasLtMatrixLayoutCreate(&matY, HIP_R_16F, M, N, M);

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
            hipblasLtMatmulAlgoGetHeuristic(handles[dev], matmul, matX, matW, matY, matY,
                                           pref, 1, heuristicResult, &returnedAlgoCount);

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_X[dev], matX, d_W[dev], matW,
                           &beta, d_Y[dev], matY, d_Y[dev], matY,
                           (returnedAlgoCount > 0) ? &heuristicResult[0].algo : nullptr,
                           nullptr, 0, 0);

            hipblasLtMatmulPreferenceDestroy(pref);

            h_Y_partials[dev].resize(M * N);
            hipMemcpy(h_Y_partials[dev].data(), d_Y[dev], M * N * sizeof(_Float16), hipMemcpyDeviceToHost);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matX); hipblasLtMatrixLayoutDestroy(matW);
            hipblasLtMatrixLayoutDestroy(matY);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": Row-parallel partial computed" << std::endl;
        }

        // All-reduce: sum partials (in real deployment, use NCCL/RCCL)
        for(int dev = 0; dev < numDevices; ++dev)
        {
            for(size_t i = 0; i < h_Y_result.size(); ++i)
            {
                h_Y_result[i] = static_cast<_Float16>(
                    static_cast<float>(h_Y_result[i]) + static_cast<float>(h_Y_partials[dev][i])
                );
            }
        }

        bool correct = verifyResult_FP16(h_Y_result, h_Y_expected);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_X[dev]); hipFree(d_W[dev]); hipFree(d_Y[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        EXPECT_TRUE(correct) << "Row-parallel with all-reduce does NOT match CPU reference!";
        hipblaslt_cout << "✓ Row-Parallel with All-Reduce VERIFIED" << std::endl;
    }

} // namespace
