/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Scaling Efficiency Tests
 * Tests strong scaling, weak scaling, and communication overhead profiling
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <cmath>
#include <chrono>

#include <gtest/gtest-spi.h>

namespace
{
    int getNumGPUs()
    {
        int numDevices = 0;
        hipGetDeviceCount(&numDevices);
        return numDevices;
    }

    void initMatrix_FP16(std::vector<_Float16>& mat, int64_t size, float base = 1.0f)
    {
        for(int64_t i = 0; i < size; ++i)
        {
            mat[i] = static_cast<_Float16>(base + (i % 100) * 0.01f);
        }
    }

    // ============================================================================
    // Test 1: Strong Scaling - Fixed problem size, vary GPU count
    // ============================================================================
    TEST(MultiGPUScaling, StrongScaling_DataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Strong Scaling Analysis ===" << std::endl;

        const int64_t M = 2048, N = 2048, K = 2048;
        const int64_t total_batches = 16; // Fixed total work
        float alpha = 1.0f, beta = 0.0f;

        std::vector<_Float16> h_A(M * K * total_batches);
        std::vector<_Float16> h_B(K * N * total_batches);

        initMatrix_FP16(h_A, M * K * total_batches, 1.0f);
        initMatrix_FP16(h_B, K * N * total_batches, 2.0f);

        // Measure time for different GPU counts: 1, 2, 4, ... up to numDevices
        std::vector<int> gpu_counts;
        for(int g = 1; g <= numDevices; g *= 2)
        {
            gpu_counts.push_back(g);
        }
        if(gpu_counts.back() != numDevices)
        {
            gpu_counts.push_back(numDevices);
        }

        std::vector<double> execution_times;

        for(int num_gpus : gpu_counts)
        {
            int64_t batches_per_gpu = total_batches / num_gpus;
            if(batches_per_gpu == 0) continue;

            std::vector<hipblasLtHandle_t> handles(num_gpus);
            std::vector<_Float16*> d_A(num_gpus), d_B(num_gpus), d_C(num_gpus);
            std::vector<hipEvent_t> start_events(num_gpus), stop_events(num_gpus);

            for(int dev = 0; dev < num_gpus; ++dev)
            {
                hipSetDevice(dev);
                hipblasLtCreate(&handles[dev]);

                hipEventCreate(&start_events[dev]);
                hipEventCreate(&stop_events[dev]);

                hipMalloc(&d_A[dev], M * K * batches_per_gpu * sizeof(_Float16));
                hipMalloc(&d_B[dev], K * N * batches_per_gpu * sizeof(_Float16));
                hipMalloc(&d_C[dev], M * N * batches_per_gpu * sizeof(_Float16));

                int64_t batch_start = dev * batches_per_gpu;
                hipMemcpy(d_A[dev], h_A.data() + batch_start * M * K,
                         M * K * batches_per_gpu * sizeof(_Float16), hipMemcpyHostToDevice);
                hipMemcpy(d_B[dev], h_B.data() + batch_start * K * N,
                         K * N * batches_per_gpu * sizeof(_Float16), hipMemcpyHostToDevice);
            }

            // Start timing
            auto start_cpu = std::chrono::high_resolution_clock::now();

            for(int dev = 0; dev < num_gpus; ++dev)
            {
                hipSetDevice(dev);
                hipEventRecord(start_events[dev]);
            }

            // Execute on all GPUs
            for(int dev = 0; dev < num_gpus; ++dev)
            {
                hipSetDevice(dev);

                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, M, N, M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, M, N, M);

                hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                 &batches_per_gpu, sizeof(batches_per_gpu));
                hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                 &batches_per_gpu, sizeof(batches_per_gpu));
                hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                 &batches_per_gpu, sizeof(batches_per_gpu));
                hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                 &batches_per_gpu, sizeof(batches_per_gpu));

                int64_t stride = M * K, stride_b = K * N, stride_c = M * N;
                hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                 &stride, sizeof(stride));
                hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                 &stride_b, sizeof(stride_b));
                hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                 &stride_c, sizeof(stride_c));
                hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                 &stride_c, sizeof(stride_c));

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmul(handles[dev], matmul, &alpha,
                               d_A[dev], matA, d_B[dev], matB,
                               &beta, d_C[dev], matC, d_C[dev], matD,
                               nullptr, nullptr, 0, 0);

                hipEventRecord(stop_events[dev]);

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }

            // Synchronize all GPUs
            for(int dev = 0; dev < num_gpus; ++dev)
            {
                hipSetDevice(dev);
                hipEventSynchronize(stop_events[dev]);
            }

            auto end_cpu = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
            execution_times.push_back(time_ms);

            // Cleanup
            for(int dev = 0; dev < num_gpus; ++dev)
            {
                hipSetDevice(dev);
                hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
                hipEventDestroy(start_events[dev]);
                hipEventDestroy(stop_events[dev]);
                hipblasLtDestroy(handles[dev]);
            }

            hipblaslt_cout << "GPUs: " << num_gpus << " | Time: " << time_ms << " ms" << std::endl;
        }

        // Calculate speedup and efficiency
        double baseline_time = execution_times[0];
        hipblaslt_cout << "\n=== Strong Scaling Results ===" << std::endl;
        for(size_t i = 0; i < gpu_counts.size(); ++i)
        {
            double speedup = baseline_time / execution_times[i];
            double efficiency = speedup / gpu_counts[i] * 100.0;
            hipblaslt_cout << "GPUs: " << gpu_counts[i]
                          << " | Speedup: " << speedup << "x"
                          << " | Efficiency: " << efficiency << "%" << std::endl;
        }

        // Test passes if we get some reasonable speedup
        double final_speedup = baseline_time / execution_times.back();
        EXPECT_GT(final_speedup, 1.5) << "Multi-GPU speedup too low!";
        hipblaslt_cout << "✓ Strong Scaling Test Complete" << std::endl;
    }

    // ============================================================================
    // Test 2: Weak Scaling - Problem size scales with GPU count
    // ============================================================================
    TEST(MultiGPUScaling, WeakScaling_DataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Weak Scaling Analysis ===" << std::endl;

        const int64_t M_base = 1024, N = 1024, K = 1024;
        const int64_t batches_per_gpu = 4; // Constant per-GPU work
        float alpha = 1.0f, beta = 0.0f;

        std::vector<int> gpu_counts;
        for(int g = 1; g <= numDevices; g *= 2)
        {
            gpu_counts.push_back(g);
        }
        if(gpu_counts.back() != numDevices)
        {
            gpu_counts.push_back(numDevices);
        }

        std::vector<double> execution_times;

        for(int num_gpus : gpu_counts)
        {
            int64_t total_batches = num_gpus * batches_per_gpu;

            std::vector<_Float16> h_A(M_base * K * total_batches);
            std::vector<_Float16> h_B(K * N * total_batches);
            initMatrix_FP16(h_A, M_base * K * total_batches, 1.0f);
            initMatrix_FP16(h_B, K * N * total_batches, 2.0f);

            std::vector<hipblasLtHandle_t> handles(num_gpus);
            std::vector<_Float16*> d_A(num_gpus), d_B(num_gpus), d_C(num_gpus);

            for(int dev = 0; dev < num_gpus; ++dev)
            {
                hipSetDevice(dev);
                hipblasLtCreate(&handles[dev]);

                hipMalloc(&d_A[dev], M_base * K * batches_per_gpu * sizeof(_Float16));
                hipMalloc(&d_B[dev], K * N * batches_per_gpu * sizeof(_Float16));
                hipMalloc(&d_C[dev], M_base * N * batches_per_gpu * sizeof(_Float16));

                int64_t batch_start = dev * batches_per_gpu;
                hipMemcpy(d_A[dev], h_A.data() + batch_start * M_base * K,
                         M_base * K * batches_per_gpu * sizeof(_Float16), hipMemcpyHostToDevice);
                hipMemcpy(d_B[dev], h_B.data() + batch_start * K * N,
                         K * N * batches_per_gpu * sizeof(_Float16), hipMemcpyHostToDevice);
            }

            auto start = std::chrono::high_resolution_clock::now();

            for(int dev = 0; dev < num_gpus; ++dev)
            {
                hipSetDevice(dev);

                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, M_base, K, M_base);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, M_base, N, M_base);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, M_base, N, M_base);

                hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                 &batches_per_gpu, sizeof(batches_per_gpu));
                hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                 &batches_per_gpu, sizeof(batches_per_gpu));
                hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                 &batches_per_gpu, sizeof(batches_per_gpu));
                hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                 &batches_per_gpu, sizeof(batches_per_gpu));

                int64_t stride = M_base * K, stride_b = K * N, stride_c = M_base * N;
                hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                 &stride, sizeof(stride));
                hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                 &stride_b, sizeof(stride_b));
                hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                 &stride_c, sizeof(stride_c));
                hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                 &stride_c, sizeof(stride_c));

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmul(handles[dev], matmul, &alpha,
                               d_A[dev], matA, d_B[dev], matB,
                               &beta, d_C[dev], matC, d_C[dev], matD,
                               nullptr, nullptr, 0, 0);

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }

            for(int dev = 0; dev < num_gpus; ++dev)
            {
                hipSetDevice(dev);
                hipDeviceSynchronize();
            }

            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            execution_times.push_back(time_ms);

            for(int dev = 0; dev < num_gpus; ++dev)
            {
                hipSetDevice(dev);
                hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
                hipblasLtDestroy(handles[dev]);
            }

            hipblaslt_cout << "GPUs: " << num_gpus << " | Total Work: " << total_batches
                          << " batches | Time: " << time_ms << " ms" << std::endl;
        }

        hipblaslt_cout << "\n=== Weak Scaling Results ===" << std::endl;
        double baseline_time = execution_times[0];
        for(size_t i = 0; i < gpu_counts.size(); ++i)
        {
            double relative_time = execution_times[i] / baseline_time;
            hipblaslt_cout << "GPUs: " << gpu_counts[i]
                          << " | Relative Time: " << relative_time
                          << " (ideal=1.0)" << std::endl;
        }

        // In weak scaling, time should remain relatively constant
        double time_variation = execution_times.back() / baseline_time;
        EXPECT_LT(time_variation, 2.0) << "Weak scaling shows poor performance!";
        hipblaslt_cout << "✓ Weak Scaling Test Complete" << std::endl;
    }

} // namespace
