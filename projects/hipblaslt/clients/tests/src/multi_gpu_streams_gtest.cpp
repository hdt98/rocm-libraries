/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Stream-Based Concurrent Execution Test Suite
 * Tests: Multiple streams per GPU, stream synchronization, concurrent GEMM
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <thread>
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

    // ----------------------------------------------------------------------------
    // Test 1: Multiple Streams with Concurrent GEMM per GPU
    // ----------------------------------------------------------------------------
    TEST(MultiGPUStreams, ConcurrentStreamsPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing concurrent streams on " << numDevices << " GPUs" << std::endl;

        const int num_streams = 4;
        const int64_t M = 256;
        const int64_t N = 256;
        const int64_t K = 256;

        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " creating " << num_streams << " concurrent streams" << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Create multiple streams
            std::vector<hipStream_t> streams(num_streams);
            for(int s = 0; s < num_streams; ++s)
            {
                hipErr = hipStreamCreate(&streams[s]);
                ASSERT_EQ(hipErr, hipSuccess);
            }

            // Setup for GEMM operations
            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);
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

            // Allocate separate memory for each stream
            std::vector<float*> d_a(num_streams);
            std::vector<float*> d_b(num_streams);
            std::vector<float*> d_c(num_streams);
            std::vector<float*> d_d(num_streams);

            for(int s = 0; s < num_streams; ++s)
            {
                hipErr = hipMalloc(&d_a[s], M * K * sizeof(float));
                ASSERT_EQ(hipErr, hipSuccess);
                hipErr = hipMalloc(&d_b[s], K * N * sizeof(float));
                ASSERT_EQ(hipErr, hipSuccess);
                hipErr = hipMalloc(&d_c[s], M * N * sizeof(float));
                ASSERT_EQ(hipErr, hipSuccess);
                hipErr = hipMalloc(&d_d[s], M * N * sizeof(float));
                ASSERT_EQ(hipErr, hipSuccess);
            }

            // Initialize data
            std::vector<float> h_a(M * K, 1.0f);
            std::vector<float> h_b(K * N, 2.0f);
            std::vector<float> h_c(M * N, 0.0f);

            // Copy data to each stream's memory asynchronously
            for(int s = 0; s < num_streams; ++s)
            {
                hipErr = hipMemcpyAsync(d_a[s], h_a.data(), M * K * sizeof(float),
                                        hipMemcpyHostToDevice, streams[s]);
                ASSERT_EQ(hipErr, hipSuccess);
                hipErr = hipMemcpyAsync(d_b[s], h_b.data(), K * N * sizeof(float),
                                        hipMemcpyHostToDevice, streams[s]);
                ASSERT_EQ(hipErr, hipSuccess);
                hipErr = hipMemcpyAsync(d_c[s], h_c.data(), M * N * sizeof(float),
                                        hipMemcpyHostToDevice, streams[s]);
                ASSERT_EQ(hipErr, hipSuccess);
            }

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
                handle, matmul, matA, matB, matC, matD, pref, 1, heuristicResult, &returnedAlgoCount);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            ASSERT_GT(returnedAlgoCount, 0);

            // Allocate workspace for each stream
            std::vector<void*> d_workspace(num_streams, nullptr);
            if(heuristicResult[0].workspaceSize > 0)
            {
                for(int s = 0; s < num_streams; ++s)
                {
                    hipErr = hipMalloc(&d_workspace[s], heuristicResult[0].workspaceSize);
                    ASSERT_EQ(hipErr, hipSuccess);
                }
            }

            // Launch GEMM on all streams concurrently
            float alpha = 1.0f;
            float beta = 0.0f;
            for(int s = 0; s < num_streams; ++s)
            {
                status = hipblasLtMatmul(handle, matmul, &alpha, d_a[s], matA, d_b[s], matB,
                                         &beta, d_c[s], matC, d_d[s], matD,
                                         &heuristicResult[0].algo, d_workspace[s],
                                         heuristicResult[0].workspaceSize, streams[s]);
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            }

            // Synchronize all streams
            for(int s = 0; s < num_streams; ++s)
            {
                hipErr = hipStreamSynchronize(streams[s]);
                ASSERT_EQ(hipErr, hipSuccess);
            }

            // Verify results from all streams
            float expected = static_cast<float>(K) * 1.0f * 2.0f;
            for(int s = 0; s < num_streams; ++s)
            {
                std::vector<float> h_d(M * N);
                hipErr = hipMemcpy(h_d.data(), d_d[s], M * N * sizeof(float), hipMemcpyDeviceToHost);
                ASSERT_EQ(hipErr, hipSuccess);

                EXPECT_NEAR(h_d[0], expected, 0.01f) << "GPU " << deviceId << " stream " << s;
            }

            hipblaslt_cout << "GPU " << deviceId << " completed " << num_streams
                           << " concurrent streams successfully" << std::endl;

            // Cleanup
            for(int s = 0; s < num_streams; ++s)
            {
                if(d_workspace[s])
                    hipFree(d_workspace[s]);
                hipFree(d_a[s]);
                hipFree(d_b[s]);
                hipFree(d_c[s]);
                hipFree(d_d[s]);
                hipStreamDestroy(streams[s]);
            }

            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Concurrent streams per GPU test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 2: Stream Dependencies with Events
    // ----------------------------------------------------------------------------
    TEST(MultiGPUStreams, StreamDependenciesWithEvents)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing stream dependencies with events across " << numDevices << " GPUs" << std::endl;

        const int64_t M = 128;
        const int64_t N = 128;
        const int64_t K = 128;

        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " testing stream dependencies" << std::endl;

            // Create 3 streams with dependencies: stream1 -> stream2 -> stream3
            hipStream_t stream1, stream2, stream3;
            hipErr = hipStreamCreate(&stream1);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipStreamCreate(&stream2);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipStreamCreate(&stream3);
            ASSERT_EQ(hipErr, hipSuccess);

            // Create events for synchronization
            hipEvent_t event1, event2;
            hipErr = hipEventCreate(&event1);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipEventCreate(&event2);
            ASSERT_EQ(hipErr, hipSuccess);

            // Allocate memory
            float *d_data1, *d_data2, *d_data3;
            size_t data_size = M * N * sizeof(float);
            hipErr = hipMalloc(&d_data1, data_size);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_data2, data_size);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_data3, data_size);
            ASSERT_EQ(hipErr, hipSuccess);

            std::vector<float> h_data(M * N, 1.0f);

            // Stream 1: Initialize data
            hipErr = hipMemcpyAsync(d_data1, h_data.data(), data_size, hipMemcpyHostToDevice, stream1);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipEventRecord(event1, stream1);
            ASSERT_EQ(hipErr, hipSuccess);

            // Stream 2: Wait for stream1, then copy data
            hipErr = hipStreamWaitEvent(stream2, event1, 0);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMemcpyAsync(d_data2, d_data1, data_size, hipMemcpyDeviceToDevice, stream2);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipEventRecord(event2, stream2);
            ASSERT_EQ(hipErr, hipSuccess);

            // Stream 3: Wait for stream2, then copy data
            hipErr = hipStreamWaitEvent(stream3, event2, 0);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMemcpyAsync(d_data3, d_data2, data_size, hipMemcpyDeviceToDevice, stream3);
            ASSERT_EQ(hipErr, hipSuccess);

            // Synchronize all streams
            hipErr = hipStreamSynchronize(stream1);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipStreamSynchronize(stream2);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipStreamSynchronize(stream3);
            ASSERT_EQ(hipErr, hipSuccess);

            // Verify final data
            std::vector<float> h_result(M * N);
            hipErr = hipMemcpy(h_result.data(), d_data3, data_size, hipMemcpyDeviceToHost);
            ASSERT_EQ(hipErr, hipSuccess);

            EXPECT_NEAR(h_result[0], 1.0f, 0.01f) << "GPU " << deviceId;

            hipblaslt_cout << "GPU " << deviceId << " stream dependency chain completed successfully" << std::endl;

            // Cleanup
            hipFree(d_data1);
            hipFree(d_data2);
            hipFree(d_data3);
            hipEventDestroy(event1);
            hipEventDestroy(event2);
            hipStreamDestroy(stream1);
            hipStreamDestroy(stream2);
            hipStreamDestroy(stream3);
        }

        hipblaslt_cout << "Stream dependencies with events test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 3: Stream Priorities
    // ----------------------------------------------------------------------------
    TEST(MultiGPUStreams, StreamPriorities)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing stream priorities across " << numDevices << " GPUs" << std::endl;

        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " testing stream priorities" << std::endl;

            // Get priority range
            int leastPriority, greatestPriority;
            hipErr = hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " priority range: [" << greatestPriority
                           << " (highest), " << leastPriority << " (lowest)]" << std::endl;

            // Create streams with different priorities
            hipStream_t highPriorityStream, lowPriorityStream, normalStream;

            hipErr = hipStreamCreateWithPriority(&highPriorityStream, hipStreamDefault, greatestPriority);
            ASSERT_EQ(hipErr, hipSuccess);

            hipErr = hipStreamCreateWithPriority(&lowPriorityStream, hipStreamDefault, leastPriority);
            ASSERT_EQ(hipErr, hipSuccess);

            hipErr = hipStreamCreate(&normalStream);
            ASSERT_EQ(hipErr, hipSuccess);

            // Verify priorities
            int highPrio, lowPrio, normalPrio;
            hipErr = hipStreamGetPriority(highPriorityStream, &highPrio);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipStreamGetPriority(lowPriorityStream, &lowPrio);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipStreamGetPriority(normalStream, &normalPrio);
            ASSERT_EQ(hipErr, hipSuccess);

            EXPECT_EQ(highPrio, greatestPriority);
            EXPECT_EQ(lowPrio, leastPriority);

            hipblaslt_cout << "GPU " << deviceId << " stream priorities verified: high=" << highPrio
                           << ", normal=" << normalPrio << ", low=" << lowPrio << std::endl;

            // Cleanup
            hipStreamDestroy(highPriorityStream);
            hipStreamDestroy(lowPriorityStream);
            hipStreamDestroy(normalStream);
        }

        hipblaslt_cout << "Stream priorities test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 4: Multi-GPU with Multi-Stream Pipeline
    // ----------------------------------------------------------------------------
    TEST(MultiGPUStreams, MultiGPUMultiStreamPipeline)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing multi-GPU multi-stream pipeline with " << numDevices << " GPUs" << std::endl;

        const int num_streams_per_gpu = 2;
        const int64_t M = 128;
        const int64_t N = 128;
        const int64_t K = 128;

        // Run concurrent operations on multiple GPUs, each with multiple streams
        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            std::vector<hipStream_t> streams(num_streams_per_gpu);
            std::vector<float*> d_a(num_streams_per_gpu);
            std::vector<float*> d_result(num_streams_per_gpu);

            // Create streams and allocate memory
            for(int s = 0; s < num_streams_per_gpu; ++s)
            {
                hipErr = hipStreamCreate(&streams[s]);
                ASSERT_EQ(hipErr, hipSuccess);

                hipErr = hipMalloc(&d_a[s], M * K * sizeof(float));
                ASSERT_EQ(hipErr, hipSuccess);
                hipErr = hipMalloc(&d_result[s], M * N * sizeof(float));
                ASSERT_EQ(hipErr, hipSuccess);
            }

            // Launch operations on all streams
            std::vector<float> h_a(M * K, static_cast<float>(deviceId + 1));
            for(int s = 0; s < num_streams_per_gpu; ++s)
            {
                hipErr = hipMemcpyAsync(d_a[s], h_a.data(), M * K * sizeof(float),
                                        hipMemcpyHostToDevice, streams[s]);
                ASSERT_EQ(hipErr, hipSuccess);
            }

            // Synchronize all streams
            for(int s = 0; s < num_streams_per_gpu; ++s)
            {
                hipErr = hipStreamSynchronize(streams[s]);
                ASSERT_EQ(hipErr, hipSuccess);
            }

            // Cleanup
            for(int s = 0; s < num_streams_per_gpu; ++s)
            {
                hipFree(d_a[s]);
                hipFree(d_result[s]);
                hipStreamDestroy(streams[s]);
            }

            hipblaslt_cout << "GPU " << deviceId << " pipeline with " << num_streams_per_gpu
                           << " streams completed" << std::endl;
        }

        hipblaslt_cout << "Multi-GPU multi-stream pipeline test passed" << std::endl;
    }

} // namespace
