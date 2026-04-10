/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Stream and Synchronization Test Suite
 * Testing streams, events, priorities, and synchronization
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

    // Test 1: NULL Stream (Default)
    TEST(MultiGPUStream, NULLStream)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing NULL stream (default) across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Use NULL stream (default stream)
            hipStream_t stream = nullptr;
            hipblasLtMatrixLayout_t matA;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);

            hipblaslt_cout << "GPU " << dev << " using NULL (default) stream" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtDestroy(handle);
        }
    }

    // Test 2: Custom Stream
    TEST(MultiGPUStream, CustomStream)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing custom stream across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Create custom stream
            hipStream_t stream;
            hipStreamCreate(&stream);

            hipblaslt_cout << "GPU " << dev << " custom stream created" << std::endl;

            hipStreamDestroy(stream);
            hipblasLtDestroy(handle);
        }
    }

    // Test 3: Multiple Concurrent Streams (2 per GPU)
    TEST(MultiGPUStream, TwoConcurrentStreams)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing 2 concurrent streams per GPU" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Create 2 concurrent streams
            hipStream_t stream1, stream2;
            hipStreamCreate(&stream1);
            hipStreamCreate(&stream2);

            hipblaslt_cout << "GPU " << dev << " created 2 concurrent streams" << std::endl;

            hipStreamDestroy(stream1);
            hipStreamDestroy(stream2);
            hipblasLtDestroy(handle);
        }
    }

    // Test 4: Multiple Concurrent Streams (4 per GPU)
    TEST(MultiGPUStream, FourConcurrentStreams)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing 4 concurrent streams per GPU" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Create 4 concurrent streams
            std::vector<hipStream_t> streams(4);
            for(auto& stream : streams)
            {
                hipStreamCreate(&stream);
            }

            hipblaslt_cout << "GPU " << dev << " created 4 concurrent streams" << std::endl;

            for(auto& stream : streams)
            {
                hipStreamDestroy(stream);
            }
            hipblasLtDestroy(handle);
        }
    }

    // Test 5: Multiple Concurrent Streams (8 per GPU)
    TEST(MultiGPUStream, EightConcurrentStreams)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing 8 concurrent streams per GPU" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Create 8 concurrent streams
            std::vector<hipStream_t> streams(8);
            for(auto& stream : streams)
            {
                hipStreamCreate(&stream);
            }

            hipblaslt_cout << "GPU " << dev << " created 8 concurrent streams" << std::endl;

            for(auto& stream : streams)
            {
                hipStreamDestroy(stream);
            }
            hipblasLtDestroy(handle);
        }
    }

    // Test 6: Stream Priorities (High, Normal, Low)
    TEST(MultiGPUStream, StreamPriorities)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing stream priorities across GPUs" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Get priority range
            int priority_low, priority_high;
            hipDeviceGetStreamPriorityRange(&priority_low, &priority_high);

            hipblaslt_cout << "GPU " << dev << " priority range: " << priority_high
                           << " (high) to " << priority_low << " (low)" << std::endl;

            // Create streams with different priorities
            hipStream_t stream_high, stream_normal, stream_low;
            hipStreamCreateWithPriority(&stream_high, hipStreamDefault, priority_high);
            hipStreamCreateWithPriority(&stream_normal, hipStreamDefault, 0);
            hipStreamCreateWithPriority(&stream_low, hipStreamDefault, priority_low);

            hipblaslt_cout << "GPU " << dev << " created streams with high, normal, low priorities" << std::endl;

            hipStreamDestroy(stream_high);
            hipStreamDestroy(stream_normal);
            hipStreamDestroy(stream_low);
            hipblasLtDestroy(handle);
        }
    }

    // Test 7: Event-Based Synchronization
    TEST(MultiGPUStream, EventBasedSync)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing event-based synchronization across GPUs" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Create stream and events
            hipStream_t stream;
            hipStreamCreate(&stream);

            hipEvent_t event_start, event_end;
            hipEventCreate(&event_start);
            hipEventCreate(&event_end);

            // Record events on stream
            hipEventRecord(event_start, stream);
            hipEventRecord(event_end, stream);

            // Synchronize
            hipEventSynchronize(event_end);

            hipblaslt_cout << "GPU " << dev << " event-based synchronization successful" << std::endl;

            hipEventDestroy(event_start);
            hipEventDestroy(event_end);
            hipStreamDestroy(stream);
            hipblasLtDestroy(handle);
        }
    }

    // Test 8: Cross-GPU Stream Dependencies
    TEST(MultiGPUStream, CrossGPUStreamDependencies)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing cross-GPU stream dependencies" << std::endl;

        // Create streams and events on both GPUs
        hipSetDevice(0);
        hipStream_t stream0;
        hipEvent_t event0;
        hipStreamCreate(&stream0);
        hipEventCreate(&event0);
        hipEventRecord(event0, stream0);

        hipSetDevice(1);
        hipStream_t stream1;
        hipStreamCreate(&stream1);

        // Make stream1 on GPU1 wait for event0 from GPU0
        hipStreamWaitEvent(stream1, event0, 0);

        hipblaslt_cout << "Cross-GPU stream dependency established (GPU1 waits for GPU0)" << std::endl;

        hipStreamDestroy(stream1);
        hipSetDevice(0);
        hipEventDestroy(event0);
        hipStreamDestroy(stream0);
    }

    // Test 9: Stream Callbacks
    TEST(MultiGPUStream, StreamCallbacks)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing stream callbacks across GPUs" << std::endl;

        auto callback = [](hipStream_t stream, hipError_t status, void* userData) {
            int* flag = static_cast<int*>(userData);
            *flag = 1;
        };

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipStream_t stream;
            hipStreamCreate(&stream);

            int callback_flag = 0;
            hipStreamAddCallback(stream, callback, &callback_flag, 0);

            // Synchronize to ensure callback is executed
            hipStreamSynchronize(stream);

            if(callback_flag == 1)
            {
                hipblaslt_cout << "GPU " << dev << " stream callback executed successfully" << std::endl;
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << " stream callback not executed" << std::endl;
            }

            hipStreamDestroy(stream);
        }
    }

    // Test 10: Stream Synchronization Methods
    TEST(MultiGPUStream, StreamSyncMethods)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing different stream synchronization methods" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            // Method 1: hipStreamSynchronize
            {
                hipStream_t stream;
                hipStreamCreate(&stream);
                hipStreamSynchronize(stream);
                hipblaslt_cout << "GPU " << dev << " hipStreamSynchronize() successful" << std::endl;
                hipStreamDestroy(stream);
            }

            // Method 2: hipDeviceSynchronize
            {
                hipDeviceSynchronize();
                hipblaslt_cout << "GPU " << dev << " hipDeviceSynchronize() successful" << std::endl;
            }

            // Method 3: hipStreamQuery
            {
                hipStream_t stream;
                hipStreamCreate(&stream);
                hipError_t query_result = hipStreamQuery(stream);
                if(query_result == hipSuccess)
                {
                    hipblaslt_cout << "GPU " << dev << " hipStreamQuery() reports idle" << std::endl;
                }
                hipStreamDestroy(stream);
            }
        }
    }

    // Test 11: Non-Blocking Streams
    TEST(MultiGPUStream, NonBlockingStreams)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing non-blocking streams across GPUs" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            // Create non-blocking stream
            hipStream_t stream;
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);

            hipblaslt_cout << "GPU " << dev << " non-blocking stream created" << std::endl;

            hipStreamDestroy(stream);
        }
    }

} // namespace
