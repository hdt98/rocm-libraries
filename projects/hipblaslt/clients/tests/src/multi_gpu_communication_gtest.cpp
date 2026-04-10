/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Communication Patterns Test Suite
 * Tests: GPU-to-GPU transfers, All-reduce, All-gather, Ring communication
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <numeric>

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
    // Test 1: Direct GPU-to-GPU Memory Copy
    // ----------------------------------------------------------------------------
    TEST(MultiGPUCommunication, DirectGPUtoGPUCopy)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing direct GPU-to-GPU memory copy" << std::endl;

        const size_t data_size = 1024 * 1024; // 1M floats
        const size_t bytes = data_size * sizeof(float);

        // Allocate on GPU 0
        auto hipErr = hipSetDevice(0);
        ASSERT_EQ(hipErr, hipSuccess);

        float *d_src;
        hipErr = hipMalloc(&d_src, bytes);
        ASSERT_EQ(hipErr, hipSuccess);

        // Initialize data on GPU 0
        std::vector<float> h_data(data_size);
        for(size_t i = 0; i < data_size; ++i)
            h_data[i] = static_cast<float>(i);

        hipErr = hipMemcpy(d_src, h_data.data(), bytes, hipMemcpyHostToDevice);
        ASSERT_EQ(hipErr, hipSuccess);

        // Allocate on GPU 1
        hipErr = hipSetDevice(1);
        ASSERT_EQ(hipErr, hipSuccess);

        float *d_dst;
        hipErr = hipMalloc(&d_dst, bytes);
        ASSERT_EQ(hipErr, hipSuccess);

        // Check P2P capability
        int canAccessPeer = 0;
        hipErr = hipDeviceCanAccessPeer(&canAccessPeer, 1, 0);
        ASSERT_EQ(hipErr, hipSuccess);

        if(canAccessPeer)
        {
            // Enable P2P
            hipErr = hipDeviceEnablePeerAccess(0, 0);
            // May already be enabled, so don't assert

            // Copy from GPU 0 to GPU 1 using P2P
            hipErr = hipMemcpyPeer(d_dst, 1, d_src, 0, bytes);
            ASSERT_EQ(hipErr, hipSuccess);

            // Verify data on GPU 1
            std::vector<float> h_result(data_size);
            hipErr = hipMemcpy(h_result.data(), d_dst, bytes, hipMemcpyDeviceToHost);
            ASSERT_EQ(hipErr, hipSuccess);

            for(size_t i = 0; i < std::min(size_t(100), data_size); ++i)
            {
                EXPECT_FLOAT_EQ(h_result[i], h_data[i]) << "Mismatch at index " << i;
            }

            hipblaslt_cout << "P2P copy successful: " << (bytes / 1024 / 1024) << " MB transferred" << std::endl;
        }
        else
        {
            hipblaslt_cout << "P2P not supported, using staged copy via host" << std::endl;

            // Copy via host memory
            std::vector<float> h_temp(data_size);
            hipErr = hipSetDevice(0);
            hipErr = hipMemcpy(h_temp.data(), d_src, bytes, hipMemcpyDeviceToHost);
            ASSERT_EQ(hipErr, hipSuccess);

            hipErr = hipSetDevice(1);
            hipErr = hipMemcpy(d_dst, h_temp.data(), bytes, hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "Staged copy via host successful" << std::endl;
        }

        // Cleanup
        hipErr = hipSetDevice(0);
        hipFree(d_src);
        hipErr = hipSetDevice(1);
        hipFree(d_dst);

        hipblaslt_cout << "Direct GPU-to-GPU copy test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 2: Ring Communication Pattern
    // ----------------------------------------------------------------------------
    TEST(MultiGPUCommunication, RingCommunicationPattern)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing ring communication pattern across " << numDevices << " GPUs" << std::endl;

        const size_t chunk_size = 256 * 1024; // 256K floats per chunk
        const size_t bytes = chunk_size * sizeof(float);

        std::vector<float*> d_data(numDevices);
        std::vector<float*> d_recv(numDevices);

        // Allocate on each GPU
        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipErr = hipMalloc(&d_data[dev], bytes);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_recv[dev], bytes);
            ASSERT_EQ(hipErr, hipSuccess);

            // Initialize with device-specific data
            std::vector<float> h_init(chunk_size, static_cast<float>(dev));
            hipErr = hipMemcpy(d_data[dev], h_init.data(), bytes, hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);
        }

        // Ring communication: each GPU sends to next GPU
        // GPU 0 -> GPU 1 -> GPU 2 -> ... -> GPU 0
        for(int dev = 0; dev < numDevices; ++dev)
        {
            int next_dev = (dev + 1) % numDevices;

            hipblaslt_cout << "Ring step: GPU " << dev << " -> GPU " << next_dev << std::endl;

            // Check P2P
            int canAccess = 0;
            auto hipErr = hipDeviceCanAccessPeer(&canAccess, next_dev, dev);
            ASSERT_EQ(hipErr, hipSuccess);

            if(canAccess)
            {
                hipErr = hipSetDevice(next_dev);
                ASSERT_EQ(hipErr, hipSuccess);
                hipErr = hipDeviceEnablePeerAccess(dev, 0);
                // May already be enabled

                // P2P copy
                hipErr = hipMemcpyPeer(d_recv[next_dev], next_dev, d_data[dev], dev, bytes);
                ASSERT_EQ(hipErr, hipSuccess);
            }
            else
            {
                // Copy via host
                std::vector<float> h_temp(chunk_size);
                hipErr = hipSetDevice(dev);
                hipErr = hipMemcpy(h_temp.data(), d_data[dev], bytes, hipMemcpyDeviceToHost);
                ASSERT_EQ(hipErr, hipSuccess);

                hipErr = hipSetDevice(next_dev);
                hipErr = hipMemcpy(d_recv[next_dev], h_temp.data(), bytes, hipMemcpyHostToDevice);
                ASSERT_EQ(hipErr, hipSuccess);
            }
        }

        // Verify: each GPU should have received data from previous GPU
        for(int dev = 0; dev < numDevices; ++dev)
        {
            int prev_dev = (dev - 1 + numDevices) % numDevices;

            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            std::vector<float> h_recv(chunk_size);
            hipErr = hipMemcpy(h_recv.data(), d_recv[dev], bytes, hipMemcpyDeviceToHost);
            ASSERT_EQ(hipErr, hipSuccess);

            // Should contain data from previous device
            EXPECT_FLOAT_EQ(h_recv[0], static_cast<float>(prev_dev))
                << "GPU " << dev << " should have data from GPU " << prev_dev;
        }

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            hipFree(d_data[dev]);
            hipFree(d_recv[dev]);
        }

        hipblaslt_cout << "Ring communication pattern test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 3: All-Reduce Sum Pattern (simplified)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUCommunication, AllReduceSum)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing all-reduce sum across " << numDevices << " GPUs" << std::endl;

        const size_t data_size = 1024;
        const size_t bytes = data_size * sizeof(float);

        std::vector<float*> d_data(numDevices);

        // Initialize each GPU with its own data
        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipErr = hipMalloc(&d_data[dev], bytes);
            ASSERT_EQ(hipErr, hipSuccess);

            // Initialize with value = device_id
            std::vector<float> h_data(data_size, static_cast<float>(dev));
            hipErr = hipMemcpy(d_data[dev], h_data.data(), bytes, hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);
        }

        // Simplified all-reduce: gather all to GPU 0, compute sum, broadcast back
        std::vector<std::vector<float>> gathered_data(numDevices);

        // Gather to GPU 0
        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            gathered_data[dev].resize(data_size);
            hipErr = hipMemcpy(gathered_data[dev].data(), d_data[dev], bytes, hipMemcpyDeviceToHost);
            ASSERT_EQ(hipErr, hipSuccess);
        }

        // Compute sum on host (in real scenario, this would be on GPU)
        std::vector<float> sum_result(data_size, 0.0f);
        for(int dev = 0; dev < numDevices; ++dev)
        {
            for(size_t i = 0; i < data_size; ++i)
            {
                sum_result[i] += gathered_data[dev][i];
            }
        }

        // Expected sum: 0 + 1 + 2 + ... + (numDevices-1) for each element
        float expected_sum = 0.0f;
        for(int dev = 0; dev < numDevices; ++dev)
            expected_sum += dev;

        EXPECT_FLOAT_EQ(sum_result[0], expected_sum);

        // Broadcast result back to all GPUs
        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipErr = hipMemcpy(d_data[dev], sum_result.data(), bytes, hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);
        }

        // Verify all GPUs have the same result
        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            std::vector<float> h_result(data_size);
            hipErr = hipMemcpy(h_result.data(), d_data[dev], bytes, hipMemcpyDeviceToHost);
            ASSERT_EQ(hipErr, hipSuccess);

            EXPECT_FLOAT_EQ(h_result[0], expected_sum) << "GPU " << dev;
        }

        hipblaslt_cout << "All-reduce sum completed: sum = " << expected_sum << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            hipFree(d_data[dev]);
        }

        hipblaslt_cout << "All-reduce sum test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 4: Broadcast Pattern
    // ----------------------------------------------------------------------------
    TEST(MultiGPUCommunication, BroadcastPattern)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing broadcast pattern from GPU 0 to all GPUs" << std::endl;

        const size_t data_size = 512 * 1024; // 512K floats
        const size_t bytes = data_size * sizeof(float);

        // Initialize data on GPU 0
        auto hipErr = hipSetDevice(0);
        ASSERT_EQ(hipErr, hipSuccess);

        float *d_src;
        hipErr = hipMalloc(&d_src, bytes);
        ASSERT_EQ(hipErr, hipSuccess);

        std::vector<float> h_src(data_size);
        for(size_t i = 0; i < data_size; ++i)
            h_src[i] = static_cast<float>(i % 1000);

        hipErr = hipMemcpy(d_src, h_src.data(), bytes, hipMemcpyHostToDevice);
        ASSERT_EQ(hipErr, hipSuccess);

        // Broadcast to all other GPUs
        std::vector<float*> d_dst(numDevices, nullptr);

        for(int dev = 1; dev < numDevices; ++dev)
        {
            hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipErr = hipMalloc(&d_dst[dev], bytes);
            ASSERT_EQ(hipErr, hipSuccess);

            // Check P2P
            int canAccess = 0;
            hipErr = hipDeviceCanAccessPeer(&canAccess, dev, 0);
            ASSERT_EQ(hipErr, hipSuccess);

            if(canAccess)
            {
                hipErr = hipDeviceEnablePeerAccess(0, 0);
                // May already be enabled

                // P2P copy from GPU 0
                hipErr = hipMemcpyPeer(d_dst[dev], dev, d_src, 0, bytes);
                ASSERT_EQ(hipErr, hipSuccess);

                hipblaslt_cout << "Broadcast GPU 0 -> GPU " << dev << " via P2P" << std::endl;
            }
            else
            {
                // Copy via host
                hipErr = hipMemcpy(d_dst[dev], h_src.data(), bytes, hipMemcpyHostToDevice);
                ASSERT_EQ(hipErr, hipSuccess);

                hipblaslt_cout << "Broadcast GPU 0 -> GPU " << dev << " via host" << std::endl;
            }
        }

        // Verify all GPUs have the same data
        for(int dev = 1; dev < numDevices; ++dev)
        {
            hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            std::vector<float> h_dst(data_size);
            hipErr = hipMemcpy(h_dst.data(), d_dst[dev], bytes, hipMemcpyDeviceToHost);
            ASSERT_EQ(hipErr, hipSuccess);

            for(size_t i = 0; i < std::min(size_t(100), data_size); ++i)
            {
                EXPECT_FLOAT_EQ(h_dst[i], h_src[i]) << "GPU " << dev << " index " << i;
            }
        }

        // Cleanup
        hipErr = hipSetDevice(0);
        hipFree(d_src);
        for(int dev = 1; dev < numDevices; ++dev)
        {
            hipErr = hipSetDevice(dev);
            if(d_dst[dev])
                hipFree(d_dst[dev]);
        }

        hipblaslt_cout << "Broadcast pattern test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 5: Scatter-Gather Pattern
    // ----------------------------------------------------------------------------
    TEST(MultiGPUCommunication, ScatterGatherPattern)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing scatter-gather pattern across " << numDevices << " GPUs" << std::endl;

        const size_t chunk_size = 256 * 1024; // 256K floats per GPU
        const size_t total_size = chunk_size * numDevices;
        const size_t chunk_bytes = chunk_size * sizeof(float);

        // Create full dataset on host
        std::vector<float> h_full_data(total_size);
        for(size_t i = 0; i < total_size; ++i)
            h_full_data[i] = static_cast<float>(i);

        // SCATTER: Distribute chunks to each GPU
        std::vector<float*> d_chunks(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipErr = hipMalloc(&d_chunks[dev], chunk_bytes);
            ASSERT_EQ(hipErr, hipSuccess);

            // Copy this GPU's chunk
            const float* chunk_start = h_full_data.data() + dev * chunk_size;
            hipErr = hipMemcpy(d_chunks[dev], chunk_start, chunk_bytes, hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "Scattered chunk " << dev << " to GPU " << dev << std::endl;
        }

        // GATHER: Collect chunks back from all GPUs
        std::vector<float> h_gathered(total_size);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            float* gather_start = h_gathered.data() + dev * chunk_size;
            hipErr = hipMemcpy(gather_start, d_chunks[dev], chunk_bytes, hipMemcpyDeviceToHost);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "Gathered chunk " << dev << " from GPU " << dev << std::endl;
        }

        // Verify gathered data matches original
        for(size_t i = 0; i < std::min(size_t(1000), total_size); ++i)
        {
            EXPECT_FLOAT_EQ(h_gathered[i], h_full_data[i]) << "Mismatch at index " << i;
        }

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            hipFree(d_chunks[dev]);
        }

        hipblaslt_cout << "Scatter-gather pattern test passed" << std::endl;
    }

} // namespace
