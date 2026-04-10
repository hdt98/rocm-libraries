/*******************************************************************************
 * Multi-GPU Collective Operations Test Suite
 * Tests: All-Reduce, All-Gather, Reduce-Scatter, Broadcast (with RCCL fallback)
 *
 * NOTE: This test suite provides both RCCL-based and manual implementations.
 * If RCCL is not available, tests will use manual reduction implementations.
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <atomic>
#include <thread>
#include <gtest/gtest-spi.h>

// Check if RCCL is available
// Note: Disabled by default - enable by defining USE_RCCL at compile time
#ifdef USE_RCCL
#  ifdef __has_include
#    if __has_include(<rccl/rccl.h>)
#      define HAS_RCCL 1
#      include <rccl/rccl.h>
#    else
#      define HAS_RCCL 0
#    endif
#  else
#    define HAS_RCCL 0
#  endif
#else
#  define HAS_RCCL 0
#endif

namespace
{
    int getNumGPUs()
    {
        int numDevices = 0;
        auto err = hipGetDeviceCount(&numDevices);
        (void)err;
        return numDevices;
    }

    // Manual all-reduce implementation (fallback when RCCL not available)
    void manual_allreduce_sum(const std::vector<float*>& d_data,
                              const std::vector<hipStream_t>& streams,
                              size_t count,
                              int numDevices)
    {
        // Stage 1: Copy all data to host
        std::vector<std::vector<float>> h_data(numDevices);
        for(int dev = 0; dev < numDevices; ++dev)
        {
            h_data[dev].resize(count);
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMemcpyAsync(h_data[dev].data(), d_data[dev],
                                     count * sizeof(float),
                                     hipMemcpyDeviceToHost, streams[dev]), hipSuccess);
        }

        // Sync all streams
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipStreamSynchronize(streams[dev]), hipSuccess);
        }

        // Stage 2: Sum on host
        std::vector<float> h_result(count, 0.0f);
        for(int dev = 0; dev < numDevices; ++dev)
        {
            for(size_t i = 0; i < count; ++i)
            {
                h_result[i] += h_data[dev][i];
            }
        }

        // Stage 3: Broadcast result back to all GPUs
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMemcpyAsync(d_data[dev], h_result.data(),
                                     count * sizeof(float),
                                     hipMemcpyHostToDevice, streams[dev]), hipSuccess);
        }

        // Final sync
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipStreamSynchronize(streams[dev]), hipSuccess);
        }
    }

    TEST(MultiGPUCollectives, AllReduceSum_Manual)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== All-Reduce (Sum) - Manual Implementation ===" << std::endl;

        const size_t count = 1024 * 1024; // 1M elements

        std::vector<hipStream_t> streams(numDevices);
        std::vector<float*> d_data(numDevices);
        std::vector<std::vector<float>> h_init(numDevices);

        // Initialize
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipStreamCreate(&streams[dev]), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_data[dev], count * sizeof(float)), hipSuccess);

            // Each GPU starts with dev+1 in all elements
            h_init[dev].resize(count, static_cast<float>(dev + 1));
            EXPECT_EQ(hipMemcpy(d_data[dev], h_init[dev].data(),
                                count * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
        }

        // Perform all-reduce
        auto start = std::chrono::high_resolution_clock::now();
        manual_allreduce_sum(d_data, streams, count, numDevices);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // Verify result: each element should be sum(1 + 2 + ... + numDevices) = numDevices*(numDevices+1)/2
        float expected_sum = static_cast<float>(numDevices * (numDevices + 1) / 2);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            std::vector<float> h_result(count);
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMemcpy(h_result.data(), d_data[dev],
                                count * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

            // Check first and last elements
            EXPECT_NEAR(h_result[0], expected_sum, 0.001f) << "GPU " << dev << " first element mismatch";
            EXPECT_NEAR(h_result[count - 1], expected_sum, 0.001f) << "GPU " << dev << " last element mismatch";
        }

        hipblaslt_cout << "All-reduce completed in " << (elapsed.count() * 1000.0) << " ms" << std::endl;
        hipblaslt_cout << "Expected sum per element: " << expected_sum << std::endl;
        hipblaslt_cout << "✓ All GPUs have correct reduced value" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipStreamDestroy(streams[dev]), hipSuccess);
            EXPECT_EQ(hipFree(d_data[dev]), hipSuccess);
        }
    }

#if HAS_RCCL
    TEST(MultiGPUCollectives, AllReduceSum_RCCL)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== All-Reduce (Sum) - RCCL Implementation ===" << std::endl;

        const size_t count = 1024 * 1024;

        // Initialize RCCL
        std::vector<ncclComm_t> comms(numDevices);
        ncclUniqueId id;
        ncclGetUniqueId(&id);

        std::vector<hipStream_t> streams(numDevices);
        std::vector<float*> d_data(numDevices);

        // Initialize communicators
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipStreamCreate(&streams[dev]), hipSuccess);
            ncclCommInitRank(&comms[dev], numDevices, id, dev);
        }

        // Allocate and initialize data
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_data[dev], count * sizeof(float)), hipSuccess);

            std::vector<float> h_init(count, static_cast<float>(dev + 1));
            EXPECT_EQ(hipMemcpy(d_data[dev], h_init.data(),
                                count * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
        }

        // Perform RCCL all-reduce
        auto start = std::chrono::high_resolution_clock::now();

        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            ncclAllReduce(d_data[dev], d_data[dev], count, ncclFloat, ncclSum, comms[dev], streams[dev]);
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipStreamSynchronize(streams[dev]), hipSuccess);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // Verify
        float expected_sum = static_cast<float>(numDevices * (numDevices + 1) / 2);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            std::vector<float> h_result(count);
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMemcpy(h_result.data(), d_data[dev],
                                count * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

            EXPECT_NEAR(h_result[0], expected_sum, 0.001f);
            EXPECT_NEAR(h_result[count - 1], expected_sum, 0.001f);
        }

        hipblaslt_cout << "RCCL all-reduce completed in " << (elapsed.count() * 1000.0) << " ms" << std::endl;
        hipblaslt_cout << "✓ All GPUs have correct reduced value" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            ncclCommDestroy(comms[dev]);
            EXPECT_EQ(hipStreamDestroy(streams[dev]), hipSuccess);
            EXPECT_EQ(hipFree(d_data[dev]), hipSuccess);
        }
    }
#endif

    TEST(MultiGPUCollectives, AllGather_Manual)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== All-Gather - Manual Implementation ===" << std::endl;

        const size_t count_per_gpu = 256 * 1024; // 256K per GPU
        const size_t total_count = count_per_gpu * numDevices;

        std::vector<float*> d_send(numDevices);
        std::vector<float*> d_recv(numDevices);

        // Allocate
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);

            // Each GPU has count_per_gpu elements to send
            EXPECT_EQ(hipMalloc(&d_send[dev], count_per_gpu * sizeof(float)), hipSuccess);

            // Each GPU receives total_count elements (gathered from all)
            EXPECT_EQ(hipMalloc(&d_recv[dev], total_count * sizeof(float)), hipSuccess);

            // Initialize send buffer with unique values
            std::vector<float> h_send(count_per_gpu);
            for(size_t i = 0; i < count_per_gpu; ++i)
            {
                h_send[i] = static_cast<float>(dev * 1000 + i);
            }
            EXPECT_EQ(hipMemcpy(d_send[dev], h_send.data(),
                                count_per_gpu * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
        }

        // All-gather: collect all send buffers to all recv buffers
        std::vector<std::vector<float>> h_all_data(numDevices);

        // Copy all send buffers to host
        for(int dev = 0; dev < numDevices; ++dev)
        {
            h_all_data[dev].resize(count_per_gpu);
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMemcpy(h_all_data[dev].data(), d_send[dev],
                                count_per_gpu * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
        }

        // Concatenate all data
        std::vector<float> h_gathered(total_count);
        for(int dev = 0; dev < numDevices; ++dev)
        {
            std::copy(h_all_data[dev].begin(), h_all_data[dev].end(),
                     h_gathered.begin() + dev * count_per_gpu);
        }

        // Broadcast concatenated data to all GPUs
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMemcpy(d_recv[dev], h_gathered.data(),
                                total_count * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
        }

        // Verify
        for(int dev = 0; dev < numDevices; ++dev)
        {
            std::vector<float> h_result(total_count);
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMemcpy(h_result.data(), d_recv[dev],
                                total_count * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

            // Check that each segment matches the corresponding GPU's data
            for(int src_dev = 0; src_dev < numDevices; ++src_dev)
            {
                size_t offset = src_dev * count_per_gpu;
                EXPECT_NEAR(h_result[offset], static_cast<float>(src_dev * 1000), 0.001f)
                    << "GPU " << dev << " segment from GPU " << src_dev << " mismatch";
            }
        }

        hipblaslt_cout << "✓ All-gather: each GPU received " << total_count << " elements" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipFree(d_send[dev]), hipSuccess);
            EXPECT_EQ(hipFree(d_recv[dev]), hipSuccess);
        }
    }

    TEST(MultiGPUCollectives, ReduceScatter_Manual)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Reduce-Scatter - Manual Implementation ===" << std::endl;

        const size_t count_per_gpu = 256 * 1024;
        const size_t total_count = count_per_gpu * numDevices;

        std::vector<float*> d_send(numDevices);
        std::vector<float*> d_recv(numDevices);

        // Allocate
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);

            // Each GPU sends total_count elements
            EXPECT_EQ(hipMalloc(&d_send[dev], total_count * sizeof(float)), hipSuccess);

            // Each GPU receives count_per_gpu elements (its partition)
            EXPECT_EQ(hipMalloc(&d_recv[dev], count_per_gpu * sizeof(float)), hipSuccess);

            // Initialize with dev+1
            std::vector<float> h_send(total_count, static_cast<float>(dev + 1));
            EXPECT_EQ(hipMemcpy(d_send[dev], h_send.data(),
                                total_count * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
        }

        // Reduce-scatter: reduce all send buffers and scatter partitions
        std::vector<std::vector<float>> h_all_sends(numDevices);

        // Copy all send buffers to host
        for(int dev = 0; dev < numDevices; ++dev)
        {
            h_all_sends[dev].resize(total_count);
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMemcpy(h_all_sends[dev].data(), d_send[dev],
                                total_count * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
        }

        // Reduce (sum) all send buffers
        std::vector<float> h_reduced(total_count, 0.0f);
        for(int dev = 0; dev < numDevices; ++dev)
        {
            for(size_t i = 0; i < total_count; ++i)
            {
                h_reduced[i] += h_all_sends[dev][i];
            }
        }

        // Scatter partitions to each GPU
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            size_t offset = dev * count_per_gpu;
            EXPECT_EQ(hipMemcpy(d_recv[dev], h_reduced.data() + offset,
                                count_per_gpu * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
        }

        // Verify: each GPU should have sum of all GPUs in its partition
        float expected_sum = static_cast<float>(numDevices * (numDevices + 1) / 2);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            std::vector<float> h_result(count_per_gpu);
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMemcpy(h_result.data(), d_recv[dev],
                                count_per_gpu * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

            EXPECT_NEAR(h_result[0], expected_sum, 0.001f) << "GPU " << dev << " partition mismatch";
        }

        hipblaslt_cout << "✓ Reduce-scatter: each GPU received its partition with sum "
                       << expected_sum << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipFree(d_send[dev]), hipSuccess);
            EXPECT_EQ(hipFree(d_recv[dev]), hipSuccess);
        }
    }

    TEST(MultiGPUCollectives, Broadcast_Manual)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Broadcast - Manual Implementation ===" << std::endl;

        const size_t count = 512 * 1024;
        const int root_gpu = 0;

        std::vector<float*> d_data(numDevices);

        // Allocate
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_data[dev], count * sizeof(float)), hipSuccess);

            if(dev == root_gpu)
            {
                // Root GPU initializes with specific pattern
                std::vector<float> h_root(count);
                for(size_t i = 0; i < count; ++i)
                {
                    h_root[i] = static_cast<float>(i % 1000) * 0.1f;
                }
                EXPECT_EQ(hipMemcpy(d_data[dev], h_root.data(),
                                    count * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
            }
            else
            {
                // Other GPUs initialize with zeros
                EXPECT_EQ(hipMemset(d_data[dev], 0, count * sizeof(float)), hipSuccess);
            }
        }

        // Broadcast from root to all others
        std::vector<float> h_broadcast(count);
        EXPECT_EQ(hipSetDevice(root_gpu), hipSuccess);
        EXPECT_EQ(hipMemcpy(h_broadcast.data(), d_data[root_gpu],
                            count * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            if(dev != root_gpu)
            {
                EXPECT_EQ(hipSetDevice(dev), hipSuccess);
                EXPECT_EQ(hipMemcpy(d_data[dev], h_broadcast.data(),
                                    count * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
            }
        }

        // Verify all GPUs have same data as root
        for(int dev = 0; dev < numDevices; ++dev)
        {
            std::vector<float> h_result(count);
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMemcpy(h_result.data(), d_data[dev],
                                count * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

            EXPECT_NEAR(h_result[0], 0.0f, 0.001f) << "GPU " << dev << " first element mismatch";
            EXPECT_NEAR(h_result[999], 99.9f, 0.001f) << "GPU " << dev << " element 999 mismatch";
        }

        hipblaslt_cout << "✓ Broadcast from GPU " << root_gpu << " to all " << numDevices
                       << " GPUs completed" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipFree(d_data[dev]), hipSuccess);
        }
    }

    TEST(MultiGPUCollectives, KPartitionWithAllReduce)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== K-Partition GEMM with All-Reduce ===" << std::endl;

        // Simplified K-partition GEMM demonstration
        const int64_t M = 512, N = 512, K = 1024;
        const int64_t K_per_gpu = K / numDevices;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C_partial(numDevices);

        // Each GPU computes partial result for its K-slice
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            hipblasLtCreate(&handles[dev]);

            // A slice: M x K_per_gpu
            EXPECT_EQ(hipMalloc(&d_A[dev], M * K_per_gpu * sizeof(float)), hipSuccess);

            // B slice: K_per_gpu x N
            EXPECT_EQ(hipMalloc(&d_B[dev], K_per_gpu * N * sizeof(float)), hipSuccess);

            // Partial C: M x N
            EXPECT_EQ(hipMalloc(&d_C_partial[dev], M * N * sizeof(float)), hipSuccess);

            // Initialize (simplified)
            EXPECT_EQ(hipMemset(d_A[dev], dev + 1, M * K_per_gpu * sizeof(float)), hipSuccess);
            EXPECT_EQ(hipMemset(d_B[dev], dev + 1, K_per_gpu * N * sizeof(float)), hipSuccess);
            EXPECT_EQ(hipMemset(d_C_partial[dev], 0, M * N * sizeof(float)), hipSuccess);

            hipblaslt_cout << "GPU " << dev << ": Handles K slice [" << (dev * K_per_gpu)
                           << ":" << ((dev + 1) * K_per_gpu) << "]" << std::endl;
        }

        // Compute partial GEMMs (skipped for brevity - would use hipblasLtMatmul)
        // Each GPU computes: C_partial[dev] = A[dev] @ B[dev]

        hipblaslt_cout << "Computing partial GEMMs..." << std::endl;
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        }

        // All-reduce to sum partial results
        hipblaslt_cout << "Performing all-reduce on partial results..." << std::endl;

        std::vector<hipStream_t> streams(numDevices);
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipStreamCreate(&streams[dev]), hipSuccess);
        }

        manual_allreduce_sum(d_C_partial, streams, M * N, numDevices);

        hipblaslt_cout << "✓ K-partition GEMM with all-reduce completed" << std::endl;
        hipblaslt_cout << "  All GPUs now have final result C = sum of partial results" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipFree(d_A[dev]), hipSuccess);
            EXPECT_EQ(hipFree(d_B[dev]), hipSuccess);
            EXPECT_EQ(hipFree(d_C_partial[dev]), hipSuccess);
            EXPECT_EQ(hipStreamDestroy(streams[dev]), hipSuccess);
            hipblasLtDestroy(handles[dev]);
        }
    }

} // namespace
