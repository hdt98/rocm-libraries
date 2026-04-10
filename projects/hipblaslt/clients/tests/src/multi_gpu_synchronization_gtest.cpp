/*******************************************************************************
 * Multi-GPU Synchronization Primitives Test Suite
 * Tests: Global barriers, hierarchical sync, lock-free primitives
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <gtest/gtest-spi.h>

namespace
{
    int getNumGPUs()
    {
        int numDevices = 0;
        auto err = hipGetDeviceCount(&numDevices);
        (void)err;
        return numDevices;
    }

    // Global barrier for multi-GPU synchronization
    class GlobalBarrier
    {
    private:
        std::atomic<int> arrived;
        std::atomic<int> generation;
        int num_participants;
        std::mutex mtx;
        std::condition_variable cv;

    public:
        GlobalBarrier(int participants) : arrived(0), generation(0), num_participants(participants) {}

        void wait()
        {
            int gen = generation.load();
            int arr = arrived.fetch_add(1) + 1;

            if(arr == num_participants)
            {
                // Last thread to arrive - reset and notify all
                arrived.store(0);
                generation.fetch_add(1);
                cv.notify_all();
            }
            else
            {
                // Wait for barrier to open
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this, gen]() { return generation.load() != gen; });
            }
        }

        void reset(int participants)
        {
            arrived.store(0);
            generation.store(0);
            num_participants = participants;
        }
    };

    // Hierarchical barrier (NUMA-aware)
    class HierarchicalBarrier
    {
    private:
        struct Group
        {
            std::atomic<int> arrived;
            std::atomic<bool> ready;
            std::mutex mtx;
            std::condition_variable cv;
            int num_members;
        };

        std::vector<Group> groups;
        GlobalBarrier global_barrier;
        std::vector<int> thread_to_group;

    public:
        HierarchicalBarrier(const std::vector<int>& group_sizes)
            : groups(group_sizes.size()), global_barrier(group_sizes.size())
        {
            int total_threads = 0;
            for(size_t g = 0; g < group_sizes.size(); ++g)
            {
                groups[g].arrived = 0;
                groups[g].ready = false;
                groups[g].num_members = group_sizes[g];
                total_threads += group_sizes[g];
            }

            // Build thread-to-group mapping
            thread_to_group.resize(total_threads);
            int thread_id = 0;
            for(size_t g = 0; g < group_sizes.size(); ++g)
            {
                for(int m = 0; m < group_sizes[g]; ++m)
                {
                    thread_to_group[thread_id++] = g;
                }
            }
        }

        void wait(int thread_id)
        {
            int group_id = thread_to_group[thread_id];
            Group& group = groups[group_id];

            // Phase 1: Local group barrier
            int arr = group.arrived.fetch_add(1) + 1;

            if(arr == group.num_members)
            {
                // Last thread in group
                group.arrived.store(0);

                // Phase 2: Global barrier (one representative per group)
                global_barrier.wait();

                // Notify group members
                group.ready.store(true);
                group.cv.notify_all();
            }
            else
            {
                // Wait for group to be released
                std::unique_lock<std::mutex> lock(group.mtx);
                group.cv.wait(lock, [&group]() { return group.ready.load(); });
            }

            // Reset ready flag for next barrier
            if(arr == 1)
            {
                group.ready.store(false);
            }
        }
    };

    TEST(MultiGPUSynchronization, GlobalBarrier)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Global Barrier Test ===" << std::endl;

        GlobalBarrier barrier(numDevices);
        std::atomic<int> counter{0};
        std::vector<int> arrival_order(numDevices);

        auto worker = [&](int device_id) {
            EXPECT_EQ(hipSetDevice(device_id), hipSuccess);

            // Phase 1: All threads initialize
            counter.fetch_add(1);

            // Wait for all threads to reach barrier
            barrier.wait();

            // After barrier, all should have incremented counter
            int count = counter.load();
            EXPECT_EQ(count, numDevices) << "Not all threads reached barrier";

            // Record arrival order for phase 2
            arrival_order[device_id] = counter.fetch_add(1);

            // Second barrier
            barrier.wait();

            // After second barrier, all should have incremented again
            count = counter.load();
            EXPECT_EQ(count, numDevices * 2) << "Barrier failed in phase 2";
        };

        std::vector<std::thread> threads;
        for(int dev = 0; dev < numDevices; ++dev)
        {
            threads.emplace_back(worker, dev);
        }

        for(auto& t : threads)
        {
            t.join();
        }

        hipblaslt_cout << "✓ All " << numDevices << " threads passed both barriers" << std::endl;
        hipblaslt_cout << "✓ Final counter value: " << counter.load() << std::endl;

        EXPECT_EQ(counter.load(), numDevices * 2);
    }

    TEST(MultiGPUSynchronization, HierarchicalBarrier)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 4) GTEST_SKIP() << "Requires 4+ GPUs for hierarchical test";

        hipblaslt_cout << "=== Hierarchical Barrier (NUMA-aware) ===" << std::endl;

        // Simulate 2 NUMA nodes with 2 GPUs each
        int gpus_per_node = numDevices / 2;
        std::vector<int> group_sizes = {gpus_per_node, numDevices - gpus_per_node};

        HierarchicalBarrier barrier(group_sizes);
        std::atomic<int> phase1_complete{0};
        std::atomic<int> phase2_complete{0};

        auto worker = [&](int thread_id) {
            int device_id = thread_id;
            EXPECT_EQ(hipSetDevice(device_id), hipSuccess);

            // Phase 1 work
            phase1_complete.fetch_add(1);

            // Hierarchical barrier (local group first, then global)
            barrier.wait(thread_id);

            // After barrier, all threads completed phase 1
            EXPECT_EQ(phase1_complete.load(), numDevices);

            // Phase 2 work
            phase2_complete.fetch_add(1);

            barrier.wait(thread_id);

            // After second barrier, all completed phase 2
            EXPECT_EQ(phase2_complete.load(), numDevices);
        };

        std::vector<std::thread> threads;
        for(int i = 0; i < numDevices; ++i)
        {
            threads.emplace_back(worker, i);
        }

        for(auto& t : threads)
        {
            t.join();
        }

        hipblaslt_cout << "✓ Hierarchical barrier with " << group_sizes.size()
                       << " groups completed successfully" << std::endl;
        hipblaslt_cout << "  Group sizes: ";
        for(auto size : group_sizes)
            hipblaslt_cout << size << " ";
        hipblaslt_cout << std::endl;
    }

    TEST(MultiGPUSynchronization, EventBasedSync)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Event-Based Multi-GPU Synchronization ===" << std::endl;

        const int64_t M = 512, N = 512, K = 512;

        std::vector<hipStream_t> streams(numDevices);
        std::vector<hipEvent_t> events(numDevices);
        std::vector<float*> d_data(numDevices);

        // Create streams and events
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipStreamCreate(&streams[dev]), hipSuccess);
            EXPECT_EQ(hipEventCreate(&events[dev]), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_data[dev], M * N * sizeof(float)), hipSuccess);
        }

        // Launch async memset on each GPU
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMemsetAsync(d_data[dev], dev, M * N * sizeof(float), streams[dev]), hipSuccess);
            EXPECT_EQ(hipEventRecord(events[dev], streams[dev]), hipSuccess);
        }

        // Create dependencies: Each GPU waits for previous GPU
        for(int dev = 1; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipStreamWaitEvent(streams[dev], events[dev - 1], 0), hipSuccess);
        }

        // Launch second operation after dependency
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMemsetAsync(d_data[dev], dev * 2, M * N * sizeof(float), streams[dev]), hipSuccess);
        }

        // Wait for all streams to complete
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipStreamSynchronize(streams[dev]), hipSuccess);
        }

        hipblaslt_cout << "✓ Event-based dependency chain executed successfully" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipStreamDestroy(streams[dev]), hipSuccess);
            EXPECT_EQ(hipEventDestroy(events[dev]), hipSuccess);
            EXPECT_EQ(hipFree(d_data[dev]), hipSuccess);
        }
    }

    TEST(MultiGPUSynchronization, StreamPriorities)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Stream Priority Test ===" << std::endl;

        // Get priority range
        int priority_low, priority_high;
        EXPECT_EQ(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high), hipSuccess);

        hipblaslt_cout << "Stream priority range: " << priority_high << " (high) to "
                       << priority_low << " (low)" << std::endl;

        std::vector<hipStream_t> high_priority_streams(numDevices);
        std::vector<hipStream_t> low_priority_streams(numDevices);

        // Create high and low priority streams
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipStreamCreateWithPriority(&high_priority_streams[dev], hipStreamDefault,
                                                   priority_high), hipSuccess);
            EXPECT_EQ(hipStreamCreateWithPriority(&low_priority_streams[dev], hipStreamDefault,
                                                   priority_low), hipSuccess);
        }

        const size_t work_size = 1024 * 1024 * 64; // 64MB

        std::vector<float*> d_high_data(numDevices), d_low_data(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_high_data[dev], work_size), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_low_data[dev], work_size), hipSuccess);
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Launch high priority work
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMemsetAsync(d_high_data[dev], 1, work_size, high_priority_streams[dev]), hipSuccess);
        }

        // Launch low priority work (should execute after high priority)
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMemsetAsync(d_low_data[dev], 2, work_size, low_priority_streams[dev]), hipSuccess);
        }

        // Wait for all work to complete
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipStreamSynchronize(high_priority_streams[dev]), hipSuccess);
            EXPECT_EQ(hipStreamSynchronize(low_priority_streams[dev]), hipSuccess);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        hipblaslt_cout << "✓ Priority streams completed in " << (elapsed.count() * 1000.0)
                       << " ms" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipStreamDestroy(high_priority_streams[dev]), hipSuccess);
            EXPECT_EQ(hipStreamDestroy(low_priority_streams[dev]), hipSuccess);
            EXPECT_EQ(hipFree(d_high_data[dev]), hipSuccess);
            EXPECT_EQ(hipFree(d_low_data[dev]), hipSuccess);
        }
    }

    TEST(MultiGPUSynchronization, LockFreeQueue)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Lock-Free Work Queue ===" << std::endl;

        // Simple lock-free queue implementation
        struct WorkItem
        {
            int64_t M, N, K;
            int priority;
        };

        const int total_work_items = 64;
        std::atomic<int> work_index{0};
        std::vector<WorkItem> work_items(total_work_items);

        // Initialize work items
        for(int i = 0; i < total_work_items; ++i)
        {
            work_items[i] = {512 + i * 32, 512 + i * 32, 512, i};
        }

        std::vector<std::atomic<int>> work_completed_per_gpu(numDevices);
        for(auto& counter : work_completed_per_gpu)
        {
            counter = 0;
        }

        auto worker = [&](int device_id) {
            EXPECT_EQ(hipSetDevice(device_id), hipSuccess);

            while(true)
            {
                // Atomically fetch next work item
                int idx = work_index.fetch_add(1);

                if(idx >= total_work_items)
                    break;

                const WorkItem& item = work_items[idx];

                // Simulate work (allocate and free)
                float* d_temp = nullptr;
                size_t size = item.M * item.K * sizeof(float);
                hipError_t err = hipMalloc(&d_temp, size);

                if(err == hipSuccess)
                {
                    EXPECT_EQ(hipMemset(d_temp, 0, size), hipSuccess);
                    EXPECT_EQ(hipFree(d_temp), hipSuccess);
                    work_completed_per_gpu[device_id].fetch_add(1);
                }
            }
        };

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        for(int dev = 0; dev < numDevices; ++dev)
        {
            threads.emplace_back(worker, dev);
        }

        for(auto& t : threads)
        {
            t.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // Report distribution
        hipblaslt_cout << "Work distribution across GPUs:" << std::endl;
        int total_completed = 0;
        for(int dev = 0; dev < numDevices; ++dev)
        {
            int completed = work_completed_per_gpu[dev].load();
            total_completed += completed;
            hipblaslt_cout << "  GPU " << dev << ": " << completed << " items ("
                           << (100.0 * completed / total_work_items) << "%)" << std::endl;
        }

        hipblaslt_cout << "Total items processed: " << total_completed << "/" << total_work_items << std::endl;
        hipblaslt_cout << "Time: " << (elapsed.count() * 1000.0) << " ms" << std::endl;

        EXPECT_EQ(total_completed, total_work_items) << "Not all work items completed";
    }

    TEST(MultiGPUSynchronization, PipelinedExecution)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 3) GTEST_SKIP() << "Requires 3+ GPUs for pipeline";

        hipblaslt_cout << "=== Pipelined Multi-GPU Execution ===" << std::endl;

        // 3-stage pipeline: Stage 0 (GPU 0) -> Stage 1 (GPU 1) -> Stage 2 (GPU 2)
        const int num_stages = std::min(3, numDevices);
        const int num_batches = 8;
        const int64_t batch_size = 512 * 512;

        std::vector<hipStream_t> streams(num_stages);
        std::vector<hipEvent_t> stage_complete(num_stages);
        std::vector<std::vector<float*>> stage_buffers(num_stages);

        // Create streams and events
        for(int stage = 0; stage < num_stages; ++stage)
        {
            EXPECT_EQ(hipSetDevice(stage), hipSuccess);
            EXPECT_EQ(hipStreamCreate(&streams[stage]), hipSuccess);
            EXPECT_EQ(hipEventCreate(&stage_complete[stage]), hipSuccess);

            stage_buffers[stage].resize(num_batches);
            for(int b = 0; b < num_batches; ++b)
            {
                EXPECT_EQ(hipMalloc(&stage_buffers[stage][b], batch_size * sizeof(float)), hipSuccess);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Pipeline execution
        for(int batch = 0; batch < num_batches; ++batch)
        {
            for(int stage = 0; stage < num_stages; ++stage)
            {
                EXPECT_EQ(hipSetDevice(stage), hipSuccess);

                // Wait for previous stage if not first stage
                if(stage > 0 && batch > 0)
                {
                    EXPECT_EQ(hipStreamWaitEvent(streams[stage], stage_complete[stage - 1], 0), hipSuccess);
                }

                // Execute stage work (memset as placeholder)
                EXPECT_EQ(hipMemsetAsync(stage_buffers[stage][batch], stage,
                                         batch_size * sizeof(float), streams[stage]), hipSuccess);

                // Record completion
                EXPECT_EQ(hipEventRecord(stage_complete[stage], streams[stage]), hipSuccess);
            }
        }

        // Wait for pipeline to drain
        for(int stage = 0; stage < num_stages; ++stage)
        {
            EXPECT_EQ(hipStreamSynchronize(streams[stage]), hipSuccess);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        hipblaslt_cout << "✓ " << num_stages << "-stage pipeline processed " << num_batches
                       << " batches in " << (elapsed.count() * 1000.0) << " ms" << std::endl;

        // Cleanup
        for(int stage = 0; stage < num_stages; ++stage)
        {
            EXPECT_EQ(hipSetDevice(stage), hipSuccess);
            EXPECT_EQ(hipStreamDestroy(streams[stage]), hipSuccess);
            EXPECT_EQ(hipEventDestroy(stage_complete[stage]), hipSuccess);

            for(int b = 0; b < num_batches; ++b)
            {
                EXPECT_EQ(hipFree(stage_buffers[stage][b]), hipSuccess);
            }
        }
    }

} // namespace
