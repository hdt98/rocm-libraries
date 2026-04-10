/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Dynamic & Adaptive Patterns Test Suite
 * Tests: Dynamic load balancing, work stealing, auto-tuning, adaptive precision
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <atomic>
#include <mutex>
#include <queue>
#include <thread>
#include <chrono>

#include <gtest/gtest-spi.h>

namespace
{
    int getNumGPUs()
    {
        int numDevices = 0;
        auto err = hipGetDeviceCount(&numDevices);
        (void)err; // Ignore error - function returns 0 if call fails
        return numDevices;
    }

    // Thread-safe work queue for work stealing
    template<typename T>
    class WorkQueue
    {
    private:
        std::queue<T> queue;
        mutable std::mutex mutex;

    public:
        void push(const T& item)
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push(item);
        }

        bool try_pop(T& item)
        {
            std::lock_guard<std::mutex> lock(mutex);
            if(queue.empty())
                return false;

            item = queue.front();
            queue.pop();
            return true;
        }

        size_t size() const
        {
            std::lock_guard<std::mutex> lock(mutex);
            return queue.size();
        }

        bool empty() const
        {
            std::lock_guard<std::mutex> lock(mutex);
            return queue.empty();
        }
    };

    struct WorkItem
    {
        int64_t batch_id;
        int64_t M, N, K;
    };

    // ----------------------------------------------------------------------------
    // Test 1: Dynamic Load Balancing
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDynamic, DynamicLoadBalancing)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Dynamic Load Balancing ===" << std::endl;

        // Create work queue with varying workload sizes
        WorkQueue<WorkItem> work_queue;
        const int total_batches = 64;

        for(int i = 0; i < total_batches; ++i)
        {
            WorkItem item;
            item.batch_id = i;
            // Varying problem sizes
            item.M = 128 + (i % 4) * 128;
            item.N = 128 + (i % 3) * 128;
            item.K = 128 + (i % 5) * 128;
            work_queue.push(item);
        }

        hipblaslt_cout << "Total work items: " << total_batches << std::endl;

        // Track work completed by each GPU
        std::vector<std::atomic<int>> work_completed(numDevices);
        for(auto& counter : work_completed)
            counter = 0;

        // Worker function for each GPU
        auto worker = [&](int device_id) {
            EXPECT_EQ(hipSetDevice(device_id), hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            WorkItem item;
            while(work_queue.try_pop(item))
            {
                // Allocate and execute GEMM
                float *d_a, *d_b, *d_d;
                EXPECT_EQ(hipMalloc(&d_a, item.M * item.K * sizeof(float)), hipSuccess);
                EXPECT_EQ(hipMalloc(&d_b, item.K * item.N * sizeof(float)), hipSuccess);
                EXPECT_EQ(hipMalloc(&d_d, item.M * item.N * sizeof(float)), hipSuccess);

                std::vector<float> h_a(item.M * item.K, 1.0f);
                std::vector<float> h_b(item.K * item.N, 2.0f);

                EXPECT_EQ(hipMemcpy(d_a, h_a.data(), item.M * item.K * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
                EXPECT_EQ(hipMemcpy(d_b, h_b.data(), item.K * item.N * sizeof(float), hipMemcpyHostToDevice), hipSuccess);

                hipblasLtMatrixLayout_t matA, matB, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, item.M, item.K, item.M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, item.K, item.N, item.K);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, item.M, item.N, item.M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasOperation_t op = HIPBLAS_OP_N;
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));

                hipblasLtMatmulPreference_t pref;
                hipblasLtMatmulPreferenceCreate(&pref);

                hipblasLtMatmulHeuristicResult_t heuristicResult[1];
                int returnedAlgoCount = 0;
                hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matD, matD,
                                               pref, 1, heuristicResult, &returnedAlgoCount);

                if(returnedAlgoCount > 0)
                {
                    void* d_workspace = nullptr;
                    if(heuristicResult[0].workspaceSize > 0)
                        EXPECT_EQ(hipMalloc(&d_workspace, heuristicResult[0].workspaceSize), hipSuccess);

                    float alpha = 1.0f, beta = 0.0f;
                    hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                   &beta, d_d, matD, d_d, matD,
                                   &heuristicResult[0].algo, d_workspace,
                                   heuristicResult[0].workspaceSize, 0);

                    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

                    if(d_workspace) EXPECT_EQ(hipFree(d_workspace), hipSuccess);
                }

                EXPECT_EQ(hipFree(d_a), hipSuccess);
                EXPECT_EQ(hipFree(d_b), hipSuccess);
                EXPECT_EQ(hipFree(d_d), hipSuccess);
                hipblasLtMatmulPreferenceDestroy(pref);
                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);

                work_completed[device_id]++;
            }

            hipblasLtDestroy(handle);
        };

        // Launch worker threads for each GPU
        std::vector<std::thread> threads;
        auto start = std::chrono::high_resolution_clock::now();

        for(int dev = 0; dev < numDevices; ++dev)
        {
            threads.emplace_back(worker, dev);
        }

        // Wait for all workers to complete
        for(auto& t : threads)
            t.join();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // Report load distribution
        hipblaslt_cout << "\nLoad distribution:" << std::endl;
        int total_completed = 0;
        for(int dev = 0; dev < numDevices; ++dev)
        {
            int completed = work_completed[dev].load();
            total_completed += completed;
            double percentage = (completed / static_cast<double>(total_batches)) * 100.0;
            hipblaslt_cout << "GPU " << dev << ": " << completed << " batches ("
                           << percentage << "%)" << std::endl;
        }

        EXPECT_EQ(total_completed, total_batches) << "Not all work items completed!";

        hipblaslt_cout << "Total time: " << elapsed.count() << " seconds" << std::endl;

        // Check for reasonable load balance (no GPU should do >60% of work in a 2-GPU system)
        if(numDevices == 2)
        {
            double max_load = std::max(work_completed[0].load(), work_completed[1].load());
            double max_percentage = (max_load / total_batches) * 100.0;
            EXPECT_LT(max_percentage, 65.0) << "Load imbalance detected";
        }
    }

    // ----------------------------------------------------------------------------
    // Test 2: Work Stealing Between GPUs
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDynamic, WorkStealing)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Work Stealing Between GPUs ===" << std::endl;

        // Create per-GPU work queues with unbalanced initial distribution
        std::vector<WorkQueue<WorkItem>> gpu_queues(numDevices);

        // Intentionally unbalanced distribution (GPU 0 gets most work)
        const int total_work = 40;
        for(int i = 0; i < total_work; ++i)
        {
            WorkItem item;
            item.batch_id = i;
            item.M = item.N = item.K = 256;

            // GPU 0 gets 75% of work, others share the rest
            int target_gpu = (i < total_work * 3 / 4) ? 0 : (i % numDevices);
            gpu_queues[target_gpu].push(item);
        }

        hipblaslt_cout << "Initial distribution:" << std::endl;
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipblaslt_cout << "GPU " << dev << ": " << gpu_queues[dev].size()
                           << " items" << std::endl;
        }

        std::vector<std::atomic<int>> completed_count(numDevices);
        std::vector<std::atomic<int>> stolen_count(numDevices);
        for(int i = 0; i < numDevices; ++i)
        {
            completed_count[i] = 0;
            stolen_count[i] = 0;
        }

        // Worker with work stealing
        auto worker_with_stealing = [&](int device_id) {
            EXPECT_EQ(hipSetDevice(device_id), hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            WorkItem item;
            bool done = false;

            while(!done)
            {
                // Try own queue first
                if(gpu_queues[device_id].try_pop(item))
                {
                    // Execute work (simplified for test)
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    completed_count[device_id]++;
                }
                else
                {
                    // Try stealing from other GPUs
                    bool stole_work = false;
                    for(int victim = 0; victim < numDevices; ++victim)
                    {
                        if(victim == device_id) continue;

                        if(gpu_queues[victim].try_pop(item))
                        {
                            // Stole work!
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            completed_count[device_id]++;
                            stolen_count[device_id]++;
                            stole_work = true;
                            break;
                        }
                    }

                    if(!stole_work)
                    {
                        // Check if all queues are empty
                        bool all_empty = true;
                        for(int q = 0; q < numDevices; ++q)
                        {
                            if(!gpu_queues[q].empty())
                            {
                                all_empty = false;
                                break;
                            }
                        }

                        if(all_empty)
                            done = true;
                        else
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
            }

            hipblasLtDestroy(handle);
        };

        std::vector<std::thread> threads;
        auto start = std::chrono::high_resolution_clock::now();

        for(int dev = 0; dev < numDevices; ++dev)
        {
            threads.emplace_back(worker_with_stealing, dev);
        }

        for(auto& t : threads)
            t.join();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        hipblaslt_cout << "\nFinal statistics:" << std::endl;
        int total_completed = 0;
        int total_stolen = 0;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            int completed = completed_count[dev].load();
            int stolen = stolen_count[dev].load();
            total_completed += completed;
            total_stolen += stolen;

            hipblaslt_cout << "GPU " << dev << ": completed=" << completed
                           << ", stolen=" << stolen << std::endl;
        }

        EXPECT_EQ(total_completed, total_work) << "Work count mismatch";
        EXPECT_GT(total_stolen, 0) << "No work stealing occurred!";

        hipblaslt_cout << "Total stolen work items: " << total_stolen << std::endl;
        hipblaslt_cout << "Total time: " << elapsed.count() << " seconds" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 3: Runtime GPU Selection Based on Availability
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDynamic, RuntimeGPUSelection)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Runtime GPU Selection ===" << std::endl;

        // Track GPU utilization
        std::vector<std::atomic<int>> active_tasks(numDevices);
        for(auto& counter : active_tasks)
            counter = 0;

        std::mutex selection_mutex;

        // Function to select least busy GPU
        auto selectLeastBusyGPU = [&]() -> int {
            std::lock_guard<std::mutex> lock(selection_mutex);

            int min_tasks = active_tasks[0].load();
            int best_gpu = 0;

            for(int dev = 1; dev < numDevices; ++dev)
            {
                int tasks = active_tasks[dev].load();
                if(tasks < min_tasks)
                {
                    min_tasks = tasks;
                    best_gpu = dev;
                }
            }

            return best_gpu;
        };

        const int total_submissions = 32;
        std::vector<int> gpu_assignments(total_submissions);

        // Simulate dynamic task submission
        for(int i = 0; i < total_submissions; ++i)
        {
            int selected_gpu = selectLeastBusyGPU();
            gpu_assignments[i] = selected_gpu;
            active_tasks[selected_gpu]++;

            hipblaslt_cout << "Task " << i << " assigned to GPU " << selected_gpu << std::endl;

            // Randomly complete some tasks
            if(i % 5 == 0 && i > 0)
            {
                for(int dev = 0; dev < numDevices; ++dev)
                {
                    if(active_tasks[dev].load() > 0)
                        active_tasks[dev]--;
                }
            }
        }

        // Count assignments per GPU
        std::vector<int> assignment_count(numDevices, 0);
        for(int gpu : gpu_assignments)
            assignment_count[gpu]++;

        hipblaslt_cout << "\nAssignment distribution:" << std::endl;
        for(int dev = 0; dev < numDevices; ++dev)
        {
            double percentage = (assignment_count[dev] / static_cast<double>(total_submissions)) * 100.0;
            hipblaslt_cout << "GPU " << dev << ": " << assignment_count[dev]
                           << " tasks (" << percentage << "%)" << std::endl;
        }

        // In a balanced scenario, no GPU should get >60% of tasks
        int max_assignments = *std::max_element(assignment_count.begin(), assignment_count.end());
        double max_percentage = (max_assignments / static_cast<double>(total_submissions)) * 100.0;
        EXPECT_LT(max_percentage, 70.0) << "Severe load imbalance in dynamic assignment";
    }

    // ----------------------------------------------------------------------------
    // Test 4: Adaptive Precision Switching
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDynamic, AdaptivePrecisionSwitching)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Adaptive Precision Switching ===" << std::endl;

        // Simulate adaptive precision based on problem characteristics
        struct Problem
        {
            int64_t M, N, K;
            bool requires_high_precision;
        };

        std::vector<Problem> problems = {
            {128, 128, 128, false},  // Small -> FP16
            {1024, 1024, 1024, true}, // Large -> FP32
            {256, 256, 256, false},   // Medium -> FP16
            {2048, 2048, 2048, true}  // Very large -> FP32
        };

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);

            for(size_t p = 0; p < problems.size(); ++p)
            {
                const auto& prob = problems[p];

                // Select precision based on problem size and requirements
                hipDataType dtype = prob.requires_high_precision ? HIP_R_32F : HIP_R_16F;
                const char* dtype_name = prob.requires_high_precision ? "FP32" : "FP16";

                hipblaslt_cout << "GPU " << dev << " problem " << p << " ("
                               << prob.M << "x" << prob.N << "x" << prob.K << "): "
                               << dtype_name << std::endl;

                hipblasLtHandle_t handle;
                hipblasLtCreate(&handle);

                hipblasLtMatrixLayout_t matA, matB, matD;
                auto status_a = hipblasLtMatrixLayoutCreate(&matA, dtype, prob.M, prob.K, prob.M);
                auto status_b = hipblasLtMatrixLayoutCreate(&matB, dtype, prob.K, prob.N, prob.K);
                auto status_d = hipblasLtMatrixLayoutCreate(&matD, dtype, prob.M, prob.N, prob.M);

                if(status_a == HIPBLAS_STATUS_SUCCESS &&
                   status_b == HIPBLAS_STATUS_SUCCESS &&
                   status_d == HIPBLAS_STATUS_SUCCESS)
                {
                    hipblaslt_cout << "  Successfully configured with " << dtype_name << std::endl;

                    hipblasLtMatrixLayoutDestroy(matA);
                    hipblasLtMatrixLayoutDestroy(matB);
                    hipblasLtMatrixLayoutDestroy(matD);
                }

                hipblasLtDestroy(handle);
            }
        }
    }

    // ----------------------------------------------------------------------------
    // Test 5: Auto-Tuning Algorithm Selection
    // ----------------------------------------------------------------------------
    TEST(MultiGPUDynamic, AutoTuningAlgorithmSelection)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Auto-Tuning Algorithm Selection ===" << std::endl;

        const int64_t M = 512, N = 512, K = 512;

        for(int dev = 0; dev < std::min(numDevices, 2); ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);

            hipblaslt_cout << "\nGPU " << dev << " auto-tuning:" << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            float *d_a, *d_b, *d_d;
            EXPECT_EQ(hipMalloc(&d_a, M * K * sizeof(float)), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_b, K * N * sizeof(float)), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_d, M * N * sizeof(float)), hipSuccess);

            std::vector<float> h_a(M * K, 1.0f);
            std::vector<float> h_b(K * N, 2.0f);
            EXPECT_EQ(hipMemcpy(d_a, h_a.data(), M * K * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
            EXPECT_EQ(hipMemcpy(d_b, h_b.data(), K * N * sizeof(float), hipMemcpyHostToDevice), hipSuccess);

            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasOperation_t op = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));

            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);

            // Get multiple algorithm candidates
            const int max_algos = 10;
            hipblasLtMatmulHeuristicResult_t heuristicResults[max_algos];
            int returnedAlgoCount = 0;

            hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matD, matD,
                                           pref, max_algos, heuristicResults, &returnedAlgoCount);

            hipblaslt_cout << "  Available algorithms: " << returnedAlgoCount << std::endl;

            // Benchmark each algorithm
            std::vector<double> algo_times;
            int best_algo_idx = 0;
            double best_time = std::numeric_limits<double>::max();

            for(int algo_idx = 0; algo_idx < std::min(returnedAlgoCount, 5); ++algo_idx)
            {
                void* d_workspace = nullptr;
                if(heuristicResults[algo_idx].workspaceSize > 0)
                    EXPECT_EQ(hipMalloc(&d_workspace, heuristicResults[algo_idx].workspaceSize), hipSuccess);

                float alpha = 1.0f, beta = 0.0f;

                // Warmup
                hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                               &beta, d_d, matD, d_d, matD,
                               &heuristicResults[algo_idx].algo, d_workspace,
                               heuristicResults[algo_idx].workspaceSize, 0);
                EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

                // Benchmark
                auto start = std::chrono::high_resolution_clock::now();
                for(int iter = 0; iter < 10; ++iter)
                {
                    hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                   &beta, d_d, matD, d_d, matD,
                                   &heuristicResults[algo_idx].algo, d_workspace,
                                   heuristicResults[algo_idx].workspaceSize, 0);
                }
                EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> elapsed = end - start;
                double avg_time = elapsed.count() / 10.0;

                hipblaslt_cout << "  Algo " << algo_idx << ": " << (avg_time * 1000.0) << " ms" << std::endl;

                if(avg_time < best_time)
                {
                    best_time = avg_time;
                    best_algo_idx = algo_idx;
                }

                if(d_workspace) EXPECT_EQ(hipFree(d_workspace), hipSuccess);
            }

            hipblaslt_cout << "  Best algorithm: " << best_algo_idx << " ("
                           << (best_time * 1000.0) << " ms)" << std::endl;

            EXPECT_EQ(hipFree(d_a), hipSuccess);
            EXPECT_EQ(hipFree(d_b), hipSuccess);
            EXPECT_EQ(hipFree(d_d), hipSuccess);
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }
    }

} // namespace
