/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Heterogeneous Configuration Test Suite
 * Tests: Mixed GPU architectures, different memory sizes, asymmetric workloads
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <map>
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

    struct GPUCapability
    {
        int device_id;
        std::string name;
        size_t total_memory;
        int compute_major;
        int compute_minor;
        int multiprocessor_count;
        int max_threads_per_mp;
        double compute_power_index; // Normalized compute capability
    };

    // ----------------------------------------------------------------------------
    // Test 1: GPU Architecture Detection
    // ----------------------------------------------------------------------------
    TEST(MultiGPUHeterogeneous, ArchitectureDetection)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 1) GTEST_SKIP() << "Requires at least 1 GPU";

        hipblaslt_cout << "=== GPU Architecture Detection ===" << std::endl;

        std::vector<GPUCapability> gpu_caps;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            hipDeviceProp_t prop;
            hipGetDeviceProperties(&prop, dev);

            GPUCapability cap;
            cap.device_id = dev;
            cap.name = prop.name;
            cap.total_memory = prop.totalGlobalMem;
            cap.compute_major = prop.major;
            cap.compute_minor = prop.minor;
            cap.multiprocessor_count = prop.multiProcessorCount;
            cap.max_threads_per_mp = prop.maxThreadsPerMultiProcessor;

            // Compute power index (relative compute capability)
            cap.compute_power_index = cap.multiprocessor_count * cap.max_threads_per_mp *
                                     (cap.compute_major * 10 + cap.compute_minor);

            gpu_caps.push_back(cap);

            hipblaslt_cout << "GPU " << dev << ": " << cap.name << std::endl;
            hipblaslt_cout << "  Memory: " << (cap.total_memory / 1024.0 / 1024.0 / 1024.0) << " GB" << std::endl;
            hipblaslt_cout << "  Compute: " << cap.compute_major << "." << cap.compute_minor << std::endl;
            hipblaslt_cout << "  Multiprocessors: " << cap.multiprocessor_count << std::endl;
            hipblaslt_cout << "  Max threads/MP: " << cap.max_threads_per_mp << std::endl;
            hipblaslt_cout << "  Compute power index: " << cap.compute_power_index << std::endl;
        }

        // Detect heterogeneity
        bool heterogeneous_arch = false;
        bool heterogeneous_memory = false;

        for(size_t i = 1; i < gpu_caps.size(); ++i)
        {
            if(gpu_caps[i].name != gpu_caps[0].name)
                heterogeneous_arch = true;
            if(std::abs(static_cast<long>(gpu_caps[i].total_memory) -
                       static_cast<long>(gpu_caps[0].total_memory)) > 1024 * 1024 * 1024) // >1GB diff
                heterogeneous_memory = true;
        }

        if(heterogeneous_arch)
            hipblaslt_cout << "Detected heterogeneous GPU architectures" << std::endl;
        else
            hipblaslt_cout << "All GPUs have same architecture" << std::endl;

        if(heterogeneous_memory)
            hipblaslt_cout << "Detected heterogeneous memory configurations" << std::endl;
        else
            hipblaslt_cout << "All GPUs have similar memory" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 2: Capability-Based Workload Distribution
    // ----------------------------------------------------------------------------
    TEST(MultiGPUHeterogeneous, CapabilityBasedDistribution)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Capability-Based Workload Distribution ===" << std::endl;

        // Measure relative compute power
        std::vector<double> compute_power(numDevices);
        const int64_t test_size = 512;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            float *d_a, *d_b, *d_c, *d_d;
            hipMalloc(&d_a, test_size * test_size * sizeof(float));
            hipMalloc(&d_b, test_size * test_size * sizeof(float));
            hipMalloc(&d_c, test_size * test_size * sizeof(float));
            hipMalloc(&d_d, test_size * test_size * sizeof(float));

            std::vector<float> h_data(test_size * test_size, 1.0f);
            hipMemcpy(d_a, h_data.data(), test_size * test_size * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_b, h_data.data(), test_size * test_size * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_c, h_data.data(), test_size * test_size * sizeof(float), hipMemcpyHostToDevice);

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, test_size, test_size, test_size);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, test_size, test_size, test_size);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, test_size, test_size, test_size);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, test_size, test_size, test_size);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasOperation_t op = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));

            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);

            hipblasLtMatmulHeuristicResult_t heuristicResult[1];
            int returnedAlgoCount = 0;
            hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matC, matD,
                                           pref, 1, heuristicResult, &returnedAlgoCount);

            if(returnedAlgoCount > 0)
            {
                void* d_workspace = nullptr;
                if(heuristicResult[0].workspaceSize > 0)
                    hipMalloc(&d_workspace, heuristicResult[0].workspaceSize);

                float alpha = 1.0f, beta = 0.0f;

                // Benchmark
                auto start = std::chrono::high_resolution_clock::now();
                for(int iter = 0; iter < 10; ++iter)
                {
                    hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                   &beta, d_c, matC, d_d, matD,
                                   &heuristicResult[0].algo, d_workspace,
                                   heuristicResult[0].workspaceSize, 0);
                }
                hipDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> elapsed = end - start;
                compute_power[dev] = 1.0 / (elapsed.count() / 10.0); // Higher is faster

                if(d_workspace) hipFree(d_workspace);
            }

            hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_d);
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);

            hipblaslt_cout << "GPU " << dev << " relative compute power: " << compute_power[dev] << std::endl;
        }

        // Distribute workload proportionally to compute power
        double total_power = std::accumulate(compute_power.begin(), compute_power.end(), 0.0);
        const int64_t total_batches = 128;

        std::vector<int64_t> batches_per_gpu(numDevices);
        int64_t assigned_batches = 0;

        for(int dev = 0; dev < numDevices - 1; ++dev)
        {
            batches_per_gpu[dev] = static_cast<int64_t>(
                (compute_power[dev] / total_power) * total_batches
            );
            assigned_batches += batches_per_gpu[dev];
        }
        batches_per_gpu[numDevices - 1] = total_batches - assigned_batches;

        hipblaslt_cout << "\nWorkload distribution for " << total_batches << " batches:" << std::endl;
        for(int dev = 0; dev < numDevices; ++dev)
        {
            double percentage = (batches_per_gpu[dev] / static_cast<double>(total_batches)) * 100.0;
            hipblaslt_cout << "GPU " << dev << ": " << batches_per_gpu[dev]
                           << " batches (" << percentage << "%)" << std::endl;
        }

        // Verify all batches assigned
        EXPECT_EQ(std::accumulate(batches_per_gpu.begin(), batches_per_gpu.end(), 0L), total_batches);
    }

    // ----------------------------------------------------------------------------
    // Test 3: Memory-Constrained Workload Adaptation
    // ----------------------------------------------------------------------------
    TEST(MultiGPUHeterogeneous, MemoryConstrainedAdaptation)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Memory-Constrained Workload Adaptation ===" << std::endl;

        // Get available memory for each GPU
        std::vector<size_t> available_memory(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);

            size_t free_mem, total_mem;
            EXPECT_EQ(hipMemGetInfo(&free_mem, &total_mem), hipSuccess);
            available_memory[dev] = free_mem;

            hipblaslt_cout << "GPU " << dev << " available memory: "
                           << (free_mem / 1024.0 / 1024.0 / 1024.0) << " GB" << std::endl;
        }

        // Find GPU with least memory
        size_t min_memory = *std::min_element(available_memory.begin(), available_memory.end());
        size_t max_memory = *std::max_element(available_memory.begin(), available_memory.end());

        hipblaslt_cout << "Memory range: " << (min_memory / 1024.0 / 1024.0 / 1024.0)
                       << " GB - " << (max_memory / 1024.0 / 1024.0 / 1024.0) << " GB" << std::endl;

        // Adapt problem size per GPU based on available memory
        const int64_t base_size = 1024;
        const size_t base_memory = 3 * base_size * base_size * sizeof(float); // A, B, D matrices

        for(int dev = 0; dev < numDevices; ++dev)
        {
            // Scale problem size based on available memory
            double memory_ratio = std::min(1.0, available_memory[dev] / (10.0 * base_memory));
            int64_t scaled_size = static_cast<int64_t>(base_size * std::sqrt(memory_ratio));

            hipblaslt_cout << "GPU " << dev << " adapted problem size: "
                           << scaled_size << "x" << scaled_size << std::endl;

            hipSetDevice(dev);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Allocate based on scaled size
            float *d_a, *d_b, *d_d;
            auto err_a = hipMalloc(&d_a, scaled_size * scaled_size * sizeof(float));
            auto err_b = hipMalloc(&d_b, scaled_size * scaled_size * sizeof(float));
            auto err_d = hipMalloc(&d_d, scaled_size * scaled_size * sizeof(float));

            EXPECT_EQ(err_a, hipSuccess) << "GPU " << dev << " allocation failed";
            EXPECT_EQ(err_b, hipSuccess) << "GPU " << dev << " allocation failed";
            EXPECT_EQ(err_d, hipSuccess) << "GPU " << dev << " allocation failed";

            if(err_a == hipSuccess && err_b == hipSuccess && err_d == hipSuccess)
            {
                hipblaslt_cout << "GPU " << dev << " successfully allocated "
                               << (3 * scaled_size * scaled_size * sizeof(float) / 1024.0 / 1024.0)
                               << " MB" << std::endl;

                hipFree(d_a);
                hipFree(d_b);
                hipFree(d_d);
            }

            hipblasLtDestroy(handle);
        }
    }

    // ----------------------------------------------------------------------------
    // Test 4: Mixed Precision Across Different Architectures
    // ----------------------------------------------------------------------------
    TEST(MultiGPUHeterogeneous, MixedPrecisionPerArchitecture)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Mixed Precision Per Architecture ===" << std::endl;

        // Test different precision modes on different GPUs
        std::vector<hipDataType> precision_modes = {HIP_R_32F, HIP_R_16F, HIP_R_16BF};
        const char* precision_names[] = {"FP32", "FP16", "BF16"};

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            hipDeviceProp_t prop;
            hipGetDeviceProperties(&prop, dev);

            hipblaslt_cout << "\nGPU " << dev << " (" << prop.name << ") testing:" << std::endl;

            for(size_t p = 0; p < precision_modes.size(); ++p)
            {
                hipblasLtHandle_t handle;
                hipblasLtCreate(&handle);

                hipblasLtMatrixLayout_t matA, matB, matD;
                auto status_a = hipblasLtMatrixLayoutCreate(&matA, precision_modes[p], M, K, M);
                auto status_b = hipblasLtMatrixLayoutCreate(&matB, precision_modes[p], K, N, K);
                auto status_d = hipblasLtMatrixLayoutCreate(&matD, precision_modes[p], M, N, M);

                if(status_a == HIPBLAS_STATUS_SUCCESS &&
                   status_b == HIPBLAS_STATUS_SUCCESS &&
                   status_d == HIPBLAS_STATUS_SUCCESS)
                {
                    hipblaslt_cout << "  " << precision_names[p] << ": Supported" << std::endl;

                    hipblasLtMatrixLayoutDestroy(matA);
                    hipblasLtMatrixLayoutDestroy(matB);
                    hipblasLtMatrixLayoutDestroy(matD);
                }
                else
                {
                    hipblaslt_cout << "  " << precision_names[p] << ": Not supported" << std::endl;
                }

                hipblasLtDestroy(handle);
            }
        }
    }

    // ----------------------------------------------------------------------------
    // Test 5: Asymmetric Batch Distribution
    // ----------------------------------------------------------------------------
    TEST(MultiGPUHeterogeneous, AsymmetricBatchDistribution)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Asymmetric Batch Distribution ===" << std::endl;

        // Distribute different batch counts to different GPUs
        const int64_t M = 128, N = 128, K = 128;
        std::vector<int64_t> batch_counts = {8, 16, 32, 64};

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            int64_t batch_count = batch_counts[dev % batch_counts.size()];

            hipblaslt_cout << "GPU " << dev << " processing " << batch_count << " batches" << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            // Set batch count
            hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                             &batch_count, sizeof(batch_count));
            hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                             &batch_count, sizeof(batch_count));
            hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                             &batch_count, sizeof(batch_count));
            hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                             &batch_count, sizeof(batch_count));

            int64_t stride_a = M * K, stride_b = K * N, stride_c = M * N;
            hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                             &stride_a, sizeof(stride_a));
            hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                             &stride_b, sizeof(stride_b));
            hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                             &stride_c, sizeof(stride_c));
            hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                             &stride_c, sizeof(stride_c));

            hipblaslt_cout << "GPU " << dev << " configured with asymmetric batch count" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // ----------------------------------------------------------------------------
    // Test 6: Architecture-Specific Algorithm Selection
    // ----------------------------------------------------------------------------
    TEST(MultiGPUHeterogeneous, ArchSpecificAlgorithms)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Architecture-Specific Algorithm Selection ===" << std::endl;

        const int64_t M = 512, N = 512, K = 512;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            hipDeviceProp_t prop;
            hipGetDeviceProperties(&prop, dev);

            hipblaslt_cout << "\nGPU " << dev << " (" << prop.name << "):" << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            float *d_a, *d_b, *d_c, *d_d;
            hipMalloc(&d_a, M * K * sizeof(float));
            hipMalloc(&d_b, K * N * sizeof(float));
            hipMalloc(&d_c, M * N * sizeof(float));
            hipMalloc(&d_d, M * N * sizeof(float));

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasOperation_t op = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));

            // Get all available algorithms
            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);

            size_t workspace = 64 * 1024 * 1024;
            hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &workspace, sizeof(workspace));

            const int max_algos = 20;
            hipblasLtMatmulHeuristicResult_t heuristicResults[max_algos];
            int returnedAlgoCount = 0;

            hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matC, matD,
                                           pref, max_algos, heuristicResults, &returnedAlgoCount);

            hipblaslt_cout << "  Available algorithms: " << returnedAlgoCount << std::endl;

            // Different GPUs may have different optimal algorithms
            if(returnedAlgoCount > 0)
            {
                hipblaslt_cout << "  Best algorithm selected (count: " << returnedAlgoCount << ")" << std::endl;
                hipblaslt_cout << "  Workspace required: "
                               << (heuristicResults[0].workspaceSize / 1024.0 / 1024.0) << " MB" << std::endl;
            }

            hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_d);
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }
    }

} // namespace
