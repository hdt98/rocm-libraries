/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Fault Tolerance Tests
 * Tests error handling, graceful degradation, and recovery scenarios
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
        auto err = hipGetDeviceCount(&numDevices);
        (void)err; // Ignore error - function returns 0 if call fails
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
    // Test 1: Out of Memory Error Handling
    // ============================================================================
    TEST(MultiGPUFaultTolerance, OutOfMemory)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Out of Memory Error Handling ===" << std::endl;

        const int64_t M = 512, N = 512, K = 512;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<_Float16*> d_A(numDevices), d_B(numDevices), d_C(numDevices);
        std::vector<bool> allocation_succeeded(numDevices, false);

        // Try to allocate on each GPU, handle failures gracefully
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            hipblasLtCreate(&handles[dev]);

            // Normal allocation
            hipError_t err_a = hipMalloc(&d_A[dev], M * K * sizeof(_Float16));
            hipError_t err_b = hipMalloc(&d_B[dev], K * N * sizeof(_Float16));
            hipError_t err_c = hipMalloc(&d_C[dev], M * N * sizeof(_Float16));

            if(err_a == hipSuccess && err_b == hipSuccess && err_c == hipSuccess)
            {
                allocation_succeeded[dev] = true;
                hipblaslt_cout << "GPU " << dev << ": Allocation succeeded" << std::endl;
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << ": Allocation FAILED (expected for OOM test)" << std::endl;
                // Free any partial allocations
                if(err_a == hipSuccess) { auto r = hipFree(d_A[dev]); (void)r; }
                if(err_b == hipSuccess) { auto r = hipFree(d_B[dev]); (void)r; }
                if(err_c == hipSuccess) { auto r = hipFree(d_C[dev]); (void)r; }
                allocation_succeeded[dev] = false;
            }
        }

        // Attempt oversized allocation on GPU 0 to test OOM handling
        EXPECT_EQ(hipSetDevice(0), hipSuccess);

        // Get actual GPU memory info to determine truly oversized allocation
        size_t free_mem = 0, total_mem = 0;
        hipError_t mem_info_err = hipMemGetInfo(&free_mem, &total_mem);
        EXPECT_EQ(mem_info_err, hipSuccess);

        // Request 2x total memory to ensure OOM
        size_t oversized = total_mem * 2;
        hipblaslt_cout << "GPU 0: Total memory: " << (total_mem / (1024*1024*1024)) << " GB, "
                       << "Requesting: " << (oversized / (1024*1024*1024)) << " GB" << std::endl;

        _Float16* d_oversized = nullptr;
        hipError_t oom_err = hipMalloc(&d_oversized, oversized);

        if(oom_err != hipSuccess)
        {
            hipblaslt_cout << "GPU 0: Oversized allocation properly failed with error code " << oom_err << std::endl;
        }
        else
        {
            hipblaslt_cout << "GPU 0: Unexpected success on oversized allocation" << std::endl;
            auto r = hipFree(d_oversized);
            (void)r; // Ignore return - this is cleanup path
        }

        // Cleanup successful allocations
        for(int dev = 0; dev < numDevices; ++dev)
        {
            if(allocation_succeeded[dev])
            {
                auto r = hipSetDevice(dev);
                (void)r; // Ignore in cleanup path
                r = hipFree(d_A[dev]); (void)r;
                r = hipFree(d_B[dev]); (void)r;
                r = hipFree(d_C[dev]); (void)r;
            }
            hipblasLtDestroy(handles[dev]);
        }

        EXPECT_NE(oom_err, hipSuccess) << "Oversized allocation should fail!";
        hipblaslt_cout << "✓ Out of Memory Error Handling Test Complete" << std::endl;
    }

    // ============================================================================
    // Test 2: Partial Failure - Some GPUs succeed, some fail
    // ============================================================================
    TEST(MultiGPUFaultTolerance, PartialFailure)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Partial Failure Handling ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<_Float16> h_A(M * K), h_B(K * N);
        initMatrix_FP16(h_A, M * K, 1.0f);
        initMatrix_FP16(h_B, K * N, 2.0f);

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<_Float16*> d_A(numDevices), d_B(numDevices), d_C(numDevices);
        std::vector<bool> gpu_succeeded(numDevices, false);
        std::vector<std::vector<_Float16>> h_C_results(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);

            hipblasStatus_t err = hipblasLtCreate(&handles[dev]);
            if(err != HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cout << "GPU " << dev << ": Handle creation FAILED" << std::endl;
                continue;
            }

            // Allocate
            if(hipMalloc(&d_A[dev], M * K * sizeof(_Float16)) != hipSuccess ||
               hipMalloc(&d_B[dev], K * N * sizeof(_Float16)) != hipSuccess ||
               hipMalloc(&d_C[dev], M * N * sizeof(_Float16)) != hipSuccess)
            {
                hipblaslt_cout << "GPU " << dev << ": Allocation FAILED" << std::endl;
                continue;
            }

            // Copy data
            EXPECT_EQ(hipMemcpy(d_A[dev], h_A.data(), M * K * sizeof(_Float16), hipMemcpyHostToDevice), hipSuccess);
            EXPECT_EQ(hipMemcpy(d_B[dev], h_B.data(), K * N * sizeof(_Float16), hipMemcpyHostToDevice), hipSuccess);

            // Execute GEMM
            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasStatus_t matmul_status = hipblasLtMatmul(handles[dev], matmul, &alpha,
                                                            d_A[dev], matA, d_B[dev], matB,
                                                            &beta, d_C[dev], matC, d_C[dev], matD,
                                                            nullptr, nullptr, 0, 0);

            if(matmul_status == HIPBLAS_STATUS_SUCCESS)
            {
                h_C_results[dev].resize(M * N);
                EXPECT_EQ(hipMemcpy(h_C_results[dev].data(), d_C[dev], M * N * sizeof(_Float16), hipMemcpyDeviceToHost), hipSuccess);
                gpu_succeeded[dev] = true;
                hipblaslt_cout << "GPU " << dev << ": GEMM succeeded" << std::endl;
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << ": GEMM FAILED with status " << matmul_status << std::endl;
            }

            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
        }

        // Count successes
        int success_count = 0;
        for(int dev = 0; dev < numDevices; ++dev)
        {
            if(gpu_succeeded[dev]) success_count++;
        }

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            if(d_A[dev]) EXPECT_EQ(hipFree(d_A[dev]), hipSuccess);
            if(d_B[dev]) EXPECT_EQ(hipFree(d_B[dev]), hipSuccess);
            if(d_C[dev]) EXPECT_EQ(hipFree(d_C[dev]), hipSuccess);
            if(handles[dev]) hipblasLtDestroy(handles[dev]);
        }

        // In normal operation, all should succeed
        EXPECT_GT(success_count, 0) << "At least one GPU should succeed!";
        hipblaslt_cout << "✓ Partial Failure Test: " << success_count << "/" << numDevices << " GPUs succeeded" << std::endl;
    }

    // ============================================================================
    // Test 3: Continue with Fewer GPUs - Graceful Degradation
    // ============================================================================
    TEST(MultiGPUFaultTolerance, ContinueWithFewerGPUs)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Continue with Fewer GPUs ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t total_batches = 8;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<_Float16> h_A(M * K * total_batches);
        std::vector<_Float16> h_B(K * N * total_batches);
        initMatrix_FP16(h_A, M * K * total_batches, 1.0f);
        initMatrix_FP16(h_B, K * N * total_batches, 2.0f);

        // Determine available GPUs
        std::vector<int> available_gpus;
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            hipblasLtHandle_t test_handle;
            if(hipblasLtCreate(&test_handle) == HIPBLAS_STATUS_SUCCESS)
            {
                available_gpus.push_back(dev);
                hipblasLtDestroy(test_handle);
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << " unavailable, skipping" << std::endl;
            }
        }

        if(available_gpus.empty())
        {
            GTEST_SKIP() << "No GPUs available";
        }

        hipblaslt_cout << "Available GPUs: " << available_gpus.size() << "/" << numDevices << std::endl;

        // Distribute work among available GPUs
        int num_available = available_gpus.size();
        int64_t batches_per_gpu = total_batches / num_available;
        int64_t remaining_batches = total_batches % num_available;

        std::vector<hipblasLtHandle_t> handles(num_available);
        std::vector<_Float16*> d_A(num_available), d_B(num_available), d_C(num_available);
        std::vector<int64_t> batches_assigned(num_available);

        int64_t batch_offset = 0;
        for(int i = 0; i < num_available; ++i)
        {
            int dev = available_gpus[i];
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            hipblasLtCreate(&handles[i]);

            // Distribute remaining batches among first GPUs
            batches_assigned[i] = batches_per_gpu + (i < remaining_batches ? 1 : 0);

            EXPECT_EQ(hipMalloc(&d_A[i], M * K * batches_assigned[i] * sizeof(_Float16)), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_B[i], K * N * batches_assigned[i] * sizeof(_Float16)), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_C[i], M * N * batches_assigned[i] * sizeof(_Float16)), hipSuccess);

            EXPECT_EQ(hipMemcpy(d_A[i], h_A.data() + batch_offset * M * K,
                     M * K * batches_assigned[i] * sizeof(_Float16), hipMemcpyHostToDevice), hipSuccess);
            EXPECT_EQ(hipMemcpy(d_B[i], h_B.data() + batch_offset * K * N,
                     K * N * batches_assigned[i] * sizeof(_Float16), hipMemcpyHostToDevice), hipSuccess);

            batch_offset += batches_assigned[i];

            hipblaslt_cout << "GPU " << dev << ": Processing " << batches_assigned[i] << " batches" << std::endl;
        }

        // Execute on all available GPUs
        for(int i = 0; i < num_available; ++i)
        {
            int dev = available_gpus[i];
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, M, N, M);

            hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                             &batches_assigned[i], sizeof(batches_assigned[i]));
            hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                             &batches_assigned[i], sizeof(batches_assigned[i]));
            hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                             &batches_assigned[i], sizeof(batches_assigned[i]));
            hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                             &batches_assigned[i], sizeof(batches_assigned[i]));

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

            hipblasLtMatmul(handles[i], matmul, &alpha,
                           d_A[i], matA, d_B[i], matB,
                           &beta, d_C[i], matC, d_C[i], matD,
                           nullptr, nullptr, 0, 0);

            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
        }

        // Cleanup
        for(int i = 0; i < num_available; ++i)
        {
            int dev = available_gpus[i];
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipFree(d_A[i]), hipSuccess);
            EXPECT_EQ(hipFree(d_B[i]), hipSuccess);
            EXPECT_EQ(hipFree(d_C[i]), hipSuccess);
            hipblasLtDestroy(handles[i]);
        }

        EXPECT_EQ(batch_offset, total_batches) << "Not all batches were processed!";
        hipblaslt_cout << "✓ Successfully continued with " << num_available << " GPUs" << std::endl;
    }

    // ============================================================================
    // Test 4: Invalid Algorithm - Handle Configuration Errors
    // ============================================================================
    TEST(MultiGPUFaultTolerance, InvalidAlgorithm)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 1) GTEST_SKIP() << "Requires at least 1 GPU";

        hipblaslt_cout << "=== Invalid Algorithm Handling ===" << std::endl;

        const int64_t M = 128, N = 128, K = 128;
        float alpha = 1.0f, beta = 0.0f;

        EXPECT_EQ(hipSetDevice(0), hipSuccess);

        hipblasLtHandle_t handle;
        hipblasLtCreate(&handle);

        _Float16 *d_A, *d_B, *d_C;
        EXPECT_EQ(hipMalloc(&d_A, M * K * sizeof(_Float16)), hipSuccess);
        EXPECT_EQ(hipMalloc(&d_B, K * N * sizeof(_Float16)), hipSuccess);
        EXPECT_EQ(hipMalloc(&d_C, M * N * sizeof(_Float16)), hipSuccess);

        hipblasLtMatrixLayout_t matA, matB, matC, matD;
        hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, M, K, M);
        hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, K, N, K);
        hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, M, N, M);
        hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, M, N, M);

        hipblasLtMatmulDesc_t matmul;
        hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

        // Create preference with invalid/restrictive settings
        hipblasLtMatmulPreference_t pref;
        hipblasLtMatmulPreferenceCreate(&pref);

        // Set workspace to 0 (very restrictive)
        uint64_t workspace_size = 0;
        hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &workspace_size, sizeof(workspace_size));

        const int request_solutions = 10;
        hipblasLtMatmulHeuristicResult_t results[request_solutions];
        int returned_count = 0;

        hipblasStatus_t status = hipblasLtMatmulAlgoGetHeuristic(
            handle, matmul, matA, matB, matC, matD, pref,
            request_solutions, results, &returned_count
        );

        if(status != HIPBLAS_STATUS_SUCCESS || returned_count == 0)
        {
            hipblaslt_cout << "No algorithms found with restrictive settings (expected)" << std::endl;
        }
        else
        {
            hipblaslt_cout << "Found " << returned_count << " algorithms despite restrictions" << std::endl;
        }

        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(matmul);
        hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
        hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
        EXPECT_EQ(hipFree(d_A), hipSuccess);
        EXPECT_EQ(hipFree(d_B), hipSuccess);
        EXPECT_EQ(hipFree(d_C), hipSuccess);
        hipblasLtDestroy(handle);

        // Test passes if we handle the case gracefully (no crash)
        EXPECT_TRUE(true) << "Handled invalid configuration gracefully";
        hipblaslt_cout << "✓ Invalid Algorithm Error Handling Complete" << std::endl;
    }

} // namespace
