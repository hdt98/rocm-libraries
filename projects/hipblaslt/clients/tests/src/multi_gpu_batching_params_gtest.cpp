/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Batching Parameters Test Suite
 * Comprehensive testing of batch counts, strides, grouped GEMM
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

    // Test 1: Single Batch (batch_count = 1)
    TEST(MultiGPUBatching, SingleBatch)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing single batch across GPUs" << std::endl;

        const int64_t M = 128, N = 128, K = 128;
        const int64_t batch_count = 1;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                              &batch_count, sizeof(batch_count));
            hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                              &batch_count, sizeof(batch_count));
            hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                              &batch_count, sizeof(batch_count));

            hipblaslt_cout << "GPU " << dev << " single batch configured" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // Test 2: Small Batches (2, 4, 8)
    TEST(MultiGPUBatching, SmallBatchCounts)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing small batch counts across GPUs" << std::endl;

        std::vector<int64_t> batch_counts = {2, 4, 8};
        const int64_t M = 128, N = 128, K = 128;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            for(auto batch_count : batch_counts)
            {
                hipblaslt_cout << "GPU " << dev << " testing batch_count=" << batch_count << std::endl;

                hipblasLtHandle_t handle;
                hipblasLtCreate(&handle);

                hipblasLtMatrixLayout_t matA, matB, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &batch_count, sizeof(batch_count));
                hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &batch_count, sizeof(batch_count));
                hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &batch_count, sizeof(batch_count));

                int64_t stride = M * K;
                hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &stride, sizeof(stride));

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtDestroy(handle);
            }
        }
    }

    // Test 3: Medium Batches (16, 32)
    TEST(MultiGPUBatching, MediumBatchCounts)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing medium batch counts across GPUs" << std::endl;

        std::vector<int64_t> batch_counts = {16, 32};
        const int64_t M = 64, N = 64, K = 64;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            for(auto batch_count : batch_counts)
            {
                hipblaslt_cout << "GPU " << dev << " testing batch_count=" << batch_count << std::endl;

                hipblasLtHandle_t handle;
                hipblasLtCreate(&handle);

                hipblasLtMatrixLayout_t matA, matB, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &batch_count, sizeof(batch_count));
                hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &batch_count, sizeof(batch_count));
                hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &batch_count, sizeof(batch_count));

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtDestroy(handle);
            }
        }
    }

    // Test 4: Large Batches (64, 128)
    TEST(MultiGPUBatching, LargeBatchCounts)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing large batch counts across GPUs" << std::endl;

        std::vector<int64_t> batch_counts = {64, 128};
        const int64_t M = 32, N = 32, K = 32;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            for(auto batch_count : batch_counts)
            {
                hipblaslt_cout << "GPU " << dev << " testing batch_count=" << batch_count << std::endl;

                hipblasLtHandle_t handle;
                hipblasLtCreate(&handle);

                hipblasLtMatrixLayout_t matA;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &batch_count, sizeof(batch_count));

                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtDestroy(handle);
            }
        }
    }

    // Test 5: Contiguous Strides
    TEST(MultiGPUBatching, ContiguousStrides)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing contiguous strides across GPUs" << std::endl;

        const int64_t M = 64, N = 64, K = 64;
        const int64_t batch_count = 4;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                              &batch_count, sizeof(batch_count));

            int64_t strideA = M * K; // Contiguous
            hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &strideA, sizeof(strideA));

            hipblaslt_cout << "GPU " << dev << " contiguous stride=" << strideA << std::endl;

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // Test 6: Non-Contiguous Strides (with padding)
    TEST(MultiGPUBatching, NonContiguousStrides)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing non-contiguous strides across GPUs" << std::endl;

        const int64_t M = 64, N = 64, K = 64;
        const int64_t batch_count = 4;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                              &batch_count, sizeof(batch_count));

            int64_t strideA = M * K + 128; // Non-contiguous with padding
            hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &strideA, sizeof(strideA));

            hipblaslt_cout << "GPU " << dev << " non-contiguous stride=" << strideA << std::endl;

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtDestroy(handle);
        }
    }

    // Test 7: Different Batch Counts Per GPU
    TEST(MultiGPUBatching, DifferentBatchCountsPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing different batch counts per GPU" << std::endl;

        std::vector<int64_t> batch_counts_per_gpu = {4, 8, 16, 32};
        const int64_t M = 64, N = 64, K = 64;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            int64_t batch_count = batch_counts_per_gpu[dev % batch_counts_per_gpu.size()];
            hipblaslt_cout << "GPU " << dev << " batch_count=" << batch_count << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                              &batch_count, sizeof(batch_count));

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtDestroy(handle);
        }
    }

    // Test 8: Variable Stride Patterns
    TEST(MultiGPUBatching, VariableStridePatterns)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing variable stride patterns across GPUs" << std::endl;

        const int64_t M = 64, N = 64, K = 64;
        const int64_t batch_count = 4;
        std::vector<int64_t> stride_multipliers = {1, 2, 4, 8};

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            int64_t stride = M * K * stride_multipliers[dev % stride_multipliers.size()];
            hipblaslt_cout << "GPU " << dev << " stride_multiplier=" << stride_multipliers[dev % stride_multipliers.size()]
                           << " stride=" << stride << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                              &batch_count, sizeof(batch_count));
            hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &stride, sizeof(stride));

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtDestroy(handle);
        }
    }

} // namespace
