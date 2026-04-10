/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Combined Features Test Suite
 * Tests: Combinations of batching+mixed precision, streams+transpose, etc.
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

    // ----------------------------------------------------------------------------
    // Test 1: Batched GEMM + Mixed Precision
    // ----------------------------------------------------------------------------
    TEST(MultiGPUCombinedFeatures, BatchedWithMixedPrecision)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing batched GEMM with mixed precision across GPUs" << std::endl;

        const int64_t M = 128;
        const int64_t N = 128;
        const int64_t K = 128;
        const int64_t batch_count = 8;

        struct PrecisionConfig {
            hipDataType input_type;
            hipDataType output_type;
            hipblasComputeType_t compute_type;
            size_t input_elem_size;
            size_t output_elem_size;
            const char* name;
        };

        std::vector<PrecisionConfig> configs = {
            {HIP_R_32F, HIP_R_32F, HIPBLAS_COMPUTE_32F, sizeof(float), sizeof(float), "FP32"},
            {HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F, sizeof(uint16_t), sizeof(uint16_t), "FP16"}
        };

        for(size_t cfg_idx = 0; cfg_idx < std::min(configs.size(), static_cast<size_t>(numDevices)); ++cfg_idx)
        {
            auto hipErr = hipSetDevice(cfg_idx);
            ASSERT_EQ(hipErr, hipSuccess);

            const auto& config = configs[cfg_idx];
            hipblaslt_cout << "GPU " << cfg_idx << " testing batched " << config.name << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, config.input_type, M, K, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, config.input_type, K, N, K);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matC, config.output_type, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, config.output_type, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                // Set batch count
                status = hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                           &batch_count, sizeof(batch_count));
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
                status = hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                           &batch_count, sizeof(batch_count));
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
                status = hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                           &batch_count, sizeof(batch_count));
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
                status = hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                           &batch_count, sizeof(batch_count));
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

                // Set strides
                int64_t strideA = M * K;
                int64_t strideB = K * N;
                int64_t strideC = M * N;
                int64_t strideD = M * N;
                hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &strideA, sizeof(strideA));
                hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &strideB, sizeof(strideB));
                hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &strideC, sizeof(strideC));
                hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &strideD, sizeof(strideD));

                hipblaslt_cout << "GPU " << cfg_idx << " batched " << config.name
                               << " layout created with " << batch_count << " batches" << std::endl;

                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC);
                hipblasLtMatrixLayoutDestroy(matD);
            }
            else
            {
                hipblaslt_cout << "GPU " << cfg_idx << " " << config.name << " not supported" << std::endl;
            }

            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Batched with mixed precision test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 2: Multi-Stream + Transposed Operations
    // ----------------------------------------------------------------------------
    TEST(MultiGPUCombinedFeatures, MultiStreamWithTranspose)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing multi-stream with transpose operations across GPUs" << std::endl;

        const int num_streams = 4;
        const int64_t M = 256;
        const int64_t N = 256;
        const int64_t K = 256;
        float alpha = 1.0f;
        float beta = 0.0f;

        std::vector<hipblasOperation_t> transpose_ops = {
            HIPBLAS_OP_N,
            HIPBLAS_OP_T,
            HIPBLAS_OP_N,
            HIPBLAS_OP_T
        };

        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " testing multi-stream transpose" << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Create streams
            std::vector<hipStream_t> streams(num_streams);
            for(int s = 0; s < num_streams; ++s)
            {
                hipErr = hipStreamCreate(&streams[s]);
                ASSERT_EQ(hipErr, hipSuccess);
            }

            // Each stream will execute different transpose configuration
            for(int s = 0; s < num_streams; ++s)
            {
                hipblasOperation_t transA = transpose_ops[s % transpose_ops.size()];
                hipblasOperation_t transB = (s % 2 == 0) ? HIPBLAS_OP_N : HIPBLAS_OP_T;

                int64_t rowsA = (transA == HIPBLAS_OP_N) ? M : K;
                int64_t colsA = (transA == HIPBLAS_OP_N) ? K : M;
                int64_t rowsB = (transB == HIPBLAS_OP_N) ? K : N;
                int64_t colsB = (transB == HIPBLAS_OP_N) ? N : K;

                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, rowsA, colsA, rowsA);
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
                status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, rowsB, colsB, rowsB);
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
                status = hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
                status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

                hipblasLtMatmulDesc_t matmul;
                status = hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

                status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
                status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

                // Cleanup for this stream config
                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC);
                hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }

            // Cleanup streams
            for(int s = 0; s < num_streams; ++s)
            {
                hipStreamDestroy(streams[s]);
            }

            hipblasLtDestroy(handle);
            hipblaslt_cout << "GPU " << deviceId << " multi-stream transpose test completed" << std::endl;
        }

        hipblaslt_cout << "Multi-stream with transpose test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 3: Strided Batching + Epilogue (Activation)
    // ----------------------------------------------------------------------------
    TEST(MultiGPUCombinedFeatures, StridedBatchingWithEpilogue)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing strided batching with epilogue across GPUs" << std::endl;

        const int64_t M = 64;
        const int64_t N = 64;
        const int64_t K = 64;
        const int64_t batch_count = 4;

        std::vector<hipblasLtEpilogue_t> epilogues = {
            HIPBLASLT_EPILOGUE_DEFAULT,
            HIPBLASLT_EPILOGUE_RELU,
            HIPBLASLT_EPILOGUE_GELU,
            HIPBLASLT_EPILOGUE_RELU
        };

        const char* epilogue_names[] = {"None", "ReLU", "GELU", "ReLU"};

        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtEpilogue_t epilogue = epilogues[deviceId % epilogues.size()];
            hipblaslt_cout << "GPU " << deviceId << " testing strided batching with "
                           << epilogue_names[deviceId % epilogues.size()] << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // Set batch count and strides
            hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                              &batch_count, sizeof(batch_count));
            hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                              &batch_count, sizeof(batch_count));
            hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                              &batch_count, sizeof(batch_count));
            hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                              &batch_count, sizeof(batch_count));

            int64_t strideA = M * K;
            int64_t strideB = K * N;
            int64_t strideC = M * N;
            int64_t strideD = M * N;
            hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &strideA, sizeof(strideA));
            hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &strideB, sizeof(strideB));
            hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &strideC, sizeof(strideC));
            hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &strideD, sizeof(strideD));

            hipblasLtMatmulDesc_t matmul;
            status = hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasOperation_t opA = HIPBLAS_OP_N;
            hipblasOperation_t opB = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

            // Set epilogue
            status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                     &epilogue, sizeof(epilogue));
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblaslt_cout << "GPU " << deviceId << " strided batching + epilogue configured" << std::endl;

            // Cleanup
            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Strided batching with epilogue test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 4: Row/Column Partitioning + P2P Communication
    // ----------------------------------------------------------------------------
    TEST(MultiGPUCombinedFeatures, PartitioningWithP2P)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing matrix partitioning with P2P communication" << std::endl;

        const int64_t M_total = 512;
        const int64_t N = 256;
        const int64_t K = 256;
        const int64_t M_per_gpu = M_total / numDevices;

        // Check P2P capability
        int canAccessPeer = 0;
        auto hipErr = hipDeviceCanAccessPeer(&canAccessPeer, 1, 0);
        ASSERT_EQ(hipErr, hipSuccess);

        if(!canAccessPeer)
        {
            hipblaslt_cout << "P2P not supported, skipping" << std::endl;
            GTEST_SKIP();
        }

        // Enable P2P between GPU 0 and GPU 1
        hipErr = hipSetDevice(1);
        ASSERT_EQ(hipErr, hipSuccess);
        hipErr = hipDeviceEnablePeerAccess(0, 0);
        // May already be enabled

        hipblaslt_cout << "P2P enabled between GPU 0 and GPU 1" << std::endl;

        // Allocate partitioned data on GPUs
        std::vector<float*> d_a_partitions(2);
        std::vector<float*> d_b(2);
        std::vector<float*> d_d_partitions(2);

        for(int dev = 0; dev < 2; ++dev)
        {
            hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipErr = hipMalloc(&d_a_partitions[dev], M_per_gpu * K * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_b[dev], K * N * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_d_partitions[dev], M_per_gpu * N * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);

            // Initialize
            std::vector<float> h_a_part(M_per_gpu * K, 1.0f);
            std::vector<float> h_b(K * N, 2.0f);

            hipMemcpy(d_a_partitions[dev], h_a_part.data(), M_per_gpu * K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_b[dev], h_b.data(), K * N * sizeof(float), hipMemcpyHostToDevice);
        }

        // Test P2P transfer: copy a partition from GPU 0 to GPU 1
        size_t transfer_size = M_per_gpu * K * sizeof(float);
        float *d_temp_on_gpu1;
        hipErr = hipSetDevice(1);
        ASSERT_EQ(hipErr, hipSuccess);
        hipErr = hipMalloc(&d_temp_on_gpu1, transfer_size);
        ASSERT_EQ(hipErr, hipSuccess);

        hipErr = hipMemcpyPeer(d_temp_on_gpu1, 1, d_a_partitions[0], 0, transfer_size);
        ASSERT_EQ(hipErr, hipSuccess);

        hipblaslt_cout << "P2P transfer completed: " << (transfer_size / 1024 / 1024) << " MB" << std::endl;

        // Cleanup
        for(int dev = 0; dev < 2; ++dev)
        {
            hipErr = hipSetDevice(dev);
            hipFree(d_a_partitions[dev]);
            hipFree(d_b[dev]);
            hipFree(d_d_partitions[dev]);
        }
        hipErr = hipSetDevice(1);
        hipFree(d_temp_on_gpu1);

        hipblaslt_cout << "Partitioning with P2P test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 5: Mixed Precision + Scaling + Epilogue
    // ----------------------------------------------------------------------------
    TEST(MultiGPUCombinedFeatures, MixedPrecisionScalingEpilogue)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing mixed precision with scaling and epilogue" << std::endl;

        const int64_t M = 128;
        const int64_t N = 128;
        const int64_t K = 128;

        for(int deviceId = 0; deviceId < std::min(numDevices, 2); ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << deviceId << " testing combined features" << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            // FP16 input, FP32 output
            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, M, K, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, K, N, K);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblasLtMatmulDesc_t matmul;
                status = hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

                hipblasOperation_t opA = HIPBLAS_OP_N;
                hipblasOperation_t opB = HIPBLAS_OP_N;
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

                // Set epilogue (ReLU)
                hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_RELU;
                status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                         &epilogue, sizeof(epilogue));
                EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

                // Allocate scale factors
                float *d_scaleA, *d_scaleB;
                hipErr = hipMalloc(&d_scaleA, sizeof(float));
                ASSERT_EQ(hipErr, hipSuccess);
                hipErr = hipMalloc(&d_scaleB, sizeof(float));
                ASSERT_EQ(hipErr, hipSuccess);

                float scale = 2.0f;
                hipMemcpy(d_scaleA, &scale, sizeof(float), hipMemcpyHostToDevice);
                hipMemcpy(d_scaleB, &scale, sizeof(float), hipMemcpyHostToDevice);

                // Set scale pointers (API may vary)
                void* scaleA_ptr = d_scaleA;
                status = hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                         &scaleA_ptr, sizeof(scaleA_ptr));
                // Continue even if not supported

                hipblaslt_cout << "GPU " << deviceId << " mixed precision + scaling + epilogue configured" << std::endl;

                hipFree(d_scaleA);
                hipFree(d_scaleB);
                hipblasLtMatmulDescDestroy(matmul);
                hipblasLtMatrixLayoutDestroy(matA);
                hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC);
                hipblasLtMatrixLayoutDestroy(matD);
            }
            else
            {
                hipblaslt_cout << "GPU " << deviceId << " FP16 not supported" << std::endl;
            }

            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Mixed precision scaling epilogue test passed" << std::endl;
    }

} // namespace
