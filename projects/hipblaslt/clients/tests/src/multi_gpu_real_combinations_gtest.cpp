/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Real Multi-GPU Tests - Extensive Parameter Combinations
 * Tests combinations of multiple parameters simultaneously
 * This is the most comprehensive test file covering realistic use cases
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
        hipGetDeviceCount(&numDevices);
        return numDevices;
    }

    // ============================================================================
    // Test 1: FP16 + Transpose NT + RELU - Row Partition
    // Combination: Data Type + Transpose + Epilogue + Distribution
    // ============================================================================
    TEST(MultiGPURealCombinations, FP16_NT_RELU_RowPartition)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Combination: FP16 + NT + RELU + Row Partition ===" << std::endl;

        const int64_t M_total = 1024, N = 512, K = 512;
        const int64_t M_per_gpu = M_total / numDevices;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<_Float16*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t M_local = (dev == numDevices - 1) ? (M_total - dev * M_per_gpu) : M_per_gpu;

            hipMalloc(&d_A[dev], M_local * K * sizeof(_Float16));
            hipMalloc(&d_B[dev], N * K * sizeof(_Float16)); // Transposed B: N×K
            hipMalloc(&d_C[dev], M_local * N * sizeof(_Float16));

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, M_local, K, M_local);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, K, N, N); // Transposed
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, M_local, N, M_local);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, M_local, N, M_local);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_16F);

            hipblasOperation_t opA = HIPBLAS_OP_N, opB = HIPBLAS_OP_T;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                           &opA, sizeof(opA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                           &opB, sizeof(opB));

            hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_RELU;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                           &epilogue, sizeof(epilogue));

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_A[dev], matA, d_B[dev], matB,
                           &beta, d_C[dev], matC, d_C[dev], matD,
                           nullptr, nullptr, 0, 0);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": FP16+NT+RELU, " << M_local << " rows" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "Completed 4-way combination test" << std::endl;
    }

    // ============================================================================
    // Test 2: BF16 + TT + Alpha=2.0/Beta=0.5 + GELU - Data Parallel
    // Combination: Data Type + Transpose + Scalars + Epilogue + Distribution
    // ============================================================================
    TEST(MultiGPURealCombinations, BF16_TT_Scalars_GELU_DataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Combination: BF16 + TT + Alpha/Beta + GELU + Data Parallel ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 2;
        float alpha = 2.0f, beta = 0.5f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<hip_bfloat16*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_A[dev], batches_per_gpu * K * M * sizeof(hip_bfloat16));
            hipMalloc(&d_B[dev], batches_per_gpu * N * K * sizeof(hip_bfloat16));
            hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(hip_bfloat16));

            for(int64_t b = 0; b < batches_per_gpu; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_16BF, M, K, K); // Transposed
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_16BF, K, N, N); // Transposed
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_16BF, M, N, M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_16BF, M, N, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_16BF);

                hipblasOperation_t opA = HIPBLAS_OP_T, opB = HIPBLAS_OP_T;
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                               &opA, sizeof(opA));
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                               &opB, sizeof(opB));

                hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_GELU;
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                               &epilogue, sizeof(epilogue));

                hipblasLtMatmul(handles[dev], matmul, &alpha,
                               d_A[dev] + b * K * M, matA,
                               d_B[dev] + b * N * K, matB,
                               &beta,
                               d_C[dev] + b * M * N, matC,
                               d_C[dev] + b * M * N, matD,
                               nullptr, nullptr, 0, 0);

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }

            hipDeviceSynchronize();
            hipblaslt_cout << "GPU " << dev << ": BF16+TT+Scalars+GELU, "
                          << batches_per_gpu << " batches" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "Completed 5-way combination test" << std::endl;
    }

    // ============================================================================
    // Test 3: FP32 + NN + Large Matrix + Batching + BIAS+RELU - Column Partition
    // Combination: Data Type + Transpose + Size + Batch + Epilogue + Distribution
    // ============================================================================
    TEST(MultiGPURealCombinations, FP32_NN_Large_Batch_BiasRELU_ColumnPartition)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Combination: FP32 + NN + Large + Batch + BIAS+RELU + Column Split ===" << std::endl;

        const int64_t M = 2048;
        const int64_t N_total = 2048, K = 1024;
        const int64_t N_per_gpu = N_total / numDevices;
        const int64_t batches_per_gpu = 4;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices), d_bias(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t N_local = (dev == numDevices - 1) ? (N_total - dev * N_per_gpu) : N_per_gpu;

            hipMalloc(&d_A[dev], batches_per_gpu * M * K * sizeof(float));
            hipMalloc(&d_B[dev], batches_per_gpu * K * N_local * sizeof(float));
            hipMalloc(&d_C[dev], batches_per_gpu * M * N_local * sizeof(float));
            hipMalloc(&d_bias[dev], M * sizeof(float));

            for(int64_t b = 0; b < batches_per_gpu; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N_local, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N_local, M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N_local, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasOperation_t opA = HIPBLAS_OP_N, opB = HIPBLAS_OP_N;
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                               &opA, sizeof(opA));
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                               &opB, sizeof(opB));

                hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_RELU_BIAS;
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                               &epilogue, sizeof(epilogue));

                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                               &d_bias[dev], sizeof(d_bias[dev]));

                hipblasLtMatmul(handles[dev], matmul, &alpha,
                               d_A[dev] + b * M * K, matA,
                               d_B[dev] + b * K * N_local, matB,
                               &beta,
                               d_C[dev] + b * M * N_local, matC,
                               d_C[dev] + b * M * N_local, matD,
                               nullptr, nullptr, 0, 0);

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }

            hipDeviceSynchronize();
            hipblaslt_cout << "GPU " << dev << ": Large matrix " << M << "×" << N_local << "×" << K
                          << ", " << batches_per_gpu << " batches" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]); hipFree(d_bias[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "Completed 6-way combination test with large matrices" << std::endl;
    }

    // ============================================================================
    // Test 4: All Parameters Different Per GPU (Maximum Variety)
    // Each GPU tests a unique combination of parameters
    // ============================================================================
    TEST(MultiGPURealCombinations, AllParametersDifferentPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== All Parameters Different Per GPU ===" << std::endl;

        struct GPUConfig {
            hipDataType dtype;
            hipblasOperation_t opA, opB;
            int64_t M, N, K;
            float alpha, beta;
            hipblasLtEpilogue_t epilogue;
            std::string name;
        };

        std::vector<GPUConfig> configs = {
            {HIP_R_32F, HIPBLAS_OP_N, HIPBLAS_OP_N, 256, 256, 256, 1.0f, 0.0f, HIPBLASLT_EPILOGUE_DEFAULT, "FP32+NN+256+DEFAULT"},
            {HIP_R_16F, HIPBLAS_OP_N, HIPBLAS_OP_T, 512, 512, 256, 2.0f, 0.5f, HIPBLASLT_EPILOGUE_RELU, "FP16+NT+512+RELU"},
            {HIP_R_16BF, HIPBLAS_OP_T, HIPBLAS_OP_N, 1024, 256, 512, 1.0f, 1.0f, HIPBLASLT_EPILOGUE_GELU, "BF16+TN+1024+GELU"},
            {HIP_R_32F, HIPBLAS_OP_T, HIPBLAS_OP_T, 128, 128, 128, 0.5f, 0.0f, HIPBLASLT_EPILOGUE_BIAS, "FP32+TT+128+BIAS"},
            {HIP_R_16F, HIPBLAS_OP_N, HIPBLAS_OP_N, 2048, 512, 512, 3.0f, 0.25f, HIPBLASLT_EPILOGUE_RELU_BIAS, "FP16+NN+2048+RELU_BIAS"},
            {HIP_R_32F, HIPBLAS_OP_N, HIPBLAS_OP_T, 512, 1024, 256, 1.5f, 0.75f, HIPBLASLT_EPILOGUE_GELU_BIAS, "FP32+NT+512x1024+GELU_BIAS"},
            {HIP_R_16BF, HIPBLAS_OP_N, HIPBLAS_OP_N, 256, 2048, 1024, 0.25f, 2.0f, HIPBLASLT_EPILOGUE_DEFAULT, "BF16+NN+256x2048+DEFAULT"},
            {HIP_R_16F, HIPBLAS_OP_T, HIPBLAS_OP_T, 512, 512, 512, 2.5f, 0.1f, HIPBLASLT_EPILOGUE_RELU, "FP16+TT+512+RELU"}
        };

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            int cfg_idx = dev % configs.size();
            auto& cfg = configs[cfg_idx];

            hipblaslt_cout << "GPU " << dev << ": " << cfg.name << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            size_t type_size = (cfg.dtype == HIP_R_32F) ? sizeof(float) : sizeof(_Float16);

            void *d_A, *d_B, *d_C, *d_bias = nullptr;
            int64_t dimA_rows = (cfg.opA == HIPBLAS_OP_N) ? cfg.M : cfg.K;
            int64_t dimA_cols = (cfg.opA == HIPBLAS_OP_N) ? cfg.K : cfg.M;
            int64_t dimB_rows = (cfg.opB == HIPBLAS_OP_N) ? cfg.K : cfg.N;
            int64_t dimB_cols = (cfg.opB == HIPBLAS_OP_N) ? cfg.N : cfg.K;

            hipMalloc(&d_A, dimA_rows * dimA_cols * type_size);
            hipMalloc(&d_B, dimB_rows * dimB_cols * type_size);
            hipMalloc(&d_C, cfg.M * cfg.N * type_size);

            if(cfg.epilogue == HIPBLASLT_EPILOGUE_BIAS ||
               cfg.epilogue == HIPBLASLT_EPILOGUE_RELU_BIAS ||
               cfg.epilogue == HIPBLASLT_EPILOGUE_GELU_BIAS)
            {
                hipMalloc(&d_bias, cfg.M * type_size);
            }

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            int64_t ldA = (cfg.opA == HIPBLAS_OP_N) ? dimA_rows : dimA_cols;
            int64_t ldB = (cfg.opB == HIPBLAS_OP_N) ? dimB_rows : dimB_cols;

            hipblasLtMatrixLayoutCreate(&matA, cfg.dtype, dimA_rows, dimA_cols, ldA);
            hipblasLtMatrixLayoutCreate(&matB, cfg.dtype, dimB_rows, dimB_cols, ldB);
            hipblasLtMatrixLayoutCreate(&matC, cfg.dtype, cfg.M, cfg.N, cfg.M);
            hipblasLtMatrixLayoutCreate(&matD, cfg.dtype, cfg.M, cfg.N, cfg.M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, cfg.dtype);

            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                           &cfg.opA, sizeof(cfg.opA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                           &cfg.opB, sizeof(cfg.opB));

            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                           &cfg.epilogue, sizeof(cfg.epilogue));

            if(d_bias != nullptr)
            {
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                               &d_bias, sizeof(d_bias));
            }

            hipblasLtMatmul(handle, matmul, &cfg.alpha,
                           d_A, matA, d_B, matB,
                           &cfg.beta, d_C, matC, d_C, matD,
                           nullptr, nullptr, 0, 0);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipFree(d_A); hipFree(d_B); hipFree(d_C);
            if(d_bias) hipFree(d_bias);
            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Tested maximum parameter variety across " << numDevices << " GPUs" << std::endl;
    }

    // ============================================================================
    // Test 5: Mixed Precision with Batching and Different Epilogues
    // ============================================================================
    TEST(MultiGPURealCombinations, MixedPrecision_Batching_Epilogues)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Mixed Precision + Batching + Epilogues ===" << std::endl;

        const int64_t M = 512, N = 512, K = 256;
        const int64_t batches_per_gpu = 4;
        float alpha = 1.5f, beta = 0.5f;

        std::vector<hipblasLtEpilogue_t> epilogues = {
            HIPBLASLT_EPILOGUE_DEFAULT,
            HIPBLASLT_EPILOGUE_RELU,
            HIPBLASLT_EPILOGUE_GELU,
            HIPBLASLT_EPILOGUE_BIAS
        };

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            int epilogue_idx = dev % epilogues.size();
            hipblasLtEpilogue_t epilogue = epilogues[epilogue_idx];

            hipblaslt_cout << "GPU " << dev << ": FP16->FP32 mixed precision, "
                          << batches_per_gpu << " batches, epilogue " << epilogue_idx << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            _Float16 *d_A, *d_B;
            float *d_C, *d_bias = nullptr;

            hipMalloc(&d_A, batches_per_gpu * M * K * sizeof(_Float16));
            hipMalloc(&d_B, batches_per_gpu * K * N * sizeof(_Float16));
            hipMalloc(&d_C, batches_per_gpu * M * N * sizeof(float));

            if(epilogue == HIPBLASLT_EPILOGUE_BIAS)
            {
                hipMalloc(&d_bias, M * sizeof(float));
            }

            for(int64_t b = 0; b < batches_per_gpu; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                               &epilogue, sizeof(epilogue));

                if(d_bias != nullptr)
                {
                    hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                   &d_bias, sizeof(d_bias));
                }

                hipblasLtMatmul(handle, matmul, &alpha,
                               d_A + b * M * K, matA,
                               d_B + b * K * N, matB,
                               &beta,
                               d_C + b * M * N, matC,
                               d_C + b * M * N, matD,
                               nullptr, nullptr, 0, 0);

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }

            hipDeviceSynchronize();

            hipFree(d_A); hipFree(d_B); hipFree(d_C);
            if(d_bias) hipFree(d_bias);
            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Tested mixed precision with batching and epilogues" << std::endl;
    }

} // namespace
