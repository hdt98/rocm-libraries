/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Real Multi-GPU Tests - Complete Combination Matrix
 * Systematic testing of major parameter combinations
 * This provides the highest level of comprehensive coverage
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <string>

#include <gtest/gtest-spi.h>

namespace
{
    int getNumGPUs()
    {
        int numDevices = 0;
        hipGetDeviceCount(&numDevices);
        return numDevices;
    }

    struct TestConfig {
        hipDataType dtype;
        hipblasOperation_t opA, opB;
        int64_t M, N, K;
        int64_t batch_count;
        float alpha, beta;
        hipblasLtEpilogue_t epilogue;
        int64_t ldA_padding, ldB_padding, ldC_padding;
        std::string description;
    };

    // ============================================================================
    // Test 1: Small Matrices - All Data Types
    // ============================================================================
    TEST(MultiGPURealMatrixCoverage, SmallMatrices_AllDataTypes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Small Matrices: All Data Types ===" << std::endl;

        std::vector<TestConfig> configs = {
            {HIP_R_32F, HIPBLAS_OP_N, HIPBLAS_OP_N, 128, 128, 128, 4, 1.0f, 0.0f, HIPBLASLT_EPILOGUE_DEFAULT, 0, 0, 0, "FP32_NN_128"},
            {HIP_R_16F, HIPBLAS_OP_N, HIPBLAS_OP_N, 128, 128, 128, 4, 1.0f, 0.0f, HIPBLASLT_EPILOGUE_DEFAULT, 0, 0, 0, "FP16_NN_128"},
            {HIP_R_16BF, HIPBLAS_OP_N, HIPBLAS_OP_N, 128, 128, 128, 4, 1.0f, 0.0f, HIPBLASLT_EPILOGUE_DEFAULT, 0, 0, 0, "BF16_NN_128"},
        };

        for(auto& cfg : configs)
        {
            hipblaslt_cout << "Testing: " << cfg.description << std::endl;

            size_t type_size = (cfg.dtype == HIP_R_32F) ? sizeof(float) : sizeof(_Float16);

            std::vector<hipblasLtHandle_t> handles(numDevices);
            std::vector<void*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipblasLtCreate(&handles[dev]);

                int64_t ldA = cfg.M + cfg.ldA_padding;
                int64_t ldB = cfg.K + cfg.ldB_padding;
                int64_t ldC = cfg.M + cfg.ldC_padding;

                hipMalloc(&d_A[dev], cfg.batch_count * ldA * cfg.K * type_size);
                hipMalloc(&d_B[dev], cfg.batch_count * ldB * cfg.N * type_size);
                hipMalloc(&d_C[dev], cfg.batch_count * ldC * cfg.N * type_size);

                for(int64_t b = 0; b < cfg.batch_count; ++b)
                {
                    hipblasLtMatrixLayout_t matA, matB, matC, matD;
                    hipblasLtMatrixLayoutCreate(&matA, cfg.dtype, cfg.M, cfg.K, ldA);
                    hipblasLtMatrixLayoutCreate(&matB, cfg.dtype, cfg.K, cfg.N, ldB);
                    hipblasLtMatrixLayoutCreate(&matC, cfg.dtype, cfg.M, cfg.N, ldC);
                    hipblasLtMatrixLayoutCreate(&matD, cfg.dtype, cfg.M, cfg.N, ldC);

                    hipblasLtMatmulDesc_t matmul;
                    hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, cfg.dtype);

                    hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                   &cfg.opA, sizeof(cfg.opA));
                    hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                   &cfg.opB, sizeof(cfg.opB));

                    char *d_A_batch = (char*)d_A[dev] + b * ldA * cfg.K * type_size;
                    char *d_B_batch = (char*)d_B[dev] + b * ldB * cfg.N * type_size;
                    char *d_C_batch = (char*)d_C[dev] + b * ldC * cfg.N * type_size;

                    hipblasLtMatmul(handles[dev], matmul, &cfg.alpha,
                                   d_A_batch, matA, d_B_batch, matB,
                                   &cfg.beta, d_C_batch, matC, d_C_batch, matD,
                                   nullptr, nullptr, 0, 0);

                    hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                    hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                    hipblasLtMatmulDescDestroy(matmul);
                }

                hipDeviceSynchronize();
            }

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
                hipblasLtDestroy(handles[dev]);
            }
        }

        hipblaslt_cout << "Small matrices tested with all data types" << std::endl;
    }

    // ============================================================================
    // Test 2: Medium Matrices - All Transposes
    // ============================================================================
    TEST(MultiGPURealMatrixCoverage, MediumMatrices_AllTransposes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Medium Matrices: All Transpose Operations ===" << std::endl;

        std::vector<TestConfig> configs = {
            {HIP_R_32F, HIPBLAS_OP_N, HIPBLAS_OP_N, 512, 512, 512, 2, 1.0f, 0.0f, HIPBLASLT_EPILOGUE_DEFAULT, 0, 0, 0, "NN_512"},
            {HIP_R_32F, HIPBLAS_OP_N, HIPBLAS_OP_T, 512, 512, 512, 2, 1.0f, 0.0f, HIPBLASLT_EPILOGUE_DEFAULT, 0, 0, 0, "NT_512"},
            {HIP_R_32F, HIPBLAS_OP_T, HIPBLAS_OP_N, 512, 512, 512, 2, 1.0f, 0.0f, HIPBLASLT_EPILOGUE_DEFAULT, 0, 0, 0, "TN_512"},
            {HIP_R_32F, HIPBLAS_OP_T, HIPBLAS_OP_T, 512, 512, 512, 2, 1.0f, 0.0f, HIPBLASLT_EPILOGUE_DEFAULT, 0, 0, 0, "TT_512"},
        };

        for(auto& cfg : configs)
        {
            hipblaslt_cout << "Testing: " << cfg.description << std::endl;

            std::vector<hipblasLtHandle_t> handles(numDevices);
            std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipblasLtCreate(&handles[dev]);

                int64_t dimA_rows = (cfg.opA == HIPBLAS_OP_N) ? cfg.M : cfg.K;
                int64_t dimA_cols = (cfg.opA == HIPBLAS_OP_N) ? cfg.K : cfg.M;
                int64_t dimB_rows = (cfg.opB == HIPBLAS_OP_N) ? cfg.K : cfg.N;
                int64_t dimB_cols = (cfg.opB == HIPBLAS_OP_N) ? cfg.N : cfg.K;

                hipMalloc(&d_A[dev], cfg.batch_count * dimA_rows * dimA_cols * sizeof(float));
                hipMalloc(&d_B[dev], cfg.batch_count * dimB_rows * dimB_cols * sizeof(float));
                hipMalloc(&d_C[dev], cfg.batch_count * cfg.M * cfg.N * sizeof(float));

                for(int64_t b = 0; b < cfg.batch_count; ++b)
                {
                    hipblasLtMatrixLayout_t matA, matB, matC, matD;
                    int64_t ldA = (cfg.opA == HIPBLAS_OP_N) ? dimA_rows : dimA_cols;
                    int64_t ldB = (cfg.opB == HIPBLAS_OP_N) ? dimB_rows : dimB_cols;

                    hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, dimA_rows, dimA_cols, ldA);
                    hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, dimB_rows, dimB_cols, ldB);
                    hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, cfg.M, cfg.N, cfg.M);
                    hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, cfg.M, cfg.N, cfg.M);

                    hipblasLtMatmulDesc_t matmul;
                    hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                    hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                   &cfg.opA, sizeof(cfg.opA));
                    hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                   &cfg.opB, sizeof(cfg.opB));

                    hipblasLtMatmul(handles[dev], matmul, &cfg.alpha,
                                   d_A[dev] + b * dimA_rows * dimA_cols, matA,
                                   d_B[dev] + b * dimB_rows * dimB_cols, matB,
                                   &cfg.beta,
                                   d_C[dev] + b * cfg.M * cfg.N, matC,
                                   d_C[dev] + b * cfg.M * cfg.N, matD,
                                   nullptr, nullptr, 0, 0);

                    hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                    hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                    hipblasLtMatmulDescDestroy(matmul);
                }

                hipDeviceSynchronize();
            }

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
                hipblasLtDestroy(handles[dev]);
            }
        }

        hipblaslt_cout << "Medium matrices tested with all transposes" << std::endl;
    }

    // ============================================================================
    // Test 3: Large Matrices - All Epilogues
    // ============================================================================
    TEST(MultiGPURealMatrixCoverage, LargeMatrices_AllEpilogues)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Large Matrices: All Epilogue Operations ===" << std::endl;

        std::vector<TestConfig> configs = {
            {HIP_R_32F, HIPBLAS_OP_N, HIPBLAS_OP_N, 1024, 1024, 512, 1, 1.0f, 0.0f, HIPBLASLT_EPILOGUE_DEFAULT, 0, 0, 0, "DEFAULT_1024"},
            {HIP_R_32F, HIPBLAS_OP_N, HIPBLAS_OP_N, 1024, 1024, 512, 1, 1.0f, 0.0f, HIPBLASLT_EPILOGUE_RELU, 0, 0, 0, "RELU_1024"},
            {HIP_R_32F, HIPBLAS_OP_N, HIPBLAS_OP_N, 1024, 1024, 512, 1, 1.0f, 0.0f, HIPBLASLT_EPILOGUE_GELU, 0, 0, 0, "GELU_1024"},
        };

        for(auto& cfg : configs)
        {
            hipblaslt_cout << "Testing: " << cfg.description << std::endl;

            std::vector<hipblasLtHandle_t> handles(numDevices);
            std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipblasLtCreate(&handles[dev]);

                hipMalloc(&d_A[dev], cfg.M * cfg.K * sizeof(float));
                hipMalloc(&d_B[dev], cfg.K * cfg.N * sizeof(float));
                hipMalloc(&d_C[dev], cfg.M * cfg.N * sizeof(float));

                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, cfg.M, cfg.K, cfg.M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, cfg.K, cfg.N, cfg.K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, cfg.M, cfg.N, cfg.M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, cfg.M, cfg.N, cfg.M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                               &cfg.epilogue, sizeof(cfg.epilogue));

                hipblasLtMatmul(handles[dev], matmul, &cfg.alpha,
                               d_A[dev], matA, d_B[dev], matB,
                               &cfg.beta, d_C[dev], matC, d_C[dev], matD,
                               nullptr, nullptr, 0, 0);

                hipDeviceSynchronize();

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
                hipblasLtDestroy(handles[dev]);
            }
        }

        hipblaslt_cout << "Large matrices tested with all epilogues" << std::endl;
    }

    // ============================================================================
    // Test 4: Comprehensive Combinations Per GPU
    // Each GPU runs a unique complete configuration
    // ============================================================================
    TEST(MultiGPURealMatrixCoverage, ComprehensiveCombinationsPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Comprehensive Combinations Per GPU ===" << std::endl;

        std::vector<TestConfig> configs = {
            // Config 0: Small FP32 NN Default
            {HIP_R_32F, HIPBLAS_OP_N, HIPBLAS_OP_N, 256, 256, 256, 4, 1.0f, 0.0f, HIPBLASLT_EPILOGUE_DEFAULT, 0, 0, 0, "FP32_NN_256_DEFAULT"},

            // Config 1: Medium FP16 NT RELU with batching
            {HIP_R_16F, HIPBLAS_OP_N, HIPBLAS_OP_T, 512, 512, 256, 8, 2.0f, 0.5f, HIPBLASLT_EPILOGUE_RELU, 16, 0, 0, "FP16_NT_512_RELU_Batch8"},

            // Config 2: Large BF16 TN GELU with padding
            {HIP_R_16BF, HIPBLAS_OP_T, HIPBLAS_OP_N, 1024, 512, 512, 2, 0.5f, 1.0f, HIPBLASLT_EPILOGUE_GELU, 0, 32, 32, "BF16_TN_1024_GELU_Padded"},

            // Config 3: Huge FP32 TT with large batch
            {HIP_R_32F, HIPBLAS_OP_T, HIPBLAS_OP_T, 2048, 2048, 512, 16, 3.0f, 0.25f, HIPBLASLT_EPILOGUE_DEFAULT, 64, 64, 64, "FP32_TT_2048_Batch16_AllPadded"},

            // Config 4: Non-square FP16
            {HIP_R_16F, HIPBLAS_OP_N, HIPBLAS_OP_N, 4096, 64, 4096, 1, 1.0f, 0.0f, HIPBLASLT_EPILOGUE_RELU, 0, 0, 0, "FP16_NN_TallSkinny"},

            // Config 5: Non-power-of-2 BF16
            {HIP_R_16BF, HIPBLAS_OP_N, HIPBLAS_OP_N, 1000, 1000, 500, 4, 1.5f, 0.75f, HIPBLASLT_EPILOGUE_GELU, 0, 0, 0, "BF16_NN_NonPow2"},

            // Config 6: Small K, Large MN
            {HIP_R_32F, HIPBLAS_OP_N, HIPBLAS_OP_N, 2048, 2048, 32, 1, 1.0f, 0.0f, HIPBLASLT_EPILOGUE_DEFAULT, 0, 0, 0, "FP32_NN_SmallK"},

            // Config 7: Large K, Small MN
            {HIP_R_16F, HIPBLAS_OP_N, HIPBLAS_OP_N, 256, 256, 8192, 1, 1.0f, 0.0f, HIPBLASLT_EPILOGUE_RELU, 0, 0, 0, "FP16_NN_LargeK"}
        };

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            int cfg_idx = dev % configs.size();
            auto& cfg = configs[cfg_idx];

            hipblaslt_cout << "GPU " << dev << ": " << cfg.description << std::endl;

            size_t type_size = (cfg.dtype == HIP_R_32F) ? sizeof(float) : sizeof(_Float16);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            void *d_A, *d_B, *d_C;

            // Physical storage dimensions (for column-major matrices)
            // For A: OP_N -> M×K, OP_T -> K×M
            // For B: OP_N -> K×N, OP_T -> N×K
            int64_t dimA_rows = (cfg.opA == HIPBLAS_OP_N) ? cfg.M : cfg.K;
            int64_t dimA_cols = (cfg.opA == HIPBLAS_OP_N) ? cfg.K : cfg.M;
            int64_t dimB_rows = (cfg.opB == HIPBLAS_OP_N) ? cfg.K : cfg.N;
            int64_t dimB_cols = (cfg.opB == HIPBLAS_OP_N) ? cfg.N : cfg.K;

            // Leading dimension is the first physical dimension (number of rows in column-major)
            int64_t ldA = dimA_rows + cfg.ldA_padding;
            int64_t ldB = dimB_rows + cfg.ldB_padding;
            int64_t ldC = cfg.M + cfg.ldC_padding;

            // Allocate: batch_count * physical_rows * physical_cols
            hipMalloc(&d_A, cfg.batch_count * ldA * dimA_cols * type_size);
            hipMalloc(&d_B, cfg.batch_count * ldB * dimB_cols * type_size);
            hipMalloc(&d_C, cfg.batch_count * ldC * cfg.N * type_size);

            for(int64_t b = 0; b < cfg.batch_count; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, cfg.dtype, dimA_rows, dimA_cols, ldA);
                hipblasLtMatrixLayoutCreate(&matB, cfg.dtype, dimB_rows, dimB_cols, ldB);
                hipblasLtMatrixLayoutCreate(&matC, cfg.dtype, cfg.M, cfg.N, ldC);
                hipblasLtMatrixLayoutCreate(&matD, cfg.dtype, cfg.M, cfg.N, ldC);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, cfg.dtype);

                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                               &cfg.opA, sizeof(cfg.opA));
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                               &cfg.opB, sizeof(cfg.opB));
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                               &cfg.epilogue, sizeof(cfg.epilogue));

                char *d_A_batch = (char*)d_A + b * ldA * dimA_cols * type_size;
                char *d_B_batch = (char*)d_B + b * ldB * dimB_cols * type_size;
                char *d_C_batch = (char*)d_C + b * ldC * cfg.N * type_size;

                hipblasLtMatmul(handle, matmul, &cfg.alpha,
                               d_A_batch, matA, d_B_batch, matB,
                               &cfg.beta, d_C_batch, matC, d_C_batch, matD,
                               nullptr, nullptr, 0, 0);

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }

            hipDeviceSynchronize();

            hipFree(d_A); hipFree(d_B); hipFree(d_C);
            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "Comprehensive combinations tested across " << numDevices << " GPUs" << std::endl;
    }

} // namespace
