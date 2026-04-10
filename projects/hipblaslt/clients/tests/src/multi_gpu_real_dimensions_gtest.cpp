/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Real Multi-GPU Tests - Comprehensive Dimension Coverage
 * Tests all matrix dimension categories: tiny, small, medium, large, huge
 * Non-power-of-2, extreme aspect ratios, varying K dimensions
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

    // ============================================================================
    // Test 1: Tiny Matrices (16-64)
    // ============================================================================
    TEST(MultiGPURealDimensions, TinyMatrices_DataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Tiny Matrices: 16×16×16, 32×32×32, 64×64×64 ===" << std::endl;

        std::vector<std::tuple<int64_t, int64_t, int64_t>> sizes = {
            {16, 16, 16},
            {32, 32, 32},
            {64, 64, 64}
        };

        float alpha = 1.0f, beta = 0.0f;
        const int64_t batches_per_gpu = 4;

        for(auto& [M, N, K] : sizes)
        {
            hipblaslt_cout << "Testing " << M << "×" << N << "×" << K << std::endl;

            std::vector<hipblasLtHandle_t> handles(numDevices);
            std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipblasLtCreate(&handles[dev]);

                hipMalloc(&d_A[dev], batches_per_gpu * M * K * sizeof(float));
                hipMalloc(&d_B[dev], batches_per_gpu * K * N * sizeof(float));
                hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(float));

                for(int64_t b = 0; b < batches_per_gpu; ++b)
                {
                    hipblasLtMatrixLayout_t matA, matB, matC, matD;
                    hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                    hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                    hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
                    hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                    hipblasLtMatmulDesc_t matmul;
                    hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                    hipblasLtMatmul(handles[dev], matmul, &alpha,
                                   d_A[dev] + b * M * K, matA,
                                   d_B[dev] + b * K * N, matB,
                                   &beta,
                                   d_C[dev] + b * M * N, matC,
                                   d_C[dev] + b * M * N, matD,
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

        hipblaslt_cout << "Tiny matrices tested across " << numDevices << " GPUs" << std::endl;
    }

    // ============================================================================
    // Test 2: Huge Matrices (4096-16384)
    // ============================================================================
    TEST(MultiGPURealDimensions, HugeMatrices_RowPartition)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Huge Matrices: 4096×4096×512, 8192×4096×1024 ===" << std::endl;

        std::vector<std::tuple<int64_t, int64_t, int64_t>> sizes = {
            {4096, 4096, 512},
            {8192, 4096, 1024}
        };

        float alpha = 1.0f, beta = 0.0f;

        for(auto& [M_total, N, K] : sizes)
        {
            hipblaslt_cout << "Testing " << M_total << "×" << N << "×" << K << std::endl;

            int64_t M_per_gpu = M_total / numDevices;

            std::vector<hipblasLtHandle_t> handles(numDevices);
            std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipblasLtCreate(&handles[dev]);

                int64_t M_local = (dev == numDevices - 1) ? (M_total - dev * M_per_gpu) : M_per_gpu;

                hipMalloc(&d_A[dev], M_local * K * sizeof(float));
                hipMalloc(&d_B[dev], K * N * sizeof(float));
                hipMalloc(&d_C[dev], M_local * N * sizeof(float));

                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M_local, K, M_local);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M_local, N, M_local);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M_local, N, M_local);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmul(handles[dev], matmul, &alpha,
                               d_A[dev], matA, d_B[dev], matB,
                               &beta, d_C[dev], matC, d_C[dev], matD,
                               nullptr, nullptr, 0, 0);

                hipDeviceSynchronize();

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);

                hipblaslt_cout << "GPU " << dev << ": " << M_local << " rows processed" << std::endl;
            }

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
                hipblasLtDestroy(handles[dev]);
            }
        }

        hipblaslt_cout << "Huge matrices tested" << std::endl;
    }

    // ============================================================================
    // Test 3: Non-Power-of-2 Dimensions
    // ============================================================================
    TEST(MultiGPURealDimensions, NonPowerOf2_DataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Non-Power-of-2: 100, 300, 500, 1000, 1500, 3000 ===" << std::endl;

        std::vector<std::tuple<int64_t, int64_t, int64_t>> sizes = {
            {100, 100, 100},
            {300, 300, 300},
            {500, 500, 500},
            {1000, 1000, 1000},
            {1500, 1500, 500},
            {3000, 1000, 500}
        };

        float alpha = 1.0f, beta = 0.0f;
        const int64_t batches_per_gpu = 2;

        for(auto& [M, N, K] : sizes)
        {
            hipblaslt_cout << "Testing " << M << "×" << N << "×" << K << std::endl;

            std::vector<hipblasLtHandle_t> handles(numDevices);
            std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipblasLtCreate(&handles[dev]);

                hipMalloc(&d_A[dev], batches_per_gpu * M * K * sizeof(float));
                hipMalloc(&d_B[dev], batches_per_gpu * K * N * sizeof(float));
                hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(float));

                for(int64_t b = 0; b < batches_per_gpu; ++b)
                {
                    hipblasLtMatrixLayout_t matA, matB, matC, matD;
                    hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                    hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                    hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
                    hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                    hipblasLtMatmulDesc_t matmul;
                    hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                    hipblasLtMatmul(handles[dev], matmul, &alpha,
                                   d_A[dev] + b * M * K, matA,
                                   d_B[dev] + b * K * N, matB,
                                   &beta,
                                   d_C[dev] + b * M * N, matC,
                                   d_C[dev] + b * M * N, matD,
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

        hipblaslt_cout << "Non-power-of-2 dimensions tested" << std::endl;
    }

    // ============================================================================
    // Test 4: Extreme Aspect Ratios - Tall & Skinny
    // ============================================================================
    TEST(MultiGPURealDimensions, TallSkinny_RowPartition)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Tall & Skinny: High M, Low N ===" << std::endl;

        std::vector<std::tuple<int64_t, int64_t, int64_t>> sizes = {
            {4096, 64, 4096},   // Very tall
            {8192, 128, 512},   // Extremely tall
            {2048, 32, 2048},   // Tall with small N
            {16384, 64, 1024}   // Extreme
        };

        float alpha = 1.0f, beta = 0.0f;

        for(auto& [M_total, N, K] : sizes)
        {
            hipblaslt_cout << "Testing " << M_total << "×" << N << "×" << K << std::endl;

            int64_t M_per_gpu = M_total / numDevices;

            std::vector<hipblasLtHandle_t> handles(numDevices);
            std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipblasLtCreate(&handles[dev]);

                int64_t M_local = (dev == numDevices - 1) ? (M_total - dev * M_per_gpu) : M_per_gpu;

                hipMalloc(&d_A[dev], M_local * K * sizeof(float));
                hipMalloc(&d_B[dev], K * N * sizeof(float));
                hipMalloc(&d_C[dev], M_local * N * sizeof(float));

                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M_local, K, M_local);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M_local, N, M_local);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M_local, N, M_local);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmul(handles[dev], matmul, &alpha,
                               d_A[dev], matA, d_B[dev], matB,
                               &beta, d_C[dev], matC, d_C[dev], matD,
                               nullptr, nullptr, 0, 0);

                hipDeviceSynchronize();

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);

                hipblaslt_cout << "GPU " << dev << ": " << M_local << "×" << N << " processed" << std::endl;
            }

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
                hipblasLtDestroy(handles[dev]);
            }
        }

        hipblaslt_cout << "Tall & skinny matrices tested" << std::endl;
    }

    // ============================================================================
    // Test 5: Extreme Aspect Ratios - Short & Wide
    // ============================================================================
    TEST(MultiGPURealDimensions, ShortWide_ColumnPartition)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Short & Wide: Low M, High N ===" << std::endl;

        std::vector<std::tuple<int64_t, int64_t, int64_t>> sizes = {
            {64, 4096, 64},     // Very wide
            {128, 8192, 512},   // Extremely wide
            {32, 2048, 2048},   // Wide with small M
            {64, 16384, 1024}   // Extreme
        };

        float alpha = 1.0f, beta = 0.0f;

        for(auto& [M, N_total, K] : sizes)
        {
            hipblaslt_cout << "Testing " << M << "×" << N_total << "×" << K << std::endl;

            int64_t N_per_gpu = N_total / numDevices;

            std::vector<hipblasLtHandle_t> handles(numDevices);
            std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipblasLtCreate(&handles[dev]);

                int64_t N_local = (dev == numDevices - 1) ? (N_total - dev * N_per_gpu) : N_per_gpu;

                hipMalloc(&d_A[dev], M * K * sizeof(float));
                hipMalloc(&d_B[dev], K * N_local * sizeof(float));
                hipMalloc(&d_C[dev], M * N_local * sizeof(float));

                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N_local, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N_local, M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N_local, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmul(handles[dev], matmul, &alpha,
                               d_A[dev], matA, d_B[dev], matB,
                               &beta, d_C[dev], matC, d_C[dev], matD,
                               nullptr, nullptr, 0, 0);

                hipDeviceSynchronize();

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);

                hipblaslt_cout << "GPU " << dev << ": " << M << "×" << N_local << " processed" << std::endl;
            }

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
                hipblasLtDestroy(handles[dev]);
            }
        }

        hipblaslt_cout << "Short & wide matrices tested" << std::endl;
    }

    // ============================================================================
    // Test 6: Varying K Dimension - Small K
    // ============================================================================
    TEST(MultiGPURealDimensions, SmallK_LargeMN)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Small K, Large M,N ===" << std::endl;

        std::vector<std::tuple<int64_t, int64_t, int64_t>> sizes = {
            {2048, 2048, 32},   // Very small K
            {4096, 4096, 64},   // Small K
            {1024, 1024, 16},   // Extremely small K
            {3000, 3000, 48}    // Small K, non-power-of-2
        };

        float alpha = 1.0f, beta = 0.0f;

        for(auto& [M, N, K] : sizes)
        {
            hipblaslt_cout << "Testing " << M << "×" << N << "×" << K << std::endl;

            std::vector<hipblasLtHandle_t> handles(numDevices);
            std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipblasLtCreate(&handles[dev]);

                hipMalloc(&d_A[dev], M * K * sizeof(float));
                hipMalloc(&d_B[dev], K * N * sizeof(float));
                hipMalloc(&d_C[dev], M * N * sizeof(float));

                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmul(handles[dev], matmul, &alpha,
                               d_A[dev], matA, d_B[dev], matB,
                               &beta, d_C[dev], matC, d_C[dev], matD,
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

        hipblaslt_cout << "Small K tested" << std::endl;
    }

    // ============================================================================
    // Test 7: Varying K Dimension - Large K
    // ============================================================================
    TEST(MultiGPURealDimensions, LargeK_SmallMN)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Large K, Small M,N ===" << std::endl;

        std::vector<std::tuple<int64_t, int64_t, int64_t>> sizes = {
            {512, 512, 8192},   // Very large K
            {256, 256, 16384},  // Extremely large K
            {1024, 1024, 4096}, // Large K
            {500, 500, 5000}    // Large K, non-power-of-2
        };

        float alpha = 1.0f, beta = 0.0f;

        for(auto& [M, N, K_total] : sizes)
        {
            hipblaslt_cout << "Testing " << M << "×" << N << "×" << K_total << " (K-partition)" << std::endl;

            int64_t K_per_gpu = K_total / numDevices;

            std::vector<std::vector<float>> h_C_partials(numDevices);
            std::vector<hipblasLtHandle_t> handles(numDevices);
            std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipblasLtCreate(&handles[dev]);

                int64_t K_local = (dev == numDevices - 1) ? (K_total - dev * K_per_gpu) : K_per_gpu;

                hipMalloc(&d_A[dev], M * K_local * sizeof(float));
                hipMalloc(&d_B[dev], K_local * N * sizeof(float));
                hipMalloc(&d_C[dev], M * N * sizeof(float));

                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K_local, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K_local, N, K_local);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmul(handles[dev], matmul, &alpha,
                               d_A[dev], matA, d_B[dev], matB,
                               &beta, d_C[dev], matC, d_C[dev], matD,
                               nullptr, nullptr, 0, 0);

                h_C_partials[dev].resize(M * N);
                hipMemcpy(h_C_partials[dev].data(), d_C[dev], M * N * sizeof(float), hipMemcpyDeviceToHost);

                hipDeviceSynchronize();

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);

                hipblaslt_cout << "GPU " << dev << ": K=" << K_local << " processed" << std::endl;
            }

            // Reduction
            std::vector<float> h_C_final(M * N, 0.0f);
            for(int dev = 0; dev < numDevices; ++dev)
            {
                for(size_t i = 0; i < h_C_final.size(); ++i)
                {
                    h_C_final[i] += h_C_partials[dev][i];
                }
            }

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
                hipblasLtDestroy(handles[dev]);
            }
        }

        hipblaslt_cout << "Large K tested" << std::endl;
    }

    // ============================================================================
    // Test 8: All Size Categories Per GPU
    // ============================================================================
    TEST(MultiGPURealDimensions, AllSizesPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== All Size Categories Per GPU ===" << std::endl;

        std::vector<std::tuple<int64_t, int64_t, int64_t, std::string>> sizes = {
            {32, 32, 32, "Tiny"},
            {128, 128, 128, "Small"},
            {512, 512, 512, "Medium"},
            {1024, 1024, 1024, "Large"},
            {2048, 2048, 512, "XLarge"},
            {4096, 2048, 512, "Huge"},
            {100, 100, 100, "NonPow2-Small"},
            {1000, 1000, 500, "NonPow2-Large"}
        };

        float alpha = 1.0f, beta = 0.0f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            int size_idx = dev % sizes.size();
            auto& [M, N, K, name] = sizes[size_idx];

            hipblaslt_cout << "GPU " << dev << ": " << name << " " << M << "×" << N << "×" << K << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            float *d_A, *d_B, *d_C;
            hipMalloc(&d_A, M * K * sizeof(float));
            hipMalloc(&d_B, K * N * sizeof(float));
            hipMalloc(&d_C, M * N * sizeof(float));

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasLtMatmul(handle, matmul, &alpha,
                           d_A, matA, d_B, matB,
                           &beta, d_C, matC, d_C, matD,
                           nullptr, nullptr, 0, 0);

            hipDeviceSynchronize();

            hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipFree(d_A); hipFree(d_B); hipFree(d_C);
            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "All size categories tested across " << numDevices << " GPUs" << std::endl;
    }

} // namespace
