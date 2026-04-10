/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Real Multi-GPU Tests - Leading Dimensions & Strides
 * Tests padded matrices (64, 128, 256-byte alignment)
 * Non-contiguous strides for batched GEMM
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
    // Test 1: Padded Leading Dimension A (ldA > M)
    // ============================================================================
    TEST(MultiGPURealLeadingStride, PaddedLdA_DataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Padded ldA (ldA > M) ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 2;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<int64_t> padding_values = {16, 32, 64};

        for(auto padding : padding_values)
        {
            int64_t ldA = M + padding;
            hipblaslt_cout << "Testing ldA=" << ldA << " (M=" << M << ", padding=" << padding << ")" << std::endl;

            std::vector<hipblasLtHandle_t> handles(numDevices);
            std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipblasLtCreate(&handles[dev]);

                hipMalloc(&d_A[dev], batches_per_gpu * ldA * K * sizeof(float));
                hipMalloc(&d_B[dev], batches_per_gpu * K * N * sizeof(float));
                hipMalloc(&d_C[dev], batches_per_gpu * M * N * sizeof(float));

                for(int64_t b = 0; b < batches_per_gpu; ++b)
                {
                    hipblasLtMatrixLayout_t matA, matB, matC, matD;
                    hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, ldA); // Padded!
                    hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                    hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
                    hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                    hipblasLtMatmulDesc_t matmul;
                    hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                    hipblasLtMatmul(handles[dev], matmul, &alpha,
                                   d_A[dev] + b * ldA * K, matA,
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

        hipblaslt_cout << "Padded ldA tested" << std::endl;
    }

    // ============================================================================
    // Test 2: Padded Leading Dimension B (ldB > K)
    // ============================================================================
    TEST(MultiGPURealLeadingStride, PaddedLdB_RowPartition)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Padded ldB (ldB > K) ===" << std::endl;

        const int64_t M_total = 1024, N = 512, K = 512;
        const int64_t M_per_gpu = M_total / numDevices;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<int64_t> padding_values = {16, 32, 64};

        for(auto padding : padding_values)
        {
            int64_t ldB = K + padding;
            hipblaslt_cout << "Testing ldB=" << ldB << " (K=" << K << ", padding=" << padding << ")" << std::endl;

            std::vector<hipblasLtHandle_t> handles(numDevices);
            std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipblasLtCreate(&handles[dev]);

                int64_t M_local = (dev == numDevices - 1) ? (M_total - dev * M_per_gpu) : M_per_gpu;

                hipMalloc(&d_A[dev], M_local * K * sizeof(float));
                hipMalloc(&d_B[dev], ldB * N * sizeof(float)); // Padded!
                hipMalloc(&d_C[dev], M_local * N * sizeof(float));

                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M_local, K, M_local);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, ldB); // Padded!
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
            }

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
                hipblasLtDestroy(handles[dev]);
            }
        }

        hipblaslt_cout << "Padded ldB tested" << std::endl;
    }

    // ============================================================================
    // Test 3: Padded Leading Dimension C (ldC > M)
    // ============================================================================
    TEST(MultiGPURealLeadingStride, PaddedLdC_ColumnPartition)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Padded ldC (ldC > M) ===" << std::endl;

        const int64_t M = 512, N_total = 1024, K = 512;
        const int64_t N_per_gpu = N_total / numDevices;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<int64_t> padding_values = {16, 32, 64, 128};

        for(auto padding : padding_values)
        {
            int64_t ldC = M + padding;
            hipblaslt_cout << "Testing ldC=" << ldC << " (M=" << M << ", padding=" << padding << ")" << std::endl;

            std::vector<hipblasLtHandle_t> handles(numDevices);
            std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipblasLtCreate(&handles[dev]);

                int64_t N_local = (dev == numDevices - 1) ? (N_total - dev * N_per_gpu) : N_per_gpu;

                hipMalloc(&d_A[dev], M * K * sizeof(float));
                hipMalloc(&d_B[dev], K * N_local * sizeof(float));
                hipMalloc(&d_C[dev], ldC * N_local * sizeof(float)); // Padded!

                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N_local, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N_local, ldC); // Padded!
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N_local, ldC); // Padded!

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

        hipblaslt_cout << "Padded ldC tested" << std::endl;
    }

    // ============================================================================
    // Test 4: All Matrices Padded (ldA, ldB, ldC all > dimensions)
    // ============================================================================
    TEST(MultiGPURealLeadingStride, AllPadded_DataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== All Matrices Padded (ldA, ldB, ldC) ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 2;
        const int64_t padding = 32;
        float alpha = 1.0f, beta = 0.0f;

        int64_t ldA = M + padding;
        int64_t ldB = K + padding;
        int64_t ldC = M + padding;

        hipblaslt_cout << "ldA=" << ldA << ", ldB=" << ldB << ", ldC=" << ldC << std::endl;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_A[dev], batches_per_gpu * ldA * K * sizeof(float));
            hipMalloc(&d_B[dev], batches_per_gpu * ldB * N * sizeof(float));
            hipMalloc(&d_C[dev], batches_per_gpu * ldC * N * sizeof(float));

            for(int64_t b = 0; b < batches_per_gpu; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, ldA);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, ldB);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, ldC);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, ldC);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmul(handles[dev], matmul, &alpha,
                               d_A[dev] + b * ldA * K, matA,
                               d_B[dev] + b * ldB * N, matB,
                               &beta,
                               d_C[dev] + b * ldC * N, matC,
                               d_C[dev] + b * ldC * N, matD,
                               nullptr, nullptr, 0, 0);

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }

            hipDeviceSynchronize();
            hipblaslt_cout << "GPU " << dev << ": All padded matrices processed" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "All padded matrices tested" << std::endl;
    }

    // ============================================================================
    // Test 5: Alignment Testing - 64-byte, 128-byte, 256-byte
    // ============================================================================
    TEST(MultiGPURealLeadingStride, AlignmentTesting)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Alignment Testing (64, 128, 256-byte) ===" << std::endl;

        const int64_t M = 253, N = 253, K = 253; // Odd sizes to test alignment
        float alpha = 1.0f, beta = 0.0f;

        // Calculate padding for different alignments (assuming float = 4 bytes)
        std::vector<std::tuple<int64_t, std::string>> alignments = {
            {((M + 15) / 16) * 16, "64-byte"},   // 16 floats = 64 bytes
            {((M + 31) / 32) * 32, "128-byte"},  // 32 floats = 128 bytes
            {((M + 63) / 64) * 64, "256-byte"}   // 64 floats = 256 bytes
        };

        for(auto& [aligned_ld, name] : alignments)
        {
            hipblaslt_cout << "Testing " << name << " alignment: ldA=" << aligned_ld << std::endl;

            std::vector<hipblasLtHandle_t> handles(numDevices);
            std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

            for(int dev = 0; dev < numDevices; ++dev)
            {
                hipSetDevice(dev);
                hipblasLtCreate(&handles[dev]);

                hipMalloc(&d_A[dev], aligned_ld * K * sizeof(float));
                hipMalloc(&d_B[dev], aligned_ld * N * sizeof(float));
                hipMalloc(&d_C[dev], aligned_ld * N * sizeof(float));

                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, aligned_ld);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, aligned_ld);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, aligned_ld);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, aligned_ld);

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

        hipblaslt_cout << "Alignment testing completed" << std::endl;
    }

    // ============================================================================
    // Test 6: Non-Contiguous Strides (Batched GEMM with gaps)
    // ============================================================================
    TEST(MultiGPURealLeadingStride, NonContiguousStrides_Batched)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Non-Contiguous Strides (with gaps) ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        const int64_t batches_per_gpu = 4;
        float alpha = 1.0f, beta = 0.0f;

        // Standard stride vs. strided with gaps
        int64_t stride_standard = M * K;
        int64_t stride_gap = M * K + 128; // Add 128 element gap

        hipblaslt_cout << "Standard stride: " << stride_standard
                      << ", Stride with gap: " << stride_gap << std::endl;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            // Allocate with gaps
            hipMalloc(&d_A[dev], batches_per_gpu * stride_gap * sizeof(float));
            hipMalloc(&d_B[dev], batches_per_gpu * (K * N + 128) * sizeof(float));
            hipMalloc(&d_C[dev], batches_per_gpu * (M * N + 128) * sizeof(float));

            for(int64_t b = 0; b < batches_per_gpu; ++b)
            {
                hipblasLtMatrixLayout_t matA, matB, matC, matD;
                hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
                hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                // Access with strided gaps
                hipblasLtMatmul(handles[dev], matmul, &alpha,
                               d_A[dev] + b * stride_gap, matA,
                               d_B[dev] + b * (K * N + 128), matB,
                               &beta,
                               d_C[dev] + b * (M * N + 128), matC,
                               d_C[dev] + b * (M * N + 128), matD,
                               nullptr, nullptr, 0, 0);

                hipblasLtMatrixLayoutDestroy(matA); hipblasLtMatrixLayoutDestroy(matB);
                hipblasLtMatrixLayoutDestroy(matC); hipblasLtMatrixLayoutDestroy(matD);
                hipblasLtMatmulDescDestroy(matmul);
            }

            hipDeviceSynchronize();
            hipblaslt_cout << "GPU " << dev << ": Non-contiguous strides processed" << std::endl;
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_A[dev]); hipFree(d_B[dev]); hipFree(d_C[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "Non-contiguous strides tested" << std::endl;
    }

    // ============================================================================
    // Test 7: Different Padding Per GPU
    // ============================================================================
    TEST(MultiGPURealLeadingStride, DifferentPaddingPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Different Padding Per GPU ===" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<int64_t> padding_values = {0, 16, 32, 64, 8, 24, 48, 128};

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            int pad_idx = dev % padding_values.size();
            int64_t padding = padding_values[pad_idx];
            int64_t ldA = M + padding;

            hipblaslt_cout << "GPU " << dev << ": padding=" << padding << ", ldA=" << ldA << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            float *d_A, *d_B, *d_C;
            hipMalloc(&d_A, ldA * K * sizeof(float));
            hipMalloc(&d_B, K * N * sizeof(float));
            hipMalloc(&d_C, ldA * N * sizeof(float));

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, ldA);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, ldA);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, ldA);

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

        hipblaslt_cout << "Different padding tested across " << numDevices << " GPUs" << std::endl;
    }

} // namespace
