// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side loader for the kpack GEMM example. Loads a single fp32 GEMM kernel
// from a kpack archive, runs it on the detected GPU, and verifies results
// against a naive CPU reference.
//
// This example demonstrates:
//   - Loading a GEMM kernel from a kpack archive (same as vector_add example)
//   - 2D tiled grid launch through hipModuleLaunchKernel
//   - GemmTile1DPartitioner flattens ceil(M/M_TILE) * ceil(N/N_TILE) into 1D grid
//   - CPU reference GEMM for correctness verification

#include "gemm_args.hpp"

#include <rocm_ck/gpu_arch.hpp>
#include <rocm_ck/hip_check.hpp>

#include <hip/hip_runtime.h>

#include <rocm_kpack/kpack.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace kpack_gemm;

/// Naive triple-loop CPU reference GEMM: C = A * B
/// A [M x K] RowMajor, B [K x N] ColumnMajor, C [M x N] RowMajor.
///
/// This is deliberately unoptimized — it exists only for correctness checking.
/// For the ColumnMajor B layout, element B[k][n] is at b[n * stride_B + k].
static void cpu_gemm(const float* a,
                     const float* b,
                     float* c,
                     int M,
                     int N,
                     int K,
                     int stride_A,
                     int stride_B,
                     int stride_C)
{
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for(int k = 0; k < K; ++k)
            {
                // A is RowMajor: A[i][k] = a[i * stride_A + k]
                // B is ColumnMajor: B[k][j] = b[j * stride_B + k]
                sum += a[i * stride_A + k] * b[j * stride_B + k];
            }
            // C is RowMajor: C[i][j] = c[i * stride_C + j]
            c[i * stride_C + j] = sum;
        }
    }
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::fprintf(stderr, "Usage: %s <path-to-gemm.kpack>\n", argv[0]);
        return 1;
    }

    // --- Open the kpack archive ---
    kpack_archive_t archive = nullptr;
    kpack_error_t kerr      = kpack_open(argv[1], &archive);
    if(kerr != KPACK_SUCCESS)
    {
        std::fprintf(stderr, "Failed to open archive '%s' (error %d)\n", argv[1], kerr);
        return 1;
    }

    size_t arch_count = 0;
    kpack_get_architecture_count(archive, &arch_count);
    std::printf("Opened %s — architectures:", argv[1]);
    for(size_t i = 0; i < arch_count; ++i)
    {
        const char* arch_name = nullptr;
        kpack_get_architecture(archive, i, &arch_name);
        std::printf("%s %s", (i > 0 ? "," : ""), arch_name);
    }
    std::printf("\n");

    // --- Detect current GPU architecture ---
    std::string gpu_arch = rocm_ck::get_gpu_arch();
    if(gpu_arch.empty())
    {
        std::fprintf(stderr, "Failed to detect GPU architecture\n");
        kpack_close(archive);
        return 1;
    }
    std::printf("Detected GPU: %s\n", gpu_arch.c_str());

    // --- Problem dimensions (all divisible by tile dims for no-pad kernel) ---
    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 256;

    // Strides match BLAS conventions
    constexpr int stride_A = K; // A [M x K] RowMajor
    constexpr int stride_B = K; // B [K x N] ColumnMajor
    constexpr int stride_C = N; // C [M x N] RowMajor

    // --- Initialize host matrices with small integers for exact CPU reference ---
    // Values are kept small (0-15) so fp32 accumulation is exact even for K=256.
    // Worst case: sum of 256 products of (15 * 15) = 256 * 225 = 57600, well
    // within fp32's exact integer range (2^24 = 16777216).
    std::vector<float> host_a(M * K);
    std::vector<float> host_b(K * N);
    std::vector<float> host_c(M * N, 0.0f);
    std::vector<float> ref_c(M * N, 0.0f);

    for(int i = 0; i < M * K; ++i)
        host_a[i] = static_cast<float>(i % 16);
    for(int i = 0; i < K * N; ++i)
        host_b[i] = static_cast<float>(i % 16);

    // --- CPU reference ---
    cpu_gemm(host_a.data(), host_b.data(), ref_c.data(), M, N, K, stride_A, stride_B, stride_C);

    // --- Load kernel code object ---
    void* kernel_code_object       = nullptr;
    size_t kernel_code_object_size = 0;
    kerr                           = kpack_get_kernel(
        archive, "gemm_fp32", gpu_arch.c_str(), &kernel_code_object, &kernel_code_object_size);
    if(kerr != KPACK_SUCCESS)
    {
        std::fprintf(stderr, "gemm_fp32: no kernel for '%s' (error %d)\n", gpu_arch.c_str(), kerr);
        kpack_close(archive);
        return 1;
    }

    // --- Create HIP module and get function ---
    hipModule_t module            = nullptr;
    hipFunction_t kernel_function = nullptr;
    HIP_CHECK(hipModuleLoadData(&module, kernel_code_object));
    HIP_CHECK(hipModuleGetFunction(&kernel_function, module, "gemm_fp32"));

    // --- Allocate device buffers ---
    void* device_a = nullptr;
    void* device_b = nullptr;
    void* device_c = nullptr;
    HIP_CHECK(hipMalloc(&device_a, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_b, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_c, M * N * sizeof(float)));

    HIP_CHECK(hipMemcpy(device_a, host_a.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_b, host_b.data(), K * N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(device_c, 0, M * N * sizeof(float)));

    // --- Launch kernel ---
    // GemmTile1DPartitioner flattens the 2D tile grid into 1D:
    //   grid = ceil(M / M_TILE) * ceil(N / N_TILE)
    // For M=512, N=512, M_TILE=128, N_TILE=128:
    //   grid = (512/128) * (512/128) = 4 * 4 = 16 workgroups
    const int grid_m    = (M + M_TILE - 1) / M_TILE;
    const int grid_n    = (N + N_TILE - 1) / N_TILE;
    const int grid_size = grid_m * grid_n;

    GemmArgs kernel_args = {
        .M        = M,
        .N        = N,
        .K        = K,
        .stride_A = stride_A,
        .stride_B = stride_B,
        .stride_C = stride_C,
        .a        = device_a,
        .b        = device_b,
        .c        = device_c,
    };
    size_t kernel_args_size = sizeof(kernel_args);
    void* launch_config[]   = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                               &kernel_args,
                               HIP_LAUNCH_PARAM_BUFFER_SIZE,
                               &kernel_args_size,
                               HIP_LAUNCH_PARAM_END};

    std::printf("gemm_fp32: M=%d, N=%d, K=%d, grid=%d, block=%d\n", M, N, K, grid_size, BLOCK_SIZE);

    HIP_CHECK(hipModuleLaunchKernel(kernel_function,
                                    grid_size,
                                    1,
                                    1, // grid (1D)
                                    BLOCK_SIZE,
                                    1,
                                    1,       // block
                                    0,       // shared mem
                                    nullptr, // stream
                                    nullptr, // kernel params (unused)
                                    launch_config));
    HIP_CHECK(hipDeviceSynchronize());

    // --- Download and verify ---
    HIP_CHECK(hipMemcpy(host_c.data(), device_c, M * N * sizeof(float), hipMemcpyDeviceToHost));

    // Tolerance: fp32 with small-integer inputs should produce exact results,
    // but we use a small tolerance for safety (rounding in MFMA instructions).
    constexpr float tol = 1e-5f;
    bool passed         = true;
    for(int i = 0; i < M * N; ++i)
    {
        if(std::fabs(host_c[i] - ref_c[i]) > tol * std::fabs(ref_c[i]) + tol)
        {
            int row = i / N;
            int col = i % N;
            std::fprintf(stderr,
                         "  MISMATCH at (%d,%d): got %f, expected %f (diff=%e)\n",
                         row,
                         col,
                         host_c[i],
                         ref_c[i],
                         host_c[i] - ref_c[i]);
            passed = false;
            break;
        }
    }

    std::printf("gemm_fp32: %s\n", passed ? "PASSED" : "FAILED");

    // --- Cleanup ---
    HIP_CHECK(hipFree(device_a));
    HIP_CHECK(hipFree(device_b));
    HIP_CHECK(hipFree(device_c));
    HIP_CHECK(hipModuleUnload(module));
    kpack_free_kernel(kernel_code_object);
    kpack_close(archive);

    return passed ? 0 : 1;
}
