// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side loader for the kpack GEMM example. Loads fp32, fp16, and bf16
// GEMM kernels from a kpack archive, runs each on the detected GPU, and
// verifies results against a naive CPU reference.
//
// This example demonstrates:
//   - GemmSignature schema: compile-time type resolution via make_kernel()
//   - Multi-variant iteration with typed buffers (float_to_typed / typed_to_float)
//   - Per-type tolerance for correctness verification

#include "gemm_api.hpp"
#include "gemm_args.hpp"

#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/gpu_arch.hpp>
#include <rocm_ck/hip_check.hpp>

#include <hip/hip_runtime.h>

#include <rocm_kpack/kpack.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using rocm_ck::BLOCK_SIZE;
using rocm_ck::DataType;
using rocm_ck::GemmArgs;
using rocm_ck::GemmKernel;
using rocm_ck::GemmSignature;
using rocm_ck::M_TILE;
using rocm_ck::make_kernel;
using rocm_ck::N_TILE;

// ============================================================================
// Variant table — compile-time resolved via GemmSignature schema
// ============================================================================

struct GemmVariant
{
    const char* name;
    GemmKernel kernel;
};

static constexpr GemmVariant ALL_GEMM_VARIANTS[] = {
    {"gemm_fp32", make_kernel(GemmSignature{.dtype = DataType::FP32})},
    {"gemm_fp16", make_kernel(GemmSignature{.dtype = DataType::FP16})},
    {"gemm_bf16", make_kernel(GemmSignature{.dtype = DataType::BF16})},
};

// ============================================================================
// CPU reference GEMM (always fp32)
// ============================================================================

/// Naive triple-loop CPU reference GEMM: C = A * B
/// A [M x K] RowMajor, B [K x N] ColumnMajor, C [M x N] RowMajor.
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
                sum += a[i * stride_A + k] * b[j * stride_B + k];
            }
            c[i * stride_C + j] = sum;
        }
    }
}

// ============================================================================
// Main
// ============================================================================

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

    constexpr int stride_A = K; // A [M x K] RowMajor
    constexpr int stride_B = K; // B [K x N] ColumnMajor
    constexpr int stride_C = N; // C [M x N] RowMajor

    // --- CPU reference (fp32, runs once) ---
    // Values kept small (0-7) so all types are exact. Worst case accumulation:
    // 256 × (7 × 7) = 12544, within fp16 exact range.
    std::vector<float> ref_a(M * K);
    std::vector<float> ref_b(K * N);
    std::vector<float> ref_c(M * N, 0.0f);

    for(int i = 0; i < M * K; ++i)
        ref_a[i] = static_cast<float>(i % 8);
    for(int i = 0; i < K * N; ++i)
        ref_b[i] = static_cast<float>(i % 8);

    cpu_gemm(ref_a.data(), ref_b.data(), ref_c.data(), M, N, K, stride_A, stride_B, stride_C);

    // --- Grid launch constants (same tile config for all variants) ---
    const int grid_m    = (M + M_TILE - 1) / M_TILE;
    const int grid_n    = (N + N_TILE - 1) / N_TILE;
    const int grid_size = grid_m * grid_n;

    // --- Run each variant ---
    bool all_passed  = true;
    int variants_run = 0;

    for(const auto& variant : ALL_GEMM_VARIANTS)
    {
        const GemmKernel& k = variant.kernel;

        // Load kernel code object
        void* kernel_code_object = nullptr;
        size_t code_object_size  = 0;
        kerr                     = kpack_get_kernel(
            archive, variant.name, gpu_arch.c_str(), &kernel_code_object, &code_object_size);
        if(kerr != KPACK_SUCCESS)
        {
            std::fprintf(stderr,
                         "%s: no kernel for '%s' (error %d), skipping\n",
                         variant.name,
                         gpu_arch.c_str(),
                         kerr);
            continue;
        }

        // Create HIP module and get function
        hipModule_t module            = nullptr;
        hipFunction_t kernel_function = nullptr;
        HIP_CHECK(hipModuleLoadData(&module, kernel_code_object));
        HIP_CHECK(hipModuleGetFunction(&kernel_function, module, variant.name));

        // Element sizes in bytes
        int a_elem_bytes = rocm_ck::data_type_bits(k.a_dtype) / 8;
        int b_elem_bytes = rocm_ck::data_type_bits(k.b_dtype) / 8;
        int c_elem_bytes = rocm_ck::data_type_bits(k.c_dtype) / 8;

        // Fill typed host buffers from float values
        std::vector<char> host_a(M * K * a_elem_bytes);
        std::vector<char> host_b(K * N * b_elem_bytes);
        std::vector<char> host_c(M * N * c_elem_bytes, 0);

        for(int i = 0; i < M * K; ++i)
            rocm_ck::float_to_typed(k.a_dtype, ref_a[i], &host_a[i * a_elem_bytes]);
        for(int i = 0; i < K * N; ++i)
            rocm_ck::float_to_typed(k.b_dtype, ref_b[i], &host_b[i * b_elem_bytes]);

        // Allocate device buffers
        void* device_a = nullptr;
        void* device_b = nullptr;
        void* device_c = nullptr;
        HIP_CHECK(hipMalloc(&device_a, M * K * a_elem_bytes));
        HIP_CHECK(hipMalloc(&device_b, K * N * b_elem_bytes));
        HIP_CHECK(hipMalloc(&device_c, M * N * c_elem_bytes));

        HIP_CHECK(hipMemcpy(device_a, host_a.data(), M * K * a_elem_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(device_b, host_b.data(), K * N * b_elem_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(device_c, 0, M * N * c_elem_bytes));

        // Launch kernel
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

        std::printf("%s: M=%d, N=%d, K=%d, grid=%d, block=%d\n",
                    variant.name,
                    M,
                    N,
                    K,
                    grid_size,
                    BLOCK_SIZE);

        HIP_CHECK(hipModuleLaunchKernel(kernel_function,
                                        grid_size,
                                        1,
                                        1,
                                        BLOCK_SIZE,
                                        1,
                                        1,
                                        0,
                                        nullptr,
                                        nullptr,
                                        launch_config));
        HIP_CHECK(hipDeviceSynchronize());

        // Download and verify
        HIP_CHECK(hipMemcpy(host_c.data(), device_c, M * N * c_elem_bytes, hipMemcpyDeviceToHost));

        float tol   = rocm_ck::tolerance_for(k.c_dtype);
        bool passed = true;
        for(int i = 0; i < M * N; ++i)
        {
            float got      = rocm_ck::typed_to_float(k.c_dtype, &host_c[i * c_elem_bytes]);
            float expected = ref_c[i];
            if(std::fabs(got - expected) > tol * std::fabs(expected) + tol)
            {
                int row = i / N;
                int col = i % N;
                std::fprintf(stderr,
                             "  MISMATCH at (%d,%d): got %f, expected %f (diff=%e)\n",
                             row,
                             col,
                             got,
                             expected,
                             got - expected);
                passed = false;
                break;
            }
        }

        std::printf("%s: %s\n", variant.name, passed ? "PASSED" : "FAILED");
        if(!passed)
            all_passed = false;
        ++variants_run;

        // Cleanup this variant
        HIP_CHECK(hipFree(device_a));
        HIP_CHECK(hipFree(device_b));
        HIP_CHECK(hipFree(device_c));
        HIP_CHECK(hipModuleUnload(module));
        kpack_free_kernel(kernel_code_object);
    }

    kpack_close(archive);

    if(variants_run == 0)
    {
        std::fprintf(stderr, "No variants ran successfully\n");
        return 1;
    }

    return all_passed ? 0 : 1;
}
