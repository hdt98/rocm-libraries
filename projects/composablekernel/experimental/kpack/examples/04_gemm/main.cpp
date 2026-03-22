// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side loader for the kpack GEMM example. Loads fp32, fp16, bf16,
// fp16_w32, and fp16_add GEMM kernels from a kpack archive, runs each on
// the detected GPU, and verifies results against a naive CPU reference.
//
// This example demonstrates:
//   - GemmConfig schema: compile-time type + tile geometry via make_kernel()
//   - Multi-variant iteration with per-variant grid launch parameters
//   - Fused epilogue: E = A*B + D0 with GemmArgs1D
//   - Per-type tolerance for correctness verification

#include "cpu_ref.hpp"
#include "gemm_api.hpp"
#include "gemm_args.hpp"

#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/hip_check.hpp>
#include <rocm_ck/kpack_module.hpp>
#include <rocm_ck/typed_buffer.hpp>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

using rocm_ck::DataType;
using rocm_ck::EpilogueOp;
using rocm_ck::GemmArgs;
using rocm_ck::GemmArgs1D;
using rocm_ck::GemmConfig;
using rocm_ck::GemmKernel;
using rocm_ck::make_kernel;

// ============================================================================
// Variant table — compile-time resolved via GemmConfig schema
// ============================================================================

struct GemmVariant
{
    const char* name;
    GemmKernel kernel;
};

static constexpr GemmVariant ALL_GEMM_VARIANTS[] = {
    {"gemm_fp32",
     make_kernel(GemmConfig{.signature = {.dtype = DataType::FP32},
                            .algorithm = {.block_tile  = {128, 128, 32},
                                          .block_warps = {2, 2, 1},
                                          .warp_tile   = {16, 16, 16}}})},
    {"gemm_fp16",
     make_kernel(GemmConfig{.signature = {.dtype = DataType::FP16},
                            .algorithm = {.block_tile  = {128, 128, 32},
                                          .block_warps = {2, 2, 1},
                                          .warp_tile   = {16, 16, 16}}})},
    {"gemm_bf16",
     make_kernel(GemmConfig{.signature = {.dtype = DataType::BF16},
                            .algorithm = {.block_tile  = {128, 128, 32},
                                          .block_warps = {2, 2, 1},
                                          .warp_tile   = {16, 16, 16}}})},
    {"gemm_fp16_w32",
     make_kernel(GemmConfig{.signature = {.dtype = DataType::FP16},
                            .algorithm = {.block_tile  = {128, 128, 32},
                                          .block_warps = {2, 2, 1},
                                          .warp_tile   = {32, 32, 16}}})},
    {"gemm_fp16_add",
     make_kernel(GemmConfig{.signature = {.dtype       = DataType::FP16,
                                          .epilogue_op = EpilogueOp::Add,
                                          .d0_dtype    = DataType::FP16},
                            .algorithm = {.block_tile  = {128, 128, 32},
                                          .block_warps = {2, 2, 1},
                                          .warp_tile   = {16, 16, 16}}})},
};

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
    rocm_ck::KpackArchive archive;
    if(!archive.open(argv[1]))
        return 1;

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
    // D0 values are small (0-3) — worst case fused result: 12544 + 3 = 12547.
    std::vector<float> ref_a(M * K);
    std::vector<float> ref_b(K * N);
    std::vector<float> ref_c(M * N, 0.0f);
    std::vector<float> ref_d0(M * N);

    for(int i = 0; i < M * K; ++i)
        ref_a[i] = static_cast<float>(i % 8);
    for(int i = 0; i < K * N; ++i)
        ref_b[i] = static_cast<float>(i % 8);
    for(int i = 0; i < M * N; ++i)
        ref_d0[i] = static_cast<float>(i % 4);

    cpu_gemm(ref_a.data(), ref_b.data(), ref_c.data(), M, N, K, stride_A, stride_B, stride_C);

    // Fused reference: E = A*B + D0
    std::vector<float> ref_add(M * N);
    cpu_gemm_add(ref_add.data(), ref_c.data(), ref_d0.data(), M * N);

    // --- Run each variant ---
    bool all_passed  = true;
    int variants_run = 0;

    for(const auto& variant : ALL_GEMM_VARIANTS)
    {
        const GemmKernel& k = variant.kernel;

        // Per-variant grid dimensions from tile geometry
        int grid_m    = (M + k.block_tile.m - 1) / k.block_tile.m;
        int grid_n    = (N + k.block_tile.n - 1) / k.block_tile.n;
        int grid_size = grid_m * grid_n;

        // Load kernel
        rocm_ck::KpackKernel kernel;
        if(!kernel.load(archive, variant.name))
            continue;

        // Allocate typed device buffers
        rocm_ck::TypedBuffer buf_a(k.a_dtype, M * K);
        rocm_ck::TypedBuffer buf_b(k.b_dtype, K * N);
        rocm_ck::TypedBuffer buf_c(k.c_dtype, M * N);

        buf_a.upload(ref_a.data());
        buf_b.upload(ref_b.data());
        buf_c.zero();

        // Allocate D0 tensor (used by fused epilogue variants)
        std::unique_ptr<rocm_ck::TypedBuffer> buf_d0;
        if(k.num_d_tensors >= 1)
        {
            buf_d0 = std::make_unique<rocm_ck::TypedBuffer>(k.d0_dtype, M * N);
            buf_d0->upload(ref_d0.data());
        }

        std::printf("%s: M=%d, N=%d, K=%d, grid=%d, block=%d\n",
                    variant.name,
                    M,
                    N,
                    K,
                    grid_size,
                    k.thread_block_size);

        // Build args struct — type depends on D tensor count
        GemmArgs base_args = {
            .M        = M,
            .N        = N,
            .K        = K,
            .stride_A = stride_A,
            .stride_B = stride_B,
            .stride_C = stride_C,
            .a        = buf_a.ptr(),
            .b        = buf_b.ptr(),
            .c        = buf_c.ptr(),
        };
        constexpr int stride_D0 = N; // D0 [M x N] RowMajor
        GemmArgs1D fused_args   = {
              .M         = M,
              .N         = N,
              .K         = K,
              .stride_A  = stride_A,
              .stride_B  = stride_B,
              .stride_E  = stride_C, // output stride matches C layout
              .a         = buf_a.ptr(),
              .b         = buf_b.ptr(),
              .e         = buf_c.ptr(),
              .stride_D0 = stride_D0,
              ._pad0     = 0,
              .d0        = (k.num_d_tensors >= 1) ? buf_d0->ptr() : nullptr,
        };

        // Launch — select args struct by D tensor count
        void* args_ptr          = (k.num_d_tensors == 0) ? static_cast<void*>(&base_args)
                                                         : static_cast<void*>(&fused_args);
        size_t kernel_args_size = (k.num_d_tensors == 0) ? sizeof(base_args) : sizeof(fused_args);
        void* launch_config[]   = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                   args_ptr,
                                   HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                   &kernel_args_size,
                                   HIP_LAUNCH_PARAM_END};

        HIP_CHECK(hipModuleLaunchKernel(kernel.function(),
                                        grid_size,
                                        1,
                                        1,
                                        k.thread_block_size,
                                        1,
                                        1,
                                        0,
                                        nullptr,
                                        nullptr,
                                        launch_config));
        HIP_CHECK(hipDeviceSynchronize());

        // Download and verify
        std::vector<float> result(M * N);
        buf_c.download(result.data());

        // Select correct reference: plain GEMM or fused
        const float* ref = (k.num_d_tensors == 0) ? ref_c.data() : ref_add.data();

        float tol   = rocm_ck::tolerance_for(k.c_dtype);
        bool passed = true;
        for(int i = 0; i < M * N; ++i)
        {
            if(std::fabs(result[i] - ref[i]) > tol * std::fabs(ref[i]) + tol)
            {
                int row = i / N;
                int col = i % N;
                std::fprintf(stderr,
                             "  MISMATCH at (%d,%d): got %f, expected %f (diff=%e)\n",
                             row,
                             col,
                             result[i],
                             ref[i],
                             result[i] - ref[i]);
                passed = false;
                break;
            }
        }

        std::printf("%s: %s\n", variant.name, passed ? "PASSED" : "FAILED");
        if(!passed)
            all_passed = false;
        ++variants_run;
    }

    if(variants_run == 0)
    {
        std::fprintf(stderr, "No variants ran successfully\n");
        return 1;
    }

    return all_passed ? 0 : 1;
}
