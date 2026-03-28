// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side loader for the kpack GEMM example. Loads fp32, fp16, bf16,
// fp16_w32, fp16_add, and fp16_add_relu GEMM kernels from a kpack archive,
// runs each on the detected GPU, and verifies against CPU reference.
//
// This example demonstrates:
//   - Operator-centric Signature schema with GemmOp, AddOp, ReluOp
//   - Multi-variant iteration with per-variant grid launch parameters
//   - Epilogue composition via operator chaining (e.g., GemmOp + AddOp + ReluOp)
//   - Per-type tolerance for correctness verification

#include "cpu_ref.hpp"
#include "gemm_variants.hpp"

#include <rocm_ck/args.hpp>
#include <rocm_ck/datatype_convert.hpp>
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

    // Fused references
    std::vector<float> ref_add(M * N);
    cpu_gemm_add(ref_add.data(), ref_c.data(), ref_d0.data(), M * N);

    std::vector<float> ref_add_relu(M * N);
    cpu_gemm_add_relu(ref_add_relu.data(), ref_c.data(), ref_d0.data(), M * N);

    // --- Run each variant ---
    bool all_passed  = true;
    int variants_run = 0;

    for(const rocm_ck::GemmVariant& variant : rocm_ck::gemm_variants)
    {
        const rocm_ck::GemmSpec& spec = variant.spec;

        // Per-variant grid dimensions from tile geometry
        int grid_m    = (M + spec.block_tile.m - 1) / spec.block_tile.m;
        int grid_n    = (N + spec.block_tile.n - 1) / spec.block_tile.n;
        int grid_size = grid_m * grid_n;

        // Load kernel
        rocm_ck::KpackKernel kernel;
        if(!kernel.load(archive, variant.name))
            continue;

        // Allocate typed device buffers from physical tensor table
        rocm_ck::TypedBuffer buf_a(spec.lhs().dtype, M * K);
        rocm_ck::TypedBuffer buf_b(spec.rhs().dtype, K * N);
        rocm_ck::TypedBuffer buf_c(spec.output().dtype, M * N);

        buf_a.upload(ref_a.data());
        buf_b.upload(ref_b.data());
        buf_c.zero();

        // D0 tensor (e.g., bias) is present when num_physical_tensors > 3
        bool has_d0 = spec.num_physical_tensors > 3;

        std::unique_ptr<rocm_ck::TypedBuffer> buf_d0;
        if(has_d0)
        {
            buf_d0 = std::make_unique<rocm_ck::TypedBuffer>(spec.d0().dtype, M * N);
            buf_d0->upload(ref_d0.data());
        }

        std::printf("%s: M=%d, N=%d, K=%d, grid=%d, block=%d\n",
                    variant.name,
                    M,
                    N,
                    K,
                    grid_size,
                    spec.thread_block_size);

        // Build generic Args — slot indices from role-based accessors
        rocm_ck::Args kernel_args{};
        // lhs [M x K] RowMajor
        kernel_args.tensors[spec.lhs().args_slot] = {
            buf_a.ptr(), rocm_ck::make_shape(M, K), rocm_ck::make_strides(stride_A, 1)};
        // rhs [K x N] ColumnMajor
        kernel_args.tensors[spec.rhs().args_slot] = {
            buf_b.ptr(), rocm_ck::make_shape(K, N), rocm_ck::make_strides(1, stride_B)};
        // Output [M x N] RowMajor
        kernel_args.tensors[spec.output().args_slot] = {
            buf_c.ptr(), rocm_ck::make_shape(M, N), rocm_ck::make_strides(stride_C, 1)};
        // D0 [M x N] RowMajor (optional, for fused epilogue)
        if(has_d0)
        {
            constexpr int stride_D0                  = N;
            kernel_args.tensors[spec.d0().args_slot] = {
                buf_d0->ptr(), rocm_ck::make_shape(M, N), rocm_ck::make_strides(stride_D0, 1)};
        }

        void* args_ptr          = &kernel_args;
        size_t kernel_args_size = sizeof(kernel_args);
        void* launch_config[]   = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                   args_ptr,
                                   HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                   &kernel_args_size,
                                   HIP_LAUNCH_PARAM_END};

        HIP_CHECK(hipModuleLaunchKernel(kernel.function(),
                                        grid_size,
                                        1,
                                        1,
                                        spec.thread_block_size,
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

        // Select correct reference based on epilogue op chain.
        // Priority: GEMM+Add+ReLU > GEMM+Add > plain GEMM
        const float* ref = ref_c.data();
        if(spec.has_epilogue_op(rocm_ck::EpilogueOp::Add) &&
           spec.has_epilogue_op(rocm_ck::EpilogueOp::Relu))
            ref = ref_add_relu.data();
        else if(spec.has_epilogue_op(rocm_ck::EpilogueOp::Add))
            ref = ref_add.data();

        float tolerance = rocm_ck::tolerance_for(spec.output().dtype);
        bool passed     = true;
        for(int i = 0; i < M * N; ++i)
        {
            if(std::fabs(result[i] - ref[i]) > tolerance * std::fabs(ref[i]) + tolerance)
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
