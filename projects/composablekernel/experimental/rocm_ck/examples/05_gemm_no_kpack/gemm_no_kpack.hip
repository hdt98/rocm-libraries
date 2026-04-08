// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Minimal GEMM example without kpack. A conventional single-file HIP program:
// one compilation, <<<>>> launch, no runtime module loading.
//
// Kernel: FP16 GEMM with fused Add (bias) + ReLU epilogue.
//   E = relu(A * B + bias)

#include <hip/hip_runtime.h>

#include <rocm_ck/args.hpp>
#include <rocm_ck/gemm_spec.hpp>
#include <rocm_ck/hip_check.hpp>
#include <rocm_ck/layout.hpp>
#include <rocm_ck/typed_buffer.hpp>
#include <rocm_ck/verify.hpp>

// Device-only: Avoid having the host compiler pass parse all the CK Tile sources.
#ifdef __HIP_DEVICE_COMPILE__
#include <rocm_ck/gemm_dev.hpp>
#endif

#include <cstdio>
#include <vector>

// ============================================================================
// Kernel spec — constexpr, visible to both host and device passes
// ============================================================================

static constexpr auto kSpec = rocm_ck::makeSpec(
    rocm_ck::Signature{.dtype = rocm_ck::DataType::FP16,
                       .ops   = {rocm_ck::GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                 rocm_ck::AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                 rocm_ck::ReluOp{.in = "D", .out = "E"}}},
    rocm_ck::GemmAlgorithm{.block_tile  = {128, 128, 32},
                           .block_waves = {2, 2, 1},
                           .wave_tile   = {16, 16, 16}},
    rocm_ck::TargetSet::cdna());

// ============================================================================
// Kernel
// ============================================================================

__global__ void gemm_fp16_add_relu(rocm_ck::Args args)
{
#ifdef __HIP_DEVICE_COMPILE__
    rocm_ck::run<kSpec>(args);
#endif
}

// ============================================================================
// Host
// ============================================================================

int main()
{
    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 256;

    // --- Input data (fp32, small values for exact fp16 accumulation) ---
    std::vector<float> h_a(M * K);
    std::vector<float> h_b(K * N);
    std::vector<float> h_bias(M * N);

    for(int i = 0; i < M * K; ++i)
        h_a[i] = static_cast<float>(i % 8);
    for(int i = 0; i < K * N; ++i)
        h_b[i] = static_cast<float>(i % 8);
    for(int i = 0; i < M * N; ++i)
        h_bias[i] = static_cast<float>(i % 4);

    // --- CPU reference: E = relu(A * B + bias) ---
    // Strides come from the spec — the default B layout is Col-major.
    auto [a_sm, a_sk] = rocm_ck::layoutStrides(kSpec.lhs().layout, M, K);
    auto [b_sk, b_sn] = rocm_ck::layoutStrides(kSpec.rhs().layout, K, N);
    auto [e_sm, e_sn] = rocm_ck::layoutStrides(kSpec.output().layout, M, N);
    auto [d_sm, d_sn] = rocm_ck::layoutStrides(kSpec.d0().layout, M, N);

    std::vector<float> ref(M * N, 0.0f);
    for(int m = 0; m < M; ++m)
        for(int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for(int k = 0; k < K; ++k)
                sum += h_a[m * a_sm + k * a_sk] * h_b[k * b_sk + n * b_sn];
            float val = sum + h_bias[m * d_sm + n * d_sn];
            ref[m * e_sm + n * e_sn] = val > 0.f ? val : 0.f;
        }

    // --- Device buffers ---
    rocm_ck::TypedBuffer buf_a(kSpec.lhs().dtype, M * K);
    rocm_ck::TypedBuffer buf_b(kSpec.rhs().dtype, K * N);
    rocm_ck::TypedBuffer buf_bias(kSpec.d0().dtype, M * N);
    rocm_ck::TypedBuffer buf_e(kSpec.output().dtype, M * N);

    buf_a.upload(h_a.data());
    buf_b.upload(h_b.data());
    buf_bias.upload(h_bias.data());
    buf_e.zero();

    // --- Build Args ---
    rocm_ck::Args args{};
    args.tensors[kSpec.lhs().args_slot]    = {buf_a.ptr(),    rocm_ck::makeShape(M, K), rocm_ck::makeStrides(a_sm, a_sk)};
    args.tensors[kSpec.rhs().args_slot]    = {buf_b.ptr(),    rocm_ck::makeShape(K, N), rocm_ck::makeStrides(b_sk, b_sn)};
    args.tensors[kSpec.output().args_slot] = {buf_e.ptr(),    rocm_ck::makeShape(M, N), rocm_ck::makeStrides(e_sm, e_sn)};
    args.tensors[kSpec.d0().args_slot]     = {buf_bias.ptr(), rocm_ck::makeShape(M, N), rocm_ck::makeStrides(d_sm, d_sn)};

    // --- Launch ---
    int grid_m = (M + kSpec.block_tile.m - 1) / kSpec.block_tile.m;
    int grid_n = (N + kSpec.block_tile.n - 1) / kSpec.block_tile.n;

    std::printf("gemm_fp16_add_relu: M=%d, N=%d, K=%d, grid=%dx1x1, block=%d\n",
                M, N, K, grid_m * grid_n, kSpec.workgroup_size);

    gemm_fp16_add_relu<<<dim3(grid_m * grid_n), dim3(kSpec.workgroup_size)>>>(args);
    HIP_CHECK(hipDeviceSynchronize());

    // --- Verify ---
    std::vector<float> result(M * N);
    buf_e.download(result.data());

    auto vr = rocm_ck::verify(result.data(), ref.data(), M * N, kSpec.output().dtype);
    rocm_ck::reportVerify("gemm_fp16_add_relu", vr);

    return vr.passed ? 0 : 1;
}
