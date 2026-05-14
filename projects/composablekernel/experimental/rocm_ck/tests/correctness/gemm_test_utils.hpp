// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Minimal GEMM test harness for correctness tests.
//
// Provides Args construction from a GemmSpec and grid computation.
// Strides are computed once here from the spec — both fill kernels
// and compute kernels read from the same Args struct.

#pragma once

#include <rocm_ck/args.hpp>
#include <rocm_ck/gemm_spec.hpp>
#include <rocm_ck/hip_check.hpp>
#include <rocm_ck/layout.hpp>
#include <rocm_ck/typed_buffer.hpp>
#include <rocm_ck/verify.hpp>

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <string_view>

namespace rocm_ck::test {

/// Skip the current test if no GPU is available.
inline void skipIfNoGpu()
{
    int count = 0;
    auto err  = hipGetDeviceCount(&count);
    if(err != hipSuccess || count == 0)
        GTEST_SKIP() << "No GPU available";
}

/// Build rocm_ck::Args for a plain GEMM (no epilogue D tensors).
/// Strides are derived from the spec's layout for each tensor.
/// This is the single source of truth — fill and compute both use this Args.
template <typename Spec>
rocm_ck::Args buildGemmArgs(const Spec& spec,
                            void* a_ptr,
                            void* b_ptr,
                            void* c_ptr,
                            int M,
                            int N,
                            int K)
{
    auto [a_sm, a_sk] = layoutStrides(spec.lhs().layout, M, K);
    auto [b_sk, b_sn] = layoutStrides(spec.rhs().layout, K, N);
    auto [c_sm, c_sn] = layoutStrides(spec.output().layout, M, N);

    rocm_ck::Args args{};
    args.tensors[spec.lhs().args_slot] =
        {a_ptr, rocm_ck::makeShape(M, K), rocm_ck::makeStrides(a_sm, a_sk)};
    args.tensors[spec.rhs().args_slot] =
        {b_ptr, rocm_ck::makeShape(K, N), rocm_ck::makeStrides(b_sk, b_sn)};
    args.tensors[spec.output().args_slot] =
        {c_ptr, rocm_ck::makeShape(M, N), rocm_ck::makeStrides(c_sm, c_sn)};

    return args;
}

/// Compute grid dimensions for a GEMM launch from the spec.
template <typename Spec>
dim3 gemmGrid(const Spec& spec, int M, int N)
{
    int grid_m = (M + spec.block_tile.m - 1) / spec.block_tile.m;
    int grid_n = (N + spec.block_tile.n - 1) / spec.block_tile.n;
    return dim3(static_cast<unsigned>(grid_m * grid_n), 1u,
                static_cast<unsigned>(spec.k_batch));
}

/// CPU reference GEMM: C = A * B, accumulating in FP64.
/// Reads tensor slots and strides from the spec via Args.
/// Inputs are float arrays (downloaded via TypedBuffer), which faithfully
/// represent the on-device dtype values.  FP64 accumulation gives a
/// high-precision reference; dtype-specific tolerances in verify() handle
/// the comparison against the GPU result.
template <typename Spec>
void cpuGemm(const Spec& spec,
             const float* a,
             const float* b,
             float* c,
             int M,
             int N,
             int K,
             const rocm_ck::Args& args)
{
    int64_t a_s0 = args.tensors[spec.lhs().args_slot].strides[0];
    int64_t a_s1 = args.tensors[spec.lhs().args_slot].strides[1];
    int64_t b_s0 = args.tensors[spec.rhs().args_slot].strides[0];
    int64_t b_s1 = args.tensors[spec.rhs().args_slot].strides[1];
    int64_t c_s0 = args.tensors[spec.output().args_slot].strides[0];
    int64_t c_s1 = args.tensors[spec.output().args_slot].strides[1];

    for(int m = 0; m < M; ++m)
        for(int n = 0; n < N; ++n)
        {
            double sum = 0.0;
            for(int k = 0; k < K; ++k)
                sum += static_cast<double>(a[m * a_s0 + k * a_s1])
                     * static_cast<double>(b[k * b_s0 + n * b_s1]);
            c[m * c_s0 + n * c_s1] = static_cast<float>(sum);
        }
}

/// Print a 2D matrix to a file, reading elements with strides from Args.
/// Pass "-" as path to write to stdout.
inline void dumpMatrix(const float* host_data,
                       const rocm_ck::Args& args,
                       int tensor_slot,
                       std::string_view path)
{
    int M      = args.tensors[tensor_slot].lengths[0];
    int N      = args.tensors[tensor_slot].lengths[1];
    int64_t s0 = args.tensors[tensor_slot].strides[0];
    int64_t s1 = args.tensors[tensor_slot].strides[1];

    FILE* f = (path == "-") ? stdout : std::fopen(path.data(), "w");
    if(!f)
    {
        std::fprintf(stderr, "dumpMatrix: cannot open %s\n", path.data());
        return;
    }

    std::fprintf(f, "# %d x %d  strides=(%ld, %ld)\n", M, N, s0, s1);
    for(int m = 0; m < M; ++m)
    {
        for(int n = 0; n < N; ++n)
        {
            if(n > 0)
                std::fputc(' ', f);
            std::fprintf(f, "%g", static_cast<double>(host_data[m * s0 + n * s1]));
        }
        std::fputc('\n', f);
    }

    if(f != stdout)
        std::fclose(f);
}

} // namespace rocm_ck::test
