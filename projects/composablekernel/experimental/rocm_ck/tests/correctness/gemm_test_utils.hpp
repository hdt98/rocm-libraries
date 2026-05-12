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

/// CPU reference GEMM: C = A * B using strides from Args tensor slots.
/// Reads strides from the Args struct so it matches whatever layout
/// the spec set.  Replace with a GPU reference later.
inline void cpuGemm(const float* a,
                    const float* b,
                    float* c,
                    int M,
                    int N,
                    int K,
                    const rocm_ck::Args& args,
                    int a_slot,
                    int b_slot,
                    int c_slot)
{
    int64_t a_s0 = args.tensors[a_slot].strides[0];
    int64_t a_s1 = args.tensors[a_slot].strides[1];
    int64_t b_s0 = args.tensors[b_slot].strides[0];
    int64_t b_s1 = args.tensors[b_slot].strides[1];
    int64_t c_s0 = args.tensors[c_slot].strides[0];
    int64_t c_s1 = args.tensors[c_slot].strides[1];

    for(int m = 0; m < M; ++m)
        for(int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for(int k = 0; k < K; ++k)
                sum += a[m * a_s0 + k * a_s1] * b[k * b_s0 + n * b_s1];
            c[m * c_s0 + n * c_s1] = sum;
        }
}

} // namespace rocm_ck::test
