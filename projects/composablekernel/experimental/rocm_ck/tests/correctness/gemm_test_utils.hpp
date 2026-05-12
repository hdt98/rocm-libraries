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

} // namespace rocm_ck::test
