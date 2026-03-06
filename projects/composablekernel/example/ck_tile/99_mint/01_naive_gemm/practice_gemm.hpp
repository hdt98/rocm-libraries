// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <mint/mint.h>

// NOTE: The hierarchical implementation (host_level/, block_level/, warp_level/)
// demonstrates the intended structure but has compilation issues due to current
// MINT API limitations with constexpr in device code.
//
// The simple implementation below works and demonstrates MINT tensor concepts.
//
// Uncomment to use hierarchical version (currently has build errors):
// #include "host_level/grid_gemm.hpp"
// #include "block_level/block_gemm_pipeline_agmem_bgmem_creg.hpp"
// #include "warp_level/block_gemm_asmem_bsmem_creg.hpp"

#include "gemm_simple_mint.hpp"

namespace mint_gemm {

using namespace mint;
using namespace mint::tensor;

// Problem definition
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename AccDataType_,
          index_t kMPerBlock_,
          index_t kNPerBlock_,
          index_t kKPerBlock_,
          index_t kBlockSize_>
struct GemmProblem
{
    using ADataType = ADataType_;
    using BDataType = BDataType_;
    using CDataType = CDataType_;
    using AccDataType = AccDataType_;

    struct BlockGemmShape
    {
        static constexpr index_t kM = kMPerBlock_;
        static constexpr index_t kN = kNPerBlock_;
        static constexpr index_t kK = kKPerBlock_;
    };

    static constexpr index_t kBlockSize = kBlockSize_;
};

// Main GEMM kernel - delegates to simple implementation
template <typename Problem>
__global__ void gemm_kernel(const typename Problem::ADataType* __restrict__ p_a,
                           const typename Problem::BDataType* __restrict__ p_b,
                           typename Problem::CDataType* __restrict__ p_c,
                           index_t M,
                           index_t N,
                           index_t K)
{
    // Directly call simple MINT GEMM implementation
    mint_gemm_simple::gemm_simple_impl<
        typename Problem::ADataType,
        typename Problem::BDataType,
        typename Problem::CDataType,
        typename Problem::AccDataType,
        Problem::BlockGemmShape::kM,
        Problem::BlockGemmShape::kN,
        Problem::BlockGemmShape::kK,
        Problem::kBlockSize>(p_a, p_b, p_c, M, N, K);
}

} // namespace mint_gemm
