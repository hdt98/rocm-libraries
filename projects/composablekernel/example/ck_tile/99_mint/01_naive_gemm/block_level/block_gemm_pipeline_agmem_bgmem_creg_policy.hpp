// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <mint/mint.h>
#include "../warp_level/block_gemm_asmem_bsmem_creg.hpp"

namespace mint {

// Policy for block-level GEMM pipeline
// Defines how to distribute block tiles across threads
struct BlockGemmPipelineAGmemBGmemCRegPolicy
{
    // Create distribution for A block tile (global memory to shared memory)
    template <typename Problem>
    MINT_HOST_DEVICE static constexpr auto MakeABlockCopyDistribution()
    {
        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;
        using ThreadBlock = thread_in_this_block<Problem::kBlockSize>;

        return tensor::make_simple_distribution(
            index_sequence<kMPerBlock, kKPerBlock>{},
            ThreadBlock{},
            sequence<alias_t, alias_t{"M"}, alias_t{"K"}>{},
            sequence<alias_t, alias_t{"Thread"}>{});
    }

    // Create distribution for B block tile (global memory to shared memory)
    template <typename Problem>
    MINT_HOST_DEVICE static constexpr auto MakeBBlockCopyDistribution()
    {
        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;
        using ThreadBlock = thread_in_this_block<Problem::kBlockSize>;

        return tensor::make_simple_distribution(
            index_sequence<kNPerBlock, kKPerBlock>{},
            ThreadBlock{},
            sequence<alias_t, alias_t{"N"}, alias_t{"K"}>{},
            sequence<alias_t, alias_t{"Thread"}>{});
    }

    // Get warp-level GEMM operator
    template <typename Problem>
    MINT_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        return BlockGemmASmemBSmemCReg<Problem>{};
    }
};

} // namespace mint
