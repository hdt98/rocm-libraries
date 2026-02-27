// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// MHC Shape V5 - Defines tile distribution variables
// Following the pattern from ElementWiseShape
template <typename BlockGemmShape_, typename ComputeDataType_>
struct MHCShapeV5
{
    using BlockGemmShape  = remove_cvref_t<BlockGemmShape_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;

    // Block tile sizes from BlockGemmShape
    static constexpr index_t kBlockM = BlockGemmShape::kM;
    static constexpr index_t kBlockN = BlockGemmShape::kN;
    static constexpr index_t kBlockK = BlockGemmShape::kK;

    // Warp configuration from BlockGemmShape
    static constexpr index_t kWarpPerBlockM = BlockGemmShape::BlockWarps::at(number<0>{});
    static constexpr index_t kWarpPerBlockN = BlockGemmShape::BlockWarps::at(number<1>{});
    static constexpr index_t kWarpPerBlockK = BlockGemmShape::BlockWarps::at(number<2>{});

    // For X loading (M × K tile)
    // We want to use all threads efficiently
    static constexpr index_t kThreadPerWarpM = 16; // Threads per warp in M dimension
    static constexpr index_t kThreadPerWarpK =
        get_warp_size() / kThreadPerWarpM; // Remaining threads in K

    // Vector sizes for memory coalescing
    static constexpr index_t kVectorM = min(4, kBlockM / (kWarpPerBlockM * kThreadPerWarpM));
    static constexpr index_t kVectorK = kBlockK / kThreadPerWarpK;

    // Repeat to cover full tile
    static constexpr index_t kRepeatM = kBlockM / (kWarpPerBlockM * kThreadPerWarpM * kVectorM);
    static constexpr index_t kRepeatK = 1; // Usually no repeat in K

    // For Phi loading (N × K tile)
    static constexpr index_t kThreadPerWarpN = 16; // Threads per warp in N dimension
    // kThreadPerWarpK already defined above

    static constexpr index_t kVectorN = min(4, kBlockN / (kWarpPerBlockN * kThreadPerWarpN));
    // kVectorK already defined above

    static constexpr index_t kRepeatN = kBlockN / (kWarpPerBlockN * kThreadPerWarpN * kVectorN);
    // kRepeatK already defined above

    // Total block size
    static constexpr index_t kBlockSize =
        get_warp_size() * kWarpPerBlockM * kWarpPerBlockN * kWarpPerBlockK;

    // Validation
    static_assert(kRepeatM * kWarpPerBlockM * kThreadPerWarpM * kVectorM == kBlockM,
                  "M dimension doesn't tile correctly!");
    static_assert(kRepeatK * kWarpPerBlockK * kThreadPerWarpK * kVectorK == kBlockK,
                  "K dimension doesn't tile correctly!");
    static_assert(kRepeatN * kWarpPerBlockN * kThreadPerWarpN * kVectorN == kBlockN,
                  "N dimension doesn't tile correctly!");
};

} // namespace ck_tile
