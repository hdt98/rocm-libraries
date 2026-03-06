// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <mint/mint.h>

namespace mint {

// Policy for warp-level GEMM configuration using MINT
// This is analogous to CK_Tile's BlockGemmASmemBSmemCRegPolicy
struct BlockGemmASmemBSmemCRegPolicy
{
    // Warp tile sizes for matrix multiplication
    // Each warp processes a 32x32x8 tile
    static constexpr index_t kMPerWarp = 32;
    static constexpr index_t kNPerWarp = 32;
    static constexpr index_t kKPerWarp = 8;

    // Warp configuration: 4 warps in M dimension, 1 warp in N dimension
    static constexpr index_t kMWarp = 4;
    static constexpr index_t kNWarp = 1;
};

} // namespace mint
