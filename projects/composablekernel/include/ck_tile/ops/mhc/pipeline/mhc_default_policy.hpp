// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

// Default policy for MHC kernel
// This policy provides warp gemm configuration for MHC operations
struct MHCDefaultPolicy
{
    // Check if the argument types are supported
    template <typename ADataType, typename BDataType, typename CDataType>
    CK_TILE_HOST_DEVICE static constexpr bool IsSupportedArguments()
    {
        // A and B must be the same type
        if constexpr(!std::is_same_v<ADataType, BDataType>)
        {
            return false;
        }

        // C must be float for all supported input types
        if constexpr(!std::is_same_v<CDataType, float>)
        {
            return false;
        }

        // Check supported input types
        constexpr bool is_supported_type =
            std::is_same_v<ADataType, bf16_t> || std::is_same_v<ADataType, half_t> ||
            std::is_same_v<ADataType, fp8_t> || std::is_same_v<ADataType, float>;

        return is_supported_type;
    }

    // Provide warp gemm configuration for various data types and block shapes
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        using ADataType = typename Problem::ADataType;
        using BDataType = typename Problem::BDataType;
        using CDataType = typename Problem::CDataType;

        // Validate argument types:
        // Check if we support this configuration with MFMA
        static_assert(IsSupportedArguments<ADataType, BDataType, CDataType>(),
                      "MHC kernel requires: A and B must be the same type"
                      ", and C must be float");

        // Read warp configuration directly from Problem's BlockGemmShape
        // This allows the Problem definition to control the warp layout
        constexpr index_t kWarpM = Problem::BlockGemmShape::WarpTile::at(number<0>{});
        constexpr index_t kWarpN = Problem::BlockGemmShape::WarpTile::at(number<1>{});
        constexpr index_t kWarpK = Problem::BlockGemmShape::WarpTile::at(number<2>{});

        constexpr index_t kMWarp = Problem::BlockGemmShape::BlockWarps::at(number<0>{});
        constexpr index_t kNWarp = Problem::BlockGemmShape::BlockWarps::at(number<1>{});

        // Create WarpGemm dispatcher with dimensions from BlockGemmShape
        using WG = WarpGemmDispatcher<ADataType,
                                      BDataType,
                                      CDataType,
                                      kWarpM,
                                      kWarpN,
                                      kWarpK,
                                      true,  // TransposeC
                                      false, // SwizzleA
                                      false, // UseStructuredSparsity
                                      WGAttrNumAccessEnum::Single>;

        return make_tuple(WG{}, kMWarp, kNWarp);
    }
};

} // namespace ck_tile
