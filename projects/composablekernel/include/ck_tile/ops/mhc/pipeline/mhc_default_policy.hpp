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

        // Check supported input types: bf16, fp16 (half_t), fp8, or float
        constexpr bool is_supported_type =
            std::is_same_v<ADataType, bf16_t> || std::is_same_v<ADataType, half_t> ||
            std::is_same_v<ADataType, fp8_t> || std::is_same_v<ADataType, float>;

        return is_supported_type;
    }

    // Helper to compute warp tile dimensions based on data type and block shape
    template <typename ADataType, index_t kM, index_t kN, index_t kK>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpTileDimensions()
    {
        // Standard warp tile size for MFMA operations
        constexpr index_t kWarpM = 16;
        constexpr index_t kWarpN = 16;

        // K dimension varies by data type to match MFMA instruction capabilities
        constexpr index_t kWarpK =
            std::is_same_v<ADataType, float> ? 16 : // f32: 16x16x16
                std::is_same_v<ADataType, bf16_t> ? (kK == 32 ? 32 : 16)
                                                  : // bf16: 16x16x16 or 16x16x32
                std::is_same_v<ADataType, half_t> ? (kK == 32 ? 32 : 16)
                                                  : // fp16: 16x16x16 or 16x16x32
                std::is_same_v<ADataType, fp8_t> ? (kK == 32 ? 32 : 16)
                                                 : // fp8: 16x16x16 or 16x16x32
                16;                                // default

        return make_tuple(number<kWarpM>{}, number<kWarpN>{}, number<kWarpK>{});
    }

    // Helper to compute number of warps in M and N dimensions
    template <index_t kM, index_t kN, index_t kWarpM, index_t kWarpN>
    CK_TILE_HOST_DEVICE static constexpr auto GetNumWarps()
    {
        // Special case: asymmetric 16x32 block uses 2 warps in N dimension
        if constexpr(kM == 16 && kN == 32)
        {
            return make_tuple(number<1>{}, number<2>{}); // 1 warp in M, 2 warps in N
        }
        else
        {
            // Standard case: 1 warp in both dimensions
            return make_tuple(number<1>{}, number<1>{}); // 1 warp in M, 1 warp in N
        }
    }

    // Provide warp gemm configuration for various data types and block shapes
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        using ADataType = typename Problem::ADataType;
        using BDataType = typename Problem::BDataType;
        using CDataType = typename Problem::CDataType;

        constexpr index_t kM = Problem::BlockGemmShape::kM;
        constexpr index_t kN = Problem::BlockGemmShape::kN;
        constexpr index_t kK = Problem::BlockGemmShape::kK;

        // Validate argument types
        static_assert(IsSupportedArguments<ADataType, BDataType, CDataType>(),
                      "MHC kernel requires: A and B must be the same type (bf16, fp16, fp8, or "
                      "float), and C must be float");

        // Check if we support this configuration with MFMA
        if constexpr(IsSupportedArguments<ADataType, BDataType, CDataType>())
        {
            // Get warp tile dimensions based on data type and block shape
            constexpr auto warp_dims = GetWarpTileDimensions<ADataType, kM, kN, kK>();
            constexpr index_t kWarpM = warp_dims.template at<0>().value;
            constexpr index_t kWarpN = warp_dims.template at<1>().value;
            constexpr index_t kWarpK = warp_dims.template at<2>().value;

            // Get number of warps in M and N dimensions
            constexpr auto num_warps = GetNumWarps<kM, kN, kWarpM, kWarpN>();
            constexpr index_t kMWarp = num_warps.template at<0>().value;
            constexpr index_t kNWarp = num_warps.template at<1>().value;

            // Create WarpGemm dispatcher with computed dimensions
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
        else
        {
            // For unsupported data types, delegate to default policy
            return BlockGemmASmemBSmemCRegV1DefaultPolicy::GetWarpGemmMWarpNWarp<Problem>();
        }
    }

    // Get shared memory size needed for the kernel
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // For MHC, we need shared memory for the BlockGemm operations
        // The size depends on the block shape and data types
        // This is a placeholder - actual size calculation would depend on
        // the specific BlockGemm implementation requirements
        return 0; // Will be calculated by BlockGemm internally
    }
};

} // namespace ck_tile
