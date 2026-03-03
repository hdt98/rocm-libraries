// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_gemm_shape.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp"

namespace ck_tile {

// MHC Problem - Small Tiles for Higher Occupancy
// Based on profiling analysis: reduce LDS usage to allow more blocks per CU
// Strategy: M=16, N=16, K=32 to minimize LDS footprint

template <typename XDataType_, typename ComputeDataType_, typename YDataType_>
struct MHCProblemSmallTiles
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;

    using PhiDataType = XDataType;
    using ADataType   = XDataType;
    using BDataType   = PhiDataType;
    using CDataType   = ComputeDataType;

    static constexpr index_t kMTile = 32;

    // Balanced tiles: M=32, N=32, K=32
    // LDS usage: (32×32 + 32×32) × 2 bytes = 4KB (same as baseline M=16,N=16,K=64)
    // Register usage: result_tile = 32×32 = 1024 elements (4x baseline)
    // BUT: Half the K dimension means half the K-tile iterations (8 instead of 16)
    // This reduces the time result_tile must stay alive, potentially reducing spilling
    using BlockGemmShape =
        TileGemmShape<sequence<32, 32, 32>, sequence<1, 1, 1>, sequence<32, 32, 16>>;

    static constexpr index_t VectorSizeA = 4;
    static constexpr index_t VectorSizeB = 4;

    using BlockShape = Generic2dBlockShape<sequence<1, 64>, sequence<1, 64>, sequence<1, 1>>;

    using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    using AsDataTypeTuple = tuple<ADataType>;
    using BsDataTypeTuple = tuple<BDataType>;
    using AsLayoutTuple   = tuple<ALayout>;
    using BsLayoutTuple   = tuple<BLayout>;

    using AElementWise = identity;
    using BElementWise = identity;

    static constexpr bool TransposeC            = false;
    static constexpr bool kPadM                 = true;
    static constexpr bool kPadN                 = true;
    static constexpr bool kPadK                 = true;
    static constexpr bool Preshuffle            = false;
    static constexpr auto Scheduler             = GemmPipelineScheduler::Intrawave;
    static constexpr index_t NumWaveGroups      = 1;
    static constexpr index_t VectorLoadSize     = 16;
    static constexpr index_t kBlockSize         = BlockShape::BlockSize;
    static constexpr bool DoubleSmemBuffer      = true;
    static constexpr bool UseStructuredSparsity = false;
    static constexpr bool FixedVectorSize       = false;

    struct Traits
    {
        static constexpr bool UsePersistentKernel = false;
    };

    CK_TILE_HOST static const std::string GetName() { return "MHCProblemSmallTiles_M16_N16_K32"; }

    CK_TILE_HOST_DEVICE static constexpr auto MakeXLoadTileDistribution()
    {
        return GemmPipelineAGmemBGmemCRegV1DefaultPolicy::MakeADramTileDistribution<
            MHCProblemSmallTiles>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakePhiLoadTileDistribution()
    {
        return GemmPipelineAGmemBGmemCRegV1DefaultPolicy::MakeBDramTileDistribution<
            MHCProblemSmallTiles>();
    }
};

// MHC Problem - Small Tiles with Lower LDS (P0 improvement)
// Same as MHCProblemSmallTiles but K=16 to halve LDS per block (~2KB vs ~4KB).
// Trade-off: 2x more K-tile iterations per block; use when "Insufficient CU LDS" is high.
template <typename XDataType_, typename ComputeDataType_, typename YDataType_>
struct MHCProblemSmallTilesLowLDS
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;

    using PhiDataType = XDataType;
    using ADataType   = XDataType;
    using BDataType   = PhiDataType;
    using CDataType   = ComputeDataType;

    static constexpr index_t kMTile = 32;

    // M=32, N=32, K=16 → half the LDS of SmallTiles, same result_tile size
    using BlockGemmShape =
        TileGemmShape<sequence<32, 32, 16>, sequence<1, 1, 1>, sequence<32, 32, 16>>;

    static constexpr index_t VectorSizeA = 4;
    static constexpr index_t VectorSizeB = 4;

    using BlockShape = Generic2dBlockShape<sequence<1, 64>, sequence<1, 64>, sequence<1, 1>>;

    using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    using AsDataTypeTuple = tuple<ADataType>;
    using BsDataTypeTuple = tuple<BDataType>;
    using AsLayoutTuple   = tuple<ALayout>;
    using BsLayoutTuple   = tuple<BLayout>;

    using AElementWise = identity;
    using BElementWise = identity;

    static constexpr bool TransposeC            = false;
    static constexpr bool kPadM                 = true;
    static constexpr bool kPadN                 = true;
    static constexpr bool kPadK                 = true;
    static constexpr bool Preshuffle            = false;
    static constexpr auto Scheduler             = GemmPipelineScheduler::Intrawave;
    static constexpr index_t NumWaveGroups      = 1;
    static constexpr index_t VectorLoadSize     = 16;
    static constexpr index_t kBlockSize         = BlockShape::BlockSize;
    static constexpr bool DoubleSmemBuffer      = true;
    static constexpr bool UseStructuredSparsity = false;
    static constexpr bool FixedVectorSize       = false;

    struct Traits
    {
        static constexpr bool UsePersistentKernel = false;
    };

    CK_TILE_HOST static const std::string GetName()
    {
        return "MHCProblemSmallTilesLowLDS_M32_N32_K16";
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeXLoadTileDistribution()
    {
        return GemmPipelineAGmemBGmemCRegV1DefaultPolicy::MakeADramTileDistribution<
            MHCProblemSmallTilesLowLDS>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakePhiLoadTileDistribution()
    {
        return GemmPipelineAGmemBGmemCRegV1DefaultPolicy::MakeBDramTileDistribution<
            MHCProblemSmallTilesLowLDS>();
    }
};

} // namespace ck_tile
