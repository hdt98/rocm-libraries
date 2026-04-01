// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/common/generic_2d_block_shape.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp"

namespace ck_tile {

// MHC Problem - Reusing GEMM Pipeline Tile Distributions
template <typename XDataType_, typename ComputeDataType_, typename YDataType_, index_t MTile_ = 16>
struct MHCProblemGemmDist
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;

    using PhiDataType = XDataType;
    using ADataType   = XDataType;
    using BDataType   = PhiDataType;
    using CDataType   = ComputeDataType;

    static constexpr index_t kMTile = MTile_;

    // Validate MTile value at compile time
    static_assert(MTile_ == 16 || MTile_ == 32 || MTile_ == 64 || MTile_ == 128,
                  "MHC kernel only supports MTile values of 16, 32, 64, or 128. ");

    // BlockGemmShape defines the warp grid and tile sizes
    // Use ONLY WarpTile configs supported by WarpGemmDispatcher for bf16!
    // M=16: 1 warp, WarpTile 16×16×32
    // M=32: 1 warp, WarpTile 32×32×16
    // M=64: 2 warps (2×1), WarpTile 32×32×16, Block K=64
    // M=128: 4 warps (4×1), WarpTile 32×32×16, Block K=64
    using BlockGemmShape = std::conditional_t<
        MTile_ == 16,
        TileGemmShape<sequence<16, 32, 64>, sequence<1, 1, 1>, sequence<16, 16, 32>>,
        std::conditional_t<
            MTile_ == 32,
            TileGemmShape<sequence<32, 32, 64>, sequence<1, 1, 1>, sequence<32, 32, 16>>,
            std::conditional_t<
                MTile_ == 64,
                TileGemmShape<sequence<64, 32, 64>, sequence<2, 1, 1>, sequence<32, 32, 16>>,
                TileGemmShape<sequence<128, 32, 64>, sequence<4, 1, 1>, sequence<32, 32, 16>>>>>;

    // Standard vector sizes for all configurations
    static constexpr index_t VectorSizeA = 4;
    static constexpr index_t VectorSizeB = 4;

    // Adaptive block size based on number of warps
    // M=16, M=32: 1 warp = 64 threads
    // M=64: 2 warps (2×1 grid) = 128 threads
    // M=128: 4 warps (4×1 grid) = 256 threads
    using BlockShape = std::conditional_t<
        MTile_ == 16 || MTile_ == 32,
        Generic2dBlockShape<sequence<1, 64>, sequence<1, 64>, sequence<1, 1>>,
        std::conditional_t<
            MTile_ == 64,
            Generic2dBlockShape<sequence<1, 128>, sequence<1, 128>, sequence<1, 1>>,
            Generic2dBlockShape<sequence<1, 256>, sequence<1, 256>, sequence<1, 1>>>>;

    // Layouts: X is row-major, Phi is column-major (transposed for GEMM)
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
        return MTile_ == 16   ? "MHCProblemGemmDist_M16"
               : MTile_ == 32 ? "MHCProblemGemmDist_M32"
               : MTile_ == 64 ? "MHCProblemGemmDist_M64"
                              : "MHCProblemGemmDist_M128";
    }

    // Use default GEMM tile distributions - they are proven to work with multi-warp!

    CK_TILE_HOST_DEVICE static constexpr auto MakeXLoadTileDistribution()
    {
        // X is A matrix in GEMM (M × K, row-major)
        return GemmPipelineAGmemBGmemCRegV1DefaultPolicy::MakeADramTileDistribution<
            MHCProblemGemmDist>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakePhiLoadTileDistribution()
    {
        // Phi is B matrix in GEMM (N × K, column-major for transposed access)
        return GemmPipelineAGmemBGmemCRegV1DefaultPolicy::MakeBDramTileDistribution<
            MHCProblemGemmDist>();
    }
};

} // namespace ck_tile
