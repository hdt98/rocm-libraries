// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_gemm_shape.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp"

namespace ck_tile {

// MHC Problem - K=128 Configuration
// Minimal change from baseline: just increase K tile
//
// Key configuration:
// - M=32 (same as baseline)
// - N=32 (same as baseline)
// - K=128 (2x larger - fewer syncs, more work per iteration)
// - 1 warp (same as baseline)
// - BlockSize=64 threads
// - Expected: 28-32 TFLOPS

template <typename XDataType_, typename ComputeDataType_, typename YDataType_>
struct MHCProblemK128
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;

    using PhiDataType = XDataType;
    using ADataType   = XDataType;
    using BDataType   = PhiDataType;
    using CDataType   = ComputeDataType;

    // Minimal change: just increase K
    static constexpr index_t kMTile = 32;  // Same as baseline
    static constexpr index_t kNTile = 32;  // Same as baseline
    static constexpr index_t kKTile = 128; // 2x larger!

    // Logical N for MHC (actual output dimension)
    static constexpr index_t kNTileLogical = 24; // 2n + n^2 for n=4

    // Single-warp configuration (same as baseline)
    // BlockGemmShape: <M, N, K> block tile, <M, N, K> warp grid, <M, N, K> warp tile
    // M=32, N=32, K=128
    // Warp grid: 1×1×1 (single warp)
    // Warp tile: 32×32×16 (warp processes 32×32×16)
    using BlockGemmShape = TileGemmShape<sequence<32, 32, 128>, // Block tile
                                         sequence<1, 1, 1>,     // Warp grid (single warp)
                                         sequence<32, 32, 16>>; // Warp tile

    // 1 warp = 64 threads
    static constexpr index_t kBlockSize = 64;

    using BlockShape = Generic2dBlockShape<sequence<1, 64>, sequence<1, 64>, sequence<1, 1>>;

    // Vector sizes
    static constexpr index_t VectorSizeA = 4;
    static constexpr index_t VectorSizeB = 4;

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
    static constexpr bool DoubleSmemBuffer      = false; // V3 uses single buffer
    static constexpr bool UseStructuredSparsity = false;
    static constexpr bool FixedVectorSize       = false;

    struct Traits
    {
        static constexpr bool UsePersistentKernel = false;
    };

    CK_TILE_HOST static const std::string GetName() { return "MHCProblemK128_M32_N32_K128_1Warp"; }

    // Use default GEMM tile distributions
    CK_TILE_HOST_DEVICE static constexpr auto MakeXLoadTileDistribution()
    {
        // X is A matrix in GEMM (M × K, row-major)
        return GemmPipelineAGmemBGmemCRegV1DefaultPolicy::MakeADramTileDistribution<
            MHCProblemK128>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakePhiLoadTileDistribution()
    {
        // Phi is B matrix in GEMM (N × K, column-major for transposed access)
        return GemmPipelineAGmemBGmemCRegV1DefaultPolicy::MakeBDramTileDistribution<
            MHCProblemK128>();
    }
};

} // namespace ck_tile
