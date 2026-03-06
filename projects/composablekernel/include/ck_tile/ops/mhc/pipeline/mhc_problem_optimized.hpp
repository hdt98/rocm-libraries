// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_gemm_shape.hpp"

namespace ck_tile {

// MHC Problem Optimized - Single-Pass GEMM (No Split-K)
// Key optimizations:
// 1. Large K tile to process entire C dimension in one pass
// 2. Larger M tiles (64/128) for more work per block
// 3. N=24 to match exact output dimension (no padding)
template <typename XDataType_, typename ComputeDataType_, typename YDataType_, index_t MTile_ = 64>
struct MHCProblemOptimized
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;
    using PhiDataType     = XDataType;
    using ADataType       = XDataType;
    using BDataType       = PhiDataType;
    using CDataType       = ComputeDataType;

    static constexpr index_t kMTile = MTile_;

    // Optimized GEMM with larger M tiles
    // Keep K=64 and WarpGemm same as baseline
    // Use NumWarps=1 to avoid multi-warp accumulation issues
    using BlockGemmShape = std::conditional_t<
        MTile_ == 64,
        TileGemmShape<sequence<64, 32, 64>, sequence<1, 1, 1>, sequence<64, 32, 16>>,
        std::conditional_t<
            MTile_ == 128,
            TileGemmShape<sequence<128, 32, 64>, sequence<1, 1, 1>, sequence<128, 32, 16>>,
            TileGemmShape<sequence<32, 32, 64>, sequence<1, 1, 1>, sequence<32, 32, 16>>>>;

    static constexpr index_t VectorSizeA = 4;
    static constexpr index_t VectorSizeB = 4;

    // Keep BlockSize=64 (1 warp) for all M tiles to avoid multi-warp issues
    // Larger M tiles will use M0 (grid/repeat) dimension instead
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
        return MTile_ == 32   ? "MHCProblemOptimized_M32_SinglePass"
               : MTile_ == 64 ? "MHCProblemOptimized_M64_SinglePass"
                              : "MHCProblemOptimized_M128_SinglePass";
    }

    // LDS block descriptors (3D + padding)
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = BlockGemmShape::kM;
        constexpr index_t kKPerBlock = BlockGemmShape::kK;

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / 8>{}, number<kMPerBlock>{}, number<8>{}),
            make_tuple(number<(kMPerBlock + 1) * 8>{}, number<8>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return a_lds_block_desc;
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = BlockGemmShape::kN;
        constexpr index_t kKPerBlock = BlockGemmShape::kK;

        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / 8>{}, number<kNPerBlock>{}, number<8>{}),
            make_tuple(number<(kNPerBlock + 1) * 8>{}, number<8>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return b_lds_block_desc;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
        constexpr index_t smem_size_a =
            sizeof(ADataType) * MakeALdsBlockDescriptor().get_element_space_size() / PackedSize;
        return smem_size_a;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeB()
    {
        constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;
        constexpr index_t smem_size_b =
            sizeof(BDataType) * MakeBLdsBlockDescriptor().get_element_space_size() / PackedSize;
        return smem_size_b;
    }

    // Tile distributions optimized for larger tiles
    CK_TILE_HOST_DEVICE static constexpr auto MakeXLoadTileDistribution()
    {
        constexpr index_t kMPerBlock = BlockGemmShape::kM;
        constexpr index_t kKPerBlock = BlockGemmShape::kK;
        constexpr index_t K1         = 16 / sizeof(XDataType);
        constexpr index_t K0         = kKPerBlock / K1;
        constexpr index_t M2         = get_warp_size() / K0;
        constexpr index_t M1         = kBlockSize / get_warp_size();
        constexpr index_t M0         = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakePhiLoadTileDistribution()
    {
        constexpr index_t kNPerBlock = BlockGemmShape::kN;
        constexpr index_t kKPerBlock = BlockGemmShape::kK;
        constexpr index_t K1         = VectorLoadSize / sizeof(PhiDataType);
        constexpr index_t K0         = kKPerBlock / K1;
        constexpr index_t N2         = get_warp_size() / K0;
        constexpr index_t N1         = kBlockSize / get_warp_size();
        constexpr index_t N0         = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }
};

} // namespace ck_tile
