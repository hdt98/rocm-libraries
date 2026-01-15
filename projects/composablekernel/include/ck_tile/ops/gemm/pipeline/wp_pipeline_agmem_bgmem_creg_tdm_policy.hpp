// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

struct UniversalWeightPreshufflePipelineAgBgCrTDMPolicy
    : public GemmPipelineAgBgCrCompTDMDefaultPolicy<false>
{

    using Base = GemmPipelineAgBgCrCompTDMDefaultPolicy<false>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        constexpr index_t smem_size_a = sizeof(typename Problem::ADataType) *
                                        MakeALdsBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_a;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size_a = GetSmemSizeA<Problem>();

        return smem_size_a;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemPackA()
    {
        return Problem::VectorLoadSize / sizeof(typename Problem::ADataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetKBPerLoad()
    {
        using TileShape = typename Problem::BlockGemmShape;
        return TileShape::WarpTile::at(I2) / 2;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeBFlatDramTileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape;

        constexpr index_t kNPerBlock = TileShape::kN;
        constexpr index_t kKPerBlock = TileShape::kK;
        constexpr index_t NIterPerWarp =
            kNPerBlock / TileShape::BlockWarps::at(I1) / TileShape::WarpTile::at(I1);
        constexpr index_t KIterPerWarp = kKPerBlock / TileShape::WarpTile::at(I2);

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t WaveNum   = BlockSize / WaveSize;

        constexpr index_t PackedSize = numeric_traits<typename Problem::BDataType>::PackedSize;

        constexpr index_t KBPerLoad     = GetKBPerLoad<Problem>();
        constexpr index_t MaxVecSize    = 16 / sizeof(typename Problem::BDataType) * PackedSize;
        constexpr index_t KItemsPerLoad = min(KBPerLoad, MaxVecSize);
        constexpr index_t KFragment     = KBPerLoad / KItemsPerLoad;

        constexpr index_t KThdPerWave = WaveSize;
        constexpr index_t KWavePerBlk = 1;
        constexpr index_t KRepeat     = KIterPerWarp;
        static_assert(TileShape::flatKPerWarp == KThdPerWave * KBPerLoad, "wrong");

        constexpr index_t NBPerLoad   = 1;
        constexpr index_t NThdPerWave = 1;
        constexpr index_t NWavePerBlk = TileShape::BlockWarps::at(number<1>{}); // N_Warp
        constexpr index_t NRepeat     = NIterPerWarp;

        constexpr index_t WaveRepeat = WaveNum / TileShape::flatNPerWarp;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<WaveRepeat>,
                tuple<sequence<NRepeat, NWavePerBlk, NThdPerWave, NBPerLoad>, // second
                                                                              // direction
                      sequence<KRepeat,
                               KFragment,
                               KWavePerBlk,
                               KThdPerWave,
                               KItemsPerLoad>>, // first
                                                // direction
                // wave in blk,     // thd in wave
                // <M, K>           // <M, K>
                tuple<sequence<0, 1, 2>, sequence<1, 2>>, // which direction
                tuple<sequence<0, 1, 2>, sequence<2, 3>>, // which index
                // <repeat, vec_load>
                sequence<1, 2, 2, 2>,
                sequence<0, 0, 1, 4>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockWeightPreshuffle()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;
        using BTypeToUse =
            std::conditional_t<std::is_same_v<typename Problem::BDataType, ck_tile::pk_int4_t>,
                               typename Problem::ADataType,
                               typename Problem::BDataType>;
        using WarpGemm = WarpGemmDispatcher<typename Problem::ADataType,
                                            BTypeToUse,
                                            typename Problem::CDataType,
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC>;

        using BlockWeightPreshufflePolicy =
            BlockWeightPreshuffleASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                              typename Problem::BDataType,
                                                              typename Problem::CDataType,
                                                              BlockWarps,
                                                              WarpGemm>;
        return BlockWeightPreshuffleASmemBRegCReg<Problem, BlockWeightPreshufflePolicy>{};
    }
};

} // namespace ck_tile
