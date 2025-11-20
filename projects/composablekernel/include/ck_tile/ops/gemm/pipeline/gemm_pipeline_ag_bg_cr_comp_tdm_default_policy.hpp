// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

enum class MultiCastDirection
{
    kM,
    kN,
    kMN
};

// Default policy for GemmPipelineAgBgCrCompTDM
template <bool WaveSpecialized = false>
struct GemmPipelineAgBgCrCompTDMDefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompTDMDefaultPolicy<WaveSpecialized>>
{
    using Base = UniversalGemmBasePolicy<GemmPipelineAgBgCrCompTDMDefaultPolicy<WaveSpecialized>>;

    static constexpr index_t VecByteSize = 16;
    // currently implement basic situation: the tile is divided into same parts
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        constexpr index_t BlockSize = Problem::kBlockSize;
        // for wave specialized policy, only one wave per workgroup will load A / B matrix from DRAM
        // to LDS
        constexpr index_t warpNum = WaveSpecialized ? 1 : (BlockSize / get_warp_size());

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        using ALayout = remove_cvref_t<
            std::tuple_element_t<number<0>{}, remove_cvref_t<typename Problem::AsLayoutTuple>>>;

        // Tile : MPerBlock X KPerBlock
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            if constexpr(!WaveSpecialized)
            {
                static_assert(MPerBlock % warpNum == 0, "MPerBlock should be divided by warpNum");
            }
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, MPerBlock / warpNum>, sequence<KPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
        // Tile : KPerBlock * MPerBlock
        else
        {
            if constexpr(!WaveSpecialized)
            {
                static_assert(KPerBlock % warpNum == 0, "KPerBlock should be divided by warpNum");
            }
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, KPerBlock / warpNum>, sequence<MPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        constexpr index_t BlockSize = Problem::kBlockSize;
        // for wave specialized policy, only one wave per workgroup will load A / B matrix from DRAM
        // to LDS
        constexpr index_t warpNum = WaveSpecialized ? 1 : (BlockSize / get_warp_size());

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        using BLayout = remove_cvref_t<
            std::tuple_element_t<number<0>{}, remove_cvref_t<typename Problem::BsLayoutTuple>>>;

        // Tile : KPerBlock X NPerBlock
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            if constexpr(!WaveSpecialized)
            {
                static_assert(KPerBlock % warpNum == 0, "KPerBlock should be divided by warpNum");
            }
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, KPerBlock / warpNum>, sequence<NPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
        // Tile : NPerBlock * KPerBlock
        else
        {
            if constexpr(!WaveSpecialized)
            {
                static_assert(NPerBlock % warpNum == 0, "NPerBlock should be divided by warpNum");
            }
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, NPerBlock / warpNum>, sequence<KPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t MPerBlock  = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock  = Problem::BlockGemmShape::kK;
        constexpr index_t AVectorLen = VecByteSize / sizeof(typename Problem::ADataType);
        return make_naive_tensor_descriptor(make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                            make_tuple(number<KPerBlock>{}, number<1>{}),
                                            number<AVectorLen>{},
                                            number<1>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t NPerBlock  = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock  = Problem::BlockGemmShape::kK;
        constexpr index_t BVectorLen = VecByteSize / sizeof(typename Problem::BDataType);
        return make_naive_tensor_descriptor(make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                                            make_tuple(number<KPerBlock>{}, number<1>{}),
                                            number<BVectorLen>{},
                                            number<1>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool isClusterLaunch()
    {
        constexpr index_t clusterM = Problem::BlockGemmShape::kclusterM;
        constexpr index_t clusterN = Problem::BlockGemmShape::kclusterN;
        constexpr index_t clusterK = Problem::BlockGemmShape::kclusterK;
        // cluster launch is enabled only when TilePartitioner uses cluster tile gemm shape and
        // cluster size > 1
        return is_cluster_tile_gemm_shape<typename Problem::BlockGemmShape>::value &&
               (clusterM * clusterN * clusterK > 1);
    }

    template <MultiCastDirection Direction, typename Problem>
    CK_TILE_DEVICE static uint16_t GetTDMWorkgroupMask(dim3 block_id_in_cluster)
    {
        constexpr index_t MCluster = Problem::BlockGemmShape::kclusterM;
        constexpr index_t NCluster = Problem::BlockGemmShape::kclusterN;

        auto is_participant = [&](auto i_m, auto i_n) {
            if constexpr(Direction == MultiCastDirection::kM)
            {
                return i_m == block_id_in_cluster.x;
            }
            else if constexpr(Direction == MultiCastDirection::kN)
            {
                return (i_n == block_id_in_cluster.y);
            }
            else // Direction == MultiCastDirection::kMN
            {
                return (i_m == block_id_in_cluster.x) || (i_n == block_id_in_cluster.y);
            }
        };

        // Iterate over all possible (m, n) block coordinates in the cluster. If the current (m,
        // n) block is a participant according to the multicast direction, set the corresponding bit
        // in the mask. for matmul AxB, A broadcasts from M direction, B broadcasts from N
        // direction.
        uint16_t block_id_mask = 0;
        static_for<0, NCluster, 1>{}([&](auto n) {
            static_for<0, MCluster, 1>{}([&](auto m) {
                if(is_participant(m, n))
                {
                    block_id_mask |= (1 << (n * MCluster + m));
                }
            });
        });
        return block_id_mask;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

#if defined(__gfx950__)
        constexpr index_t vector_size =
            DS_READ_TR_SIZE() / sizeof(typename Problem::ComputeDataType);
        constexpr index_t thread_elements =
            WarpTile::at(Base::I1) * WarpTile::at(Base::I2) / get_warp_size();
        constexpr auto wg_attr_num_access =
            !(Base::template is_a_load_tr<Problem> || Base::template is_b_load_tr<Problem>)
                ? WGAttrNumAccessEnum::Single
            : vector_size == thread_elements     ? WGAttrNumAccessEnum::Single
            : vector_size * 2 == thread_elements ? WGAttrNumAccessEnum::Double
            : vector_size * 4 == thread_elements ? WGAttrNumAccessEnum::Quad
                                                 : WGAttrNumAccessEnum::Invalid;
#else
        constexpr auto wg_attr_num_access = WGAttrNumAccessEnum::Single;
#endif
        using WarpGemm = WarpGemmDispatcher<typename Problem::ADataType,
                                            typename Problem::BDataType,
                                            typename Problem::CDataType, // AccDataType
                                            WarpTile::at(Base::I0),
                                            WarpTile::at(Base::I1),
                                            WarpTile::at(Base::I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            wg_attr_num_access>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};

// Type aliases for backward compatibility
using GemmPipelineAgBgCrCompTDMWaveSpecializedPolicy = GemmPipelineAgBgCrCompTDMDefaultPolicy<true>;

} // namespace ck_tile
