// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/pipeline/sageattn_v3_preprocess_problem.hpp"

namespace ck_tile {

template <typename Problem_>
struct SA3QKPrepPolicy
{
    using Problem = remove_cvref_t<Problem_>;
    using InputT  = typename Problem::InputT;

    static constexpr index_t kRows             = Problem::kRows;
    static constexpr index_t kCols             = Problem::kCols;
    static constexpr index_t kBlockSize        = Problem::kBlockSize;
    static constexpr index_t kScaleGranularity = Problem::kScaleGranularity;
    static constexpr index_t kGroups           = Problem::kGroups;
    static constexpr index_t kG                = Problem::kG;
    static constexpr index_t kLoadVec          = Problem::kLoadVec;
    static constexpr index_t kVecPerG          = Problem::kVecPerG;
    static constexpr index_t kThreadsPerCol    = Problem::kThreadsPerCol;

    // kLdsPad=8 eliminates intra-warp LDS bank conflicts.
    static constexpr index_t kLdsPad       = 8;
    static constexpr index_t kLdsRowStride = kCols + kLdsPad;
    static constexpr index_t kQTileBytes =
        kRows * kLdsRowStride * static_cast<index_t>(sizeof(InputT));

    // smem_mean: float[kCols] appended after Q tile; caches per-column mean for RunQQuantize.
    static constexpr index_t kSmemMeanOffset = kQTileBytes;
    static constexpr index_t kSmemMeanBytes  = kCols * static_cast<index_t>(sizeof(float));

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kSmemMeanOffset + kSmemMeanBytes;
    }

    // K kernel smem: kCols floats for k_mean cache.
    CK_TILE_HOST_DEVICE static constexpr index_t GetKSmemSize()
    {
        return kCols * static_cast<index_t>(sizeof(float));
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeQKTileDstr()
    {
        constexpr index_t kWarps       = kBlockSize / 64;
        constexpr index_t kRowsPerWarp = 64 / kGroups;
        static_assert(kBlockSize % 64 == 0, "kBlockSize must be a multiple of 64 (warp size)");
        static_assert(64 % kGroups == 0, "kGroups must divide 64 (warp size)");
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<kWarps, kRowsPerWarp>, sequence<kGroups, kVecPerG, kLoadVec>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<0>, sequence<1, 0>>,
                sequence<2, 2>,
                sequence<1, 2>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeMeanReduceTileDstr()
    {
        constexpr index_t kWarps_      = kBlockSize / 64;
        constexpr index_t kColsPerWarp = kCols / kWarps_;
        static_assert(kBlockSize % 64 == 0, "kBlockSize must be a multiple of 64");
        static_assert(kCols % kWarps_ == 0, "kCols must be divisible by kWarps");
        static_assert(kColsPerWarp * kThreadsPerCol == 64,
                      "kColsPerWarp * kThreadsPerCol must equal warp size (64)");
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<kThreadsPerCol>,
                                       tuple<sequence<kWarps_, kColsPerWarp, 1>>,
                                       tuple<sequence<1>, sequence<1, 0>>,
                                       tuple<sequence<0>, sequence<1, 0>>,
                                       sequence<1>,
                                       sequence<2>>{});
    }
};

template <typename Problem_>
struct SA3KMeanPolicy
{
    using Problem = remove_cvref_t<Problem_>;

    static constexpr index_t kWarps        = Problem::kWarps;
    static constexpr index_t kVec          = Problem::kVec;
    static constexpr index_t kColsPerWarp  = Problem::kColsPerWarp;
    static constexpr index_t kRowsPerGroup = Problem::kRowsPerGroup;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    CK_TILE_HOST_DEVICE static constexpr auto MakeAccDstr()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<kRowsPerGroup>,
                tuple<sequence<kWarps, kColsPerWarp, kVec>>,
                tuple<sequence<1>, sequence<1, 0>>,
                tuple<sequence<0>, sequence<1, 0>>,
                sequence<1>,
                sequence<2>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeKLoadDstr()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<kRowsPerGroup>, sequence<kWarps, kColsPerWarp, kVec>>,
                tuple<sequence<2>, sequence<2, 1>>,
                tuple<sequence<0>, sequence<1, 0>>,
                sequence<2>,
                sequence<2>>{});
    }
};

template <typename Problem_>
struct SA3VPrepPolicy
{
    using Problem = remove_cvref_t<Problem_>;
    using InputT  = typename Problem::InputT;

    static constexpr index_t kVHdimTile       = Problem::kVHdimTile;
    static constexpr index_t kVGroup          = Problem::kVGroup;
    static constexpr index_t kBlockSize       = Problem::kBlockSize;
    static constexpr index_t kLoadVec         = Problem::kLoadVec;
    static constexpr index_t kVGroupsPerBlock = Problem::kVGroupsPerBlock;

    static constexpr index_t kSmemVBytes =
        kVGroupsPerBlock * kVGroup * kVHdimTile * static_cast<index_t>(sizeof(InputT));

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return kSmemVBytes; }

    CK_TILE_HOST_DEVICE static constexpr auto MakeSingleGroupLoadDstr()
    {
        constexpr index_t kDPacks        = kVHdimTile / kLoadVec;
        constexpr index_t kWarps         = kBlockSize / 64;
        constexpr index_t kRowsPerWarp   = 64 / kDPacks;
        constexpr index_t kRowsPerThread = kVGroup / (kVGroupsPerBlock * kLoadVec);

        static_assert(kBlockSize % 64 == 0, "kBlockSize must be multiple of 64");
        static_assert(64 % kDPacks == 0, "kDPacks must divide 64 for lane decomposition");
        static_assert(kVGroup % (kVGroupsPerBlock * kLoadVec) == 0,
                      "kVGroup must be divisible by R*kLoadVec");
        static_assert(kWarps * kRowsPerWarp * kRowsPerThread == kVGroup,
                      "Distribution must exactly cover kVGroup rows");

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<kWarps, kRowsPerWarp, kRowsPerThread>, sequence<kDPacks, kLoadVec>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<0>, sequence<1, 0>>,
                sequence<2, 1>,
                sequence<1, 2>>{});
    }
};

} // namespace ck_tile
