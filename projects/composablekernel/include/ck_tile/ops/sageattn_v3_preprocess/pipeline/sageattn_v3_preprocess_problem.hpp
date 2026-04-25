// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename InputT_, index_t kRows_, index_t kCols_>
struct SA3QKPrepProblem
{
    using InputT = InputT_;

    static constexpr index_t kRows             = kRows_;
    static constexpr index_t kCols             = kCols_;
    static constexpr index_t kScaleGranularity = 32;
    static constexpr index_t kBlockSize        = kRows * (kCols / kScaleGranularity);

    static_assert(kCols % kScaleGranularity == 0,
                  "kCols must be divisible by kScaleGranularity for MXFP4");
    static_assert(kBlockSize == kRows * (kCols / kScaleGranularity),
                  "kBlockSize must equal kRows * kGroups for full QK tile coverage");

    static constexpr index_t kGroups       = kCols / kScaleGranularity;
    static constexpr index_t kG            = kScaleGranularity;
    static constexpr index_t kLoadVec      = 16 / static_cast<index_t>(sizeof(InputT));
    static constexpr index_t kVecPerG      = kG / kLoadVec;
    static constexpr index_t kThreadsPerCol = kBlockSize / kCols;

    static_assert(kG % kLoadVec == 0, "kG must be divisible by kLoadVec (128-bit load unit)");
    static_assert(kBlockSize % kCols == 0,
                  "kBlockSize must be divisible by kCols for parallel mean reduction");
};

template <typename InputT_, index_t kCols_>
struct SA3KMeanProblem
{
    using InputT = InputT_;

    static constexpr index_t kCols         = kCols_;
    static constexpr index_t kWarps        = 4;
    static constexpr index_t kBlockSize    = kWarps * 64;
    static constexpr index_t kVec          = 16 / static_cast<index_t>(sizeof(InputT));
    static constexpr index_t kColGroups    = kCols / kVec;
    static constexpr index_t kColsPerWarp  = kColGroups / kWarps;
    static constexpr index_t kRowsPerGroup = 64 / kColsPerWarp;
    static constexpr index_t kChunkRows    = 256;

    static_assert(kCols % kVec == 0, "kCols must be divisible by kVec (16/sizeof(InputT))");
    static_assert(kColGroups % kWarps == 0, "kColGroups must be divisible by kWarps=4");
    static_assert((kRowsPerGroup & (kRowsPerGroup - 1)) == 0,
                  "kRowsPerGroup must be a power of 2 for XOR butterfly reduce");
};

template <typename InputT_, index_t kVHdimTile_>
struct SA3VPrepProblem
{
    using InputT = InputT_;

    static constexpr index_t kVHdimTile       = kVHdimTile_;
    static constexpr index_t kVGroup          = 32;
    static constexpr index_t kScaleGranularity = 32;
    static constexpr index_t kLoadVec         = 16 / static_cast<index_t>(sizeof(InputT));
    static constexpr index_t kVGroupR =
        (static_cast<index_t>(sizeof(float)) / static_cast<index_t>(sizeof(InputT))) * 2;
    static constexpr index_t kVGroupsPerBlock = kVGroupR;
    static constexpr index_t kBlockSize       = kVGroupsPerBlock * kVHdimTile;

    static_assert(kVGroup == kScaleGranularity, "kVGroup must equal kScaleGranularity (32)");
    static_assert(kVHdimTile % kLoadVec == 0, "kVHdimTile must be divisible by kLoadVec");
    static_assert(kBlockSize <= 1024, "kBlockSize must not exceed 1024");
};

} // namespace ck_tile
