// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename InputT_, index_t kQMeanGroupSize_, index_t kHeadDim_>
struct SA3QKPrepProblem
{
    using InputT = InputT_;

    static constexpr index_t kQMeanGroupSize             = kQMeanGroupSize_;
    static constexpr index_t kHeadDim             = kHeadDim_;
    static constexpr index_t kScaleGranularity = 32;
    static constexpr index_t kBlockSize        = kQMeanGroupSize * (kHeadDim / kScaleGranularity);

    static_assert(kHeadDim % kScaleGranularity == 0,
                  "kHeadDim must be divisible by kScaleGranularity for MXFP4");
    static_assert(kBlockSize == kQMeanGroupSize * (kHeadDim / kScaleGranularity),
                  "kBlockSize must equal kQMeanGroupSize * kGroups for full QK tile coverage");

    static constexpr index_t kGroups       = kHeadDim / kScaleGranularity;
    static constexpr index_t kG            = kScaleGranularity;
    static constexpr index_t kLoadVec      = 16 / static_cast<index_t>(sizeof(InputT));
    static constexpr index_t kVecPerG      = kG / kLoadVec;
    static constexpr index_t kThreadsPerCol = kBlockSize / kHeadDim;

    static_assert(kG % kLoadVec == 0, "kG must be divisible by kLoadVec (128-bit load unit)");
    static_assert(kBlockSize % kHeadDim == 0,
                  "kBlockSize must be divisible by kHeadDim for parallel mean reduction");
};

template <typename InputT_, index_t kHeadDim_>
struct SA3KMeanProblem
{
    using InputT = InputT_;

    static constexpr index_t kHeadDim         = kHeadDim_;
    static constexpr index_t kWarps        = 4;
    static constexpr index_t kBlockSize    = kWarps * 64;
    static constexpr index_t kVec          = 16 / static_cast<index_t>(sizeof(InputT));
    static constexpr index_t kColGroups    = kHeadDim / kVec;
    static constexpr index_t kColsPerWarp  = kColGroups / kWarps;
    static constexpr index_t kRowsPerGroup = 64 / kColsPerWarp;
    static constexpr index_t kChunkRows    = 256;

    static_assert(kHeadDim % kVec == 0, "kHeadDim must be divisible by kVec (16/sizeof(InputT))");
    static_assert(kColGroups % kWarps == 0, "kColGroups must be divisible by kWarps=4");
    static_assert((kRowsPerGroup & (kRowsPerGroup - 1)) == 0,
                  "kRowsPerGroup must be a power of 2 for XOR butterfly reduce");
};

template <typename InputT_, index_t kVHdimTile_>
struct SA3VPrepProblem
{
    using InputT = InputT_;

    static constexpr index_t kVHdimTile           = kVHdimTile_;
    static constexpr index_t kScaleGranularity   = 32;
    static constexpr index_t kFmhaTileN0         = 128;
    static constexpr index_t kScaleGroupsPerTile = kFmhaTileN0 / kScaleGranularity;
    static constexpr index_t kLoadVec            = 16 / static_cast<index_t>(sizeof(InputT));
    static constexpr index_t kBlockSize          = kScaleGroupsPerTile * kVHdimTile;

    static_assert(kFmhaTileN0 % kScaleGranularity == 0,
                  "kFmhaTileN0 must be divisible by kScaleGranularity");
    static_assert(kVHdimTile % kLoadVec == 0, "kVHdimTile must be divisible by kLoadVec");
    static_assert(kBlockSize <= 1024, "kBlockSize must not exceed 1024");
};

} // namespace ck_tile
