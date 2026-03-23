// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Pack kScaleGranularity floats -> kScaleGranularity/2 packed FP4 bytes via gfx950 intrinsic.
// The intrinsic is an in-out accumulator: slot 0 initializes `x`, later slots build on it.
// The uninitialized warning is intentionally suppressed; slot 0 initializes before any read.
template <index_t kScaleGranularity>
CK_TILE_DEVICE void
PackFP4Group(const float* __restrict__ group_data, uint8_t* __restrict__ hat_group, float scale)
{
    static_assert(kScaleGranularity % 8 == 0,
                  "kScaleGranularity must be a multiple of 8 for FP4 packing");

    for(index_t j = 0; j < kScaleGranularity / 8; j++)
    {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
        uint32_t x;
        x = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
            x, group_data[8 * j + 0], group_data[8 * j + 1], scale, 0);
        x = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
            x, group_data[8 * j + 2], group_data[8 * j + 3], scale, 1);
        x = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
            x, group_data[8 * j + 4], group_data[8 * j + 5], scale, 2);
        x = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
            x, group_data[8 * j + 6], group_data[8 * j + 7], scale, 3);
#pragma clang diagnostic pop
        *reinterpret_cast<uint32_t*>(hat_group + j * 4) = x;
    }
}

} // namespace ck_tile
