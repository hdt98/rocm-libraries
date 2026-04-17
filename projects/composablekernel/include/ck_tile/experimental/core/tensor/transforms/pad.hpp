// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transforms/pad.hpp
 *  @brief TransformImpl specialization for PAD (shifted mapping with bounds).
 */

#pragma once

#include "ck_tile/experimental/core/tensor/transform_impl.hpp"

namespace ck_tile {

/** @brief Padding transform: expands a dimension with boundary elements.
 *
 *  Definition:  1 dim (unpadded) --> 1 dim (padded, larger)
 *  Traversal:   1 upper --> 1 lower    (subtract left_pad to map padded
 *                                       index into the unpadded space;
 *                                       right_pad affects bounds only)
 *
 *  ndim_upper  = 1, ndim_lower = 1
 *  lengths[0]      = unpadded dimension length
 *  coefficients[0] = left_pad amount
 *  coefficients[1] = right_pad amount
 *  skip_bounds_check = true to skip validation (e.g., halo regions)
 *
 *  Example: Pad with unpadded_length=10, left_pad=2, right_pad=4
 *
 *    Definition direction (bottom-up):
 *
 *      Base dim (10 elements):      [0  1  2  3  4  5  6  7  8  9]
 *                                      |
 *                                      v
 *      User dim (16 elements):  [P  P  0  1  2  3  4  5  6  7  8  9  P  P  P  P]
 *                                ^  ^                                  ^  ^  ^  ^
 *                               left_pad = 2                          right_pad = 4
 *
 *      The user sees 16 elements. Indices [0,1] and [12,13,14,15]
 *      are padding. Indices [2..11] map to base elements [0..9].
 *
 *    Traversal direction (top-down) --- what mapIndices computes:
 *
 *      User provides:      padded_idx = 5
 *                               |
 *                               v
 *                        .------'------.
 *                        | 5 - 2 = 3   |   subtract left_pad
 *                        '-------.-----'
 *                                |
 *                                v
 *      To base:           unpadded_idx = 3
 *
 *      mapIndices(5) = 3
 *
 *      Boundary check: index must be in [left_pad, left_pad + length).
 *      Rejects both left and right padding zones.
 *        isValidUpper(1) = false  (in left padding zone)
 *        isValidUpper(5) = true   (maps to base index 3)
 *        isValidUpper(13) = false (in right padding zone)
 *
 *  skip_bounds_check: Set to true for halo regions where out-of-bounds
 *  access is intentional (the caller handles boundary conditions).
 */
template <>
struct TransformImpl<TransformType::PAD>
{
    /** @brief Forward: upper to lower by subtracting left_pad.
     *
     *  lower = upper - left_pad. The right_pad does not affect the
     *  index mapping itself --- it only affects bounds checking in
     *  isValidUpper() and the padded length in upperLength().
     */
    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& t, index_t* lower, const index_t* upper)
    {
        lower[0] = upper[0] - t.coefficients[0]; // subtract left_pad
    }

    /** @brief Reverse: recover padded index by adding left_pad back.
     *
     *  upper = lower + left_pad. Inverse of mapIndices.
     */
    static CK_TILE_HOST_DEVICE constexpr void
    reverseMapIndices(const CoordinateTransform& t, index_t* upper, const index_t* lower)
    {
        upper[0] = lower[0] + t.coefficients[0]; // add left_pad
    }

    /** @brief Check if upper index is within the valid (non-padded) range.
     *
     *  The padded dimension has three zones:
     *    [0, left_pad)                         --> left padding (invalid)
     *    [left_pad, left_pad + unpadded_length) --> valid (maps to base)
     *    [left_pad + unpadded_length, total)    --> right padding (invalid)
     *
     *  Returns false if the index falls in either padding zone.
     *  Returns true unconditionally if skip_bounds_check is set.
     */
    static CK_TILE_HOST_DEVICE constexpr bool isValidUpper(const CoordinateTransform& t,
                                                           const index_t* upper)
    {
        if(t.skip_bounds_check)
        {
            return true;
        }
        index_t left_pad      = t.coefficients[0];
        index_t padded_length = t.lengths[0] + t.coefficients[0] + t.coefficients[1];
        return upper[0] >= left_pad && upper[0] < padded_length - t.coefficients[1];
    }

    /** @brief Upper dimension length = unpadded + left + right. */
    static CK_TILE_HOST_DEVICE constexpr index_t upperLength(const CoordinateTransform& t,
                                                             index_t /*i*/)
    {
        return t.lengths[0] + t.coefficients[0] + t.coefficients[1];
    }
};

} // namespace ck_tile
