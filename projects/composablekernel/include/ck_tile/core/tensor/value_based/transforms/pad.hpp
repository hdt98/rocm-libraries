// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transforms/pad.hpp
 *  @brief TransformImpl specialization for PAD (shifted mapping with bounds).
 */

#pragma once

#include "ck_tile/core/tensor/value_based/transform_impl.hpp"

namespace ck_tile {

/** @brief Padding transform: expands a dimension with boundary elements.
 *
 *  Definition:  1 dim (unpadded) --> 1 dim (padded, larger)
 *  Traversal:   1 input --> 1 output   (subtract left_pad to map into
 *                                       the unpadded coordinate space)
 *
 *  ndim_input  = 1, ndim_output = 1
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
 *      Boundary check: input must be in [left_pad, left_pad + length).
 *        isValidInput(1) = false  (in left padding zone)
 *        isValidInput(5) = true   (maps to base index 3)
 *        isValidInput(13) = false (in right padding zone)
 *
 *  skip_bounds_check: Set to true for halo regions where out-of-bounds
 *  access is intentional (the caller handles boundary conditions).
 */
template <>
struct TransformImpl<TransformType::PAD>
{
    /** @brief Map input to output by subtracting left_pad. */
    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& t, index_t* output, const index_t* input)
    {
        output[0] = input[0] - t.coefficients[0]; // subtract left_pad
    }

    /** @brief Check if input is within the valid (non-padded) range.
     *
     *  Valid range: [left_pad, left_pad + unpadded_length)
     *  Returns true unconditionally if skip_bounds_check is set.
     */
    static CK_TILE_HOST_DEVICE constexpr bool isValidInput(const CoordinateTransform& t,
                                                           const index_t* input)
    {
        if(t.skip_bounds_check)
        {
            return true;
        }
        index_t left_pad      = t.coefficients[0];
        index_t padded_length = t.lengths[0] + t.coefficients[0] + t.coefficients[1];
        return input[0] >= left_pad && input[0] < padded_length - t.coefficients[1];
    }

    /** @brief Input dimension length = unpadded + left + right. */
    static CK_TILE_HOST_DEVICE constexpr index_t input_length(const CoordinateTransform& t,
                                                              index_t /*i*/)
    {
        return t.lengths[0] + t.coefficients[0] + t.coefficients[1];
    }
};

} // namespace ck_tile
