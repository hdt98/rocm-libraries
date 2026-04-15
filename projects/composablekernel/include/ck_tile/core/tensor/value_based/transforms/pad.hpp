// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transforms/pad.hpp
 *  @brief TransformImpl specialization for PAD (shifted mapping with bounds).
 */

#pragma once

#include "ck_tile/core/tensor/value_based/transform_impl.hpp"

namespace ck_tile {

/** @brief Shifted mapping with optional bounds checking.
 *
 *  Maps output = input - left_pad. The padded dimension has length
 *  = original_length + left_pad + right_pad.
 *
 *  Covers left-only (right_pad=0), right-only (left_pad=0), and
 *  both-sides padding in a single transform type.
 *
 *  - lengths[0] = unpadded dimension length
 *  - coefficients[0] = left_pad amount
 *  - coefficients[1] = right_pad amount
 *  - skip_bounds_check = true to skip validation (e.g., halo regions)
 *  - ndim_input = 1, ndim_output = 1
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
