// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transforms/unmerge.hpp
 *  @brief TransformImpl specialization for UNMERGE (composition during traversal).
 */

#pragma once

#include "ck_tile/core/tensor/value_based/transform_impl.hpp"

namespace ck_tile {

/** @brief Split 1 memory dim into N user-facing component dims.
 *
 *  During graph traversal (user -> memory), the unmerge receives N input
 *  values (the components from the user) and COMPOSES them into 1 output
 *  value (the flat index toward memory) using multiply-accumulate.
 *
 *  This matches v1's unmerge::calculate_lower_index which takes N upper
 *  values and produces 1 lower value.
 *
 *  - ndim_input = N (user side: the N component values)
 *  - ndim_output = 1 (memory side: the single flat value)
 *  - lengths[0..N-1] = component sizes
 *  - coefficients[0..N-1] = strides (prefix products of subsequent lengths)
 *
 *  Example: component_lengths = {4, 8}
 *    strides (coefficients) = {8, 1}
 *    input = {2, 3} -> output = 2*8 + 3*1 = 19
 */
template <>
struct TransformImpl<TransformType::UNMERGE>
{
    /** @brief Compose N inputs into 1 output via multiply-accumulate.
     *
     *  Uses pre-computed strides (in coefficients[]) to flatten the
     *  multi-dimensional input into a single flat index.
     */
    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& t, index_t* output, const index_t* input)
    {
        output[0] = 0;
        for(index_t d = 0; d < t.ndim_input; ++d)
        {
            output[0] += input[d] * t.coefficients[d];
        }
    }

    /** @brief Check all component input indices are within [0, length). */
    static CK_TILE_HOST_DEVICE constexpr bool isValidInput(const CoordinateTransform& t,
                                                           const index_t* input)
    {
        for(index_t d = 0; d < t.ndim_input; ++d)
        {
            if(input[d] < 0 || input[d] >= t.lengths[d])
            {
                return false;
            }
        }
        return true;
    }

    /** @brief Get the length of the i-th input (component) dimension. */
    static CK_TILE_HOST_DEVICE constexpr index_t input_length(const CoordinateTransform& t,
                                                              index_t i)
    {
        return t.lengths[i];
    }
};

} // namespace ck_tile
