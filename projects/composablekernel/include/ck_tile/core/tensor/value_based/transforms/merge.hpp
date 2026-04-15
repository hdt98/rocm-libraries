// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transforms/merge.hpp
 *  @brief TransformImpl specialization for MERGE (decomposition during traversal).
 */

#pragma once

#include "ck_tile/core/tensor/value_based/transform_impl.hpp"

namespace ck_tile {

/** @brief Merge N component dims into 1 user-facing dim.
 *
 *  During graph traversal (user -> memory), the merge receives 1 input
 *  value (the merged/flattened index from the user) and DECOMPOSES it
 *  into N output values (the components toward memory) using pre-computed
 *  magic division constants.
 *
 *  This matches v1's merge_v2_magic_division::calculate_lower_index which
 *  takes 1 upper value and produces N lower values.
 *
 *  - ndim_input = 1 (user side: the single merged value)
 *  - ndim_output = N (memory side: the N components)
 *  - lengths[0..N-1] = component sizes
 *  - coefficients[0..N-2] = divisors (product of subsequent lengths)
 *  - magic_divs[0..N-2] = pre-computed magic division constants
 *
 *  Example: component_lengths = {4, 8}
 *    input = 19 -> output = {2, 3} (19 / 8 = 2 remainder 3)
 */
template <>
struct TransformImpl<TransformType::MERGE>
{
    /** @brief Decompose 1 input into N outputs via magic division.
     *
     *  Iterates from most significant component (d=0) to least significant.
     *  Uses pre-computed magic_divs for quotients and coefficients (divisors)
     *  for remainder computation.
     */
    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& t, index_t* output, const index_t* input)
    {
        index_t remaining = input[0];

        for(index_t d = 0; d < t.ndim_output - 1; ++d)
        {
            index_t quotient = static_cast<index_t>(
                do_magic_div(static_cast<uint32_t>(remaining), t.magic_divs[d]));
            output[d] = quotient;
            remaining -= quotient * t.coefficients[d]; // coefficients[d] = divisor
        }

        output[t.ndim_output - 1] = remaining;
    }

    /** @brief Check the merged input index is within [0, product_of_lengths). */
    static CK_TILE_HOST_DEVICE constexpr bool isValidInput(const CoordinateTransform& t,
                                                           const index_t* input)
    {
        index_t product = 1;
        for(index_t d = 0; d < t.ndim_output; ++d)
        {
            product *= t.lengths[d];
        }
        return input[0] >= 0 && input[0] < product;
    }

    /** @brief Input dimension length = product of all component lengths. */
    static CK_TILE_HOST_DEVICE constexpr index_t input_length(const CoordinateTransform& t,
                                                              index_t /*i*/)
    {
        index_t product = 1;
        for(index_t d = 0; d < t.ndim_output; ++d)
        {
            product *= t.lengths[d];
        }
        return product;
    }
};

} // namespace ck_tile
