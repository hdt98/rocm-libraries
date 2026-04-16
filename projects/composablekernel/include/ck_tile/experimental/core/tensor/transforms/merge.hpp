// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transforms/merge.hpp
 *  @brief TransformImpl specialization for MERGE (decomposition during traversal).
 */

#pragma once

#include "ck_tile/experimental/core/tensor/transform_impl.hpp"

namespace ck_tile {

/** @brief Combine N component dims into 1 user-facing dim.
 *
 *  Definition:  N base dims --> 1 user dim   (flatten / combine)
 *  Traversal:   1 input --> N outputs        (DECOMPOSE via magic division)
 *
 *  ndim_input  = 1   (receives 1 merged value from above)
 *  ndim_output = N   (sends N component values below)
 *  lengths[0..N-1]      = component sizes
 *  coefficients[0..N-2] = divisors (product of subsequent lengths)
 *  magic_divs[0..N-2]   = pre-computed magic division constants
 *
 *  Merge is a CONSTRAINED, INVERTIBLE Embed:
 *    - Like Embed, the definition is a linear combination: flat = sum(c[i]*s[i])
 *    - Unlike Embed, the strides are exactly the prefix products of the
 *      component lengths. This constraint makes it invertible via division.
 *    - During traversal, mapIndices computes the INVERSE (decomposition),
 *      which is why it needs magic division.
 *
 *  Example: Merge with component_lengths = {8, 8}
 *
 *    Definition direction (bottom-up):
 *
 *      Base dims:      dim 0        dim 1          Merged dim:
 *                     (K_div8)     (K_mod8)            (K)
 *                        |            |                 ^
 *                        |            |                 |
 *                        '-------.----'           .-----'
 *                                |                |
 *                          K = i0 * 8 + i1        |  strides = {8, 1}
 *                                |                |  from prefix products
 *                                '================.  of lengths {8, 8}
 *
 *    Traversal direction (top-down) --- what mapIndices computes:
 *
 *      User provides:      K = 19
 *                            |
 *                            v
 *                     .------'------.
 *                     | 19 / 8 = 2  |   magic division
 *                     | 19 % 8 = 3  |   (remainder)
 *                     '----.----.---'
 *                          |    |
 *                          v    v
 *      To base:         K_div8=2  K_mod8=3
 *
 *      mapIndices(19) = {2, 3}
 *
 *  Contrast with Embed: Embed also computes a linear combination, but
 *  it sits at the base and only runs forward (N-->1) during traversal.
 *  Merge sits above the base and must run inverse (1-->N) during
 *  traversal, which is why it needs magic division constants.
 *  Merge is essentially an Embed that also carries its own inverse.
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
