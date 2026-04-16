// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transforms/pass_through.hpp
 *  @brief TransformImpl specialization for PASS_THROUGH (identity mapping).
 */

#pragma once

#include "ck_tile/core/tensor/value_based/transform_impl.hpp"

namespace ck_tile {

/** @brief Identity mapping: passes a single dimension through unchanged.
 *
 *  Definition:  1 dim --> 1 dim (same value, same length)
 *  Traversal:   1 dim --> 1 dim (identity in both directions)
 *
 *  ndim_input = 1, ndim_output = 1
 *  lengths[0] = dimension length
 *
 *  PassThrough is its own inverse --- the same operation in both
 *  the definition and traversal directions.
 *
 *  Example: PassThrough(M=128)
 *
 *    User dim (M)              Base dim (M)
 *        |                         ^
 *        v    definition (up)      |
 *       idx ===================> idx        (same value)
 *        |    traversal (down)     ^
 *        '========================'
 *
 *    mapIndices(5) = 5
 *
 *  Used when a dimension needs to pass through a transform layer
 *  without modification --- e.g., the M dimension in a GEMM LDS
 *  descriptor where only K is being reshaped.
 */
template <>
struct TransformImpl<TransformType::PASS_THROUGH>
{
    /** @brief Map input to output (identity). */
    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& /*t*/, index_t* output, const index_t* input)
    {
        output[0] = input[0];
    }

    /** @brief Always valid — identity cannot produce out-of-bounds. */
    static CK_TILE_HOST_DEVICE constexpr bool isValidInput(const CoordinateTransform& /*t*/,
                                                           const index_t* /*input*/)
    {
        return true;
    }

    /** @brief Input dimension length (same as output for identity). */
    static CK_TILE_HOST_DEVICE constexpr index_t input_length(const CoordinateTransform& t,
                                                              index_t /*i*/)
    {
        return t.lengths[0];
    }
};

} // namespace ck_tile
