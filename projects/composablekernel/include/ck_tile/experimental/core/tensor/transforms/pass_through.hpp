// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transforms/pass_through.hpp
 *  @brief TransformImpl specialization for PASS_THROUGH (identity mapping).
 */

#pragma once

#include "ck_tile/experimental/core/tensor/transform_impl.hpp"

namespace ck_tile {

/** @brief Identity mapping: passes a single dimension through unchanged.
 *
 *  Definition:  1 dim --> 1 dim (same value, same length)
 *  Traversal:   1 dim --> 1 dim (identity in both directions)
 *
 *  ndim_upper = 1, ndim_lower = 1
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
    /** @brief Forward: upper to lower (identity). */
    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& /*t*/, index_t* lower, const index_t* upper)
    {
        lower[0] = upper[0];
    }

    /** @brief Reverse: identity (same as forward). */
    static CK_TILE_HOST_DEVICE constexpr void
    reverseMapIndices(const CoordinateTransform& /*t*/, index_t* upper, const index_t* lower)
    {
        upper[0] = lower[0];
    }

    /** @brief Always valid — identity cannot produce out-of-bounds. */
    static CK_TILE_HOST_DEVICE constexpr bool isValidUpper(const CoordinateTransform& /*t*/,
                                                           const index_t* /*upper*/)
    {
        return true;
    }

    /** @brief Upper dimension length (same as lower for identity). */
    static CK_TILE_HOST_DEVICE constexpr index_t upperLength(const CoordinateTransform& t,
                                                             index_t /*i*/)
    {
        return t.lengths[0];
    }
};

} // namespace ck_tile
