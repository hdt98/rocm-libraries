// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transforms/unmerge.hpp
 *  @brief TransformImpl specialization for UNMERGE (composition during traversal).
 */

#pragma once

#include "ck_tile/experimental/core/tensor/transform_impl.hpp"

namespace ck_tile {

/** @brief Split 1 base dim into N user-facing component dims.
 *
 *  Definition:  1 base dim --> N user dims   (split / expand / unflatten)
 *  Traversal:   N upper --> 1 lower           (COMPOSE via multiply-accumulate)
 *
 *  ndim_upper  = N   (receives N component values from above)
 *  ndim_lower = 1   (sends 1 flat value below)
 *  lengths[0..N-1]      = component sizes
 *  coefficients[0..N-1] = strides (prefix products of subsequent lengths)
 *
 *  Unmerge is the INVERSE of Merge:
 *    - Merge definition: N --> 1 (flatten). Merge traversal: 1 --> N (decompose).
 *    - Unmerge definition: 1 --> N (split). Unmerge traversal: N --> 1 (compose).
 *    - Unmerge's traversal (compose) does the SAME MATH as Merge's definition.
 *    - Merge's traversal (decompose) does the SAME MATH as Unmerge's definition.
 *
 *  Example: Unmerge with component_lengths = {4, 8}
 *
 *    Definition direction (bottom-up):
 *
 *      Base dim:        flat index          User dims:    dim 0     dim 1
 *                          (32)                         (rows=4)  (cols=8)
 *                           |                              ^         ^
 *                           |                              |         |
 *                     .-----'-----.                  .-----'----.----'
 *                     | 19 / 8 = 2|   division       |          |
 *                     | 19 % 8 = 3|   (split)        |          |
 *                     '----.------'                   |          |
 *                          '==========================.==========.
 *
 *    Traversal direction (top-down) --- what mapIndices computes:
 *
 *      User provides:     row = 2       col = 3
 *                           |             |
 *                           v             v
 *                     .-----'-----'-------.
 *                     | 2 * 8  +  3 * 1   |   multiply-accumulate
 *                     | = 16   +  3       |   strides = {8, 1}
 *                     | = 19              |
 *                     '---------.--------.'
 *                               |
 *                               v
 *      To base:            flat = 19
 *
 *      mapIndices({2, 3}) = 19
 *
 *  Note: Unmerge's traversal is a simple multiply-accumulate (no magic
 *  division needed), because composing components into a flat index
 *  is cheap. Magic division is only needed in Merge's traversal, where
 *  a flat index must be decomposed back into components.
 */
template <>
struct TransformImpl<TransformType::UNMERGE>
{
    /** @brief Compose N upper values into 1 lower value via multiply-accumulate.
     *
     *  Uses pre-computed strides (in coefficients[]) to flatten the
     *  multi-dimensional input into a single flat index.
     */
    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& t, index_t* lower, const index_t* upper)
    {
        lower[0] = 0;
        for(index_t d = 0; d < t.ndim_upper; ++d)
        {
            lower[0] += upper[d] * t.coefficients[d];
        }
    }

    /** @brief Reverse: decompose 1 flat value into N components (magic division). */
    static CK_TILE_HOST_DEVICE constexpr void
    reverseMapIndices(const CoordinateTransform& t, index_t* upper, const index_t* lower)
    {
        index_t remaining = lower[0];
        for(index_t d = 0; d < t.ndim_upper - 1; ++d)
        {
            index_t quotient =
                static_cast<index_t>(doMagicDiv(static_cast<uint32_t>(remaining), t.magic_divs[d]));
            upper[d] = quotient;
            remaining -= quotient * t.coefficients[d];
        }
        upper[t.ndim_upper - 1] = remaining;
    }

    /** @brief Check all upper component indices are within [0, length). */
    static CK_TILE_HOST_DEVICE constexpr bool isValidUpper(const CoordinateTransform& t,
                                                           const index_t* upper)
    {
        for(index_t d = 0; d < t.ndim_upper; ++d)
        {
            if(upper[d] < 0 || upper[d] >= t.lengths[d])
            {
                return false;
            }
        }
        return true;
    }

    /** @brief Get the length of the i-th upper (component) dimension. */
    static CK_TILE_HOST_DEVICE constexpr index_t upperLength(const CoordinateTransform& t,
                                                             index_t i)
    {
        return t.lengths[i];
    }
};

} // namespace ck_tile
