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
 *  Traversal:   N inputs --> 1 output        (COMPOSE via multiply-accumulate)
 *
 *  ndim_input  = N   (receives N component values from above)
 *  ndim_output = 1   (sends 1 flat value below)
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
