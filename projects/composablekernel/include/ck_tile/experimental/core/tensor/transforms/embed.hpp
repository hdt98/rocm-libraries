// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transforms/embed.hpp
 *  @brief TransformImpl specialization for EMBED (linear combination).
 */

#pragma once

#include "ck_tile/experimental/core/tensor/transform_impl.hpp"

namespace ck_tile {

/** @brief Linear combination with strides. Computes a 1D memory offset
 *  from N-dimensional indices.
 *
 *  Definition:  N dims --> 1 dim   (weighted sum with strides)
 *  Traversal:   N dims --> 1 dim   (same --- Embed is always at the base)
 *
 *  ndim_input = N, ndim_output = 1
 *  lengths[0..N-1]      = size of each input dimension
 *  coefficients[0..N-1] = stride of each input dimension
 *
 *  Embed always sits at the BASE of the transform stack. It defines
 *  the physical memory layout. During traversal it runs FORWARD
 *  (N dims --> 1 offset). The memory offset is the end of the line,
 *  so no transform ever needs to undo an Embed.
 *
 *  Compare with Merge: Merge also does a linear combination in its
 *  definition direction, but during traversal it must run in REVERSE
 *  (1 --> N, decomposing via magic division). Merge is essentially
 *  an Embed that also carries its own inverse. Embed does not need
 *  an inverse because it only ever runs forward.
 *
 *  Example: Embed with lengths={8, 128, 8}, strides={1032, 8, 1}
 *
 *    Base dims:     dim 0       dim 1      dim 2
 *                  (K_div8)      (M)      (K_mod8)
 *                    |           |           |
 *                    v           v           v
 *                .---'----------'-----------.
 *                |                           |
 *                | offset = 2*1032 + 5*8 + 3 |
 *                |        = 2064 + 40 + 3    |
 *                |        = 2107             |
 *                |                           |
 *                '-----------.---------------'
 *                            |
 *                            v
 *                    Memory offset (1D)
 *
 *    mapIndices({2, 5, 3}) = 2107
 *
 *    The stride 1032 = (128+1)*8 includes 1 element of padding per
 *    row for LDS bank conflict avoidance. These are arbitrary strides
 *    that do not need to be prefix products of the lengths.
 */
template <>
struct TransformImpl<TransformType::EMBED>
{
    /** @brief Compute output = sum(input[i] * coefficient[i]). */
    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& t, index_t* output, const index_t* input)
    {
        output[0] = 0;
        for(index_t d = 0; d < t.ndim_input; ++d)
        {
            output[0] += input[d] * t.coefficients[d];
        }
    }

    /** @brief Check all input indices are within [0, length). */
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

    /** @brief Get the length of the i-th input dimension. */
    static CK_TILE_HOST_DEVICE constexpr index_t input_length(const CoordinateTransform& t,
                                                              index_t i)
    {
        return t.lengths[i];
    }
};

} // namespace ck_tile
