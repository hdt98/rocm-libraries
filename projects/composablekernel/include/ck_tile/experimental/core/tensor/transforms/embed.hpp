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
 *  ndim_upper = N, ndim_lower = 1
 *  lengths[0..N-1]      = size of each upper dimension
 *  coefficients[0..N-1] = stride of each upper dimension
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
    /** @brief Forward: N upper dims to 1 lower offset via linear combination. */
    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& t, index_t* lower, const index_t* upper)
    {
        lower[0] = 0;
        for(index_t d = 0; d < t.ndim_upper; ++d)
        {
            lower[0] += upper[d] * t.coefficients[d];
        }
    }

    /** @brief Reverse: decompose offset into N indices via magic division on strides. */
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

    /** @brief Check all upper indices are within [0, length). */
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

    /** @brief Get the length of the i-th upper dimension. */
    static CK_TILE_HOST_DEVICE constexpr index_t upperLength(const CoordinateTransform& t,
                                                             index_t i)
    {
        return t.lengths[i];
    }
};

} // namespace ck_tile
