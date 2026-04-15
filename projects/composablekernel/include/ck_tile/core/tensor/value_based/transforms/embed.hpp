// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transforms/embed.hpp
 *  @brief TransformImpl specialization for EMBED (linear combination).
 */

#pragma once

#include "ck_tile/core/tensor/value_based/transform_impl.hpp"

namespace ck_tile {

/** @brief Linear combination: output[0] = sum(input[i] * coefficient[i]).
 *
 *  Maps N input dimensions to 1 output dimension via a weighted sum.
 *  Typically used to compute a linear memory offset from multi-dimensional
 *  indices with strides.
 *
 *  - lengths[0..N-1] = input dimension sizes
 *  - coefficients[0..N-1] = strides (weights for the linear combination)
 *  - ndim_input = N, ndim_output = 1
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
