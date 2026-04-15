// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transforms/pass_through.hpp
 *  @brief TransformImpl specialization for PASS_THROUGH (identity mapping).
 */

#pragma once

#include "ck_tile/core/tensor/value_based/transform_impl.hpp"

namespace ck_tile {

/** @brief Identity mapping: output[0] = input[0].
 *
 *  The simplest transform — passes a single dimension through unchanged.
 *  Always valid (no bounds checking needed).
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
