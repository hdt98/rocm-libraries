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
 *  ndim_input = 1, ndim_output = 1
 */
template <>
struct TransformImpl<TransformType::PASS_THROUGH>
{
    // ── Schema ──
    struct Schema
    {
        index_t dim_length = 0;

        private:
        static_array<uint8_t, MAX_TRANSFORM_DATA_SIZE - sizeof(index_t)> _pad{};
    };
    static_assert(sizeof(Schema) == MAX_TRANSFORM_DATA_SIZE);

    // ── Schema conversion ──
    static constexpr Schema readSchema(const CoordinateTransform& t)
    {
        return bit_cast<Schema>(t.data);
    }
    static constexpr void writeSchema(CoordinateTransform& t, const Schema& d)
    {
        t.data = bit_cast<TransformDataBuffer>(d);
    }

    // ── Operations ──

    /** @brief Forward: input to output (identity). */
    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& /*t*/, index_t* output, const index_t* input)
    {
        output[0] = input[0];
    }

    /** @brief Reverse: identity (same as forward). */
    static CK_TILE_HOST_DEVICE constexpr void
    reverseMapIndices(const CoordinateTransform& /*t*/, index_t* input, const index_t* output)
    {
        input[0] = output[0];
    }

    /** @brief Always valid — identity cannot produce out-of-bounds. */
    static CK_TILE_HOST_DEVICE constexpr bool isValidInput(const CoordinateTransform& /*t*/,
                                                           const index_t* /*input*/)
    {
        return true;
    }

    /** @brief Upper dimension length. */
    static CK_TILE_HOST_DEVICE constexpr index_t inputLength(const CoordinateTransform& t,
                                                             index_t /*i*/)
    {
        auto d = readSchema(t);
        return d.dim_length;
    }
};

} // namespace ck_tile
