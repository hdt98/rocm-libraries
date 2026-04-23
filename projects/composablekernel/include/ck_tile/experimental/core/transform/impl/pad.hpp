// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief TransformImpl specialization for PAD (shifted mapping with bounds).
// IWYU pragma: private, include "ck_tile/experimental/core/transform/transform_impl.hpp"
#pragma once

#include "ck_tile/experimental/core/transform/transform_impl.hpp"

namespace ck_tile::core::transform::detail {

// Padding: expands a dimension with boundary elements.
//   Definition:  1 dim (unpadded) --> 1 dim (padded, larger)
//   Traversal:   1 input          --> 1 output  (subtract left_pad to map
//                                                padded index into the
//                                                unpadded space; right_pad
//                                                affects bounds only)
// ndim_input == 1, ndim_output == 1.
template <>
struct TransformImpl<TransformType::PAD>
{
    // ── Schema ──
    struct Schema
    {
        index_t dim_length = 0;
        index_t left_pad   = 0;
        index_t right_pad  = 0;

        private:
        static_array<uint8_t, MAX_TRANSFORM_DATA_SIZE - 3 * sizeof(index_t)> _pad{};
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

    // Forward: output = input - left_pad. right_pad does not affect the
    // index mapping itself; it only affects bounds checking in isValidInput()
    // and the padded length in inputLength().
    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& t, index_t* output, const index_t* input)
    {
        auto d    = readSchema(t);
        output[0] = input[0] - d.left_pad;
    }

    // Reverse: recover padded index by adding left_pad back.
    static CK_TILE_HOST_DEVICE constexpr void
    reverseMapIndices(const CoordinateTransform& t, index_t* input, const index_t* output)
    {
        auto d   = readSchema(t);
        input[0] = output[0] + d.left_pad;
    }

    // Padded dimension has three zones:
    //   [0, left_pad)                       --> left padding (invalid)
    //   [left_pad, left_pad + dim_length)   --> valid (maps to base)
    //   [left_pad + dim_length, total)      --> right padding (invalid)
    // Returns false if the index falls in either padding zone; returns true
    // unconditionally if skip_bounds_check is set.
    static CK_TILE_HOST_DEVICE constexpr bool isValidInput(const CoordinateTransform& t,
                                                           const index_t* input)
    {
        if(t.skip_bounds_check)
        {
            return true;
        }
        auto d = readSchema(t);
        return input[0] >= d.left_pad && input[0] < d.left_pad + d.dim_length;
    }

    // Upper dimension length = dim_length + left_pad + right_pad.
    static CK_TILE_HOST_DEVICE constexpr index_t inputLength(const CoordinateTransform& t,
                                                             index_t /*i*/)
    {
        auto d = readSchema(t);
        return d.dim_length + d.left_pad + d.right_pad;
    }
};

} // namespace ck_tile::core::transform::detail
