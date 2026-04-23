// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief TransformImpl specialization for PASS_THROUGH (identity mapping).
// IWYU pragma: private, include "ck_tile/experimental/core/transform/transform_impl.hpp"
#pragma once

#include "ck_tile/experimental/core/transform/transform_impl.hpp"

namespace ck_tile::core::transform::detail {

// Identity: 1 input dim → 1 output dim (same value, same length).
// ndim_input == 1, ndim_output == 1.
template <>
struct TransformImpl<TransformType::PASS_THROUGH>
{
    struct Schema
    {
        index_t dim_length = 0;

        private:
        static_array<uint8_t, MAX_TRANSFORM_DATA_SIZE - sizeof(index_t)> _pad{};
    };
    static_assert(sizeof(Schema) == MAX_TRANSFORM_DATA_SIZE);

    static constexpr Schema readSchema(const CoordinateTransform& t)
    {
        return bit_cast<Schema>(t.data);
    }

    static constexpr void writeSchema(CoordinateTransform& t, const Schema& d)
    {
        t.data = bit_cast<TransformDataBuffer>(d);
    }

    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& /*t*/, index_t* output, const index_t* input)
    {
        output[0] = input[0];
    }

    // XOR is its own inverse — same operation in both directions; PassThrough
    // is even simpler.
    static CK_TILE_HOST_DEVICE constexpr void
    reverseMapIndices(const CoordinateTransform& /*t*/, index_t* input, const index_t* output)
    {
        input[0] = output[0];
    }

    // Identity cannot produce out-of-bounds — always valid.
    static CK_TILE_HOST_DEVICE constexpr bool isValidInput(const CoordinateTransform& /*t*/,
                                                           const index_t* /*input*/)
    {
        return true;
    }

    static CK_TILE_HOST_DEVICE constexpr index_t inputLength(const CoordinateTransform& t,
                                                             index_t /*i*/)
    {
        auto d = readSchema(t);
        return d.dim_length;
    }
};

} // namespace ck_tile::core::transform::detail
