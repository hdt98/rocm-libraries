// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief TransformImpl specialization for MERGE (decomposition during traversal).
// IWYU pragma: private, include "ck_tile/experimental/core/transform/transform_impl.hpp"
#pragma once

#include "ck_tile/experimental/core/transform/transform_impl.hpp"

namespace ck_tile::core::transform::detail {

// Combine N component dims into 1 user-facing dim.
//   Definition:  N base dims  --> 1 user dim   (flatten / combine)
//   Traversal:   1 input      --> N output     (decompose via magic division)
// ndim_input == 1, ndim_output == N.
template <>
struct TransformImpl<TransformType::MERGE>
{
    // ── Schema ──
    struct Schema
    {
        static_array<index_t, MAX_TENSOR_DIMS> component_lengths{};
        static_array<index_t, MAX_TENSOR_DIMS> strides{};
        static_array<MagicDivConstants, MAX_TENSOR_DIMS> magic_divs{};

        private:
        static_array<uint8_t,
                     MAX_TRANSFORM_DATA_SIZE - sizeof(static_array<index_t, MAX_TENSOR_DIMS>) -
                         sizeof(static_array<index_t, MAX_TENSOR_DIMS>) -
                         sizeof(static_array<MagicDivConstants, MAX_TENSOR_DIMS>)>
            _pad{};
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

    // Decompose 1 input value into N output values via magic division.
    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& t, index_t* output, const index_t* input)
    {
        auto d            = readSchema(t);
        index_t remaining = input[0];

        for(index_t i = 0; i < t.ndim_output - 1; ++i)
        {
            index_t quotient =
                static_cast<index_t>(doMagicDiv(static_cast<uint32_t>(remaining), d.magic_divs[i]));
            output[i] = quotient;
            remaining -= quotient * d.strides[i];
        }
        output[t.ndim_output - 1] = remaining;
    }

    // Reverse: compose N components into 1 flat value (multiply-accumulate).
    static CK_TILE_HOST_DEVICE constexpr void
    reverseMapIndices(const CoordinateTransform& t, index_t* input, const index_t* output)
    {
        auto d   = readSchema(t);
        input[0] = 0;
        for(index_t i = 0; i < t.ndim_output; ++i)
        {
            input[0] += output[i] * d.strides[i];
        }
    }

    // Check the merged input index is within [0, product_of_lengths).
    static CK_TILE_HOST_DEVICE constexpr bool isValidInput(const CoordinateTransform& t,
                                                           const index_t* input)
    {
        auto d          = readSchema(t);
        index_t product = 1;
        for(index_t i = 0; i < t.ndim_output; ++i)
        {
            product *= d.component_lengths[i];
        }
        return input[0] >= 0 && input[0] < product;
    }

    // Upper dimension length = product of all component lengths.
    static CK_TILE_HOST_DEVICE constexpr index_t inputLength(const CoordinateTransform& t,
                                                             index_t /*i*/)
    {
        auto d          = readSchema(t);
        index_t product = 1;
        for(index_t i = 0; i < t.ndim_output; ++i)
        {
            product *= d.component_lengths[i];
        }
        return product;
    }
};

} // namespace ck_tile::core::transform::detail
