// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief TransformImpl specialization for UNMERGE (composition during traversal).
// IWYU pragma: private, include "ck_tile/experimental/core/transform/transform_impl.hpp"
#pragma once

#include "ck_tile/experimental/core/transform/transform_impl.hpp"

namespace ck_tile::core::transform::detail {

// Split 1 base dim into N user-facing component dims.
//   Definition:  1 base dim   --> N user dims   (split / expand / unflatten)
//   Traversal:   N input      --> 1 output      (compose via multiply-accumulate)
// ndim_input == N, ndim_output == 1.
template <>
struct TransformImpl<TransformType::UNMERGE>
{
    // ── Schema ──
    // Same layout as TransformImpl<MERGE>::Schema — they are inverses using the same data.
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

    // Compose N input values into 1 output value via multiply-accumulate.
    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& t, index_t* output, const index_t* input)
    {
        auto d    = readSchema(t);
        output[0] = 0;
        for(index_t i = 0; i < t.ndim_input; ++i)
        {
            output[0] += input[i] * d.strides[i];
        }
    }

    // Reverse: decompose 1 flat value into N components (magic division).
    static CK_TILE_HOST_DEVICE constexpr void
    reverseMapIndices(const CoordinateTransform& t, index_t* input, const index_t* output)
    {
        auto d            = readSchema(t);
        index_t remaining = output[0];
        for(index_t i = 0; i < t.ndim_input - 1; ++i)
        {
            index_t quotient =
                static_cast<index_t>(doMagicDiv(static_cast<uint32_t>(remaining), d.magic_divs[i]));
            input[i] = quotient;
            remaining -= quotient * d.strides[i];
        }
        input[t.ndim_input - 1] = remaining;
    }

    // Check all input component indices are within [0, length).
    static CK_TILE_HOST_DEVICE constexpr bool isValidInput(const CoordinateTransform& t,
                                                           const index_t* input)
    {
        auto d = readSchema(t);
        for(index_t i = 0; i < t.ndim_input; ++i)
        {
            if(input[i] < 0 || input[i] >= d.component_lengths[i])
                return false;
        }
        return true;
    }

    // Get the length of the i-th input (component) dimension.
    static CK_TILE_HOST_DEVICE constexpr index_t inputLength(const CoordinateTransform& t,
                                                             index_t i)
    {
        auto d = readSchema(t);
        return d.component_lengths[i];
    }
};

} // namespace ck_tile::core::transform::detail
