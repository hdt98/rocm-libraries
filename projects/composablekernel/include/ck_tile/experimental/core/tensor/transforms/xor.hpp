// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transforms/xor.hpp
 *  @brief TransformImpl specialization for XOR (2D swizzle for LDS bank conflicts).
 */

#pragma once

#include "ck_tile/experimental/core/tensor/transform_impl.hpp"

namespace ck_tile {

/** @brief 2D XOR permutation for LDS bank conflict avoidance.
 *
 *  Swizzles the second dimension by XORing it with the first dimension
 *  (modulo the second dimension's length). This creates a diagonal access
 *  pattern that avoids LDS bank conflicts.
 *
 *  Definition:  2 dims --> 2 dims
 *  Traversal:   2 input --> 2 output
 *
 *  output[0] = input[0]
 *  output[1] = input[1] ^ (input[0] % length_1)
 *
 *  XOR is its own inverse — applying it twice recovers the original indices.
 *
 *  ndim_input = 2, ndim_output = 2
 */
template <>
struct TransformImpl<TransformType::XOR>
{
    // ── Schema ──
    struct Schema
    {
        index_t length_0 = 0;
        index_t length_1 = 0;

        private:
        static_array<uint8_t, MAX_TRANSFORM_DATA_SIZE - 2 * sizeof(index_t)> _pad{};
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

    /** @brief Forward: swizzle dim 1 by XORing with dim 0 (mod length_1). */
    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& t, index_t* output, const index_t* input)
    {
        auto d    = readSchema(t);
        output[0] = input[0];
        output[1] = input[1] ^ (input[0] % d.length_1);
    }

    /** @brief Reverse: XOR is self-inverse — same operation undoes itself. */
    static CK_TILE_HOST_DEVICE constexpr void
    reverseMapIndices(const CoordinateTransform& t, index_t* input, const index_t* output)
    {
        auto d   = readSchema(t);
        input[0] = output[0];
        input[1] = output[1] ^ (output[0] % d.length_1);
    }

    /** @brief Always valid — XOR preserves index range for power-of-2 length_1
     *  (the typical LDS use case). For non-power-of-2, results may exceed bounds. */
    static CK_TILE_HOST_DEVICE constexpr bool isValidInput(const CoordinateTransform& /*t*/,
                                                           const index_t* /*input*/)
    {
        return true;
    }

    /** @brief Get the length of the i-th input dimension. */
    static CK_TILE_HOST_DEVICE constexpr index_t inputLength(const CoordinateTransform& t,
                                                             index_t i)
    {
        auto d = readSchema(t);
        return (i == 0) ? d.length_0 : d.length_1;
    }
};

} // namespace ck_tile
