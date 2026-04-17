// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transforms/embed.hpp
 *  @brief TransformImpl specialization for EMBED (linear combination with strides).
 */

#pragma once

#include "ck_tile/experimental/core/tensor/transform_impl.hpp"

namespace ck_tile {

/** @brief Linear combination with strides. Computes a 1D memory offset
 *  from N-dimensional indices. Sits at the BASE of the transform stack.
 *
 *  ndim_upper = N, ndim_lower = 1
 */
template <>
struct TransformImpl<TransformType::EMBED>
{
    // ── Schema ──
    // Embed maps ALL tensor dims to a single offset, so it needs
    // MAX_TENSOR_DIMS capacity (64 entries per array).
    struct Schema
    {
        static_array<index_t, MAX_TENSOR_DIMS> dim_lengths{};
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

    /** @brief Forward: N upper dims to 1 lower offset via linear combination. */
    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& t, index_t* lower, const index_t* upper)
    {
        auto d   = readSchema(t);
        lower[0] = 0;
        for(index_t i = 0; i < t.ndim_upper; ++i)
        {
            lower[0] += upper[i] * d.strides[i];
        }
    }

    /** @brief Reverse: decompose offset into N indices via magic division on strides. */
    static CK_TILE_HOST_DEVICE constexpr void
    reverseMapIndices(const CoordinateTransform& t, index_t* upper, const index_t* lower)
    {
        auto d            = readSchema(t);
        index_t remaining = lower[0];
        for(index_t i = 0; i < t.ndim_upper - 1; ++i)
        {
            index_t quotient =
                static_cast<index_t>(doMagicDiv(static_cast<uint32_t>(remaining), d.magic_divs[i]));
            upper[i] = quotient;
            remaining -= quotient * d.strides[i];
        }
        upper[t.ndim_upper - 1] = remaining;
    }

    /** @brief Check all upper indices are within [0, dim_length). */
    static CK_TILE_HOST_DEVICE constexpr bool isValidUpper(const CoordinateTransform& t,
                                                           const index_t* upper)
    {
        auto d = readSchema(t);
        for(index_t i = 0; i < t.ndim_upper; ++i)
        {
            if(upper[i] < 0 || upper[i] >= d.dim_lengths[i])
                return false;
        }
        return true;
    }

    /** @brief Get the length of the i-th upper dimension. */
    static CK_TILE_HOST_DEVICE constexpr index_t upperLength(const CoordinateTransform& t,
                                                             index_t i)
    {
        auto d = readSchema(t);
        return d.dim_lengths[i];
    }
};

} // namespace ck_tile
