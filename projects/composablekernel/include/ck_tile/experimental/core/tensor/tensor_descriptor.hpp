// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file tensor_descriptor.hpp
 *  @brief Pure metadata describing a tensor's shape and memory layout.
 *
 *  TensorDescriptor is user-facing metadata: lengths, strides, ndim,
 *  element_space_size. It knows nothing about coordinate transforms or
 *  transform graphs — those are internal implementation details.
 *
 *  A TransformGraph can be constructed FROM a descriptor via
 *  make_transform_graph(), which creates an Embed transform from the
 *  descriptor's lengths and strides.
 *
 *  TensorDescriptor is a structural NTTP (pure aggregate, defaulted ==).
 */

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/static_array.hpp"
#include "ck_tile/core/numeric/integer.hpp"
namespace ck_tile {

/// Maximum tensor dimensions. Determines capacity of TensorDescriptor,
/// DimIds routing arrays, Embed schema, and TransformGraph endpoints.
inline constexpr index_t MAX_TENSOR_DIMS = 64;

/** @brief Pure metadata describing a tensor's shape and memory layout.
 *
 *  Structural NTTP — can be used as template<TensorDescriptor D>.
 *  Contains only the information needed to describe a tensor's geometry:
 *  how many dimensions, how large each dimension is, and what stride
 *  each dimension has in memory.
 *
 *  Does NOT contain:
 *  - Data pointers (those belong in TensorView)
 *  - Transform graphs (those are constructed from a descriptor when needed)
 *  - Coordinate state (that belongs in TensorCoordinate)
 */
struct TensorDescriptor
{
    static_array<index_t, MAX_TENSOR_DIMS> lengths{};
    static_array<index_t, MAX_TENSOR_DIMS> strides{};
    index_t ndim               = 0;
    index_t element_space_size = 0;

    constexpr bool operator==(const TensorDescriptor&) const = default;
};

/** @brief Create a tensor descriptor with explicit strides.
 *
 *  @tparam NDim  Number of dimensions (deduced from array size)
 *  @param lengths  Size of each dimension
 *  @param strides  Stride of each dimension (in elements)
 *  @return TensorDescriptor with the given shape and strides
 *
 *  Example: make_tensor_descriptor({8, 128, 8}, {1032, 8, 1})
 */
template <index_t NDim>
CK_TILE_HOST_DEVICE constexpr TensorDescriptor
make_tensor_descriptor(const static_array<index_t, NDim>& lengths,
                       const static_array<index_t, NDim>& strides)
{
    static_assert(NDim >= 1, "make_tensor_descriptor: need at least 1 dim");
    static_assert(NDim <= MAX_TENSOR_DIMS,
                  "make_tensor_descriptor: too many dims (max MAX_TENSOR_DIMS)");

    TensorDescriptor desc{};
    desc.ndim = NDim;

    for(index_t i = 0; i < NDim; ++i)
    {
        desc.lengths[i] = lengths[i];
        desc.strides[i] = strides[i];
    }

    // element_space_size = 1 + sum((length[i] - 1) * stride[i])
    desc.element_space_size = 1;
    for(index_t i = 0; i < NDim; ++i)
    {
        desc.element_space_size += (lengths[i] - 1) * strides[i];
    }

    // Zero-fill unused slots for NTTP deduplication
    for(index_t i = NDim; i < MAX_TENSOR_DIMS; ++i)
    {
        desc.lengths[i] = 0;
        desc.strides[i] = 0;
    }

    return desc;
}

/** @brief Create a tensor descriptor with packed (row-major) strides.
 *
 *  Strides are computed automatically: stride[i] = product(lengths[i+1..N-1]).
 *  The last dimension is contiguous (stride = 1).
 *
 *  @tparam NDim  Number of dimensions (deduced from array size)
 *  @param lengths  Size of each dimension
 *  @return TensorDescriptor with row-major strides
 *
 *  Example: make_tensor_descriptor({4, 8}) -> strides = {8, 1}
 */
template <index_t NDim>
CK_TILE_HOST_DEVICE constexpr TensorDescriptor
make_tensor_descriptor(const static_array<index_t, NDim>& lengths)
{
    static_assert(NDim >= 1, "make_tensor_descriptor: need at least 1 dim");
    static_assert(NDim <= MAX_TENSOR_DIMS,
                  "make_tensor_descriptor: too many dims (max MAX_TENSOR_DIMS)");

    // Compute row-major strides: stride[i] = product(lengths[i+1..N-1])
    static_array<index_t, NDim> strides{};
    strides[NDim - 1] = 1;
    for(index_t i = NDim - 1; i > 0; --i)
    {
        strides[i - 1] = strides[i] * lengths[i];
    }

    return make_tensor_descriptor(lengths, strides);
}

} // namespace ck_tile
