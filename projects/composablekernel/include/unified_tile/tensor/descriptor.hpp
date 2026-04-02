// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../config.hpp"

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
#endif

#ifdef UNIFIED_TILE_BACKEND_MINT
#include "mint/mint.h"
#endif

namespace unified_tile {
namespace descriptor {

// Forward declaration
template <backend_type Backend>
struct tensor_descriptor_traits;

// ============================================================================
// CK_Tile Backend Traits
// ============================================================================

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
template <>
struct tensor_descriptor_traits<backend_type::ck_tile>
{
    template <typename... Lengths,
              typename... Strides,
              ck_tile::index_t GuaranteedLastDimensionVectorLength  = -1,
              ck_tile::index_t GuaranteedLastDimensionVectorStride = -1>
    CK_TILE_HOST_DEVICE static constexpr auto make_naive(
        const ck_tile::tuple<Lengths...>& lengths,
        const ck_tile::tuple<Strides...>& strides,
        ck_tile::number<GuaranteedLastDimensionVectorLength> vec_len =
            ck_tile::number<-1>{},
        ck_tile::number<GuaranteedLastDimensionVectorStride> vec_stride =
            ck_tile::number<-1>{})
    {
        return ck_tile::make_naive_tensor_descriptor(
            lengths, strides, vec_len, vec_stride);
    }

    template <typename... Lengths,
              ck_tile::index_t GuaranteedLastDimensionVectorLength = -1>
    CK_TILE_HOST_DEVICE static constexpr auto make_naive_packed(
        const ck_tile::tuple<Lengths...>& lengths,
        ck_tile::number<GuaranteedLastDimensionVectorLength> vec_len =
            ck_tile::number<-1>{})
    {
        return ck_tile::make_naive_tensor_descriptor_packed(lengths, vec_len);
    }
};
#endif

// ============================================================================
// MINT Backend Traits
// ============================================================================

#ifdef UNIFIED_TILE_BACKEND_MINT

namespace detail {

template <mint::index_t N, mint::index_t... Is>
constexpr auto
make_top_dim_aliases_impl(mint::index_sequence<Is...>)
{
    return mint::sequence<mint::index_t, (Is + 1)...>{};
}

template <mint::index_t N>
constexpr auto make_top_dim_aliases()
{
    return make_top_dim_aliases_impl<N>(mint::make_index_sequence<N>{});
}

template <mint::index_t BottomAlias = 0>
constexpr auto make_bottom_dim_alias()
{
    return mint::integral_constant<mint::index_t, BottomAlias>{};
}

} // namespace detail

template <>
struct tensor_descriptor_traits<backend_type::mint>
{
    template <mint::index_t N>
    MINT_HOST_DEVICE static constexpr auto
    make_naive_packed(const mint::nd_index<N>& lengths)
    {
        constexpr auto dim_seq    = detail::make_top_dim_aliases<N>();
        constexpr auto offset_alias = detail::make_bottom_dim_alias<0>();
        return mint::tensor::make_aliased_naive_packed_tensor_descriptor(
            dim_seq, offset_alias, lengths);
    }

    template <mint::index_t N>
    MINT_HOST_DEVICE static constexpr auto
    make_naive(const mint::nd_index<N>& lengths,
               const mint::nd_index<N>& strides)
    {
        constexpr auto dim_seq    = detail::make_top_dim_aliases<N>();
        constexpr auto offset_alias = detail::make_bottom_dim_alias<0>();
        return mint::tensor::make_aliased_naive_tensor_descriptor(
            dim_seq, offset_alias, lengths, strides);
    }
};
#endif

// ============================================================================
// Public Unified API
// ============================================================================

#ifdef UNIFIED_TILE_BACKEND_CK_TILE

/// @brief Create naive tensor descriptor with lengths and strides (CK_Tile)
template <typename... Lengths,
          typename... Strides,
          ck_tile::index_t GuaranteedLastDimensionVectorLength  = -1,
          ck_tile::index_t GuaranteedLastDimensionVectorStride = -1>
CK_TILE_HOST_DEVICE constexpr auto make_strided_descriptor(
    const ck_tile::tuple<Lengths...>& lengths,
    const ck_tile::tuple<Strides...>& strides,
    ck_tile::number<GuaranteedLastDimensionVectorLength> vec_len =
        ck_tile::number<-1>{},
    ck_tile::number<GuaranteedLastDimensionVectorStride> vec_stride =
        ck_tile::number<-1>{})
{
    return tensor_descriptor_traits<backend_type::ck_tile>::make_naive(
        lengths, strides, vec_len, vec_stride);
}

/// @brief Create naive packed tensor descriptor (CK_Tile)
template <typename... Lengths,
          ck_tile::index_t GuaranteedLastDimensionVectorLength = -1>
CK_TILE_HOST_DEVICE constexpr auto make_packed_descriptor(
    const ck_tile::tuple<Lengths...>& lengths,
    ck_tile::number<GuaranteedLastDimensionVectorLength> vec_len =
        ck_tile::number<-1>{})
{
    return tensor_descriptor_traits<backend_type::ck_tile>::make_naive_packed(
        lengths, vec_len);
}

/// @brief Create naive packed descriptor (convenience, variadic)
template <typename... Dims>
CK_TILE_HOST_DEVICE constexpr auto make_descriptor(Dims... dims)
{
    return tensor_descriptor_traits<backend_type::ck_tile>::make_naive_packed(
        ck_tile::make_tuple(dims...));
}

/// @brief Create descriptor with strides (convenience)
template <typename... Lengths, typename... Strides>
CK_TILE_HOST_DEVICE constexpr auto make_descriptor_with_strides(
    const ck_tile::tuple<Lengths...>& lengths,
    const ck_tile::tuple<Strides...>& strides)
{
    return tensor_descriptor_traits<backend_type::ck_tile>::make_naive(
        lengths, strides);
}

#else // UNIFIED_TILE_BACKEND_MINT

/// @brief Create tensor descriptor with lengths and strides (MINT)
template <mint::index_t N>
MINT_HOST_DEVICE constexpr auto make_strided_descriptor(
    const mint::nd_index<N>& lengths,
    const mint::nd_index<N>& strides)
{
    return tensor_descriptor_traits<backend_type::mint>::make_naive(
        lengths, strides);
}

/// @brief Create packed tensor descriptor (MINT)
template <mint::index_t N>
MINT_HOST_DEVICE constexpr auto
make_packed_descriptor(const mint::nd_index<N>& lengths)
{
    return tensor_descriptor_traits<backend_type::mint>::make_naive_packed(
        lengths);
}

/// @brief Create packed descriptor (convenience, variadic)
template <typename... Dims>
MINT_HOST_DEVICE constexpr auto make_descriptor(Dims... dims)
{
    return tensor_descriptor_traits<backend_type::mint>::make_naive_packed(
        mint::nd_index<sizeof...(Dims)>{
            static_cast<mint::index_t>(dims)...});
}

/// @brief Create descriptor with strides (convenience)
template <mint::index_t N>
MINT_HOST_DEVICE constexpr auto make_descriptor_with_strides(
    const mint::nd_index<N>& lengths,
    const mint::nd_index<N>& strides)
{
    return tensor_descriptor_traits<backend_type::mint>::make_naive(
        lengths, strides);
}

#endif // UNIFIED_TILE_BACKEND

// ============================================================================
// Query Functions
// ============================================================================

/// @brief Get number of (top-level / logical) dimensions
template <typename Descriptor>
UNIFIED_TILE_HOST_DEVICE constexpr auto
get_num_dimensions(const Descriptor& desc)
{
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    return desc.get_num_of_dimension();
#else
    (void)desc;
    return Descriptor::top_ndim();
#endif
}

/// @brief Get dimension lengths
template <typename Descriptor>
UNIFIED_TILE_HOST_DEVICE constexpr auto get_lengths(const Descriptor& desc)
{
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    return desc.get_lengths();
#else
    return desc.top_lengths();
#endif
}

} // namespace descriptor
} // namespace unified_tile
