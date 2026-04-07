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

template <>
struct tensor_descriptor_traits<backend_type::mint>
{
    /// @brief Create packed descriptor with string aliases.
    /// @tparam kAliases  String aliases for each dimension (e.g., "M", "K")
    template <mint::alias_t... kAliases>
    MINT_HOST_DEVICE static constexpr auto
    make_naive_packed(mint::sequence<mint::alias_t, kAliases...>,
                      const mint::nd_index<sizeof...(kAliases)>& lengths)
    {
        return mint::tensor::make_aliased_naive_packed_tensor_descriptor(
            mint::sequence<mint::alias_t, kAliases...>{},
            mint::integral_constant<mint::alias_t, "Offset">{},
            lengths);
    }

    /// @brief Create strided descriptor with string aliases.
    template <mint::alias_t... kAliases>
    MINT_HOST_DEVICE static constexpr auto
    make_naive(mint::sequence<mint::alias_t, kAliases...>,
               const mint::nd_index<sizeof...(kAliases)>& lengths,
               const mint::nd_index<sizeof...(kAliases)>& strides)
    {
        return mint::tensor::make_aliased_naive_tensor_descriptor(
            mint::sequence<mint::alias_t, kAliases...>{},
            mint::integral_constant<mint::alias_t, "Offset">{},
            lengths,
            strides);
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

/// @brief Create packed descriptor with explicit string aliases (MINT).
/// This is the primary API for MINT - aliases must match distribution aliases.
/// Example: make_aliased_descriptor<"M", "K">(m_size, k_size)
template <mint::alias_t... kAliases, typename... Dims>
MINT_HOST_DEVICE constexpr auto make_aliased_descriptor(Dims... dims)
{
    static_assert(sizeof...(kAliases) == sizeof...(Dims),
                  "Number of aliases must match number of dimensions");
    return tensor_descriptor_traits<backend_type::mint>::make_naive_packed(
        mint::sequence<mint::alias_t, kAliases...>{},
        mint::nd_index<sizeof...(Dims)>{
            static_cast<mint::index_t>(dims)...});
}

/// @brief Create strided descriptor with explicit string aliases (MINT).
template <mint::alias_t... kAliases>
MINT_HOST_DEVICE constexpr auto make_aliased_strided_descriptor(
    const mint::nd_index<sizeof...(kAliases)>& lengths,
    const mint::nd_index<sizeof...(kAliases)>& strides)
{
    return tensor_descriptor_traits<backend_type::mint>::make_naive(
        mint::sequence<mint::alias_t, kAliases...>{},
        lengths,
        strides);
}

/// @brief Create packed descriptor with default aliases "D0","D1",... (MINT).
/// Use make_aliased_descriptor<"M","K"> when aliases must match a distribution.
template <typename... Dims>
MINT_HOST_DEVICE constexpr auto make_descriptor(Dims... dims)
{
    if constexpr(sizeof...(Dims) == 1)
    {
        return make_aliased_descriptor<"D0">(dims...);
    }
    else if constexpr(sizeof...(Dims) == 2)
    {
        return make_aliased_descriptor<"D0", "D1">(dims...);
    }
    else if constexpr(sizeof...(Dims) == 3)
    {
        return make_aliased_descriptor<"D0", "D1", "D2">(dims...);
    }
    else if constexpr(sizeof...(Dims) == 4)
    {
        return make_aliased_descriptor<"D0", "D1", "D2", "D3">(dims...);
    }
}

/// @brief Create strided descriptor with default aliases (MINT).
template <mint::index_t N>
MINT_HOST_DEVICE constexpr auto make_descriptor_with_strides(
    const mint::nd_index<N>& lengths,
    const mint::nd_index<N>& strides)
{
    if constexpr(N == 1)
    {
        return make_aliased_strided_descriptor<"D0">(lengths, strides);
    }
    else if constexpr(N == 2)
    {
        return make_aliased_strided_descriptor<"D0", "D1">(lengths, strides);
    }
    else if constexpr(N == 3)
    {
        return make_aliased_strided_descriptor<"D0", "D1", "D2">(
            lengths, strides);
    }
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
