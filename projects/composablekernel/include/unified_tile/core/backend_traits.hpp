// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../config.hpp"

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    #include "ck_tile/core.hpp"
#else
    #include "mint/poly/fundamental_morpher.hpp"
    #include "mint/tensor/tensor_descriptor_helper.h"
    #include "mint/tile/tile_overview.hpp"
#endif

namespace unified_tile {

// ============================================================================
// Backend Traits Implementation (Template Specializations)
// ============================================================================

/// @brief Backend-specific type traits
/// @tparam Backend The backend type
template <backend_type Backend>
struct backend_traits_impl;

// ============================================================================
// CK_Tile Backend Traits
// ============================================================================

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
template <>
struct backend_traits_impl<backend_type::ck_tile> {
    // ------------------------------------------------------------------------
    // Fundamental Types
    // ------------------------------------------------------------------------

    /// @brief Index type (signed integer for tensor dimensions)
    using index_t = ck_tile::index_t;

    /// @brief Compile-time integer constant
    template <index_t N>
    using number = ck_tile::number<N>;

    /// @brief Compile-time integer sequence
    template <index_t... Is>
    using sequence = ck_tile::sequence<Is...>;

    /// @brief Static array type
    template <typename T, index_t N>
    using array = ck_tile::array<T, N>;

    /// @brief Multi-dimensional index (tuple of indices)
    template <index_t NDim>
    using multi_index = decltype(ck_tile::make_tuple(
        ck_tile::number<0>{} /* repeated NDim times */));

    // ------------------------------------------------------------------------
    // Tensor Types (Using decltype for actual return types)
    // ------------------------------------------------------------------------

    /// @brief Tensor descriptor type - use decltype of make_naive_tensor_descriptor
    /// Returns: tensor_descriptor<Transforms, LowerDims, UpperDims, TopDims, ElementSpaceSize, ...>
    /// Full signature: make_naive_tensor_descriptor(lengths, strides, vector_length, vector_stride)
    /// Example: make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(K, 1), number<8>{}, number<1>{})
    template <typename Lengths,
              typename Strides,
              index_t VectorLength = -1,
              index_t VectorStride = -1>
    using tensor_descriptor_t = decltype(ck_tile::make_naive_tensor_descriptor(
        std::declval<Lengths>(),
        std::declval<Strides>(),
        ck_tile::number<VectorLength>{},
        ck_tile::number<VectorStride>{}));

    /// @brief Tensor view type - use decltype of make_tensor_view
    /// Returns: tensor_view<BufferView, TensorDesc, MemOp>
    template <typename DataType, typename Descriptor>
    using tensor_view_t = decltype(ck_tile::make_tensor_view<
        ck_tile::address_space_enum::global>(
        std::declval<DataType*>(),
        std::declval<Descriptor>()));

    /// @brief Tile window type - use decltype of make_tile_window
    /// Returns: tile_window_with_static_distribution<...>
    template <typename TensorView, typename WindowLengths, typename Distribution>
    using tile_window_t = decltype(ck_tile::make_tile_window(
        std::declval<TensorView>(),
        std::declval<WindowLengths>(),
        ck_tile::make_tuple(0, 0),  // origin
        std::declval<Distribution>()));

    /// @brief Tile distribution encoding - use decltype of make_static_tile_distribution
    /// Returns: static_tile_distribution<...>
    template <typename DistributionEncoding>
    using tile_distribution_t = decltype(ck_tile::make_static_tile_distribution(
        std::declval<DistributionEncoding>()));

    /// @brief Distributed tensor - use decltype of make_static_distributed_tensor
    /// Returns: static_distributed_tensor<...>
    template <typename Distribution>
    using distributed_tensor_t = decltype(ck_tile::make_static_distributed_tensor<
        float /* DataType placeholder */>(
        std::declval<Distribution>()));

    // ------------------------------------------------------------------------
    // Feature Flags
    // ------------------------------------------------------------------------

    /// @brief CK_Tile uses pre-computed coordinates for tile windows
    static constexpr bool has_pre_computed_coords = true;

    /// @brief CK_Tile does not use named dimension aliases
    static constexpr bool has_named_dimensions = false;

    /// @brief CK_Tile has built-in tile distribution encoding
    static constexpr bool has_tile_distribution_encoding = true;

    /// @brief CK_Tile supports space-filling curve patterns
    static constexpr bool has_space_filling_curves = true;

    /// @brief CK_Tile does not use polymorpher composition
    static constexpr bool has_morpher_composition = false;

    /// @brief Backend name for debugging
    static constexpr const char* backend_name = "CK_Tile";
};
#endif

// ============================================================================
// MINT Backend Traits
// ============================================================================

#ifdef UNIFIED_TILE_BACKEND_MINT
template <>
struct backend_traits_impl<backend_type::mint> {
    // ------------------------------------------------------------------------
    // Fundamental Types
    // ------------------------------------------------------------------------

    /// @brief Index type (signed integer for tensor dimensions)
    using index_t = mint::index_t;

    /// @brief Compile-time integer constant
    template <index_t N>
    using number = mint::constant<N>;

    /// @brief Compile-time integer sequence
    template <index_t... Is>
    using sequence = mint::index_sequence<Is...>;

    /// @brief Static array type
    template <typename T, index_t N>
    using array = mint::nd_array<T, N>;

    /// @brief Multi-dimensional index
    template <index_t NDim>
    using multi_index = mint::nd_index<NDim>;

    // ------------------------------------------------------------------------
    // Tensor Types (Using decltype for actual return types)
    // ------------------------------------------------------------------------

    /// @brief Tensor descriptor type - use decltype of make_aliased_naive_packed_tensor_descriptor
    /// Full signature: make_aliased_naive_packed_tensor_descriptor(dim_aliases, offset_alias, lengths)
    /// Example: make_aliased_naive_packed_tensor_descriptor(
    ///            sequence<alias_t, alias_t{"M"}, alias_t{"K"}>{},
    ///            index_constant<-1>{},
    ///            nd_index<2>{M, K})
    template <typename DimAliasSeq, typename OffsetAlias, typename Lengths>
    using tensor_descriptor_t = decltype(mint::make_aliased_naive_packed_tensor_descriptor(
        std::declval<DimAliasSeq>(),
        std::declval<OffsetAlias>(),
        std::declval<Lengths>()));

    /// @brief Tensor view type - use decltype of make_tensor_view
    /// Signature: make_tensor_view(descriptor, memory_view)
    /// Example: make_tensor_view(desc, make_global_memory_view(p, size))
    template <typename Descriptor, typename MemoryView>
    using tensor_view_t = decltype(mint::make_tensor_view(
        std::declval<Descriptor>(),
        std::declval<MemoryView>()));

    /// @brief Distributed window type - use decltype of make_distributed_window
    /// Returns: distributed_window<...>
    template <typename TensorView, typename DistributionDesc>
    using distributed_window_t = decltype(mint::make_distributed_window(
        std::declval<TensorView>(),
        mint::nd_index<2>{0, 0},  // origin
        std::declval<DistributionDesc>(),
        mint::constant<mint::make_element_layout_from_distribution(
            std::declval<DistributionDesc>())>{},
        mint::constant<mint::this_thread_block{}>{}));

    /// @brief Polymorpher-based distribution - use decltype of make_simple_distribution
    /// Returns: distributed_tensor_descriptor<polymorpher, ...>
    template <typename TileLengths, typename PartitionStrategy>
    using distribution_desc_t = decltype(mint::make_simple_distribution(
        std::declval<TileLengths>(),
        std::declval<PartitionStrategy>(),
        mint::sequence<mint::alias_t>{},  // tile aliases
        mint::sequence<mint::alias_t>{}));  // partition aliases

    /// @brief Distributed tensor - use decltype of make_distributed_tensor_vgpr
    /// Returns: distributed_tensor<...>
    template <typename DataType, typename DistributionDesc>
    using distributed_tensor_t = decltype(mint::make_distributed_tensor_vgpr<DataType>(
        std::declval<DistributionDesc>()));

    // ------------------------------------------------------------------------
    // Feature Flags
    // ------------------------------------------------------------------------

    /// @brief MINT computes coordinates on-the-fly using polymorphers
    static constexpr bool has_pre_computed_coords = false;

    /// @brief MINT uses named dimension aliases extensively
    static constexpr bool has_named_dimensions = true;

    /// @brief MINT does not have tile distribution encoding (uses polymorphers)
    static constexpr bool has_tile_distribution_encoding = false;

    /// @brief MINT does not use space-filling curves
    static constexpr bool has_space_filling_curves = false;

    /// @brief MINT uses polymorpher composition for distributions
    static constexpr bool has_morpher_composition = true;

    /// @brief Backend name for debugging
    static constexpr const char* backend_name = "MINT";
};
#endif

// ============================================================================
// Unified Backend Traits (Active Backend)
// ============================================================================

/// @brief Type traits for the currently active backend
/// This provides a unified interface regardless of which backend is selected
using backend_traits = backend_traits_impl<active_backend>;

// ============================================================================
// Convenience Type Aliases
// ============================================================================

/// @brief Index type from active backend
using index_t = typename backend_traits::index_t;

/// @brief Compile-time integer constant from active backend
template <index_t N>
using number = typename backend_traits::template number<N>;

/// @brief Integer sequence from active backend
template <index_t... Is>
using sequence = typename backend_traits::template sequence<Is...>;

/// @brief Static array from active backend
template <typename T, index_t N>
using array = typename backend_traits::template array<T, N>;

/// @brief Multi-dimensional index from active backend
template <index_t NDim>
using multi_index = typename backend_traits::template multi_index<NDim>;

// ============================================================================
// Backend Feature Queries
// ============================================================================

/// @brief Check if active backend has pre-computed coordinates
inline constexpr bool has_pre_computed_coords = backend_traits::has_pre_computed_coords;

/// @brief Check if active backend uses named dimensions
inline constexpr bool has_named_dimensions = backend_traits::has_named_dimensions;

/// @brief Check if active backend has tile distribution encoding
inline constexpr bool has_tile_distribution_encoding = backend_traits::has_tile_distribution_encoding;

/// @brief Check if active backend supports space-filling curves
inline constexpr bool has_space_filling_curves = backend_traits::has_space_filling_curves;

/// @brief Check if active backend uses morpher composition
inline constexpr bool has_morpher_composition = backend_traits::has_morpher_composition;

/// @brief Get active backend name
inline constexpr const char* backend_name = backend_traits::backend_name;

} // namespace unified_tile
