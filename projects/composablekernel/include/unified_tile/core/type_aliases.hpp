// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../config.hpp"
#include "backend_traits.hpp"

/**
 * @file type_aliases.hpp
 * @brief Common type aliases for backend-agnostic code
 *
 * This file provides convenient type aliases that automatically resolve
 * to the correct backend-specific types based on compile-time configuration.
 */

namespace unified_tile {

// ============================================================================
// Fundamental Type Aliases (from backend_traits)
// ============================================================================

/// @brief Index type from active backend
/// CK_Tile: ck_tile::index_t (int32_t)
/// MINT: mint::index_t (int64_t)
using index_t = typename backend_traits::index_t;

/// @brief Compile-time integer constant from active backend
/// CK_Tile: ck_tile::number<N>
/// MINT: mint::constant<N>
template <index_t N>
using number = typename backend_traits::template number<N>;

/// @brief Integer sequence from active backend
/// CK_Tile: ck_tile::sequence<Is...>
/// MINT: mint::index_sequence<Is...>
template <index_t... Is>
using sequence = typename backend_traits::template sequence<Is...>;

/// @brief Static array from active backend
/// CK_Tile: ck_tile::array<T, N>
/// MINT: mint::nd_array<T, N>
template <typename T, index_t N>
using array = typename backend_traits::template array<T, N>;

/// @brief Multi-dimensional index from active backend
/// CK_Tile: tuple of numbers
/// MINT: mint::nd_index<NDim>
template <index_t NDim>
using multi_index = typename backend_traits::template multi_index<NDim>;

// ============================================================================
// Helper Type Aliases for Common Use Cases
// ============================================================================

/// @brief 1D index (single integer)
using index_1d = multi_index<1>;

/// @brief 2D index (row, col)
using index_2d = multi_index<2>;

/// @brief 3D index (batch, row, col)
using index_3d = multi_index<3>;

/// @brief 4D index (batch, channel, height, width)
using index_4d = multi_index<4>;

// ============================================================================
// Backend-Specific Wrappers
// ============================================================================

#ifdef UNIFIED_TILE_BACKEND_CK_TILE

/// @brief CK_Tile tuple type
template <typename... Ts>
using tuple = decltype(ck_tile::make_tuple(std::declval<Ts>()...));

/// @brief Create a tuple (CK_Tile)
template <typename... Ts>
UNIFIED_TILE_HOST_DEVICE constexpr auto make_tuple(Ts&&... args)
{
    return ck_tile::make_tuple(std::forward<Ts>(args)...);
}

/// @brief Create a sequence (CK_Tile)
template <index_t... Is>
UNIFIED_TILE_HOST_DEVICE constexpr auto make_sequence()
{
    return ck_tile::sequence<Is...>{};
}

/// @brief Create a 2D multi-index (CK_Tile)
UNIFIED_TILE_HOST_DEVICE constexpr auto make_multi_index(index_t m, index_t n)
{
    return ck_tile::make_tuple(m, n);
}

/// @brief Create a 3D multi-index (CK_Tile)
UNIFIED_TILE_HOST_DEVICE constexpr auto
make_multi_index(index_t b, index_t m, index_t n)
{
    return ck_tile::make_tuple(b, m, n);
}

#else // UNIFIED_TILE_BACKEND_MINT

/// @brief MINT tuple type
template <typename... Ts>
using tuple = mint::tuple<Ts...>;

/// @brief Create a tuple (MINT)
template <typename... Ts>
UNIFIED_TILE_HOST_DEVICE constexpr auto make_tuple(Ts&&... args)
{
    return mint::make_tuple(std::forward<Ts>(args)...);
}

/// @brief Create a sequence (MINT)
template <index_t... Is>
UNIFIED_TILE_HOST_DEVICE constexpr auto make_sequence()
{
    return mint::index_sequence<Is...>{};
}

/// @brief Create a 2D multi-index (MINT)
UNIFIED_TILE_HOST_DEVICE constexpr auto make_multi_index(index_t m, index_t n)
{
    return mint::nd_index<2>{m, n};
}

/// @brief Create a 3D multi-index (MINT)
UNIFIED_TILE_HOST_DEVICE constexpr auto
make_multi_index(index_t b, index_t m, index_t n)
{
    return mint::nd_index<3>{b, m, n};
}

#endif // UNIFIED_TILE_BACKEND

// ============================================================================
// Common Utility Functions
// ============================================================================

/// @brief Create a number (compile-time constant)
template <index_t N>
UNIFIED_TILE_HOST_DEVICE constexpr auto make_number()
{
    return number<N>{};
}

/// @brief Get value from number type
template <typename NumberType>
UNIFIED_TILE_HOST_DEVICE constexpr auto get_value(NumberType)
{
    return NumberType::value;
}

// ============================================================================
// Type Deduction Helpers
// ============================================================================

/// @brief Deduce index_t from any backend type
template <typename T>
struct extract_index_type
{
    using type = index_t;
};

/// @brief Deduce data type from tensor view
template <typename TensorView>
struct extract_data_type
{
    using type = typename TensorView::DataType;
};

/// @brief Helper alias for extract_data_type
template <typename TensorView>
using extract_data_type_t = typename extract_data_type<TensorView>::type;

// ============================================================================
// Size Computation Helpers
// ============================================================================

/// @brief Calculate total size from multi-dimensional index
template <index_t NDim>
UNIFIED_TILE_HOST_DEVICE constexpr index_t
compute_total_size(const multi_index<NDim>& idx)
{
    index_t size = 1;
    for(index_t i = 0; i < NDim; ++i)
    {
        size *= idx[i];
    }
    return size;
}

/// @brief Calculate 2D size (M * N)
UNIFIED_TILE_HOST_DEVICE constexpr index_t compute_2d_size(index_t M, index_t N)
{
    return M * N;
}

/// @brief Calculate 3D size (B * M * N)
UNIFIED_TILE_HOST_DEVICE constexpr index_t
compute_3d_size(index_t B, index_t M, index_t N)
{
    return B * M * N;
}

// ============================================================================
// Debug Helpers
// ============================================================================

#if UNIFIED_TILE_DEBUG

/// @brief Print type information at compile time (causes compilation to fail
/// with type info)
template <typename T>
struct print_type;

/// @brief Print value at compile time
template <auto Value>
struct print_value;

#endif // UNIFIED_TILE_DEBUG

} // namespace unified_tile
