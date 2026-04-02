// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../config.hpp"
#include "backend_traits.hpp"
#include <type_traits>

/**
 * @file type_concepts.hpp
 * @brief C++20 concepts for compile-time type safety
 *
 * This file defines concepts that enforce type requirements at compile time,
 * providing better error messages and documentation for the unified tile API.
 */

namespace unified_tile {

// ============================================================================
// Core Tensor Concepts
// ============================================================================

/// @brief Concept for tensor descriptor types
/// A tensor descriptor must provide dimension information
template <typename T>
concept TensorDescriptor = requires(const T& desc) {
    // Must be able to query number of dimensions
    { desc.get_num_of_dimension() } -> std::convertible_to<index_t>;

    // Must be able to get dimension lengths
    { desc.get_lengths() };
};

/// @brief Concept for tensor view types
/// A tensor view combines a buffer with a descriptor
template <typename T>
concept TensorView = requires(T& view) {
    // Must have an associated data type
    typename T::DataType;

    // Must be able to get the underlying tensor descriptor
    { view.get_tensor_descriptor() };
};

/// @brief Concept for tile window types
/// A tile window represents a movable window over a tensor
template <typename T>
concept TileWindow = requires(T& window) {
    // Must be able to get window origin (position in tensor)
    { window.get_window_origin() };

    // Must be able to get window lengths (tile dimensions)
    { window.get_window_lengths() };
};

/// @brief Concept for tile distribution types
/// A distribution describes how work is partitioned across threads
template <typename T>
concept TileDistribution = requires(const T& dist) {
    // Must be able to query distribution properties
    // This is backend-specific, so we keep it minimal for now
    // CK_Tile: has tile_distribution_encoding methods
    // MINT: has polymorpher with dimension queries
    sizeof(T);  // Just ensure it's a complete type
};

/// @brief Concept for distributed tensor types
/// A distributed tensor stores data in VGPRs with a distribution pattern
template <typename T>
concept DistributedTensor = requires(T& tensor) {
    // Must have an associated data type
    typename T::DataType;

    // Must support some form of element access
    // The exact interface varies by backend
    sizeof(T);  // Ensure complete type
};

// ============================================================================
// Backend-Specific Concepts (Optional - More Strict Checking)
// ============================================================================

#ifdef UNIFIED_TILE_BACKEND_CK_TILE

/// @brief Concept for CK_Tile tensor descriptor
template <typename T>
concept CKTileTensorDescriptor = TensorDescriptor<T> && requires(const T& desc) {
    // CK_Tile specific: coordinate transforms
    { desc.calculate_offset(desc.get_lengths()) } -> std::convertible_to<index_t>;
};

/// @brief Concept for CK_Tile tensor view
template <typename T>
concept CKTileTensorView = TensorView<T> && requires(T& view) {
    // CK_Tile specific: buffer_view
    { view.get_buffer_view() };
};

#endif // UNIFIED_TILE_BACKEND_CK_TILE

#ifdef UNIFIED_TILE_BACKEND_MINT

/// @brief Concept for MINT tensor descriptor
template <typename T>
concept MINTTensorDescriptor = TensorDescriptor<T> && requires(const T& desc) {
    // MINT specific: bottom and top dimensions
    { desc.bottom_ndim() } -> std::convertible_to<index_t>;
    { desc.top_ndim() } -> std::convertible_to<index_t>;
};

/// @brief Concept for MINT tensor view
template <typename T>
concept MINTTensorView = TensorView<T> && requires(T& view) {
    // MINT specific: memory_view
    { view.get_memory_view() };
};

#endif // UNIFIED_TILE_BACKEND_MINT

// ============================================================================
// Utility Concepts
// ============================================================================

/// @brief Concept for integral constant types
/// Checks if type represents a compile-time integer constant
template <typename T>
concept IntegralConstant = requires {
    { T::value } -> std::convertible_to<index_t>;
};

/// @brief Concept for compile-time sequence types
/// Checks if type is an index sequence (like sequence<0, 1, 2>)
template <typename T>
concept IndexSequence = requires {
    { T::size() } -> std::convertible_to<index_t>;
};

/// @brief Concept for multi-dimensional index types
/// Checks if type can be used as a multi-dimensional index
template <typename T>
concept MultiDimensionalIndex = requires(T idx) {
    // Must support subscript operator
    { idx[0] } -> std::convertible_to<index_t>;

    // Must have size information
    { idx.size() } -> std::convertible_to<index_t>;
};

/// @brief Concept for static array types
/// Checks if type behaves like a static array
template <typename T>
concept StaticArray = requires(T arr) {
    // Must support subscript operator
    { arr[0] };

    // Must have size method or static size member
    { arr.size() } -> std::convertible_to<index_t>;
};

/// @brief Concept for pointer types
/// Checks if type is a pointer (raw or smart)
template <typename T>
concept Pointer = std::is_pointer_v<T>;

/// @brief Concept for arithmetic types (integer or floating point)
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

// ============================================================================
// Type Trait Helpers (for non-concept code)
// ============================================================================

/// @brief Check if type is a tensor descriptor at compile time
template <typename T>
inline constexpr bool is_tensor_descriptor_v = TensorDescriptor<T>;

/// @brief Check if type is a tensor view at compile time
template <typename T>
inline constexpr bool is_tensor_view_v = TensorView<T>;

/// @brief Check if type is a tile window at compile time
template <typename T>
inline constexpr bool is_tile_window_v = TileWindow<T>;

/// @brief Check if type is a tile distribution at compile time
template <typename T>
inline constexpr bool is_tile_distribution_v = TileDistribution<T>;

/// @brief Check if type is a distributed tensor at compile time
template <typename T>
inline constexpr bool is_distributed_tensor_v = DistributedTensor<T>;

// ============================================================================
// Example Concept-Constrained Functions
// ============================================================================

/// @brief Get number of dimensions from any tensor descriptor
/// @tparam Desc Must satisfy TensorDescriptor concept
template <TensorDescriptor Desc>
UNIFIED_TILE_HOST_DEVICE constexpr auto get_num_dimensions(const Desc& desc) {
    return desc.get_num_of_dimension();
}

/// @brief Get tensor descriptor from any tensor view
/// @tparam View Must satisfy TensorView concept
template <TensorView View>
UNIFIED_TILE_HOST_DEVICE constexpr auto get_descriptor(View& view) {
    return view.get_tensor_descriptor();
}

/// @brief Get window origin from any tile window
/// @tparam Window Must satisfy TileWindow concept
template <TileWindow Window>
UNIFIED_TILE_HOST_DEVICE constexpr auto get_origin(Window& window) {
    return window.get_window_origin();
}

// ============================================================================
// Concept Checking Utilities
// ============================================================================

/// @brief Compile-time assertion that type satisfies a concept
/// Usage: static_assert(satisfies<MyType, TensorDescriptor>, "MyType must be a TensorDescriptor");
template <typename T, template<typename> class Concept>
inline constexpr bool satisfies = Concept<T>;

/// @brief Helper to print concept satisfaction at compile time (for debugging)
/// Usage: concept_check<MyType, TensorDescriptor>()
template <typename T, template<typename> class Concept>
constexpr void concept_check() {
    static_assert(Concept<T>, "Type does not satisfy the required concept");
}

} // namespace unified_tile
