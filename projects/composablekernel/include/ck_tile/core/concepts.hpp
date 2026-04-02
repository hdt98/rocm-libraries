// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <type_traits>

namespace ck_tile {

// Concept for arithmetic types (integral or floating-point)
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

// Concept for integral types
template <typename T>
concept Integral = std::integral<T>;

// Concept for floating-point types
template <typename T>
concept FloatingPoint = std::floating_point<T>;

// Concept for types that can be used as tile dimensions
template <typename T>
concept TileDimension = std::is_integral_v<T> && (sizeof(T) <= sizeof(int));

// Concept for types that are trivially copyable (important for GPU data transfer)
template <typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

// Concept for types that can be used in GPU tiles (trivially copyable and destructible)
template <typename T>
concept TileCompatible = std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>;

} // namespace ck_tile
