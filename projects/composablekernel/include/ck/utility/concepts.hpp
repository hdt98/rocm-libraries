// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <type_traits>

namespace ck {

// Concept for arithmetic types (integral or floating-point)
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

// Concept for integral types
template <typename T>
concept Integral = std::integral<T>;

// Concept for floating-point types
template <typename T>
concept FloatingPoint = std::floating_point<T>;

// Concept for types that can be used as tensor indices
template <typename T>
concept IndexType = std::is_integral_v<T> && std::is_signed_v<T>;

// Concept for types that are trivially copyable (important for GPU data transfer)
template <typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

// Concept for types that can be used in GPU kernels (trivially copyable and destructible)
template <typename T>
concept GpuCompatible = std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>;

} // namespace ck
