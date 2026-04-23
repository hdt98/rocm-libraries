// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief Opaque byte-buffer storage backing every CoordinateTransform.
///
/// Every CoordinateTransform stores its parameters in a fixed-size,
/// type-erased byte buffer. Each TransformImpl<Type> specialization
/// reinterprets the buffer through a typed Schema (via constexpr bit_cast)
/// to access named, transform-specific fields. The buffer's capacity must
/// accommodate the largest Schema across all transform types.
///
/// Internal: not part of the user-facing transform API.
// IWYU pragma: private, include "ck_tile/experimental/core/transform/coordinate_transform.hpp"
#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/static_array.hpp"
#include "ck_tile/core/numeric/integer.hpp"

#include <stdint.h>

namespace ck_tile::core::transform::detail {

// Size of the opaque data buffer in CoordinateTransform.
// The largest Schema is Embed with MAX_TENSOR_DIMS (64) dimensions, requiring
//   64 dim_lengths     (256 B)
// + 64 strides         (256 B)
// + 64 magic divisors  (512 B)
// = 1024 B + 8 B alignment slack = 1032 B.
inline constexpr index_t MAX_TRANSFORM_DATA_SIZE = 1032;

// Opaque byte buffer; reinterpreted via bit_cast in each TransformImpl<Type>.
using TransformDataBuffer = static_array<uint8_t, MAX_TRANSFORM_DATA_SIZE>;

} // namespace ck_tile::core::transform::detail
