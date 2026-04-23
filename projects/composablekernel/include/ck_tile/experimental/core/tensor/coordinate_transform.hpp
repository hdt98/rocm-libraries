// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file coordinate_transform.hpp
 *  @brief Pure data struct describing a single coordinate transform.
 *
 *  CoordinateTransform carries an opaque data buffer interpreted by
 *  TransformImpl<Type> specializations via typed schema structs.
 *
 *  Each TransformImpl defines:
 *    - A nested Schema struct with named fields + private padding
 *    - readSchema(CoordinateTransform) to read buffer -> typed Schema via bit_cast
 *    - writeSchema(CoordinateTransform&, Schema) to write typed Schema -> buffer via bit_cast
 *
 *  This design means:
 *    - CoordinateTransform is a structural NTTP (aggregate, defaulted ==)
 *    - Adding a new transform does NOT modify this struct
 *    - Field names are self-documenting per-transform (e.g., PadData::left_pad)
 *    - No field interpretation table needed
 *
 *  WARNING: Do NOT replace static_array with ck_tile::array or std::array.
 *  - ck_tile::array has user-provided constructors -> not a structural type -> breaks NTTP.
 *  - std::array requires #include <array> -> breaks hipRTC.
 *  - static_array is an aggregate, no stdlib, device-compatible, structural with ==.
 */

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/static_array.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/experimental/core/tensor/tensor_descriptor.hpp" // MAX_TENSOR_DIMS
#include "ck_tile/experimental/core/tensor/transform_type.hpp"
#include "ck_tile/experimental/core/tensor/magic_division.hpp"

namespace ck_tile {

/// Alias for dimension index arrays used in graph routing.
using DimIds = static_array<index_t, MAX_TENSOR_DIMS>;

/// Size of the opaque data buffer in CoordinateTransform.
/// Must accommodate the largest Schema: Embed with MAX_TENSOR_DIMS (64) dimensions
/// requires 64 dim_lengths (256B) + 64 strides (256B) + 64 magic_divs (512B) = 1024B.
inline constexpr index_t MAX_TRANSFORM_DATA_SIZE = 1032;

/// Type alias for the opaque data buffer.
using TransformDataBuffer = static_array<uint8_t, MAX_TRANSFORM_DATA_SIZE>;

/** @brief Pure data describing a single coordinate mapping.
 *
 *  The data buffer is opaque — each TransformImpl<Type> defines a typed
 *  Schema struct with named fields. The Schema is the same
 *  size as the buffer (padded with a private _pad member). Conversion
 *  between buffer and schema uses constexpr bit_cast.
 *
 *  Example usage in a TransformImpl:
 *    auto d = readSchema(t);        // bit_cast<Schema>(t.data) — typed view
 *    output[0] = d.left_pad;     // self-documenting field access
 *
 *  Unused buffer bytes MUST be zero-initialized for NTTP deduplication.
 *  Factory functions guarantee this via schema default initialization.
 */
struct CoordinateTransform
{
    TransformType type     = TransformType::UNDEFINED;
    index_t ndim_input     = 0; ///< Dims on the user side (top of stack)
    index_t ndim_output    = 0; ///< Dims on the memory side (bottom of stack)
    bool skip_bounds_check = false;
    bool is_bijective      = false;

    TransformDataBuffer data{}; ///< Opaque — interpreted by TransformImpl via schema

    constexpr bool operator==(const CoordinateTransform&) const = default;
};

} // namespace ck_tile
