// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file coordinate_transform.hpp
 *  @brief Pure data struct describing a single coordinate transform.
 *
 *  CoordinateTransform carries ONLY data — no dispatch logic, no mapping functions.
 *  Behavior is provided by TransformImpl<Type> specializations (static polymorphism).
 *
 *  This separation means:
 *  - CoordinateTransform is a structural NTTP (pure aggregate, defaulted ==)
 *  - Adding a new transform type does NOT modify this struct
 *  - Each TransformImpl<Type> is self-contained in its own file
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
#include "ck_tile/core/tensor/value_based/transform_type.hpp"
#include "ck_tile/core/tensor/value_based/magic_division.hpp"

namespace ck_tile {

/// Maximum dimensions per transform. Observed max in ck_tile: 5 (3D conv merge).
inline constexpr index_t MAX_DIMS_PER_TRANSFORM = 5;

/// Alias for dimension index arrays used in graph routing.
using DimIds = static_array<index_t, MAX_DIMS_PER_TRANSFORM>;

/** @brief Pure data describing a single coordinate mapping.
 *
 *  Each CoordinateTransform describes WHAT a transform does (its parameters),
 *  not HOW it does it (that is in TransformImpl<Type>).
 *
 *  Field interpretation by TransformType:
 *
 *  | Type          | lengths[]                | coefficients[]           | magic_divs[]   |
 *  |---------------|--------------------------|--------------------------|----------------|
 *  | PASS_THROUGH  | [0]=dim_length           | --                       | --             |
 *  | PAD           | [0]=unpadded_length      | [0]=left_pad,[1]=right   | --             |
 *  | EMBED         | dim_lengths[0..N-1]      | strides[0..N-1]          | --             |
 *  | MERGE         | component_lengths[0..N-1]| divisors[0..N-2]         | [0..N-2]       |
 *  | UNMERGE       | component_lengths[0..N-1]| strides[0..N-1] (prefix) | --             |
 *  | XOR           | [0],[1]=dim_lengths      | --                       | --             |
 *  | OFFSET        | [0]=dim_length           | [0]=offset_value         | --             |
 *  | FREEZE        | [0]=frozen_index         | --                       | --             |
 *  | SLICE         | [0]=unsliced_length      | [0]=begin,[1]=end        | --             |
 *  | MODULO        | [0]=modulus              | [0]=dim_length           | --             |
 *
 *  Note on MERGE vs UNMERGE (direction matches v1's upper/lower convention):
 *  - MERGE: ndim_input=1 (user-facing merged value), ndim_output=N (memory-side components).
 *    During traversal: DECOMPOSES 1 merged value into N components via magic division.
 *    lengths = component sizes, coefficients = divisors, magic_divs = pre-computed constants.
 *  - UNMERGE: ndim_input=N (user-facing components), ndim_output=1 (memory-side flat value).
 *    During traversal: COMPOSES N components into 1 flat value via multiply-accumulate.
 *    lengths = component sizes, coefficients = strides (prefix products).
 *
 *  Unused array slots MUST be zero-initialized for NTTP deduplication.
 *  Factory functions (make_pass_through, make_merge, etc.) guarantee this.
 */
struct CoordinateTransform
{
    TransformType type  = TransformType::UNDEFINED;
    index_t ndim_input  = 0; ///< Number of input dimensions (consumed by this transform)
    index_t ndim_output = 0; ///< Number of output dimensions (produced by this transform)

    static_array<index_t, MAX_DIMS_PER_TRANSFORM> lengths{};
    static_array<index_t, MAX_DIMS_PER_TRANSFORM> coefficients{};
    static_array<MagicDivConstants, MAX_DIMS_PER_TRANSFORM> magic_divs{};
    bool skip_bounds_check = false;

    // Pure data — no member functions for dispatch.
    // Behavior is in TransformImpl<Type> specializations.

    constexpr bool operator==(const CoordinateTransform&) const = default;
};

} // namespace ck_tile
