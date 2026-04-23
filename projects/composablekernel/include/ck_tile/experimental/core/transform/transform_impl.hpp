// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transform_impl.hpp
 *  @brief Static dispatch interface for coordinate transform algorithms.
 *
 *  Each TransformType has a TransformImpl specialization providing:
 *  - mapIndices(): the forward index mapping (traversal direction)
 *  - reverseMapIndices(): the inverse index mapping (reverse traversal)
 *  - isValidInput(): bounds checking
 *  - inputLength(): query input dimension length
 *
 *  Specializations are in separate files under transforms/.
 *  The primary template uses = delete to produce a clear compile error
 *  if an unimplemented transform type is used.
 *
 *  To add a new transform type:
 *    1. Add enum value to TransformType (in transform_type.hpp)
 *    2. Create transforms/new_type.hpp with TransformImpl<NEW_TYPE>
 *    3. #include it at the bottom of this file
 *    4. Add factory function in make_transform.hpp
 *    No existing code modified (open/closed principle).
 *
 *  @tparam Type  The transform type (compile-time constant, typically from an NTTP)
 */

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/experimental/core/transform/coordinate_transform.hpp"

namespace ck_tile::core::transform::detail {

// Primary template — intentionally deleted to force specialization. Using
// an unimplemented TransformType value produces a clear compile error
// pointing at the missing specialization.
template <TransformType Type>
struct TransformImpl
{
    static CK_TILE_HOST_DEVICE constexpr void
    mapIndices(const CoordinateTransform& t, index_t* output, const index_t* input) = delete;

    static CK_TILE_HOST_DEVICE constexpr void
    reverseMapIndices(const CoordinateTransform& t, index_t* input, const index_t* output) = delete;

    static CK_TILE_HOST_DEVICE constexpr bool isValidInput(const CoordinateTransform& t,
                                                           const index_t* input) = delete;

    static CK_TILE_HOST_DEVICE constexpr index_t inputLength(const CoordinateTransform& t,
                                                             index_t i) = delete;
};

} // namespace ck_tile::core::transform::detail

// Include all specializations.
// To add a new transform, add a #include here and create the corresponding file.
#include "ck_tile/experimental/core/transform/impl/pass_through.hpp"
#include "ck_tile/experimental/core/transform/impl/embed.hpp"
#include "ck_tile/experimental/core/transform/impl/merge.hpp"
#include "ck_tile/experimental/core/transform/impl/unmerge.hpp"
#include "ck_tile/experimental/core/transform/impl/pad.hpp"
#include "ck_tile/experimental/core/transform/impl/xor.hpp"
