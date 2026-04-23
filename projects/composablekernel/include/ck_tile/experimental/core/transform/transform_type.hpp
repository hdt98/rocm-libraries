// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transform_type.hpp
 *  @brief Enumeration of coordinate transform types for value-based transform graphs.
 */

#pragma once

#include <stdint.h>

namespace ck_tile::core::transform {

/** @brief Identifies the coordinate mapping algorithm a transform performs.
 *
 *  Each value corresponds to a TransformImpl specialization that provides
 *  the actual mapping logic. See transforms/ directory for implementations.
 *
 *  To add a new transform type:
 *    1. Add the enum value here
 *    2. Create transforms/new_type.hpp with TransformImpl<NEW_TYPE>
 *    3. Include it in transform_impl.hpp
 *    4. Add a factory function in make_transform.hpp
 */
enum struct TransformType : uint8_t
{
    PASS_THROUGH, ///< Identity: output[0] = input[0]
    PAD,          ///< Shifted with optional bounds: output = input - left_pad
    EMBED,        ///< Linear combination: output = sum(input[i] * coefficient[i])
    MERGE,        ///< N inputs -> 1 output (flattening: output = sum(input[i] * stride[i]))
    UNMERGE,      ///< 1 input -> N outputs (decomposition via magic division)
    XOR,          ///< 2D XOR permutation for LDS bank conflict avoidance
    OFFSET,       ///< Constant shift: output = input + offset_value
    FREEZE,       ///< Fixes a dimension to a compile-time constant
    INSERT,       ///< Adds a new dimension with no memory mapping
    REPLICATE,    ///< Broadcasts: N outputs with no inputs
    SLICE,        ///< Subrange: output = input within [begin, end)
    MODULO,       ///< Modular arithmetic: output = input % modulus
    UNDEFINED     ///< Sentinel for unused array slots
};

} // namespace ck_tile::core::transform
