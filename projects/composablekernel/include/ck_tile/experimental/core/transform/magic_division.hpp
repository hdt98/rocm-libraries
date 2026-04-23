// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file magic_division.hpp
 *  @brief Value-based magic division constants for structural NTTP usage.
 *
 *  Provides constexpr computation and CK_TILE_HOST_DEVICE execution of magic
 *  number division. Unlike ck_tile::magic_division which returns ck_tile::tuple
 *  (incompatible with structured bindings), this returns a plain aggregate
 *  suitable for storage in structural NTTP types.
 *
 *  Magic division replaces integer division (expensive on GPU) with a
 *  multiply-shift sequence using pre-computed constants. Valid for divisors
 *  in range [1, INT32_MAX] and dividends in range [0, INT32_MAX].
 */

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include <stdint.h>

namespace ck_tile::core::transform::detail {

/** @brief Pre-computed magic division constants for one divisor.
 *
 *  Structural type (aggregate, defaulted ==) — safe for use in NTTPs.
 *  Compute via computeMagicDiv(). Execute via doMagicDiv().
 */
struct MagicDivConstants
{
    uint32_t multiplier = 0;
    uint32_t shift      = 0;

    constexpr bool operator==(const MagicDivConstants&) const = default;
};

/** @brief Compute magic division constants for a given divisor at compile time.
 *
 *  @param divisor  The divisor (must be >= 1 and <= INT32_MAX)
 *  @return Pre-computed multiplier and shift for use with doMagicDiv()
 *
 *  @pre divisor >= 1 && divisor <= INT32_MAX
 */
constexpr MagicDivConstants computeMagicDiv(uint32_t divisor)
{
    uint32_t shift_val = 0;

    while((1U << shift_val) < divisor)
    {
        shift_val++;
    }

    uint64_t tmp            = static_cast<uint64_t>((1UL << shift_val) - divisor) << 32;
    uint32_t multiplier_val = static_cast<uint32_t>(tmp / divisor + 1);

    return {multiplier_val, shift_val};
}

/** @brief Perform magic division at runtime.
 *
 *  Computes dividend / divisor using pre-computed magic constants.
 *  On device, uses __umulhi intrinsic when not in constexpr context.
 *  In constexpr context, uses portable 64-bit arithmetic.
 *
 *  @param dividend  The value to divide (must be non-negative)
 *  @param md        Pre-computed magic constants from computeMagicDiv()
 *  @return dividend / original_divisor
 */
CK_TILE_HOST_DEVICE constexpr uint32_t doMagicDiv(uint32_t dividend, MagicDivConstants md)
{
#ifdef __HIP_DEVICE_COMPILE__
    if(!__builtin_is_constant_evaluated())
    {
        uint32_t tmp = __umulhi(dividend, md.multiplier);
        return (tmp + dividend) >> md.shift;
    }
#endif
    uint32_t tmp = static_cast<uint32_t>((static_cast<uint64_t>(dividend) * md.multiplier) >> 32);
    return (tmp + dividend) >> md.shift;
}

} // namespace ck_tile::core::transform::detail
