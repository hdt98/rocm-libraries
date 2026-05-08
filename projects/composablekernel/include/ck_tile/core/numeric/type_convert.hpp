// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#if CK_TILE_USE_CUSTOM_DATA_TYPE
#include "ck_tile/core/utility/type_traits.hpp"
#else
#include "ck_tile/core/utility/bit_cast.hpp"
#if defined(__gfx125__)
#include "ck_tile/core/numeric/mxfp_scale.hpp"
#endif

#include <stdint.h>
#include <type_traits>
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wno-unknown-warning-option"
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
namespace ck_tile {

#if CK_TILE_USE_CUSTOM_DATA_TYPE
template <typename Y, typename X>
CK_TILE_HOST_DEVICE constexpr remove_cvref_t<Y> type_convert(const X& x)
{
    return static_cast<Y>(x);
}
#else
// Convert X to Y, both X and Y are non-const data types.
template <typename Y,
          typename X,
          std::enable_if_t<!(std::is_const_v<Y> || std::is_const_v<X>), bool> = false>
CK_TILE_HOST_DEVICE constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);
    return static_cast<Y>(x);
}

// Convert X to Y, either X or Y is a const data type.
template <typename Y,
          typename X,
          std::enable_if_t<std::is_const_v<Y> || std::is_const_v<X>, bool> = false>
CK_TILE_HOST_DEVICE constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);

    using non_const_y = std::remove_const_t<Y>;
    using non_const_x = std::remove_const_t<X>;
    return static_cast<Y>(type_convert<non_const_y, non_const_x>(x));
}

template <typename Y, typename X>
CK_TILE_HOST_DEVICE constexpr Y scaled_type_convert(X x, float scale);

#if defined(__gfx125__)
// Declare a template function for wave-wise scaled conversion
/* scale is packed 4 form, see details for FP8/BF8, FP4, FP6 */
template <typename Y, typename X, int Scale_sel>
struct pk4scaled_type_convert_impl
{
    CK_TILE_DEVICE static constexpr Y run(X x, Packed4Scale_E8M0 scale);
};

template <typename Y, typename X, int Scale_sel = 0>
CK_TILE_DEVICE constexpr Y pk4scaled_type_convert(X x, Packed4Scale_E8M0 scale)
{
    return pk4scaled_type_convert_impl<Y, X, Scale_sel>::run(x, scale);
}

// pk6scaled_type_convert for FP6 E2M3 and BF6 E3M2
template <typename Y, typename X, int Scale_sel>
struct pk6scaled_type_convert_impl
{
    CK_TILE_DEVICE static constexpr Y run(X x, Packed4Scale_E8M0 scale);
};

template <typename Y, typename X, int Scale_sel = 0>
CK_TILE_DEVICE constexpr Y pk6scaled_type_convert(X x, Packed4Scale_E8M0 scale)
{
    return pk6scaled_type_convert_impl<Y, X, Scale_sel>::run(x, scale);
}

#endif

[[deprecated("Use (ck_tile::numeric_traits<ck_tile::tf32_t>::exp_mask << 23) "
             "instead")]] static constexpr uint32_t float32_exponent_mask = 0x7f800000u;

enum class [[deprecated]] tf32_rounding_mode
{
    trunc = 0, // truncate
    rne   = 1, // round to nearest even (RTNE)
};

template <tf32_rounding_mode rounding>
[[deprecated(
    "Use ck_tile::type_convert<ck_tile::tf32_t> instead")]] CK_TILE_HOST_DEVICE constexpr float
float_to_tf32(float x)
{
    uint32_t i = bit_cast<uint32_t>(x);
    if constexpr(rounding == tf32_rounding_mode::rne)
    {
        // RTNE rounding.
        if((i & float32_exponent_mask) != float32_exponent_mask)
        {
            // Add rounding bias for round-to-nearest-even (RTNE) before truncating:
            //  - 0xfff is the rounding bias corresponding to the 13 fraction bits that
            //    will be discarded.
            //  - (i >> 13) & 1 extracts the least significant of those discarded bits and
            //    adding it implements "ties to even" (round half-way cases to even).
            i += 0xfff + ((i >> 13) & 1);
        }
    }
    // Zero out the lowest 13 fraction bits to form the TF32-like value.
    i &= 0xFFFFE000u;
    return bit_cast<float>(i);
}

class tfloat32_t; // TODO: remove when the below function is removed

template <typename Y,
          tf32_rounding_mode rounding,
          std::enable_if_t<std::is_same_v<Y, tfloat32_t>, bool> = false>
[[deprecated(
    "Use ck_tile::type_convert<ck_tile::tf32_t> instead")]] CK_TILE_HOST_DEVICE constexpr float
type_convert(float x)
{
    return float_to_tf32<rounding>(x);
}

#endif

} // namespace ck_tile
#pragma clang diagnostic pop
