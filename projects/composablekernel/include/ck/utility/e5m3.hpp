// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef CK_CODE_GEN_RTC

#include "ck/utility/scale_utils.hpp"
#include "ck/utility/type.hpp"

namespace ck {

struct e5m3_scale_t
{
    using type   = uint8_t;
    using Format = utils::ScaleFormat<5, 3>;

    static constexpr int exponent_bits = Format::exponent_bits;
    static constexpr int mantissa_bits = Format::mantissa_bits;
    static constexpr type value_mask   = Format::value_mask;
    static constexpr type nan_mask     = Format::nan_mask;
    static constexpr type max_finite   = Format::max_finite;
    static constexpr int bias          = Format::bias;

    type data;

    __host__ __device__ constexpr e5m3_scale_t() : data{type{}} {}
    __host__ __device__ constexpr explicit e5m3_scale_t(type init)
        : data{static_cast<type>(init & value_mask)}
    {
    }
    __host__ __device__ constexpr explicit e5m3_scale_t(int init)
        : data{static_cast<type>(static_cast<type>(init) & value_mask)}
    {
    }
    __host__ __device__ explicit e5m3_scale_t(float scale) : data{Format::encode(scale)} {}

    __host__ __device__ explicit operator float() const { return Format::decode(data); }

    __host__ __device__ constexpr bool operator==(const e5m3_scale_t& other) const
    {
        return data == other.data && !is_nan();
    }

    __host__ __device__ constexpr bool operator!=(const e5m3_scale_t& other) const
    {
        return !(*this == other);
    }

    __host__ __device__ constexpr bool is_nan() const { return Format::is_nan(data); }
};

namespace utils {

template <>
__host__ __device__ inline constexpr int32_t get_exponent_value<e5m3_scale_t>(e5m3_scale_t x)
{
    return e5m3_scale_t::Format::exponent(x.data);
}

} // namespace utils

} // namespace ck

#endif
