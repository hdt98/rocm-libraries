// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_DRIVER_UTILS_HPP
#define GUARD_MIOPEN_DRIVER_UTILS_HPP

#include <cmath>
#include <limits>
#include <type_traits>

// Driver-local replacements for simple internal MIOpen utility functions.

namespace driver_utils {

// ULP-based floating-point equality (replaces miopen::float_equal).
template <class T, class U>
bool float_equal(T x, U y)
{
    using C = std::common_type_t<T, U>;
    auto cx = static_cast<C>(x);
    auto cy = static_cast<C>(y);
    return std::isfinite(static_cast<double>(cx)) && std::isfinite(static_cast<double>(cy)) &&
           std::nextafter(cx, std::numeric_limits<C>::lowest()) <= cy &&
           std::nextafter(cx, std::numeric_limits<C>::max()) >= cy;
}

// Exact floating-point equality for sentinel values (replaces miopen::float_equal_sentinel).
template <class T, class U>
bool float_equal_sentinel(T x, U y)
{
    using C = std::common_type_t<T, U>;
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#endif
    return static_cast<C>(x) == static_cast<C>(y);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
}

} // namespace driver_utils

#endif // GUARD_MIOPEN_DRIVER_UTILS_HPP
