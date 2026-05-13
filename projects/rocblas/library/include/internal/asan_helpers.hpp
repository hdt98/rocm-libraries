// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCBLAS_ASAN_HELPERS_HPP
#define ROCBLAS_ASAN_HELPERS_HPP

inline constexpr bool rocblas_enable_asan = false;

namespace rocblas
{

    template <bool B, auto IfTrue, auto IfFalse>
    inline constexpr auto conditional_v = B ? IfTrue : IfFalse;

} // namespace rocblas

#endif
