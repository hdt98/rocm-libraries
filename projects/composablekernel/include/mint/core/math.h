#pragma once
#include "mint/core/arithmetic_type.h"
#include "mint/core/is_mint_arithmetic.h"
#include "mint/core/type_traits.h"

namespace mint {
namespace math {

template <class T>
  requires is_mint_arithmetic_v<T>
constexpr T max(const T& a, const T& b) {
  return a >= b ? a : b;
}

template <class T>
  requires is_mint_arithmetic_v<T>
constexpr T abs(const T& a) {
  return a >= 0 ? a : -a;
}

template <class T>
  requires is_mint_arithmetic_v<T>
constexpr T min(const T& a, const T& b) {
  return a < b ? a : b;
}

template <class T>
  requires(is_integral_v<T>)
MINT_HOST_DEVICE constexpr T integer_divide_ceiling(T dividend, T divisor) {
  return (dividend - T{1}) / divisor + T{1};
}

// FIXME: safe to use __builtin_clz(uint32_t) fpr int32_t type?
template <class T>
  requires(is_integral_v<T>)
MINT_HOST_DEVICE constexpr T clz(T x) {
#if MINT_HACK_HAS_CONSTEXPR_CLZ
#if defined(__CUDA_ARCH__)
  if constexpr (is_same_v<T, int32_t>)
    return __clz(x);
  else
    static_assert(false, "wrong! not supported");
#elif 1
  return std::countl_zero(x);
#else
  for (index_t i = 31; i >= 0; --i) {
    if ((1 << i) & x)
      return T(31 - i);
  }
  return 32;
#endif
#else // #if MINT_HACK_HAS_CONSTEXPR_CLZ
#if 1
  if constexpr (is_same_v<T, int32_t>)
    // FIXME: safe to use __builtin_clz(uint32_t) fpr int32_t type?
    return __builtin_clz(x);
  else
    static_assert(dependent_false<T>{}, "wrong! not supported");
#else
  for (index_t i = 31; i >= 0; --i) {
    if ((1 << i) & x)
      return T(31 - i);
  }
  return 32;
#endif
#endif // #if MINT_HACK_HAS_CONSTEXPR_CLZ
}

template <typename T>
  requires(is_integral_v<T>)
MINT_HOST_DEVICE constexpr T integer_log2_floor(T x) {
  return int32_t(31 - clz(x));
}

template <typename T>
  requires(is_integral_v<T>)
MINT_HOST_DEVICE constexpr T integer_log2_ceiling(T x) {
  int32_t a = int32_t(31 - clz(x));
  a += (x & (x - 1)) != 0; // Round up, add 1 if not a power of 2.
  return a;
}

template <typename T>
  requires(is_integral_v<T>)
MINT_HOST_DEVICE constexpr T is_power_of_2_integer(T x) {
  return x == (1 << int32_t(31 - clz(x)));
}

struct fast_div_mod {
  int32_t divisor;
  uint32_t multiplier;
  uint32_t shift_right;

  constexpr fast_div_mod() = default;

  MINT_HOST_DEVICE constexpr fast_div_mod(int div) : divisor(div) {
    if (div >= 1) {
      shift_right = integer_log2_ceiling(div);
      uint64_t one = 1;
      multiplier =
          ((one << 32) * ((one << shift_right) - div)) / div + 1;
    } else {
      multiplier = 0;
      shift_right = 0;
    }
  }

  /// For int32_t input
  MINT_HOST_DEVICE constexpr void
  fast_divmod(int32_t& quotient, int32_t& remainder, int32_t dividend) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (__builtin_is_constant_evaluated()) {
      quotient = int32_t((static_cast<int64_t>(dividend) * multiplier) >> 32);
    } else {
      quotient = __umulhi(dividend, multiplier);
    }
#else // #if defined(__CUDA_ARCH__)
    quotient = int32_t((static_cast<int64_t>(dividend) * multiplier) >> 32);
#endif // #if defined(__CUDA_ARCH__)
    quotient = (quotient + dividend) >> shift_right;
    remainder = dividend - (quotient * divisor);
  }

  /// For int64_t input
  MINT_HOST_DEVICE constexpr void
  fast_divmod(int32_t& quotient, int64_t& remainder, int64_t dividend) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (__builtin_is_constant_evaluated()) {
      quotient = ((dividend * multiplier) >> 32);
    } else {
      quotient = __umulhi(dividend, multiplier);
    }
#else // #if defined(__CUDA_ARCH__)
    quotient = ((dividend * multiplier) >> 32);
#endif // #if defined(__CUDA_ARCH__)
    quotient = (quotient + dividend) >> shift_right;
    remainder = dividend - (quotient * divisor);
  }

  MINT_HOST_DEVICE constexpr void
  operator()(int32_t& quotient, int32_t& remainder, int32_t dividend) const {
    fast_divmod(quotient, remainder, dividend);
  }

  constexpr bool operator==(const fast_div_mod&) const = default;
};

} // namespace math
} // namespace mint
