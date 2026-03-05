#pragma once
#include <mint/core/index_t.h>
#include <type_traits>

namespace mint {

using std::bool_constant;
using std::integral_constant;

template <index_t kI>
using index_constant = integral_constant<index_t, kI>;

// Helper to convert character digits (Decimal) to index_constant
template <char... kDigits>
MINT_HOST_DEVICE consteval auto operator""_ic() {
  constexpr auto to_integer = []() {
    index_t value = 0;
    for (char digit : {kDigits...})
      value = value * 10 + (digit - '0');
    return value;
  };
  return index_constant<to_integer()>{};
}

template <index_t kX, index_t kY>
MINT_HOST_DEVICE constexpr index_constant<kX + kY> operator+(
    index_constant<kX>,
    index_constant<kY>) {
  return {};
}

template <index_t kX, index_t kY>
MINT_HOST_DEVICE constexpr index_constant<kX - kY> operator-(
    index_constant<kX>,
    index_constant<kY>) {
  return {};
}

} // namespace mint
