#pragma once
#include <mint/config.h>
#include <mint/core/array.h>

namespace mint {

template <index_t kN>
using nd_index = array<index_t, kN>;

template <index_t kN>
using ref_nd_index = ref_array<index_t, kN>;

template <index_t kN>
MINT_HOST_DEVICE constexpr nd_index<kN> operator+(
    const nd_index<kN>& lhs,
    const nd_index<kN>& rhs) {
  nd_index<kN> ret;
  for (index_t i = 0; i < kN; i++) {
    ret[i] = lhs[i] + rhs[i];
  }
  return ret;
}

template <index_t kN>
MINT_HOST_DEVICE constexpr nd_index<kN> operator-(
    const nd_index<kN>& lhs,
    const nd_index<kN>& rhs) {
  nd_index<kN> ret;
  for (index_t i = 0; i < kN; i++) {
    ret[i] = lhs[i] - rhs[i];
  }
  return ret;
}

template <index_t kN>
MINT_HOST_DEVICE constexpr nd_index<kN> operator*(
    const nd_index<kN>& lhs,
    const nd_index<kN>& rhs) {
  nd_index<kN> ret;
  for (index_t i = 0; i < kN; i++) {
    ret[i] = lhs[i] * rhs[i];
  }
  return ret;
}

template <index_t kN>
MINT_HOST_DEVICE constexpr nd_index<kN> operator/(
    const nd_index<kN>& lhs,
    const nd_index<kN>& rhs) {
  nd_index<kN> ret;
  for (index_t i = 0; i < kN; i++) {
    ret[i] = lhs[i] / rhs[i];
  }
  return ret;
}

template <index_t kN>
MINT_HOST_DEVICE constexpr const ref_nd_index<kN>& operator+=(
    const ref_nd_index<kN>& lhs,
    const nd_index<kN>& rhs) {
  for (index_t i = 0; i < kN; i++) {
    lhs[i] += rhs[i];
  }
  return lhs;
}

template <index_t kN, class Fn, index_t empty = 0>
MINT_HOST_DEVICE constexpr index_t nd_reduce(const nd_index<kN>& nd, Fn f) {
  if (kN == 0) {
    return empty;
  } else {
    index_t ret = nd[0];
    for (index_t i = 1; i < kN; i++) {
      ret = f(ret, nd[i]);
    }
    return ret;
  }
}

} // namespace mint
