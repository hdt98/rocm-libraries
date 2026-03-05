#pragma once
#include <mint/config.h>
#include <mint/core/array.h>
#include <mint/core/integer_sequence.h>
#include <mint/core/same_tuple.h>
#include <mint/core/sequence.h>
#include <mint/core/static_map.h>

namespace mint {

template <class T, index_t kSize>
MINT_HOST_DEVICE constexpr auto array_to_static_map(
    const array<T, kSize>& arr) {
  static_map<T, index_t, kSize> ret;
  for (index_t i = 0; i < kSize; i++)
    ret[arr[i]] = i;
  return ret;
}

} // namespace mint
