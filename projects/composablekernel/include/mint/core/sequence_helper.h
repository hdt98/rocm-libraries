#pragma once
#include <mint/config.h>
#include <mint/core/integer_sequence.h>
#include <mint/core/sequence.h>

namespace mint {

namespace impl {

template <auto kArr, index_t... kIs>
MINT_HOST_DEVICE consteval auto to_sequence_impl(sequence<index_t, kIs...>) {
  using value_type = typename remove_cvref_t<decltype(kArr)>::value_type;
  return sequence<value_type, kArr[kIs]...>{};
}

} // namespace impl

// convert to sequence
//   Arr could be: array, same_tuple, sequence
template <auto kArr>
  requires(kArr.size() >= 0)
MINT_HOST_DEVICE consteval auto to_sequence() {
  return impl::to_sequence_impl<kArr>(
      make_integer_sequence<index_t, kArr.size()>{});
}

} // namespace mint
