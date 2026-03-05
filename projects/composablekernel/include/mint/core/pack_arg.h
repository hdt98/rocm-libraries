#pragma once
#include <mint/config.h>
#include <mint/core/integer_sequence.h>

namespace mint {

namespace impl {

template <class F, index_t... kIs>
MINT_HOST_DEVICE constexpr auto pack_arg_impl(F&& f, index_sequence<kIs...>) {
  return [&f](auto&& arg) { return f(arg.template at<kIs>()...); };
}

} // namespace impl

// for function f(Xs&&... xs),
//   pack_arg(f) returns a lambda g(auto&& arr),
//   that is equivalent to f(arr[0], arr[1], ...)
template <index_t kN, class F>
MINT_HOST_DEVICE constexpr auto pack_arg(F&& f) {
  return impl::pack_arg_impl(f, make_index_sequence<kN>{});
}

} // namespace mint
