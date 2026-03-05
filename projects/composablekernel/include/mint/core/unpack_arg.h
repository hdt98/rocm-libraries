#pragma once
#include <mint/config.h>

namespace mint {

template <class T, index_t kN>
class array;

template <class T, T... kIs>
class sequence;

template <class... Ts>
struct tuple;

template <class... Ts>
MINT_HOST_DEVICE constexpr tuple<unwrap_ref_decay_t<Ts>...> make_tuple(
    Ts&&... ts);

// for function f(sequence<T, xs...>&& arr),
//   unpack_sequence_arg(f) returns a lambda g(integral_constant<T, xs>...),
//   which is equivalent to f(arr)
template <index_t kN, class T, class F>
MINT_HOST_DEVICE constexpr auto unpack_sequence_arg_as_integral_constants(
    F&& f) {
  return [&f](auto&&... xs) {
    return f(sequence<T, remove_cvref_t<decltype(xs)>{}...>{});
  };
}

// for function f(tuple<Xs...>&& arr),
//   unpack_tuple_arg(f) returns a lambda g(xs...),
//   which is equivalent to f(tuple<Xs...>{xs...})
template <index_t kN, class F>
MINT_HOST_DEVICE constexpr auto unpack_tuple_arg(F&& f) {
  return [&f](auto&&... xs) { return f(mint::make_tuple(xs...)); };
}

// for function f(array<T, kN>&& arr),
//   unpack_array_arg(f) returns a lambda g(T{xs}...),
//   which is equivalent to f(array<T, kN>{xs...})
template <index_t kN, class F>
MINT_HOST_DEVICE constexpr auto unpack_array_arg(F&& f) {
  return [&f](auto&& x, auto&&... xs) {
    using T = remove_cvref_t<decltype(x)>;
    return f(array<T, sizeof...(xs) + 1>{x, xs...});
  };
}

} // namespace mint
