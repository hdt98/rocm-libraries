#pragma once
#include <mint/core/integral_constant.h>
#include <mint/core/type_traits.h>
#include <type_traits>

namespace mint {

#if 1
// FIXME: std::integral_constant is only implemented for integral,
//   need a general-purpose constant wrapper for any type and value
template <auto kValue>
using constant = integral_constant<remove_cvref_t<decltype(kValue)>, kValue>;
#else
// General-purpose constant wrapper for any type and value
template <class T, T v>
struct value_constant {
  using value_type = T;
  static constexpr value_type value = v;
  using type = value_constant;
  constexpr operator value_type() const noexcept {
    return value;
  }
};

template <auto kValue>
using constant = value_constant<std::remove_cvref_t<decltype(kValue)>, kValue>;
#endif

namespace impl {

template <class T>
struct is_constant_impl {
  static constexpr bool value = false;
};

template <auto kValue>
struct is_constant_impl<constant<kValue>> {
  static constexpr bool value = true;
};

} // namespace impl

template <class T>
using is_constant = impl::is_constant_impl<remove_cvref_t<T>>;

template <class T>
constexpr bool is_constant_v = is_constant<T>::value;

} // namespace mint
