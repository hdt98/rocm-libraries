#pragma once
#include <mint/config.h>
#include <mint/core/constant.h>
#include <mint/core/dependent_false.h>
#include <mint/core/type_traits.h>

namespace mint {

template <class T>
  requires(requires(T t) { t.print(); })
MINT_HOST_DEVICE void print_item(const T& x) {
  x.print();
}

template <class T>
  requires(!(requires(T t) { t.print(); }))
MINT_HOST_DEVICE void print_item(const T& x) {
  if constexpr (is_integral_v<T>)
    printf("%d", static_cast<int32_t>(x));
  else if constexpr (is_floating_point_v<T>)
    printf("%f", static_cast<float>(x));
  else if constexpr (is_same_v<T, char>)
    printf("%c", x);
  else
    static_assert(dependent_false<T>::value, "wrong! not supported type");
}

// TODO: remove this when contant is no longer aliased to integer_constant
template <auto kValue>
  requires(requires(remove_cvref_t<decltype(kValue)> t) { t.print(); })
MINT_HOST_DEVICE void print_item(constant<kValue>) {
  kValue.print();
}

// TODO: remove this when contant is no longer aliased to integer_constant
template <auto kValue>
  requires(!(requires(remove_cvref_t<decltype(kValue)> t) { t.print(); }))
MINT_HOST_DEVICE void print_item(constant<kValue>) {
  print_item(kValue);
}

template <class... Ts>
MINT_HOST_DEVICE void print_items(const Ts&... xs) {
  ((print_item(xs), printf(", ")), ...);
}

} // namespace mint
