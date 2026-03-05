#pragma once
#include <mint/core/type_traits.h>

namespace mint {

template <class Y, class X>
  requires(sizeof(Y) == sizeof(X))
MINT_HOST_DEVICE auto as_type(X& x) -> Y& {
  return *reinterpret_cast<Y*>(&x);
}

template <class Y, class X>
  requires(sizeof(Y) == sizeof(X))
MINT_HOST_DEVICE auto as_type(const X& x) -> const Y& {
  return *reinterpret_cast<const Y*>(&x);
}

} // namespace mint
