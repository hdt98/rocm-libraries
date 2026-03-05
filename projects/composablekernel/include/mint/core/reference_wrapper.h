#pragma once
#include <functional>

namespace mint {
#if 1
using std::reference_wrapper;
#else

template <class T>
struct reference_wrapper {
  using value_type = remove_reference_t<T>;

  MINT_HOST_DEVICE constexpr reference_wrapper() = delete;

  template <class X>
  MINT_HOST_DEVICE constexpr reference_wrapper(X&& x) : r_{x} {}

  MINT_HOST_DEVICE constexpr value_type& get() const {
    return r_;
  }

  MINT_HOST_DEVICE constexpr value_type& get() {
    return r_;
  }

  MINT_HOST_DEVICE constexpr operator value_type&() const {
    return r_;
  }

  MINT_HOST_DEVICE constexpr operator value_type&() {
    return r_;
  }

  value_type& r_;
};

#endif
} // namespace mint
