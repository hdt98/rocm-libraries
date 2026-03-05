#pragma once
#include <mint/config.h>
#include <mint/core/index_t.h>
#include <mint/core/print.h>

namespace mint {

template <class T, T... kIs>
class sequence {
 public:
  using value_type = remove_cvref_t<T>;

  MINT_HOST_DEVICE static consteval index_t size() {
    return sizeof...(kIs);
  }

  MINT_HOST_DEVICE constexpr value_type operator[](index_t i) const {
    return at(i);
  }

  MINT_HOST_DEVICE static constexpr value_type at(index_t i) {
    constexpr value_type values[] = {kIs..., value_type{}};
    return values[i];
  }

  template <index_t kI>
  MINT_HOST_DEVICE static consteval value_type at() {
    constexpr value_type values[] = {kIs..., value_type{}};
    return values[kI];
  }

  MINT_HOST_DEVICE void print() const {
    printf("sequence: {");
    for (index_t i = 0; i < size(); i++) {
      print_item(at(i));
      printf(", ");
    }
    printf("}");
  }
};

} // namespace mint
