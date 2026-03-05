#pragma once
#include <mint/config.h>
#include <mint/core/array.h>
#include <mint/core/index_t.h>
#include <mint/core/initializer_list.h>

namespace mint {

template <class T, index_t kN, index_t... kNs>
struct nd_array {
  using sub_nd_array_type = nd_array<T, kNs...>;

 public:
  using value_type = T;

  constexpr nd_array() = default;

  MINT_HOST_DEVICE constexpr nd_array(initializer_list<sub_nd_array_type> init)
      : data_{} {
    if constexpr (kN > 0)
      std::copy(init.begin(), init.end(), data_.begin());
  }

  constexpr bool operator==(const nd_array&) const = default;

  MINT_HOST_DEVICE static constexpr index_t ndim() {
    return sizeof...(kNs) + 1;
  }

  MINT_HOST_DEVICE static constexpr auto lengths()
      -> array<index_t, sizeof...(kNs) + 1> {
    return {kN, kNs...};
  }

  template <class... Ts>
    requires(sizeof...(Ts) == sizeof...(kNs))
  MINT_HOST_DEVICE constexpr const T& at(index_t i, Ts... is) const {
    return data_.at(i).at(is...);
  }

  template <class... Ts>
    requires(sizeof...(Ts) == sizeof...(kNs))
  MINT_HOST_DEVICE constexpr T& at(index_t i, Ts... is) {
    return data_.at(i).at(is...);
  }

  template <class... Ts>
    requires(sizeof...(Ts) == sizeof...(kNs))
  MINT_HOST_DEVICE constexpr const T& operator()(index_t i, Ts... is) const {
    return data_.at(i).at(is...);
  }

  template <class... Ts>
    requires(sizeof...(Ts) == sizeof...(kNs))
  MINT_HOST_DEVICE constexpr T& operator()(index_t i, Ts... is) {
    return data_.at(i).at(is...);
  }

  MINT_HOST_DEVICE constexpr auto operator[](index_t i) const
      -> const sub_nd_array_type& {
    return data_[i];
  }

  MINT_HOST_DEVICE constexpr auto operator[](index_t i) -> sub_nd_array_type& {
    return data_[i];
  }

  MINT_HOST_DEVICE constexpr void fill(const T& v) {
    for (index_t i = 0; i < kN; i++)
      data_.at(i).fill(v);
  }

  MINT_HOST_DEVICE void print() const {
    //  printf("nd_array {");
    //  printf("lengths: ");
    //  lengths().print();
    //  printf("data: ");
    data_.print();
    //  printf("}");
  }

  array<sub_nd_array_type, kN> data_;
};

template <class T, index_t kN>
struct nd_array<T, kN> {
  using value_type = T;

  constexpr nd_array() = default;

  MINT_HOST_DEVICE constexpr nd_array(initializer_list<T> init) : data_{} {
    std::copy(init.begin(), init.end(), data_.begin());
  }

  constexpr bool operator==(const nd_array&) const = default;

  MINT_HOST_DEVICE static constexpr auto lengths() -> array<index_t, 1> {
    return {kN};
  }

  MINT_HOST_DEVICE constexpr const T& at(index_t i) const {
    return data_.at(i);
  }

  MINT_HOST_DEVICE constexpr T& at(index_t i) {
    return data_.at(i);
  }

  MINT_HOST_DEVICE constexpr const T& operator[](index_t i) const {
    return data_[i];
  }

  MINT_HOST_DEVICE constexpr T& operator[](index_t i) {
    return data_[i];
  }

  MINT_HOST_DEVICE constexpr void fill(const T& v) {
    data_.fill(v);
  }

  MINT_HOST_DEVICE void print() const {
    //  printf("nd_array {");
    //  printf("lengths: ");
    //  lengths().print();
    //  printf("data: ");
    data_.print();
    //  printf("}");
  }

  array<T, kN> data_;
};

} // namespace mint
