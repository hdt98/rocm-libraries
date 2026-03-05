#pragma once
#include <mint/config.h>
#include <mint/core/index_t.h>
#include <mint/core/print.h>
#include <mint/core/reference_wrapper.h>
#include <mint/core/type_traits.h>

namespace mint {

template <class T, index_t kN>
struct ref_array {
  using value_type = remove_cvref_t<T>;

  reference_wrapper<value_type> data_ref_[kN];

  constexpr ref_array() = default;

  template <class... Xs>
    requires(sizeof...(Xs) == kN)
  MINT_HOST_DEVICE constexpr ref_array(Xs&&... xs) : data_ref_{xs...} {}

  MINT_HOST_DEVICE static consteval index_t size() {
    return kN;
  }

  MINT_HOST_DEVICE constexpr value_type& at(index_t i) const {
    return data_ref_[i].get();
  }

  MINT_HOST_DEVICE constexpr value_type& at(index_t i) {
    return data_ref_[i].get();
  }

  MINT_HOST_DEVICE constexpr value_type& operator[](index_t i) const {
    return data_ref_[i].get();
  }

  MINT_HOST_DEVICE constexpr value_type& operator[](index_t i) {
    return data_ref_[i].get();
  }

  MINT_HOST_DEVICE constexpr void fill(const value_type& v) {
    for (index_t i = 0; i < kN; i++)
      data_ref_[i].get() = v;
  }

  // impl for subset_reference
  template <index_t kM, array<index_t, kM> kSubset, index_t... kIs>
    requires(kM == sizeof...(kIs) && kM <= size())
  MINT_HOST_DEVICE constexpr auto subset_reference_impl(index_sequence<kIs...>)
      const -> const ref_array<value_type, kSubset.size()> {
    return {(*this)[kSubset[kIs]]...};
  }

  // impl for subset_reference
  template <index_t kM, array<index_t, kM> kSubset, index_t... kIs>
    requires(kM == sizeof...(kIs) && kM <= size())
  MINT_HOST_DEVICE constexpr auto subset_reference_impl(index_sequence<kIs...>)
      -> ref_array<value_type, kSubset.size()> {
    return {(*this)[kSubset[kIs]]...};
  }

  template <index_t kM, array<index_t, kM> kSubset>
    requires(kM <= size())
  MINT_HOST_DEVICE constexpr auto subset_reference() const
      -> const ref_array<value_type, kM> {
    return subset_reference_impl<kM, kSubset>(make_index_sequence<kM>{});
  }

  template <index_t kM, array<index_t, kM> kSubset>
    requires(kM <= size())
  MINT_HOST_DEVICE constexpr auto subset_reference()
      -> ref_array<value_type, kM> {
    return subset_reference_impl<kM, kSubset>(make_index_sequence<kM>{});
  }

  MINT_HOST_DEVICE void print() const {
    printf("ref_array {size %d, data[", size());
    for (index_t i = 0; i < kN; i++) {
      print_item((*this)[i]);
      printf(", ");
    }
    printf("]}");
  }
};

template <class T, index_t kN>
struct const_ref_array {
  using value_type = remove_cvref_t<T>;

  reference_wrapper<const value_type> data_ref_[kN];

  constexpr const_ref_array() = default;

  template <class... Xs>
    requires(sizeof...(Xs) == kN)
  MINT_HOST_DEVICE constexpr const_ref_array(Xs&&... xs) : data_ref_{xs...} {}

  MINT_HOST_DEVICE static consteval index_t size() {
    return kN;
  }

  MINT_HOST_DEVICE constexpr const value_type& at(index_t i) const {
    return data_ref_[i].get();
  }

  MINT_HOST_DEVICE constexpr const value_type& at(index_t i) {
    return data_ref_[i].get();
  }

  MINT_HOST_DEVICE constexpr const value_type& operator[](index_t i) const {
    return data_ref_[i].get();
  }

  MINT_HOST_DEVICE constexpr const value_type& operator[](index_t i) {
    return data_ref_[i].get();
  }

  // impl for subset_reference
  template <auto kSubset, index_t... kIs>
    requires(
        is_same_v<typename decltype(kSubset)::value_type, index_t> &&
        kSubset.size() == sizeof...(kIs) && kSubset.size() <= size())
  MINT_HOST_DEVICE constexpr auto subset_reference_impl(index_sequence<kIs...>)
      const -> const const_ref_array<value_type, kSubset.size()> {
    return {(*this)[kSubset[kIs]]...};
  }

  // impl for subset_reference
  template <auto kSubset, index_t... kIs>
    requires(
        is_same_v<typename decltype(kSubset)::value_type, index_t> &&
        kSubset.size() == sizeof...(kIs) && kSubset.size() <= size())
  MINT_HOST_DEVICE constexpr auto subset_reference_impl(index_sequence<kIs...>)
      -> const_ref_array<value_type, kSubset.size()> {
    return {(*this)[kSubset[kIs]]...};
  }

  template <auto kSubset>
    requires(
        is_same_v<typename decltype(kSubset)::value_type, index_t> &&
        kSubset.size() <= size())
  MINT_HOST_DEVICE constexpr auto subset_reference() const
      -> const const_ref_array<value_type, kSubset.size()> {
    return subset_reference_impl<kSubset>(
        make_index_sequence<kSubset.size()>{});
  }

  template <auto kSubset>
    requires(
        is_same_v<typename decltype(kSubset)::value_type, index_t> &&
        kSubset.size() <= size())
  MINT_HOST_DEVICE constexpr auto subset_reference()
      -> const_ref_array<value_type, kSubset.size()> {
    return subset_reference_impl<kSubset>(
        make_index_sequence<kSubset.size()>{});
  }

  MINT_HOST_DEVICE void print() const {
    printf("const_ref_array {size %d, data[", size());
    for (index_t i = 0; i < kN; i++) {
      print_item((*this)[i]);
      printf(", ");
    }
    printf("]}");
  }
};

template <class T>
struct ref_array<T, 0> {
  using value_type = remove_cvref_t<T>;

  constexpr ref_array() = default;

  MINT_HOST_DEVICE static consteval index_t size() {
    return 0;
  }

  MINT_HOST_DEVICE constexpr value_type& at(index_t) const {
    return value_type{};
  }

  MINT_HOST_DEVICE constexpr value_type& at(index_t) {
    return value_type{};
  }

  MINT_HOST_DEVICE constexpr value_type& operator[](index_t) const {
    return value_type{};
  }

  MINT_HOST_DEVICE constexpr value_type& operator[](index_t) {
    return value_type{};
  }

  MINT_HOST_DEVICE constexpr void fill(const value_type&) {}

  template <auto kSubset>
    requires(
        is_same_v<typename decltype(kSubset)::value_type, index_t> &&
        kSubset.size() == 0)
  MINT_HOST_DEVICE constexpr auto subset_reference() const
      -> const ref_array<value_type, 0> {
    return *this;
  }

  template <auto kSubset>
    requires(
        is_same_v<typename decltype(kSubset)::value_type, index_t> &&
        kSubset.size() == 0)
  MINT_HOST_DEVICE constexpr auto subset_reference()
      -> ref_array<value_type, 0> {
    return *this;
  }

  MINT_HOST_DEVICE void print() const {
    printf("ref_array {size %d, data[", size());
    printf("]}");
  }
};

template <class T>
struct const_ref_array<T, 0> {
  using value_type = remove_cvref_t<T>;

  constexpr const_ref_array() = default;

  MINT_HOST_DEVICE static consteval index_t size() {
    return 0;
  }

  MINT_HOST_DEVICE constexpr const value_type& at(index_t) const {
    return value_type{};
  }

  MINT_HOST_DEVICE constexpr const value_type& at(index_t) {
    return value_type{};
  }

  MINT_HOST_DEVICE constexpr const value_type& operator[](index_t) const {
    return value_type{};
  }

  MINT_HOST_DEVICE constexpr const value_type& operator[](index_t) {
    return value_type{};
  }

  template <auto kSubset>
    requires(
        is_same_v<typename decltype(kSubset)::value_type, index_t> &&
        kSubset.size() == 0)
  MINT_HOST_DEVICE constexpr auto subset_reference() const
      -> const const_ref_array<value_type, 0> {
    return *this;
  }

  template <auto kSubset>
    requires(
        is_same_v<typename decltype(kSubset)::value_type, index_t> &&
        kSubset.size() == 0)
  MINT_HOST_DEVICE constexpr auto subset_reference()
      -> const_ref_array<value_type, 0> {
    return *this;
  }

  MINT_HOST_DEVICE void print() const {
    printf("const_ref_array {size %d, data[", size());
    printf("]}");
  }
};
} // namespace mint
