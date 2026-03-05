#pragma once
#include <mint/config.h>
#include <mint/core/for.h>
#include <mint/core/index_t.h>
#include <mint/core/initializer_list.h>
#include <mint/core/print.h>
#include <mint/core/ref_array.h>
#include <mint/core/sequence.h>
#include <mint/core/std.h>

namespace mint {

#if 1
template <class T, index_t kN>
class array : public std::array<T, kN> {
  using std::array<T, kN>::array;
  using base_type = std::array<T, kN>;

 public:
  using value_type = base_type::value_type;

  MINT_HOST_DEVICE constexpr array(initializer_list<T> init) {
#if 0
    // nvcc produce wrong result for this impl
    if constexpr (kN > 0)
      std::copy(init.begin(), init.end(), base_type::begin());
#else
    // nvcc produce correct result for this impl
    if constexpr (kN > 0) {
      auto it = init.begin();
      unroll_for_n<kN>()([&](auto i) {
        if (it != init.end()) {
          at(i) = *it;
          it++;
        }
      });
    }
#endif
  }

  MINT_HOST_DEVICE static consteval index_t size() {
    return kN;
  }

  MINT_HOST_DEVICE constexpr const T& operator[](index_t i) const {
    return *(base_type::cbegin() + i);
  }

  MINT_HOST_DEVICE constexpr T& operator[](index_t i) {
    return *(base_type::begin() + i);
  }

  MINT_HOST_DEVICE constexpr const value_type& at(index_t i) const {
    return *(base_type::cbegin() + i);
  }

  MINT_HOST_DEVICE constexpr value_type& at(index_t i) {
    return *(base_type::begin() + i);
  }

  MINT_HOST_DEVICE constexpr const T& operator()(index_t i) const {
    return *(base_type::cbegin() + i);
  }

  MINT_HOST_DEVICE constexpr T& operator()(index_t i) {
    return *(base_type::begin() + i);
  }

  template <index_t kI>
  MINT_HOST_DEVICE constexpr const value_type& at() const {
    return *(base_type::cbegin() + kI);
  }

  template <index_t kI>
  MINT_HOST_DEVICE constexpr value_type& at() {
    return *(base_type::begin() + kI);
  }

  MINT_HOST_DEVICE constexpr array& fill(const value_type& v) {
    base_type::fill(v);
    return *this;
  }

#if 1
  // impl for subset_reference
  template <index_t kM, array<index_t, kM> kSubset, index_t... kIs>
    requires(kM == sizeof...(kIs) && kM <= size())
  MINT_HOST_DEVICE constexpr auto subset_reference_impl(index_sequence<kIs...>)
      const -> const_ref_array<value_type, kSubset.size()> {
    return {(*this)[kSubset[kIs]]...};
  }

  // impl for subset_reference
  template <index_t kM, array<index_t, kM> kSubset, index_t... kIs>
    requires(kM == sizeof...(kIs) && kM <= size())
  MINT_HOST_DEVICE constexpr auto subset_reference_impl(index_sequence<kIs...>)
      -> ref_array<value_type, kM> {
    return {(*this)[kSubset[kIs]]...};
  }

  template <index_t kM, array<index_t, kM> kSubset>
    requires(kM <= size())
  MINT_HOST_DEVICE constexpr auto subset_reference() const
      -> const_ref_array<value_type, kM> {
    return subset_reference_impl<kM, kSubset>(make_index_sequence<kM>{});
  }

  template <index_t kM, array<index_t, kM> kSubset>
    requires(kM <= size())
  MINT_HOST_DEVICE constexpr auto subset_reference()
      -> ref_array<value_type, kM> {
    return subset_reference_impl<kM, kSubset>(make_index_sequence<kM>{});
  }
#endif

  template <index_t kM, array<index_t, kM> kSubset>
    requires(kM <= size())
  MINT_HOST_DEVICE constexpr auto get_subset() const {
    array<T, kM> ret = {};
    for (index_t i = 0; i < kM; i++)
      ret[i] = at(kSubset[i]);
    return ret;
  }

  template <index_t kM, array<index_t, kM> kSubset>
    requires(kM <= size())
  MINT_HOST_DEVICE constexpr void set_subset(const array<T, kM>& in) {
    for (index_t i = 0; i < kM; i++)
      at(kSubset[i]) = in[i];
  }

  MINT_HOST_DEVICE void print() const {
    printf("array {size %d, data[", size());
    for (index_t i = 0; i < kN; i++) {
      print_item((*this)[i]);
      printf(", ");
    }
    printf("]}");
  }
};
#else
static_assert(false, "disabled array impl");
template <class T, index_t kN>
struct array {
  using value_type = remove_cv_t<T>;

  value_type data_[kN];

  MINT_HOST_DEVICE constexpr array() {
    unroll_for_n<kN>()([&](auto i) { data_[i] = T{}; });
  }

  MINT_HOST_DEVICE constexpr array(initializer_list<value_type> init) {
#if 0
    // nvcc produce wrong result for this impl
    if constexpr (kN > 0)
      std::copy(init.begin(), init.end(), data_);
#else
    // nvcc produce correct result for this impl
    if constexpr (kN > 0) {
      auto it = init.begin();
      unroll_for_n<kN>()([&](auto i) {
        if (it != init.end()) {
          at(i) = *it;
          it++;
        }
      });
    }
#endif
  }

  constexpr bool operator==(const array&) const = default;

  MINT_HOST_DEVICE static consteval index_t size() {
    return kN;
  }

  MINT_HOST_DEVICE constexpr const value_type* begin() const {
    return data_;
  }

  MINT_HOST_DEVICE constexpr value_type* begin() {
    return data_;
  }

  MINT_HOST_DEVICE constexpr const value_type* end() const {
    return data_ + kN;
  }

  MINT_HOST_DEVICE constexpr value_type* end() {
    return data_ + kN;
  }

  MINT_HOST_DEVICE constexpr const value_type& at(index_t i) const {
    return data_[i];
  }

  MINT_HOST_DEVICE constexpr value_type& at(index_t i) {
    return data_[i];
  }

  MINT_HOST_DEVICE constexpr const value_type& operator[](index_t i) const {
    return data_[i];
  }

  MINT_HOST_DEVICE constexpr value_type& operator[](index_t i) {
    return data_[i];
  }

  MINT_HOST_DEVICE constexpr const T& operator()(index_t i) const {
    return data_[i];
  }

  MINT_HOST_DEVICE constexpr T& operator()(index_t i) {
    return data_[i];
  }

  template <index_t kI>
  MINT_HOST_DEVICE constexpr const value_type& at() const {
    return data_[kI];
  }

  template <index_t kI>
  MINT_HOST_DEVICE constexpr value_type& at() {
    return data_[kI];
  }

  MINT_HOST_DEVICE constexpr array& fill(const value_type& v) {
    for (index_t i = 0; i < kN; i++)
      data_[i] = v;

    return *this;
  }

#if 1
  // impl for subset_reference
  template <index_t kM, array<index_t, kM> kSubset, index_t... kIs>
    requires(kM == sizeof...(kIs) && kM <= size())
  MINT_HOST_DEVICE constexpr auto subset_reference_impl(index_sequence<kIs...>)
      const -> const_ref_array<value_type, kSubset.size()> {
    return {(*this)[kSubset[kIs]]...};
  }

  // impl for subset_reference
  template <index_t kM, array<index_t, kM> kSubset, index_t... kIs>
    requires(kM == sizeof...(kIs) && kM <= size())
  MINT_HOST_DEVICE constexpr auto subset_reference_impl(index_sequence<kIs...>)
      -> ref_array<value_type, kM> {
    return {(*this)[kSubset[kIs]]...};
  }

  template <index_t kM, array<index_t, kM> kSubset>
    requires(kM <= size())
  MINT_HOST_DEVICE constexpr auto subset_reference() const
      -> const_ref_array<value_type, kM> {
    return subset_reference_impl<kM, kSubset>(make_index_sequence<kM>{});
  }

  template <index_t kM, array<index_t, kM> kSubset>
    requires(kM <= size())
  MINT_HOST_DEVICE constexpr auto subset_reference()
      -> ref_array<value_type, kM> {
    return subset_reference_impl<kM, kSubset>(make_index_sequence<kM>{});
  }
#endif

  template <index_t kM, array<index_t, kM> kSubset>
    requires(kM <= size())
  MINT_HOST_DEVICE constexpr auto get_subset() const {
    array<T, kM> ret;
    unroll_for_n<kM>()([this, &ret](auto i) { ret[i] = (*this)[kSubset[i]]; });
    return ret;
  }

  template <index_t kM, array<index_t, kM> kSubset>
    requires(kM <= size())
  MINT_HOST_DEVICE constexpr void set_subset(const array<T, kM>& in) {
    unroll_for_n<kM>()([this, &in](auto i) { (*this)[kSubset[i]] = in[i]; });
  }

  MINT_HOST_DEVICE void print() const {
    printf("array {size %d, data[", size());
    for (index_t i = 0; i < kN; i++) {
      print_item((*this)[i]);
      printf(", ");
    }
    printf("]}");
  }
};

template <class T>
struct array<T, 0> {
  using value_type = remove_cv_t<T>;

  value_type dummy_data_ = value_type{};

  constexpr array() = default;

  MINT_HOST_DEVICE constexpr array(initializer_list<value_type>) {}

  constexpr bool operator==(const array&) const = default;

  MINT_HOST_DEVICE static consteval index_t size() {
    return 0;
  }

  MINT_HOST_DEVICE constexpr const value_type* begin() const {
    return &dummy_data_;
  }

  MINT_HOST_DEVICE constexpr value_type* begin() {
    return &dummy_data_;
  }

  MINT_HOST_DEVICE constexpr const value_type* end() const {
    return &dummy_data_;
  }

  MINT_HOST_DEVICE constexpr value_type* end() {
    return &dummy_data_;
  }

  MINT_HOST_DEVICE constexpr const value_type& at(index_t) const {
    return dummy_data_;
  }

  MINT_HOST_DEVICE constexpr value_type& at(index_t) {
    return dummy_data_;
  }

  MINT_HOST_DEVICE constexpr const value_type& operator[](index_t) const {
    return dummy_data_;
  }

  MINT_HOST_DEVICE constexpr value_type& operator[](index_t) {
    return dummy_data_;
  }

  MINT_HOST_DEVICE constexpr void fill(const value_type&) {}

  MINT_HOST_DEVICE void print() const {
    printf("array {size 0, data[]}");
  }
};
#endif

namespace impl {

template <class T, index_t... kIs, class F>
MINT_HOST_DEVICE constexpr auto generate_array_impl(
    const F& f,
    index_sequence<kIs...>) {
  return mint::array<T, sizeof...(kIs)>{f(kIs)...};
}

} // namespace impl

template <index_t kN, class F>
MINT_HOST_DEVICE constexpr auto generate_array(const F& f) {
  return impl::generate_array_impl<decltype(f(0))>(
      f, make_index_sequence<kN>{});
}

// convert to array
//   Arr could be: array, same_tuple, sequence
template <class Arr>
  requires(Arr::size() >= 0)
MINT_HOST_DEVICE constexpr auto to_array(const Arr& in) {
  array<typename Arr::value_type, Arr::size()> ret;
  static_for_n<Arr::size()>()([&](auto i) { ret[i] = in[i]; });
  return ret;
}

// reorder array
template <auto kNewToOld, class T, index_t kN>
  requires(kNewToOld.size() == kN)
MINT_HOST_DEVICE constexpr auto reorder(const array<T, kN>& in) {
  array<T, kN> ret;
  for (index_t i = 0; i < kN; i++)
    ret[i] = in[kNewToOld[i]];
  return ret;
}

} // namespace mint
