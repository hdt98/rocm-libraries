#pragma once
#include <mint/core/for.h>
#include <mint/core/tuple.h>

namespace mint {
namespace impl {

template <class T, class Seq>
struct same_tuple_impl;

template <class T, index_t... kIs>
struct same_tuple_impl<T, index_sequence<kIs...>> {
  using type = tuple<decltype((kIs, T{}))...>;
};

} // namespace impl

template <class T, index_t kN>
struct same_tuple : impl::same_tuple_impl<T, make_index_sequence<kN>>::type {
  using base_type =
      typename impl::same_tuple_impl<T, make_index_sequence<kN>>::type;

 public:
  using value_type = remove_cv_t<T>;

  MINT_HOST_DEVICE constexpr same_tuple() : base_type{} {}

  MINT_HOST_DEVICE constexpr same_tuple(const base_type& in) : base_type{in} {}

  MINT_HOST_DEVICE constexpr same_tuple(initializer_list<T> init)
      : base_type{} {
    auto it = init.begin();
    static_for_n<kN>()([this, &it, &init](auto i) {
      if (it != init.end()) {
        base_type::operator[](i) = *it;
        it++;
      }
    });
  }

  template <index_t kI>
  MINT_HOST_DEVICE constexpr const T& operator[](index_constant<kI>) const {
    return base_type::template at<kI>();
  }

  template <index_t kI>
  MINT_HOST_DEVICE constexpr T& operator[](index_constant<kI>) {
    return base_type::template at<kI>();
  }

  MINT_HOST_DEVICE constexpr same_tuple& fill(const T& v) {
    static_for_n<kN>()([&](auto i) { base_type::template at<i.value>() = v; });
    return *this;
  }

  template <auto kSubset>
    requires(is_same_v<typename decltype(kSubset)::value_type, index_t>)
  MINT_HOST_DEVICE constexpr auto get_subset() const
      -> same_tuple<T, kSubset.size()> {
    return base_type::template get_subset<kSubset>();
  }

#if 0
  template <auto kSubset>
    requires(
        is_same_v<typename decltype(kSubset)::value_type, index_t> &&
        kSubset.size() <= base_type::size())
  MINT_HOST_DEVICE constexpr void set_subset(
      const same_tuple<T, kSubset.size()>& in) {
    base_type::template set_subset<kSubset>(in);
  }
#endif
};

} // namespace mint
