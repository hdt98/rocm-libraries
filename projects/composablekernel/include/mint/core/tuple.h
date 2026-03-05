#pragma once
#include <mint/config.h>
#include <mint/core/index_t.h>
#include <mint/core/initializer_list.h>
#include <mint/core/integer_sequence.h>
#include <mint/core/integral_constant.h>
#include <mint/core/nd_index.h>
#include <mint/core/print.h>
#include <mint/core/type_traits.h>

namespace mint {

template <auto kArr>
  requires(kArr.size() >= 0)
MINT_HOST_DEVICE consteval auto to_sequence();

template <index_t kN>
MINT_HOST_DEVICE constexpr auto static_for_n();

#if defined(__CUDACC__)
namespace impl {

template <index_t kI, typename... Ts>
struct type_pack_element;

template <index_t kI, typename T, typename... Ts>
struct type_pack_element<kI, T, Ts...> : type_pack_element<kI - 1, Ts...> {};

template <typename T, typename... Ts>
struct type_pack_element<0, T, Ts...> {
  using type = T;
};

} // namespace impl

template <index_t kI, typename... Ts>
using type_pack_element_t = typename impl::type_pack_element<kI, Ts...>::type;
#else
template <index_t kI, typename... Ts>
using type_pack_element_t = __type_pack_element<kI, Ts...>;
#endif

namespace impl {

template <index_t kI, class T>
struct tuple_elem {
  T data_;

  MINT_HOST_DEVICE constexpr tuple_elem() : data_{} {}
  MINT_HOST_DEVICE constexpr tuple_elem(T&& t) : data_{t} {}
  MINT_HOST_DEVICE constexpr tuple_elem(const T& t) : data_{t} {}
  MINT_HOST_DEVICE constexpr tuple_elem(T& t) : data_{t} {}

  MINT_HOST_DEVICE constexpr bool operator==(const tuple_elem& other) const {
    return data_ == other.data_;
  }

  MINT_HOST_DEVICE constexpr const T& get_impl() const {
    return data_;
  }

  MINT_HOST_DEVICE constexpr T& get_impl() {
    return data_;
  }
};

template <class Seq, class... Ts>
struct tuple_impl;

template <index_t... kIs, class... Ts>
  requires(sizeof...(Ts) > 0 && sizeof...(kIs) == sizeof...(Ts))
struct tuple_impl<index_sequence<kIs...>, Ts...> : tuple_elem<kIs, Ts>... {
  constexpr tuple_impl() = default;

  constexpr bool operator==(const tuple_impl&) const = default;

  MINT_HOST_DEVICE constexpr tuple_impl(Ts... ts)
      : tuple_elem<kIs, Ts>{std::forward<Ts>(ts)}... {}

  template <index_t kI>
    requires(kI < sizeof...(Ts))
  MINT_HOST_DEVICE constexpr auto at() const
      -> const type_pack_element_t<kI, Ts...>& {
    using elem_type = type_pack_element_t<kI, Ts...>;
    return static_cast<const tuple_elem<kI, elem_type>*>(this)->get_impl();
  }

  template <index_t kI>
    requires(kI < sizeof...(Ts))
  MINT_HOST_DEVICE constexpr auto at() -> type_pack_element_t<kI, Ts...>& {
    using elem_type = type_pack_element_t<kI, Ts...>;
    return static_cast<tuple_elem<kI, elem_type>*>(this)->get_impl();
  }
};

template <>
struct tuple_impl<index_sequence<>> {
  constexpr tuple_impl() = default;

  constexpr bool operator==(const tuple_impl&) const = default;
};

} // namespace impl

template <class... Ts>
struct tuple : impl::tuple_impl<make_index_sequence<sizeof...(Ts)>, Ts...> {
  using impl::tuple_impl<make_index_sequence<sizeof...(Ts)>, Ts...>::tuple_impl;
  using base_type = impl::tuple_impl<make_index_sequence<sizeof...(Ts)>, Ts...>;

  constexpr bool operator==(const tuple&) const = default;

  MINT_HOST_DEVICE static consteval index_t size() {
    return sizeof...(Ts);
  }

  template <index_t kI>
  MINT_HOST_DEVICE constexpr auto at() const
      -> const type_pack_element_t<kI, Ts...>& {
    return base_type::template at<kI>();
  }

  template <index_t kI>
  MINT_HOST_DEVICE constexpr auto at() -> type_pack_element_t<kI, Ts...>& {
    return base_type::template at<kI>();
  }

  template <index_t kI>
  MINT_HOST_DEVICE constexpr auto at(index_constant<kI>) const
      -> const type_pack_element_t<kI, Ts...>& {
    return base_type::template at<kI>();
  }

  template <index_t kI>
  MINT_HOST_DEVICE constexpr auto at(index_constant<kI>)
      -> type_pack_element_t<kI, Ts...>& {
    return base_type::template at<kI>();
  }

  template <index_t kI>
  MINT_HOST_DEVICE constexpr auto operator[](index_constant<kI>) const
      -> const type_pack_element_t<kI, Ts...>& {
    return base_type::template at<kI>();
  }

  template <index_t kI>
  MINT_HOST_DEVICE constexpr auto operator[](index_constant<kI>)
      -> type_pack_element_t<kI, Ts...>& {
    return base_type::template at<kI>();
  }

  // WARNING: Don't use! For C++ structured binding only!
  template <index_t kI>
  MINT_HOST_DEVICE constexpr auto get() const
      -> const type_pack_element_t<kI, Ts...>& {
    return base_type::template at<kI>();
  }

  // WARNING: Don't use! For C++ structured binding only!
  template <index_t kI>
  MINT_HOST_DEVICE constexpr auto get() -> type_pack_element_t<kI, Ts...>& {
    return base_type::template at<kI>();
  }

  template <index_t... kIs>
  MINT_HOST_DEVICE constexpr auto get_subset_impl(
      index_sequence<kIs...>) const {
    return tuple<type_pack_element_t<kIs, Ts...>...>{at<kIs>()...};
  }

  template <index_t kM, nd_index<kM> kSubset>
    requires(kM <= size())
  MINT_HOST_DEVICE constexpr auto get_subset() const {
    return get_subset_impl(to_sequence<kSubset>());
  }

  template <index_t kM, nd_index<kM> kSubset, class... Xs>
    requires(sizeof...(Xs) == kM && kM <= size())
  MINT_HOST_DEVICE constexpr void set_subset(const tuple<Xs...>& in) {
    static_for_n<sizeof...(Xs)>()([&](auto i) { at<kSubset[i]>() = in[i]; });
  }

  MINT_HOST_DEVICE void print() const {
    printf("tuple {size %d, data[", size());
    static_for_n<size()>()([this](auto i) {
      print_item(this->template at<i.value>());
      printf(", ");
    });
    printf("]}");
  }
};

// FIXME: ambiguous with std::make_tuple
template <class... Ts>
MINT_HOST_DEVICE constexpr tuple<unwrap_ref_decay_t<Ts>...> make_tuple(
    Ts&&... ts) {
  return {std::forward<Ts>(ts)...};
}

template <class... Args>
MINT_HOST_DEVICE constexpr tuple<Args&...> tie(Args&... args) noexcept {
  return {args...};
}

template <class... Ts>
MINT_HOST_DEVICE constexpr tuple<Ts...> tuple_cat(const tuple<Ts...>& t) {
  return t;
}

namespace impl {

template <class... Xs, class... Ys, index_t... kIs, index_t... kJs>
MINT_HOST_DEVICE constexpr tuple<Xs..., Ys...> tuple_cat_impl(
    const tuple<Xs...>& x,
    const tuple<Ys...>& y,
    index_sequence<kIs...>,
    index_sequence<kJs...>) {
  return {x.template at<kIs>()..., y.template at<kJs>()...};
}

} // namespace impl

template <class... Xs, class... Ys>
MINT_HOST_DEVICE constexpr tuple<Xs..., Ys...> tuple_cat(
    const tuple<Xs...>& x,
    const tuple<Ys...>& y) {
  return impl::tuple_cat_impl(
      x,
      y,
      make_index_sequence<sizeof...(Xs)>{},
      make_index_sequence<sizeof...(Ys)>{});
}

template <class Tuple0, class Tuple1, class... Tuples>
MINT_HOST_DEVICE constexpr auto
tuple_cat(const Tuple0& t0, const Tuple1& t1, const Tuples&... ts) {
  return tuple_cat(tuple_cat(t0, t1), ts...);
}

namespace impl {

template <index_t... kIs, class F>
MINT_HOST_DEVICE constexpr auto generate_tuple_impl(
    const F& f,
    index_sequence<kIs...>) {
  return mint::make_tuple(f(index_constant<kIs>{})...);
}

} // namespace impl

template <index_t kN, class F>
MINT_HOST_DEVICE constexpr auto generate_tuple(const F& f) {
  return impl::generate_tuple_impl(f, make_index_sequence<kN>{});
}

template <index_t kN, class Ts>
constexpr auto generate_repeated_tuple(const Ts& value) {
  return generate_tuple<kN>([&](auto /*i*/) { return value; });
}

} // namespace mint

namespace std {
// WARNING: Don't use! For C++ structured binding only!
template <class... Ts>
struct tuple_size<mint::tuple<Ts...>>
    : std::integral_constant<std::size_t, sizeof...(Ts)> {};

// WARNING: Don't use! For C++ structured binding only!
template <std::size_t kI, class... Ts>
struct tuple_element<kI, mint::tuple<Ts...>> {
  using type = mint::type_pack_element_t<kI, Ts...>;
};

} // namespace std
