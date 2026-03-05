#pragma once
#include <mint/core/bind.h>
#include <mint/core/index_t.h>
#include <mint/core/initializer_list.h>
#include <mint/core/integer_sequence.h>
#include <mint/core/integral_constant.h>
#include <mint/core/pack_param.h>
#include <mint/core/unpack_arg.h>
#include <functional>

namespace mint {

// for_n
struct for_n {
  MINT_HOST_DEVICE constexpr for_n(index_t n) : n_{n} {}

  template <class F>
  MINT_HOST_DEVICE constexpr auto operator()(F&& f) const {
    for (index_t i = 0; i < n_; i++)
      f(i);
  }

  const index_t n_;
};

// for_nd
template <index_t kNDim>
  requires(kNDim > 0)
struct for_nd {
  MINT_HOST_DEVICE constexpr for_nd(initializer_list<index_t> ndims) {
    std::copy(ndims.begin(), ndims.begin() + kNDim, ndims_);
  }

  MINT_HOST_DEVICE constexpr void operator()(auto&& f) const {
    if constexpr (kNDim > 1) {
      auto next = for_nd<kNDim - 1>{};
      std::copy(ndims_ + 1, ndims_ + kNDim, next.ndims_);
      for (index_t i = 0; i < ndims_[0]; i++)
        next(bind_front(f, i));
    } else {
      for (index_t i = 0; i < ndims_[0]; i++)
        f(i);
    }
  }

#if 0 // debug
  const index_t ndims_[kNDim];
#else
  // FIXME: add back const
  index_t ndims_[kNDim];
#endif
};

// unroll_for_seq
template <class T>
struct unroll_for_seq;

template <index_t... kIs>
struct unroll_for_seq<index_sequence<kIs...>> {
  MINT_HOST_DEVICE constexpr void operator()(auto&& f) const {
    if constexpr (sizeof...(kIs) > 0)
      (f(kIs), ...);
  }
};

// unroll_for_n
// return func that takes f(index_t), loop over (i=0:kN) and call f(i)
template <index_t kN>
MINT_HOST_DEVICE constexpr auto unroll_for_n() {
  return unroll_for_seq<make_index_sequence<kN>>();
}

// unroll_for_nd
// return func that takes f(index_t, index_t...),
// loop over (i=0:kN, is=0:kNs, ...) and call f(i, is...)
template <index_t kN, index_t... kNs>
struct unroll_for_nd {
  MINT_HOST_DEVICE constexpr void operator()(auto&& f) const {
    unroll_for_n<kN>()(
        [&f](index_t i) { unroll_for_nd<kNs...>()(bind_front(f, i)); });
  }
};

template <index_t kN>
struct unroll_for_nd<kN> {
  MINT_HOST_DEVICE constexpr void operator()(auto&& f) const {
    unroll_for_n<kN>()(f);
  }
};

// static_for_seq
template <class T>
struct static_for_seq;

template <index_t... kIs>
struct static_for_seq<index_sequence<kIs...>> {
  MINT_HOST_DEVICE constexpr void operator()(auto&& f) const {
    if constexpr (sizeof...(kIs) > 0)
      (f(index_constant<kIs>{}), ...);
  }
};

// static_for_n
template <index_t kN>
MINT_HOST_DEVICE constexpr auto static_for_n() {
  return static_for_seq<make_index_sequence<kN>>();
}

namespace impl {

// static_for_nd_impl
// return func that takes f(index_constant<I0>, index_constant<I1>, ...)
// loop over (i=0:kN, is=0:kNs, ...) and call f(i0, i1, ...)
template <index_t kN, index_t... kNs>
struct static_for_nd_impl {
  MINT_HOST_DEVICE constexpr void operator()(auto&& f) const {
    static_for_n<kN>()(
        [&f](auto i) { static_for_nd_impl<kNs...>()(bind_front(f, i)); });
  }
};

template <index_t kN>
struct static_for_nd_impl<kN> {
  MINT_HOST_DEVICE constexpr void operator()(auto&& f) const {
    static_for_n<kN>()(f);
  }
};

} // namespace impl

// static_for_nd
// return func that takes f(index_constant<I0>, index_constant<I1>, ...)
// loop over (i=0:kN, is=0:kNs, ...) and call f(i0, i1, ...)
template <index_t... kNs>
struct static_for_nd {
  MINT_HOST_DEVICE constexpr void operator()(auto&& f) const {
    if constexpr (sizeof...(kNs) == 0)
      f();
    else
      impl::static_for_nd_impl<kNs...>()(f);
  }
};

template <index_t kN>
struct static_for_nd<kN> {
  MINT_HOST_DEVICE constexpr void operator()(auto&& f) const {
    static_for_n<kN>()(f);
  }
};

// same as unroll_for_nd, but kNs is a container
template <auto kNs>
using unroll_for_nd2 = pack_param<unroll_for_nd, kNs>::type;

// same as static_for_nd, but kNs is a container
template <auto kNs>
using static_for_nd2 = pack_param<static_for_nd, kNs>::type;

// same as unroll_for_nd, except that:
//  1. kNs is a container
//  2. lambda function signature: f(nd_index<kNs>{xs...}), instead of f(xs...)
template <auto kNs>
  requires requires {
    { kNs.size() } -> std::convertible_to<index_t>;
  }
struct unroll_for_nd3 {
  using base_type = pack_param<unroll_for_nd, kNs>::type;

  MINT_HOST_DEVICE constexpr void operator()(auto&& f) const {
    base_type{}(unpack_array_arg<kNs.size()>(f));
  }
};

// same as static_for_nd, except that:
//  1. kNs is a container
//  2. lambda function signature: f(index_sequence<xs...>), instead of f(xs...)
template <auto kNs>
  requires requires {
    { kNs.size() } -> std::convertible_to<index_t>;
  }
struct static_for_nd3 {
  using base_type = pack_param<static_for_nd, kNs>::type;

  MINT_HOST_DEVICE constexpr void operator()(auto&& f) const {
    base_type{}(
        unpack_sequence_arg_as_integral_constants<kNs.size(), index_t>(f));
  }
};

} // namespace mint
