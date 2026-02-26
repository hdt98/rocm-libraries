// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include <stdint.h>
#include <utility>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
namespace ck_tile {

namespace detail {

struct swallow
{
    template <typename... Ts>
    CK_TILE_HOST_DEVICE constexpr swallow(Ts&&...)
    {
    }
};

template <class>
struct static_for_impl;

template <index_t... Is>
struct static_for_impl<sequence<Is...>>
{
    template <class F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        swallow{(f(number<Is>{}), 0)...};
    }
};

} // namespace detail

// F signature: F(number<Iter>)
template <index_t NBegin, index_t NEnd, index_t Increment>
struct static_for
{
    CK_TILE_HOST_DEVICE constexpr static_for()
    {
        static_assert(Increment != 0 && (NEnd - NBegin) % Increment == 0,
                      "Wrong! should satisfy (NEnd - NBegin) % Increment == 0");
        static_assert((Increment > 0 && NBegin <= NEnd) || (Increment < 0 && NBegin >= NEnd),
                      "wrongs! should (Increment > 0 && NBegin <= NEnd) || (Increment < 0 && "
                      "NBegin >= NEnd)");
    }

    template <class F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        detail::static_for_impl<typename arithmetic_sequence_gen<NBegin, NEnd, Increment>::type>{}(
            f);
    }
};

namespace detail {

template <typename T, T... Is>
struct applier
{
    template <typename F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        // tweak -fbracket-depth if compilation fails. Clang default limit is 256
        (f(number<Is>{}), ...);
    }
};

template <int32_t Size> // == sizeof...(Is)
using make_applier = __make_integer_seq<applier, index_t, Size>;

} // namespace detail

template <index_t N>
struct static_for<0, N, 1> : detail::make_applier<N>
{
    using detail::make_applier<N>::operator();
};

template <typename... Ts>
struct static_for_product;
template <index_t... Is>
struct static_for_product<static_for<Is...>> : public static_for<Is...>
{
};
template <index_t... Is>
struct static_for_product<sequence<Is...>> : public static_for<Is...>
{
};
template <index_t I>
struct static_for_product<number<I>> : public static_for<0, I, 1>
{
};
template <typename First, typename... Rest>
struct static_for_product<First, Rest...>
{
    template <typename F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        static_for_product<First>{}([=](auto I) {
            static_for_product<Rest...>{}([=](auto... Is) { //
                f(I, Is...);
            });
        });
    }
};

struct identity
{
    template <typename T>
    CK_TILE_HOST_DEVICE constexpr T&& operator()(T&& arg) const noexcept
    {
        return std::forward<T>(arg);
    }
};

// Similar to identity, but takes an additional index parameter as the first argument.
// The index is ignored and only the second argument (value) is forwarded.
// Useful for indexed element-wise operations where the functor signature requires an index.
struct idx_identity
{
    template <typename I, typename T>
    CK_TILE_HOST_DEVICE constexpr T&& operator()(I&& /*idx*/, T&& arg) const noexcept
    {
        return std::forward<T>(arg);
    }
};

namespace detail {

// Compile-time index decomposition for flat loop → multi-index conversion.
// Uses pre-computed strides to convert a linear index into N-dimensional coordinates.
template <class OrderedLengths, class IndexSeq>
struct index_decomposer;

template <index_t... Ls, index_t... Is>
struct index_decomposer<sequence<Ls...>, sequence<Is...>>
{
    static constexpr index_t NDim                      = sizeof...(Ls);
    static constexpr detail::index_array<NDim> lengths = {{Ls...}};

    static constexpr detail::index_array<NDim> compute_all_strides()
    {
        detail::index_array<NDim> result{};
        if constexpr(NDim > 0)
        {
            result[NDim - 1] = 1;
            for(index_t i = NDim - 2; i >= 0; --i)
            {
                result[i] = result[i + 1] * lengths[i + 1];
            }
        }
        return result;
    }

    static constexpr detail::index_array<NDim> strides = compute_all_strides();

    // Compile-time decomposition: linear index → sequence of ordered indices
    template <index_t LinearIdx>
    using decompose = sequence<((LinearIdx / strides[Is]) % lengths[Is])...>;
};

} // namespace detail

// Compile-time N-dimensional loop with static multi-indices.
// Uses O(1) template instantiation depth via flat loop with index decomposition,
// avoiding the recursive template structures of the old implementation.
//
// Lengths is sequence<...>, the size of each dimension for N-dimensional loop
// Orders is sequence<...>, the iteration order of dimensions
//
// Example:
//   static_ford<sequence<2, 3>>{}([](auto multi_id) {
//       constexpr auto i = multi_id[number<0>{}];  // 0, 0, 0, 1, 1, 1
//       constexpr auto j = multi_id[number<1>{}];  // 0, 1, 2, 0, 1, 2
//   });
template <class Lengths,
          class Orders = typename arithmetic_sequence_gen<0, Lengths::size(), 1>::type>
struct static_ford
{
    static constexpr index_t NDim      = Lengths::size();
    static constexpr index_t TotalSize = Lengths::product();

    CK_TILE_HOST_DEVICE constexpr static_ford()
    {
        static_assert(NDim > 0, "wrong! Lengths is empty");
        static_assert(NDim == Orders::size(), "wrong! inconsistent size");
    }

    using Decomposer =
        detail::index_decomposer<remove_cvref_t<decltype(Lengths::reorder_new_to_old(Orders{}))>,
                                 make_index_sequence<NDim>>;

    // F signature: F(sequence<...> multi_id)
    // multi_id is the unordered multi-index
    template <class F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        static_for<0, TotalSize, 1>{}([&](auto linear_idx) {
            using OrderedIdx = typename Decomposer::template decompose<linear_idx.value>;
            f(OrderedIdx::reorder_old_to_new(Orders{}));
        });
    }
};

namespace detail {

template <typename Indices>
struct unpack_impl;

template <index_t... Is>
struct unpack_impl<sequence<Is...>>
{
    template <typename F, typename X>
    CK_TILE_HOST_DEVICE constexpr auto operator()(F&& f, X&& x) const
    {
#if 0
        return std::forward<F>(f)(std::forward<X>(x).at(number<Is>{})...);
#else
        return std::forward<F>(f)(std::forward<X>(x).template at<Is>()...);
#endif
    }
};

template <typename Seq0, typename Seq1>
struct unpack2_impl;

// TODO: remove this, after properly implementing unpack that takes any number of containers
template <index_t... Is, index_t... Js>
struct unpack2_impl<sequence<Is...>, sequence<Js...>>
{
    template <typename F, typename X, typename Y>
    CK_TILE_HOST_DEVICE constexpr auto operator()(F&& f, X&& x, Y&& y) const
    {
#if 0
        return std::forward<F>(f)(std::forward<X>(x).at(number<Is>{})...,
                                  std::forward<Y>(y).at(number<Js>{})...);
#else
        return std::forward<F>(f)(std::forward<X>(x).template at<Is>()...,
                                  std::forward<Y>(y).template at<Js>()...);
#endif
    }
};

} // namespace detail

template <typename F, typename X>
CK_TILE_HOST_DEVICE constexpr auto unpack(F&& f, X&& x)
{
    using X_ = remove_reference_t<X>;
    return detail::unpack_impl<typename arithmetic_sequence_gen<0, X_::size(), 1>::type>{}(
        std::forward<F>(f), std::forward<X>(x));
}

// TODO: properly implement unpack that takes any number of containers
template <typename F, typename X, typename Y>
CK_TILE_HOST_DEVICE constexpr auto unpack2(F&& f, X&& x, Y&& y)
{
    using X_ = remove_reference_t<X>;
    using Y_ = remove_reference_t<Y>;
    return detail::unpack2_impl<typename arithmetic_sequence_gen<0, X_::size(), 1>::type,
                                typename arithmetic_sequence_gen<0, Y_::size(), 1>::type>{}(
        std::forward<F>(f), std::forward<X>(x), std::forward<Y>(y));
}

// z = predicate ? x : y
template <bool predicate, typename X, typename Y>
constexpr auto conditional_expr(X&& x, Y&& y)
{
    if constexpr(predicate)
    {
        return std::forward<X>(x);
    }
    else
    {
        return std::forward<Y>(y);
    }
}

} // namespace ck_tile
#pragma clang diagnostic pop
