// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/math.hpp"
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

// Note: Multi-arg static_for_product<First, Rest...> specialization is defined
// after static_ford (which it depends on). See below.

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

// Helper array for storing compile-time indices
template <index_t N>
struct index_array
{
    index_t data[N > 0 ? N : 1];

    CK_TILE_HOST_DEVICE constexpr index_t operator[](index_t i) const { return data[i]; }
};

// Common base for static_ford providing compile-time constants
template <class Lengths, class Orders>
struct ford_base
{
    static constexpr index_t NDim = Lengths::size();

    static constexpr index_t TotalSize =
        reduce_on_sequence(Lengths{}, multiplies<>{}, number<1>{});

    static constexpr auto OrderedLengths = Lengths::reorder_new_to_old(Orders{});

    using OrderedLengthsType = remove_cvref_t<decltype(OrderedLengths)>;

    CK_TILE_HOST_DEVICE constexpr ford_base()
    {
        static_assert(Lengths::size() > 0, "wrong! Lengths is empty");
        static_assert(Lengths::size() == Orders::size(), "wrong! inconsistent size");
    }
};

// Index decomposer: converts linear index to N-dimensional indices
// Uses precomputed strides for O(1) decomposition per dimension
template <class OrderedLengths, class IndexSeq>
struct index_decomposer;

template <index_t... Ls, index_t... Is>
struct index_decomposer<sequence<Ls...>, sequence<Is...>>
{
    static constexpr index_t NDim = sizeof...(Ls);

    static constexpr index_array<NDim> lengths = {{Ls...}};

    // Compute strides: strides[i] = product of lengths[i+1..N-1]
    // strides[N-1] = 1, strides[i] = strides[i+1] * lengths[i+1]
    static constexpr index_array<NDim> compute_strides()
    {
        index_array<NDim> result{};
        if constexpr(NDim > 0)
        {
            result.data[NDim - 1] = 1;
            for(index_t i = NDim - 2; i >= 0; --i)
            {
                result.data[i] = result.data[i + 1] * lengths[i + 1];
            }
        }
        return result;
    }

    static constexpr index_array<NDim> strides = compute_strides();

    // Compile-time decomposition: ordered_idx[i] = (linear_idx / strides[i]) % lengths[i]
    template <index_t LinearIdx>
    using decompose = sequence<((LinearIdx / strides[Is]) % lengths[Is])...>;
};

} // namespace detail

// Optimized N-dimensional loop using flat iteration with index decomposition.
// Uses O(1) template instantiation depth instead of recursive templates.
//
// Lengths is sequence<...>, it is the length of each dimension for N-dimensional loop
// Orders is sequence<...>, it is the order of dimension in which static_ford
// will loop over each dimension
template <class Lengths,
          class Orders = typename arithmetic_sequence_gen<0, Lengths::size(), 1>::type>
struct static_ford : detail::ford_base<Lengths, Orders>
{
    using Base = detail::ford_base<Lengths, Orders>;
    using Decomposer =
        detail::index_decomposer<typename Base::OrderedLengthsType,
                                 typename arithmetic_sequence_gen<0, Base::NDim, 1>::type>;

    // F signature: F(sequence<...> multi_id)
    // multi_id is the unordered multi-index
    template <class F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        // Single flat loop with index decomposition - O(1) template depth
        static_for<0, Base::TotalSize, 1>{}([&](auto linear_idx) {
            using OrderedIdx = typename Decomposer::template decompose<linear_idx.value>;
            f(OrderedIdx::reorder_old_to_new(Orders{}));
        });
    }
};

// ============================================================================
// Optimized static_for_product implementation (multi-arg specialization)
// Uses static_ford for O(1) template depth instead of recursive lambdas
// ============================================================================

namespace detail {

// Helper to extract iteration count from dimension specifiers
template <typename T>
struct product_dim_length;

// static_for<Begin, End, Step> -> number of iterations
template <index_t Begin, index_t End, index_t Step>
struct product_dim_length<static_for<Begin, End, Step>>
{
    static constexpr index_t value = (End - Begin) / Step;
};

// sequence<Is...> -> iterate over each element (size of sequence)
template <index_t... Is>
struct product_dim_length<sequence<Is...>>
{
    static constexpr index_t value = sizeof...(Is);
};

// number<N> -> iterate 0 to N-1
template <index_t N>
struct product_dim_length<number<N>>
{
    static constexpr index_t value = N;
};

// Helper to convert linear index component to the actual iteration value
template <typename T>
struct product_dim_value;

// static_for<Begin, End, Step> -> Begin + idx * Step
template <index_t Begin, index_t End, index_t Step>
struct product_dim_value<static_for<Begin, End, Step>>
{
    template <index_t Idx>
    static constexpr index_t get = Begin + Idx * Step;
};

// sequence<Is...> -> Is[idx]
template <index_t... Is>
struct product_dim_value<sequence<Is...>>
{
    template <index_t Idx>
    static constexpr index_t get = sequence<Is...>::at(Idx);
};

// number<N> -> idx directly
template <index_t N>
struct product_dim_value<number<N>>
{
    template <index_t Idx>
    static constexpr index_t get = Idx;
};

// Helper to call f with unpacked number<> arguments from a sequence
template <typename... Dims>
struct product_caller;

template <typename Dim>
struct product_caller<Dim>
{
    template <typename F, index_t Idx>
    CK_TILE_HOST_DEVICE static constexpr void call(F& f)
    {
        f(number<product_dim_value<Dim>::template get<Idx>>{});
    }
};

template <typename Dim0, typename Dim1, typename... Rest>
struct product_caller<Dim0, Dim1, Rest...>
{
    template <typename F, index_t Idx0, index_t Idx1, index_t... IdxRest>
    CK_TILE_HOST_DEVICE static constexpr void call(F& f)
    {
        f(number<product_dim_value<Dim0>::template get<Idx0>>{},
          number<product_dim_value<Dim1>::template get<Idx1>>{},
          number<product_dim_value<Rest>::template get<IdxRest>>{}...);
    }
};

// Dispatcher that extracts indices from sequence and calls product_caller
template <typename... Dims>
struct product_dispatch;

template <typename Dim>
struct product_dispatch<Dim>
{
    template <typename F, index_t Idx>
    CK_TILE_HOST_DEVICE static constexpr void call(F& f, sequence<Idx>)
    {
        product_caller<Dim>::template call<F, Idx>(f);
    }
};

template <typename Dim0, typename Dim1>
struct product_dispatch<Dim0, Dim1>
{
    template <typename F, index_t Idx0, index_t Idx1>
    CK_TILE_HOST_DEVICE static constexpr void call(F& f, sequence<Idx0, Idx1>)
    {
        product_caller<Dim0, Dim1>::template call<F, Idx0, Idx1>(f);
    }
};

template <typename Dim0, typename Dim1, typename Dim2>
struct product_dispatch<Dim0, Dim1, Dim2>
{
    template <typename F, index_t Idx0, index_t Idx1, index_t Idx2>
    CK_TILE_HOST_DEVICE static constexpr void call(F& f, sequence<Idx0, Idx1, Idx2>)
    {
        product_caller<Dim0, Dim1, Dim2>::template call<F, Idx0, Idx1, Idx2>(f);
    }
};

template <typename Dim0, typename Dim1, typename Dim2, typename Dim3>
struct product_dispatch<Dim0, Dim1, Dim2, Dim3>
{
    template <typename F, index_t Idx0, index_t Idx1, index_t Idx2, index_t Idx3>
    CK_TILE_HOST_DEVICE static constexpr void call(F& f, sequence<Idx0, Idx1, Idx2, Idx3>)
    {
        product_caller<Dim0, Dim1, Dim2, Dim3>::template call<F, Idx0, Idx1, Idx2, Idx3>(f);
    }
};

} // namespace detail

// Optimized multi-dimensional product iteration using flat loop + index decomposition.
// Avoids recursive lambda nesting that causes O(N) unique lambda type instantiations.
//
// Usage: static_for_product<number<2>, number<3>>{}([](auto i, auto j) { ... });
//        static_for_product<sequence<1,2,3>, number<4>>{}([](auto i, auto j) { ... });
template <typename First, typename... Rest>
struct static_for_product<First, Rest...>
{
    // Build lengths sequence for static_ford
    using Lengths = sequence<detail::product_dim_length<First>::value,
                             detail::product_dim_length<Rest>::value...>;

    template <typename F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        // Use static_ford for flat iteration with O(1) template depth
        static_ford<Lengths>{}([&](auto idx) {
            // Convert multi-index to actual iteration values and call f
            detail::product_dispatch<First, Rest...>::call(f, idx);
        });
    }
};

// ============================================================================
// End of static_for_product implementation
// ============================================================================

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
