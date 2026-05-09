// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_DRIVER_FORD_HPP
#define GUARD_MIOPEN_DRIVER_FORD_HPP

// Driver-local multi-dimensional for-loop utilities.
// Replaces miopen::ford, miopen::par_ford, and miopen::par_for.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <numeric>
#include <thread>
#include <tuple>
#include <vector>

namespace driver_ford {

// Thread that auto-joins on destruction.
struct joinable_thread : std::thread
{
    template <class... Xs>
    joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...) {} // NOLINT

    joinable_thread& operator=(joinable_thread&& other) = default;
    joinable_thread(joinable_thread&& other)            = default;
    ~joinable_thread()
    {
        if(this->joinable())
            this->join();
    }
};

// Parallel for: splits [0, n) across threads.
template <class F>
void par_for(std::size_t n, F f)
{
    constexpr std::size_t min_grain = 8;
    const auto threadsize =
        std::min<std::size_t>(std::thread::hardware_concurrency(), n / min_grain);
    if(threadsize <= 1)
    {
        for(std::size_t i = 0; i < n; i++)
            f(i);
    }
    else
    {
        std::vector<joinable_thread> threads(threadsize);
        const std::size_t grainsize = std::ceil(static_cast<double>(n) / threads.size());
        std::size_t work            = 0;
        std::generate(threads.begin(), threads.end(), [&]() {
            auto result = joinable_thread([=, &f] {
                std::size_t start = work;
                std::size_t last  = std::min(n, work + grainsize);
                for(std::size_t i = start; i < last; i++)
                    f(i);
            });
            work += grainsize;
            return result;
        });
    }
}

// Parallel for with explicit max threads.
template <class F>
void par_for(std::size_t n, std::size_t max_threads, F f)
{
    const auto threadsize = std::min<std::size_t>(std::thread::hardware_concurrency(), max_threads);
    if(threadsize <= 1 || n <= 1)
    {
        for(std::size_t i = 0; i < n; i++)
            f(i);
    }
    else
    {
        auto actual_threads = std::min(threadsize, n);
        std::vector<joinable_thread> threads(actual_threads);
        const std::size_t grainsize = std::ceil(static_cast<double>(n) / threads.size());
        std::size_t work            = 0;
        std::generate(threads.begin(), threads.end(), [&]() {
            auto result = joinable_thread([=, &f] {
                std::size_t start = work;
                std::size_t last  = std::min(n, work + grainsize);
                for(std::size_t i = start; i < last; i++)
                    f(i);
            });
            work += grainsize;
            return result;
        });
    }
}

namespace detail {

template <class F, class T, std::size_t... Ns>
auto unpack_impl(F f, std::index_sequence<Ns...>, T&& x)
{
    return f(std::get<Ns>(x)...);
}

template <class F, class T>
auto unpack(F f, T&& x)
{
    using type = std::remove_cvref_t<T>;
    return unpack_impl(
        f, std::make_index_sequence<std::tuple_size<type>::value>(), std::forward<T>(x));
}

} // namespace detail

// Multi-dimensional sequential for-loop.
// Usage: ford(n1, n2, n3)([&](int i, int j, int k) { ... });
struct ford_impl
{
    template <class F>
    void operator()(F f) const { f(); }

    template <class F, class T, class... Ts>
    void operator()(F f, T x, Ts... xs) const
    {
        for(T i = 0; i < x; i++)
            (*this)([&](Ts... is) { f(i, is...); }, xs...);
    }
};

template <class T>
struct ford_wrapper
{
    template <class... Ts>
    auto operator()(Ts... xs) const
    {
        return std::bind(T{}, std::placeholders::_1, xs...);
    }
};

inline constexpr ford_wrapper<ford_impl> ford{};

// Multi-dimensional parallel for-loop.
// Usage: par_ford(n1, n2, n3)([&](int i, int j, int k) { ... });
struct par_ford_impl
{
    template <class F, class... Ts>
    void operator()(F f, Ts... xs) const
    {
        using array_type = std::array<std::size_t, sizeof...(Ts)>;
        array_type lens  = {{static_cast<std::size_t>(xs)...}};
        array_type strides;
        strides.fill(1);
        std::partial_sum(
            lens.rbegin(), lens.rend() - 1, strides.rbegin() + 1, std::multiplies<std::size_t>());
        auto size = std::accumulate(
            lens.begin(), lens.end(), static_cast<std::size_t>(1), std::multiplies<std::size_t>());
        par_for(size, [&](std::size_t i) {
            array_type indices;
            std::transform(strides.begin(),
                           strides.end(),
                           lens.begin(),
                           indices.begin(),
                           [&](size_t stride, size_t len) { return (i / stride) % len; });
            detail::unpack(f, indices);
        });
    }
};

inline constexpr ford_wrapper<par_ford_impl> par_ford{};

} // namespace driver_ford

#endif // GUARD_MIOPEN_DRIVER_FORD_HPP
