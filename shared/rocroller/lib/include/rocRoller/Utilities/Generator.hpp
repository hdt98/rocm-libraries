/*******************************************************************************
   *
   * MIT License
   *
   * Copyright 2024-2025 AMD ROCm(TM) Software
   *
   * Permission is hereby granted, free of charge, to any person obtaining a copy
   * of this software and associated documentation files (the "Software"), to deal
   * in the Software without restriction, including without limitation the rights
   * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   * copies of the Software, and to permit persons to whom the Software is
   * furnished to do so, subject to the following conditions:
   *
   * The above copyright notice and this permission notice shall be included in
   * all copies or substantial portions of the Software.
   *
   * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   * SOFTWARE.
   *
   *******************************************************************************/
#pragma once

#if defined(__clang__)
#include <coroutine>
#define co_yield_(arg) co_yield arg
#else
#include <coroutine>

#define co_yield_(arg)                             \
    do                                             \
    {                                              \
        auto generator_tmp_object_ = arg;          \
        co_yield std::move(generator_tmp_object_); \
    } while(0)
#endif
#include "Concepts.hpp"
#include <iterator>
#include <optional>
#include <ranges>
#include <tuple>
#include <utility>
namespace rocRoller
{

    template <std::movable T>
    class Generator
    {
    public:
        class promise_type
        {
        public:
            using Handle = std::coroutine_handle<promise_type>;

            Generator<T> get_return_object() noexcept;

            static constexpr void return_void() noexcept {}

            void unhandled_exception() noexcept;

            static constexpr std::suspend_always initial_suspend() noexcept
            {
                return {};
            }

            static constexpr std::suspend_always final_suspend() noexcept
            {
                return {};
            }

            std::suspend_always yield_value(T v) noexcept;

            auto yield_value(Generator<T>&& subGen) noexcept;

            template <CInputRangeOf<T> ARange>
            auto yield_value(ARange&& r) noexcept;

            auto yield_value(std::initializer_list<T> r) noexcept;

            inline void check_exception() const;

            inline std::optional<T> const& value() const noexcept;

            inline bool has_value() const noexcept;

            void advance() noexcept;

        private:
            mutable std::optional<T> m_value;

            mutable Handle m_leaf{nullptr};

            mutable promise_type* m_parent{nullptr};

            mutable std::exception_ptr m_exception{nullptr};

            friend class Generator<T>;
        };

        class Iterator
        {
        public:
            using difference_type = std::ptrdiff_t;
            using value_type      = T;
            using Handle          = std::coroutine_handle<promise_type>;

            Iterator() noexcept = default;

            explicit Iterator(Handle coroutine) noexcept;

            Iterator& operator++();

            void operator++(int);

            T const& operator*() const;

            T const* operator->() const;

            bool        operator==(std::default_sentinel_t) const noexcept;
            bool        operator!=(std::default_sentinel_t) const noexcept;
            friend bool operator==(std::default_sentinel_t s, Iterator const& it) noexcept
            {
                return it == s;
            }

        private:
            bool isDone() const noexcept;

            mutable Handle m_coroutine{nullptr};
        };

        using iterator   = std::common_iterator<Iterator, std::default_sentinel_t>;
        using Handle     = std::coroutine_handle<promise_type>;
        using value_type = T;

        explicit Generator(Handle coroutine) noexcept;

        Generator() noexcept = default;

        ~Generator();

        Generator(Generator const&) = delete;
        Generator& operator=(Generator const&) = delete;

        Generator(Generator&& other) noexcept;
        Generator& operator=(Generator&& other) noexcept;

        iterator begin();

        iterator end() noexcept;

        template <template <typename...> typename Container>
        Container<T> to();

        template <std::predicate<T> Predicate>
        Generator<T> filter(Predicate predicate);

        template <std::invocable<T> Func>
        Generator<std::invoke_result_t<Func, T>> map(Func func);

        Generator<T> take(size_t n);

        std::optional<T> only();

        bool empty();

    private:
        Handle m_coroutine{nullptr};
    };

    template <std::ranges::input_range Range, typename Predicate>
    requires(std::predicate<Predicate, std::ranges::range_value_t<Range>>)
        Generator<std::ranges::range_value_t<Range>> filter(Predicate predicate, Range range);
    template <std::ranges::input_range Range, typename Func>
    requires(std::invocable<Func, std::ranges::range_value_t<Range>>)
        Generator<std::invoke_result_t<Func, std::ranges::range_value_t<Range>>> map(Func  func,
                                                                                     Range range);
    template <typename T>
    Generator<T> take(size_t n, Generator<T> gen);
    template <std::ranges::input_range Range>
    std::optional<std::ranges::range_value_t<Range>> only(Range range);
    template <std::ranges::input_range Range>
    bool empty(Range range);
    template <std::ranges::forward_range ARange, std::ranges::forward_range... Rest>
    Generator<std::tuple<std::ranges::range_value_t<ARange>, std::ranges::range_value_t<Rest>...>>
        zip(ARange const& a, Rest const&... rest);
    template <std::ranges::input_range ARange, std::ranges::input_range... Rest>
    Generator<std::tuple<std::ranges::range_value_t<ARange>, std::ranges::range_value_t<Rest>...>>
        zip(ARange&& a, Rest&&... rest);
}
#include "Generator_impl.hpp"
