/*******************************************************************************
   *
   * MIT License
   *
   * Copyright 2024-2025 AMD ROCm(TM) Software
   *
   * [License text same as above...]
   *
   *******************************************************************************/
#pragma once
#include "Generator.hpp"
namespace rocRoller
{

    template <std::movable T>
    Generator<T> Generator<T>::promise_type::get_return_object() noexcept
    {
        return Generator<T>{Handle::from_promise(*this)};
    }
    template <std::movable T>
    void Generator<T>::promise_type::unhandled_exception() noexcept
    {

        m_exception = std::current_exception();
        m_value.reset();
        m_leaf = nullptr;
    }
    template <std::movable T>
    inline void Generator<T>::promise_type::check_exception() const
    {

        if(m_exception) [[unlikely]]
        {
            std::exception_ptr exc = nullptr;
            std::swap(exc, m_exception);
            std::rethrow_exception(exc);
        }
    }
    template <std::movable T>
    inline std::optional<T> const& Generator<T>::promise_type::value() const noexcept
    {
        return m_value;
    }
    template <std::movable T>
    inline bool Generator<T>::promise_type::has_value() const noexcept
    {
        return m_value.has_value();
    }

    template <std::movable T>
    std::suspend_always Generator<T>::promise_type::yield_value(T v) noexcept
    {

        m_value = std::move(v);
        m_leaf  = nullptr;
        return {};
    }

    template <std::movable T>
    auto Generator<T>::promise_type::yield_value(Generator<T>&& subGen) noexcept
    {

        struct SubGenAwaitable
        {
            Generator<T>  gen;
            promise_type* self;
            bool          exhausted;
            SubGenAwaitable(Generator<T>&& g, promise_type* s) noexcept
                : gen(std::move(g))
                , self(s)
                , exhausted(false)
            {
            }

            bool await_ready() noexcept
            {
                if(!gen.m_coroutine) [[unlikely]]
                {

                    exhausted = true;
                    return true;
                }
                auto& subPromise = gen.m_coroutine.promise();

                subPromise.m_parent = self;

                Handle leaf = gen.m_coroutine;
                while(leaf.promise().m_leaf)
                {
                    leaf = leaf.promise().m_leaf;
                }

                self->m_leaf = leaf;

                if(!leaf.done())
                {
                    leaf.resume();
                }

                auto& leafPromise = leaf.promise();
                if(leafPromise.m_value.has_value()) [[likely]]
                {

                    self->m_value = std::move(leafPromise.m_value);
                    leafPromise.m_value.reset();
                    exhausted = false;
                    return false;
                }

                if(leafPromise.m_leaf) [[unlikely]]
                {

                    self->m_leaf = leafPromise.m_leaf;
                    exhausted    = false;
                    return false;
                }

                self->m_leaf = nullptr;
                exhausted    = true;
                return true;
            }
            void await_suspend(std::coroutine_handle<promise_type>) noexcept {}
            void await_resume() noexcept {}
        };

        m_value.reset();
        return SubGenAwaitable{std::move(subGen), this};
    }

    template <std::movable T>
    template <CInputRangeOf<T> ARange>
    auto Generator<T>::promise_type::yield_value(ARange&& r) noexcept
    {

        auto rangeToGenerator = [](auto begin, auto end) -> Generator<T> {
            for(; begin != end; ++begin)
            {
                co_yield *begin;
            }
        };

        return yield_value(rangeToGenerator(r.begin(), r.end()));
    }
    template <std::movable T>
    auto Generator<T>::promise_type::yield_value(std::initializer_list<T> r) noexcept
    {
        return yield_value<std::initializer_list<T>>(std::move(r));
    }

    template <std::movable T>
    void Generator<T>::promise_type::advance() noexcept
    {

        m_value.reset();

        Handle selfHandle = Handle::from_promise(*this);

        while(true)
        {

            Handle current = m_leaf ? m_leaf : selfHandle;

            while(current.promise().m_leaf)
            {
                current = current.promise().m_leaf;
            }

            if(current != selfHandle)
            {
                m_leaf = current;
            }

            if(!current.done()) [[likely]]
            {
                current.resume();
            }

            auto& currentPromise = current.promise();
            if(currentPromise.m_exception) [[unlikely]]
            {

                m_exception                = currentPromise.m_exception;
                currentPromise.m_exception = nullptr;
                m_leaf                     = nullptr;
                return;
            }

            if(currentPromise.m_value.has_value()) [[likely]]
            {

                if(current != selfHandle)
                {
                    m_value = std::move(currentPromise.m_value);
                    currentPromise.m_value.reset();
                }

                return;
            }

            if(currentPromise.m_leaf) [[unlikely]]
            {

                m_leaf = currentPromise.m_leaf;
                continue;
            }

            if(current == selfHandle)
            {

                m_leaf = nullptr;
                return;
            }

            promise_type* parent = currentPromise.m_parent;
            if(!parent) [[unlikely]]
            {

                m_leaf = nullptr;
                return;
            }

            parent->m_leaf = nullptr;

            Handle parentHandle = Handle::from_promise(*parent);

            if(parentHandle == selfHandle)
            {
                m_leaf = nullptr;
            }
            else
            {
                m_leaf = parentHandle;
            }
        }
    }

    template <std::movable T>
    Generator<T>::Iterator::Iterator(Handle coroutine) noexcept
        : m_coroutine{coroutine}
    {
    }
    template <std::movable T>
    auto Generator<T>::Iterator::operator++() -> Iterator&
    {

        m_coroutine.promise().check_exception();

        m_coroutine.promise().advance();
        return *this;
    }
    template <std::movable T>
    void Generator<T>::Iterator::operator++(int)
    {

        ++(*this);
    }
    template <std::movable T>
    T const& Generator<T>::Iterator::operator*() const
    {
        m_coroutine.promise().check_exception();
        auto const& val = m_coroutine.promise().value();
        if(!val) [[unlikely]]
        {
            throw std::runtime_error("Generator::Iterator: dereferencing exhausted iterator");
        }
        return *val;
    }
    template <std::movable T>
    T const* Generator<T>::Iterator::operator->() const
    {
        return &(**this);
    }
    template <std::movable T>
    bool Generator<T>::Iterator::isDone() const noexcept
    {
        if(!m_coroutine) [[unlikely]]
        {
            return true;
        }

        auto const& promise = m_coroutine.promise();
        if(promise.has_value()) [[likely]]
        {
            return false;
        }
        return m_coroutine.done();
    }
    template <std::movable T>
    bool Generator<T>::Iterator::operator==(std::default_sentinel_t) const noexcept
    {
        return isDone();
    }
    template <std::movable T>
    bool Generator<T>::Iterator::operator!=(std::default_sentinel_t s) const noexcept
    {
        return !(*this == s);
    }

    template <std::movable T>
    Generator<T>::Generator(Handle coroutine) noexcept
        : m_coroutine{coroutine}
    {
    }
    template <std::movable T>
    Generator<T>::~Generator()
    {
        if(m_coroutine)
        {
            m_coroutine.destroy();
        }
    }
    template <std::movable T>
    Generator<T>::Generator(Generator&& other) noexcept
        : m_coroutine{other.m_coroutine}
    {
        other.m_coroutine = nullptr;
    }
    template <std::movable T>
    auto Generator<T>::operator=(Generator&& other) noexcept -> Generator&
    {
        if(this != &other)
        {
            if(m_coroutine)
            {
                m_coroutine.destroy();
            }
            m_coroutine       = other.m_coroutine;
            other.m_coroutine = nullptr;
        }
        return *this;
    }
    template <std::movable T>
    auto Generator<T>::begin() -> iterator
    {

        if(m_coroutine) [[likely]]
        {
            m_coroutine.promise().advance();
        }
        return iterator{Iterator{m_coroutine}};
    }
    template <std::movable T>
    auto Generator<T>::end() noexcept -> iterator
    {
        return iterator{std::default_sentinel};
    }
    template <std::movable T>
    template <template <typename...> typename Container>
    Container<T> Generator<T>::to()
    {
        auto first = begin();
        auto last  = end();
        return Container<T>(first, last);
    }
    template <std::movable T>
    template <std::predicate<T> Predicate>
    Generator<T> Generator<T>::filter(Predicate predicate)
    {
        return rocRoller::filter(std::move(predicate), std::move(*this));
    }
    template <std::movable T>
    template <std::invocable<T> Func>
    Generator<std::invoke_result_t<Func, T>> Generator<T>::map(Func func)
    {
        return rocRoller::map(std::move(func), std::move(*this));
    }
    template <std::movable T>
    Generator<T> Generator<T>::take(size_t n)
    {
        return rocRoller::take(n, std::move(*this));
    }
    template <std::movable T>
    std::optional<T> Generator<T>::only()
    {
        return rocRoller::only(std::move(*this));
    }
    template <std::movable T>
    bool Generator<T>::empty()
    {
        return begin() == end();
    }

    template <std::ranges::input_range Range, typename Predicate>
    requires(std::predicate<Predicate, std::ranges::range_value_t<Range>>)
        Generator<std::ranges::range_value_t<Range>> filter(Predicate predicate, Range range)
    {
        for(auto&& val : range)
        {
            if(predicate(val))
            {
                co_yield val;
            }
        }
    }
    template <std::ranges::input_range Range, typename Func>
    requires(std::invocable<Func, std::ranges::range_value_t<Range>>)
        Generator<std::invoke_result_t<Func, std::ranges::range_value_t<Range>>> map(Func  func,
                                                                                     Range range)
    {
        for(auto&& val : range)
        {
            co_yield func(val);
        }
    }
    template <typename T>
    Generator<T> take(size_t n, Generator<T> gen)
    {
        auto it = gen.begin();
        for(size_t i = 0; i < n && it != gen.end(); ++i, ++it)
        {
            co_yield *it;
        }
    }
    template <std::ranges::input_range T>
    Generator<std::ranges::range_value_t<T>> take(size_t n, T gen)
    {
        auto it = gen.begin();
        for(size_t i = 0; i < n && it != gen.end(); ++i, ++it)
        {
            co_yield *it;
        }
    }
    template <std::ranges::input_range Range>
    std::optional<std::ranges::range_value_t<Range>> only(Range range)
    {
        auto iter = range.begin();
        if(iter == range.end())
        {
            return {};
        }
        auto value = *iter;
        ++iter;
        if(iter == range.end())
        {
            return value;
        }
        return {};
    }
    template <std::ranges::input_range Range>
    bool empty(Range range)
    {
        return range.begin() == range.end();
    }
    template <std::ranges::forward_range ARange, std::ranges::forward_range... Rest>
    Generator<std::tuple<std::ranges::range_value_t<ARange>, std::ranges::range_value_t<Rest>...>>
        zip(ARange const& a, Rest const&... rest)
    {
        if constexpr(sizeof...(Rest) == 0)
        {
            for(auto const& elem : a)
            {
                co_yield std::make_tuple(elem);
            }
        }
        else
        {
            auto aIter    = a.begin();
            auto restGen  = zip(rest...);
            auto restIter = restGen.begin();
            for(; aIter != a.end() && restIter != restGen.end(); ++aIter, ++restIter)
            {
                co_yield std::tuple_cat(std::make_tuple(*aIter), *restIter);
            }
        }
    }
    template <std::ranges::input_range ARange, std::ranges::input_range... Rest>
    Generator<std::tuple<std::ranges::range_value_t<ARange>, std::ranges::range_value_t<Rest>...>>
        zip(ARange&& a, Rest&&... rest)
    {
        if constexpr(sizeof...(Rest) == 0)
        {
            for(auto&& elem : a)
            {
                co_yield std::make_tuple(elem);
            }
        }
        else
        {
            auto aIter    = a.begin();
            auto restGen  = zip(std::forward<Rest>(rest)...);
            auto restIter = restGen.begin();
            for(; aIter != a.end() && restIter != restGen.end(); ++aIter, ++restIter)
            {
                co_yield std::tuple_cat(std::make_tuple(*aIter), *restIter);
            }
        }
    }
}
