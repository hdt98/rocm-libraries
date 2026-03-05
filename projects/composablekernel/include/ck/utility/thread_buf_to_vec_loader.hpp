// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/data_type.hpp"

namespace ck {

template <typename... Funcs>
struct FunctorInvoker
{
    ck::Tuple<Funcs...> funcs;

    __host__ __device__ constexpr FunctorInvoker(Funcs... fs) : funcs(ck::forward<Funcs>(fs)...) {}

    template <index_t I>
    __host__ __device__ constexpr void operator()(ck::Number<I> i) const
    {
        invoke(i, std::index_sequence_for<Funcs...>{});
    }

    template <index_t I, std::size_t... Is>
    __host__ __device__ constexpr void invoke(ck::Number<I> i, std::index_sequence<Is...>) const
    {
        (funcs[ck::Number<static_cast<index_t>(Is)>{}](i), ...);
    }
};

// required for CTAD to work with __host__ __device__ qualifiers
template <typename... Fs>
__host__ __device__ constexpr auto MakeFunctorInvoker(Fs&&... fs)
{
    return FunctorInvoker<Fs...>{ck::forward<Fs&&>(fs)...};
}

template <typename T, index_t ik, typename Enable = void>
struct IndexEval;

struct Ik
{
};

template <typename L, typename R>
struct Add
{
};

template <typename L, typename R>
struct Mult
{
};

template <typename L, typename R>
struct Div
{
};

template <typename L, typename R>
struct Mod
{
};

template <typename T, index_t ik>
struct IndexEval<T, ik, std::enable_if_t<!std::is_same_v<decltype(T::value), void>, void>>
{
    static constexpr index_t value = T::value;
};

template <index_t ik>
struct IndexEval<Ik, ik, void>
{
    static constexpr index_t value = ik;
};

template <typename L, typename R, index_t ik>
struct IndexEval<Add<L, R>,
                 ik,
                 std::enable_if_t<!std::is_same_v<decltype(IndexEval<L, ik>::value), void> &&
                                      !std::is_same_v<decltype(IndexEval<R, ik>::value), void>,
                                  void>>
{
    static constexpr index_t value = IndexEval<L, ik>::value + IndexEval<R, ik>::value;
};

template <typename L, typename R, index_t ik>
struct IndexEval<Mult<L, R>,
                 ik,
                 std::enable_if_t<!std::is_same_v<decltype(IndexEval<L, ik>::value), void> &&
                                      !std::is_same_v<decltype(IndexEval<R, ik>::value), void>,
                                  void>>
{
    static constexpr index_t value = IndexEval<L, ik>::value * IndexEval<R, ik>::value;
};

template <typename L, typename R, index_t ik>
struct IndexEval<Div<L, R>,
                 ik,
                 std::enable_if_t<!std::is_same_v<decltype(IndexEval<L, ik>::value), void> &&
                                      !std::is_same_v<decltype(IndexEval<R, ik>::value), void> &&
                                      (IndexEval<R, ik>::value != 0),
                                  void>>
{
    static constexpr index_t value = IndexEval<L, ik>::value / IndexEval<R, ik>::value;
};

template <typename L, typename R, index_t ik>
struct IndexEval<Mod<L, R>,
                 ik,
                 std::enable_if_t<!std::is_same_v<decltype(IndexEval<L, ik>::value), void> &&
                                      !std::is_same_v<decltype(IndexEval<R, ik>::value), void>,
                                  void>>
{
    static constexpr index_t value = IndexEval<L, ik>::value % IndexEval<R, ik>::value;
};

template <typename ThreadVec,
          typename ThreadBuf,
          typename ThreadDesc,
          typename ComputeType,
          typename... IdxWrapper>
struct thread_buf_to_vec_loader
{
    ThreadVec& thread_vec;
    ThreadBuf& thread_buf;

    __host__ __device__ constexpr thread_buf_to_vec_loader(ThreadVec& tv, ThreadBuf& tb)
        : thread_vec(tv), thread_buf(tb)
    {
    }

    template <index_t ik>
    __host__ __device__ constexpr void operator()(Number<ik>) const
    {
        constexpr auto thread_desc = ThreadDesc{};
        thread_vec.template AsType<ComputeType>()(Number<ik>{}) =
            thread_buf[Number<thread_desc.CalculateOffset(
                ck::make_tuple(Number<IndexEval<IdxWrapper, ik>::value>{}...))>{}];
    }
};
} // namespace ck
