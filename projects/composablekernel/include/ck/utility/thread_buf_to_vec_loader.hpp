// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/index_expression.hpp"

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


using Ik = index_expr::Ik;

template <typename L, typename R>
using Add = index_expr::Add<L, R>;

template <typename L, typename R>
using Mult = index_expr::Mult<L, R>;

template <typename L, typename R>
using Div = index_expr::Div<L, R>;

template <typename L, typename R>
using Mod = index_expr::Mod<L, R>;

template <typename T, index_t ik, typename Enable = void>
struct IndexEval
{
    static constexpr index_t value = index_expr::eval_v<T, ik>;
};

template <typename ThreadVec,
          typename ThreadBuf,
          typename ThreadDesc,
          typename ComputeType,
          typename... IdxExpr>
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
        constexpr auto idx_tuple = ck::make_tuple(Number<index_expr::eval_v<IdxExpr, ik>>{}...);
        constexpr auto offset    = thread_desc.CalculateOffset(idx_tuple);

        auto& target = thread_vec.template AsType<ComputeType>()(Number<ik>{});
        target       = thread_buf[Number<offset>{}];
    }
};
} // namespace ck

