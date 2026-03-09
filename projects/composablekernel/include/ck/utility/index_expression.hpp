// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck::index_expr {

struct Ik {};
template <typename L, typename R> struct Add {};
template <typename L, typename R> struct Mult {};
template <typename L, typename R> struct Div {};
template <typename L, typename R> struct Mod {};

template <index_t ik, index_t v>
constexpr index_t eval(Number<v>) { return v; }

template <index_t ik>
constexpr index_t eval(Ik) { return ik; }

template <index_t ik, typename L, typename R>
constexpr index_t eval(Add<L, R>) {
    return eval<ik>(L{}) + eval<ik>(R{});
}

template <index_t ik, typename L, typename R>
constexpr index_t eval(Mult<L, R>) {
    return eval<ik>(L{}) * eval<ik>(R{});
}

template <index_t ik, typename L, typename R>
constexpr index_t eval(Div<L, R>) {
    constexpr index_t d = eval<ik>(R{});
    static_assert(d != 0, "ck::index_expr::Div: division by zero in compile-time index expression");
    return eval<ik>(L{}) / d;
}

template <index_t ik, typename L, typename R>
constexpr index_t eval(Mod<L, R>) {
    return eval<ik>(L{}) % eval<ik>(R{});
}

template <typename Expr, index_t ik>
inline constexpr index_t eval_v = eval<ik>(Expr{});

} // namespace ck::index_expr
