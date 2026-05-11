// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side copy of the ReduceExtremeOp_t enum and reduce_func templates.
// Canonical kernel-side definition: src/kernels/MIOpenReduceExtreme.hpp
#pragma once

#include <miopen/miopen.h>

enum class ReduceExtremeOp_t
{
    Argmin = 1,
    Argmax,
    Min,
    Max,
    First_ = Argmin,
    Last_  = Max,
};

static_assert(MIOPEN_REDUCE_EXTREME_ARGMIN == static_cast<int>(ReduceExtremeOp_t::Argmin));
static_assert(MIOPEN_REDUCE_EXTREME_ARGMAX == static_cast<int>(ReduceExtremeOp_t::Argmax));
static_assert(MIOPEN_REDUCE_EXTREME_MIN == static_cast<int>(ReduceExtremeOp_t::Min));
static_assert(MIOPEN_REDUCE_EXTREME_MAX == static_cast<int>(ReduceExtremeOp_t::Max));

template <typename T1, typename T2, ReduceExtremeOp_t op>
struct reduce_func
{
    inline constexpr void calculate(T1& a, T1 b, T2& c, T2 d) const;
};

template <typename T1, typename T2>
struct reduce_func<T1, T2, ReduceExtremeOp_t::Max>
{
    inline constexpr void calculate(T1& a, T1 b, T2& c, T2 d) const
    {
        if(a < b)
        {
            a = b;
            c = d;
        }
    }
};

template <typename T1, typename T2>
struct reduce_func<T1, T2, ReduceExtremeOp_t::Min>
{
    inline constexpr void calculate(T1& a, T1 b, T2& c, T2 d) const
    {
        if(a > b)
        {
            a = b;
            c = d;
        }
    }
};

template <typename T1, typename T2>
struct reduce_func<T1, T2, ReduceExtremeOp_t::Argmax>
{
    inline constexpr void calculate(T1& a, T1 b, T2& c, T2 d) const
    {
        if(a < b)
        {
            a = b;
            c = d;
        }
    }
};

template <typename T1, typename T2>
struct reduce_func<T1, T2, ReduceExtremeOp_t::Argmin>
{
    inline constexpr void calculate(T1& a, T1 b, T2& c, T2 d) const
    {
        if(a > b)
        {
            a = b;
            c = d;
        }
    }
};
