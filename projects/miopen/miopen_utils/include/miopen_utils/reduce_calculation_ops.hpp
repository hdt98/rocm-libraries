// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side copy of the ReduceCalculationOp_t enum and reduce_func templates.
// Canonical kernel-side definition: src/kernels/MIOpenReduceCalculation.hpp
#pragma once

#include <miopen/miopen.h>

enum class ReduceCalculationOp_t
{
    Prod = 1,
    Sum,
    First_ = Prod,
    Last_  = Sum,
};

static_assert(MIOPEN_REDUCE_CALCULATION_PROD == static_cast<int>(ReduceCalculationOp_t::Prod));
static_assert(MIOPEN_REDUCE_CALCULATION_SUM == static_cast<int>(ReduceCalculationOp_t::Sum));

template <typename T, ReduceCalculationOp_t op>
struct reduce_func
{
    inline constexpr void calculate(T& a, T b) const;
};

template <typename T>
struct reduce_func<T, ReduceCalculationOp_t::Prod>
{
    inline constexpr void calculate(T& a, T b) const { a *= b; }
};

template <typename T>
struct reduce_func<T, ReduceCalculationOp_t::Sum>
{
    inline constexpr void calculate(T& a, T b) const { a += b; }
};
