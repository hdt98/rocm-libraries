// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/check_err.hpp"

namespace ck_tile {

/// Computes relative and absolute error tolerance thresholds for GEMM verification.
/// Accounts for split-K (or multi-WG) accumulation error in addition to per-element
/// compute error.
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculateRtolAtol(const index_t k_dim,
                       const index_t k_batch,
                       const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto relative_tolerance =
        get_relative_threshold<ComputeType, CDataType, AccDataType>(
            integer_divide_ceil(k_dim, k_batch));
    const auto absolute_tolerance =
        get_absolute_threshold<ComputeType, CDataType, AccDataType>(
            max_accumulated_value / k_batch, integer_divide_ceil(k_dim, k_batch));
    // Calculate error due to multiple WGs working in the same C macro tile
    const auto relative_tolerance_split_k =
        get_relative_threshold<CDataType, CDataType, CDataType>(k_batch);
    const auto absolute_tolerance_split_k =
        get_absolute_threshold<CDataType, CDataType, CDataType>(max_accumulated_value, k_batch);
    // Use higher threshold
    return make_tuple(std::max(relative_tolerance, relative_tolerance_split_k),
                      std::max(absolute_tolerance, absolute_tolerance_split_k));
}

} // namespace ck_tile
