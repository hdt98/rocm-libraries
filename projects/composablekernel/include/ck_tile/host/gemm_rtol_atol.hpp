// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/check_err.hpp"

namespace ck_tile {

/// Computes relative and absolute error tolerance thresholds for GEMM verification
/// without split-K accumulation error (single-WG, no split-K).
/// @return tuple<double, double> where [0] = relative tolerance (rtol),
///                                     [1] = absolute tolerance (atol).
/// @note Does NOT support pk_fp4_t operand types. For block-scale quantized GEMMs
///       using pk_fp4_t, use the local calculate_rtol_atol in
///       example/ck_tile/38_block_scale_gemm/gemm_utils.hpp instead.
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculateRtolAtol(const index_t k_dim, const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    const auto relative_tolerance =
        get_relative_threshold<ComputeType, CDataType, AccDataType>(k_dim);
    const auto absolute_tolerance =
        get_absolute_threshold<ComputeType, CDataType, AccDataType>(max_accumulated_value, k_dim);
    return make_tuple(relative_tolerance, absolute_tolerance);
}

/// Computes relative and absolute error tolerance thresholds for GEMM verification.
/// Accounts for split-K (or multi-WG) accumulation error in addition to per-element
/// compute error.
/// @return tuple<double, double> where [0] = relative tolerance (rtol),
///                                     [1] = absolute tolerance (atol).
/// @note Does NOT support pk_fp4_t operand types. For block-scale quantized GEMMs
///       using pk_fp4_t, use the local calculate_rtol_atol in
///       example/ck_tile/38_block_scale_gemm/gemm_utils.hpp instead.
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

/// 5-parameter variant for multi-D GEMMs where a D tensor participates in the accumulation
/// and its precision affects the effective compute type (3-way minimum-precision selection).
/// D0DataType = first D-tensor (bias/residual input); EDataType = output tensor (replaces C).
/// @return tuple<double, double> where [0] = relative tolerance (rtol),
///                                     [1] = absolute tolerance (atol).
template <typename ADataType,
          typename BDataType,
          typename D0DataType,
          typename EDataType,
          typename AccDataType>
auto calculateRtolAtol(const index_t k_dim,
                       const index_t k_batch,
                       const float max_accumulated_value)
{
    using ComputeTypeAB =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    using ComputeType =
        std::conditional_t<sizeof(ComputeTypeAB) < sizeof(D0DataType), ComputeTypeAB, D0DataType>;
    // Calculate thresholds
    const auto relative_tolerance =
        get_relative_threshold<ComputeType, EDataType, AccDataType>(
            integer_divide_ceil(k_dim, k_batch));
    const auto absolute_tolerance =
        get_absolute_threshold<ComputeType, EDataType, AccDataType>(
            max_accumulated_value / k_batch, integer_divide_ceil(k_dim, k_batch));
    // Calculate error due to multiple WGs working in the same E macro tile
    const auto relative_tolerance_split_k =
        get_relative_threshold<EDataType, EDataType, EDataType>(k_batch);
    const auto absolute_tolerance_split_k =
        get_absolute_threshold<EDataType, EDataType, EDataType>(max_accumulated_value, k_batch);
    // Use higher threshold
    return make_tuple(std::max(relative_tolerance, relative_tolerance_split_k),
                      std::max(absolute_tolerance, absolute_tolerance_split_k));
}

} // namespace ck_tile
