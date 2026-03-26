// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Sinkhorn kernel
// Handles packed MHC data layout: [H^pre (1xn), H^post (1xn), H^res (nxn)]
//
// Key design: Each thread processes ONE complete nxn matrix from ONE batch element
template <typename YDataType,
          typename ComputeDataType = float,
          index_t N_               = 4,
          bool UseLogDomain_       = true>
struct MHCSinkhornKernel
{
    static constexpr index_t kN         = N_;
    static constexpr index_t kNSquared  = kN * kN;
    static constexpr index_t kBlockSize = 64;
    static constexpr bool UseLogDomain  = UseLogDomain_;

    CK_TILE_HOST static constexpr index_t BlockSize() { return kBlockSize; }

    // Log-sum-exp function for numerical stability (only used in log domain)
    CK_TILE_DEVICE ComputeDataType log_sum_exp(const ComputeDataType* log_values, index_t n) const
    {
        if constexpr(!UseLogDomain)
            return 0.0f; // Not used in non-log domain

        // Find max for numerical stability
        ComputeDataType max_val = log_values[0];
        for(index_t i = 1; i < n; ++i)
        {
            max_val = ck_tile::max(max_val, log_values[i]);
        }

        if(::isinf(max_val) && max_val < 0)
        {
            return -INFINITY;
        }

        // Compute log(sum(exp(log_values - max_val))) + max_val
        ComputeDataType sum = 0.0f;
        for(index_t i = 0; i < n; ++i)
        {
            sum += ck_tile::exp(log_values[i] - max_val);
        }

        return ck_tile::log(sum) + max_val;
    }

    // Each thread processes one batch element's nxn matrix
    CK_TILE_DEVICE void operator()(YDataType* p_output,
                                   index_t batch,
                                   index_t output_dim,
                                   index_t n,
                                   index_t sinkhorn_iters) const
    {
        // Calculate block's starting batch index
        constexpr index_t kBlockM = kBlockSize;
        const index_t iM          = get_block_id() * kBlockM;

        // Create 3D tensor view covering the batch data [batch, n, n]
        auto matrix_tensor_naive = make_naive_tensor_view<address_space_enum::global>(
            p_output + 2 * n, // Start at first batch's H^res
            make_tuple(batch, n, n),
            make_tuple(output_dim,
                       n,
                       1), // Stride: output_dim between batches, n between rows, 1 within row
            number<1>{},
            number<1>{});

        auto matrix_tensor = pad_tensor_view(
            matrix_tensor_naive,
            make_tuple(number<kBlockM>{}, number<kN>{}, number<kN>{}),
            sequence<true, false, false>{});

        // Create tile distribution for batch processing
        // Each thread processes one complete nxn matrix
        constexpr auto tile_dist = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<kBlockSize, 1>, sequence<1, kN>, sequence<1, kN>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<0, 0>>,
                sequence<2, 3>,
                sequence<1, 1>>{});

        // Create tile window at the block's starting batch position
        auto input_window = make_tile_window(
            matrix_tensor, make_tuple(number<kN>{}, number<kN>{}), {iM, 0, 0}, tile_dist);

        // Load matrix
        auto input_tile = make_static_distributed_tensor<YDataType>(tile_dist);
        load_tile(input_tile, input_window);

        // Process in appropriate domain (log or standard)
        ComputeDataType matrix[kNSquared];

        // Manually loop through the tile's thread buffer to extract values
        // Each thread owns the entire n×n matrix
        auto& thread_buf = input_tile.get_thread_buffer();

        // Find max for numerical stability
        ComputeDataType max_val = -INFINITY;
        for(index_t i = 0; i < kNSquared; ++i)
        {
            ComputeDataType val = type_convert<ComputeDataType>(thread_buf[i]);
            max_val             = ck_tile::max(max_val, val);
        }

        // Process values based on domain
        if constexpr(UseLogDomain)
        {
            // Log domain: convert to log domain with max subtracted for stability
            for(index_t i = 0; i < kNSquared; ++i)
            {
                ComputeDataType val = type_convert<ComputeDataType>(thread_buf[i]);
                // In log domain: log(exp(val - max_val)) = val - max_val
                matrix[i] = val - max_val;
            }
        }
        else
        {
            // Standard domain: exponentiate with max subtracted for numerical stability
            for(index_t i = 0; i < kNSquared; ++i)
            {
                ComputeDataType val = type_convert<ComputeDataType>(thread_buf[i]);
                matrix[i]           = ck_tile::exp(val - max_val);
            }
        }

        // Check if matrix is degenerate (domain-specific checks)
        bool is_degenerate = false;
        if constexpr(UseLogDomain)
        {
            // Log domain: check individual values
            constexpr ComputeDataType min_log_val = -80.0f; // exp(-80) ≈ 1e-35
            is_degenerate                         = true;
            for(index_t i = 0; i < kNSquared; ++i)
            {
                if(matrix[i] > min_log_val)
                {
                    is_degenerate = false;
                    break;
                }
            }
        }
        else
        {
            // Standard domain: check matrix sum
            ComputeDataType matrix_sum = 0.0f;
            for(index_t i = 0; i < kNSquared; ++i)
            {
                matrix_sum += matrix[i];
            }
            constexpr ComputeDataType min_matrix_sum = 1e-30f;
            is_degenerate                            = (matrix_sum < min_matrix_sum);
        }

        if(is_degenerate)
        {
            // Set to uniform distribution
            ComputeDataType uniform_val = 1.0f / static_cast<ComputeDataType>(kNSquared);

            // Write uniform distribution back to the tile
            auto set_uniform = [&](YDataType& x) { x = type_convert<YDataType>(uniform_val); };
            tile_elementwise_inout(set_uniform, input_tile);

            // Store the uniform distribution using ck-tile API
            store_tile(input_window, input_tile);
            return;
        }

        // Sinkhorn iterations (domain-specific)
        for(index_t iter = 0; iter < sinkhorn_iters; ++iter)
        {
            // Normalize rows
            for(index_t row = 0; row < kN; ++row)
            {
                if constexpr(UseLogDomain)
                {
                    // Log domain normalization
                    ComputeDataType row_values[kN];
                    for(index_t col = 0; col < kN; ++col)
                    {
                        row_values[col] = matrix[row * kN + col];
                    }

                    ComputeDataType log_row_sum = log_sum_exp(row_values, kN);

                    // Handle degenerate row
                    if(::isinf(log_row_sum) && log_row_sum < 0)
                    {
                        // Set row to uniform in log domain: log(1/kN)
                        ComputeDataType log_uniform =
                            -ck_tile::log(static_cast<ComputeDataType>(kN));
                        for(index_t col = 0; col < kN; ++col)
                        {
                            matrix[row * kN + col] = log_uniform;
                        }
                    }
                    else
                    {
                        // Normalize: log(val/sum) = log(val) - log(sum)
                        for(index_t col = 0; col < kN; ++col)
                        {
                            matrix[row * kN + col] -= log_row_sum;
                        }
                    }
                }
                else
                {
                    // Standard domain normalization
                    ComputeDataType row_sum = 0.0f;
                    for(index_t col = 0; col < kN; ++col)
                    {
                        row_sum += matrix[row * kN + col];
                    }

                    if(row_sum > 1e-12f)
                    {
                        for(index_t col = 0; col < kN; ++col)
                        {
                            matrix[row * kN + col] /= row_sum;
                        }
                    }
                    else
                    {
                        // Set to uniform distribution if row sum is too small
                        ComputeDataType uniform_val = 1.0f / static_cast<ComputeDataType>(kN);
                        for(index_t col = 0; col < kN; ++col)
                        {
                            matrix[row * kN + col] = uniform_val;
                        }
                    }
                }
            }

            // Normalize columns (similar pattern)
            for(index_t col = 0; col < kN; ++col)
            {
                if constexpr(UseLogDomain)
                {
                    // Log domain column normalization
                    ComputeDataType col_values[kN];
                    for(index_t row = 0; row < kN; ++row)
                    {
                        col_values[row] = matrix[row * kN + col];
                    }

                    ComputeDataType log_col_sum = log_sum_exp(col_values, kN);

                    // Handle degenerate column
                    if(::isinf(log_col_sum) && log_col_sum < 0)
                    {
                        // Set column to uniform in log domain
                        ComputeDataType log_uniform =
                            -ck_tile::log(static_cast<ComputeDataType>(kN));
                        for(index_t row = 0; row < kN; ++row)
                        {
                            matrix[row * kN + col] = log_uniform;
                        }
                    }
                    else
                    {
                        // Normalize: log(val/sum) = log(val) - log(sum)
                        for(index_t row = 0; row < kN; ++row)
                        {
                            matrix[row * kN + col] -= log_col_sum;
                        }
                    }
                }
                else
                {
                    // Standard domain column normalization
                    ComputeDataType col_sum = 0.0f;
                    for(index_t row = 0; row < kN; ++row)
                    {
                        col_sum += matrix[row * kN + col];
                    }

                    if(col_sum > 1e-12f)
                    {
                        for(index_t row = 0; row < kN; ++row)
                        {
                            matrix[row * kN + col] /= col_sum;
                        }
                    }
                    else
                    {
                        // Set to uniform distribution if column sum is too small
                        ComputeDataType uniform_val = 1.0f / static_cast<ComputeDataType>(kN);
                        for(index_t row = 0; row < kN; ++row)
                        {
                            matrix[row * kN + col] = uniform_val;
                        }
                    }
                }
            }
        }

        // Store result back
        index_t elem_idx = 0;
        if constexpr(UseLogDomain)
        {
            // Log domain: convert back from log domain
            auto convert_from_log = [&](YDataType& x) {
                ComputeDataType log_val = matrix[elem_idx++];
                ComputeDataType val     = ck_tile::exp(log_val);
                val                     = ck_tile::max(val, static_cast<ComputeDataType>(1e-35f));
                val                     = ck_tile::min(val, static_cast<ComputeDataType>(1.0f));
                x                       = type_convert<YDataType>(val);
            };
            tile_elementwise_inout(convert_from_log, input_tile);
        }
        else
        {
            // Standard domain: direct store
            auto store_result = [&](YDataType& x) {
                ComputeDataType val = matrix[elem_idx++];
                x                   = type_convert<YDataType>(val);
            };
            tile_elementwise_inout(store_result, input_tile);
        }

        store_tile(input_window, input_tile);
    }
};

// Dispatcher wrapper that selects the appropriate template instantiation
template <typename YDataType, typename ComputeDataType = float, bool UseLogDomain = true>
struct MHCSinkhornKernelDispatcher
{
    static constexpr index_t kBlockSize = 64;

    CK_TILE_HOST static constexpr index_t BlockSize() { return kBlockSize; }

    // Check if the kernel supports the given n value
    CK_TILE_HOST static constexpr bool IsSupportedArgument(index_t n)
    {
        return (n == 4) || (n == 8); // Supported template instantiations
    }

    CK_TILE_DEVICE void operator()(YDataType* p_output,
                                   index_t batch,
                                   index_t output_dim,
                                   index_t n,
                                   index_t sinkhorn_iters) const
    {
        // Dispatch to appropriate template instantiation based on n
        if(n == 4)
        {
            MHCSinkhornKernel<YDataType, ComputeDataType, 4, UseLogDomain>{}(
                p_output, batch, output_dim, n, sinkhorn_iters);
        }
        else if(n == 8)
        {
            MHCSinkhornKernel<YDataType, ComputeDataType, 8, UseLogDomain>{}(
                p_output, batch, output_dim, n, sinkhorn_iters);
        }
        else
        {
            assert(false && "MHCSinkhornKernelDispatcher only supports n in the set (4, 8)");
        }
    }
};

} // namespace ck_tile
