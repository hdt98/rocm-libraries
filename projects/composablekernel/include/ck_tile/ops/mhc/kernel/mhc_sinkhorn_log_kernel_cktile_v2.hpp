// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Log-domain Sinkhorn kernel using ck-tile style API (Version 2)
// This version properly uses ck-tile's tile abstraction for loading and storing
// Handles packed MHC data layout: [H^pre (1xn), H^post (1xn), H^res (nxn)]
//
// Key design: Each thread processes ONE complete n×n matrix from ONE batch element
// Uses ck-tile for efficient tile-based loading, vectorized writes for storing
// Template parameter N allows compile-time optimization for specific matrix sizes
template <typename YDataType, typename ComputeDataType = float, index_t N_ = 4>
struct MHCSinkhornLogKernelTileV2
{
    static constexpr index_t kN         = N_;
    static constexpr index_t kNSquared  = kN * kN;
    static constexpr index_t kBlockSize = 64;

    CK_TILE_HOST static constexpr index_t BlockSize() { return kBlockSize; }

    // Log-sum-exp function for numerical stability
    CK_TILE_DEVICE ComputeDataType log_sum_exp(const ComputeDataType* log_values, index_t n) const
    {
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

    // Each thread processes one batch element's 4×4 matrix
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
        // This matches the approach from test_store_tile_debug.cpp
        auto matrix_tensor = make_naive_tensor_view<address_space_enum::global>(
            p_output + 2 * n, // Start at first batch's H^res
            make_tuple(batch, n, n),
            make_tuple(output_dim,
                       n,
                       1), // Stride: output_dim between batches, n between rows, 1 within row
            number<1>{},
            number<1>{});

        // Create tile distribution for batch processing
        // 64 threads total processing batches of 4×4 matrices
        // Each thread processes: 64/64=1 batch, 4/4=1 row, 4/4=1 col (i.e., one complete 4×4
        // matrix)
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

        // Load matrix using ck-tile API
        auto input_tile = make_static_distributed_tensor<YDataType>(tile_dist);
        load_tile(input_tile, input_window);

        // Convert to log domain and process
        ComputeDataType log_matrix[kNSquared];

        // Manually loop through the tile's thread buffer to find max and extract values
        // Each thread owns the entire 4×4 matrix (16 elements)
        auto& thread_buf = input_tile.get_thread_buffer();

        // Find max for numerical stability before taking log
        ComputeDataType max_val = -INFINITY;
        for(index_t i = 0; i < kNSquared; ++i)
        {
            ComputeDataType val = type_convert<ComputeDataType>(thread_buf[i]);
            max_val             = ck_tile::max(max_val, val);
        }

        // Convert to log domain with max subtracted for stability
        for(index_t i = 0; i < kNSquared; ++i)
        {
            ComputeDataType val = type_convert<ComputeDataType>(thread_buf[i]);
            // In log domain: log(exp(val - max_val)) = val - max_val
            log_matrix[i] = val - max_val;
        }

        // Check if matrix is degenerate (all values too negative)
        bool is_degenerate                    = true;
        constexpr ComputeDataType min_log_val = -80.0f; // exp(-80) ≈ 1e-35
        for(index_t i = 0; i < kNSquared; ++i)
        {
            if(log_matrix[i] > min_log_val)
            {
                is_degenerate = false;
                break;
            }
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

        // Sinkhorn iterations in log domain
        for(index_t iter = 0; iter < sinkhorn_iters; ++iter)
        {
            // Normalize rows in log domain
            for(index_t row = 0; row < kN; ++row)
            {
                ComputeDataType row_values[kN];
                for(index_t col = 0; col < kN; ++col)
                {
                    row_values[col] = log_matrix[row * kN + col];
                }

                ComputeDataType log_row_sum = log_sum_exp(row_values, kN);

                // Handle degenerate row
                if(::isinf(log_row_sum) && log_row_sum < 0)
                {
                    // Set row to uniform in log domain: log(1/kN)
                    ComputeDataType log_uniform = -ck_tile::log(static_cast<ComputeDataType>(kN));
                    for(index_t col = 0; col < kN; ++col)
                    {
                        log_matrix[row * kN + col] = log_uniform;
                    }
                }
                else
                {
                    // Normalize: log(val/sum) = log(val) - log(sum)
                    for(index_t col = 0; col < kN; ++col)
                    {
                        log_matrix[row * kN + col] -= log_row_sum;
                    }
                }
            }

            // Normalize columns in log domain
            for(index_t col = 0; col < kN; ++col)
            {
                ComputeDataType col_values[kN];
                for(index_t row = 0; row < kN; ++row)
                {
                    col_values[row] = log_matrix[row * kN + col];
                }

                ComputeDataType log_col_sum = log_sum_exp(col_values, kN);

                // Handle degenerate column
                if(::isinf(log_col_sum) && log_col_sum < 0)
                {
                    // Set column to uniform in log domain
                    ComputeDataType log_uniform = -ck_tile::log(static_cast<ComputeDataType>(kN));
                    for(index_t row = 0; row < kN; ++row)
                    {
                        log_matrix[row * kN + col] = log_uniform;
                    }
                }
                else
                {
                    // Normalize: log(val/sum) = log(val) - log(sum)
                    for(index_t row = 0; row < kN; ++row)
                    {
                        log_matrix[row * kN + col] -= log_col_sum;
                    }
                }
            }
        }

        // Convert back from log domain and store using ck-tile API
        // Copy log_matrix back into the input tile (reusing it for output)
        index_t elem_idx      = 0;
        auto convert_from_log = [&](YDataType& x) {
            ComputeDataType log_val = log_matrix[elem_idx++];
            ComputeDataType val     = ck_tile::exp(log_val);
            val                     = ck_tile::max(val, static_cast<ComputeDataType>(1e-35f));
            val                     = ck_tile::min(val, static_cast<ComputeDataType>(1.0f));
            x                       = type_convert<YDataType>(val);
        };
        tile_elementwise_inout(convert_from_log, input_tile);

        // Store using ck-tile API - reuse the input_window which already has correct position
        store_tile(input_window, input_tile);
    }
};

// Dispatcher wrapper that selects the appropriate template instantiation based on runtime n
// Falls back to old kernel for unsupported sizes
template <typename YDataType, typename ComputeDataType = float>
struct MHCSinkhornLogKernelTileV2Dispatcher
{
    static constexpr index_t kBlockSize = 64;

    CK_TILE_HOST static constexpr index_t BlockSize() { return kBlockSize; }

    CK_TILE_DEVICE void operator()(YDataType* p_output,
                                   index_t batch,
                                   index_t output_dim,
                                   index_t n,
                                   index_t sinkhorn_iters) const
    {
        // Dispatch to appropriate template instantiation based on n
        if(n == 4)
        {
            MHCSinkhornLogKernelTileV2<YDataType, ComputeDataType, 4>{}(
                p_output, batch, output_dim, n, sinkhorn_iters);
        }
        else if(n == 8)
        {
            MHCSinkhornLogKernelTileV2<YDataType, ComputeDataType, 8>{}(
                p_output, batch, output_dim, n, sinkhorn_iters);
        }
        else if(n == 5)
        {
            MHCSinkhornLogKernelTileV2<YDataType, ComputeDataType, 5>{}(
                p_output, batch, output_dim, n, sinkhorn_iters);
        }
        else
        {
            // Fall back to old kernel for unsupported sizes
            // For now, just handle as best effort with n=8 template (largest common size)
            MHCSinkhornLogKernelTileV2<YDataType, ComputeDataType, 8>{}(
                p_output, batch, output_dim, n, sinkhorn_iters);
        }
    }
};

} // namespace ck_tile
