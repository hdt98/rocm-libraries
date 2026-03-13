// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Standard Sinkhorn kernel using ck-tile style API (Version 2)
// This version properly uses ck-tile's tile abstraction for loading and storing
// Handles packed MHC data layout: [H^pre (1xn), H^post (1xn), H^res (nxn)]
//
// Key design: Each thread processes ONE complete n×n matrix from ONE batch element
// Uses ck-tile for efficient tile-based loading, vectorized writes for storing
// Template parameter N allows compile-time optimization for specific matrix sizes
template <typename YDataType, typename ComputeDataType = float, index_t N_ = 4>
struct MHCSinkhornKernelTileV2
{
    static constexpr index_t kN         = N_;
    static constexpr index_t kNSquared  = kN * kN;
    static constexpr index_t kBlockSize = 64;

    CK_TILE_HOST static constexpr index_t BlockSize() { return kBlockSize; }

    // Each thread processes one batch element's n×n matrix
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
        // 64 threads total processing batches of n×n matrices
        // Each thread processes: 64/64=1 batch, n/n=1 row, n/n=1 col (i.e., one complete n×n
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

        // Process in standard (non-log) domain
        ComputeDataType matrix[kNSquared];

        // Manually loop through the tile's thread buffer to extract values
        // Each thread owns the entire n×n matrix
        auto& thread_buf = input_tile.get_thread_buffer();

        // Find max for numerical stability (like old kernel)
        ComputeDataType max_val = -INFINITY;
        for(index_t i = 0; i < kNSquared; ++i)
        {
            ComputeDataType val = type_convert<ComputeDataType>(thread_buf[i]);
            max_val             = ck_tile::max(max_val, val);
        }

        // Exponentiate with max subtracted for numerical stability (like old kernel)
        for(index_t i = 0; i < kNSquared; ++i)
        {
            ComputeDataType val = type_convert<ComputeDataType>(thread_buf[i]);
            matrix[i]           = ck_tile::exp(val - max_val);
        }

        // Check if matrix is degenerate (all values near zero) - use matrix sum like old kernel
        ComputeDataType matrix_sum = 0.0f;
        for(index_t i = 0; i < kNSquared; ++i)
        {
            matrix_sum += matrix[i];
        }

        constexpr ComputeDataType min_matrix_sum = 1e-30f;
        bool is_degenerate                       = (matrix_sum < min_matrix_sum);

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

        // Sinkhorn iterations in standard domain
        for(index_t iter = 0; iter < sinkhorn_iters; ++iter)
        {
            // Normalize rows (use same thresholds as old kernel)
            for(index_t row = 0; row < kN; ++row)
            {
                ComputeDataType row_sum = 0.0f;
                for(index_t col = 0; col < kN; ++col)
                {
                    row_sum += matrix[row * kN + col];
                }

                if(row_sum > 1e-12f) // Use same threshold as old kernel
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

            // Normalize columns (use same thresholds as old kernel)
            for(index_t col = 0; col < kN; ++col)
            {
                ComputeDataType col_sum = 0.0f;
                for(index_t row = 0; row < kN; ++row)
                {
                    col_sum += matrix[row * kN + col];
                }

                if(col_sum > 1e-12f) // Use same threshold as old kernel
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

        // Store result back using ck-tile API
        // Copy matrix back into the input tile (reusing it for output)
        index_t elem_idx  = 0;
        auto store_result = [&](YDataType& x) {
            ComputeDataType val = matrix[elem_idx++];
            // Don't clamp to 1e-35f minimum like I was doing - just store the result
            x = type_convert<YDataType>(val);
        };
        tile_elementwise_inout(store_result, input_tile);

        // Store using ck-tile API - reuse the input_window which already has correct position
        store_tile(input_window, input_tile);
    }
};

// Dispatcher wrapper that selects the appropriate template instantiation based on runtime n
// Falls back to old kernel for unsupported sizes
template <typename YDataType, typename ComputeDataType = float>
struct MHCSinkhornKernelTileV2Dispatcher
{
    static constexpr index_t kBlockSize = 64;

    CK_TILE_HOST static constexpr index_t BlockSize() { return kBlockSize; }

    // Check if V2 kernel supports the given n value
    CK_TILE_HOST static constexpr bool IsSupportedArgument(index_t n)
    {
        return (n == 4) || (n == 5) || (n == 8); // Supported template instantiations
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
            MHCSinkhornKernelTileV2<YDataType, ComputeDataType, 4>{}(
                p_output, batch, output_dim, n, sinkhorn_iters);
        }
        else if(n == 8)
        {
            MHCSinkhornKernelTileV2<YDataType, ComputeDataType, 8>{}(
                p_output, batch, output_dim, n, sinkhorn_iters);
        }
        else if(n == 5)
        {
            MHCSinkhornKernelTileV2<YDataType, ComputeDataType, 5>{}(
                p_output, batch, output_dim, n, sinkhorn_iters);
        }
        else
        {
            // This should not happen if IsSupportedArgument is checked properly
            // For safety, handle as best effort with n=8 template (largest common size)
            MHCSinkhornKernelTileV2<YDataType, ComputeDataType, 8>{}(
                p_output, batch, output_dim, n, sinkhorn_iters);
        }
    }
};

} // namespace ck_tile
