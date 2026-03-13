// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Non-log domain Sinkhorn kernel with support for arbitrary n
// Standard Sinkhorn-Knopp algorithm in probability space
template <typename YDataType, typename ComputeDataType = float>
struct MHCSinkhornKernel
{
    static constexpr index_t kMaxN      = 16; // Support up to n=16
    static constexpr index_t kBlockSize = 64;

    CK_TILE_HOST static constexpr index_t BlockSize() { return kBlockSize; }

    // Each thread processes one batch element's n×n matrix
    CK_TILE_DEVICE void operator()(YDataType* p_output,
                                   index_t batch,
                                   index_t output_dim,
                                   index_t n,
                                   index_t sinkhorn_iters) const
    {
        const index_t batch_idx = get_block_id() * get_block_size() + get_thread_id();

        if(batch_idx >= batch)
            return;

        // H^res starts at index 2*n for this batch element
        const index_t h_res_start = batch_idx * output_dim + 2 * n;
        const index_t n_squared   = n * n;

        // Load matrix into local memory
        ComputeDataType matrix[kMaxN * kMaxN];

        // Find max for numerical stability
        ComputeDataType max_val = -INFINITY;
        for(index_t i = 0; i < n_squared; ++i)
        {
            ComputeDataType val = type_convert<ComputeDataType>(p_output[h_res_start + i]);
            max_val             = ck_tile::max(max_val, val);
        }

        // Exponentiate with max subtracted for numerical stability
        for(index_t i = 0; i < n_squared; ++i)
        {
            ComputeDataType val = type_convert<ComputeDataType>(p_output[h_res_start + i]);
            matrix[i]           = ck_tile::exp(val - max_val);
        }

        // Check if matrix is degenerate (all values near zero)
        ComputeDataType matrix_sum = 0.0f;
        for(index_t i = 0; i < n_squared; ++i)
        {
            matrix_sum += matrix[i];
        }

        constexpr ComputeDataType min_matrix_sum = 1e-30f;
        if(matrix_sum < min_matrix_sum)
        {
            // Set to uniform distribution
            ComputeDataType uniform_val = 1.0f / static_cast<ComputeDataType>(n_squared);
            for(index_t i = 0; i < n_squared; ++i)
            {
                p_output[h_res_start + i] = type_convert<YDataType>(uniform_val);
            }
            return;
        }

        // Sinkhorn iterations
        for(index_t iter = 0; iter < sinkhorn_iters; ++iter)
        {
            // Normalize rows
            for(index_t row = 0; row < n; ++row)
            {
                ComputeDataType row_sum = 0.0f;
                for(index_t col = 0; col < n; ++col)
                {
                    row_sum += matrix[row * n + col];
                }

                if(row_sum > 1e-12f)
                {
                    for(index_t col = 0; col < n; ++col)
                    {
                        matrix[row * n + col] /= row_sum;
                    }
                }
                else
                {
                    // Set to uniform distribution if row sum is too small
                    ComputeDataType uniform_val = 1.0f / static_cast<ComputeDataType>(n);
                    for(index_t col = 0; col < n; ++col)
                    {
                        matrix[row * n + col] = uniform_val;
                    }
                }
            }

            // Normalize columns
            for(index_t col = 0; col < n; ++col)
            {
                ComputeDataType col_sum = 0.0f;
                for(index_t row = 0; row < n; ++row)
                {
                    col_sum += matrix[row * n + col];
                }

                if(col_sum > 1e-12f)
                {
                    for(index_t row = 0; row < n; ++row)
                    {
                        matrix[row * n + col] /= col_sum;
                    }
                }
                else
                {
                    // Set to uniform distribution if column sum is too small
                    ComputeDataType uniform_val = 1.0f / static_cast<ComputeDataType>(n);
                    for(index_t row = 0; row < n; ++row)
                    {
                        matrix[row * n + col] = uniform_val;
                    }
                }
            }
        }

        // Write back to global memory
        for(index_t i = 0; i < n_squared; ++i)
        {
            p_output[h_res_start + i] = type_convert<YDataType>(matrix[i]);
        }
    }
};

} // namespace ck_tile
