// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Log-domain Sinkhorn kernel for MHC H^res normalization
// More numerically stable than standard Sinkhorn for extreme values
template <typename YDataType, typename ComputeDataType = float>
struct MHCSinkhornLogKernel
{
    static constexpr index_t kN         = 4;
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

        // Handle case where all values are -inf
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

    CK_TILE_DEVICE void operator()(YDataType* p_output,
                                   index_t batch,
                                   index_t output_dim,
                                   index_t n,
                                   index_t sinkhorn_iters) const
    {
        // Each thread processes one batch element's 4×4 matrix independently
        const index_t batch_idx = get_block_id() * get_block_size() + get_thread_id();

        if(batch_idx >= batch)
            return;

        // H^res starts at index 2*n for this batch element
        const index_t h_res_start = batch_idx * output_dim + 2 * n;

        // Load matrix and convert to log domain
        ComputeDataType log_matrix[kNSquared];

        // Vectorized load from global memory
        constexpr index_t kVecSize = 4;
        using VecType              = ext_vector_t<YDataType, kVecSize>;

        // Find max for numerical stability before taking log
        ComputeDataType max_val = -INFINITY;
        for(index_t row = 0; row < kN; ++row)
        {
            VecType vec = *c_style_pointer_cast<const VecType*>(p_output + h_res_start + row * kN);
            for(index_t col = 0; col < kN; ++col)
            {
                ComputeDataType val = type_convert<ComputeDataType>(vec[col]);
                max_val             = ck_tile::max(max_val, val);
            }
        }

        // Convert to log domain with max subtracted for stability
        for(index_t row = 0; row < kN; ++row)
        {
            VecType vec = *c_style_pointer_cast<const VecType*>(p_output + h_res_start + row * kN);
            for(index_t col = 0; col < kN; ++col)
            {
                ComputeDataType val = type_convert<ComputeDataType>(vec[col]);
                // In log domain: log(exp(val - max_val)) = val - max_val
                log_matrix[row * kN + col] = val - max_val;
            }
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
            for(index_t row = 0; row < kN; ++row)
            {
                VecType vec{};
                for(index_t col = 0; col < kN; ++col)
                {
                    vec[col] = type_convert<YDataType>(uniform_val);
                }
                *c_style_pointer_cast<VecType*>(p_output + h_res_start + row * kN) = vec;
            }
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

        // Convert back from log domain and write to global memory
        for(index_t row = 0; row < kN; ++row)
        {
            VecType vec{};
            for(index_t col = 0; col < kN; ++col)
            {
                // Convert from log domain: exp(log_val)
                ComputeDataType log_val = log_matrix[row * kN + col];
                ComputeDataType val     = ck_tile::exp(log_val);

                // Clamp to reasonable range to avoid numerical issues
                val = ck_tile::max(val, static_cast<ComputeDataType>(1e-35f));
                val = ck_tile::min(val, static_cast<ComputeDataType>(1.0f));

                vec[col] = type_convert<YDataType>(val);
            }
            *c_style_pointer_cast<VecType*>(p_output + h_res_start + row * kN) = vec;
        }
    }
};

} // namespace ck_tile
