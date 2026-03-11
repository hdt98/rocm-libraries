// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Dedicated Sinkhorn kernel for MHC H^res normalization
// Direct memory access version - each thread independently processes one 4×4 matrix
template <typename YDataType, typename ComputeDataType = float>
struct MHCSinkhornKernel
{
    static constexpr index_t kN         = 4;
    static constexpr index_t kNSquared  = kN * kN;
    static constexpr index_t kBlockSize = 64; // For __launch_bounds__

    CK_TILE_HOST static constexpr index_t BlockSize() { return kBlockSize; }

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

        // Direct memory access: load 4×4 matrix into registers with vectorization
        ComputeDataType matrix[kNSquared];

        // Vectorized load from global memory (row-major layout)
        // Load 4 rows of 4 elements each using vector loads for coalescing
        constexpr index_t kVecSize = 4;
        using VecType              = ext_vector_t<YDataType, kVecSize>;

        for(index_t row = 0; row < kN; ++row)
        {
            VecType vec = *c_style_pointer_cast<const VecType*>(p_output + h_res_start + row * kN);
            for(index_t col = 0; col < kN; ++col)
            {
                matrix[row * kN + col] = type_convert<ComputeDataType>(vec[col]);
            }
        }

        // Apply Sinkhorn-Knopp algorithm
        // Find max for numerical stability
        ComputeDataType max_val = matrix[0];
        for(index_t i = 1; i < kNSquared; ++i)
        {
            max_val = ck_tile::max(max_val, matrix[i]);
        }

        // Exponentiate with max subtracted
        for(index_t i = 0; i < kNSquared; ++i)
        {
            matrix[i] = ck_tile::exp(matrix[i] - max_val);
        }

        // Check if matrix is degenerate (all values near zero)
        // This is a safety check for extreme numerical underflow
        ComputeDataType matrix_sum = 0.0f;
        for(index_t i = 0; i < kNSquared; ++i)
        {
            matrix_sum += matrix[i];
        }

        // Only treat as degenerate if the sum is EXTREMELY small
        // This prevents valid small values from being incorrectly handled
        constexpr ComputeDataType min_matrix_sum = 1e-30f;
        if(matrix_sum < min_matrix_sum)
        {
            ComputeDataType uniform_val = 1.0f / static_cast<ComputeDataType>(kNSquared);
            for(index_t i = 0; i < kNSquared; ++i)
            {
                matrix[i] = uniform_val;
            }
            // Skip Sinkhorn iterations for degenerate case
            // Store back to global memory
            for(index_t row = 0; row < kN; ++row)
            {
                using VecTypeUniform = ext_vector_t<YDataType, kN>;
                VecTypeUniform vec{};
                for(index_t col = 0; col < kN; ++col)
                {
                    vec[col] = type_convert<YDataType>(uniform_val);
                }
                *c_style_pointer_cast<VecTypeUniform*>(p_output + h_res_start + row * kN) = vec;
            }
            return;
        }

        // Sinkhorn iterations
        for(index_t iter = 0; iter < sinkhorn_iters; ++iter)
        {
            // Normalize rows
            for(index_t row = 0; row < kN; ++row)
            {
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

            // Normalize columns
            for(index_t col = 0; col < kN; ++col)
            {
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

        // Vectorized write back to global memory
        for(index_t row = 0; row < kN; ++row)
        {
            VecType vec{};
            for(index_t col = 0; col < kN; ++col)
            {
                vec[col] = type_convert<YDataType>(matrix[row * kN + col]);
            }
            *c_style_pointer_cast<VecType*>(p_output + h_res_start + row * kN) = vec;
        }
    }
};

} // namespace ck_tile
