// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Naive Sinkhorn-Knopp iteration with the whole matrix loaded within the thread
template <typename InDistributedTensor,
          typename OutDistributedTensor,
          typename ComputeDataType = float>
CK_TILE_DEVICE void sinkhorn_knopp_naive_full(const InDistributedTensor& input_tile,
                                              OutDistributedTensor& out_tile,
                                              index_t iterations)
{
    using InDataType  = typename InDistributedTensor::DataType;
    using OutDataType = typename OutDistributedTensor::DataType;

    // DEBUG: Commented out for normal testing
    // bool is_batch_0 = (get_block_id() * get_block_size() + get_thread_id()) == 0;

    // Run the first steps iteration of the Sinkhorn-Knopp algorithm
    // Exponentiate the input to make it strictly positive
    auto exp_func = [](InDataType x) -> ComputeDataType {
        return ck_tile::exp(type_convert<ComputeDataType>(x));
    };

    auto compute_tile    = tile_elementwise_in(exp_func, input_tile);
    constexpr auto spans = compute_tile.get_distributed_spans();

    // Loop the Sinkhorn-Knopp normalization for rows and columns
    for(int i = 0; i < iterations; i++)
    {
        sweep_tile_span(spans[number<0>{}], [&](const auto idx0) {
            // 1. Compute row sums
            ComputeDataType row_sum = 0.0;
            sweep_tile_span(spans[number<1>{}], [&](const auto idx1) {
                constexpr auto idx = make_tuple(idx0, idx1);
                row_sum += compute_tile(idx);
            });

            // 2. Divide values in the row by the sum
            sweep_tile_span(spans[number<1>{}], [&](const auto idx1) {
                constexpr auto c_idx = make_tuple(idx0, idx1);
                compute_tile(c_idx)  = compute_tile(c_idx) / row_sum;
            });
        });

        // Repeat for columns
        sweep_tile_span(spans[number<1>{}], [&](const auto idx1) {
            ComputeDataType col_sum = 0.0;
            sweep_tile_span(spans[number<0>{}], [&](const auto idx0) {
                constexpr auto idx = make_tuple(idx0, idx1);
                col_sum += compute_tile(idx);
            });

            sweep_tile_span(spans[number<0>{}], [&](const auto idx0) {
                constexpr auto idx = make_tuple(idx0, idx1);
                compute_tile(idx)  = compute_tile(idx) / col_sum;
            });
        });
    }

    // Copy compute_tile to out_tile element by element
    // Cannot use store_tile because out_tile is a distributed tensor, not a tile window
    index_t batch_idx = get_block_id() * get_block_size() + get_thread_id();

    sweep_tile_span(spans[number<0>{}], [&](const auto idx0) {
        sweep_tile_span(spans[number<1>{}], [&](const auto idx1) {
            constexpr auto idx = make_tuple(idx0, idx1);

            if(batch_idx == 0 || batch_idx == 1)
            {
                // Get the actual matrix indices for debugging
                const auto tile_idx_compute = get_x_indices_from_distributed_indices(
                    compute_tile.get_tile_distribution(), idx);
                const auto tile_idx_out =
                    get_x_indices_from_distributed_indices(out_tile.get_tile_distribution(), idx);

                printf("[COPY DEBUG Batch %ld] compute[%ld,%ld]=%f -> out[%ld,%ld]\n",
                       static_cast<long>(batch_idx),
                       static_cast<long>(tile_idx_compute.at(number<0>{})),
                       static_cast<long>(tile_idx_compute.at(number<1>{})),
                       static_cast<float>(compute_tile(idx)),
                       static_cast<long>(tile_idx_out.at(number<0>{})),
                       static_cast<long>(tile_idx_out.at(number<1>{})));
            }

            out_tile(idx) = type_convert<OutDataType>(compute_tile(idx));
        });
    });
}

// Distributed version of Sinkhorn-Knopp for data spread across multiple threads
// Each thread owns part of the matrix, we gather to shared memory, process, then scatter
template <typename InDistributedTensor,
          typename OutDistributedTensor,
          typename ComputeDataType = float>
CK_TILE_DEVICE void sinkhorn_knopp_naive_full_distributed(InDistributedTensor& input_tile,
                                                          OutDistributedTensor& out_tile,
                                                          index_t iterations)
{
    using OutDataType = typename OutDistributedTensor::DataType;

    // Assume 4×4 matrix for MHC
    constexpr index_t kN        = 4;
    constexpr index_t kNSquared = kN * kN;

    // Each thread uses registers to store its own 4×4 matrix
    // No shared memory needed - each thread works independently
    ComputeDataType reg_matrix[kNSquared];

    // Step 1: Gather data from distributed tensor to registers
    const auto spans = input_tile.get_distributed_spans();
    sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
        sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
            constexpr auto idx = make_tuple(idx0, idx1);
            const auto tile_idx =
                get_x_indices_from_distributed_indices(input_tile.get_tile_distribution(), idx);
            const index_t row        = tile_idx.at(number<0>{});
            const index_t col        = tile_idx.at(number<1>{});
            const index_t linear_idx = row * kN + col;
            reg_matrix[linear_idx]   = type_convert<ComputeDataType>(input_tile(idx));
        });
    });

    // Step 2: Each thread independently performs Sinkhorn on its own matrix
    // Find max for numerical stability
    ComputeDataType max_val = reg_matrix[0];
    for(index_t i = 1; i < kNSquared; ++i)
    {
        max_val = ck_tile::max(max_val, reg_matrix[i]);
    }

    // Exponentiate with max subtracted
    for(index_t i = 0; i < kNSquared; ++i)
    {
        reg_matrix[i] = ck_tile::exp(reg_matrix[i] - max_val);
    }

    // Sinkhorn iterations
    for(index_t iter = 0; iter < iterations; ++iter)
    {
        // Normalize rows
        for(index_t row = 0; row < kN; ++row)
        {
            ComputeDataType row_sum = 0.0;
            for(index_t col = 0; col < kN; ++col)
            {
                row_sum += reg_matrix[row * kN + col];
            }

            if(row_sum > 1e-12)
            {
                for(index_t col = 0; col < kN; ++col)
                {
                    reg_matrix[row * kN + col] /= row_sum;
                }
            }
        }

        // Normalize columns
        for(index_t col = 0; col < kN; ++col)
        {
            ComputeDataType col_sum = 0.0;
            for(index_t row = 0; row < kN; ++row)
            {
                col_sum += reg_matrix[row * kN + col];
            }

            if(col_sum > 1e-12)
            {
                for(index_t row = 0; row < kN; ++row)
                {
                    reg_matrix[row * kN + col] /= col_sum;
                }
            }
        }
    }

    // Step 3: Scatter results back to distributed tensor
    sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
        sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
            constexpr auto idx = make_tuple(idx0, idx1);
            const auto tile_idx =
                get_x_indices_from_distributed_indices(out_tile.get_tile_distribution(), idx);
            const index_t row        = tile_idx.at(number<0>{});
            const index_t col        = tile_idx.at(number<1>{});
            const index_t linear_idx = row * kN + col;
            out_tile(idx)            = type_convert<OutDataType>(reg_matrix[linear_idx]);
        });
    });
}

} // namespace ck_tile
