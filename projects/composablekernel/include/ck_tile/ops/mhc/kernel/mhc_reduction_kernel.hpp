// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
#include "ck_tile/ops/reduce/pipeline/reduce2d_shape.hpp"
#include "ck_tile/ops/reduce/pipeline/reduce2d_problem.hpp"
#include "ck_tile/ops/reduce/pipeline/reduce2d_default_policy.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_problem.hpp"

// MHC Reduction Kernel
// ======================================================================
// Implementing equations 16, 17 and 18 of https://arxiv.org/abs/2512.24880
//
// Pattern:
// 1. Workspace tensor layout: [batch, grid_k, output_dim]
// 2. Output tensor layout: [batch * output_dim] (1D merged for coalesced writes)
// 3. Each block handles Block_M elements from the merged dimension
// 4. Reduce over grid_k dimension with block_reduce2d
// 5. Apply warp-level and cross-warp synchronization
// 6. Store results with activation/scaling using coalesced tile writes

namespace ck_tile {

template <typename Problem_, typename Activation_ = element_wise::Sigmoid>
struct MHCReductionKernel
{
    using Activation      = ck_tile::remove_cvref_t<Activation_>;
    using Problem         = ck_tile::remove_cvref_t<Problem_>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;

    // Reduction shape for the input processing
    // Block_M covers the merged [batch * output_dim] dimension
    // Block_N covers the grid_k reduction dimension
    using BlockWarps  = ck_tile::sequence<1, 1>;
    using BlockTile   = ck_tile::sequence<64, 1>; // Block_M=64, Block_N=1
    using WarpTile    = ck_tile::sequence<64, 1>;
    using ThreadTile  = ck_tile::sequence<1, 1>;
    using ReduceShape = ck_tile::Reduce2dShape<BlockWarps, BlockTile, WarpTile, ThreadTile>;

    static constexpr index_t kBlockSize = ReduceShape::BlockSize;
    static constexpr bool kOutputIndex  = false;

    // Workspace layout: [batch, grid_k, output_dim]
    // Reduce over grid_k (dimension 1), keep batch and output_dim (dimensions 0, 2)
    static constexpr auto kept_dims   = ck_tile::sequence<0, 2>{};
    static constexpr auto reduce_dims = ck_tile::sequence<1>{};
    static constexpr index_t Rank     = 3;

    using ReduceProblem = ck_tile::Reduce2dProblem<ComputeDataType,
                                                   ComputeDataType,
                                                   ComputeDataType,
                                                   ReduceShape,
                                                   ck_tile::ReduceOp::Add,
                                                   decltype(kept_dims),
                                                   decltype(reduce_dims),
                                                   Rank,
                                                   kOutputIndex>;
    using Policy        = ck_tile::Reduce2dDefaultPolicy;

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? kBlockSize / 2 : kBlockSize;
    }

    CK_TILE_DEVICE void operator()(const ComputeDataType* p_workspace,
                                   const ComputeDataType* p_partial_norms,
                                   YDataType* p_output,
                                   index_t batch,
                                   index_t nC,
                                   index_t output_dim,
                                   index_t n,
                                   index_t grid_k,
                                   float alpha_pre,
                                   float alpha_post,
                                   float alpha_res,
                                   float bias) const
    {
        using S       = ReduceShape;
        const auto iM = get_block_id() * S::Block_M;

        // Workspace layout: [batch, grid_k, output_dim]
        const index_t total_output = batch * output_dim;

        // Create 3D workspace tensor view
        auto workspace_tensor_3d = make_naive_tensor_view<address_space_enum::global>(
            p_workspace,
            make_tuple(batch, grid_k, output_dim),
            make_tuple(grid_k * output_dim, output_dim, 1),
            number<1>{},
            number<1>{});

        // Transform to merge batch and output_dim into first dimension
        const auto merge_transform = make_merge_transform(make_tuple(batch, output_dim));
        auto workspace_tensor =
            transform_tensor_view(workspace_tensor_3d,
                                  make_tuple(merge_transform, make_pass_through_transform(grid_k)),
                                  make_tuple(sequence<0, 2>{}, sequence<1>{}),
                                  make_tuple(sequence<0>{}, sequence<1>{}));

        // Pad tensor to block tile size
        auto workspace_padded =
            pad_tensor_view(workspace_tensor,
                            make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                            sequence<false, false>{});

        // Create tile window for input (workspace)
        auto x_window =
            make_tile_window(workspace_padded,
                             make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                             {iM, 0},
                             Policy::template MakeXBlockTileDistribution<ReduceProblem>());

        __shared__ char smem[Policy::template GetSmemSize<ReduceProblem>()];

        // Calculate number of iterations over grid_k dimension
        index_t num_n_tile_iteration =
            amd_wave_read_first_lane(integer_divide_ceil(grid_k, S::Block_N));

        // Get block reduction operators
        auto block_reduce2d      = Policy::template GetBlockReduce2d<ReduceProblem>();
        auto block_reduce2d_sync = Policy::template GetBlockReduce2dSync<ReduceProblem>();
        auto block_reduce2d_cross_warp_sync =
            Policy::template GetBlockReduce2dCrossWarpSync<ReduceProblem>();

        // Initialize reduction accumulator
        using XTensorType = decltype(load_tile(x_window));
        auto y_compute    = block_reduce2d.template MakeYBlockTile<XTensorType>();
        auto reduce_func  = ck_tile::ReduceOp::Add{};
        set_tile(y_compute, reduce_func.template GetIdentityValue<ComputeDataType>());

        // Main reduction loop - iterate over grid_k dimension
        for(int iN = amd_wave_read_first_lane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x = load_tile(x_window);
            block_reduce2d(x, y_compute, reduce_func);
            move_tile_window(x_window, {0, S::Block_N});
        }

        // Warp-level reduction
        block_reduce2d_sync(y_compute, reduce_func);

        // Cross-warp reduction
        block_reduce2d_cross_warp_sync(y_compute, smem, reduce_func);

        // Apply element-wise transformations in place
        // We iterate through the distributed spans and apply transforms based on position
        constexpr auto result_spans = decltype(y_compute)::get_distributed_spans();

        sweep_tile_span(result_spans[number<0>{}], [&](auto idx0) {
            // Get the tile index from the distributed index
            const auto tile_idx = get_x_indices_from_distributed_indices(
                y_compute.get_tile_distribution(), make_tuple(idx0));

            const index_t local_idx  = tile_idx.at(number<0>{});
            const index_t global_idx = iM + local_idx;

            // Get reduced value using tile's operator[]
            ComputeDataType value = y_compute[make_tuple(idx0)];
            ComputeDataType final_value;

            // Check bounds
            if(global_idx >= total_output)
            {
                final_value = type_convert<ComputeDataType>(0);
            }
            else
            {
                const index_t global_m = global_idx / output_dim;
                const index_t global_n = global_idx % output_dim;

                // Compute norm from partial norms for this batch element
                //
                // Note: the GEMM/norm kernel, using the split-k setup, writes the partial norm
                //  in a coalesced way [grid_k, batch]. However this kernel is setup to reduce
                //. per batch, hence reads the partial norm in a strided way [batch, grid_k]
                ComputeDataType sum_squares = 0.0f;
                for(index_t k = 0; k < grid_k; ++k)
                {
                    sum_squares += p_partial_norms[k * batch + global_m];
                }
                const ComputeDataType sqrt_nC_val = ck_tile::sqrt(static_cast<ComputeDataType>(nC));
                ComputeDataType norm              = ck_tile::sqrt(sum_squares) / sqrt_nC_val;
                norm                              = (norm > 1e-12f) ? norm : 1.0f;

                // Apply activation and scaling based on output section
                if(global_n < n)
                {
                    // H^pre: (alpha_pre/norm) * sigma(value) + bias
                    YDataType activated;
                    Activation{}(activated, type_convert<YDataType>(value));
                    final_value =
                        (alpha_pre / norm) * type_convert<ComputeDataType>(activated) + bias;
                }
                else if(global_n < 2 * n)
                {
                    // H^post: (alpha_post/norm) * 2*sigma(value) + bias
                    YDataType activated;
                    Activation{}(activated, type_convert<YDataType>(value));
                    final_value =
                        (alpha_post / norm) * 2.0f * type_convert<ComputeDataType>(activated) +
                        bias;
                }
                else
                {
                    // H^res: (alpha_res/norm) * value + bias
                    final_value = (alpha_res / norm) * value + bias;
                }
            }

            // Store transformed value back using tile's operator()
            y_compute(make_tuple(idx0)) = final_value;
        });

        // Create 1D output tensor view for coalesced store
        // Output layout: [batch * output_dim] (merged)
        // Vector size for coalesced access
        constexpr index_t output_vector_size = ck_tile::min(
            static_cast<index_t>(16 / sizeof(YDataType)), static_cast<index_t>(S::Block_M));

        auto y_tensor_view =
            make_naive_tensor_view<address_space_enum::global>(p_output,
                                                               make_tuple(total_output),
                                                               make_tuple(1),
                                                               number<output_vector_size>{},
                                                               number<1>{});

        // Pad output tensor to block tile size
        auto y_tensor_padded =
            pad_tensor_view(y_tensor_view, make_tuple(number<S::Block_M>{}), sequence<false>{});

        // Create tile window for output with the same distribution as y_compute
        // This ensures store_tile correctly maps tile elements to memory
        auto y_window = make_tile_window(y_tensor_padded,
                                         make_tuple(number<S::Block_M>{}),
                                         {iM},
                                         y_compute.get_tile_distribution());

        // Store with coalesced access pattern
        store_tile(y_window, cast_tile<YDataType>(y_compute));
    }
};

} // namespace ck_tile
