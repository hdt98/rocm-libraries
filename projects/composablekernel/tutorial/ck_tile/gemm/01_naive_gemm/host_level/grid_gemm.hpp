// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

// GridGemm: the host-level dispatcher that maps one thread block to one tile of C.
//
// Responsibilities:
//   1. Extract M, N, K from the tensor descriptors.
//   2. Map the linear block ID to a 2D tile coordinate (iM, iN).
//   3. Create tile windows over A and B at the correct offset.
//   4. Allocate LDS and call the block pipeline.
//   5. Cast the accumulator (AccDataType) to CDataType, apply CElementFunction, store to C.
template <typename Problem, typename Policy>
struct GridGemm
{
    using ADataType        = typename Problem::ADataType;
    using BDataType        = typename Problem::BDataType;
    using CDataType        = typename Problem::CDataType;
    using AccDataType      = typename Problem::AccDataType;
    using CElementFunction = typename Problem::CElementFunction;

    static constexpr auto kMPerBlock = Policy::kMPerBlock;
    static constexpr auto kNPerBlock = Policy::kNPerBlock;
    static constexpr auto kKPerBlock = Policy::kKPerBlock;

    template <typename AGridTensorView, typename BGridTensorView, typename CGridTensorView>
    CK_TILE_DEVICE void operator()(const AGridTensorView& a_grid,
                                   const BGridTensorView& b_grid,
                                   CGridTensorView& c_grid,
                                   const CElementFunction& c_element_func) const
    {
        // Extract problem dimensions from the tensor descriptors at runtime.
        // These are runtime values (M, N, K are passed as kernel arguments).
        const auto M = a_grid.get_tensor_descriptor().get_length(number<0>{});
        const auto N = c_grid.get_tensor_descriptor().get_length(number<1>{});
        const auto K = a_grid.get_tensor_descriptor().get_length(number<1>{});

        // get_block_id(): returns the linear thread block index within the grid (blockIdx.x).
        // The grid is 1D here (kGridSize blocks total).
        const auto id_block = get_block_id();

        // Number of tiles needed to cover the M and N dimensions of C.
        // integer_divide_ceil(x, y) = (x + y - 1) / y -- handles non-divisible sizes.
        const auto num_tile_m = integer_divide_ceil(M, kMPerBlock);
        const auto num_tile_n = integer_divide_ceil(N, kNPerBlock);

        // MakeBlock2TileMap returns a lambda that converts a linear block_id to a 2D tile index.
        // N-first ordering: adjacent block IDs map to adjacent N tiles (same M row),
        // so they access the same A rows but different B columns.
        const auto block2tile = Policy::template MakeBlock2TileMap<Problem>(num_tile_m, num_tile_n);

        const auto id_tile = block2tile(id_block);

        // __builtin_amdgcn_readfirstlane: broadcasts a VGPR value from lane 0 to all lanes.
        // The compiler then treats iM/iN as SGPRs (scalar registers) rather than VGPRs.
        // This is important because tile window origins are uniform across all threads in a
        // block -- making them scalar reduces VGPR pressure significantly.
        const auto iM = __builtin_amdgcn_readfirstlane(id_tile.template at<0>() * kMPerBlock);
        const auto iN = __builtin_amdgcn_readfirstlane(id_tile.template at<1>() * kNPerBlock);

        // A block window: covers rows [iM, iM+kMPerBlock) and initially K columns [0, kKPerBlock).
        // The block pipeline will slide this window along K in each loop iteration.
        auto a_block_window = make_tile_window(
            a_grid, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {iM, 0});

        // B block window: covers rows [iN, iN+kNPerBlock) and initially K columns [0, kKPerBlock).
        // B is stored as [N, K] (N as leading dimension, K is contiguous).
        auto b_block_window = make_tile_window(
            b_grid, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {iN, 0});

        // GetBlockGemmPipeline returns the block pipeline object (compile-time, no runtime cost).
        constexpr auto block_gemm_pipeline = Policy::template GetBlockGemmPipeline<Problem>();

        // Allocate LDS (shared memory) for the block pipeline.
        // GetStaticLdsSize() is a compile-time constant, so the array size is known at compile
        // time. The pipeline uses LDS to stage A and B tiles for MFMA consumption.
        __shared__ char p_smem_char[block_gemm_pipeline.GetStaticLdsSize()];

        // Run the block pipeline: iterates K/kKPerBlock times, accumulates into acc_block_tile.
        // acc_block_tile is in AccDataType (float) precision in registers.
        const auto acc_block_tile =
            block_gemm_pipeline(a_block_window, b_block_window, K / kKPerBlock, p_smem_char);

        // Cast accumulator tile from AccDataType to CDataType, applying CElementFunction.
        // tile_elementwise_in applies the lambda element-wise, producing a new tile.
        // type_convert<CDataType>(acc): FP32 -> FP16 conversion (or identity if types match).
        // c_element_func(x): apply any epilogue (e.g., bias add, activation); identity here.
        const auto c_block_tile = tile_elementwise_in(
            [&](const auto& acc) { return c_element_func(type_convert<CDataType>(acc)); },
            acc_block_tile);

        // Create the C tile window at this block's output position.
        // store_tile uses the C tile's distribution to scatter per-thread register data
        // back to the correct global memory addresses (vectorized, coalesced stores).
        auto c_window = make_tile_window(
            c_grid, make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}), {iM, iN});

        store_tile(c_window, c_block_tile);
    }
};

} // namespace ck_tile
