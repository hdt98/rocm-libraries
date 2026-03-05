// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <mint/mint.h>

namespace mint_tutorial {

using namespace mint;
using namespace mint::tensor;
using namespace mint::tile::rocm;  // Use rocm namespace for ROCm platform

/**
 * @brief MINT-based copy kernel using warp-level operations
 *
 * This kernel demonstrates the MINT programming model:
 * 1. Create tensor views using packed tensor view API
 * 2. Define distributed tensors for VGPR storage using make_simple_distribution
 * 3. Use warp-level masked load/store operations
 *
 * @tparam RowPerTile Number of rows per warp tile (typically 16 or 32)
 * @tparam ColPerTile Number of cols per warp tile (typically 16 or 32)
 * @tparam T Data type (fp16/half only for MINT)
 */
template <index_t RowPerTile, index_t ColPerTile, typename T>
__global__ void MintCopyKernel(T* p_dst, const T* p_src, index_t M, index_t N)
{
    // Step 1: Create tensor views using MINT's packed tensor view API
    const auto src_view = make_global_packed_tensor_view(
        p_src, nd_index<2>{M, N});

    const auto dst_view = make_global_packed_tensor_view(
        p_dst, nd_index<2>{M, N});

    // Step 2: Define distributed tensor for VGPR storage
    // Use make_simple_distribution to create a warp-distributed tile
    constexpr auto tile_lengths = index_sequence<RowPerTile, ColPerTile>{};
    constexpr auto partition_info = thread_in_this_warp{};
    constexpr auto tile_aliases = sequence<alias_t, alias_t{"r"}, alias_t{"c"}>{};
    constexpr auto part_aliases = sequence<alias_t, alias_t{"w"}>{};

    constexpr auto dstr_desc = make_simple_distribution(
        tile_lengths,
        partition_info,
        tile_aliases,
        part_aliases);

    auto dstr_tensor = make_distributed_tensor_vgpr<T>(constant<dstr_desc>{});

    // Step 3: Define masks (empty - no boundary handling in this simple version)
    const auto src_mask = tuple<>{};
    const auto dst_mask = tuple<>{};

    // Step 4: Main copy loop
    // Iterate over rows, then columns
    // Each warp processes RowPerTile x ColPerTile elements per iteration

    index_t row = 0;
    do {
        index_t col = 0;
        do {
            // Warp-level masked load: reads from global memory into distributed tensor
            warp::masked_load(src_view, {row, col}, dstr_tensor, src_mask);

            // Warp-level masked store: writes from distributed tensor to global memory
            warp::masked_store(dst_view, {row, col}, dstr_tensor, dst_mask);

            col += ColPerTile;
        } while (col < N);
        row += RowPerTile;
    } while (row < M);
}

/**
 * @brief MINT-based copy kernel with element-wise access pattern
 *
 * This variant shows how to access individual elements within the distributed tensor,
 * which can be useful for element-wise transformations or data conversion.
 *
 * Note: This is currently a pass-through copy, but demonstrates the pattern
 * for future element-wise transformations.
 */
template <index_t RowPerTile, index_t ColPerTile, typename T>
__global__ void MintCopyKernelElementWise(T* p_dst, const T* p_src, index_t M, index_t N)
{
    // Create tensor views
    const auto src_view = make_global_packed_tensor_view(
        p_src, nd_index<2>{M, N});
    const auto dst_view = make_global_packed_tensor_view(
        p_dst, nd_index<2>{M, N});

    // Define distributed tensor using make_simple_distribution
    constexpr auto tile_lengths = index_sequence<RowPerTile, ColPerTile>{};
    constexpr auto partition_info = thread_in_this_warp{};
    constexpr auto tile_aliases = sequence<alias_t, alias_t{"r"}, alias_t{"c"}>{};
    constexpr auto part_aliases = sequence<alias_t, alias_t{"w"}>{};

    constexpr auto dstr_desc = make_simple_distribution(
        tile_lengths,
        partition_info,
        tile_aliases,
        part_aliases);

    auto dstr_tensor = make_distributed_tensor_vgpr<T>(constant<dstr_desc>{});

    const auto src_mask = tuple<>{};
    const auto dst_mask = tuple<>{};

    // Main loop - same as basic version but could add element-wise ops
    index_t row = 0;
    do {
        index_t col = 0;
        do {
            warp::masked_load(src_view, {row, col}, dstr_tensor, src_mask);

            // Here you could perform element-wise operations on dstr_tensor
            // Example (pseudocode):
            // for each element e in dstr_tensor:
            //     e = transform(e)

            warp::masked_store(dst_view, {row, col}, dstr_tensor, dst_mask);

            col += ColPerTile;
        } while (col < N);
        row += RowPerTile;
    } while (row < M);
}

} // namespace mint_tutorial
