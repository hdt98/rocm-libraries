// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <mint/mint.h>

namespace mint_tutorial {

using namespace mint;
using namespace mint::tensor;
using namespace mint::tile::rocm;  // Use rocm namespace for ROCm platform

/**
 * @brief MINT copy kernel using warp-level tile operations
 *
 * This demonstrates MINT's warp-level API using runtime tensor dimensions.
 *
 * @tparam RowPerTile Rows per warp tile
 * @tparam ColPerTile Cols per warp tile
 * @tparam T Data type
 */
template <index_t RowPerTile, index_t ColPerTile, typename T>
__global__ void MintCopyKernelTiled(T* __restrict__ p_dst,
                                     const T* __restrict__ p_src,
                                     index_t M,
                                     index_t N)
{
    // Create packed tensor views with runtime dimensions
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

    // Masks (empty for no boundary checking)
    const auto mask = tuple<>{};

    // Main copy loop
    // Each block processes one row of tiles
    const index_t block_row = blockIdx.x * RowPerTile;

    // Iterate over columns
    for (index_t col = 0; col < N; col += ColPerTile)
    {
        // Warp-level load: cooperatively load tile into VGPRs
        warp::masked_load(src_view, {block_row, col}, dstr_tensor, mask);

        // Warp-level store: cooperatively store tile from VGPRs to global memory
        warp::masked_store(dst_view, {block_row, col}, dstr_tensor, mask);
    }
}

} // namespace mint_tutorial
