// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../config.hpp"
#include "partition.hpp"

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/tensor/tile_window_utils.hpp"
#endif

#ifdef UNIFIED_TILE_BACKEND_MINT
#include "mint/tensor/distributed_window.h"
#include "mint/tensor/tensor_descriptor_helper.h"
#endif

namespace unified_tile {
namespace window {

// ============================================================================
// CK_Tile Backend
// ============================================================================

#ifdef UNIFIED_TILE_BACKEND_CK_TILE

/// @brief Create a tile window with distribution (CK_Tile).
/// Combines a tensor view, window lengths, origin, and distribution into
/// a tile_window_with_static_distribution that maps threads to data.
///
/// @param view       Tensor view (global or shared memory)
/// @param lengths    Tile size per dimension (compile-time)
/// @param origin     Starting offset in the view
/// @param dstr       Tile distribution (from make_block_copy_*_distribution)
template <typename TensorView,
          typename WindowLengths,
          typename Distribution>
CK_TILE_DEVICE constexpr auto
make_tile_window(const TensorView& view,
                 const WindowLengths& lengths,
                 const ck_tile::multi_index<TensorView::get_num_of_dimension()>& origin,
                 const Distribution& dstr)
{
    return ck_tile::make_tile_window(view, lengths, origin, dstr);
}

/// @brief Create a tile window without distribution (CK_Tile).
/// Returns a tile_window_with_static_lengths (logical view only).
template <typename TensorView, typename WindowLengths>
CK_TILE_DEVICE constexpr auto
make_tile_window(const TensorView& view,
                 const WindowLengths& lengths,
                 const ck_tile::multi_index<TensorView::get_num_of_dimension()>& origin)
{
    return ck_tile::make_tile_window(view, lengths, origin);
}

/// @brief Move a tile window by a step along each dimension.
template <typename TileWindow>
CK_TILE_DEVICE void move_window(TileWindow& window,
                                 const typename TileWindow::BottomTensorIndex& step)
{
    window.move(step);
}

#else // UNIFIED_TILE_BACKEND_MINT

/// @brief Create a distributed window with distribution (MINT).
/// The element_layout is auto-generated from distribution's element_lengths().
/// The partition is auto-generated from BlockSize (extracted from distribution).
///
/// @param view       Tensor view (global or shared memory)
/// @param lengths    Tile size per dimension (unused by MINT, kept for API parity)
/// @param origin     Starting offset in the view
/// @param dstr       Distribution (distributed_tensor_descriptor)
template <typename TensorView,
          typename WindowLengths,
          typename Distribution>
MINT_DEVICE constexpr auto
make_tile_window(const TensorView& view,
                 const WindowLengths&,
                 const mint::nd_index<TensorView::ndim()>& origin,
                 const Distribution&)
{
    // Distribution is a distributed_tensor_descriptor with all info in NTTPs.
    // Default-construct to get a constexpr value for mint::constant<>.
    constexpr auto dstr_val = Distribution{};
    constexpr auto block_size = Distribution::partition_size();
    using Partition = block_partition<static_cast<int>(block_size)>;

    return mint::tensor::make_distributed_window(
        view,
        origin,
        mint::constant<dstr_val>{},
        mint::constant<Partition{}>{});
}

/// @brief Move a distributed window by a delta along each dimension.
template <typename TileWindow>
MINT_DEVICE void move_window(TileWindow& window,
                              const mint::nd_index<TileWindow::dstr_tensor_desc().top_ndim()>& step)
{
    mint::tensor::move_window(window, step);
}

#endif // UNIFIED_TILE_BACKEND

} // namespace window
} // namespace unified_tile
