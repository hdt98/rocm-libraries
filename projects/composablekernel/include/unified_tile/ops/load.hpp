// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../config.hpp"

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
#include "ck_tile/core/tensor/load_tile.hpp"
#endif

#ifdef UNIFIED_TILE_BACKEND_MINT
#include "mint/tile/generic/load_no_shuffle_vectorized.h"
#endif

namespace unified_tile {
namespace ops {

// ============================================================================
// CK_Tile Backend
// ============================================================================

#ifdef UNIFIED_TILE_BACKEND_CK_TILE

/// @brief Load data through a tile window into a new distributed tensor.
/// Returns a static_distributed_tensor with per-thread data.
template <typename TileWindow>
CK_TILE_DEVICE auto load_tile(const TileWindow& window)
{
    return ck_tile::load_tile(window);
}

/// @brief Load data through a tile window into an existing distributed tensor.
template <typename DistributedTensor, typename TileWindow>
CK_TILE_DEVICE void load_tile(DistributedTensor& dst, const TileWindow& window)
{
    ck_tile::load_tile(dst, window);
}

#else // UNIFIED_TILE_BACKEND_MINT

// ============================================================================
// MINT Backend: auto-derive vector/freeze params from distribution
// ============================================================================
// For a 2D block copy distribution with element dims {Dim0_elem, Dim1_elem}:
//   - Vector dim = Dim1_elem (inner/contiguous element dim)
//   - Vector length = distribution's element length for that dim
//   - Freeze dims = tuple({Dim1_alias}, {Dim0_alias})
//     (when iterating Dim0_elem, freeze Dim1; when iterating Dim1_elem, freeze Dim0)
// ============================================================================

namespace detail {

/// @brief Auto-derive load parameters and call masked_load_no_shuffle_vectorized.
/// Works for 2D block copy distributions where element_ndim() == 2.
template <typename TileWindow>
MINT_DEVICE auto load_tile_impl(const TileWindow& window)
{
    using namespace mint;

    // Extract distribution info from the window type
    constexpr auto dstr = TileWindow::dstr_tensor_desc();
    static_assert(dstr.element_ndim() == 2,
                  "Auto-derive only supports 2D distributions (2 element dims)");

    constexpr auto elem_aliases = dstr.element_dim_aliases();
    constexpr auto top_aliases = dstr.top_dim_aliases();
    constexpr auto elem_lengths = dstr.element_lengths();

    // Vector dim = last element dim (inner/contiguous), e.g. "K_1"
    constexpr auto vector_dims = array<alias_t, 1>{elem_aliases[1]};
    constexpr auto vector_lengths = array<index_t, 1>{elem_lengths[1]};

    // Freeze dims: when moving elem[0], freeze top[1]; when moving elem[1], freeze top[0]
    constexpr auto freeze_dims = mint::make_tuple(
        array<alias_t, 1>{top_aliases[1]},
        array<alias_t, 1>{top_aliases[0]});

    return tile::generic::experimental::masked_load_no_shuffle_vectorized<
        vector_dims,
        vector_lengths,
        freeze_dims>(window, tuple<>{});
}

} // namespace detail

/// @brief Load data through a distributed window into a new distributed tensor.
/// Auto-derives vector dims, lengths, and freeze dims from the window's distribution.
template <typename TileWindow>
MINT_DEVICE auto load_tile(const TileWindow& window)
{
    return detail::load_tile_impl(window);
}

/// @brief Load data through a distributed window into an existing distributed tensor.
template <typename DistributedTensor, typename TileWindow>
MINT_DEVICE void load_tile(DistributedTensor& dst, const TileWindow& window)
{
    dst = detail::load_tile_impl(window);
}

#endif // UNIFIED_TILE_BACKEND

} // namespace ops
} // namespace unified_tile
