// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../config.hpp"

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
#include "ck_tile/core/tensor/store_tile.hpp"
#endif

#ifdef UNIFIED_TILE_BACKEND_MINT
#include "mint/tile/generic/store_no_shuffle_vectorized.h"
#endif

namespace unified_tile {
namespace ops {

// ============================================================================
// CK_Tile Backend
// ============================================================================

#ifdef UNIFIED_TILE_BACKEND_CK_TILE

/// @brief Store a distributed tensor through a tile window.
/// Writes per-thread data from the distributed tensor to memory via the window.
template <typename TileWindow, typename DistributedTensor>
CK_TILE_DEVICE void store_tile(TileWindow& window,
                                const DistributedTensor& tile)
{
    ck_tile::store_tile(window, tile);
}

#else // UNIFIED_TILE_BACKEND_MINT

// ============================================================================
// MINT Backend: auto-derive vector/freeze params from distribution
// ============================================================================
// Same pattern as load.hpp:
//   - Vector dim = last element dim (inner/contiguous)
//   - Freeze dims = tuple({top[1]}, {top[0]}) for 2-element-dim distributions
// ============================================================================

namespace detail {

/// @brief Auto-derive store parameters and call masked_store_no_shuffle_vectorized.
/// Works for 2D block copy distributions where element_ndim() == 2.
template <typename TileWindow, typename DistributedTensor>
MINT_DEVICE void store_tile_impl(const TileWindow& window,
                                  const DistributedTensor& tile)
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

    tile::generic::experimental::masked_store_no_shuffle_vectorized<
        vector_dims,
        vector_lengths,
        freeze_dims>(window, tuple<>{}, tile);
}

} // namespace detail

/// @brief Store a distributed tensor through a distributed window.
/// Auto-derives vector dims, lengths, and freeze dims from the window's distribution.
template <typename TileWindow, typename DistributedTensor>
MINT_DEVICE void store_tile(const TileWindow& window,
                             const DistributedTensor& tile)
{
    detail::store_tile_impl(window, tile);
}

#endif // UNIFIED_TILE_BACKEND

} // namespace ops
} // namespace unified_tile
