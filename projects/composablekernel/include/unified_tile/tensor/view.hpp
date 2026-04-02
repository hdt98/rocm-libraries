// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../config.hpp"
#include "address_space.hpp"

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
#include "ck_tile/core/tensor/buffer_view.hpp"
#include "ck_tile/core/tensor/tensor_view.hpp"
#endif

#ifdef UNIFIED_TILE_BACKEND_MINT
#include "mint/tensor/tensor_view.h"
#include "mint/core/memory/simt/global_memory_view.h"
#include "mint/core/memory/simt/shared_memory_view.h"
#endif

namespace unified_tile {
namespace view {

// ============================================================================
// CK_Tile Backend
// ============================================================================

#ifdef UNIFIED_TILE_BACKEND_CK_TILE

/// @brief Create a tensor view from pointer + descriptor (CK_Tile)
/// @tparam AddrSpace The address space (global, shared, vgpr)
/// @param p Pointer to memory
/// @param desc Tensor descriptor
template <address_space AddrSpace, typename DataType, typename... DescTs>
CK_TILE_HOST_DEVICE constexpr auto
make_tensor_view(DataType* p, const ck_tile::tensor_descriptor<DescTs...>& desc)
{
    if constexpr(AddrSpace == address_space::global)
    {
        return ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
            p, desc);
    }
    else if constexpr(AddrSpace == address_space::shared)
    {
        return ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            p, desc);
    }
    else if constexpr(AddrSpace == address_space::vgpr)
    {
        return ck_tile::make_tensor_view<ck_tile::address_space_enum::vgpr>(
            p, desc);
    }
}

/// @brief Create a tensor view from pointer + lengths + strides (CK_Tile)
template <address_space AddrSpace,
          typename DataType,
          typename... Lengths,
          typename... Strides>
CK_TILE_HOST_DEVICE constexpr auto make_tensor_view(
    DataType* p,
    const ck_tile::tuple<Lengths...>& lengths,
    const ck_tile::tuple<Strides...>& strides)
{
    if constexpr(AddrSpace == address_space::global)
    {
        return ck_tile::make_naive_tensor_view<
            ck_tile::address_space_enum::global>(p, lengths, strides);
    }
    else if constexpr(AddrSpace == address_space::shared)
    {
        return ck_tile::make_naive_tensor_view<
            ck_tile::address_space_enum::lds>(p, lengths, strides);
    }
    else if constexpr(AddrSpace == address_space::vgpr)
    {
        return ck_tile::make_naive_tensor_view<
            ck_tile::address_space_enum::vgpr>(p, lengths, strides);
    }
}

/// @brief Apply padding to a tensor view (CK_Tile)
/// Pads dimensions where DoPads is true to be multiples of tile_lengths.
/// @param tensor_view The tensor view to pad
/// @param tile_lengths Tile size per dimension
/// @param do_pads sequence<bool,...> indicating which dims to pad
template <typename TensorView, typename TileLengths, typename DoPads>
CK_TILE_HOST_DEVICE constexpr auto pad_view(const TensorView& tensor_view,
                                             const TileLengths& tile_lengths,
                                             DoPads)
{
    return ck_tile::pad_tensor_view(tensor_view, tile_lengths, DoPads{});
}

#else // UNIFIED_TILE_BACKEND_MINT

/// @brief Create a tensor view from pointer + descriptor (MINT, global)
template <address_space AddrSpace, typename DataType, typename Descriptor>
MINT_HOST_DEVICE constexpr auto make_tensor_view(DataType* p,
                                                  const Descriptor& desc)
{
    if constexpr(AddrSpace == address_space::global)
    {
        auto mem_view =
            mint::make_global_memory_view(p, desc.bottom_lengths()[0]);
        return mint::tensor::make_tensor_view(desc, mem_view);
    }
    else if constexpr(AddrSpace == address_space::shared)
    {
        auto mem_view =
            mint::make_shared_memory_view(p, desc.bottom_lengths()[0]);
        return mint::tensor::make_tensor_view(desc, mem_view);
    }
}

/// @brief Apply padding to a tensor view (MINT - no-op, masking at load/store)
/// Returns the view unchanged. MINT handles boundaries via masks.
template <typename TensorView, typename TileLengths, typename DoPads>
MINT_HOST_DEVICE constexpr auto pad_view(const TensorView& tensor_view,
                                          const TileLengths&,
                                          DoPads)
{
    return tensor_view;
}

#endif // UNIFIED_TILE_BACKEND

// ============================================================================
// Query Functions (common to both backends)
// ============================================================================

/// @brief Get the total element space size of a tensor view
/// CK_TILE: view.get_tensor_descriptor().get_element_space_size()
/// MINT:    view.memory_view().size()
template <typename TensorView>
UNIFIED_TILE_HOST_DEVICE constexpr auto get_view_size(const TensorView& view)
{
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    return view.get_tensor_descriptor().get_element_space_size();
#else
    return view.memory_view().size();
#endif
}

} // namespace view
} // namespace unified_tile
