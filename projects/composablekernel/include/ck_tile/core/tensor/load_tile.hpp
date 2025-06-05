// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/tensor/tile_window_linear.hpp"
#include "ck_tile/core/tensor/null_tile_window.hpp"
#include "ck_tile/core/tensor/null_tensor.hpp"

namespace ck_tile {

template <typename TileWindow_, index_t i_access = -1, bool oob_conditional_check = true>
CK_TILE_DEVICE auto load_tile(const TileWindow_& tile_window,
                              number<i_access>                     = {},
                              bool_constant<oob_conditional_check> = {})
{
    return tile_window.load(number<i_access>{}, bool_constant<oob_conditional_check>{});
}

template <typename DistributedTensor_,
          typename TileWindow_,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto load_tile(DistributedTensor_& dst_tile,
                              const TileWindow_& tile_window,
                              number<i_access>                     = {},
                              bool_constant<oob_conditional_check> = {})
{
    return tile_window.load(dst_tile, number<i_access>{}, bool_constant<oob_conditional_check>{});
}

/**
 * @brief Loads a tile of data using inline assembly.
 *
 * @note Bare in mind that loading data this way, you have to manually initialize your
 *       thread buffer and synchronize load afterwards in order to make sure it's done before
 *       using loaded data from registers
 *       @see `tile_window_with_static_distribution::init_raw()` and `buffer_view.hpp`
 *       @see  `buffer_load_fence()`
 */
template <typename T,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          index_t i_access           = -1,
          bool oob_conditional_check = true,
          bool pre_nop               = false>
CK_TILE_DEVICE auto load_tile_raw(T& tile,
                                  const tile_window_with_static_distribution<BottomTensorView_,
                                                                             WindowLengths_,
                                                                             TileDistribution_,
                                                                             NumCoord>& tile_window,
                                  number<i_access>                     = {},
                                  bool_constant<oob_conditional_check> = {},
                                  bool_constant<pre_nop>               = {})
{
    tile_window.load_raw(
        tile, number<i_access>{}, bool_constant<oob_conditional_check>{}, bool_constant<pre_nop>{});
}

template <typename T,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename LinearBottomDims_,
          index_t i_access           = -1,
          bool oob_conditional_check = true,
          bool pre_nop               = false>
CK_TILE_DEVICE auto load_tile_raw(T& tile,
                                  const tile_window_linear<BottomTensorView_,
                                                           WindowLengths_,
                                                           TileDistribution_,
                                                           LinearBottomDims_>& tile_window,
                                  number<i_access>                     = {},
                                  bool_constant<oob_conditional_check> = {},
                                  bool_constant<pre_nop>               = {})
{
    tile_window.load_raw(
        tile, number<i_access>{}, bool_constant<oob_conditional_check>{}, bool_constant<pre_nop>{});
}

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto tr_load_tile(const tile_window_with_static_distribution<BottomTensorView_,
                                                                            WindowLengths_,
                                                                            TileDistribution_,
                                                                            NumCoord>& tile_window,
                                 number<i_access>                     = {},
                                 bool_constant<oob_conditional_check> = {})
{
    return tile_window.tr_load(number<i_access>{}, bool_constant<oob_conditional_check>{});
}

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename LinearBottomDims_,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto tr_load_tile(const tile_window_linear<BottomTensorView_,
                                                          WindowLengths_,
                                                          TileDistribution_,
                                                          LinearBottomDims_>& tile_window,
                                 number<i_access>                     = {},
                                 bool_constant<oob_conditional_check> = {})
{
    return tile_window.tr_load(number<i_access>{}, bool_constant<oob_conditional_check>{});
}

// TODO : be careful that async_load_tile_to_lds's logic is different from async_load_tile_raw; only
// support builtin async load in gfx13; LdsTileWindow_ only supports tile_window_with_static_lengths
template <typename LdsBottomTensorView_,
          typename GlobalBottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto async_load_tile_to_lds(
    tile_window_with_static_lengths<LdsBottomTensorView_, WindowLengths_>& lds_tile_window,
    const tile_window_with_static_distribution<GlobalBottomTensorView_,
                                               WindowLengths_,
                                               TileDistribution_,
                                               NumCoord>& global_tile_window,
    number<i_access>                     = {},
    bool_constant<oob_conditional_check> = {})
{
    using lds_data_type    = typename remove_cvref_t<decltype(lds_tile_window)>::DataType;
    using global_data_type = typename remove_cvref_t<decltype(global_tile_window)>::DataType;
    static_assert(std::is_same_v<lds_data_type, global_data_type>,
                  "currently lds and global's data type should be the same!");

    return global_tile_window.async_load_to_lds(lds_tile_window);
}

template <typename LdsTileWindow_,
          typename TileWindow_,
          index_t i_access           = -1,
          bool oob_conditional_check = true,
          bool pre_nop               = false>
CK_TILE_DEVICE auto async_load_tile_raw(LdsTileWindow_&& lds_tile,
                                        const TileWindow_& tile_window,
                                        number<i_access>                     = {},
                                        bool_constant<oob_conditional_check> = {},
                                        bool_constant<pre_nop>               = {})
{
    return tile_window.async_load_raw(lds_tile,
                                      number<i_access>{},
                                      bool_constant<oob_conditional_check>{},
                                      bool_constant<pre_nop>{});
}

CK_TILE_DEVICE auto async_load_fence(index_t cnt = 0)
{
#if defined(__gfx13__)
    asm volatile("s_wait_loadcnt %0" : : "n"(cnt) : "memory");
#else
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
#endif
}

template <typename WindowLengths>
CK_TILE_DEVICE auto load_tile(const null_tile_window<WindowLengths>&)
{
    return null_tensor{};
}

template <typename T, typename WindowLengths>
CK_TILE_DEVICE auto load_tile_raw(T& /*null_tile*/, const null_tile_window<WindowLengths>&)
{
}

} // namespace ck_tile
