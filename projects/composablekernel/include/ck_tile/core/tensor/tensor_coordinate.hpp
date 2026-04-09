// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tensor_adaptor_coordinate.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/tensor/im2col_coordinate.hpp"

namespace ck_tile {

template <index_t NDimHidden, typename TopDimensionHiddenIds>
struct tensor_coordinate
    : public tensor_adaptor_coordinate<NDimHidden, sequence<0>, TopDimensionHiddenIds>
{
    using Base = tensor_adaptor_coordinate<NDimHidden, sequence<0>, TopDimensionHiddenIds>;

    // TODO make these private
    static constexpr index_t ndim_top_ = TopDimensionHiddenIds::size();

    using HiddenIndex = multi_index<NDimHidden>;
    using TopIndex    = multi_index<ndim_top_>;

    public:
    CK_TILE_HOST_DEVICE constexpr tensor_coordinate() = default;

    CK_TILE_HOST_DEVICE constexpr tensor_coordinate(const HiddenIndex& idx_hidden)
        : Base{idx_hidden}
    {
    }

    // construct from TensorAdaptorCoordinte base class
    CK_TILE_HOST_DEVICE constexpr tensor_coordinate(const Base& adaptor_coord) : Base{adaptor_coord}
    {
    }

    CK_TILE_HOST_DEVICE constexpr auto get_index() const { return Base::get_top_index(); }

    CK_TILE_HOST_DEVICE constexpr index_t get_offset() const
    {
        return Base::get_bottom_index()[number<0>{}];
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_hidden_index() const
    {
        return Base::get_hidden_index();
    }

    CK_TILE_HOST_DEVICE auto& get_hidden_index() { return Base::get_hidden_index(); }
};

// ---------------------------------------------------------------------------
// is_im2col_coordinate<T>
//
// True for any Im2ColCoordinate<Tensor> specialization.
// Used in the fast paths of make/move_tensor_coordinate and validity checks.
// ---------------------------------------------------------------------------
template <typename T>
struct is_im2col_coordinate : std::false_type
{
};

template <Im2ColTensor Tensor>
struct is_im2col_coordinate<Im2ColCoordinate<Tensor>> : std::true_type
{
};

template <typename T>
inline constexpr bool is_im2col_coordinate_v = is_im2col_coordinate<remove_cvref_t<T>>::value;

// ---------------------------------------------------------------------------
// make_tensor_coordinate
//
// Fast path: when the descriptor has im2col metadata (has_im2col_meta_v<>),
// return an Im2ColCoordinate<Tensor> initialized from (m_gemm, k/n_gemm).
// The tensor kind is read from the descriptor's im2col_tensor_kind static
// member (defaults to FwdInput for backward compatibility).
//
// Generic path: delegate to the full transform-chain adaptor.
// ---------------------------------------------------------------------------
template <typename TensorDesc, typename TopIndex>
CK_TILE_HOST_DEVICE constexpr auto make_tensor_coordinate(const TensorDesc& tensor_desc,
                                                          const TopIndex& idx_top)
{
    if constexpr(has_im2col_meta_v<TensorDesc>)
    {
        // Fast tiled path: use precomputed metadata for direct offset calculation.
        // idx_top contains the absolute (m_gemm, k_gemm / n_gemm) indices.
        constexpr Im2ColTensor kind = im2col_tensor_kind_of_v<TensorDesc>;
        Im2ColCoordinate<kind> coord;
        coord.init(static_cast<index_t>(idx_top[number<0>{}]),
                   static_cast<index_t>(idx_top[number<1>{}]),
                   tensor_desc.get_im2col_meta());
        return coord;
    }
    else
    {
        const auto adaptor_coord = make_tensor_adaptor_coordinate(tensor_desc, idx_top);
        return tensor_coordinate<
            TensorDesc::get_num_of_hidden_dimension(),
            remove_cvref_t<decltype(TensorDesc::get_top_dimension_hidden_ids())>>{adaptor_coord};
    }
}

// ---------------------------------------------------------------------------
// move_tensor_coordinate
//
// Fast path: when the coordinate is any Im2ColCoordinate<Tensor>, use its
// move_step() method which separates M and K/N movements.
//   - K/N-only step (dm==0): incremental update, 0 divmod for FwdOutput.
//   - General step:          full reinit (3-5 divmod depending on tensor kind).
//
// Generic path: delegate to move_tensor_adaptor_coordinate.
// ---------------------------------------------------------------------------
template <bool JudgeDoTransforms = true, typename TensorDesc, typename TensorCoord, typename Index>
CK_TILE_HOST_DEVICE constexpr void
move_tensor_coordinate(const TensorDesc& tensor_desc, TensorCoord& coord, const Index& coord_step)
{
    if constexpr(is_im2col_coordinate_v<TensorCoord>)
    {
        // Fast tiled path: coord_step is the delta in (M, K/N) global space.
        coord.move_step(static_cast<index_t>(coord_step[number<0>{}]),
                        static_cast<index_t>(coord_step[number<1>{}]),
                        tensor_desc.get_im2col_meta());
    }
    else
    {
        move_tensor_adaptor_coordinate(tensor_desc, coord, coord_step);
    }
}

// ---------------------------------------------------------------------------
// coordinate_has_valid_offset_assuming_top_index_is_valid
//
// Fast path: for any Im2ColCoordinate<Tensor>, validity is precomputed in
// coord.valid during init() and move_step().
//
// Generic path: delegate to adaptor validity check.
// ---------------------------------------------------------------------------
template <typename TensorDesc, typename TensorCoord>
CK_TILE_HOST_DEVICE constexpr bool
coordinate_has_valid_offset_assuming_top_index_is_valid(const TensorDesc& tensor_desc,
                                                        const TensorCoord& coord)
{
    if constexpr(is_im2col_coordinate_v<TensorCoord>)
    {
        return coord.is_valid();
    }
    else
    {
        return adaptor_coordinate_is_valid_assuming_top_index_is_valid(tensor_desc, coord);
    }
}

template <typename TensorDesc, typename TensorCoord>
CK_TILE_HOST_DEVICE constexpr bool coordinate_has_valid_offset(const TensorDesc& tensor_desc,
                                                               const TensorCoord& coord)
{
    if constexpr(is_im2col_coordinate_v<TensorCoord>)
    {
        return coord.is_valid();
    }
    else
    {
        return adaptor_coordinate_is_valid(tensor_desc, coord);
    }
}

template <index_t N, typename T>
CK_TILE_HOST_DEVICE void print(const tensor_coordinate<N, T>& coord)
{
    print(static_cast<typename tensor_coordinate<N, T>::Base>(coord));
}

} // namespace ck_tile
