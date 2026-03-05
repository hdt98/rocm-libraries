#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>

namespace mint {
namespace tile {
namespace generic {
namespace impl {

template <
    class TensorView,
    class TensorMask,
    auto kTilerDims,
    auto kTileDesc,
    auto kElementTensorDesc,
    auto kPartition,
    class Memory>
MINT_HOST_DEVICE void masked_store_no_shuffle_impl(
    const typename mint::tensor::impl::tiler_impl<
        TensorView,
        kTilerDims,
        kTileDesc,
        kElementTensorDesc,
        kPartition>& tiler,
    const TensorMask& dst_tensor_mask,
    const mint::tensor::
        distributed_tensor<kTileDesc, kElementTensorDesc, Memory>& src) {
  using value_type = typename TensorView::value_type;

  const TensorView& dst_tensor_view = tiler.tensor_view();

  auto dst_coord = tiler.coord_;

  //
  constexpr auto get_old_idx = []<index_t ndim>(
                                   const nd_index<ndim>& lengths,
                                   const nd_index<ndim>& idx) {
    if constexpr (ndim == 0) {
      return nd_index<0>{};
    } else {
      auto ret = idx;
      ret[ndim - 1]--;
      bool borrow = ret[ndim - 1] < 0;
      for (index_t i = ndim - 1; i > 0; i--) {
        if (borrow) {
          ret[i] = lengths[i] - 1;
          ret[i - 1]--;
          borrow = ret[i - 1] < 0;
        }
      }
      return ret;
    }
  };

  //
  constexpr auto get_snake_idx = []<index_t ndim>(
                                     const nd_index<ndim>& lengths,
                                     const nd_index<ndim>& idx) {
    if constexpr (ndim == 0) {
      return nd_index<0>{};
    } else {
      nd_index<ndim> ret{};
      ret[0] = idx[0];
      index_t cnt = idx[0];
      for (index_t i = 1; i < ndim; i++) {
        ret[i] = (cnt % 2 == 0) ? idx[i] : lengths[i] - 1 - idx[i];
        cnt = cnt * lengths[i] + idx[i];
      }
      return ret;
    }
  };

  constexpr auto tile_dims = std::remove_cvref_t<decltype(tiler)>::tile_dims();

  constexpr auto get_full_idx = [tile_dims](auto tile_idx) {
    nd_index<TensorView::ndim()> ret;
    // tiler dim first
    for (index_t i = 0; i < kTilerDims.size(); i++) {
      ret[kTilerDims[i]] = 0;
    }
    // tile dim second
    for (index_t i = 0; i < tile_dims.size(); i++) {
      ret[tile_dims[i]] = tile_idx[i];
    }
    return ret;
  };

  constexpr index_t elem_top_ndim = kElementTensorDesc.top_ndim();
  constexpr auto elem_top_lengths = kTileDesc.element_lengths();

  // loop
  static_for_nd2<elem_top_lengths>()([&](auto... is) {
    constexpr auto idx = nd_index<elem_top_ndim>{is...};
    constexpr auto old_idx = get_old_idx(elem_top_lengths, idx);
    constexpr auto snake_idx = get_snake_idx(elem_top_lengths, idx);
    constexpr auto snake_old_idx = get_snake_idx(elem_top_lengths, old_idx);

    // move forward
    if constexpr (idx != nd_index<elem_top_ndim>{}.fill(0)) {
      constexpr auto tile_top_idx_delta =
          kTileDesc.top_index_delta(snake_idx - snake_old_idx);
      move_coordinate(
          dst_coord,
          dst_tensor_view.tensor_desc(),
          get_full_idx(tile_top_idx_delta));
    }

    constexpr index_t elem_offset =
        kElementTensorDesc.calculate_bottom_index(snake_idx)[0];
    value_type v = src.memory().template at<elem_offset>();

    if (is_unmasked(dst_coord, dst_tensor_view.tensor_desc(), dst_tensor_mask))
      dst_tensor_view.element(dst_coord) = v;
  });
}

} // namespace impl

template <
    class TensorView,
    index_t TilerNDim,
    nd_index<TilerNDim> kTilerDims,
    class TensorMask,
    auto kTileDesc,
    auto kElementTensorDesc,
    auto kPartition,
    class Memory>
MINT_HOST_DEVICE void masked_store_no_shuffle_z2(
    const TensorView& view,
    integral_constant<nd_index<TilerNDim>, kTilerDims>,
    const nd_index<TilerNDim>& tiler_idx,
    constant<kPartition> /*partition*/,
    const TensorMask& dst_tensor_mask,
    const mint::tensor::
        distributed_tensor<kTileDesc, kElementTensorDesc, Memory>& src) {
  constexpr auto dstrNDim = TensorView::ndim() - TilerNDim;

  const auto morpher = view.tensor_desc_.polymorpher().morphers()[0_ic];
  constexpr auto top_lengths = decltype(morpher)::kTopLengths;

  constexpr auto dstr_dims = [&]() {
    nd_index<dstrNDim> ret;
    index_t cnt = 0;
    for (index_t i = 0; i < TensorView::ndim(); i++) {
      if (std::find(kTilerDims.begin(), kTilerDims.end(), i) ==
          kTilerDims.end())
        ret[cnt++] = i;
    }
    return ret;
  }();

  // get dstr top lengths
  constexpr auto dstr_top_lengths = [&]() {
    nd_index<dstrNDim> ret;
    for (index_t i = 0; i < dstrNDim; i++) {
      ret[i] = top_lengths[dstr_dims[i]];
    }
    return ret;
  }();

  const auto dstr_layout = tensor::make_simple_distribution_z2(
      constant<dstr_top_lengths>{}, kPartition);

  constexpr auto element_layout =
      tensor::make_aliased_naive_packed_tensor_descriptor(
          make_index_sequence<dstr_layout.element_ndim()>{},
          index_constant<-1>{},
          dstr_layout.element_lengths());

  const auto tiler = tensor::impl::tiler_impl<
      TensorView,
      kTilerDims,
      dstr_layout,
      element_layout,
      kPartition>{view, tiler_idx};

  impl::masked_store_no_shuffle_impl(tiler, dst_tensor_mask, src);
}

template <
    class TensorView,
    auto kTilerDims,
    auto kTileDesc,
    auto kElementTensorDesc,
    class Memory,
    auto kPartition,
    class TensorMask = tuple<>>
MINT_HOST_DEVICE void masked_store_no_shuffle(
    const TensorView& /*view*/,
    const tensor::impl::tiler_impl<
        TensorView,
        kTilerDims,
        kTileDesc,
        kElementTensorDesc,
        kPartition>& tiler,
    const tensor::distributed_tensor<kTileDesc, kElementTensorDesc, Memory>&
        src_dstr_tensor,
    const TensorMask& dst_tensor_mask = tuple<>{}) {
  impl::masked_store_no_shuffle_impl(tiler, dst_tensor_mask, src_dstr_tensor);
}

template <
    class TensorView,
    index_t TilerNDim,
    nd_index<TilerNDim> kTilerDims,
    class DstrTensor,
    class TensorMask = tuple<>,
    auto kPartition>
MINT_HOST_DEVICE void masked_store_no_shuffle(
    const TensorView& view,
    integral_constant<nd_index<TilerNDim>, kTilerDims> tiler_dims,
    const nd_index<TilerNDim>& tiler_idx,
    constant<kPartition> partition,
    const DstrTensor& src_dstr_tensor,
    const TensorMask& dst_tensor_mask = tuple<>{}) {
  const auto tiler = tensor::make_tiler(
      view, tiler_dims, tiler_idx, src_dstr_tensor, partition);

  masked_store_no_shuffle(view, tiler, src_dstr_tensor, dst_tensor_mask);
}

} // namespace generic
} // namespace tile
} // namespace mint
