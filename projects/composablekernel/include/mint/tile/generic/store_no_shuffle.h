#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>

namespace mint {
namespace tile {
namespace generic {

// generic scope, no_shuffle, masked
template <
    class TensorView,
    class TensorMask,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition,
    class Memory>
MINT_HOST_DEVICE void masked_store_no_shuffle(
    const mint::tensor::distributed_window<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>& dst_win,
    const TensorMask& dst_tensor_mask,
    const mint::tensor::
        distributed_tensor<kDstrTensorDesc, kElementTensorDesc, Memory>& src) {
  using value_type = typename TensorView::value_type;

  const TensorView& dst_tensor_view = dst_win.tensor_view();

  auto dst_coord = dst_win.coord_;

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

  constexpr index_t elem_top_ndim = kElementTensorDesc.top_ndim();
  constexpr auto elem_top_lengths = kDstrTensorDesc.element_lengths();

  // loop
  static_for_nd2<elem_top_lengths>()([&](auto... is) {
    constexpr auto idx = nd_index<elem_top_ndim>{is...};
    constexpr auto old_idx = get_old_idx(elem_top_lengths, idx);
    constexpr auto snake_idx = get_snake_idx(elem_top_lengths, idx);
    constexpr auto snake_old_idx = get_snake_idx(elem_top_lengths, old_idx);

    // move forward
    if constexpr (idx != nd_index<elem_top_ndim>{}.fill(0)) {
      constexpr auto dstr_top_idx_delta =
          kDstrTensorDesc.top_index_delta(snake_idx - snake_old_idx);
      move_coordinate(
          dst_coord, dst_tensor_view.tensor_desc(), dstr_top_idx_delta);
    }

    constexpr index_t elem_offset =
        kElementTensorDesc.calculate_bottom_index(snake_idx)[0];
    value_type v = src.memory().template at<elem_offset>();

    if (is_unmasked(dst_coord, dst_tensor_view.tensor_desc(), dst_tensor_mask))
      dst_tensor_view.element(dst_coord) = v;
  });
}

template <
    class TensorView,
    class TensorMask,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition,
    class Memory>
MINT_HOST_DEVICE void masked_store_no_shuffle(
    const TensorView& view,
    const nd_index<TensorView::ndim()>& idx,
    constant<kPartition> partition,
    const TensorMask& dst_tensor_mask,
    const mint::tensor::
        distributed_tensor<kDstrTensorDesc, kElementTensorDesc, Memory>& src) {
  const auto dst_win = make_distributed_window(
      view,
      idx,
      constant<kDstrTensorDesc>{},
      constant<kElementTensorDesc>{},
      partition);

  masked_store_no_shuffle(dst_win, dst_tensor_mask, src);
}

} // namespace generic
} // namespace tile
} // namespace mint
