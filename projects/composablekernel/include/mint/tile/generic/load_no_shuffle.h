#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>

namespace mint {
namespace tile {
namespace generic {
namespace impl {

// generic scope, no shuffle
template <
    class TensorView,
    class TensorMask,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition>
  requires(
      kDstrTensorDesc.can_bottom_up() &&
      kElementTensorDesc.bottom_ndim() == 1 &&
      kElementTensorDesc.top_ndim() == kDstrTensorDesc.element_ndim())
MINT_HOST_DEVICE auto masked_load_no_shuffle_impl(
    const TensorView& src_tensor_view,
    const TensorMask& src_tensor_mask,
    const typename mint::tensor::impl::distributed_window_impl<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>::coord_type& src_coord_in) {
  using namespace mint::tensor;
  using value_type = typename TensorView::value_type;
  using mem_type =
      owned_vgpr_memory<value_type, kDstrTensorDesc.element_size()>;
  using dstr_coord_type [[maybe_unused]] =
      coordinate<decltype(kDstrTensorDesc.tensor_desc())>;

  distributed_tensor<kDstrTensorDesc, kElementTensorDesc, mem_type> dst{};

  auto src_coord = src_coord_in;

  //
  constexpr auto get_old_idx = []<index_t ndim>(
                                   const nd_index<ndim>& lengths,
                                   const nd_index<ndim>& idx) {
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
  };

  //
  constexpr auto get_snake_idx = []<index_t ndim>(
                                     const nd_index<ndim>& lengths,
                                     const nd_index<ndim>& idx) {
    nd_index<ndim> ret{};
    ret[0] = idx[0];
    index_t cnt = idx[0];
    for (index_t i = 1; i < ndim; i++) {
      ret[i] = (cnt % 2 == 0) ? idx[i] : lengths[i] - 1 - idx[i];
      cnt = cnt * lengths[i] + idx[i];
    }
    return ret;
  };

  constexpr index_t elem_top_ndim = kElementTensorDesc.top_ndim();
  constexpr auto elem_top_lengths = kDstrTensorDesc.element_lengths();

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
          src_coord, src_tensor_view.tensor_desc(), dstr_top_idx_delta);
    }

    // FIXME: need to pass in customized value
    value_type v = value_type{0};
    if (is_unmasked(src_coord, src_tensor_view.tensor_desc(), src_tensor_mask))
      v = src_tensor_view.element(src_coord);

    constexpr index_t elem_offset =
        kElementTensorDesc.calculate_bottom_index(snake_idx)[0];
    dst.memory().template at<elem_offset>() = v;
  });

  return dst;
}

} // namespace impl

// generic scope, no shuffle, masked
template <
    class TensorView,
    class TensorMask = tuple<>,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition>
  requires(
      kDstrTensorDesc.can_bottom_up() && kElementTensorDesc.bottom_ndim() == 1)
MINT_HOST_DEVICE auto masked_load_no_shuffle(
    const mint::tensor::impl::distributed_window_impl<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>& src_win,
    const TensorMask& src_tensor_mask = tuple<>{}) {
  return impl::masked_load_no_shuffle_impl<
      TensorView,
      TensorMask,
      kDstrTensorDesc,
      kElementTensorDesc,
      kPartition>(src_win.tensor_view(), src_tensor_mask, src_win.coord_);
}

template <
    class TensorView,
    class TensorMask = tuple<>,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    class Memory,
    auto kPartition>
  requires(kDstrTensorDesc.can_bottom_up())
MINT_HOST_DEVICE void masked_load_no_shuffle(
    const TensorView& view,
    const nd_index<TensorView::ndim()>& idx,
    mint::tensor::
        distributed_tensor<kDstrTensorDesc, kElementTensorDesc, Memory>& dstr,
    constant<kPartition> partition,
    const TensorMask& src_tensor_mask = tuple<>{}) {
  const auto src_win = make_distributed_window(
      view,
      idx,
      constant<kDstrTensorDesc>{},
      constant<kElementTensorDesc>{},
      partition);

  dstr = impl::masked_load_no_shuffle_impl<
      TensorView,
      TensorMask,
      kDstrTensorDesc,
      kElementTensorDesc,
      kPartition>(view, src_tensor_mask, src_win.coord_);
}

} // namespace generic
} // namespace tile
} // namespace mint
