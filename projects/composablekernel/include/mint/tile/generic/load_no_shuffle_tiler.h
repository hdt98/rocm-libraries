#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>
#include <algorithm>

namespace mint {
namespace tile {
namespace generic {
namespace impl {

// Helper to check if two dimension arrays have any overlapping elements
template <auto kDims1, auto kDims2>
constexpr bool has_dimension_overlap() {
  for (index_t i = 0; i < kDims1.size(); i++) {
    if (std::find(kDims2.begin(), kDims2.end(), kDims1[i]) != kDims2.end()) {
      return true;
    }
  }
  return false;
}

// generic scope, no shuffle
template <
    class TensorView,
    class TensorMask,
    auto kTilerDims,
    auto kTileDesc,
    auto kElementTensorDesc,
    auto kPartition>
  requires(
      kTileDesc.can_bottom_up() && kElementTensorDesc.bottom_ndim() == 1 &&
      kElementTensorDesc.top_ndim() == kTileDesc.element_ndim())
MINT_HOST_DEVICE auto masked_load_no_shuffle_impl(
    const typename mint::tensor::impl::tiler_impl<
        TensorView,
        kTilerDims,
        kTileDesc,
        kElementTensorDesc,
        kPartition>& tiler,
    const TensorMask& src_tensor_mask) {
  using namespace mint::tensor;
  using value_type = typename TensorView::value_type;
  using mem_type = owned_vgpr_memory<value_type, kTileDesc.element_size()>;
  using tile_coord_type [[maybe_unused]] =
      coordinate<decltype(kTileDesc.tensor_desc())>;

  distributed_tensor<kTileDesc, kElementTensorDesc, mem_type> dst{};

  auto src_coord = tiler.coord_;

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

  constexpr auto tile_dims = std::remove_cvref_t<decltype(tiler)>::tile_dims();

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

  const auto src_tensor_view = tiler.tensor_view();

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
          src_coord,
          src_tensor_view.tensor_desc(),
          get_full_idx(tile_top_idx_delta));
    }

    // FIXME: need to pass in customized value
    value_type v = value_type{0};
    if (is_unmasked(src_coord, src_tensor_view.tensor_desc(), src_tensor_mask))
        [[likely]] {
      v = src_tensor_view.element(src_coord);
    }

    constexpr index_t elem_offset =
        kElementTensorDesc.calculate_bottom_index(snake_idx)[0];
    dst.memory().template at<elem_offset>() = v;
  });

  return dst;
}

} // namespace impl

template <
    class TensorView,
    index_t kTilerNDim,
    nd_index<kTilerNDim> kTilerDims,
    class TensorMask = tuple<>,
    auto kPartition>
MINT_HOST_DEVICE auto masked_load_no_shuffle_z2(
    const TensorView& view,
    integral_constant<nd_index<kTilerNDim>, kTilerDims>,
    const nd_index<kTilerNDim>& tiler_idx,
    constant<kPartition> /*partition*/,
    const TensorMask& src_tensor_mask = tuple<>{}) {
  constexpr auto dstr_ndim = TensorView::ndim() - kTilerNDim;

  auto morphers = view.tensor_desc_.polymorpher();
  static_assert(
      morphers.num_morpher_ == 1,
      "TensorView must contain exactly one morpher");

  const auto morpher = view.tensor_desc_.polymorpher().morphers()[0_ic];

  constexpr auto top_lengths = decltype(morpher)::kTopLengths;

  constexpr auto dstr_dims = [&]() {
    nd_index<dstr_ndim> ret;
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
    nd_index<dstr_ndim> ret;
    for (index_t i = 0; i < dstr_ndim; i++) {
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

  return impl::masked_load_no_shuffle_impl(tiler, src_tensor_mask);
}

template <
    class TensorView,
    auto kTilerDims,
    auto kTileDesc,
    auto kElementTensorDesc,
    class Memory,
    auto kPartition,
    class TensorMask>
MINT_HOST_DEVICE void masked_load_no_shuffle(
    const tensor::impl::tiler_impl<
        TensorView,
        kTilerDims,
        kTileDesc,
        kElementTensorDesc,
        kPartition>& tiler,
    tensor::distributed_tensor<kTileDesc, kElementTensorDesc, Memory>&
        dstr_tensor,
    const TensorMask& src_tensor_mask) {
  dstr_tensor = impl::masked_load_no_shuffle_impl(tiler, src_tensor_mask);
}

template <
    class TensorView,
    auto kTilerDims,
    auto kTileDesc,
    auto kElementTensorDesc,
    auto kPartition,
    class TensorMask>
MINT_HOST_DEVICE auto masked_load_no_shuffle(
    const tensor::impl::tiler_impl<
        TensorView,
        kTilerDims,
        kTileDesc,
        kElementTensorDesc,
        kPartition>& tiler,
    const TensorMask& src_tensor_mask) {
  return impl::masked_load_no_shuffle_impl(tiler, src_tensor_mask);
}

template <
    class TensorView,
    index_t kTilerNDim,
    nd_index<kTilerNDim> kTilerDims,
    class DstrTensor,
    class TensorMask,
    auto kPartition>
MINT_HOST_DEVICE void masked_load_no_shuffle(
    const TensorView& view,
    integral_constant<nd_index<kTilerNDim>, kTilerDims> tiler_dims,
    const nd_index<kTilerNDim>& tiler_idx,
    DstrTensor& dstr_tensor,
    constant<kPartition> partition,
    const TensorMask& src_tensor_mask) {
  const auto tiler =
      tensor::make_tiler(view, tiler_dims, tiler_idx, dstr_tensor, partition);

  masked_load_no_shuffle(tiler, dstr_tensor, src_tensor_mask);
}

} // namespace generic
} // namespace tile
} // namespace mint
