#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor/tensor_descriptor_helper.h>
#include <mint/tensor/tensor_view.h>

namespace mint {
namespace tensor {
namespace impl {

template <
    class TensorView,
    auto kLoopDims,
    auto kTileDesc,
    auto kElementTensorDesc,
    auto kPartition>
  requires(
      TensorView::ndim() == (kTileDesc.top_ndim() + kLoopDims.size()) &&
      kTileDesc.element_ndim() == kElementTensorDesc.top_ndim() &&
      kElementTensorDesc.bottom_ndim() == 1)
struct tiler_impl {
  using tensor_view_type = remove_reference_t<TensorView>;
  using tensor_desc_type = typename tensor_view_type::tensor_desc_type;
  using coord_type = coordinate<tensor_desc_type>;
  using tile_desc_type = remove_cvref_t<decltype(kTileDesc)>;
  // using element_tensor_desc_type =
  // remove_cvref_t<decltype(kElementTensorDesc)>;
  using value_type = typename tensor_view_type::value_type;
  using tile_coord_type = coordinate<typename tile_desc_type::tensor_desc_type>;

  const tensor_view_type tensor_view_;
  coord_type coord_;
  index_t coord_byte_offset_;
  tile_coord_type tile_coord_;

 public:
  MINT_HOST_DEVICE static consteval index_t ndim() {
    return tile_desc_type::top_ndim() + kLoopDims.size();
  }

  MINT_HOST_DEVICE index_t byte_offset() const noexcept {
    return coord_byte_offset_;
  }

  MINT_HOST_DEVICE static consteval index_t tile_ndim() {
    return tile_desc_type::top_ndim();
  }

  MINT_HOST_DEVICE static consteval index_t loop_ndim() {
    return kLoopDims.size();
  }

  MINT_HOST_DEVICE static consteval auto tile_lengths() {
    return tile_desc_type::top_lengths();
  }

  MINT_HOST_DEVICE static consteval auto tile_dims() {
    nd_index<tile_ndim()> ret;
    index_t cnt = 0;
    for (index_t i = 0; i < TensorView::ndim(); i++) {
      if (std::find(kLoopDims.begin(), kLoopDims.end(), i) == kLoopDims.end())
        ret[cnt++] = i;
    }
    return ret;
  }

  static constexpr auto kTileNDim = tile_ndim();
  static constexpr auto kTileDims = tile_dims();

  MINT_HOST_DEVICE void set_tile_index(
      const nd_index<kLoopDims.size()>& loop_idx) {
    const auto top_idx =
        merge_loop_and_tile_idx(loop_idx, tile_coord_.get_top_index());
    coord_.set_top_index_and_propagate_down(
        tensor_view_.tensor_desc(), top_idx);
    update_byte_offset();
  }

  MINT_HOST_DEVICE void move_tiler(
      const nd_index<kLoopDims.size()>& loop_delta_idx) {
    const auto new_idx = inc_loop_idx(coord_.get_top_index(), loop_delta_idx);
    coord_.set_top_index_and_propagate_down(
        tensor_view_.tensor_desc(), new_idx);
    update_byte_offset();
  }

  MINT_HOST_DEVICE constexpr tiler_impl(
      const tensor_view_type& tensor_view_in,
      const nd_index<kLoopDims.size()>& loop_idx)
      : tensor_view_{tensor_view_in},
        coord_{},
        coord_byte_offset_{},
        tile_coord_{} {
    static_assert(
        tile_desc_type::top_ndim() + kLoopDims.size() ==
        coord_type::top_ndim());

    initialize_tile_coord();
    const auto top_idx =
        merge_loop_and_tile_idx(loop_idx, tile_coord_.get_top_index());
    coord_ = coord_type{tensor_view_.tensor_desc(), top_idx};
    update_byte_offset();
  }

  MINT_HOST_DEVICE const tensor_view_type& tensor_view() const noexcept {
    return tensor_view_;
  }

  MINT_HOST_DEVICE static consteval auto tile_desc() noexcept {
    return kTileDesc;
  }

  MINT_HOST_DEVICE static consteval auto element_tensor_desc() noexcept {
    return kElementTensorDesc;
  }

  template <alias_t alias>
  MINT_HOST_DEVICE auto alias_to_index() {
    constexpr auto index = tensor_desc_type::template top_alias_to_dim<alias>();
    return coord_.all_index()[index];
  }

 private:
  template <class TileIdx>
  MINT_HOST_DEVICE static constexpr nd_index<ndim()> merge_loop_and_tile_idx(
      const nd_index<kLoopDims.size()>& loop_idx,
      const TileIdx& tile_idx) {
    nd_index<ndim()> ret;
    for (index_t i = 0; i < kLoopDims.size(); i++) {
      ret[kLoopDims[i]] = loop_idx[i];
    }

    for (index_t i = 0; i < kTileNDim; i++) {
      ret[kTileDims[i]] = tile_idx[i];
    }
    return ret;
  }

  MINT_HOST_DEVICE static constexpr nd_index<ndim()> inc_loop_idx(
      const nd_index<coord_type::top_ndim()>& old_idx,
      const nd_index<kLoopDims.size()>& loop_delta_idx) {
    nd_index<ndim()> ret;
    for (index_t i = 0; i < kLoopDims.size(); i++) {
      ret[kLoopDims[i]] = old_idx[kLoopDims[i]] + loop_delta_idx[i];
    }

    for (index_t i = 0; i < kTileNDim; i++) {
      ret[kTileDims[i]] = old_idx[kTileDims[i]];
    }
    return ret;
  }

  MINT_HOST_DEVICE void initialize_tile_coord() {
    tile_coord_.all_index()
        .template subset_reference<
            tile_desc_type::element_ndim(),
            tile_desc_type::element_dims()>()
        .fill(0);
    tile_coord_.all_index()
        .template set_subset<
            tile_desc_type::partition_ndim(),
            tile_desc_type::partition_dims()>(kPartition.my_partition_idx());
    kTileDesc.tensor_desc().polymorpher().propagate_index_bottom_up(
        tile_coord_.all_index());
  }

  MINT_HOST_DEVICE void update_byte_offset() {
    coord_byte_offset_ = coord_.get_bottom_index()[0] * sizeof(value_type);
  }
};

} // namespace impl

template <
    class TensorView,
    auto kLoopDims,
    auto kTileDesc,
    auto kElementTensorDesc,
    auto kPartition>
  requires(
      TensorView::ndim() == (kTileDesc.top_ndim() + kLoopDims.size()) &&
      kTileDesc.element_ndim() == kElementTensorDesc.top_ndim() &&
      kElementTensorDesc.bottom_ndim() == 1)
struct tiler : impl::tiler_impl<
                   TensorView,
                   kLoopDims,
                   kTileDesc,
                   kElementTensorDesc,
                   kPartition> {
  using base = impl::tiler_impl<
      TensorView,
      kLoopDims,
      kTileDesc,
      kElementTensorDesc,
      kPartition>;
  using tensor_view_type = typename base::tensor_view_type;

  using impl::tiler_impl<
      TensorView,
      kLoopDims,
      kTileDesc,
      kElementTensorDesc,
      kPartition>::tiler_impl;

  MINT_HOST_DEVICE static consteval bool is_const_tiler() {
    return false;
  }
};

template <
    class TensorView,
    auto kLoopDims,
    auto kTileDesc,
    auto kElementTensorDesc,
    auto kPartition>
  requires(
      TensorView::ndim() == (kTileDesc.top_ndim() + kLoopDims.size()) &&
      kTileDesc.element_ndim() == kElementTensorDesc.top_ndim() &&
      kElementTensorDesc.bottom_ndim() == 1)
struct const_tiler : impl::tiler_impl<
                         TensorView,
                         kLoopDims,
                         kTileDesc,
                         kElementTensorDesc,
                         kPartition> {
  using base = impl::tiler_impl<
      TensorView,
      kLoopDims,
      kTileDesc,
      kElementTensorDesc,
      kPartition>;
  using tensor_view_type = typename base::tensor_view_type;

  using impl::tiler_impl<
      TensorView,
      kLoopDims,
      kTileDesc,
      kElementTensorDesc,
      kPartition>::tiler_impl;

  MINT_HOST_DEVICE static consteval bool is_const_tiler() {
    return true;
  }
};

template <
    class TensorView,
    index_t kLoopNDim,
    nd_index<kLoopNDim> kLoopDims,
    auto kTileDesc,
    auto kElementTensorDesc,
    auto kPartition,
    class Memory>
  requires(not TensorView::is_const_tensor_view())
MINT_HOST_DEVICE auto make_tiler(
    const TensorView& view,
    integral_constant<nd_index<kLoopNDim>, kLoopDims>,
    const nd_index<kLoopNDim>& loop_idx,
    distributed_tensor<kTileDesc, kElementTensorDesc, Memory>,
    constant<kPartition>) {
  return tiler<
      TensorView,
      kLoopDims,
      kTileDesc,
      kElementTensorDesc,
      kPartition>{view, loop_idx};
}

template <
    class TensorView,
    index_t kLoopNDim,
    nd_index<kLoopNDim> kLoopDims,
    auto kTileDesc,
    auto kElementTensorDesc,
    auto kPartition,
    class Memory>
  requires(TensorView::is_const_tensor_view())
MINT_HOST_DEVICE auto make_tiler(
    const TensorView& view,
    integral_constant<nd_index<kLoopNDim>, kLoopDims>,
    const nd_index<kLoopNDim>& loop_idx,
    distributed_tensor<kTileDesc, kElementTensorDesc, Memory>,
    constant<kPartition>) {
  return const_tiler<
      TensorView,
      kLoopDims,
      kTileDesc,
      kElementTensorDesc,
      kPartition>{view, loop_idx};
}

} // namespace tensor
} // namespace mint
