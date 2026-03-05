#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>
#include <mint/tile/generic/load_no_shuffle.h>
#include <mint/tile/generic/load_no_shuffle_tiler.h>
#include <mint/tile/generic/load_no_shuffle_vectorized.h>

namespace mint {
namespace tile {
namespace simt {
namespace warp {

// warp load, no shuffle, masked
template <
    class TensorView,
    class TensorMask = tuple<>,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition>
MINT_HOST_DEVICE auto masked_load(
    const mint::tensor::impl::distributed_window_impl<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>& src_win,
    const TensorMask& src_mask = tuple<>{}) {
  return ::mint::tile::generic::masked_load_no_shuffle(src_win, src_mask);
}

template <
    class TensorView,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    class Memory,
    class TensorMask = tuple<>,
    auto kPartition = thread_in_this_warp{}>
  requires(kDstrTensorDesc.can_bottom_up())
MINT_HOST_DEVICE void masked_load(
    const TensorView& view,
    const nd_index<TensorView::ndim()>& idx,
    mint::tensor::
        distributed_tensor<kDstrTensorDesc, kElementTensorDesc, Memory>& dstr,
    const TensorMask& src_mask = tuple<>{},
    constant<kPartition> partition = constant<kPartition>{}) {
  ::mint::tile::generic::masked_load_no_shuffle(
      view, idx, dstr, partition, src_mask);
}

template <
    auto kElementVectorDimAliases, // dim_alias of element vector dims
    auto kElementVectorLengths, // vector_lengths of element vector dims
    class TensorView,
    class TensorMask = tuple<>,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    class Memory,
    auto kPartition = thread_in_this_warp{}>
  requires(kDstrTensorDesc.can_bottom_up())
MINT_HOST_DEVICE void masked_load_vectorized(
    const TensorView& view,
    const nd_index<TensorView::ndim()>& idx,
    mint::tensor::
        distributed_tensor<kDstrTensorDesc, kElementTensorDesc, Memory>& dstr,
    const TensorMask& src_mask = tuple<>{},
    constant<kPartition> partition = constant<kPartition>{}) {
  ::mint::tile::generic::experimental::masked_load_no_shuffle_vectorized<
      kElementVectorDimAliases,
      kElementVectorLengths>(view, idx, dstr, partition, src_mask);
}

template <
    class TensorView,
    index_t kTilerNDim,
    nd_index<kTilerNDim> kTilerDims,
    class DstrTensor,
    class TensorMask = tuple<>,
    auto kPartition = thread_in_this_warp{}>
MINT_HOST_DEVICE void masked_load(
    const TensorView& view,
    integral_constant<nd_index<kTilerNDim>, kTilerDims> tiler_dims,
    const nd_index<kTilerNDim>& tiler_idx,
    DstrTensor& dstr_tensor,
    const TensorMask& src_mask = tuple<>{},
    constant<kPartition> partition = constant<kPartition>{}) {
  ::mint::tile::generic::masked_load_no_shuffle(
      view, tiler_dims, tiler_idx, dstr_tensor, partition, src_mask);
}

template <
    class TensorView,
    index_t kTilerNDim,
    nd_index<kTilerNDim> kTilerDims,
    class TensorMask = tuple<>,
    auto kPartition = thread_in_this_warp{}>
MINT_HOST_DEVICE auto masked_load_z2(
    const TensorView& view,
    integral_constant<nd_index<kTilerNDim>, kTilerDims> tiler_dims,
    const nd_index<kTilerNDim>& tiler_idx,
    const TensorMask& src_mask = tuple<>{},
    constant<kPartition> partition = constant<kPartition>{}) {
  return ::mint::tile::generic::masked_load_no_shuffle_z2(
      view, tiler_dims, tiler_idx, partition, src_mask);
}

template <
    class TensorView,
    class Tiler,
    class DstrTensor,
    class TensorMask = tuple<>>
MINT_HOST_DEVICE void masked_load(
    const Tiler& tiler,
    DstrTensor& dstr_tensor,
    const TensorMask& src_tensor_mask = tuple<>{}) {
  ::mint::tile::generic::masked_load_no_shuffle(
      tiler, dstr_tensor, src_tensor_mask);
}

template <class TensorView, class Tiler, class TensorMask = tuple<>>
MINT_HOST_DEVICE auto masked_load(
    const Tiler& tiler,
    const TensorMask& src_tensor_mask = tuple<>{}) {
  return ::mint::tile::generic::masked_load_no_shuffle(tiler, src_tensor_mask);
}

} // namespace warp
} // namespace simt
} // namespace tile
} // namespace mint
