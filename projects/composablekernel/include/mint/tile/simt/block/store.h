#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>
#include <mint/tile/generic/store_no_shuffle.h>
#include <mint/tile/generic/store_no_shuffle_tiler.h>
#include <mint/tile/generic/store_no_shuffle_vectorized.h>

namespace mint {
namespace tile {
namespace simt {
namespace block {

// block store, no shuffle
template <
    class TensorView,
    class TensorMask,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition,
    class Memory>
MINT_HOST_DEVICE void masked_store(
    const mint::tensor::distributed_window<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>& dst_win,
    const TensorMask& dst_mask,
    const mint::tensor::
        distributed_tensor<kDstrTensorDesc, kElementTensorDesc, Memory>& src) {
  mint::tile::generic::masked_store_no_shuffle(dst_win, dst_mask, src);
}

template <
    class TensorView,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    class Memory,
    class TensorMask = tuple<>,
    auto kPartition>
MINT_HOST_DEVICE void masked_store(
    const TensorView& view,
    const nd_index<TensorView::ndim()>& idx,
    constant<kPartition> partition,
    const mint::tensor::
        distributed_tensor<kDstrTensorDesc, kElementTensorDesc, Memory>& src,
    const TensorMask& dst_mask = tuple<>{}) {
  ::mint::tile::generic::masked_store_no_shuffle(
      view, idx, partition, dst_mask, src);
}

template <
    auto kElementVectorDimAliases, // dim_alias of element vector dims
    auto kElementVectorLengths, // vector_lengths of element vector dims
    class TensorView,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    class Memory,
    class TensorMask = tuple<>,
    auto kPartition>
MINT_HOST_DEVICE void masked_store_vectorized(
    const TensorView& view,
    const nd_index<TensorView::ndim()>& idx,
    constant<kPartition> partition,
    const mint::tensor::
        distributed_tensor<kDstrTensorDesc, kElementTensorDesc, Memory>& src,
    const TensorMask& dst_mask = tuple<>{}) {
  ::mint::tile::generic::experimental::masked_store_no_shuffle_vectorized<
      kElementVectorDimAliases,
      kElementVectorLengths>(view, idx, partition, dst_mask, src);
}

template <
    class TensorView,
    index_t kTilerNDim,
    nd_index<kTilerNDim> kTilerDims,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    class Memory,
    class TensorMask = tuple<>,
    auto kPartition>
MINT_HOST_DEVICE void masked_store(
    const TensorView& view,
    integral_constant<nd_index<kTilerNDim>, kTilerDims> tiler_dims,
    const nd_index<kTilerNDim>& tiler_idx,
    constant<kPartition> partition,
    const mint::tensor::
        distributed_tensor<kDstrTensorDesc, kElementTensorDesc, Memory>& src,
    const TensorMask& dst_mask = tuple<>{}) {
  ::mint::tile::generic::masked_store_no_shuffle(
      view, tiler_dims, tiler_idx, partition, src, dst_mask);
}

} // namespace block
} // namespace simt
} // namespace tile
} // namespace mint
