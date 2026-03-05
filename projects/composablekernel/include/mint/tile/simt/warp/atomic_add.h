#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>
#include <mint/tile/generic/atomic_add_no_shuffle.h>

namespace mint {
namespace tile {
namespace simt {
namespace warp {

// warp atomic add, no shuffle
template <
    class TensorView,
    class TensorMask,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition,
    class Memory>
MINT_HOST_DEVICE void masked_atomic_add(
    const ::mint::tensor::distributed_window<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>& dst_win,
    const TensorMask& dst_mask,
    const ::mint::tensor::
        distributed_tensor<kDstrTensorDesc, kElementTensorDesc, Memory>& src) {
  ::mint::tile::generic::masked_atomic_add_no_shuffle(dst_win, dst_mask, src);
}

} // namespace warp
} // namespace simt
} // namespace tile
} // namespace mint
