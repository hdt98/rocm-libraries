#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>
#include <mint/tile/generic/elementwise_no_shuffle.h>

namespace mint {
namespace tile {
namespace simt {
namespace warp {

template <class... Tensors, class FElementwise>
MINT_HOST_DEVICE void elementwise(
    const FElementwise& f_elementwise,
    Tensors&... tensors) {
  mint::tile::generic::elementwise_no_shuffle(f_elementwise, tensors...);
}

} // namespace warp
} // namespace simt
} // namespace tile
} // namespace mint
