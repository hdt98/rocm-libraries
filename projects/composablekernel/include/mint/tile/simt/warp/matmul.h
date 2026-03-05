#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>
#include <mint/tile/generic/matmul_no_shuffle.h>

namespace mint {
namespace tile {
namespace simt {
namespace warp {

// C[M, N] += A[M, K] * B[K, N]
// warp matmul, no shuffle
template <class CTensor, class ATensor, class BTensor>
MINT_HOST_DEVICE void
matmul_mn_mk_kn_no_shuffle(CTensor& c, const ATensor& a, const BTensor& b) {
  ::mint::tile::generic::matmul_mn_mk_kn_no_shuffle(c, a, b);
}

// C[M, N] += A[M, K] * B[N, K]
// warp matmul, no shuffle
template <class CTensor, class ATensor, class BTensor>
MINT_HOST_DEVICE void
matmul_mn_mk_nk_no_shuffle(CTensor& c, const ATensor& a, const BTensor& b) {
  ::mint::tile::generic::matmul_mn_mk_nk_no_shuffle(c, a, b);
}

} // namespace warp
} // namespace simt
} // namespace tile
} // namespace mint
