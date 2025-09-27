// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

namespace origami {
namespace math {

/**
 * Performs `(n + d - 1) / d`, but is robust against the case where
 * `(n + d - 1)` would overflow.
 */
template <typename N, typename D>
__device__ __host__ inline constexpr N safe_ceil_div(N n, D d) {
  // Static cast to undo integral promotion.
  return static_cast<N>(d == 0 ? 0 : (n / d + (n % d != 0 ? 1 : 0)));
}

}  // namespace math
}  // namespace origami