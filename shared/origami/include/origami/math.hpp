// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>

namespace origami {
namespace math {

/**
 * @brief Performs `(n + d - 1) / d`, but is robust against the case where
 * `(n + d - 1)` would overflow.
 *
 */
template <typename N, typename D>
inline constexpr N safe_ceil_div(N n, D d) {
  // Static cast to undo integral promotion.
  return static_cast<N>(d == 0 ? 0 : (n / d + (n % d != 0 ? 1 : 0)));
}

/**
 * @brief Round elements so the contiguous dimension is a multiple of 128B
 * 
 */
inline std::size_t round_elems_to_128B(std::size_t elems, int bytes_per_elem) {
  const std::size_t bytes         = elems * static_cast<std::size_t>(bytes_per_elem);
  const std::size_t rounded_bytes = (bytes + 127) / 128 * 128;
  return rounded_bytes / static_cast<std::size_t>(bytes_per_elem);
}

}  // namespace math
}  // namespace origami
