/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "origami/hardware.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"

#include <vector>

namespace origami {
namespace streamk {

/**
 * @brief Returns number of k-iterations (depth-U steps) per output tile.
 *
 * Matches Tensile `getItersPerTile` when `mt_k` is the kernel depthU / BLK_K.
 */
inline constexpr size_t num_iters_per_tile(size_t mt_k, size_t k) {
  return math::safe_ceil_div(k, mt_k);
}

/**
 * @brief Total K-iterations across all output tiles (StreamK stream length in steps).
 */
inline constexpr size_t num_iters_total(size_t output_tiles, size_t iters_per_tile) {
  return output_tiles * iters_per_tile;
}

/**
 * @brief StreamK iterations assigned to each workgroup (ceil of stream / grid).
 */
inline constexpr size_t num_iters_per_cta(size_t iters_total, size_t grid) {
  return grid == 0 ? 0 : math::safe_ceil_div(iters_total, grid);
}

/**
 * @brief Fixup peer count (v2) for uneven StreamK schedules (partial / multi-tile WGs).
 */
inline constexpr size_t num_fixup_peers_v2(size_t g,
                                           size_t iters_total,
                                           size_t iters_per_tile,
                                           size_t iters_per_cta) {
  if (g == 0) return 0;
  size_t hasFixup = (iters_total % g == 0 && iters_per_cta % iters_per_tile == 0) ? 0 : 1;
  return math::safe_ceil_div(iters_per_tile, iters_per_cta) + hasFixup;
}

/**
 * @brief Number of output tiles.
 *
 * @param mt_m Tile size in M-dimension.
 * @param mt_n Tile size in N-dimension.
 * @param m Matrix's m-dimension.
 * @param n Matrix's n-dimension.
 * @param batch Number of batches.
 * @return size_t Total number of output tiles.
 */
size_t compute_number_of_output_tiles(size_t mt_m, size_t mt_n, size_t m, size_t n, size_t batch);

/**
 * @brief Select the best reduction strategy for StreamK.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param algorithm Grid selection algorithm
 * @return reduction_t Selected reduction strategy
 */
reduction_t select_reduction(const problem_t& problem,
                             const hardware_t& hardware,
                             const config_t& config,
                             grid_selection_t algorithm);

/**
 * @brief Based on the provided kernel config, select the best grid dimension.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param grid_selection_t grid selection algorithm (@see origami::grid_selection_t)
 * @param max_cus Maximum number of CUs to use.
 * @return size_t Dimensions of the grid launched.
 */
size_t select_grid_size(const problem_t& problem,
                        const hardware_t& hardware,
                        const config_t& config,
                        grid_selection_t algorithm,
                        size_t max_cus = 0);

}  // namespace streamk
}  // namespace origami
