// Copyright Advanced Micro Devices, Inconfig., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/streamk.hpp"
#include "origami/hardware.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"
#include "origami/utils.hpp"

namespace origami {
namespace streamk {
constexpr size_t num_iters_total(size_t output_tiles, size_t iters_per_tile) {
  return output_tiles * iters_per_tile;
}

constexpr size_t num_iters_per_tile(size_t BLK_K, size_t k) {
  return math::safe_ceil_div(k, BLK_K);
}

constexpr size_t num_iters_per_cta(size_t iters_total, int g) {
  return math::safe_ceil_div(iters_total, g);
}

constexpr size_t number_of_output_tiles(size_t BLK_M,
                                        size_t BLK_N,
                                        size_t m,
                                        size_t n,
                                        size_t batch) {
  size_t m_tiles = math::safe_ceil_div(m, BLK_M);
  size_t n_tiles = math::safe_ceil_div(n, BLK_N);
  return m_tiles * n_tiles * batch;
}

constexpr size_t num_fixup_peers(size_t g,
                                 size_t iters_total,
                                 size_t iters_per_tile,
                                 size_t iters_per_cta) {
  // If tiles don't evenly divide there are always at least 2 fixup peers, and more if
  // iters_per_tile > iters_per_cta
  size_t hasFixup =
      (iters_total % g == 0 &&               // Check if some WGs have more iters than others
       iters_per_cta % iters_per_tile == 0)  // Check if WGs have an even number of full tiles
          ? 0
          : 1;
  return math::safe_ceil_div(iters_per_tile, iters_per_cta) + hasFixup;
}

std::tuple<double, size_t, size_t, double> predicted_runtime(size_t BLK_M,
                                                             size_t BLK_N,
                                                             size_t BLK_K,
                                                             size_t m,
                                                             size_t n,
                                                             size_t k,
                                                             size_t batch,
                                                             int g,
                                                             double a,
                                                             double b,
                                                             double c,
                                                             double d) {
  size_t output_tiles   = number_of_output_tiles(BLK_M, BLK_N, m, n, batch);
  size_t iters_per_tile = num_iters_per_tile(BLK_K, k);
  size_t iters_total    = num_iters_total(output_tiles, iters_per_tile);
  size_t iters_per_cta  = num_iters_per_cta(iters_total, g);
  size_t fixup_peers    = num_fixup_peers(g, iters_total, iters_per_tile, iters_per_cta);

  size_t remainder_tiles = output_tiles % g;
  double k_split_ratio   = remainder_tiles / static_cast<double>(g);

  double cache_penalty = 0.0;
  if (fixup_peers >= 1) {
    // Calculate the ideal equal split ratio
    double ideal_split_ratio = 1.0 / fixup_peers;

    // Measure deviation from the ideal equal split
    double imbalance = 1 / std::abs(k_split_ratio - ideal_split_ratio);

    // Scale the penalty by the imbalance and the per-collaborator cost (d)
    cache_penalty = d * imbalance * fixup_peers;
  }

  // Include the cache penalty in the runtime prediction
  double runtime =
      a + (b * (fixup_peers > 1)) + (c * iters_per_cta) + (d * (fixup_peers - 1)) + cache_penalty;

  return std::make_tuple(runtime, iters_per_cta, fixup_peers, cache_penalty);
}

size_t get_workspace(size_t x,
                     size_t y,
                     size_t mt_m,
                     size_t mt_n,
                     size_t bpe_c,
                     size_t grid,
                     size_t tiles,
                     reduction_t reduction) {
  size_t size = 0;
  if (reduction == reduction_t::Tree) {
    if (tiles % grid == 0) {
      size_t tileSize = mt_m * mt_n * bpe_c;
      size += tileSize * grid;
    }
  } else if (reduction == reduction_t::Parallel) {
    size_t splitSize  = x * y * bpe_c;
    size_t splitCount = grid / tiles;
    size += splitSize * splitCount;
  }
  return size;
}

reduction_t select_reduction(size_t x,
                             size_t y,
                             size_t z,
                             size_t batch,
                             size_t mt_m,
                             size_t mt_n,
                             size_t mt_k,
                             const hardware_t& hardware,
                             grid_selection_t algorithm) {
  reduction_t reductionStrat = reduction_t::Tree;

  if (algorithm == grid_selection_t::k_split_aware) {
    size_t tiles    = number_of_output_tiles(mt_m, mt_n, x, y, batch);
    size_t cu_count = hardware.N_CU;
    size_t iters_per_tile =
        num_iters_per_tile(z, mt_k);  // std::max(size_t(1), math::safe_ceil_div(z, mt_k));

    if (tiles < cu_count) {
      // For problems with large k and low number of tiles, use parallel reduction
      // TODO Benchmark to check if limits are correct
      constexpr int MinItersForParallel = 64;
      constexpr int MaxTilesForParallel = 16;
      if (iters_per_tile >= MinItersForParallel && tiles <= MaxTilesForParallel)
        reductionStrat = reduction_t::Parallel;
    }
  }

  return reductionStrat;
}

std::size_t grid_min_resources(const problem_t& problem,
                               const hardware_t& hardware,
                               const config_t& config) {
  std::size_t cu_count = hardware.N_CU;
  std::size_t tiles    = number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);
  return std::min(cu_count, tiles);
}

std::size_t grid_energy_aware(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config) {
  std::size_t cu_count = hardware.N_CU;
  std::size_t tiles    = number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);
  std::size_t sk_grid = cu_count;

  if (tiles > sk_grid) {
    for (std::size_t i = 1; i <= 32; i *= 2) {
      std::size_t tilesPerCU  = math::safe_ceil_div(i * tiles, cu_count);
      std::size_t reducedGrid = math::safe_ceil_div(i * tiles, tilesPerCU);
      float utilization       = float(reducedGrid) / float(cu_count);
      if (utilization > 0.75f) {
        if (utilization < 1.0f) sk_grid = reducedGrid;
        break;
      }
    }
  }
  return std::min(sk_grid, tiles);
}

std::size_t grid_reduction_cost_aware(const problem_t& problem,
                                      const hardware_t& hardware,
                                      const config_t& config) {
  // Fixed overhead alpha (a), fixed-size cost incurred by
  // each work-group, e.g. the grid launch latency, the initial
  // compulsary cache misses, the cost of writing the final output tile
  // to config.
  // double a = 5544 + 9130;
  double a = 2.772 + 4.565;  // 5.04 + 8.30;

  // Beta (b) incorporates conditional costs of outputting temporary partial
  // sums for scenarios where the number of output tiles does not quantize
  // perfectly across the number of processors.
  double b = 3.01;  // 5.47; 6017;

  // c represents instruction and stall workload of each MAC-iteration.
  double c = 2.2935;  // 4.17; 4587;

  // Delta (d) is the cost of reading and accumulating the partial sums from
  // other work-groups covering the same tile.
  double d = 10.22;  // 18.59; 20449;

  std::pair<size_t, double> min_grid_runtime;
  min_grid_runtime.second = std::numeric_limits<double>::max();

  
  std::size_t grid_start = 1;
  std::size_t grid_end = hardware.N_CU;
  
  // Predict the number of CTAs to use between 1 and 304
  std::size_t g = grid_start;
  for (; g <= static_cast<size_t>(grid_end); ++g) {
    auto [runtime, iters_per_cta, fixup_peers, cache_penalty] = predicted_runtime(config.mt.m,
                                                                                  config.mt.n,
                                                                                  config.mt.k,
                                                                                  problem.size.m,
                                                                                  problem.size.n,
                                                                                  problem.size.k,
                                                                                  problem.batch,
                                                                                  g,
                                                                                  a,
                                                                                  b,
                                                                                  c,
                                                                                  d);

    if (min_grid_runtime.second > runtime) {
      min_grid_runtime.first  = g;
      min_grid_runtime.second = runtime;
    }
  }

  return min_grid_runtime.first;
}

std::size_t grid_data_parallel(const problem_t& problem, const config_t& config) {
  return number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);
}

std::size_t grid_analytical(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config) {
  std::size_t biggest_allowable_split = 8;

  // Extract values from problem and config
  std::size_t M             = problem.size.m;
  std::size_t N             = problem.size.n;
  std::size_t K             = problem.size.k;
  std::size_t batch         = problem.batch;
  bool transpose_a          = problem.transpose_a;
  bool transpose_b          = problem.transpose_b;
  data_type_t mi_dtype      = problem.mi_dtype;
  std::size_t mx_block_size = problem.mx_block_size;

  // Calculate element sizes
  std::size_t element_size_A   = data_type_to_bits(problem.a_dtype);
  std::size_t element_size_B   = data_type_to_bits(problem.b_dtype);
  std::size_t element_size_out = data_type_to_bits(problem.d_dtype);

  // Extract dimensions for grid calculation
  std::size_t MT_M = config.mt.m;
  std::size_t MT_N = config.mt.n;

  // compute how many 32×32 tiles are needed in each dim,
  // then multiply to get total grid size:
  std::size_t grid = ((M + MT_M - 1) / MT_M) * ((N + MT_N - 1) / MT_N) * batch;

  std::size_t max_hw_split = std::floor(hardware.N_CU / grid);
  std::size_t MAX_SPLIT    = std::min(biggest_allowable_split, max_hw_split);

  std::size_t best_split = 1;
  double best_latency    = std::numeric_limits<double>::infinity();

  for (size_t split = 1; split <= MAX_SPLIT; ++split) {
    double latency = compute_total_latency(hardware, problem, config, split);

    if (latency < best_latency) {
      best_latency = latency;
      best_split   = split;
    }
  }
  size_t best_grid = best_split * grid;

  // you now have both `grid` and `best_split`—
  // return whichever is appropriate (here we stick with split):
  return best_grid;
}

std::size_t grid_k_split_aware(const problem_t& problem,
                               const hardware_t& hardware,
                               const config_t& config) {
  std::size_t cu_count = hardware.N_CU;
  std::size_t tiles    = number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);

  std::size_t sk_grid = tiles;  // Fallback if no good fractional tile is found

  const std::size_t iters_per_tile = num_iters_per_tile(problem.size.k, config.mt.k);

  const std::size_t tile_size = config.mt.m * config.mt.n * config.workspace_size_per_elem_c;

  // More tiles than CUs
  // Distribute tiles evenly across maximum number of CUs
  // Split remaining tiles as evenly as possible for better caching
  if (tiles > cu_count) {
    std::size_t virt_cu_count = cu_count;
    if (config.occupancy > 1) virt_cu_count *= config.occupancy;

    const std::vector<double> tile_fractions = {
        0.0, 1.0 / 2.0, 1.0 / 8.0, 1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0};
    const std::size_t min_even_tiles = tiles / virt_cu_count;

    for (double frac : tile_fractions) {
      const std::size_t frac_grid =
          static_cast<std::size_t>((tiles / (min_even_tiles + frac)) + 0.5);

      // Check if higher occupancy would cause excessive workspace requirements (set current limit
      // to 128MB)
      if ((tiles % frac_grid != 0) && (tile_size * frac_grid > 128ull * 1024ull * 1024ull))
        continue;

      if (frac_grid <= virt_cu_count) {
        sk_grid = frac_grid;
        break;
      }
    }

    // Fewer tiles than CUs
    // Split tiles evenly in k-dimension
    // Attempt to maximize CU utilization, up to a peak number of splits
    // Max splitting is currently constant, but should be dependant on K dimension
  } else if (tiles < cu_count) {
    // For problems with large k and low number of tiles, use parallel reduction
    // TODO Benchmark to check if limits are correct
    // constexpr int MinItersForParallel = 64;
    // constexpr int MaxTilesForParallel = 16;
    constexpr int MinItersPerCU = 8;

    if (config.reduction_strategy == reduction_t::Parallel) {
      std::size_t virt_cu_count = cu_count;
      // TODO check if using occupancy info makes workspace too large
      // if (occupancy > 1)
      //     virt_cu_count *= occupancy;

      // Find max splitting factor to use as much of GPU as possible
      const std::size_t maxSplitsForTiles = virt_cu_count / tiles;

      // Find max splitting factor to ensure each CU has a minimum number of iterations to do
      const std::size_t maxSplitsForIters = iters_per_tile / MinItersPerCU;

      const std::size_t maxSplits = std::min(maxSplitsForTiles, maxSplitsForIters);
      sk_grid                     = tiles * maxSplits;
    } else {
      const std::vector<std::size_t> tile_fractions = {16, 12, 8, 6, 4, 3, 2, 1};
      for (std::size_t frac : tile_fractions) {
        const std::size_t splitGrid  = tiles * frac;
        const std::size_t itersPerCU = iters_per_tile / frac;
        if (splitGrid <= cu_count && itersPerCU >= MinItersPerCU) {
          sk_grid = splitGrid;
          break;
        }
      }
    }
  }

  if (tiles % sk_grid != 0 && tile_size * sk_grid > config.workspace_size) sk_grid = tiles;

  return sk_grid;
}

}  // namespace streamk
}  // namespace origami
