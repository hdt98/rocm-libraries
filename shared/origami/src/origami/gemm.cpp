// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

#include "origami/hardware.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"

#include "origami/gemm.hpp"
#include "origami/streamk.hpp"

namespace origami {

// Computes the number of active compute units if there is only one wave and it is partial
// Otherwise, returns hardware.N_CU
std::tuple<size_t, size_t, size_t> compute_cu_occupancy(const problem_t& problem,
                                                        const hardware_t& hardware,
                                                        const config_t& config,
                                                        grid_selection_t grid_selection) {
  // Number of output MTs
  std::size_t num_mts = streamk::compute_number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);

  reduction_t rt = streamk::select_reduction(problem, hardware, config, grid_selection);

  // For now, restrict the grid-size algorithm to k_split_aware.
  std::size_t num_wgs =
      streamk::select_grid_size(problem, hardware, config, grid_selection_t::k_split_aware);

  // output variables
  std::size_t num_active_cus = num_wgs < hardware.N_CU ? num_wgs : hardware.N_CU;
  // There are cases in which StreamK combines multiple output MTs and assigns to 1 WG.
  // That means, we artifically observe one full wave, but that is not what actually happens
  // under the hood. From a theoretical point of view, these distributions change all of the
  // computations in Origami. With current implementation, it is hard to capture that behaviour
  // analytically.
  // So for now, if the num_wgs is less than the numMTs, we calculate num_waves based on the
  // numMTs. Otherwise, we use num_wgs to compute num_waves.
  std::size_t num_waves    = num_wgs > num_mts ? math::safe_ceil_div(num_wgs, hardware.N_CU)
                                               : math::safe_ceil_div(num_mts, hardware.N_CU);
  std::size_t split_factor = math::safe_ceil_div(num_wgs, num_mts);

  if (hardware_t::is_debug_enabled()) {
    hardware.log_debug("num_macro_tiles", num_mts);
    hardware.log_debug("num_workgroups", num_wgs);
    hardware.log_debug("num_active_cus", num_active_cus);
    hardware.log_debug("num_waves", num_waves);
    hardware.log_debug("split_factor", split_factor);
  }

  return std::make_tuple(num_active_cus, num_waves, split_factor);
}

// Work (M×N×K) utilization over the launched macro-tile grid and K loop.
inline double calculate_work_utilization(const problem_t& problem, const config_t& config) {
  const std::size_t M = problem.size.m;
  const std::size_t N = problem.size.n;
  const std::size_t K = problem.size.k;

  const std::size_t MT_M = config.mt.m;
  const std::size_t MT_N = config.mt.n;
  const std::size_t MT_K = config.mt.k;

  if (M == 0 || N == 0 || K == 0 || MT_M == 0 || MT_N == 0 || MT_K == 0) { return 1.0; }

  const double launched_M =
      static_cast<double>(math::safe_ceil_div(M, MT_M)) * static_cast<double>(MT_M);
  const double launched_N =
      static_cast<double>(math::safe_ceil_div(N, MT_N)) * static_cast<double>(MT_N);
  const double launched_K =
      static_cast<double>(math::safe_ceil_div(K, MT_K)) * static_cast<double>(MT_K);

  const double useful_volume   = static_cast<double>(M) * N * K;
  const double launched_volume = launched_M * launched_N * launched_K;

  if (launched_volume < 1.0) return 1.0;  // guard tiny cases
  return useful_volume / launched_volume;
}

// Output (M×N) utilization with optional vectorization alignment.
inline double calculate_output_utilization(const problem_t& problem,
                                           const config_t& config,
                                           std::size_t vector_elems = 1) {
  const std::size_t M = problem.size.m;
  const std::size_t N = problem.size.n;

  const std::size_t MT_M = config.mt.m;
  const std::size_t MT_N = config.mt.n;

  if (M == 0 || N == 0 || MT_M == 0 || MT_N == 0) { return 1.0; }

  const double launched_M =
      static_cast<double>(math::safe_ceil_div(M, MT_M)) * static_cast<double>(MT_M);
  const double launched_N =
      static_cast<double>(math::safe_ceil_div(N, MT_N)) * static_cast<double>(MT_N);

  // Model vector store/load width: tails scalarize.
  const std::size_t M_vec = (vector_elems > 1) ? (M / vector_elems) * vector_elems : M;
  const std::size_t N_vec = (vector_elems > 1) ? (N / vector_elems) * vector_elems : N;

  const double useful   = static_cast<double>(M_vec) * static_cast<double>(N_vec);
  const double launched = launched_M * launched_N;

  if (launched < 1.0) return 1.0;
  return useful / launched;
}

/* ---------------------------------------------------------------------------------------- */
/* Compute-related functions                                                                */
/* ---------------------------------------------------------------------------------------- */
std::size_t compute_number_matrix_instructions(dim3_t mt, dim3_t mi) {
  // Compute the number of Matrix Instructions required in each dim.
  std::size_t num_m_instrs = math::safe_ceil_div(mt.m, mi.m);
  std::size_t num_n_instrs = math::safe_ceil_div(mt.n, mi.n);
  std::size_t num_k_instrs = math::safe_ceil_div(mt.k, mi.k);

  // Total number of matrix instructions.
  std::size_t num_matrix_instrs = num_m_instrs * num_n_instrs * num_k_instrs;

  return num_matrix_instrs;
}

// Compute arithmic intensity
double arithmetic_intensity(double m, double n, double k, double bytes_per_element) {
  // Numerator: 2.0 * m * n * k
  // Denominator: (m*n + n*k + m*k) * bytes_per_element
  double numerator   = 2.0 * m * n * k;
  double denominator = (m * n + n * k + m * k) * bytes_per_element;

  return numerator / denominator;
}

// Compute cvt overhead in tf32 emulation
double compute_cvt_overhead(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config) {
  // Wave tile sizes
  // TODO: Use kernel's actual wavetiles.
  const double wave_tile_m = config.mt.m / 2.0;
  const double wave_tile_n = config.mt.n / 2.0;
  const double wave_tile_k = config.mt.k / config.mi.k;

  // MFMA count and cycles
  const double N_MI = (wave_tile_m / config.mi.m) * (wave_tile_n / config.mi.n) * wave_tile_k;

  // TF32 emu: 3× BF16 MI issue slots
  const double num_mfma = 3.0 * static_cast<double>(N_MI);

  // Cycle scale per MI (use BF16 MI latency as the basic timing quantum)
  const double L_MI_bf16 =
      hardware.get_mi_latency(config.mi.m, config.mi.n, config.mi.k, data_type_t::BFloat16);
  const double mfma_cycles = num_mfma * L_MI_bf16;

  // 2) Bytes (per K-slice), using ceil-div to whole bytes
  int a_bytes = data_type_to_bytes(problem.a_dtype);
  int b_bytes = data_type_to_bytes(problem.b_dtype);

  const double bytesA = static_cast<double>(wave_tile_m) * config.mt.k * a_bytes;
  const double bytesB = static_cast<double>(wave_tile_n) * config.mt.k * b_bytes;

  const double mt_bytesA = static_cast<double>(config.mt.mk()) * a_bytes;

  // 3) Modeled transfer quanta (128B lines)
  //      dsA = bytesA / (128 * MI_M)
  //      dsB = bytesB / (128 * MI_N)
  //      GR  = dsA  (global->LDS modeled equal to A-side DS)
  const double dsA = (bytesA / 128.0) / static_cast<double>(config.mi.m);  // LDS->VGPR for A
  const double dsB = (bytesB / 128.0) / static_cast<double>(config.mi.n);  // LDS->VGPR for B
  const double GR  = dsA;                                                  // Global->LDS reads
  const double LR  = dsA + dsB;                                            // total DS->VGPR

  // 4) Heuristic cycle weights (scaled to MI latency).
  //    Preserves your A=104, B=8, C=4 when L_MI_bf16 == 16.
  // 24 vector instructions per 2 ds_reads (16x16x32)
  // 24 vector instructions per 2 ds_reads for A and for B.
  // 3 instructions per fp32 value read; number ds_read * size
  const double A = (104.0 / 16.0) * L_MI_bf16;  // CVT per LR-sized chunk (DS->VGPR)
  const double B = (8.0 / 16.0) * L_MI_bf16;    // hidden per spare MFMA slot
  // MI16: 16 - 4 (12 cycles), for those 4 cycles, VGPRs are locked. 8 cycles to do anything.
  const double C = (4.0 / 16.0) * L_MI_bf16;  // hidden per (LR+GR) slot     // MI16
  // 32 cycles (mfma), 4 cycles, 28, 4 vgpr lock, 24 cycles left.
  // 24: 6 conv instructions, 3 ds_reads, ~6 grs

  // 5) Exposed vs hidden CVT
  const double spare_mfma = std::max(0.0, num_mfma - LR - GR);
  const double cvt        = A * dsA;                         // only DS->VGPR contributes CVT
  const double H          = B * spare_mfma + C * (LR + GR);  // hidden cycles
  const double overhead   = std::max(cvt - H, 0.0);

  // 6) Efficiency
  const double denom = mfma_cycles + overhead;
  const double eff   = (denom > 0.0) ? (mfma_cycles / denom) : 1;

  return overhead;
}

// Compute cvt overhead in x1 TF32 emulation (SS_BSS path on gfx950, etc.)
inline double compute_cvt_overhead_x1(const problem_t& problem,
                                      const hardware_t& hardware,
                                      const config_t& config) {
  // --- Shorthands -----------------------------------------------------------
  const double MT_M = static_cast<double>(config.mt.m);
  const double MT_N = static_cast<double>(config.mt.n);
  const double MT_K = static_cast<double>(config.mt.k);

  const double MI_M = static_cast<double>(config.mi.m);
  const double MI_N = static_cast<double>(config.mi.n);
  const double MI_K = static_cast<double>(config.mi.k);

  // Guard against invalid MI tile
  if (MI_M <= 0.0 || MI_N <= 0.0 || MI_K <= 0.0) return 0.0;

  const int a_bytes = data_type_to_bytes(problem.a_dtype);
  const int b_bytes = data_type_to_bytes(problem.b_dtype);

  // In X1 TF32 GEMMs, we do two v_cvt_pk_bf16_f32 per wave tile, plus ds_write_b64.
  // These vector ops can overlap with MFMA/memory, so we model exposed overhead.

  // --- 1) Wave tile decomposition --------------------------------------------
  const double wave_tile_m = MT_M / 2.0;
  const double wave_tile_n = MT_N / 2.0;
  const double wave_tile_k = MT_K / MI_K;

  // --- MFMA count & cycles --------------------------------------------------
  const double N_MI     = (wave_tile_m / MI_M) * (wave_tile_n / MI_N) * wave_tile_k;
  const double num_mfma = N_MI;
  const double L_MI     = hardware.get_mi_latency(
      static_cast<int>(MI_M), static_cast<int>(MI_N), static_cast<int>(MI_K), problem.mi_dtype);
  const double mfma_cycles = num_mfma * L_MI;  // (not directly used, kept for clarity)

  // --- 2) Bytes (per K-slice) ----------------------------------------------
  const double bytesA = wave_tile_m * MT_K * static_cast<double>(a_bytes);
  const double bytesB = wave_tile_n * MT_K * static_cast<double>(b_bytes);

  // --- 3) Modeled transfer quanta (128B lines) ------------------------------
  // dsA = (bytesA / 128) / MI_M, dsB = (bytesB / 128) / MI_N
  // GR  = dsA (global->LDS modeled equal to A-side DS), LR = dsA + dsB
  const double dsA = (bytesA / 128.0) / MI_M;  // LDS->VGPR for A
  const double dsB = (bytesB / 128.0) / MI_N;  // LDS->VGPR for B
  const double GR  = dsA;                      // Global->LDS reads
  const double LR  = dsA + dsB;                // total DS->VGPR

  // --- 5) Exposed vs hidden CVT --------------------------------------------
  // Spare MFMA "slots" that can hide vector work:
  const double spare_mfma = std::max(0.0, num_mfma - LR - GR);

  // Two cvt per ds_write (SS_BSS). Each cvt latency ~4 cycles.
  // Scaled by MI latency per original heuristic:
  //   cvt = (2 * 4 / 16 * L_MI) * LR
  const double cvt = (2.0 * 4.0 / 16.0 * L_MI) * LR;

  // Heuristic hidden portion:
  //   H = (8/16 * L_MI) * spare_mfma + (4/16 * L_MI) * (LR + GR)
  const double H = (8.0 / 16.0 * L_MI) * spare_mfma + (4.0 / 16.0 * L_MI) * (LR + GR);

  const double overhead = std::max(cvt - H, 0.0);
  return overhead;
}

double compute_mt_compute_latency(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config) {
  // Compute the number of matrix instructions
  std::size_t N_MI = compute_number_matrix_instructions(config.mt, config.mi);
  // Latency of a single MT_M x MT_N x MT_K tile is the latency of one
  // MI multiplied by number of MI per MT_M x MT_N x MT_K.
  std::size_t L_MI =
      hardware.get_mi_latency(config.mi.m, config.mi.n, config.mi.k, problem.mi_dtype);
  std::size_t L_MT = L_MI * N_MI;

  int a_bytes = data_type_to_bytes(problem.a_dtype);
  int b_bytes = data_type_to_bytes(problem.b_dtype);

  return L_MT;
}

/* ---------------------------------------------------------------------------------------- */
/* Memory-related functions                                                                 */
/* ---------------------------------------------------------------------------------------- */

bool check_lds_capacity(const hardware_t& hardware,
                        dim3_t mt,
                        data_type_t a_dtype,
                        data_type_t b_dtype) {
  // A and B loads in bytes.
  std::size_t a_loads_in_bytes = mt.mk() * data_type_to_bytes(a_dtype);
  std::size_t b_loads_in_bytes = mt.nk() * data_type_to_bytes(b_dtype);

  // Total A/B loads in bytes is LDS usage.
  std::size_t lds_usage = a_loads_in_bytes + b_loads_in_bytes;

  if (lds_usage > hardware.lds_capacity) {
    return false;  // Exceeded available LDS capacity.
  } else {
    return true;  // Within available LDS capacity.
  }
}

// Compute limited achievable memory bandwidth based on active CUs
double compute_mem_bw_from_occupancy(const hardware_t& hardware, size_t num_active_cus) {
  const double CUs = static_cast<double>(num_active_cus);

  if (num_active_cus > hardware.N_CU) return 1.0;

  const double bw_limited = std::get<0>(hardware.mem_bw_per_wg_coefficients) * CUs * CUs +
                            std::get<1>(hardware.mem_bw_per_wg_coefficients) * CUs +
                            std::get<2>(hardware.mem_bw_per_wg_coefficients);

  return std::min(bw_limited, 1.0);
}

double estimate_l2_hit(const problem_t& problem,
                       const hardware_t& hardware,
                       const config_t& config,
                       std::size_t splitting_factor) {
  // Compute grid dimensions
  std::size_t grid_m                = math::safe_ceil_div(problem.size.m, config.mt.m);
  std::size_t grid_n                = math::safe_ceil_div(problem.size.n, config.mt.n);
  std::size_t total_workgroups      = grid_m * grid_n;
  std::size_t concurrent_workgroups = std::min(total_workgroups, hardware.N_CU);

  // Distribute CUs per XCD
  // Modify cu_per_xcd to only take into account the CUs that might share same K-tiles
  // This is to factor in the effect of splitting on L2
  std::size_t effective_cus = math::safe_ceil_div(concurrent_workgroups, splitting_factor);
  std::size_t cu_per_xcd =
      std::max(math::safe_ceil_div(effective_cus, hardware.NUM_XCD), static_cast<size_t>(1));

  // N dimension of mem1 tile is divided by whichever is smaller between WGM and grid
  std::size_t l2_n = std::min(std::size_t(config.workgroup_mapping), grid_n);
  std::size_t l2_m = math::safe_ceil_div(cu_per_xcd, l2_n);

  // If a single mem1 tile is larger than the grid, extend M dimension
  if (l2_m > grid_m) {
    int num_wraps = (l2_m / grid_m) - 1;  // how many times we wrap
    l2_n += (num_wraps * config.workgroup_mapping);
    l2_m = grid_m;
  }

  // Clamp mem1 tile dimensions to at least 1 and at most grid size
  l2_m = std::max(std::min(grid_m, l2_m), std::size_t(1));
  l2_n = std::max(std::min(grid_n, l2_n), std::size_t(1));

  // Compute "uncached" reads based on mem1 tile dimensions
  long long l2_A_uncached_reads = static_cast<long long>(l2_m) * config.mt.mk();
  long long l2_B_uncached_reads = static_cast<long long>(l2_n) * config.mt.nk();
  long long uncached_read       = l2_A_uncached_reads + l2_B_uncached_reads;

  // If bigger than cache capacity, reduce mem1 tile size and recompute uncached reads
  while (l2_A_uncached_reads + l2_B_uncached_reads >
         hardware.L2_capacity / data_type_to_bytes(problem.a_dtype)) {
    // Reduce M dimension by 1
    l2_m -= 1;
    if (l2_m < 1) {
      // We cannot shrink any more without going to zero or negative
      l2_m = 1;
      break;
    }

    l2_A_uncached_reads = static_cast<long long>(l2_m) * config.mt.mk();
    l2_B_uncached_reads = static_cast<long long>(l2_n) * config.mt.nk();
  }

  l2_A_uncached_reads = static_cast<long long>(l2_m) * config.mt.mk();
  l2_B_uncached_reads = static_cast<long long>(l2_n) * config.mt.nk();

  // Total reads considering repeated usage
  long long l2_A_reads = static_cast<long long>(l2_m) * l2_n * config.mt.mk();
  long long l2_B_reads = static_cast<long long>(l2_n) * l2_m * config.mt.nk();

  long long total_reads         = std::max(l2_A_reads + l2_B_reads, 1LL);
  long long total_uncached_read = l2_A_uncached_reads + l2_B_uncached_reads;
  long long cached_reads        = total_reads - total_uncached_read;

  double l2_hit = static_cast<double>(cached_reads) / static_cast<double>(total_reads);

  // Guard against numeric anomalies
  if (l2_hit > 1.0) {
    std::cerr << "mem1 hit was greater than 1, which isn't possible.\n"
              << "Problem Size: " << problem.size.m << "x" << problem.size.n << "x"
              << problem.size.k << "\n"
              << "Macro-Tile:  " << config.mt.m << "x" << config.mt.n << "x" << config.mt.k << "\n"
              << "cu_per_xcd:  " << cu_per_xcd << "\n"
              << "l2_m: " << l2_m << ", l2_n: " << l2_n << ", l2_hit: " << l2_hit << "\n";
  }

  if (hardware_t::is_debug_enabled()) {
    hardware.log_debug("l2_tile_m", l2_m);
    hardware.log_debug("l2_tile_n", l2_n);
  }

  return l2_hit;
}

double estimate_mall_hit(const problem_t& problem,
                         const hardware_t& hardware,
                         const config_t& config,
                         std::size_t num_active_cus,
                         std::size_t splitting_factor) {
  // Compute grid dimensions
  std::size_t workgroups_m = math::safe_ceil_div(problem.size.m, config.mt.m);
  std::size_t workgroups_n = math::safe_ceil_div(problem.size.n, config.mt.n);

  // Tile dimensions in MALL.
  size_t mall_tile_m =
      math::safe_ceil_div(num_active_cus, static_cast<size_t>(config.workgroup_mapping));
  std::size_t mall_tile_n =
      std::min(std::size_t(config.workgroup_mapping), workgroups_n);  // N dimension of mem2 tile

  // If a single mem2 tile is larger than the grid, extend its M dimension
  if (mall_tile_m > workgroups_m) {
    int num_wraps = (mall_tile_m / workgroups_m) - 1;
    mall_tile_n += (num_wraps * config.workgroup_mapping);
    mall_tile_m = workgroups_m;
  }

  // Clamp the tile dimensions to valid ranges
  mall_tile_m = std::max(std::min(workgroups_m, mall_tile_m), std::size_t(1));
  mall_tile_n = std::max(std::min(workgroups_n, mall_tile_n), std::size_t(1));

  // Unique “uncached” entries of A/B for this XCD
  int mall_A_uncached_reads = mall_tile_m * config.mt.mk();
  int mall_B_uncached_reads = mall_tile_n * config.mt.nk();
  int total_uncached_read   = mall_A_uncached_reads + mall_B_uncached_reads;

  // Total A/B reads considering repeated usage
  long long mall_A_reads = static_cast<long long>(mall_tile_m) * mall_tile_n * config.mt.mk();
  long long mall_B_reads = static_cast<long long>(mall_tile_n) * mall_tile_m * config.mt.nk();

  // Avoid division by zero
  long long total_reads  = std::max(mall_A_reads + mall_B_reads, 1LL);
  long long cached_reads = total_reads - total_uncached_read;

  double mall_hit = static_cast<double>(cached_reads) / static_cast<double>(total_reads);

  if (hardware_t::is_debug_enabled()) {
    hardware.log_debug("mall_tile_m", mall_tile_m);
    hardware.log_debug("mall_tile_n", mall_tile_n);
  }

  return mall_hit;
}

/**
 * @brief L2 hit rate from a global (problem-wide) perspective using the refactored API.
 * Computes in BYTES to correctly handle differing A/B dtypes.
 */
double compute_l2_hit_rate_global(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config) {
  const std::size_t l2_capacity_bytes = static_cast<std::size_t>(hardware.L2_capacity) * 1024ull;
  if (l2_capacity_bytes == 0) { throw std::runtime_error("L2 Capacity is zero"); }

  const std::size_t M    = problem.size.m;
  const std::size_t N    = problem.size.n;
  const std::size_t MT_M = config.mt.m;
  const std::size_t MT_N = config.mt.n;
  const std::size_t MT_K = config.mt.k;

  const std::size_t grid_m = math::safe_ceil_div(M, MT_M);
  const std::size_t grid_n = math::safe_ceil_div(N, MT_N);
  if (grid_m == 0 || grid_n == 0) {
    throw std::runtime_error("compute_l2_hit_rate_global: grid dims cannot be zero");
  }

  const std::size_t a_bytes = static_cast<std::size_t>(data_type_to_bytes(problem.a_dtype));
  const std::size_t b_bytes = static_cast<std::size_t>(data_type_to_bytes(problem.b_dtype));

  const double a_working_set_bytes =
      static_cast<double>(grid_m) * MT_M * MT_K * static_cast<double>(a_bytes);
  const double b_working_set_bytes =
      static_cast<double>(grid_n) * MT_N * MT_K * static_cast<double>(b_bytes);
  const double total_working_set_bytes = a_working_set_bytes + b_working_set_bytes;

  // Capacity check: if it doesn't fit, global reuse breaks → very low hit rate.
  if (total_working_set_bytes > static_cast<double>(l2_capacity_bytes)) {
    return 0.1;  // floor (tunable)
  }

  // Idealized hit rate assuming capacity fits (compute in BYTES)
  //    Total bytes if nothing were cached across the full grid sweep:
  const double total_A_reads_elems = static_cast<double>(grid_m) * grid_n * MT_M * MT_K;
  const double total_B_reads_elems = static_cast<double>(grid_m) * grid_n * MT_N * MT_K;

  const double total_A_reads_bytes = total_A_reads_elems * static_cast<double>(a_bytes);
  const double total_B_reads_bytes = total_B_reads_elems * static_cast<double>(b_bytes);

  //    First-touch (uncached) bytes: one full column for A, one full row for B.
  const double uncached_A_reads_bytes = static_cast<double>(grid_m) * MT_M * MT_K * a_bytes;
  const double uncached_B_reads_bytes = static_cast<double>(grid_n) * MT_N * MT_K * b_bytes;

  const double total_reads_bytes = total_A_reads_bytes + total_B_reads_bytes;
  if (total_reads_bytes == 0.0) {
    return 1.0;  // No reads → perfect hit rate
  }

  const double cached_reads_bytes = (total_A_reads_bytes - uncached_A_reads_bytes) +
                                    (total_B_reads_bytes - uncached_B_reads_bytes);

  return cached_reads_bytes / total_reads_bytes;
}

double compute_memory_latency(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              std::size_t num_active_cus,
                              std::size_t splitting_factor) {
  // ---- helpers -------------------------------------------------------------
  const auto a_bytes = data_type_to_bytes(problem.a_dtype);
  const auto b_bytes = data_type_to_bytes(problem.b_dtype);
  const auto a_bits  = data_type_to_bits(problem.a_dtype);
  const auto b_bits  = data_type_to_bits(problem.b_dtype);

  const bool a_trans = (problem.a_transpose == transpose_t::T);
  const bool b_trans = (problem.b_transpose == transpose_t::T);

  const std::size_t MT_M = config.mt.m;
  const std::size_t MT_N = config.mt.n;
  const std::size_t MT_K = config.mt.k;

  // round elements so the contiguous dimension is a multiple of 128B
  auto round_elems_to_128B = [](std::size_t elems, int bytes_per_elem) -> std::size_t {
    const std::size_t bytes         = elems * static_cast<std::size_t>(bytes_per_elem);
    const std::size_t rounded_bytes = (bytes + 127) / 128 * 128;
    return rounded_bytes / static_cast<std::size_t>(bytes_per_elem);
  };

  // Estimate per-iteration L2 hit rate
  double H_mem1 = estimate_l2_hit(problem, hardware, config, splitting_factor);

  // Global cap on L2 hit-rate (prevents impossible cache residency claims)
  // (Assumes capacity is given in KiB, convert to bytes)
  const std::size_t l2_bytes = static_cast<std::size_t>(hardware.L2_capacity) * 1024ull;
  double H_mem1_global       = compute_l2_hit_rate_global(problem, hardware, config);
  H_mem1                     = std::min(H_mem1, H_mem1_global);
  if (H_mem1 == 0.0) { H_mem1 = 0.5; }

  // Estimate LLC hit rate
  double H_mem2 = estimate_mall_hit(problem, hardware, config, num_active_cus, splitting_factor);

  // Round loads to 128 Bytes
  std::size_t MT_M_rA = round_elems_to_128B(MT_M, a_bytes);
  std::size_t MT_N_rA = round_elems_to_128B(MT_N, a_bytes);
  std::size_t MT_K_rA = round_elems_to_128B(MT_K, a_bytes);

  std::size_t MT_M_rB = round_elems_to_128B(MT_M, b_bytes);
  std::size_t MT_N_rB = round_elems_to_128B(MT_N, b_bytes);
  std::size_t MT_K_rB = round_elems_to_128B(MT_K, b_bytes);

  // Match original conditional "unrounding" (which depends on which dims are contiguous):
  // - NN: A(M contiguous), B(K contiguous)  -> don't round N for A and K for A; for B rely on B's
  // dims
  // - TN: A(K contiguous), B(K contiguous)  -> don't round M and N for A
  // - NT: A(M contiguous), B(N contiguous)  -> don't round K for B
  // - TT: A(K contiguous), B(N contiguous)  -> handled by leaving the other dims rounded
  if (!a_trans && !b_trans) {        // NN
    MT_K_rA = MT_K;                  // K is not contiguous in A tensor
    MT_N_rB = MT_N;                  // N is not contiguous in NN
  } else if (a_trans && !b_trans) {  // TN
    MT_M_rA = MT_M;                  // M is not contiguous in A tensor in TN
    MT_N_rB = MT_N;                  // N is not contiguous in B tensor in TN
  } else if (!a_trans && b_trans) {  // NT
    MT_K_rA = MT_K;                  // K is not contiguous in A tensor in NT
    MT_K_rB = MT_K;                  // K is not contiguous in B tensor in NT
  } else if (a_trans && b_trans) {   // TT
    MT_M_rA = MT_M;                  // M is not contiguous in A in TT
    MT_N_rA = MT_N;                  // K is not contiguous in B in TT
  }

  // Compute elements loaded for each operand per tile
  const std::size_t Ld_A_elems = (a_trans ? (MT_K_rA * MT_M_rA) : (MT_M_rA * MT_K_rA));
  const std::size_t Ld_B_elems = (b_trans ? (MT_N_rB * MT_K_rB) : (MT_K_rB * MT_N_rB));

  // Convert to bytes
  std::size_t Ld_CU_bytes = Ld_A_elems * static_cast<std::size_t>(a_bytes) +
                            Ld_B_elems * static_cast<std::size_t>(b_bytes);

  // Block-scaled data (assume 8-bit scales, 1 byte per scale)
  if (a_bits < 8 && problem.a_mx_block_size != 0) {
    const std::size_t num_scales_A = math::safe_ceil_div(
        std::size_t{MT_M} * MT_K, static_cast<std::size_t>(problem.a_mx_block_size));
    Ld_CU_bytes += num_scales_A;
  }
  if (b_bits < 8 && problem.b_mx_block_size != 0) {
    const std::size_t num_scales_B = math::safe_ceil_div(
        std::size_t{MT_N} * MT_K, static_cast<std::size_t>(problem.b_mx_block_size));
    Ld_CU_bytes += num_scales_B;
  }

  // ---- Aggregate across active CUs --------------------------------------
  const double total_Ld = static_cast<double>(Ld_CU_bytes) * static_cast<double>(num_active_cus);

  // ---- mem1 bandwidth limit based on occupancy (simple linear model) -----------------------
  const double mem1_bw_limited =
      static_cast<double>(num_active_cus) / static_cast<double>(hardware.N_CU);
  const double limited_mem1_bw = hardware.mem1_perf_ratio * mem1_bw_limited;

  // ---- mem1 latency -----------------------------------------------------
  const double L_mem_mem1 = (limited_mem1_bw > 0.0) ? (total_Ld / limited_mem1_bw) : 0.0;

  // ---- mem2 limit from occupancy ---------------------------------------
  const double bw_limited = compute_mem_bw_from_occupancy(hardware, num_active_cus);

  // ----  Loads that reach each level --------------------------------------
  double Ld_mem2 = (1.0 - H_mem1) * total_Ld;
  double Ld_MEM  = (1.0 - H_mem2) * Ld_mem2;

  // ----  Enforce whole-problem minimum loads when we can fit M/N in the CUs
  // Replicates the “mall tile” min-load logic from the original version.
  const std::size_t M = problem.size.m;
  const std::size_t N = problem.size.n;
  const std::size_t K = problem.size.k;
  const std::size_t B = problem.batch;

  // Number of workgroup tiles across M and N
  const std::size_t grid_m = math::safe_ceil_div(M, MT_M);
  const std::size_t grid_n = math::safe_ceil_div(N, MT_N);

  // MALL tile dimensions
  std::size_t mall_m =
      math::safe_ceil_div(num_active_cus, static_cast<std::size_t>(config.workgroup_mapping));
  std::size_t mall_n = std::min(static_cast<std::size_t>(config.workgroup_mapping), grid_n);

  // Handle wrap-around case (when mall_m exceeds grid_m)
  if (mall_m > grid_m) {
    const std::size_t num_wraps = mall_m / grid_m;
    mall_n += (num_wraps * static_cast<std::size_t>(config.workgroup_mapping));
    mall_m = grid_m;
  }

  // Clamp to valid ranges
  mall_m = std::max<std::size_t>(1, std::min(grid_m, mall_m));
  mall_n = std::max<std::size_t>(1, std::min(grid_n, mall_n));

  // Minimum unique bytes needed from main memory to feed concurrent WGs (per batch)
  const double min_load_bytes =
      static_cast<double>((mall_m * MT_M * K * static_cast<std::size_t>(a_bytes)) +  // A
                          (mall_n * MT_N * K * static_cast<std::size_t>(b_bytes))    // B
      );

  // Apply batching to the minimum itself and clamp both levels
  const double min_load_total = min_load_bytes * static_cast<double>(B);
  Ld_MEM                      = std::max(Ld_MEM, min_load_total);
  Ld_mem2                     = std::max(Ld_mem2, min_load_total);

  // ----  mem2 latency ----------------------------------------------------
  const double limited_mem2_bw = hardware.mem2_perf_ratio * bw_limited;
  const double L_mem_mem2      = (limited_mem2_bw > 0.0) ? (Ld_mem2 / limited_mem2_bw) : 0.0;

  // ----  Main memory latency --------------------------------------------
  const double limited_mem_bw = hardware.mem3_perf_ratio * bw_limited;
  double L_mem_MEM            = (limited_mem_bw > 0.0) ? (Ld_MEM / limited_mem_bw) : 0.0;
  L_mem_MEM += 200.0;  // Fixed load latency

  // ---- Worst-case bound ------------------------------------------------
  double L_mem = std::max({L_mem_mem1, L_mem_mem2, L_mem_MEM});

  if (hardware_t::is_debug_enabled()) {
    hardware.log_debug("mem1_perf_ratio", hardware.mem1_perf_ratio);
    hardware.log_debug("mem2_perf_ratio", hardware.mem2_perf_ratio);
    hardware.log_debug("mem3_perf_ratio", hardware.mem3_perf_ratio);
    hardware.log_debug("mem_bw_per_wg_coeff_0", std::get<0>(hardware.mem_bw_per_wg_coefficients));
    hardware.log_debug("mem_bw_per_wg_coeff_1", std::get<1>(hardware.mem_bw_per_wg_coefficients));
    hardware.log_debug("mem_bw_per_wg_coeff_2", std::get<2>(hardware.mem_bw_per_wg_coefficients));
    hardware.log_debug("mem1_hit_ratio", H_mem1);
    hardware.log_debug("mem2_hit_ratio", H_mem2);
    hardware.log_debug("total_load_bytes", total_Ld);
    hardware.log_debug("load_mem2_bytes", Ld_mem2);
    hardware.log_debug("load_main_mem_bytes", Ld_MEM);
    hardware.log_debug("latency_mem1_cycles", L_mem_mem1);
    hardware.log_debug("latency_mem2_cycles", L_mem_mem2);
    hardware.log_debug("latency_main_mem_cycles", L_mem_MEM);
    hardware.log_debug("MT_M%128B_A", (config.mt.m * a_bytes) % 128);
    hardware.log_debug("MT_N%128B_B", (config.mt.n * b_bytes) % 128);
    hardware.log_debug("MT_K%128B_A", (config.mt.k * a_bytes) % 128);
    hardware.log_debug("MT_K%128B_B", (config.mt.k * b_bytes) % 128);
    hardware.log_debug("tile_arith_intensity",
                       (double)(MT_M) * (double)(MT_N) * (double)(MT_K) /
                           ((double)(MT_M) * (double)(MT_K) + (double)(MT_N) * (double)(MT_K)));
  }

  return L_mem;
}
double compute_tile_latency(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config,
                            std::size_t num_active_cus,
                            std::size_t splitting_factor) {
  // --- Per-tile latencies (compute & memory) -----------------------------
  const double L_compute = compute_mt_compute_latency(problem, hardware, config);
  const double L_mem =
      compute_memory_latency(problem, hardware, config, num_active_cus, splitting_factor);

  // --- Common shorthands ----------------------------------------------------
  const std::size_t M = problem.size.m;
  const std::size_t N = problem.size.n;
  const std::size_t K = problem.size.k;
  const std::size_t B = problem.batch;

  const std::size_t MT_M = config.mt.m;
  const std::size_t MT_N = config.mt.n;
  const std::size_t MT_K = config.mt.k;

  const std::size_t a_bytes = static_cast<std::size_t>(data_type_to_bytes(problem.a_dtype));
  const std::size_t d_bytes = static_cast<std::size_t>(data_type_to_bytes(problem.d_dtype));

  auto round_elems_to_128B = [](std::size_t elems, std::size_t bytes_per_elem) -> std::size_t {
    const std::size_t bytes   = elems * bytes_per_elem;
    const std::size_t rounded = ((bytes + 127) / 128) * 128;
    return rounded / bytes_per_elem;
  };

  // --- Utilization (work + output) -----------------------------------------
  // Matches original behavior: penalties applied to both compute & memory bounds.
  const double utilization        = calculate_work_utilization(problem, config);
  const double output_utilization = calculate_output_utilization(problem, config, 1UL);

  const double effective_tile_penalty = (utilization > 1e-9) ? (1.0 / utilization) : 1.0;
  const double output_utilization_penalty =
      (output_utilization > 1e-9) ? (1.0 / output_utilization) : 1.0;

  // ---  WG setup ----------------------------------------------------------
  double L_WG_setup = 1.0;  // WG_setup_Latency

  // ---Prologue ----------------------------------------------------------
  // Original used 1.5× L_mem (comment said 2.2×, code used 1.5×). Keep code behavior.
  double L_prologue = 1.5 * L_mem;  // empirically chosen

  // ---  Epilogue ----------------------------------------------------------
  // Original writes C from all active CUs at mem3 bandwidth limited by occupancy,
  // with MT_M rounded to 128B using *A*’s element size (quirk preserved).
  const double mem_bw_occ         = compute_mem_bw_from_occupancy(hardware, num_active_cus);
  const double mem_bw_occ_limited = hardware.mem3_perf_ratio * mem_bw_occ;

  const std::size_t MT_M_rounded_128B = round_elems_to_128B(MT_M, a_bytes);

  double L_epilogue =
      (static_cast<double>(num_active_cus / std::max<std::size_t>(splitting_factor, 1)) *
       static_cast<double>(MT_M_rounded_128B) * static_cast<double>(MT_N) *
       static_cast<double>(d_bytes)) /
      std::max(1e-9, mem_bw_occ_limited);

  // One compute iteration happens in the prologue (carry original behavior)
  L_epilogue += L_compute * effective_tile_penalty;

  // --- Occupancy scaling for prologue/epilogue (empirical 0.95^occ) --------
  const std::size_t grid_m = math::safe_ceil_div(M, MT_M);
  const std::size_t grid_n = math::safe_ceil_div(N, MT_N);

  // Original: real_occupancy = min(occupancy,
  //   ceil(grid_m*grid_n*batch*splittingFactor / N_CU)).
  // We don’t receive an explicit "occupancy" in the new API, so we replicate the
  // *computed* part (which was the binding term in practice).
  const std::size_t real_occupancy = std::max<std::size_t>(
      1,
      math::safe_ceil_div(grid_m * grid_n * B * std::max<std::size_t>(splitting_factor, 1),
                          std::max<std::size_t>(hardware.N_CU, 1)));

  L_prologue *= std::pow(0.95, static_cast<double>(real_occupancy));
  L_epilogue *= std::pow(0.95, static_cast<double>(real_occupancy));

  // ---  K-split reduction overhead --------------------------------------
  if (splitting_factor > 1) {
    const std::size_t n_partials = splitting_factor - 1;

    // Only the reduction CU reads from all splits (original logic):
    const double partial_read_bytes = static_cast<double>(grid_m) * grid_n * n_partials *
                                      static_cast<double>(MT_M_rounded_128B) * MT_N * d_bytes;

    // All CUs write (once for each partial, and once by the reduction CU for output):
    const double partial_write_bytes = static_cast<double>(grid_m) * grid_n *
                                       static_cast<double>(MT_M_rounded_128B) * MT_N * d_bytes;

    const double partial_readwrite_bytes = partial_read_bytes + partial_write_bytes;

    // 64 threads active in a SIMD; exposed to at least the latency of reducing splitting_factor
    // tiles.
    const double partial_adds =
        (static_cast<double>(MT_M) * MT_N * static_cast<double>(splitting_factor)) / 64.0;

    const double mem_bw_occ2         = compute_mem_bw_from_occupancy(hardware, num_active_cus);
    const double mem_bw_occ_limited2 = hardware.mem3_perf_ratio * mem_bw_occ2;

    const double L_reduce = partial_readwrite_bytes / std::max(1e-9, mem_bw_occ_limited2);

    L_epilogue += L_reduce + partial_adds + 10000.0;
  }

  // -- Conversion overheads (TF32 emu & special BF16 path on gfx950) ---
  double L_cvt        = 0.0;
  const bool tf32_emu = (problem.mi_dtype == data_type_t::XFloat32) &&
                        (hardware.arch == hardware_t::architecture_t::gfx950);

  if (tf32_emu) {
    L_cvt = compute_cvt_overhead(problem, hardware, config);
  } else {
    const bool ss_bss_bf16 = (problem.a_dtype == data_type_t::Float) &&
                             (problem.b_dtype == data_type_t::Float) &&
                             (problem.mi_dtype == data_type_t::BFloat16) &&
                             (hardware.arch == hardware_t::architecture_t::gfx950);
    if (ss_bss_bf16) { L_cvt = compute_cvt_overhead_x1(problem, hardware, config); }
  }

  // ---Single-iteration latency & penalties ------------------------------
  double L_tile_single = (std::max(L_compute, L_mem) * effective_tile_penalty) + L_cvt;

  // The prologue overhead is also reduced by tile underutilization
  double L_prologue_eff = L_prologue * effective_tile_penalty;

  // --- Number of K-iterations (≥ 1, excluding epilogue) ------------------
  const long k_per_split =
      static_cast<long>(math::safe_ceil_div(K, std::max<std::size_t>(splitting_factor, 1)));
  long num_iter =
      static_cast<long>(math::safe_ceil_div(static_cast<std::size_t>(k_per_split), MT_K)) - 1;
  if (num_iter < 1) num_iter = 1;

  // Zero padding in K on the last iteration → penalty scaled by remainder fraction
  if (K % MT_K != 0) {
    const double problem_k_quant =
        static_cast<double>(K % MT_K) / std::max(1.0, static_cast<double>(K));
    L_epilogue += problem_k_quant * 50000.0;  // empirical 50k cycles full-remainder penalty
  }

  // (Optional in original, commented out; keep behavior: do NOT apply)
  // L_epilogue *= output_utilization_penalty;

  // --- Total tile latency ------------------------------------------------
  double L_tile_total =
      (L_tile_single * static_cast<double>(num_iter)) + L_prologue_eff +
      (L_epilogue * 2.0)  // epilogue counted twice in original
      + L_WG_setup +
      (500.0 *
       static_cast<double>(num_iter));  // loop trailer: 7 inst × 4 cycles ≈ 28 → 500 empirically

  // --- Debug logs -----------------------------------------------------------
  if (hardware_t::is_debug_enabled()) {
    const double problem_k_quant = (K % MT_K) / std::max(1.0, static_cast<double>(K));
    hardware.log_debug("Iteration Compute Latency", L_compute);
    hardware.log_debug("L_mem", L_mem);
    hardware.log_debug("L_cvt", L_cvt);
    hardware.log_debug("L_tile_single", L_tile_single);
    hardware.log_debug("num_iter", static_cast<double>(num_iter));
    hardware.log_debug("L_prologue", L_prologue);
    hardware.log_debug("L_epilogue", L_epilogue);
    hardware.log_debug("L_tile_total", L_tile_total);
    hardware.log_debug("Effective Tile Penalty", effective_tile_penalty);
    hardware.log_debug("Problem K quant", problem_k_quant);
    hardware.log_debug("K quant overhead", (problem_k_quant * 50000.0));
    hardware.log_debug("Problem Tile Quant", utilization);
    hardware.log_debug("Real Occupancy", static_cast<double>(real_occupancy));
    hardware.log_debug("Output Utilization Penalty", output_utilization_penalty);
    hardware.log_debug("Output Utilization", output_utilization);

    std::string bound_source;
    double iteration_bound = 0.0;
    if (L_compute >= L_mem) {
      iteration_bound = L_compute + L_cvt;
      bound_source    = "Compute";
    } else {
      iteration_bound = L_mem + L_cvt;
      bound_source    = "Memory";
    }
    hardware.log_debug("Iteration Bound",
                       bound_source + " (" + std::to_string(iteration_bound) + ")");
    hardware.log_debug("K % MT_K", static_cast<double>(K % MT_K));
  }

  return L_tile_total;
}

double compute_timestep_latency(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                std::size_t num_active_cus,
                                std::size_t splitting_factor) {
  // Assume latency of a wave is latency of a single k-complete output tile.
  double L_wave = compute_tile_latency(problem, hardware, config, num_active_cus, splitting_factor);

  return L_wave;
}

// Compute the total latency of a gemm based on the latency of one wave multiplied by the number
// of waves A wave is defined as : The time it takes for one CU to complete one K-complete output
// tile
double compute_total_latency(const problem_t& problem,
                             const hardware_t& hardware,
                             const config_t& config) {
  if (hardware_t::is_debug_enabled()) {
    hardware.log_debug("problem_size",
                       std::to_string(int(problem.size.m)) + "x" +
                           std::to_string(int(problem.size.n)) + "x" +
                           std::to_string(int(problem.size.k)));
    hardware.log_debug("macro_tile",
                       std::to_string(int(config.mt.m)) + "x" + std::to_string(int(config.mt.n)) +
                           "x" + std::to_string(int(config.mt.k)));
    // hardware.log_debug("a_dtype", problem.a_dtype);
    // hardware.log_debug("b_dtype", problem.b_dtype);
  }

  // 0) Short-circuit
  // We don't need to compute latency for all MTs. With this, we can shortcut.
  bool shortCircuit = true;
  if (shortCircuit) {
    // When problem dimensions are small enough that we can fit them in one tile, we should do so.
    // This short circuit condition also decreases selection latency when problems are very small
    // :)
    // TODO 256 and 256 here should be largest M and N tile dimensions in library
    if (problem.size.m <= 256 && problem.size.n <= 256 && problem.size.k < 1024 &&
        problem.batch != 1 && (config.mt.m < problem.size.m || config.mt.n < problem.size.n))
      return std::numeric_limits<double>::max();

    // Use Dot2 only for M < 3
    if (config.mi.m == 1 && config.mi.n == 1 && config.mi.k == 64 && problem.size.m > 2)
      return std::numeric_limits<double>::max();
  }

  // 1) Find CU occupancy
  auto [num_active_cus, num_waves, splitting_factor] =
      compute_cu_occupancy(problem, hardware, config, grid_selection_t::k_split_aware);

  // Compute latency of a single timestep (each CU processing one workgroup)
  double L_wave =
      compute_timestep_latency(problem, hardware, config, num_active_cus, splitting_factor);

  // Compute latency for all waves and return it as the latency for the MT/problem
  double total_latency = L_wave * num_waves;

  // Custom heuristics to adjust the latency value
  // TODO These are quantifying effects that don't work in the current math.
  // TODO THESE SHOULD BE TEMPORARY FIXES AND BE MORE SOLIDLY INTEGRATED LATER
  bool heuristics = hardware_t::is_heuristics_enabled();

  std::size_t M     = problem.size.m;
  std::size_t N     = problem.size.n;
  std::size_t K     = problem.size.k;
  std::size_t batch = problem.batch;

  bool a_trans = problem.a_transpose == transpose_t::T;
  bool b_trans = problem.b_transpose == transpose_t::T;

  std::size_t MT_M = config.mt.m;
  std::size_t MT_N = config.mt.n;
  std::size_t MT_K = config.mt.k;
  std::size_t MI_M = config.mi.m;
  std::size_t MI_N = config.mi.n;
  std::size_t MI_K = config.mi.k;

  // Heuristics for TF32
  bool tf32_emu = ((problem.mi_dtype == data_type_t::XFloat32) &&
                   (hardware.arch == hardware_t::architecture_t::gfx950));
  if (tf32_emu && heuristics) {
    // The kernel for this is more optimized (Custom kernel NT)
    if ((!a_trans && b_trans) && MT_M == 256 && MT_N == 256 && MT_K == 32) {
      total_latency = total_latency * 0.6;
    }

    // The kernel for this is more optimized (Custom kernel NN)
    if ((!a_trans && !b_trans) && MT_M == 256 && MT_N == 256 && MT_K == 32) {
      total_latency = total_latency * 0.8;
    }

    // The kernel for this is more optimized (Custom kernel TN)
    if ((a_trans && !b_trans) && MT_M == 256 && MT_N == 256 && MT_K == 32) {
      total_latency = total_latency * 0.8;
    }

    // Bias large DU where K-dimension is large and M and N are small.
    if ((K >= (M * 16) && K >= (N * 16)) && (MT_K >= 128)) { total_latency = total_latency * 0.5; }
  }

  if (heuristics && !tf32_emu) {
    // Penalize tiles that lead to edge waste
    const size_t numMT_M = math::safe_ceil_div(M, MT_M);
    const size_t numMT_N = math::safe_ceil_div(N, MT_N);
    const double waste   = static_cast<double>(numMT_M * MT_M * numMT_N * MT_N) / (M * N);
    double edge_penalty  = std::pow(waste, 0.8);
    if (batch > 10)
      edge_penalty = std::pow(waste, 1.5) * std::log10((double)batch) *
                     std::pow((double)numMT_M * numMT_N, 0.2);
    total_latency = total_latency * edge_penalty;

    // Penalize K iterations
    std::size_t K_iters = math::safe_ceil_div(K, MT_K);
    if (K_iters <= 2) {
      total_latency = total_latency * 8;
    } else if (K_iters <= 4) {
      total_latency = total_latency * 4;
    } else if (K_iters <= 8) {
      total_latency = total_latency * 2.1;
    }

    // Bias toward not splitting for small K values
    // This should actually come from SK grid prediction
    if (splitting_factor > 1 && K < 2048) { total_latency = total_latency * splitting_factor; }

    // There is no case where a kernel with MT_K > K wins unless K < MI_K.
    // Unless it is Dot2.
    if (K < MT_K && MI_M != 1) total_latency = total_latency * (MT_K - K);

    // Bias Model towards at least one dim being power of 2
    bool MT_M_is_power_two = (MT_M > 0) && (MT_M & (MT_M - 1)) == 0;
    bool MT_N_is_power_two = (MT_N > 0) && (MT_N & (MT_N - 1)) == 0;
    if (!MT_M_is_power_two && !MT_N_is_power_two) { total_latency = total_latency * 1.1; }

    // Bias Model towards both dims being a power of 2
    if (MT_M_is_power_two && MT_N_is_power_two) { total_latency = total_latency * 0.9; }

    // Bias toward 512 tiles for sizes "very skinny" sizes
    // "very skinny" definition: either N or M less than 16 (1 tile) and the other one requires
    // more than 100 waves (100*numCUs tiles)
    if (M < 16 && N > 100 * hardware.N_CU * 512 && MT_N == 512) {
      total_latency = total_latency * 0.25;
    }
    if (N < 16 && M > 100 * hardware.N_CU * 512 && MT_M == 512) {
      total_latency = total_latency * 0.25;
    }

    // DOT2 Kernels
    if (MI_M == 1 && MI_N == 1 && MI_K == 64) {
      // Bias DOT2 kernels in which the tile dimensions in M and K are equal to the problem
      // dimensions
      if (MT_M == M || MT_K == K) { total_latency = total_latency * 0.8; }
    }

    // Heuristics for FP16
    if (data_type_to_bits(problem.a_dtype) == 16) {
      // These kernels are more optimized (Custom kernels)
      // All layouts
      if (MT_M == 256 && MT_N == 256 && MT_K == 64) {
        total_latency = total_latency * 0.85;
        if ((a_trans && !b_trans) && (M == MT_M && N > 256 * MT_N && K >= 4 * MT_K)) {
          total_latency = total_latency * 0.3;
        }
      }

      // The kernel for this is less optimized, for some reason
      if (MT_M == 256 && MT_N == 16 && MT_K == 128) { total_latency = total_latency * 2; }

      // The kernel for this is less optimized, for some reason
      if (MT_M == 16 && MT_N == 256 && MT_K == 128) { total_latency = total_latency * 2; }
    }

    // Heuristics for FP8
    if (data_type_to_bits(problem.a_dtype) == 8) {
      // The kernel for this is more optimized (Custom kernel)
      if (a_trans && !b_trans && MT_M == 256 && MT_N == 256 && MT_K == 128) {
        total_latency = total_latency * 0.8;
      }

      // Bias towards dimensions divisible by 64 for 8-bit datatypes
      if ((MT_M > 64) && (MT_M % 64 != 0)) { total_latency = total_latency * 1.2; }
      if ((MT_N > 64) && (MT_N % 64 != 0)) { total_latency = total_latency * 1.2; }
    }
  }

  if (hardware_t::is_debug_enabled()) {
    hardware.log_debug("total_latency_with_heuristics", total_latency);
    hardware.print_debug_info();
  }

  return total_latency;
}

// Extract analytical metrics from a GEMM computation and return as map
std::unordered_map<std::string, std::string> extract_analytical_metrics(const hardware_t& hardware,
                                                                        const problem_t& problem,
                                                                        const config_t& config) {
  // Clear any previous debug info
  hardware.clear_debug();

  // Enable metrics collection mode to bypass environment variable check
  hardware.set_metrics_collection_mode(true);

  // Extract basic parameters for forced logging
  std::size_t M                = problem.size.m;
  std::size_t N                = problem.size.n;
  std::size_t K                = problem.size.k;
  std::size_t batch            = problem.batch;
  bool a_trans                 = problem.a_transpose == transpose_t::T;
  bool b_trans                 = problem.b_transpose == transpose_t::T;
  std::size_t element_size_A   = data_type_to_bits(problem.a_dtype);
  std::size_t element_size_B   = data_type_to_bits(problem.b_dtype);
  std::size_t element_size_out = data_type_to_bits(problem.d_dtype);
  data_type_t mi_datatype      = problem.mi_dtype;
  std::size_t a_mx_block_size  = problem.a_mx_block_size;
  std::size_t b_mx_block_size  = problem.b_mx_block_size;

  std::size_t MT_M = config.mt.m;
  std::size_t MT_N = config.mt.n;
  std::size_t MT_K = config.mt.k;
  std::size_t MI_M = config.mi.m;
  std::size_t MI_N = config.mi.n;
  std::size_t MI_K = config.mi.k;
  int WGM          = config.workgroup_mapping;

  // Log basic problem and configuration parameters
  hardware.log_debug(
      "problem_size",
      std::to_string(int(M)) + "x" + std::to_string(int(N)) + "x" + std::to_string(int(K)));
  hardware.log_debug("macro_tile",
                     std::to_string(int(MT_M)) + "x" + std::to_string(int(MT_N)) + "x" +
                         std::to_string(int(MT_K)));
  hardware.log_debug("micro_tile",
                     std::to_string(int(MI_M)) + "x" + std::to_string(int(MI_N)) + "x" +
                         std::to_string(int(MI_K)));
  hardware.log_debug("element_size_a_bits", element_size_A);
  hardware.log_debug("element_size_b_bits", element_size_B);
  hardware.log_debug("element_size_out_bits", element_size_out);
  hardware.log_debug("batch_size", batch);
  hardware.log_debug("a_transpose", a_trans ? "true" : "false");
  hardware.log_debug("b_transpose", b_trans ? "true" : "false");
  hardware.log_debug("workgroup_mapping", WGM);
  hardware.log_debug("a_mx_block_size", a_mx_block_size);
  hardware.log_debug("b_mx_block_size", b_mx_block_size);

  // Run compute_total_latency which will populate debug_info with all intermediate values
  // since we enabled metrics collection mode
  double total_latency = compute_total_latency(problem, hardware, config);

  // Disable metrics collection mode
  hardware.set_metrics_collection_mode(false);

  // Add the final latency to metrics
  hardware.log_debug_force("final_total_latency", total_latency);

  // Return all collected metrics
  return hardware.get_analytical_metrics();
}

// Extract analytical metrics and export to CSV
void extract_analytical_metrics_csv(const hardware_t& hardware,
                                    const problem_t& problem,
                                    const config_t& config,
                                    const std::string& filename) {
  // Extract metrics
  extract_analytical_metrics(hardware, problem, config);

  // Export to CSV
  hardware.extract_analytical_metrics_csv(filename);
}

}  // namespace origami
