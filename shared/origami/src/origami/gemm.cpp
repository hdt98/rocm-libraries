// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <set>
#include <tuple>

#include "origami/math.hpp"
#include "origami/types.hpp"
#include "origami/hardware.hpp"

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

  bool a_trans = problem.a_transpose;
  bool b_trans = problem.b_transpose;

  int a_bytes = data_type_to_bytes(problem.a_dtype);
  int b_bytes = data_type_to_bytes(problem.b_dtype);

  // TN
  if (a_trans && !b_trans) {
    // We want to penalize tiles that can't be coalesced for T,N where K is contiguous dimension.
    // In this case, that's when the K dimension is indivisible by 128 bytes.
    if (config.mt.k * a_bytes % 128 != 0) { L_MT *= 1.5; }
    if (config.mt.k * b_bytes % 128 != 0) { L_MT *= 1.5; }
  }

  // NT: A is contiguous in M and B is contiguous in N
  if (!a_trans && b_trans) {
    // LDS Load Granularity is 128 Bytes -> If we load an amount indivisible by 128 bytes in either
    // contiguous dimesion from LDS then we will get poor LDS utilization. This actually happens as
    // more like a quantization effect where if either contiguous dimension of the tile is not
    // evenly divisible by 128-bytes We end up with inefficient loads. Multiplication by a value is
    // arbitrary, there is probably a better analytical method to quantify the true impact of this
    // Effect on the efficiency of computation.
    if ((config.mt.m * a_bytes) % 128 != 0) { L_MT *= 2; }
    if ((config.mt.n * b_bytes) % 128 != 0) { L_MT *= 2; }
    // NT Transpose Overhead Scales in both.
  }

  // TT: A is contiguous in K and B is contiguous in N
  if (a_trans && b_trans) {
    if (config.mt.k * a_bytes < 128) { L_MT *= 2; }
    if (config.mt.n * b_bytes < 128) { L_MT *= 2; }
  }

  // NN: A is contiguous in M and B is contiguous in K
  if (!a_trans && !b_trans) {
    if (config.mt.m * a_bytes < 128) { L_MT *= 2; }
    if (config.mt.k * b_bytes < 128) { L_MT *= 2; }
  }

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
  std::size_t grid_m = math::safe_ceil_div(problem.size.m, config.mt.m);
  std::size_t grid_n = math::safe_ceil_div(problem.size.n, config.mt.n);

  // Distribute CUs per XCD
  // Modify cu_per_xcd to only take into account the CUs that might share same K-tiles
  // This is to factor in the effect of splitting on L2
  std::size_t cu_per_xcd = math::safe_ceil_div(grid_m * grid_n, hardware.NUM_XCD);
  cu_per_xcd /= splitting_factor;

  // N dimension of mem1 tile is divided by whichever is smaller between WGM and grid
  std::size_t l2_n = std::min(std::size_t(config.workgroup_mapping), grid_n);
  std::size_t l2_m = cu_per_xcd / l2_n;

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
  std::size_t grid_m = math::safe_ceil_div(problem.size.m, config.mt.m);
  std::size_t grid_n = math::safe_ceil_div(problem.size.n, config.mt.n);

  // Tile dimensions in MALL.
  std::size_t mall_m = grid_m * grid_n / config.workgroup_mapping;  // M dimension of mem2 tile
  std::size_t mall_n =
      std::min(std::size_t(config.workgroup_mapping), grid_n);  // N dimension of mem2 tile

  // If a single mem2 tile is larger than the grid, extend its M dimension
  if (mall_m > grid_m) {
    int num_wraps = (mall_m / grid_m) - 1;
    mall_n += (num_wraps * config.workgroup_mapping);
    mall_m = grid_m;
  }

  // Clamp the tile dimensions to valid ranges
  mall_m = std::max(std::min(grid_m, mall_m), std::size_t(1));
  mall_n = std::max(std::min(grid_n, mall_n), std::size_t(1));

  // Unique “uncached” entries of A/B for this XCD
  int mall_A_uncached_reads = mall_m * config.mt.mk();
  int mall_B_uncached_reads = mall_n * config.mt.nk();
  int total_uncached_read   = mall_A_uncached_reads + mall_B_uncached_reads;

  // Total A/B reads considering repeated usage
  long long mall_A_reads = static_cast<long long>(mall_m) * mall_n * config.mt.mk();
  long long mall_B_reads = static_cast<long long>(mall_n) * mall_m * config.mt.nk();

  // Avoid division by zero
  long long total_reads  = std::max(mall_A_reads + mall_B_reads, 1LL);
  long long cached_reads = total_reads - total_uncached_read;

  double mall_hit = static_cast<double>(cached_reads) / static_cast<double>(total_reads);

  if (hardware_t::is_debug_enabled()) {
    hardware.log_debug("mall_tile_m", mall_m);
    hardware.log_debug("mall_tile_n", mall_n);
  }

  return mall_hit;
}

double compute_memory_latency(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              std::size_t num_active_cus,
                              std::size_t splitting_factor) {
  // 1) Estimate L2 hit-rate
  double H_mem1 = estimate_l2_hit(problem, hardware, config, splitting_factor);

  // 2) Estimate mall hit-rate
  double H_mem2 = estimate_mall_hit(problem, hardware, config, num_active_cus, splitting_factor);

  // 3) Total loads are loads from A and loads from B
  int a_bytes = data_type_to_bytes(problem.a_dtype);
  int b_bytes = data_type_to_bytes(problem.b_dtype);

  int a_bits = data_type_to_bits(problem.a_dtype);
  int b_bits = data_type_to_bits(problem.b_dtype);

  std::size_t Ld_A_value  = config.mt.mk();
  std::size_t Ld_B_value  = config.mt.nk();
  std::size_t Ld_CU_bytes = (Ld_A_value * a_bytes)     // A Bytes
                            + (Ld_B_value * b_bytes);  // B Bytes

  // Logic for block scaled datatypes (Assuming BS=32 and 8-bit scales)
  // TODO This is technically wrong, need separate flag to enable MX so we can differentiate FP8 and
  // TODO MX8, we now have access to a_dtype/b_dtype.
  if (a_bits < 8 && problem.a_mx_block_size != 0) {
    // Number of scales per tile
    std::size_t num_scales_A = math::safe_ceil_div(config.mt.mk(), problem.a_mx_block_size);
    Ld_CU_bytes += num_scales_A;  // One Byte per scale
  }
  if (b_bits < 8 && problem.b_mx_block_size != 0) {
    // Number of scales per tile
    std::size_t num_scales_B = math::safe_ceil_div(config.mt.nk(), problem.b_mx_block_size);
    Ld_CU_bytes += num_scales_B;  // One Byte per scale
  }

  // 4) total loads by all CUs
  double total_Ld = Ld_CU_bytes * static_cast<double>(num_active_cus);

  // 5) mem1‐limited factor (simple linear model)
  double mem1_bw_limited = static_cast<double>(num_active_cus) / static_cast<double>(hardware.N_CU);
  double limited_mem1_bw = hardware.mem1_perf_ratio * mem1_bw_limited;

  // 6) mem1 latency
  double L_mem_mem1 = (limited_mem1_bw > 0) ? (total_Ld / (limited_mem1_bw)) : 0.0;

  // 7) mem2‐limited from occupancy (Can't Issue enough load/stores)
  double bw_limited = compute_mem_bw_from_occupancy(hardware, num_active_cus);

  // 8) loads that reach each level
  double Ld_mem2 = (1.0 - H_mem1) * total_Ld;
  double Ld_MEM  = (1.0 - H_mem2) * Ld_mem2;

  // 9) enforce whole‐problem minimum loads
  if (num_active_cus < hardware.N_CU) {
    // TODO: Check if this math is correct:
    double min_load =
        static_cast<double>(problem.size.m * config.mt.k * splitting_factor * a_bytes +
                            problem.size.n * config.mt.k * splitting_factor * b_bytes);
    Ld_MEM  = std::max(Ld_MEM, min_load) * problem.batch;
    Ld_mem2 = std::max(Ld_mem2, min_load) * problem.batch;
  }

  // 10) mem2 latency
  double limited_mem2_bw = hardware.mem2_perf_ratio * bw_limited;
  double L_mem_mem2      = (limited_mem2_bw > 0) ? (Ld_mem2 / limited_mem2_bw) : 0.0;

  // 11) MEM latency
  double limited_mem_bw = hardware.mem3_perf_ratio * bw_limited;
  double L_mem_MEM      = (limited_mem_bw > 0) ? (Ld_MEM / limited_mem_bw) : 0.0;
  L_mem_MEM += 200;  // Load Latency

  // 12) pick the worst‐case bound
  double L_mem = std::max({L_mem_mem1, L_mem_mem2, L_mem_MEM});

  bool a_trans = problem.a_transpose;
  bool b_trans = problem.b_transpose;

  // NT
  if (!a_trans && b_trans) {
    // LDS Load Granularity is 128 Bytes -> If we load an amount indivisible by 128 bytes in either
    // contiguous dimesion from LDS then we will get poor LDS utilization. This actually happens as
    // more like a quantization effect where if either contiguous dimension of the tile is not
    // evenly divisible by 128-bytes We end up with inefficient loads. Multiplication by a value is
    // arbitrary, there is probably a better analytical method to quantify the true impact of this
    // Effect on the efficiency of computation.
    if ((config.mt.m * a_bytes) % 128 != 0) { L_mem = L_mem * 2; }
    if ((config.mt.n * b_bytes) % 128 != 0) { L_mem = L_mem * 2; }
  }

  // TT : A is contiguous in K and B is contiguous in N
  if (a_trans && b_trans) {
    if (config.mt.k * a_bytes < 128) { L_mem = L_mem * 2; }
    if (config.mt.n * b_bytes < 128) { L_mem = L_mem * 2; }
  }

  // NN : A is contiguous in M and B is contiguous in K
  if (!a_trans && !b_trans) {
    if (config.mt.m * a_bytes < 128) { L_mem = L_mem * 2; }
    if (config.mt.k * b_bytes < 128) { L_mem = L_mem * 2; }
  }

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
  }

  return L_mem;
}

double compute_tile_latency(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config,
                            std::size_t num_active_cus,
                            std::size_t splitting_factor) {
  // 1) Compute per-tile latencies
  double L_compute = compute_mt_compute_latency(problem, hardware, config);

  double L_mem =
      compute_memory_latency(problem, hardware, config, num_active_cus, splitting_factor);

  // 2) Work-group setup & iteration latencies
  double L_WG_setup = 1;  // WG_setup_Latency

  // 3) Prologue: 2.2× memory latency
  double L_prologue = 1.5 * L_mem;  // 1.5 chosen emprically

  // 4) Epilogue: writes from all active CUs with limited bandwidth
  double epilogue_limited = (static_cast<double>(num_active_cus) / hardware.N_CU);
  double limited_mem1     = hardware.mem1_perf_ratio * epilogue_limited;
  if (limited_mem1 < 1) { limited_mem1 = 10; }
  double L_epilogue =
      (static_cast<double>(num_active_cus) * config.mt.mn() * data_type_to_bytes(problem.d_dtype)) /
      limited_mem1;

  // 4') K-split reductions are globally coherent, we need to write and read split-1 MT_M*MT_N tiles
  // to coherent memory
  if (splitting_factor > 1) {
    std::size_t n_partials = splitting_factor - 1;
    double partial_readwrite_bytes =
        (2 * num_active_cus * config.mt.mn() * data_type_to_bytes(problem.d_dtype) * n_partials);
    double L_reduce = partial_readwrite_bytes / (hardware.mem3_perf_ratio);
    L_epilogue += L_reduce * 1;
  }

  // 4'') tf32 emu has some more overhead
  double L_cvt  = 0;
  bool tf32_emu = ((problem.mi_dtype == data_type_t::XFloat32) &&
                   (hardware.arch == hardware_t::architecture_t::gfx950));
  if (tf32_emu) { L_cvt = compute_cvt_overhead(problem, hardware, config); }

  // 5) Single-tile latency (always additive)
  double L_tile_single = std::max(L_compute, L_mem) + L_cvt;

  // 6) Number of K-iterations (excluding epilogue), at least 1
  // long num_iter = static_cast<long>(((K + MT_K - 1) / MT_K)) - 1;
  // num_iter      = std::ceil(num_iter / splitting_factor);
  // num_iter      = std::max(num_iter, 1L);
  long k_per_splits = static_cast<long>(math::safe_ceil_div(problem.size.k, splitting_factor));
  long num_iter     = static_cast<long>(math::safe_ceil_div(k_per_splits, config.mt.k)) - 1;

  // 7) Total tile latency
  double L_tile_total =
      (L_tile_single * num_iter) + L_prologue + L_epilogue + L_WG_setup +
      (28 * num_iter);  // 7 instructions (each with 4 cycles) at the end of the loop

  if (hardware_t::is_debug_enabled()) {
    hardware.log_debug("compute_latency", L_compute);
    hardware.log_debug("memory_latency", L_mem);
    hardware.log_debug("convert_latency", L_cvt);
    hardware.log_debug("tile_latency_single", L_tile_single);
    hardware.log_debug("num_iterations", num_iter);
    hardware.log_debug("prologue_latency", L_prologue);
    hardware.log_debug("epilogue_latency", L_epilogue);
    hardware.log_debug("tile_latency_total", L_tile_total);
  }

  return L_tile_total;
}

double compute_wave_latency(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config,
                            std::size_t num_active_cus,
                            std::size_t splitting_factor) {
  // Assume latency of a wave is latency of a single k-complete output tile.
  double L_wave = compute_tile_latency(problem, hardware, config, num_active_cus, splitting_factor);

  return L_wave;
}

// Compute the total latency of a gemm based on the latency of one wave multiplied by the number of
// waves A wave is defined as : The time it takes for one CU to complete one K-complete output tile
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
    // This short circuit condition also decreases selection latency when problems are very small :)
    // TODO 256 and 256 here should be largest M and N tile dimensions in library
    if (problem.size.m <= 256 && problem.size.n <= 256 && problem.size.k < 1024 &&
        problem.batch != 1 && (config.mt.m < problem.size.m || config.mt.n < problem.size.n))
      return std::numeric_limits<double>::max();

    // Override dot2 instruction with vector lane widths
    if (config.mi.m == 0 && config.mi.n == 0 && config.mi.k == 0) {
      // We only use Dot2 for NN layout where M < 3
      if (problem.size.m > 2 || problem.a_transpose || problem.b_transpose)
        return std::numeric_limits<double>::max();

      config.mi.m = 1;
      config.mi.n = 1;
      config.mi.k = 64;
    }
  }

  // 1) Find CU occupancy
  auto [num_active_cus, num_waves, splitting_factor] =
      compute_cu_occupancy(problem, hardware, config, grid_selection_t::k_split_aware);

  // 2) Compute latency of a wave
  // Compute latency of a wave
  double L_wave = compute_wave_latency(problem, hardware, config, num_active_cus, splitting_factor);

  // Compute latency for all waves and return it as the latency for the MT/problem
  double total_latency = L_wave * num_waves;

  // 3) Customized heuristics
  // TODO These are quantifying effects that don't work in the current math.
  // TODO THESE SHOULD BE TEMPORARY FIXES AND BE MORE SOLIDLY INTEGRATED LATER
  bool heuristics = hardware_t::is_heuristics_enabled();

  std::size_t M     = problem.size.m;
  std::size_t N     = problem.size.n;
  std::size_t K     = problem.size.k;
  std::size_t batch = problem.batch;

  bool a_trans = problem.a_transpose;
  bool b_trans = problem.b_transpose;

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
  bool a_trans                 = problem.a_transpose;
  bool b_trans                 = problem.b_transpose;
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
