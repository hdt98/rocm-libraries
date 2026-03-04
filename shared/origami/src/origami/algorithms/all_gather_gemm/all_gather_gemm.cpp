// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/algorithms/all_gather_gemm/all_gather_gemm.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

#include "origami/gemm.hpp"
#include "origami/math.hpp"

// ═══════════════════════════════════════════════════════════════════════════
//  origami:: (top-level resource constraints)
// ═══════════════════════════════════════════════════════════════════════════

namespace origami {

resource_constraints_t resource_constraints_t::unconstrained(const hardware_t& hw) {
  resource_constraints_t rc;
  rc.available_cus = hw.N_CU;

  // Derive peak HBM bandwidth from the GPU architecture.  Origami's internal
  // memory model doesn't carry a single "peak BW" number (it derives per-WG
  // bandwidth from occupancy), so we keep a per-arch lookup here.  The values
  // must stay consistent with network_t::for_architecture().
  switch (hw.arch) {
    case hardware_t::architecture_t::gfx942:
    case hardware_t::architecture_t::gfx950:
      rc.available_hbm_bw_gbps = 5300.0;  // MI300X: 8× HBM3 stacks
      break;
    case hardware_t::architecture_t::gfx90a:
      rc.available_hbm_bw_gbps = 3200.0;  // MI250X: 4× HBM2e stacks
      break;
    default:
      rc.available_hbm_bw_gbps = 5300.0;  // fallback to MI300X
      break;
  }

  // hardware.L2_capacity uses Origami's internal unit convention where
  // (L2_capacity * 1024) gives the value that compute_l2_hit_rate_global()
  // uses as "capacity in bytes".  We store effective_l2_bytes in the same
  // derived-byte space so the constraint adapter can convert back cleanly:
  //   hw_copy.L2_capacity = effective_l2_bytes / 1024.0
  rc.effective_l2_bytes = static_cast<double>(hw.L2_capacity) * 1024.0;

  // MALL capacity estimate.  On MI300X, MALL is ~6× the per-L2-domain
  // capacity.  This is a rough heuristic -- Origami's estimate_mall_hit()
  // derives MALL capacity from hardware internals, and the constraint adapter
  // does not currently modify MALL in the hardware copy.
  rc.effective_mall_bytes = hw.has_MALL()
      ? static_cast<double>(hw.L2_capacity) * 1024.0 * 6.0
      : 0.0;
  return rc;
}

}  // namespace origami

// ═══════════════════════════════════════════════════════════════════════════
//  origami::algorithms::all_gather_gemm
// ═══════════════════════════════════════════════════════════════════════════

namespace origami::algorithms::all_gather_gemm {

// ── Internal helpers ────────────────────────────────────────────────────
//
// These are file-local (static) and not exposed in the header.

/// Origami's compute_tile_latency() asserts that the config has valid MI
/// dimensions (mi.m > 0, mi.n > 0, mi.k > 0) and positive occupancy.
/// Callers of our model may not set these because they're GEMM-internal
/// details, so we fill them in from hardware tables here.
static origami::config_t ensure_valid_gemm_config(
    const origami::config_t&   config,
    const origami::hardware_t& hardware,
    const origami::problem_t&  problem) {
  origami::config_t cfg = config;
  if (cfg.mi.m == 0 || cfg.mi.n == 0 || cfg.mi.k == 0) {
    cfg.mi = hardware.get_recommended_matrix_instruction(problem.mi_dtype);
  }
  // The fused kernel runs one WG per CU (large tiles), so occupancy = 1.
  if (cfg.occupancy <= 0) cfg.occupancy = 1;
  return cfg;
}

/// The fused-kernel problem type (all_gather_matmul_problem_t) inherits from
/// problem_t and adds comm fields (world_size, link_bw_gbps).  Origami's GEMM
/// functions take plain problem_t, so we need to strip the comm fields.
///
/// We construct a new problem_t rather than just slicing because the derived
/// type might have additional fields that would get lost in a reference cast,
/// and we want to be explicit about which fields the GEMM model sees.
static origami::problem_t make_gemm_subproblem(
    const all_gather_matmul_problem_t& p) {
  origami::problem_t sub;
  sub.size        = p.size;       // Full (M, N, K) -- NOT per-GPU sizes
  sub.batch       = p.batch;
  sub.a_dtype     = p.a_dtype;
  sub.b_dtype     = p.b_dtype;
  sub.c_dtype     = p.c_dtype;
  sub.d_dtype     = p.d_dtype;
  sub.mi_dtype    = p.mi_dtype;
  sub.a_transpose = p.a_transpose;
  sub.b_transpose = p.b_transpose;
  return sub;
}

/// Compute the time for one GEMM WG to complete its output tile using
/// Origami's detailed GEMM model, plus the fused-kernel flag-poll overhead.
///
/// WHY ORIGAMI INSTEAD OF ROOFLINE:
///   The old model (compute_gemm_wg_time_us) used:
///     total_flops / (roofline_tflops / num_cus) × occupancy_factor
///   with hardcoded 1300 TFLOPS and 5300 GB/s.  This misses:
///     - Instruction-level compute: Origami looks up MI latencies from
///       hardware tables (e.g. MFMA_F16_16x16x32 takes X cycles on gfx942)
///     - Memory hierarchy: L2 hit rate estimation from tile/problem geometry,
///       MALL hit rate from CU occupancy, per-level BW scaling
///     - Utilization: partial tiles at problem edges waste CU time
///     - Prologue/epilogue: cold-cache first iteration, output writes
///
/// WHY FLAG POLL IS ADDITIVE:
///   In the fused kernel, each GEMM WG must poll num_flag_groups_k flags
///   across its K-loop (one acquire-atomic per flag group boundary).  The
///   poll time is additive to the GEMM tile time because the WG stalls on
///   each poll until the corresponding data is available.  The stall time
///   is flag_poll_us per flag group (includes atomic latency + spin-wait).
///
/// CYCLES → MICROSECONDS:
///   Origami returns tile latency in clock cycles (at compute_clock_ghz).
///   cycles / (GHz × 1e3) = cycles / (1e9 cycles/s × 1e-6 s/μs) = μs.
static double origami_gemm_wg_time_us(
    const origami::problem_t&  problem,
    const origami::hardware_t& hardware,
    const origami::config_t&   config,
    std::size_t                num_active_cus,
    std::size_t                num_flag_groups,
    double                     flag_poll_us) {
  auto valid_cfg = ensure_valid_gemm_config(config, hardware, problem);

  double tile_lat_cycles = origami::compute_tile_latency(
      problem, hardware, valid_cfg, num_active_cus, /*split*/ 1);

  double tile_lat_us = tile_lat_cycles / (hardware.compute_clock_ghz * 1e3);

  double flag_us = static_cast<double>(num_flag_groups) * flag_poll_us;

  return tile_lat_us + flag_us;
}

/// Predict the total wall time of a standalone GEMM (no communication) using
/// Origami's per-tile model.  This gives us the "compute_time_ms" baseline
/// against which we compare comm_time_ms to determine pipeline depth.
///
/// WAVE MODEL:
///   A GEMM of M×N with tiles of bm×bn produces ceil(M/bm) × ceil(N/bn)
///   output tiles.  The GPU processes these in waves of min(num_tiles, N_CU)
///   tiles each.  Total time = tile_latency × num_waves.
///
///   We use min(num_tiles, N_CU) as the active CU count passed to Origami
///   because when tiles < CUs, only that many CUs are actually occupied.
///   This affects Origami's memory BW model (fewer concurrent readers →
///   different BW-from-occupancy) and L2 hit rate (fewer concurrent tiles →
///   less cache contention).  For typical fused all-gather problems
///   (M=131072, tiles=4096) this saturates at N_CU.
static double origami_total_gemm_ms(
    const origami::problem_t&  problem,
    const origami::hardware_t& hardware,
    const origami::config_t&   config) {
  auto valid_cfg = ensure_valid_gemm_config(config, hardware, problem);

  std::size_t bm = valid_cfg.mt.m;
  std::size_t bn = valid_cfg.mt.n;
  std::size_t M  = problem.size.m;
  std::size_t N  = problem.size.n;

  std::size_t num_tiles = math::safe_ceil_div(M, bm) * math::safe_ceil_div(N, bn);
  std::size_t num_active_cus = std::min(num_tiles, hardware.N_CU);

  double tile_lat_cycles = origami::compute_tile_latency(
      problem, hardware, valid_cfg, num_active_cus, /*split*/ 1);

  std::size_t num_waves = math::safe_ceil_div(num_tiles, num_active_cus);

  // cycles / (GHz × 1e6) = cycles / (1e9 cycles/s × 1e-3 s/ms) = ms
  double total_cycles = tile_lat_cycles * static_cast<double>(num_waves);
  return total_cycles / (hardware.compute_clock_ghz * 1e6);
}

// ── network_t factories ─────────────────────────────────────────────────
//
// Calibrated timing constants are architecture-specific.  Each architecture
// entry below was derived from kernel trace analysis on the corresponding GPU.
// When adding a new architecture, run the fused kernel with known problem
// sizes, measure the resulting wall times, and fit the constants to match.

network_t network_t::for_architecture(hardware_t::architecture_t arch,
                                      double link_bw,
                                      std::size_t world_size) {
  network_t n;
  n.link_bw_gbps = link_bw;
  n.world_size   = world_size;

  switch (arch) {
    // MI300X / MI300A — 8 XCDs, XGMI interconnect.
    // Calibrated from MI300X kernel traces; see derive_params.py for
    // the original methodology.
    // MI300X: 8× HBM3 stacks → 5.3 TB/s peak.
    // MI325X/MI350X (gfx950): same memory interface class.
    case hardware_t::architecture_t::gfx942:
    case hardware_t::architecture_t::gfx950:
      n.peak_hbm_bw_gbps     = 5300.0;
      n.scheduling_factor    = 4.5;   // measured wall / CU-queue lower bound
      n.gather_overhead      = 1.5;   // XGMI protocol overhead multiplier
      n.write_bw_per_wg_gbps = 15.0;  // sustained per-WG HBM write throughput
      n.flag_poll_us         = 2.5;   // acquire-atomic + spin-wait per flag
      n.flag_store_us        = 5.0;   // release-atomic after data write
      break;

    // MI200 series — single GCD, Infinity Fabric interconnect.
    // MI250X: 4× HBM2e stacks → ~3.2 TB/s peak.
    // No fused-kernel traces available yet; using MI300X values as a starting
    // point with a higher scheduling factor to account for the single-GCD
    // scheduler and lower Infinity Fabric bandwidth.
    case hardware_t::architecture_t::gfx90a:
      n.peak_hbm_bw_gbps     = 3200.0;
      n.scheduling_factor    = 5.5;
      n.gather_overhead      = 1.8;
      n.write_bw_per_wg_gbps = 12.0;
      n.flag_poll_us         = 3.0;
      n.flag_store_us        = 6.0;
      break;

    // Fallback for architectures without calibration data.
    // Uses MI300X values as the best available estimate.
    default:
      n.peak_hbm_bw_gbps     = 5300.0;
      n.scheduling_factor    = 4.5;
      n.gather_overhead      = 1.5;
      n.write_bw_per_wg_gbps = 15.0;
      n.flag_poll_us         = 2.5;
      n.flag_store_us        = 5.0;
      break;
  }

  return n;
}

network_t network_t::mi300x_defaults(double link_bw, std::size_t world_size) {
  return for_architecture(hardware_t::architecture_t::gfx942,
                          link_bw, world_size);
}

// ── Constraint adapter ──────────────────────────────────────────────────
//
// DESIGN DECISION: modify inputs, not outputs.
//
// An earlier version of this function called Origami's compute_tile_latency()
// to get the total, then separately called compute_mt_compute_latency() and
// compute_memory_latency(), applied a BW scaling factor to the memory part,
// and recomposed with: k_iters × max(compute, adjusted_memory).
//
// This was wrong because compute_tile_latency() does NOT simply return
// max(compute, memory) × k_iters.  It models:
//   - prologue (cache-cold first K-iteration, scaled by occupancy)
//   - epilogue (output writes, K-split reduction, K-padding penalty)
//   - utilization penalties from partial tiles
//   - main_loop_efficiency (pipeline stalls)
//   - heuristic weights on each component (weight_compute, weight_memory, etc.)
//
// All of that was being thrown away by the manual recomposition.
//
// The new approach: copy hardware_t, reduce L2_capacity to model cache
// pressure from fetcher writes, and call compute_tile_latency() exactly
// once.  Origami's full model runs with the modified inputs and produces
// a correctly-composed result.

double compute_constrained_tile_latency(const origami::problem_t&              problem,
                                        const origami::hardware_t&             hardware,
                                        const origami::config_t&               config,
                                        const origami::resource_constraints_t& constraints,
                                        std::size_t                            split_factor) {
  auto valid_cfg = ensure_valid_gemm_config(config, hardware, problem);

  origami::hardware_t hw_constrained(hardware);

  // Reduce L2 capacity to model cache pressure from fetcher staged_a writes.
  //
  // WHY: fetcher WGs write gathered A tiles into an HBM staging buffer.
  // These writes flow through L2 and evict GEMM's B-tile cache entries.
  // The effect: GEMM sees more L2 misses → more DRAM accesses → higher
  // memory latency.  By reducing L2_capacity in the hardware copy, we
  // cause Origami's compute_l2_hit_rate_global() to return a lower hit rate,
  // which propagates through the memory latency model correctly.
  //
  // The conversion (effective_l2_bytes / 1024.0) reverses the stored format:
  // effective_l2_bytes = L2_capacity × 1024 in unconstrained(), so dividing
  // by 1024 gives us back the hardware's internal unit.  Origami internally
  // computes: compute_l2_hit_rate_global(problem, hw, config, L2_capacity * 1024)
  // so the ratio is preserved correctly.
  if (constraints.effective_l2_bytes > 0.0 &&
      constraints.effective_l2_bytes < static_cast<double>(hardware.L2_capacity) * 1024.0) {
    hw_constrained.L2_capacity =
        static_cast<std::size_t>(constraints.effective_l2_bytes / 1024.0);
  }

  // Clamp CUs to at least 1 to avoid division by zero in Origami internals.
  auto cus = std::max(constraints.available_cus, std::size_t{1});

  // Origami's compute_tile_latency does all of this in one call:
  //   1. compute_mt_compute_latency: MI instruction count × per-MI cycles
  //   2. compute_memory_latency: L2 hit → MALL hit → DRAM, BW from occupancy
  //   3. max(compute, memory) × main_loop_efficiency × utilization_penalty
  //   4. prologue (memory latency × occupancy_factor)
  //   5. epilogue (output writes, K-split, K-padding)
  //   6. × heuristic weights (calibrated per-architecture)
  double tile_lat_cycles = origami::compute_tile_latency(
      problem, hw_constrained, valid_cfg, cus, split_factor);

  return tile_lat_cycles / (hardware.compute_clock_ghz * 1e3);
}

// ── Communication model ─────────────────────────────────────────────────

/// Link-limited all-gather time.
///
/// In the all-gather, each GPU receives data from (world_size - 1) peers.
/// The aggregate bandwidth across all incoming links is:
///   link_bw_gbps × (world_size - 1)
///
/// The total remote data volume is:
///   M × K_local × (world_size - 1) × dtype_bytes
/// because each peer sends one K_local slice, and there are (ws-1) peers.
///
/// Note: this is a lower bound.  The actual comm time is higher due to
/// gather_overhead, flag signaling, per-WG write BW limits, and the fact
/// that links aren't perfectly saturated in practice.
double compute_comm_time_ms(const all_gather_matmul_problem_t& problem,
                            const network_t&                   network) {
  double k_local       = static_cast<double>(problem.k_local());
  double dtype_bytes   = origami::data_type_to_bytes(problem.a_dtype);
  double total_remote  = static_cast<double>(problem.size.m) * k_local *
                         static_cast<double>(network.world_size - 1) * dtype_bytes;
  double total_link_bw = network.link_bw_gbps *
                         static_cast<double>(network.world_size - 1);
  // bytes / (GB/s × 1e9) = seconds; × 1e3 = milliseconds
  return total_remote / (total_link_bw * 1e9) * 1e3;
}

// ── Per-WG fetch timing ─────────────────────────────────────────────────
//
// Models the time for one fetcher WG to process all its assigned flag groups.
//
// PHYSICAL MODEL:
//   For each flag group, the fetcher must do three things sequentially:
//
//   1. XGMI gather: read remote data from peer GPU memory via XGMI links.
//      Only (world_size-1)/world_size of the data is remote (the local
//      slice is already in this GPU's HBM).
//      Time = remote_bytes / (link_bw × 1e3) × gather_overhead
//      The 1e3 converts GB/s to bytes/μs.  gather_overhead (1.5×) accounts
//      for XGMI protocol framing and descriptor setup.
//
//   2. HBM write: write the gathered data to the staging buffer in HBM.
//      This writes ALL data (local + remote) since the staging buffer is
//      a contiguous copy.
//      Time = total_bytes / (write_bw_per_wg × 1e3)
//
//   3. Flag store: release-atomic to signal that this flag group is ready.
//      Time = flag_store_us (calibrated constant, ~5 μs on MI300X)
//
//   The per-flag-group time is max(xgmi, write) + flag_store.
//   XGMI and HBM write can overlap (the GPU can pipeline DMA reads with
//   register-to-HBM writes), so we take the max, not the sum.

double compute_fetch_wg_time_us(std::size_t bm, std::size_t bk,
                                std::size_t kpf,
                                std::size_t world_size,
                                double link_bw_gbps,
                                double dtype_bytes,
                                std::size_t num_fgs_per_wg,
                                const network_t& network) {
  // Data volume per flag group: bm rows × (kpf × bk) columns × dtype_bytes.
  // kpf (k_per_flag) is the number of K-blocks per flag, so the K-extent
  // covered is kpf × bk elements.
  double bytes_per_fg  = static_cast<double>(bm * kpf * bk) * dtype_bytes;
  double remote_frac   = static_cast<double>(world_size - 1) / world_size;
  double remote_bytes  = bytes_per_fg * remote_frac;

  double xgmi_us  = remote_bytes / (link_bw_gbps * 1e3) * network.gather_overhead;
  double write_us = bytes_per_fg / (network.write_bw_per_wg_gbps * 1e3);

  double per_fg_us = std::max(xgmi_us, write_us) + network.flag_store_us;
  return num_fgs_per_wg * per_fg_us;
}

/// CU-work-queue kernel time estimate.
///
/// PHYSICAL MODEL:
///   Imagine the GPU as a pool of N_CU workers draining a FIFO of WGs.
///   The ideal completion time is total_work / N_CU.  In practice, the
///   actual time is scheduling_factor × ideal because:
///     - WG dispatch is not instantaneous (HW scheduler has finite throughput)
///     - Cross-XCD coherence on MI300X (8 XCDs with separate L2 domains add
///       latency for flag visibility across XCDs)
///     - Pipeline bubbles from flag-ordering constraints (GEMM WGs in stage
///       N+1 can't start until stage N's flags are set)
///     - Load imbalance (stage 0 has N_CU fetchers, later stages have fewer)
///
///   scheduling_factor = 4.5 was calibrated by measuring the ratio of actual
///   kernel wall time to this ideal lower bound across a range of MI300X
///   problem sizes.  It's the single largest source of prediction error and
///   is the most promising target for future model refinement.
double estimate_kernel_time_ms(std::size_t total_gemm_wgs,
                               double      gemm_wg_us,
                               std::size_t total_fetch_wgs,
                               double      fetch_wg_us,
                               std::size_t num_cus,
                               double      scheduling_factor) {
  double total_cu_work_us =
      static_cast<double>(total_gemm_wgs) * gemm_wg_us +
      static_cast<double>(total_fetch_wgs) * fetch_wg_us;

  double ideal_ms     = total_cu_work_us / num_cus / 1e3;
  double estimated_ms = ideal_ms * scheduling_factor;
  return estimated_ms;
}

// ── Wavefront model ─────────────────────────────────────────────────────
//
// PHYSICAL PICTURE:
//   Within each pipeline stage, GEMM WGs don't all start at once.  They are
//   gated by flag delivery: a GEMM WG can only start when its flag group has
//   been marked ready by a fetcher.  Fetchers work in parallel but each flag
//   group takes time, so flags arrive at a finite rate.
//
//   This creates a wavefront: GEMM concurrency ramps up from 0, reaches a
//   steady state, then drains as the last GEMM WGs finish after all flags
//   are delivered.  Each phase has different resource contention:
//
//     RAMP-UP: few GEMM WGs active → low L2 contention, low HBM BW demand
//     STEADY:  peak GEMM concurrency + fetchers still writing → high L2
//              pressure, reduced effective cache capacity
//     DRAIN:   fetchers done → GEMM gets all CUs and full cache capacity
//
//   Modeling this matters because a single "average" resource profile would
//   overestimate GEMM latency during ramp-up and drain (when resources are
//   less contended) and underestimate it during steady state.

stage_profile_t build_stage_profile(const origami::hardware_t& hardware,
                                    const network_t&           network,
                                    const all_gather_matmul_config_t& config,
                                    std::size_t gemm_tiles_per_stage,
                                    double      gemm_wg_time,
                                    double      per_flag_fetch_us,
                                    double      peak_hbm_bw_gbps) {
  // CUs available for GEMM = total CUs - fetcher CUs.
  std::size_t gemm_active_cus = hardware.N_CU - config.num_fetch_sms;
  gemm_active_cus = std::max(gemm_active_cus, std::size_t{1});

  // FLAG DELIVERY RATE (Little's Law):
  //   Each of the num_fetch_sms fetcher WGs processes flag groups sequentially.
  //   Each flag group takes per_flag_fetch_us microseconds.
  //   So the aggregate flag delivery rate is:
  //     flag_rate = num_fetch_sms / per_flag_fetch_us  [flags per microsecond]
  //
  //   EFFECTIVE CONCURRENCY:
  //   At equilibrium, the number of GEMM WGs that are simultaneously active is:
  //     eff_concurrent = min(flag_rate × gemm_wg_time, gemm_active_cus)
  //
  //   This is Little's Law: L = λ × W, where:
  //     L = average number in system (concurrent GEMM WGs)
  //     λ = arrival rate (flag delivery rate)
  //     W = service time (GEMM WG time)
  //
  //   If flags arrive faster than GEMM can consume them, concurrency saturates
  //   at gemm_active_cus.  If GEMM is fast, concurrency is limited by the
  //   fetch rate.
  double flag_rate = (per_flag_fetch_us > 0.0)
      ? static_cast<double>(config.num_fetch_sms) / per_flag_fetch_us
      : static_cast<double>(gemm_active_cus);

  double eff_concurrent = std::min(flag_rate * gemm_wg_time,
                                   static_cast<double>(gemm_active_cus));
  eff_concurrent = std::max(eff_concurrent, 1.0);

  // Total HBM write BW consumed by fetchers (aggregate across all fetch WGs).
  double fetch_write_bw = static_cast<double>(config.num_fetch_sms) *
                           network.write_bw_per_wg_gbps;

  double l2_capacity_bytes = static_cast<double>(hardware.L2_capacity) * 1024.0;

  // ── RAMP-UP PHASE ──
  //
  // Duration: the time for concurrency to grow from 0 to eff_concurrent.
  //   eff_concurrent / flag_rate = how long until enough flags have been
  //   delivered to sustain peak concurrency.
  //
  // Average CU count during ramp: eff_concurrent / 2 (linear ramp).
  //
  // L2 capacity: nearly full.  During ramp-up, few GEMM WGs are competing
  // for cache, so the L2 is mostly available for GEMM's working set.
  //
  // HBM BW: proportional to the ramp-average CU fraction, minus the BW
  // consumed by fetcher writes.
  origami::resource_constraints_t ramp_gemm;
  ramp_gemm.available_cus       = std::max(std::size_t{1},
                                           static_cast<std::size_t>(eff_concurrent / 2.0));
  ramp_gemm.available_hbm_bw_gbps = std::max(1.0,
      peak_hbm_bw_gbps * (eff_concurrent / 2.0) /
          static_cast<double>(hardware.N_CU) - fetch_write_bw);
  ramp_gemm.effective_l2_bytes   = l2_capacity_bytes;
  ramp_gemm.effective_mall_bytes = hardware.has_MALL()
      ? l2_capacity_bytes * 6.0 : 0.0;

  double ramp_up_duration = (flag_rate > 0) ? eff_concurrent / flag_rate : 0.0;

  origami::stage_phase_t ramp_up;
  ramp_up.duration_us      = ramp_up_duration;
  ramp_up.gemm_constraints = ramp_gemm;

  // ── STEADY PHASE ──
  //
  // Duration: time from ramp completion until all flag groups are delivered.
  //   total_fetch_duration = gemm_tiles_per_stage / flag_rate
  //   steady_duration = total_fetch_duration - ramp_up_duration
  //
  // CUs: full eff_concurrent GEMM WGs running.
  //
  // L2 capacity: REDUCED.  Fetcher WGs are actively writing staged_a tiles
  // into HBM, and these writes flow through L2.  This evicts GEMM's cached
  // B-tiles, effectively reducing L2 capacity.  We model this as:
  //   effective_L2 = L2_capacity × (1 - 0.5 × pressure_ratio)
  // where pressure_ratio = eff_concurrent / total_CUs.
  //
  // The 0.5 coefficient is a heuristic: not all L2 lines are B-tiles (some
  // hold A tiles being read by GEMM), and the write-through traffic doesn't
  // evict everything.  0.5 was chosen to roughly match measured L2 miss rates
  // from MI300X traces of the fused kernel.
  origami::resource_constraints_t steady_gemm;
  steady_gemm.available_cus       = static_cast<std::size_t>(eff_concurrent);
  steady_gemm.available_hbm_bw_gbps = std::max(1.0, peak_hbm_bw_gbps - fetch_write_bw);
  double l2_pressure_ratio = eff_concurrent / static_cast<double>(hardware.N_CU);
  steady_gemm.effective_l2_bytes  = l2_capacity_bytes * (1.0 - 0.5 * l2_pressure_ratio);
  steady_gemm.effective_mall_bytes = ramp_gemm.effective_mall_bytes;

  double total_fetch_duration = (flag_rate > 0)
      ? static_cast<double>(gemm_tiles_per_stage) / flag_rate
      : 0.0;
  double steady_duration = std::max(0.0, total_fetch_duration - ramp_up_duration);

  origami::stage_phase_t steady;
  steady.duration_us      = steady_duration;
  steady.gemm_constraints = steady_gemm;

  // ── DRAIN PHASE ──
  //
  // Duration: all flags have been delivered, but some GEMM WGs are still
  // computing.  The drain time is the excess GEMM work beyond what was
  // completed during fetch:
  //   total_gemm_work = tiles × gemm_wg_time / gemm_active_cus
  //   drain = max(0, total_gemm_work - total_fetch_duration)
  //
  // CUs: all gemm_active_cus are available (fetchers have exited).
  // L2: full capacity (no more fetcher write pressure).
  // HBM: full peak bandwidth.
  origami::resource_constraints_t drain_gemm;
  drain_gemm.available_cus        = gemm_active_cus;
  drain_gemm.available_hbm_bw_gbps = peak_hbm_bw_gbps;
  drain_gemm.effective_l2_bytes   = l2_capacity_bytes;
  drain_gemm.effective_mall_bytes = ramp_gemm.effective_mall_bytes;

  double total_gemm_work_us =
      static_cast<double>(gemm_tiles_per_stage) * gemm_wg_time /
      static_cast<double>(gemm_active_cus);
  double drain_duration = std::max(0.0, total_gemm_work_us - total_fetch_duration);

  origami::stage_phase_t drain;
  drain.duration_us      = drain_duration;
  drain.gemm_constraints = drain_gemm;

  stage_profile_t profile;
  profile.ramp_up                       = ramp_up;
  profile.steady                        = steady;
  profile.drain                         = drain;
  profile.effective_concurrent_gemm_wgs = eff_concurrent;
  return profile;
}

/// Time-weighted average GEMM tile latency across the three phases.
///
/// Each phase has a different resource constraint profile, so we call
/// compute_constrained_tile_latency() once per phase to get the tile
/// latency under that phase's specific CU count and L2 capacity.
///
/// The weighted average is: Σ(phase_latency × phase_duration) / total_duration.
///
/// This is better than using a single average constraint because the
/// relationship between CU count and latency is nonlinear (Origami's memory
/// BW model, L2 hit rate, and prologue/epilogue all scale non-trivially
/// with CU count).
double stage_gemm_latency_us(const stage_profile_t&     profile,
                             const origami::problem_t&  gemm_problem,
                             const origami::hardware_t& hardware,
                             const origami::config_t&   gemm_config) {
  double ramp_lat  = compute_constrained_tile_latency(
      gemm_problem, hardware, gemm_config,
      profile.ramp_up.gemm_constraints, 1);
  double steady_lat = compute_constrained_tile_latency(
      gemm_problem, hardware, gemm_config,
      profile.steady.gemm_constraints, 1);
  double drain_lat  = compute_constrained_tile_latency(
      gemm_problem, hardware, gemm_config,
      profile.drain.gemm_constraints, 1);

  double total_dur = profile.ramp_up.duration_us +
                     profile.steady.duration_us +
                     profile.drain.duration_us;
  if (total_dur <= 0.0)
    return steady_lat;

  return (ramp_lat  * profile.ramp_up.duration_us +
          steady_lat * profile.steady.duration_us +
          drain_lat  * profile.drain.duration_us) / total_dur;
}

// ── predict_latency ─────────────────────────────────────────────────────
//
// This is the main prediction function.  Given a fully-specified config,
// it computes:
//   1. Per-WG GEMM time (from Origami's detailed model + flag poll overhead)
//   2. Per-WG fetch time (from XGMI + HBM write BW model)
//   3. Wavefront phase profile (ramp/steady/drain resource constraints)
//   4. CU-work-queue kernel time estimate
//   5. Pipeline estimate (alternative model for cross-checking)
//
// The primary output is est_kernel_ms (CU-work-queue model).  The pipeline
// estimate is included for comparison and sanity-checking.

all_gather_matmul_prediction_result_t predict_latency(
    const all_gather_matmul_problem_t& problem,
    const origami::hardware_t&         hardware,
    const network_t&                   network,
    const all_gather_matmul_config_t&  config) {

  const std::size_t M = problem.size.m;
  const std::size_t N = problem.size.n;
  const std::size_t K = problem.size.k;
  const std::size_t ws = problem.world_size;
  const std::size_t K_local = problem.k_local();
  const double dtype_bytes = origami::data_type_to_bytes(problem.a_dtype);
  const std::size_t num_cus = hardware.N_CU;

  const std::size_t bm = config.mt.m;
  const std::size_t bn = config.mt.n;
  const std::size_t bk = config.mt.k;

  const std::size_t num_m_tiles = (M + bm - 1) / bm;
  const std::size_t num_tiles_n = (N + bn - 1) / bn;
  const std::size_t num_k_blocks = K / bk;

  const std::size_t kpf = config.k_per_flag;
  const std::size_t num_flag_groups_k =
      num_k_blocks / std::max(kpf, std::size_t{1});

  origami::problem_t gemm_sub = make_gemm_subproblem(problem);

  // ── 1. Communication time ─────────────────────────────────────────
  // Lower bound: total remote bytes / aggregate link bandwidth.
  double comm_time_ms = compute_comm_time_ms(problem, network);

  // ── 2. GEMM timing from Origami ───────────────────────────────────
  //
  // gemm_wg_us: time for one GEMM WG to compute its output tile.
  //   Uses Origami's compute_tile_latency which accounts for MI instruction
  //   latency, L2/MALL/DRAM memory hierarchy, BW-from-occupancy, utilization
  //   penalties, and prologue/epilogue.  Plus flag-poll overhead.
  //
  // compute_time_ms: standalone GEMM time (no comm) for reference.
  //   Used to compute the comm/compute ratio and effective TFLOPS.
  //
  // effective_tflops: actual throughput predicted by Origami, NOT the
  //   simple roofline ceiling.  Accounts for all memory hierarchy and
  //   scheduling overheads.
  double gemm_wg_us = origami_gemm_wg_time_us(
      gemm_sub, hardware, config, num_cus,
      num_flag_groups_k, network.flag_poll_us);

  double compute_time_ms = origami_total_gemm_ms(
      gemm_sub, hardware, config);

  double total_flops = 2.0 * M * N * K;
  double effective_tflops = (compute_time_ms > 0)
      ? total_flops / (compute_time_ms * 1e-3) / 1e12
      : 0.0;

  // ── 3. Pipeline geometry ──────────────────────────────────────────
  //
  // The M dimension is divided into num_stages horizontal slices.
  // Each stage has m_per_stage M-tiles and gemm_tiles_per_stage total tiles.
  std::size_t num_stages   = config.num_fetch_stages;
  std::size_t m_per_stage  = (std::size_t)std::ceil(
      static_cast<double>(num_m_tiles) / num_stages);
  std::size_t gemm_tiles_per_stage = m_per_stage * num_tiles_n;

  // ── 4. Fetch WG times ─────────────────────────────────────────────
  //
  // Stage 0 uses first_stage_fetch_sms (typically all CUs) to minimize
  // startup latency.  Later stages use num_fetch_sms.  The per-WG work
  // is inversely proportional to the number of fetchers.
  std::size_t total_fg_per_stage = num_flag_groups_k * m_per_stage;
  std::size_t fsf = config.first_stage_fetch_sms;
  std::size_t nf  = config.num_fetch_sms;

  std::size_t fgs_per_wg_stg0 = std::max(std::size_t{1},
      (total_fg_per_stage + fsf - 1) / fsf);
  std::size_t fgs_per_wg_rest = std::max(std::size_t{1},
      (total_fg_per_stage + nf - 1) / nf);

  double fetch_us_stg0 = compute_fetch_wg_time_us(
      bm, bk, kpf, ws, network.link_bw_gbps, dtype_bytes,
      fgs_per_wg_stg0, network);
  double fetch_us_rest = compute_fetch_wg_time_us(
      bm, bk, kpf, ws, network.link_bw_gbps, dtype_bytes,
      fgs_per_wg_rest, network);

  // ── 5. Per-flag-group time for wavefront model ────────────────────
  // This is the time for a single fetcher to process a single flag group.
  // It drives the flag delivery rate in build_stage_profile.
  double per_fg_bytes   = static_cast<double>(bm * kpf * bk) * dtype_bytes;
  double per_fg_remote  = per_fg_bytes * static_cast<double>(ws - 1) / ws;
  double per_fg_xgmi_us = per_fg_remote / (network.link_bw_gbps * 1e3) *
                           network.gather_overhead;
  double per_fg_write_us = per_fg_bytes / (network.write_bw_per_wg_gbps * 1e3);
  double per_fg_us       = std::max(per_fg_xgmi_us, per_fg_write_us) +
                            network.flag_store_us;

  // ── 6. Wavefront model ────────────────────────────────────────────
  // Build the three-phase resource profile and compute the time-weighted
  // average tile latency under the dynamic resource constraints.
  //
  // peak_hbm_bw_gbps comes from network_t (populated by for_architecture),
  // used to partition bandwidth between fetcher and GEMM WGs in the phase
  // constraints.
  double peak_hbm_bw_gbps = network.peak_hbm_bw_gbps;
  stage_profile_t profile = build_stage_profile(
      hardware, network, config, gemm_tiles_per_stage,
      gemm_wg_us, per_fg_us, peak_hbm_bw_gbps);

  double weighted_gemm_lat_us = stage_gemm_latency_us(
      profile, gemm_sub, hardware, config);

  // ── 7. Grid geometry ──────────────────────────────────────────────
  // Total WGs in the kernel launch grid.  Stage 0 uses fsf fetchers,
  // all other stages use nf.  Each stage has gemm_tiles_per_stage GEMM WGs.
  std::size_t first_stage_size = fsf + gemm_tiles_per_stage;
  std::size_t rest_stage_size  = nf + gemm_tiles_per_stage;
  std::size_t rest_count       = (num_stages > 1) ? (num_stages - 1) : 0;
  std::size_t grid_size        = first_stage_size + rest_stage_size * rest_count;
  std::size_t total_fetch_wgs  = fsf + nf * rest_count;
  std::size_t total_gemm_wgs   = gemm_tiles_per_stage * num_stages;

  // ── 8. Kernel time estimate ───────────────────────────────────────
  // CU-work-queue model: total WG microseconds / N_CU × scheduling_factor.
  // Uses weighted-average fetch time across stage 0 and later stages.
  double avg_fetch_us_num = static_cast<double>(fsf) * fetch_us_stg0 +
                            static_cast<double>(nf * rest_count) * fetch_us_rest;
  double avg_fetch_us = avg_fetch_us_num /
                        std::max(static_cast<double>(total_fetch_wgs), 1.0);

  double est_kernel_ms = estimate_kernel_time_ms(
      total_gemm_wgs, gemm_wg_us, total_fetch_wgs, avg_fetch_us,
      num_cus, network.scheduling_factor);

  // ── 9. Pipeline estimate (alternative model) ──────────────────────
  // Simple pipeline model for cross-checking:
  //   startup  = one stage of comm (no overlapping compute)
  //   steady   = max(comm, compute) per stage × (stages - 1)
  //   drain    = one stage of compute (no overlapping comm)
  //
  // stage_compute_ms is derived from Origami's total GEMM time, scaled by
  // the fraction of M-tiles in this stage.  This is more accurate than the
  // old approach of recomputing from the simple roofline.
  double total_link_bw = network.link_bw_gbps *
                         static_cast<double>(ws - 1);
  double stage_m = static_cast<double>(m_per_stage * bm);
  double stage_comm_ms = stage_m * K_local * static_cast<double>(ws - 1) *
                         dtype_bytes / (total_link_bw * 1e9) * 1e3;
  double stage_frac = (num_m_tiles > 0)
      ? static_cast<double>(m_per_stage) / static_cast<double>(num_m_tiles)
      : 1.0;
  double stage_compute_ms = compute_time_ms * stage_frac;
  double startup_ms  = stage_comm_ms;
  double steady_ms   = std::max(stage_comm_ms, stage_compute_ms) *
                       std::max(std::size_t{0}, num_stages - 1);
  double drain_ms    = stage_compute_ms;
  double pipeline_ms = startup_ms + steady_ms + drain_ms;

  double ratio = (compute_time_ms > 0) ? comm_time_ms / compute_time_ms : 999.0;

  // ── Assemble result ───────────────────────────────────────────────
  all_gather_matmul_prediction_result_t result;
  result.total_latency_ms   = est_kernel_ms;
  result.comm_time_ms       = comm_time_ms;
  result.compute_time_ms    = compute_time_ms;
  result.pipeline_ms        = pipeline_ms;
  result.est_kernel_ms      = est_kernel_ms;
  result.roofline_tflops    = effective_tflops;
  result.comm_compute_ratio = ratio;
  result.config             = config;
  result.grid_size          = grid_size;
  result.total_fetch_wgs    = total_fetch_wgs;
  result.total_gemm_wgs     = total_gemm_wgs;

  return result;
}

// ── select_config ───────────────────────────────────────────────────────
//
// Search-based parameter selection.  Instead of heuristic choosers, we
// enumerate valid candidate configurations and evaluate each through
// predict_latency() — which calls Origami's full GEMM model, the wavefront
// model, and the CU-work-queue estimator.  The candidate with the lowest
// predicted kernel time wins.
//
// The search space is kept manageable by only considering:
//   - Tile sizes from a small set of MFMA-friendly values
//   - k_per_flag values that evenly divide the K-block count
//   - Pipeline depths as powers of 2
//   - Fetcher CU counts as a geometric sweep
//
// Total candidates are typically O(100–500), each evaluated in ~1 μs of
// host time, so the full search completes in well under 1 ms.

/// Generate valid bk values that divide both K and K_local.
static std::vector<std::size_t> generate_valid_bk(std::size_t K,
                                                  std::size_t K_local) {
  std::vector<std::size_t> result;
  for (std::size_t bk : {64, 32, 16}) {
    if (K % bk == 0 && K_local % bk == 0) {
      result.push_back(bk);
    }
  }
  if (result.empty()) result.push_back(16);
  return result;
}

/// Generate valid k_per_flag values for a given num_k_blocks.
static std::vector<std::size_t> generate_valid_kpf(std::size_t num_k_blocks,
                                                   std::size_t num_k_blocks_local) {
  std::vector<std::size_t> result;

  // K_local-aligned candidate (preferred: avoids partial-flag edge cases).
  if (num_k_blocks_local > 0 && num_k_blocks % num_k_blocks_local == 0) {
    result.push_back(num_k_blocks_local);
  }

  // Divisor-based candidates targeting different flag group counts.
  for (std::size_t target : {4, 8, 16, 32}) {
    if (num_k_blocks >= target) {
      std::size_t kpf = num_k_blocks / target;
      if (kpf > 0 && num_k_blocks % kpf == 0) {
        result.push_back(kpf);
      }
    }
  }

  // Also try kpf = 1 (finest granularity).
  result.push_back(1);

  // Deduplicate.
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

all_gather_matmul_prediction_result_t select_config(
    const all_gather_matmul_problem_t& problem,
    const origami::hardware_t&         hardware,
    const network_t&                   network) {

  const std::size_t M = problem.size.m;
  const std::size_t N = problem.size.n;
  const std::size_t K = problem.size.k;
  const std::size_t K_local = problem.k_local();
  const std::size_t num_cus = hardware.N_CU;

  auto mi = hardware.get_recommended_matrix_instruction(problem.mi_dtype);
  auto valid_bks = generate_valid_bk(K, K_local);

  all_gather_matmul_prediction_result_t best;
  best.total_latency_ms = std::numeric_limits<double>::max();

  static constexpr std::size_t tile_dims[]  = {64, 128, 256};
  static constexpr std::size_t warp_opts[]  = {4, 8};
  static constexpr std::size_t gm_opts[]    = {1, 2, 4, 8};

  for (std::size_t bm : tile_dims) {
    if (bm > M) continue;

    for (std::size_t bn : tile_dims) {
      if (bn > N) continue;

      for (std::size_t bk : valid_bks) {
        std::size_t num_k_blocks = K / bk;
        std::size_t num_k_blocks_local = K_local / bk;
        std::size_t num_m_tiles = (M + bm - 1) / bm;
        std::size_t num_n_tiles = (N + bn - 1) / bn;

        auto valid_kpfs = generate_valid_kpf(num_k_blocks, num_k_blocks_local);

        for (std::size_t kpf : valid_kpfs) {
          for (std::size_t gm : gm_opts) {

            std::size_t min_tiles_per_stage = std::max(num_cus / 4, std::size_t{16});

            for (std::size_t ns = 1; ns <= 32; ns *= 2) {
              std::size_t m_per_stage = (num_m_tiles + ns - 1) / ns;
              std::size_t tiles_per_stage = m_per_stage * num_n_tiles;
              if (tiles_per_stage < min_tiles_per_stage && ns > 1) continue;

              if (gm > 1 && m_per_stage % gm != 0) {
                m_per_stage = ((m_per_stage + gm - 1) / gm) * gm;
              }
              std::size_t actual_stages = std::max(std::size_t{1},
                  (num_m_tiles + m_per_stage - 1) / m_per_stage);

              std::size_t max_fetch = std::max(num_cus / 2, std::size_t{1});
              for (std::size_t nf = 32; nf <= max_fetch; nf = std::max(nf + nf / 2, nf + 16)) {
                for (std::size_t nw : warp_opts) {

                  all_gather_matmul_config_t cfg;
                  cfg.mt = {bm, bn, bk};
                  cfg.mi = mi;
                  cfg.occupancy             = 1;
                  cfg.num_fetch_sms         = nf;
                  cfg.k_per_flag            = kpf;
                  cfg.num_fetch_stages      = actual_stages;
                  cfg.first_stage_fetch_sms = num_cus;
                  cfg.num_warps             = nw;
                  cfg.group_size_m          = gm;

                  auto result = predict_latency(problem, hardware, network, cfg);
                  if (result.total_latency_ms < best.total_latency_ms) {
                    best = result;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return best;
}

// ── rank_configs ────────────────────────────────────────────────────────

std::vector<all_gather_matmul_prediction_result_t> rank_configs(
    const all_gather_matmul_problem_t&              problem,
    const origami::hardware_t&                      hardware,
    const network_t&                                network,
    const std::vector<all_gather_matmul_config_t>&  configs) {

  std::vector<all_gather_matmul_prediction_result_t> results;
  results.reserve(configs.size());

  for (const auto& cfg : configs) {
    results.push_back(predict_latency(problem, hardware, network, cfg));
  }

  std::sort(results.begin(), results.end(),
            [](const auto& a, const auto& b) {
              return a.total_latency_ms < b.total_latency_ms;
            });

  return results;
}

}  // namespace origami::algorithms::all_gather_gemm
