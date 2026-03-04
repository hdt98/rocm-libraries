/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025-2026 AMD ROCm(TM) Software
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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "origami/hardware.hpp"
#include "origami/types.hpp"

// ═══════════════════════════════════════════════════════════════════════════════
// Shared resource-constraint types (origami:: namespace)
//
// These live in origami:: rather than the algorithm-specific namespace because
// the idea of co-running workloads sharing a GPU's resources is general.  Any
// future fused algorithm (e.g. reduce-scatter + GEMM) can reuse these types
// to express how its workloads partition CUs, bandwidth, and cache.
// ═══════════════════════════════════════════════════════════════════════════════

namespace origami {

/**
 * @brief Resource budget visible to one side of a co-running algorithm pair.
 *
 * When two workload types share a GPU (e.g. GEMM WGs + communication fetcher
 * WGs in the fused all-gather kernel), each side sees a reduced resource
 * budget.  The analytical model needs to know these budgets to predict
 * performance under contention.
 *
 * The constraint adapter (compute_constrained_tile_latency) uses these values
 * to modify the inputs to Origami's existing GEMM model:
 *   - available_cus → passed as num_active_cus to compute_tile_latency()
 *   - effective_l2_bytes → used to shrink hardware.L2_capacity in a copy
 *
 * available_hbm_bw_gbps and effective_mall_bytes are populated for analysis /
 * debugging but are not currently consumed by the constraint adapter because
 * Origami's memory model derives bandwidth internally from CU count (via
 * compute_mem_bw_from_occupancy).
 */
struct resource_constraints_t {
  /// How many CUs are available for this workload.  In the fused kernel, GEMM
  /// gets (N_CU - num_fetch_sms) CUs, but during ramp-up only a fraction are
  /// actively computing because flags haven't arrived yet.
  std::size_t available_cus        = 0;

  /// Informational: estimated HBM read bandwidth share in GB/s.  Not
  /// consumed by the constraint adapter (Origami derives BW from CU count).
  double      available_hbm_bw_gbps = 0.0;

  /// Effective L2 capacity in bytes after accounting for cache pressure.
  /// In the fused kernel, fetcher WGs write staged_a tiles into HBM and these
  /// writes flow through L2, evicting GEMM's B-tile cache entries.  Reducing
  /// this value causes Origami's L2 hit-rate model to predict more HBM misses,
  /// increasing memory latency.
  double      effective_l2_bytes   = 0.0;

  /// Effective MALL capacity in bytes.  Currently informational -- the
  /// constraint adapter does not modify MALL in the hardware copy because
  /// Origami's estimate_mall_hit() derives capacity from hardware internals.
  double      effective_mall_bytes = 0.0;

  /// Returns constraints representing no contention (all resources available).
  static resource_constraints_t unconstrained(const hardware_t& hw);
};

/**
 * @brief One phase within a pipeline stage's wavefront lifecycle.
 *
 * Each pipeline stage goes through three phases that have different resource
 * profiles.  Modeling them separately lets us capture the performance impact
 * of the staggered GEMM WG activation pattern observed in kernel traces.
 *
 * Phases:
 *   ramp_up : flags are arriving, GEMM concurrency is growing from 0 → peak
 *   steady  : peak concurrent GEMM WGs, fetchers still running
 *   drain   : all flags delivered, GEMM WGs finish with full GPU resources
 */
struct stage_phase_t {
  double               duration_us      = 0.0;
  resource_constraints_t gemm_constraints;   ///< Resources available to GEMM during this phase
  resource_constraints_t comm_constraints;   ///< Resources available to comm (currently unused)
};

/**
 * @brief Full profile of a single pipeline stage (three phases).
 *
 * effective_concurrent_gemm_wgs is derived from Little's Law:
 *   eff_concurrent = min(flag_delivery_rate × gemm_wg_time, gemm_active_cus)
 *
 * This captures the equilibrium: if flags arrive faster than GEMM can consume
 * them, concurrency saturates at the available CU count; if GEMM is faster
 * than flag delivery, concurrency is limited by the fetch rate.
 */
struct stage_profile_t {
  stage_phase_t ramp_up;
  stage_phase_t steady;
  stage_phase_t drain;
  double        effective_concurrent_gemm_wgs = 0.0;
};

}  // namespace origami

// ═══════════════════════════════════════════════════════════════════════════════
// Fused all-gather + GEMM analytical model
//
// Namespace: origami::algorithms::all_gather_gemm
//
// This models the Iris HBM-buffered fused kernel where:
//   1. Fetcher WGs gather remote A tiles via XGMI into an HBM staging buffer,
//      setting per-tile ready flags (atomics) as each chunk arrives.
//   2. GEMM WGs poll these flags, then compute C += A_staged @ B tile by tile.
//   3. The M dimension is split into pipeline stages so that stage N+1's fetch
//      overlaps with stage N's compute.
//
// The analytical model predicts kernel wall time by composing:
//   - Per-WG GEMM latency from Origami's detailed GEMM model (instruction-
//     level compute, L2/MALL/DRAM memory hierarchy, BW-from-occupancy)
//   - Per-WG fetch latency from an XGMI + HBM write bandwidth model
//   - A wavefront concurrency model that captures the staggered activation of
//     GEMM WGs within each pipeline stage (ramp-up / steady / drain phases)
//   - A CU-work-queue model that estimates total kernel time from the aggregate
//     workgroup-microseconds divided by CU count, scaled by a calibrated
//     scheduling factor
// ═══════════════════════════════════════════════════════════════════════════════

namespace origami::algorithms::all_gather_gemm {

/**
 * @brief Network / interconnect description for multi-GPU collectives.
 *
 * This is a generic container — no architecture-specific defaults are baked
 * into the member initializers.  Use one of the factory functions to get
 * calibrated values for a specific GPU:
 *
 *   auto net = network_t::for_architecture(hardware_t::architecture_t::gfx942,
 *                                          50.0, 8);
 *
 * Or the convenience alias:
 *
 *   auto net = network_t::mi300x_defaults(50.0, 8);
 *
 * The two topology fields (link_bw_gbps, world_size) describe the physical
 * deployment and must always be supplied by the caller.  The remaining
 * calibrated fields (peak_hbm_bw_gbps, scheduling_factor, gather_overhead,
 * write_bw_per_wg_gbps, flag_poll_us, flag_store_us) are architecture-specific
 * and are filled in by the factory based on hardware specs and kernel trace
 * measurements.
 */
struct network_t {
  // ── Topology (caller-provided) ──────────────────────────────────────

  /// Per-link interconnect bandwidth in GB/s.  For XGMI (MI300X) this is
  /// ~50 GB/s per link.  For other interconnects (PCIe, NVLink) the value
  /// will differ.
  double      link_bw_gbps         = 0.0;

  /// Number of GPUs participating in the all-gather.
  std::size_t world_size           = 1;

  // ── Calibrated parameters (architecture-specific) ───────────────────

  /// Peak HBM bandwidth in GB/s for this GPU architecture.  Used by the
  /// wavefront model to partition bandwidth between fetcher and GEMM WGs
  /// in the stage resource constraints.  Populated by the factory from
  /// published hardware specs (e.g. 5300 GB/s for MI300X, ~3200 GB/s for
  /// MI250X).
  double      peak_hbm_bw_gbps     = 0.0;

  /// Ratio of measured kernel wall time to the CU-work-queue lower bound.
  /// Captures effects not modeled explicitly: WG dispatch latency, cross-XCD
  /// coherence overhead, pipeline bubbles from flag-ordering constraints, and
  /// scheduler inefficiency.
  double      scheduling_factor    = 0.0;

  /// Multiplier on the raw interconnect transfer time to account for gather
  /// protocol overhead (descriptor setup, request/response framing, etc.).
  double      gather_overhead      = 0.0;

  /// Sustained HBM write throughput per fetcher WG in GB/s.  Each fetcher WG
  /// writes gathered data from its registers to the staging buffer in HBM.
  double      write_bw_per_wg_gbps = 0.0;

  /// Time in microseconds for a GEMM WG to poll and acquire one ready flag
  /// via acquire-semantics atomic load.  Includes the atomic latency plus
  /// any spin-wait time when the flag hasn't been set yet.
  double      flag_poll_us         = 0.0;

  /// Time in microseconds for a fetcher WG to store one ready flag via
  /// release-semantics atomic store after completing the data write.
  double      flag_store_us        = 0.0;

  // ── Factory functions ───────────────────────────────────────────────

  /// Build a network_t with calibrated parameters for the given architecture.
  /// Currently supports gfx942 and gfx950 (MI300X-class).  Unsupported
  /// architectures get the same defaults as gfx942 (best available) with a
  /// warning-level fallback — callers should recalibrate for their hardware.
  static network_t for_architecture(hardware_t::architecture_t arch,
                                    double link_bw_gbps,
                                    std::size_t world_size);

  /// Convenience alias: equivalent to for_architecture(gfx942, ...).
  static network_t mi300x_defaults(double link_bw = 50.0,
                                   std::size_t world_size = 8);
};

/**
 * @brief Problem description for a fused all-gather + GEMM.
 *
 * The GEMM computes C[M, N] = all_gather(A)[M, K] @ B[K, N].
 *
 * A is distributed across world_size GPUs along the K dimension: each GPU
 * holds A_local[M, K/world_size].  The kernel gathers all slices into a
 * staging buffer and fuses the gather with the matmul.
 *
 * Inherits the standard Origami problem_t so we can pass it directly to
 * the GEMM model functions (compute_tile_latency etc.) after stripping
 * the comm-specific fields.
 */
struct all_gather_matmul_problem_t : problem_t {
  std::size_t world_size  = 8;
  double      link_bw_gbps = 50.0;

  /// Per-GPU K dimension.  The full K = k_local * world_size.
  std::size_t k_local() const { return size.k / world_size; }
};

/**
 * @brief Configuration for the HBM-buffered all-gather + GEMM kernel.
 *
 * Extends the base GEMM config (tile sizes, MI dimensions, occupancy) with
 * fused-kernel-specific staging and pipeline parameters.  All of these are
 * tunable knobs that trade off communication and compute balance.
 */
struct all_gather_matmul_config_t : config_t {
  /// Number of CUs dedicated to communication (fetching remote A tiles).
  /// The remaining (N_CU - num_fetch_sms) CUs run GEMM WGs.
  /// More fetch SMs → faster data delivery → GEMM WGs start sooner,
  /// but fewer CUs for GEMM → each GEMM wave is slower.
  std::size_t num_fetch_sms         = 64;

  /// Number of K-blocks per ready flag.  Controls the granularity of the
  /// GEMM/comm overlap.  Smaller values mean flags are more frequent (more
  /// overlap opportunities but more flag polling overhead); larger values
  /// mean coarser signaling (less overhead but less overlap).
  std::size_t k_per_flag            = 64;

  /// Number of pipeline stages along the M dimension.  Each stage fetches
  /// and computes a horizontal slice of the output.  More stages increase
  /// comm/compute overlap but add pipeline startup/drain overhead and
  /// reduce per-stage tile count (which can hurt GPU occupancy).
  std::size_t num_fetch_stages      = 8;

  /// Number of fetcher WGs for stage 0.  Stage 0 has no prior compute to
  /// overlap with, so it typically uses all CUs as fetchers to minimize
  /// startup latency.
  std::size_t first_stage_fetch_sms = 304;

  /// Warps per GEMM WG.  Larger tiles (256×256) need 8 warps to saturate
  /// the MFMA units; smaller tiles (128×128) use 4.
  std::size_t num_warps             = 8;

  /// Number of M-tiles grouped together for workgroup mapping.  Affects
  /// L2 locality: tiles in the same group share B-tile cache lines.
  std::size_t group_size_m          = 4;

  /// Layout of the staged A buffer in HBM.  "k_contiguous" means K is the
  /// fastest-varying dimension, matching the natural MFMA load pattern.
  std::string staged_a_layout       = "k_contiguous";
};

/**
 * @brief Prediction result for a fused all-gather + GEMM configuration.
 *
 * Contains the predicted performance metrics and the evaluated config.
 */
struct all_gather_matmul_prediction_result_t {
  /// Primary output: predicted total kernel wall time in milliseconds.
  /// This is the CU-work-queue estimate (est_kernel_ms).
  double total_latency_ms    = 0.0;

  /// Time to gather all remote A tiles assuming link-limited bandwidth.
  /// comm_time_ms = remote_bytes / total_link_bw.  This is the theoretical
  /// minimum comm time if all links were fully saturated with zero overhead.
  double comm_time_ms        = 0.0;

  /// Time for a standalone GEMM of the same M×N×K, predicted by Origami's
  /// detailed model.  Used to compute the comm/compute ratio.
  double compute_time_ms     = 0.0;

  /// Alternative estimate using a simple pipeline model:
  /// startup + max(comm, compute) × (stages - 1) + drain.
  /// Useful as a cross-check against the CU-work-queue estimate.
  double pipeline_ms         = 0.0;

  /// CU-work-queue estimate: (total_wg_work / num_cus) × scheduling_factor.
  /// Same value as total_latency_ms (kept for explicit API clarity).
  double est_kernel_ms       = 0.0;

  /// Effective TFLOPS: total_flops / compute_time_ms.  This is NOT the
  /// simple roofline ceiling -- it's the actual throughput predicted by
  /// Origami's model including all memory hierarchy and overhead effects.
  double roofline_tflops     = 0.0;

  /// comm_time_ms / compute_time_ms.  Values > 1 mean communication is the
  /// bottleneck; values < 1 mean compute dominates.  The pipeline model
  /// uses more stages when this ratio is high to increase overlap.
  double comm_compute_ratio  = 0.0;

  /// The config that was evaluated (echoed back for convenience).
  all_gather_matmul_config_t config;

  /// Total WGs in the kernel launch grid (fetch + GEMM, all stages).
  std::size_t grid_size       = 0;
  std::size_t total_fetch_wgs = 0;
  std::size_t total_gemm_wgs  = 0;
};

// ── Constraint adapter ──────────────────────────────────────────────────

/**
 * @brief Predict GEMM tile latency under resource constraints.
 *
 * This is the bridge between the wavefront model and Origami's GEMM model.
 * Rather than building a separate "constrained GEMM" model, we modify the
 * inputs to Origami's existing compute_tile_latency() — specifically the
 * CU count and L2 capacity — and let Origami's full model do the rest.
 *
 * Why this approach: Origami's model has ~170 lines of carefully calibrated
 * logic for prologue/epilogue, utilization penalties, memory hierarchy BW
 * scaling, and heuristic weights.  Any attempt to manually decompose and
 * recompose the result (e.g. separately scaling compute and memory latency)
 * would lose accuracy.
 *
 * @param constraints  Resource budget: CU count and effective L2 capacity.
 * @return Estimated tile latency in microseconds.
 */
double compute_constrained_tile_latency(const problem_t&             problem,
                                        const hardware_t&            hardware,
                                        const config_t&              config,
                                        const resource_constraints_t& constraints,
                                        std::size_t                  split_factor);

// ── Communication model ─────────────────────────────────────────────────

/**
 * @brief Theoretical minimum all-gather time assuming all links are saturated.
 *
 * Each GPU needs to receive (world_size - 1) slices of A from its peers.
 * Total remote bytes = M × K_local × (world_size - 1) × dtype_bytes.
 * Total available bandwidth = link_bw × (world_size - 1) links.
 *
 * This is a lower bound -- the actual comm time is higher due to gather
 * overhead, flag signaling, and per-WG write bandwidth limits.
 */
double compute_comm_time_ms(const all_gather_matmul_problem_t& problem,
                            const network_t&                   network);

/**
 * @brief Time for one fetcher WG to fetch all its assigned flag groups.
 *
 * Each flag group covers (bm × kpf × bk) elements of A.  For each group,
 * the fetcher must:
 *   1. Gather remote data via XGMI (limited by link_bw × gather_overhead)
 *   2. Write gathered data to HBM staging buffer (limited by write_bw_per_wg)
 *   3. Store the ready flag (atomic, costs flag_store_us)
 *
 * The per-group time is max(XGMI time, HBM write time) + flag store.
 * Total WG time is num_fgs_per_wg × per_group_time.
 */
double compute_fetch_wg_time_us(std::size_t bm, std::size_t bk,
                                std::size_t kpf,
                                std::size_t world_size,
                                double link_bw_gbps,
                                double dtype_bytes,
                                std::size_t num_fgs_per_wg,
                                const network_t& network);

/**
 * @brief CU-work-queue kernel time estimate.
 *
 * Models the GPU as a pool of CUs that drain a queue of WGs (both GEMM and
 * fetch).  The ideal time is total_wg_microseconds / num_cus.  The actual
 * time is higher by scheduling_factor (calibrated ~4.5× on MI300X) due to:
 *   - WG dispatch latency (the HW scheduler doesn't instantly fill all CUs)
 *   - Cross-XCD coherence (MI300X has 8 XCDs with separate L2 domains)
 *   - Pipeline bubbles (flag ordering constraints prevent perfect overlap)
 *   - Load imbalance (stage 0 vs later stages have different fetch WG counts)
 */
double estimate_kernel_time_ms(std::size_t total_gemm_wgs,
                               double      gemm_wg_us,
                               std::size_t total_fetch_wgs,
                               double      fetch_wg_us,
                               std::size_t num_cus,
                               double      scheduling_factor);

// ── Wavefront model ─────────────────────────────────────────────────────

/**
 * @brief Build a three-phase resource profile for one pipeline stage.
 *
 * Within each pipeline stage, GEMM WG activation is not instantaneous --
 * it is gated by flag delivery from fetchers.  Kernel traces show a
 * characteristic wavefront pattern:
 *
 *   Time ──────────────────────────────────────>
 *   CUs:  [fetch][fetch][fetch][fetch]...
 *         [     ][gemm1][gemm1][gemm1]...
 *         [     ][     ][gemm2][gemm2]...
 *         [     ][     ][     ][gemm3]...
 *         └─ramp─┘└──── steady ────┘└─drain─┘
 *
 * The three phases have different resource profiles:
 *   ramp_up : few GEMM WGs active, L2 pressure low, CU concurrency growing
 *   steady  : peak concurrency, fetchers still writing (L2 under pressure)
 *   drain   : fetchers done, GEMM gets all resources
 *
 * @param gemm_wg_time      Per-WG GEMM latency in microseconds (from Origami)
 * @param per_flag_fetch_us  Time for one fetcher to process one flag group
 * @param peak_hbm_bw_gbps  Peak HBM BW for phase constraint fields (informational)
 */
stage_profile_t build_stage_profile(const hardware_t& hardware,
                                    const network_t&  network,
                                    const all_gather_matmul_config_t& config,
                                    std::size_t gemm_tiles_per_stage,
                                    double      gemm_wg_time_us,
                                    double      per_flag_fetch_us,
                                    double      peak_hbm_bw_gbps);

/**
 * @brief Time-weighted average GEMM tile latency across all three phases.
 *
 * Each phase has a different duration and different resource constraints.
 * We evaluate Origami's GEMM model once per phase (with that phase's
 * constraints) and return the duration-weighted average.  This gives a
 * single "effective tile latency" that accounts for the dynamic resource
 * profile over the stage's lifetime.
 */
double stage_gemm_latency_us(const stage_profile_t& profile,
                             const problem_t&       gemm_problem,
                             const hardware_t&      hardware,
                             const config_t&        gemm_config);

// ── Top-level API ───────────────────────────────────────────────────────

/**
 * @brief Automatic parameter selection for the fused all-gather + GEMM kernel.
 *
 * Searches over every config parameter, evaluating each combination via
 * predict_latency() (which uses Origami's full GEMM model internally),
 * and returns the configuration with the lowest predicted kernel time.
 *
 * All config decisions are made by the search — no hardcoded heuristics:
 *   - block_m, block_n: {64, 128, 256}, filtered by problem dimensions
 *   - block_k: {16, 32, 64}, filtered to divide both K and K_local
 *   - k_per_flag: K_local-aligned + divisor-based candidates
 *   - group_size_m: {1, 2, 4, 8}
 *   - num_fetch_stages: powers of 2 from 1 to 32, filtered by min occupancy
 *   - num_fetch_sms: geometric sweep from 32 to N_CU/2
 *   - num_warps: {4, 8}
 *
 * first_stage_fetch_sms is always set to N_CU (stage 0 has no prior compute
 * to overlap with, so all CUs fetch for fastest startup).
 *
 * Each candidate is evaluated end-to-end through the wavefront model, so
 * the selection accounts for all interactions between parameters.
 */
all_gather_matmul_prediction_result_t select_config(
    const all_gather_matmul_problem_t& problem,
    const hardware_t&                  hardware,
    const network_t&                   network);

/**
 * @brief Evaluate a specific config.  Useful for what-if analysis.
 */
all_gather_matmul_prediction_result_t predict_latency(
    const all_gather_matmul_problem_t& problem,
    const hardware_t&                  hardware,
    const network_t&                   network,
    const all_gather_matmul_config_t&  config);

/**
 * @brief Evaluate and rank multiple configs by predicted latency (ascending).
 */
std::vector<all_gather_matmul_prediction_result_t> rank_configs(
    const all_gather_matmul_problem_t&              problem,
    const hardware_t&                               hardware,
    const network_t&                                network,
    const std::vector<all_gather_matmul_config_t>&  configs);

}  // namespace origami::algorithms::all_gather_gemm
