// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <iomanip>
#include <iostream>

#include "origami/gemm.hpp"
#include "origami/math.hpp"
#include "origami/origami.hpp"
#include "origami/streamk.hpp"
#include "origami/types.hpp"

namespace origami {

prediction_result_t select_config(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const std::vector<config_t>& configs) {
  // Use rank_configs to get configurations with latencies ranked by performance
  auto results = rank_configs(problem, hardware, configs);

  if (results.empty()) { throw std::runtime_error("No valid configs found."); }

  // Apply tie-breaking logic for configs with similar latency
  double best_latency = results.front().latency;
  size_t num_the_same = 0;

  double tiebreaker_tolerance = 0.01;

  // Count configs with similar latency (within 1%)
  for (const auto& r : results) {
    double diff = std::fabs(r.latency - best_latency);
    diff /= best_latency;
    if (diff < tiebreaker_tolerance)
      num_the_same++;
    else
      break;
  }

  if (num_the_same > 1) {
    // Apply tie-breaking using arithmetic intensity
    auto compute_arithmetic_intensity = [](const config_t& config) -> double {
      auto MT_M = config.mt.m;
      auto MT_N = config.mt.n;
      auto MT_K = config.mt.k;

      double flops          = static_cast<double>(2ull * MT_M * MT_N * MT_K);
      double memory_traffic = static_cast<double>(MT_M * MT_K + MT_N * MT_K + MT_M * MT_N);

      if (memory_traffic == 0.0) return 0.0;
      return flops / memory_traffic;
    };

    std::stable_sort(results.begin(),
                     results.begin() + num_the_same,
                     [&](const prediction_result_t& a, const prediction_result_t& b) {
                       double ai_a = compute_arithmetic_intensity(a.config);
                       double ai_b = compute_arithmetic_intensity(b.config);
                       return ai_a > ai_b;  // descending
                     });
  }

  return results.front();
}

/**
 * @brief Selects the best WGM (maximizing L2 hit rate) given fixed macro tile sizes.
 *
 * @param[in] problem Problem description (M, N, K, etc.)
 * @param[in] hardware Hardware characteristics
 * @param[in] mt Macro tile dimensions
 * @param[in] mi Matrix instruction dimensions
 * @param[in] wgms Candidate WGM values to try
 *
 * @return A pair: best predicted (l2_hit_rate, wgm).
 */
std::pair<double, size_t> select_workgroup_mapping(const problem_t& problem,
                                                   const hardware_t& hardware,
                                                   const dim3_t mt,
                                                   const dim3_t mi,
                                                   const std::vector<size_t>& wgms) {
  using WGMResult = std::pair<double, size_t>;  // (l2_hit_rate, WGM)

  config_t config{mt, mi};

  std::vector<WGMResult> valid_results;
  valid_results.reserve(wgms.size());

  // Iterate over all candidate WGM values
  for (const auto& candidate_wgm : wgms) {
    if (hardware_t::is_debug_enabled()) { std::cout << "Evaluating WGM=" << candidate_wgm << "\n"; }

    // Optionally ensure we do not exceed LDS capacity
    // (If you want to factor in WGM, add it to your check_lds_capacity signature.)
    // For now, let's just check the tile itself:
    if (!check_lds_capacity(hardware, mt, problem.a_dtype, problem.b_dtype)) {
      if (hardware_t::is_debug_enabled()) {
        std::cout << "Skipping WGM=" << candidate_wgm << " due to LDS capacity.\n";
      }
      continue;
    }

    config.workgroup_mapping = candidate_wgm;

    // Compute L2 hit rate for this WGM
    double current_hit = estimate_l2_hit(problem, hardware, config, 1 /* splittingFactor */);

    valid_results.emplace_back(current_hit, candidate_wgm);
  }

  // If no valid WGM was found, throw an error
  if (valid_results.empty()) { throw std::runtime_error("No valid WGM found."); }

  // Find the maximum L2 hit rate in valid_results
  // (Use max_element on the first value in the pair.)
  auto best_it = std::max_element(
      valid_results.begin(), valid_results.end(), [](const WGMResult& a, const WGMResult& b) {
        return a.first < b.first;  // "less" => a has smaller hit rate than b
      });

  double best_l2_hit = best_it->first;
  size_t best_wgm    = best_it->second;

  // Return (l2_hit_rate, WGM)
  return std::make_pair(best_l2_hit, best_wgm);
}

std::vector<prediction_result_t> rank_configs(const problem_t& problem,
                                              const hardware_t& hardware,
                                              const std::vector<config_t>& configs) {
  if (configs.empty()) { throw std::runtime_error("No configurations provided."); }

  std::vector<prediction_result_t> results(configs.size());

  std::transform(std::execution::seq,
                 configs.begin(),
                 configs.end(),
                 results.begin(),
                 [&](const config_t& config) -> prediction_result_t {
                   if (!check_lds_capacity(hardware, config.mt, problem.a_dtype, problem.b_dtype)) {
                     return {std::numeric_limits<double>::max(), config};
                   }
                   double latency = compute_total_latency(problem, hardware, config);
                   return {latency, config};
                 });

  results.erase(std::remove_if(results.begin(),
                               results.end(),
                               [](const prediction_result_t& p) {
                                 return p.latency == std::numeric_limits<double>::max();
                               }),
                results.end());

  std::stable_sort(results.begin(),
                   results.end(),
                   [](const prediction_result_t& a, const prediction_result_t& b) {
                     return a.latency < b.latency;
                   });

  return results;
}

prediction_result_t select_config_mnk(std::size_t M,
                                      std::size_t N,
                                      std::size_t K,
                                      const hardware_t& hardware,
                                      const std::vector<config_t>& configs) {
  // Create a default problem_t with the provided M, N, K and reasonable defaults
  problem_t problem;
  problem.size.m          = M;
  problem.size.n          = N;
  problem.size.k          = K;
  problem.batch           = 1;
  problem.a_transpose     = transpose_t::T;     // Default to T
  problem.b_transpose     = transpose_t::N;     // Default to N
  problem.a_dtype         = data_type_t::Half;  // Default to fp16
  problem.b_dtype         = data_type_t::Half;  // Default to fp16
  problem.c_dtype         = data_type_t::Half;  // Default to fp16
  problem.d_dtype         = data_type_t::Half;  // Default to fp16
  problem.mi_dtype        = data_type_t::Half;  // Default to fp16
  problem.a_mx_block_size = 0;                  // Default MX block size
  problem.b_mx_block_size = 0;                  // Default MX block size

  // Use the existing select_config function with the constructed problem
  return select_config(problem, hardware, configs);
}

std::vector<prediction_result_t> select_topk_configs(const problem_t& problem,
                                                     const hardware_t& hardware,
                                                     const std::vector<config_t>& configs,
                                                     std::size_t topk) {
  auto ranked_configs = rank_configs(problem, hardware, configs);

  // Return only the top K configurations
  std::vector<prediction_result_t> topk_configs;
  size_t count = std::min(topk, ranked_configs.size());
  topk_configs.reserve(count);
  for (size_t i = 0; i < count; ++i) { topk_configs.push_back(ranked_configs[i]); }
  return topk_configs;
}

double compute_perf_gflops(const hardware_t& hardware,
                           const problem_t& problem,
                           const double latency) {
  // Extract parameters from structured types
  size_t M     = problem.size.m;
  size_t N     = problem.size.n;
  size_t K     = problem.size.k;
  size_t batch = problem.batch;

  // Compute total FLOPs
  double total_FLOPs = 2.0 * M * N * K;  // For GEMM, each multiply-add is 2 FLOPs
  // Compute total time in seconds
  double cycles_per_second = hardware.compute_clock_ghz * 1e9;  // 1 GHz = 1e9 cycles per second

  double total_time_seconds = latency / cycles_per_second;

  // Compute performance in FLOPS
  double FLOPS = total_FLOPs / total_time_seconds;
  // Convert to GFLOPS
  double GFLOPS = FLOPS / 1e9;  // 1 TFLOP = 1e9 FLOPs
  return GFLOPS;
}
}  // namespace origami
