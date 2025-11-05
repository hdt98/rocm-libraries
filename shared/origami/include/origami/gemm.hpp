// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <vector>
#include "origami/hardware.hpp"
#include "origami/types.hpp"

namespace origami {

/**
 * @brief Struct to cache variables for a config_t for a specific problem_type_t.
 *
 */
struct config_cache_t {
  /// TF32 conversion overhead
  double L_cvt;

  /// latency for single macro-tile
  double mt_compute_latency;
  
  /// whether the config macrotile fits in LDS
  bool fits_lds_capacity;

  int a_bytes;
  int b_bytes;
  int d_bytes;

  std::size_t Ld_CU_bytes;
};

/**
 * @brief Struct storing intermediate variables.
 *
 */
struct origami_cache_t {
  std::size_t grid_m;
  std::size_t grid_n;
  std::size_t grid_k;

  std::size_t launched_m;
  std::size_t launched_n;
  std::size_t launched_k;

  std::size_t num_mts;

  std::size_t num_active_cus;
  std::size_t num_waves;
  std::size_t splitting_factor;
};

/**
 * @brief Struct used to define the type of problem for which a config can be optimized, 
 * so that certain things can be precomputed and cached for a config applied to a specific problem type.
 * 
 * This contains the architecture, as well as all info describing a problem,
 * except for the actual sizes m, n, k, batch.
 *
 */
struct problem_type_t {
  /// Transpose types (TT, TN, NT, TT.)
  transpose_t a_transpose;
  transpose_t b_transpose;

  /// Data types: A, B, C, D.
  data_type_t a_dtype;
  data_type_t b_dtype;
  data_type_t c_dtype;
  data_type_t d_dtype;

  /// Compute type.
  data_type_t mi_dtype;

  /// MX block size.
  std::size_t a_mx_block_size;
  std::size_t b_mx_block_size;

  hardware_t::architecture_t arch;

  bool operator==(const problem_type_t&) const = default;

  std::size_t hash() const {
    return std::hash<std::size_t>()(static_cast<std::size_t>(a_dtype));
  }
};

/**
 * @brief Return problem type from a problem, hardware
 * 
 */
problem_type_t get_problem_type(const problem_t& problem,
                                const hardware_t& hardware);

/**
 * @brief Create struct with precomputed variables.
 * 
 */
origami_cache_t create_origami_cache(const problem_t& problem,
                                     const hardware_t& hardware,
                                     const config_t& config);

/**
 * @brief Create struct with precomputed quantities from a config for a specific problem_type.
 * 
 */
config_cache_t create_config_cache(const problem_type_t problem_type,
                                   const hardware_t& hardware,
                                   const config_t& config);

/**
 * @brief Compute and return number of matrix-instructions.
 *
 * @param mt Macro-tile of the kernel.
 * @param mi Matrix-instruction of the kernel.
 * @return std::size_t number of matrix-instructions.
 */
std::size_t compute_number_matrix_instructions(dim3_t mt, dim3_t mi);

/**
 * @brief Compute TF32 conversion overhead.
 *
 * @param problem_type Problem description.
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return double Latency in cycles.
 */
double compute_cvt_overhead(const problem_type_t& problem_type,
                            const hardware_t& hardware,
                            const config_t& config);

/**
 * @brief Compute TF32 conversion overhead.
 *
 * @param problem_type Problem description.
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return double Latency in cycles.
 */
double compute_cvt_overhead_x1(const problem_type_t& problem_type,
                               const hardware_t& hardware,
                               const config_t& config);

/**
 * @brief Compute the latency to process a single macro-tile for the given problem and hardware.
 *
 * @param problem_type Problem description.
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return double Latency in cycles.
 */
double compute_mt_compute_latency(const problem_type_t& problem_type,
                                  const hardware_t& hardware,
                                  const config_t& config);

inline double compute_mt_compute_latency(const problem_t& problem,
                                         const hardware_t& hardware,
                                         const config_t& config) {
  return compute_mt_compute_latency(get_problem_type(problem, hardware), hardware, config);
}

/**
 * @brief Checks if the macro-tile fits in LDS (A/B loads.)
 *
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param mt Macro-tile for the kernel.
 * @param a_dtype Data type for matrix-A.
 * @param b_dtype Data type for matrix-B.
 * @return true if macro-tile fits in available LDS.
 * @return false if macro-tile does NOT fit in available LDS.
 */
bool check_lds_capacity(const hardware_t& hardware,
                        dim3_t mt,
                        data_type_t a_dtype,
                        data_type_t b_dtype);

// Computes total data loads per CU per MT from A and B
// Reads happen every MT, Writes happen every K-complete tile.
// size_t compute_cu_loads(size_t MT_M, size_t MT_N, size_t MT_K);

// Estimates the l2 hit-rate

/**
 * @brief A linear-estimation method for estimating L2-hitrate.
 *
 * @todo Parameterize this based on the space-filling curve algos.
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param origami_cache Intermediate Origami results.
 * @return double Predicted L2-hitrate.
 */
double estimate_l2_hit(const problem_t& problem,
                       const hardware_t& hardware,
                       const config_t& config,
                       const origami_cache_t& origami_cache);

inline double estimate_l2_hit(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              std::size_t splitting_factor) {
  auto cache = create_origami_cache(problem, hardware, config);
  cache.splitting_factor = splitting_factor;
  return estimate_l2_hit(problem, hardware, config, cache);
}

/**
 * @brief Estimate the MALL-hitrate (last-level cache.)
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param origami_cache Intermediate Origami results.
 * @return double Predicted MALL-hitrate.
 */
double estimate_mall_hit(const problem_t& problem,
                         const hardware_t& hardware,
                         const config_t& config,
                         const origami_cache_t& origami_cache);

inline double estimate_mall_hit(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                std::size_t num_active_cus,
                                std::size_t splitting_factor) {
  auto cache = create_origami_cache(problem, hardware, config);
  cache.num_active_cus = num_active_cus;
  cache.splitting_factor = splitting_factor;
  return estimate_mall_hit(problem, hardware, config, cache);
}

/**
 * @brief Determine the memory latency per MT_M x MT_N x MT_K Macro Tile (L_MT).
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param origami_cache Intermediate Origami results.
 * @param config_cache Precomputed values for config on specific problem_type_t.
 * @return double Latency in cycles.
 */
double compute_memory_latency(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              const origami_cache_t& origami_cache,
                              const config_cache_t& config_cache);

inline double compute_memory_latency(const problem_t& problem,
                                     const hardware_t& hardware,
                                     const config_t& config,
                                     std::size_t num_active_cus,
                                     std::size_t splitting_factor) {
  auto origami_cache = create_origami_cache(problem, hardware, config);
  origami_cache.num_active_cus = num_active_cus;
  origami_cache.splitting_factor = splitting_factor;
  auto config_cache = create_config_cache(get_problem_type(problem, hardware), hardware, config);
  return compute_memory_latency(problem, hardware, config, origami_cache, config_cache);
}

/**
 * @brief Computes the latency to compute a K-COMPLETE tile.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param origami_cache Intermediate Origami results.
 * @param config_cache Precomputed values for config on specific problem_type_t.
 * @return double Latency in cycles.
 */
double compute_tile_latency(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config,
                            const origami_cache_t& origami_cache,
                            const config_cache_t& config_cache);

inline double compute_tile_latency(const problem_t& problem,
                                   const hardware_t& hardware,
                                   const config_t& config,
                                   std::size_t num_active_cus,
                                   std::size_t splitting_factor) {
  auto origami_cache = create_origami_cache(problem, hardware, config);
  origami_cache.num_active_cus = num_active_cus;
  origami_cache.splitting_factor = splitting_factor;
  auto config_cache = create_config_cache(get_problem_type(problem, hardware), hardware, config);
  return compute_tile_latency(problem, hardware, config, origami_cache, config_cache);
}


// Computes the latency per K-complete MT wave.
// A wave is defined as : The time it takes for one CU to complete one K-complete output tile

/**
 * @brief Computes the latency per K-complete MT wave.
 * A wave is defined as the time it takes for one CU to complete one
 * K-complete output tile
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param origami_cache Intermediate Origami results.
 * @param config_cache Precomputed values for config on specific problem_type_t.
 * @return double Latency in cycles.
 */
double compute_timestep_latency(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                const origami_cache_t& origami_cache,
                                const config_cache_t& config_cache);
            
inline double compute_timestep_latency(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                std::size_t num_active_cus,
                                std::size_t splitting_factor) {
  auto origami_cache = create_origami_cache(problem, hardware, config);
  origami_cache.num_active_cus = num_active_cus;
  origami_cache.splitting_factor = splitting_factor;
  auto config_cache = create_config_cache(get_problem_type(problem, hardware), hardware, config);
  return compute_timestep_latency(problem, hardware, config, origami_cache, config_cache);
}

/**
 * @brief Compute the total latency of a gemm based on the latency of one wave multiplied by the
 * number of waves A wave is defined as the time it takes for one CU to complete one K-complete
 * output tile.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param config_cache Precomputed values for config on specific problem_type_t.
 * @return double Latency in cycles.
 */
double compute_total_latency(const problem_t& problem,
                             const hardware_t& hardware,
                             const config_t& config,
                             const config_cache_t& config_cache);

inline double compute_total_latency(const problem_t& problem,
                                    const hardware_t& hardware,
                                    const config_t& config) {
  auto config_cache = create_config_cache(get_problem_type(problem, hardware), hardware, config);
  return compute_total_latency(problem, hardware, config, config_cache);
}

// Extract analytical metrics from a GEMM computation and export to CSV
void extract_analytical_metrics_csv(const hardware_t& hardware,
                                    const problem_t& problem,
                                    const config_t& config,
                                    const std::string& filename);

// Extract analytical metrics from a GEMM computation and export to JSON
void extract_analytical_metrics_json(const hardware_t& hardware,
                                     const problem_t& problem,
                                     const config_t& config,
                                     const std::string& filename);

// Extract analytical metrics from a GEMM computation and return as map
std::unordered_map<std::string, std::string> extract_analytical_metrics(const hardware_t& hardware,
                                                                        const problem_t& problem,
                                                                        const config_t& config);
}  // namespace origami

namespace std {
template <>
struct hash<origami::problem_type_t> {
  std::size_t operator()(const origami::problem_type_t& p) const { return p.hash(); }
};
}  // namespace std