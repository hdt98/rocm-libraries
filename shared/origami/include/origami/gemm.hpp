// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <vector>
#include "origami/hardware.hpp"
#include "origami/types.hpp"

namespace origami {

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
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return double Latency in cycles.
 */
double compute_cvt_overhead(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config);

/**
 * @brief Compute the latency to process a single macro-tile for the given problem and hardware.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return double Latency in cycles.
 */
double compute_mt_compute_latency(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config);

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
 * @param splitting_factor
 * @return double Predicted L2-hitrate.
 */
double estimate_l2_hit(const problem_t& problem,
                       const hardware_t& hardware,
                       const config_t& config,
                       std::size_t splitting_factor);

/**
 * @brief Estimate the MALL-hitrate (last-level cache.)
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param num_active_cus
 * @param splitting_factor
 * @return double Predicted MALL-hitrate.
 */
double estimate_mall_hit(const problem_t& problem,
                         const hardware_t& hardware,
                         const config_t& config,
                         std::size_t num_active_cus,
                         std::size_t splitting_factor);

/**
 * @brief Determine the memory latency per MT_M x MT_N x MT_K Macro Tile (L_MT).
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param num_active_cus
 * @param splitting_factor
 * @return double Latency in cycles.
 */
double compute_memory_latency(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              std::size_t num_active_cus,
                              std::size_t splitting_factor);

/**
 * @brief Computes the latency to compute a K-COMPLETE tile.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param num_active_cus
 * @param splitting_factor
 * @return double Latency in cycles.
 */
double compute_tile_latency(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config,
                            std::size_t num_active_cus,
                            std::size_t splitting_factor);

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
 * @param num_active_cus
 * @param splitting_factor
 * @return double Latency in cycles.
 */
double compute_timestep_latency(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                std::size_t num_active_cus,
                                std::size_t splitting_factor);
/**
 * @brief Compute the total latency of a gemm based on the latency of one wave multiplied by the
 * number of waves A wave is defined as the time it takes for one CU to complete one K-complete
 * output tile.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return double Latency in cycles.
 */
double compute_total_latency(const problem_t& problem,
                             const hardware_t& hardware,
                             const config_t& config);

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
