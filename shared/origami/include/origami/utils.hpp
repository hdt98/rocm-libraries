// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <functional>
#include <set>
#include <tuple>
#include <vector>

#include "origami/types.hpp"
#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/streamk.hpp"

namespace origami {

/**
 * @brief Based on the provided problem and configs; selects the best config. 
 * 
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs Vector of all possible valid configurations.
 * @return config_t Predicted best possible configuration.
 */
config_t select_config(const problem_t& problem,
                       const hardware_t& hardware,
                       const std::vector<config_t>& configs);

/**
 * @brief Based on the provided kernel config, select the best grid dimension. 
 * 
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param grid_selection_t grid selection algorithm (@see origami::grid_selection_t)
 * @param biggest_allowable_split 
 * @return size_t Dimensions of the grid launched. 
 */
size_t select_grid_size(const problem_t& problem,
                        const hardware_t& hardware,
                        const config_t& config,
                        grid_selection_t algorithm);


/**
 * @brief Select best workgroup-mapping for the given tile size.
 * 
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param mt Macro-tile of the kernel.
 * @param mi Matrix-instruction of the kernel.
 * @param wgms List of possible workgroup-mappings.
 * @return std::pair<double, size_t> 
 */
std::pair<double, size_t> select_workgroup_mapping(
    const problem_t& problem,
    const hardware_t& hardware,
    const dim3_t mt,
    const dim3_t mi,
    const std::vector<size_t>& wgms);

/**
 * @brief Rank configurations based on predicted performance.
 * 
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs List of candidate configurations to rank
 * @return std::vector<config_t> Configurations ranked by performance (best first)
 */
std::vector<config_t> rank_configs(
    const problem_t& problem,
    const hardware_t& hardware,
    const std::vector<config_t>& configs);

/**
 * @brief Select best configuration based only on M, N, K dimensions with default settings.
 * 
 * @param M Problem dimension M 
 * @param N Problem dimension N
 * @param K Problem dimension K
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs List of candidate configurations
 * @return config_t Best configuration for the given dimensions
 */
config_t select_config_mnk(
    std::size_t M,
    std::size_t N, 
    std::size_t K,
    const hardware_t& hardware,
    const std::vector<config_t>& configs);

/**
 * @brief Select top K configurations based on performance ranking.
 * 
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs List of candidate configurations to rank
 * @param topk Number of top configurations to return
 * @return std::vector<config_t> Top K configurations ranked by performance
 */
std::vector<config_t> select_topk_configs(
    const problem_t& problem,
    const hardware_t& hardware,
    const std::vector<config_t>& configs,
    std::size_t topk);

/**
 * @brief Given a latency (populated in config), compute the achieved 
 * throughput in gflops.
 * 
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param problem Problem description (M, N, K, etc.)
 * @param config Kernel configuration/descriptor.
 * @return double Throughput in gflops/s.
 */
double compute_perf_gflops(const hardware_t& hardware,
                            const problem_t& problem,
                            const config_t& config);

}  // namespace origami
