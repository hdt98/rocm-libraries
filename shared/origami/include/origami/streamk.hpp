// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "origami/hardware.hpp"
#include "origami/types.hpp"

#include <vector>

namespace origami {
namespace streamk {

/**
 * @brief Calculate workspace size required for StreamK reduction.
 *
 * @param x Problem dimension X (M)
 * @param y Problem dimension Y (N)
 * @param mt_m Macro-tile size in M dimension
 * @param mt_n Macro-tile size in N dimension
 * @param bpe_c Bytes per element of C matrix
 * @param grid Grid size
 * @param tiles Number of tiles
 * @param reduction Reduction strategy
 * @return std::size_t Workspace size in bytes
 */
std::size_t get_workspace(std::size_t x,
                          std::size_t y,
                          std::size_t mt_m,
                          std::size_t mt_n,
                          std::size_t bpe_c,
                          std::size_t grid,
                          std::size_t tiles,
                          reduction_t reduction);

/**
 * @brief Select the best reduction strategy for StreamK.
 *
 * @param x Problem dimension X (M)
 * @param y Problem dimension Y (N)
 * @param z Problem dimension Z (K)
 * @param batch Batch size
 * @param mt_m Macro-tile size in M dimension
 * @param mt_n Macro-tile size in N dimension
 * @param mt_k Macro-tile size in K dimension
 * @param hardware Hardware characteristics
 * @param algorithm Grid selection algorithm
 * @return reduction_t Selected reduction strategy
 */
reduction_t select_reduction(std::size_t x,
                             std::size_t y,
                             std::size_t z,
                             std::size_t batch,
                             std::size_t mt_m,
                             std::size_t mt_n,
                             std::size_t mt_k,
                             const hardware_t& hardware,
                             grid_selection_t algorithm);

std::size_t grid_min_resources(const problem_t& problem,
                               const hardware_t& hardware,
                               const config_t& config);

std::size_t grid_energy_aware(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config);

std::size_t grid_reduction_cost_aware(const problem_t& p,
                                      const hardware_t& hardware,
                                      const config_t& c);

std::size_t grid_data_parallel(const problem_t& p, const config_t& c);

std::size_t grid_analytical(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config);

std::size_t grid_k_split_aware(const problem_t& problem,
                               const hardware_t& hardware,
                               const config_t& config);

}  // namespace streamk
}  // namespace origami
