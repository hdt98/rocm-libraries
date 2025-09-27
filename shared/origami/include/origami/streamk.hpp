// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "origami/hardware.hpp"
#include "origami/types.hpp"

#include <vector>

namespace origami
{
    namespace streamk
    {
        std::size_t get_workspace(
            std::size_t x,
            std::size_t y,
            std::size_t mt_m,
            std::size_t mt_n,
            std::size_t bpe_c,
            std::size_t grid,
            std::size_t tiles,
            reduction_t reduction);

        reduction_t select_reduction(
            std::size_t x,
            std::size_t y,
            std::size_t z,
            std::size_t batch,
            std::size_t mt_m,
            std::size_t mt_n,
            std::size_t mt_k,
            const hardware_t& hardware,
            grid_selection_t algorithm);

        std::size_t grid_min_resources(const problem_t& problem, 
            const hardware_t& hardware, const config_t& config);

        std::size_t grid_energy_aware(const problem_t& problem, 
            const hardware_t& hardware, const config_t& config);

        std::size_t grid_reduction_cost_aware(
                                const problem_t& p,
                                const hardware_t& hardware,
                                const config_t& c);

        std::size_t grid_data_parallel(const problem_t& p, const config_t& c);

        std::size_t grid_analytical(
            const problem_t& problem,
            const hardware_t& hardware,
            const config_t& config);

        std::size_t grid_k_split_aware(const problem_t&  problem,
                                      const hardware_t& hardware,
                                      const config_t&   config);

    } // namespace streamk
} // namespace origami
