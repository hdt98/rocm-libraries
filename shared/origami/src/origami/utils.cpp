// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/utils.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"
#include "origami/streamk.hpp"

#include <algorithm>
#include <chrono> // For timing
#include <cmath>
#include <iomanip> // For output formatting
#include <iostream>

namespace origami
{
        size_t select_grid_size(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                grid_selection_t algorithm)
        {
            switch (algorithm) {
                case grid_selection_t::min_resources:
                    return streamk::grid_min_resources(problem, hardware, config);

                case grid_selection_t::energy_aware:
                    return streamk::grid_energy_aware(problem, hardware, config);

                case grid_selection_t::reduction_cost_aware:
                    return streamk::grid_reduction_cost_aware(problem, hardware, config);

                case grid_selection_t::data_parallel:
                    return streamk::grid_data_parallel(problem, config);

                case grid_selection_t::analytical:
                    return streamk::grid_analytical(problem, hardware, config);

                case grid_selection_t::k_split_aware:
                    return streamk::grid_k_split_aware(problem, hardware, config);

                case grid_selection_t::number_of_cus:
                default:
                    return hardware.N_CU;
            }
        }

        
        config_t select_config(const problem_t& problem,
                       const hardware_t& hardware,
                       const std::vector<config_t>& configs)
        {
            // Use rank_configs to get configurations ranked by performance
            auto ranked_configs = rank_configs(problem, hardware, configs);
            
            if (ranked_configs.empty()) {
                throw std::runtime_error("No valid configs found.");
            }
            
            // Apply tie-breaking logic for configs with similar latency
            double best_latency = ranked_configs.front().latency;
            size_t num_the_same = 0;

            double tiebreaker_tolerance = 0.01;

            // Count configs with similar latency (within 1%)
            for(const auto& config : ranked_configs)
            {
                double diff = std::fabs(config.latency - best_latency);
                diff /= best_latency;
                if(diff < tiebreaker_tolerance)
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

                    double flops = static_cast<double>(2ull * MT_M * MT_N * MT_K);
                    double memory_traffic = static_cast<double>(MT_M * MT_K + MT_N * MT_K + MT_M * MT_N);

                    if(memory_traffic == 0.0) return 0.0;
                    return flops / memory_traffic;
                };
                
                std::stable_sort(ranked_configs.begin(), 
                          ranked_configs.begin() + num_the_same,
                          [&](const config_t& a, const config_t& b) {
                              double ai_a = compute_arithmetic_intensity(a);
                              double ai_b = compute_arithmetic_intensity(b);
                              return ai_a > ai_b; // descending
                          });
            }

            return ranked_configs.front();
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
        std::pair<double, size_t> select_workgroup_mapping(
            const problem_t& problem,
            const hardware_t& hardware,
            const dim3_t mt,
            const dim3_t mi,
            const std::vector<size_t>& wgms)
        {
            // Extract values from problem and tile dimensions
            std::size_t M = problem.size.m;
            std::size_t N = problem.size.n;
            std::size_t K = problem.size.k;
            std::size_t batch = problem.batch;
            
            // Extract tile dimensions
            std::size_t MT_M = mt.m;
            std::size_t MT_N = mt.n;
            std::size_t MT_K = mt.k;
            std::size_t MI_M = mi.m;
            std::size_t MI_N = mi.n;
            std::size_t MI_K = mi.k;
            
            // Calculate element size (using output type as approximation)
            std::size_t element_size = data_type_to_bits(problem.d_dtype) / 8;
            
            using WGMResult = std::pair<double, size_t>; // (l2_hit_rate, WGM)

            std::vector<WGMResult> valid_results;
            valid_results.reserve(wgms.size());

            // Iterate over all candidate WGM values
            for(const auto& candidate_wgm : wgms)
            {
                if(hardware_t::is_debug_enabled())
                {
                    std::cout << "Evaluating WGM=" << candidate_wgm << "\n";
                }

                // Optionally ensure we do not exceed LDS capacity
                // (If you want to factor in WGM, add it to your check_lds_capacity signature.)
                // For now, let's just check the tile itself:
                if(!check_lds_capacity(hardware, MT_M, MT_N, MT_K, element_size))
                {
                    if(hardware_t::is_debug_enabled())
                    {
                        std::cout << "Skipping WGM=" << candidate_wgm << " due to LDS capacity.\n";
                    }
                    continue;
                }

                // Compute L2 hit rate for this WGM
                double current_hit = estimate_l2_hit(hardware,
                                                     M,
                                                     N,
                                                     K,
                                                     batch,
                                                     MT_M,
                                                     MT_N,
                                                     MT_K,
                                                     element_size,
                                                     static_cast<int>(candidate_wgm),
                                                     1 /* splittingFactor */);

                valid_results.emplace_back(current_hit, candidate_wgm);
            }

            // If no valid WGM was found, throw an error
            if(valid_results.empty())
            {
                throw std::runtime_error("No valid WGM found.");
            }

            // Find the maximum L2 hit rate in valid_results
            // (Use max_element on the first value in the pair.)
            auto best_it = std::max_element(
                valid_results.begin(),
                valid_results.end(),
                [](const WGMResult& a, const WGMResult& b) {
                    return a.first < b.first; // "less" => a has smaller hit rate than b
                });

            double best_l2_hit = best_it->first;
            size_t best_wgm    = best_it->second;

            // Return (l2_hit_rate, WGM)
            return std::make_pair(best_l2_hit, best_wgm);
        }

        std::vector<config_t> rank_configs(
            const problem_t& problem,
            const hardware_t& hardware,
            const std::vector<config_t>& configs)
        {
            if (configs.empty()) {
                throw std::runtime_error("No configurations provided.");
            }

            // Extract problem data once
            std::size_t M = problem.size.m;
            std::size_t N = problem.size.n;
            std::size_t K = problem.size.k;
            std::size_t batch = problem.batch;
            bool transpose_a = problem.transpose_a;
            bool transpose_b = problem.transpose_b;
            data_type_t mi_dtype = problem.mi_dtype;
            std::size_t mx_block_size = problem.mx_block_size;

            // Calculate element sizes
            std::size_t element_size_A = data_type_to_bits(problem.a_dtype);
            std::size_t element_size_B = data_type_to_bits(problem.b_dtype);
            std::size_t element_size_out = data_type_to_bits(problem.d_dtype);

            std::vector<config_t> ranked_configs;
            ranked_configs.reserve(configs.size());
            
            for(auto config : configs)
            {
                // Check LDS capacity - skip invalid configurations
                if(!check_lds_capacity(hardware, config.mt.m, config.mt.n, config.mt.k, element_size_out))
                {
                    continue;
                }

                // Compute predicted latency for this configuration using structured API
                config.latency = compute_total_latency(hardware, problem, config, 0);

                ranked_configs.push_back(config);
            }

            // Sort configurations by latency (ascending - best first)
            std::stable_sort(ranked_configs.begin(), ranked_configs.end(), 
                           [](const config_t& a, const config_t& b) {
                               return a.latency < b.latency;
                           });

            return ranked_configs;
        }

        config_t select_config_mnk(
            std::size_t M,
            std::size_t N, 
            std::size_t K,
            const hardware_t& hardware,
            const std::vector<config_t>& configs)
        {
            // Create a default problem_t with the provided M, N, K and reasonable defaults
            problem_t problem;
            problem.size.m = M;
            problem.size.n = N;
            problem.size.k = K;
            problem.batch = 1;
            problem.transpose_a = true;               // Default to T
            problem.transpose_b = false;              // Default to N
            problem.a_dtype = data_type_t::Half;      // Default to fp16
            problem.b_dtype = data_type_t::Half;      // Default to fp16
            problem.c_dtype = data_type_t::Half;      // Default to fp16
            problem.d_dtype = data_type_t::Half;      // Default to fp16
            problem.mi_dtype = data_type_t::Half;     // Default to fp16
            problem.mx_block_size = 0;                // Default MX block size

            // Use the existing select_config function with the constructed problem
            return select_config(problem, hardware, configs);
        }

        std::vector<config_t> select_topk_configs(
            const problem_t& problem,
            const hardware_t& hardware,
            const std::vector<config_t>& configs,
            std::size_t topk)
        {
            // Get all configurations ranked by performance
            auto ranked_configs = rank_configs(problem, hardware, configs);
            
            // Return only the top K configurations
            if (ranked_configs.size() <= topk) {
                return ranked_configs;
            } else {
                return std::vector<config_t>(ranked_configs.begin(), ranked_configs.begin() + topk);
            }
        }

        double compute_perf_gflops(const hardware_t& hardware,
                                const problem_t& problem,
                                const config_t& config)
        {
            // Extract parameters from structured types
            size_t M = problem.size.m;
            size_t N = problem.size.n;
            size_t K = problem.size.k;
            size_t batch = problem.batch;
            
            // Compute total FLOPs
            double total_FLOPs = 2.0 * M * N * K; // For GEMM, each multiply-add is 2 FLOPs
            // Compute total time in seconds
            double cycles_per_second = hardware.compute_clock_ghz * 1e9; // 1 GHz = 1e9 cycles per second
            
            // Assume latency has been populated in the config.
            double total_time_seconds = config.latency / cycles_per_second;

            // Compute performance in FLOPS
            double FLOPS = total_FLOPs / total_time_seconds;
            // Convert to GFLOPS
            double GFLOPS = FLOPS / 1e9; // 1 TFLOP = 1e9 FLOPs
            return GFLOPS;
        }
} // namespace origami
