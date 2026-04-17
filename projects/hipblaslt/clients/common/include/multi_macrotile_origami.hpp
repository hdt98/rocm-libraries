/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 *******************************************************************************/

#pragma once

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>
#include <regex>
#include <origami/gemm.hpp>
#include <origami/hardware.hpp>
#include <origami/types.hpp>

// Structure to hold per-solution performance estimates
struct OrigamiSolutionInfo
{
    int solution_index;
    hipblasLtMatmulAlgo_t algo;
    std::string solution_name;
    float waves_count;
    size_t workspace_size;
    double estimated_gflops;  // Estimated performance in GFLOPS
    double estimated_latency_cycles;  // Estimated latency in cycles from Origami

    OrigamiSolutionInfo() : solution_index(-1), waves_count(0.0f),
                            workspace_size(0), estimated_gflops(0.0),
                            estimated_latency_cycles(0.0) {}
};

// Structure to describe a split configuration with performance data
struct OrigamiSplitConfig
{
    std::vector<int64_t> split_sizes;              // Split sizes for dimension
    std::vector<int> best_solution_indices;         // Best solution index for each sub-problem
    std::vector<OrigamiSolutionInfo> solutions;     // Solution info for each sub-problem
    double estimated_total_gflops;                  // Estimated total performance
    double estimated_total_time_us;                 // Estimated total time in microseconds
    bool valid;

    OrigamiSplitConfig() : estimated_total_gflops(0.0),
                           estimated_total_time_us(1e9), valid(false) {}
};

/**
 * @brief Parse MacroTile dimensions from solution name
 * Solution names have format like: "...MT256x96x16..."
 */
inline bool parseMacroTileFromSolutionName(const std::string& solution_name,
                                            size_t& mt_m, size_t& mt_n, size_t& mt_k)
{
    std::regex mt_regex("MT(\\d+)x(\\d+)x(\\d+)");
    std::smatch match;

    if (std::regex_search(solution_name, match, mt_regex) && match.size() == 4)
    {
        mt_m = std::stoull(match[1].str());
        mt_n = std::stoull(match[2].str());
        mt_k = std::stoull(match[3].str());
        return true;
    }
    return false;
}

/**
 * @brief Convert hipblasDatatype_t to origami::data_type_t
 */
inline origami::data_type_t hipblasltTypeToOrigamiType(hipblasDatatype_t type)
{
    switch(type)
    {
        case HIPBLAS_R_16F:
            return origami::data_type_t::Half;
        case HIPBLAS_R_32F:
            return origami::data_type_t::Float;
        case HIPBLAS_R_16B:
            return origami::data_type_t::BFloat16;
        case HIPBLAS_R_8F_E4M3:
            return origami::data_type_t::Float8;
        case HIPBLAS_R_8F_E5M2:
            return origami::data_type_t::BFloat8;
        default:
            return origami::data_type_t::Half;  // Default to Half
    }
}

/**
 * @brief Create Origami problem_t from hipBLASLt parameters
 */
inline origami::problem_t createOrigamiProblem(int64_t M, int64_t N, int64_t K,
                                                hipblasOperation_t transA,
                                                hipblasOperation_t transB,
                                                hipblasDatatype_t a_type,
                                                hipblasDatatype_t b_type,
                                                hipblasDatatype_t c_type,
                                                hipblasDatatype_t d_type,
                                                hipblasDatatype_t compute_type)
{
    origami::problem_t problem;
    problem.size = {(size_t)M, (size_t)N, (size_t)K};
    problem.batch = 1;
    problem.a_transpose = (transA == HIPBLAS_OP_T) ? origami::transpose_t::T : origami::transpose_t::N;
    problem.b_transpose = (transB == HIPBLAS_OP_T) ? origami::transpose_t::T : origami::transpose_t::N;
    problem.a_dtype = hipblasltTypeToOrigamiType(a_type);
    problem.b_dtype = hipblasltTypeToOrigamiType(b_type);
    problem.c_dtype = hipblasltTypeToOrigamiType(c_type);
    problem.d_dtype = hipblasltTypeToOrigamiType(d_type);
    problem.mi_dtype = hipblasltTypeToOrigamiType(compute_type);
    return problem;
}

/**
 * @brief Query all available solutions for a given sub-problem
 *
 * @param handle hipBLASLt handle
 * @param M,N,K Sub-problem dimensions
 * @param transA,transB Transpose operations
 * @param a_type,b_type,c_type,d_type Data types
 * @param compute_type Compute type
 * @return Vector of solution info with estimated performance
 */
inline std::vector<OrigamiSolutionInfo> queryAllSolutionsForSubProblem(
    hipblasLtHandle_t handle,
    int64_t M, int64_t N, int64_t K,
    hipblasOperation_t transA,
    hipblasOperation_t transB,
    hipDataType a_type,
    hipDataType b_type,
    hipDataType c_type,
    hipDataType d_type,
    hipblasComputeType_t compute_type)
{
    std::vector<OrigamiSolutionInfo> solutions;

    try {
        // Query all algorithms using hipblaslt_ext::getAllAlgos
        std::vector<hipblasLtMatmulHeuristicResult_t> all_algos;

        hipblaslt_ext::GemmType gemm_type = hipblaslt_ext::GemmType::HIPBLASLT_GEMM;

        hipblasStatus_t status = hipblaslt_ext::getAllAlgos(
            handle,
            gemm_type,
            transA, transB,
            a_type, b_type, c_type, d_type,
            compute_type,
            all_algos);

        if (status != HIPBLAS_STATUS_SUCCESS || all_algos.empty())
        {
            // No solutions found
            return solutions;
        }

        // Convert to our structure and estimate performance
        for (size_t i = 0; i < all_algos.size(); i++)
        {
            const auto& heuristic = all_algos[i];

            if (heuristic.state != HIPBLAS_STATUS_SUCCESS)
                continue;

            OrigamiSolutionInfo info;
            info.solution_index = i;
            info.algo = heuristic.algo;
            info.waves_count = heuristic.wavesCount;
            info.workspace_size = heuristic.workspaceSize;

            // Get solution name
            info.solution_name = hipblaslt_ext::getSolutionNameFromAlgo(handle, info.algo);

            // Try to extract MacroTile configuration and use Origami for latency estimation
            size_t mt_m, mt_n, mt_k;
            if (parseMacroTileFromSolutionName(info.solution_name, mt_m, mt_n, mt_k))
            {
                try {
                    // Create Origami problem and config
                    origami::problem_t problem = createOrigamiProblem(
                        M, N, K, transA, transB, a_type, b_type, c_type, d_type, compute_type);

                    // Create origami config from MacroTile
                    origami::config_t config;
                    config.mt = {mt_m, mt_n, mt_k};
                    config.mi = {16, 16, 1};  // Default micro-tile, could parse from solution name
                    config.occupancy = 1;     // Default occupancy
                    config.streamk = false;   // Data-parallel

                    // Get hardware info (assuming device 0)
                    origami::hardware_t hardware = origami::hardware_t::get_hardware_for_device(0);

                    // Compute total latency using Origami
                    info.estimated_latency_cycles = origami::compute_total_latency(
                        problem, hardware, config, hardware.N_CU);

                    // Convert latency (cycles) to time (microseconds)
                    // Clock frequency in MHz -> cycles to microseconds
                    double clock_mhz = hardware.f_CU / 1e6;  // Convert Hz to MHz
                    double time_us = info.estimated_latency_cycles / clock_mhz;

                    // Calculate theoretical GFLOPS from latency
                    double ops = 2.0 * M * N * K;
                    info.estimated_gflops = (ops / time_us) * 1e-3;  // Convert to GFLOPS

                } catch (...) {
                    // Fall back to wavesCount-based estimation if Origami fails
                    double peak_gflops = 1400.0 * 1024.0;  // Peak in GFLOPS
                    double utilization_factor = std::min(1.0, (double)heuristic.wavesCount);
                    info.estimated_gflops = peak_gflops * utilization_factor * 0.5;
                    info.estimated_latency_cycles = 0.0;  // Unknown
                }
            }
            else
            {
                // Fall back to wavesCount-based estimation if cannot parse MacroTile
                double peak_gflops = 1400.0 * 1024.0;  // Peak in GFLOPS
                double utilization_factor = std::min(1.0, (double)heuristic.wavesCount);
                info.estimated_gflops = peak_gflops * utilization_factor * 0.5;
                info.estimated_latency_cycles = 0.0;  // Unknown
            }

            solutions.push_back(info);
        }

    } catch (...) {
        // Error querying solutions
        std::cerr << "Error querying solutions for " << M << "x" << N << "x" << K << std::endl;
    }

    return solutions;
}

/**
 * @brief Find optimal split configuration by querying all solutions
 *
 * This function:
 * 1. Generates candidate split configurations
 * 2. For each split, queries all available solutions for each sub-problem
 * 3. Tries all combinations of solutions across sub-problems
 * 4. Estimates total latency as sum of individual latencies (sequential execution)
 * 5. Returns the split configuration with minimum total latency
 *
 * @param handle hipBLASLt handle
 * @param total_size Total dimension to split (M or N)
 * @param other_dim The other dimension (N for M-split, M for N-split)
 * @param K K dimension
 * @param num_splits Number of splits (2, 3, 4, etc.)
 * @param macrotile_size MacroTile size for alignment
 * @param is_m_split True for M-split, false for N-split
 * @param transA,transB Transpose operations
 * @param a_type,b_type,c_type,d_type Data types
 * @param compute_type Compute type
 * @return Optimal split configuration
 */
inline OrigamiSplitConfig findOptimalOrigamiSplit(
    hipblasLtHandle_t handle,
    int64_t total_size,
    int64_t other_dim,
    int64_t K,
    int num_splits,
    int macrotile_size,
    bool is_m_split,
    hipblasOperation_t transA,
    hipblasOperation_t transB,
    hipDataType a_type,
    hipDataType b_type,
    hipDataType c_type,
    hipDataType d_type,
    hipblasComputeType_t compute_type)
{
    OrigamiSplitConfig best_config;

    if (num_splits != 2)
    {
        // For now, only support 2-way splits
        // Could be extended to 3+way splits with more complex search
        std::cerr << "Origami-optimized split only supports 2-way splits for now" << std::endl;
        return best_config;
    }

    // Generate candidate split ratios
    std::vector<double> candidate_ratios = {
        0.50,  // Uniform
        0.55, 0.60, 0.65, 0.70, 0.75,  // Uneven, favoring first
        0.45, 0.40, 0.35, 0.30, 0.25   // Uneven, favoring second
    };

    std::cout << "Origami-Optimized Split: Searching over " << candidate_ratios.size()
              << " candidate splits..." << std::endl;

    for (double ratio : candidate_ratios)
    {
        // Calculate split sizes
        int64_t split1 = (int64_t)(total_size * ratio);
        split1 = (split1 / macrotile_size) * macrotile_size;  // Align
        split1 = std::max(split1, (int64_t)macrotile_size);
        split1 = std::min(split1, total_size - (int64_t)macrotile_size);
        int64_t split2 = total_size - split1;

        if (split2 < macrotile_size)
            continue;  // Skip invalid splits

        std::vector<int64_t> split_sizes = {split1, split2};

        // Determine sub-problem dimensions
        std::vector<std::pair<int64_t, int64_t>> subproblem_dims;
        if (is_m_split)
        {
            subproblem_dims = {{split1, other_dim}, {split2, other_dim}};
        }
        else  // N-split
        {
            subproblem_dims = {{other_dim, split1}, {other_dim, split2}};
        }

        // Query all solutions for each sub-problem
        std::vector<std::vector<OrigamiSolutionInfo>> all_solutions;
        for (const auto& dims : subproblem_dims)
        {
            int64_t M_sub = dims.first;
            int64_t N_sub = dims.second;

            auto solutions = queryAllSolutionsForSubProblem(
                handle, M_sub, N_sub, K,
                transA, transB,
                a_type, b_type, c_type, d_type,
                compute_type);

            if (solutions.empty())
            {
                std::cerr << "No solutions found for sub-problem " << M_sub << "x" << N_sub << "x" << K << std::endl;
                goto next_ratio;  // Skip this split ratio
            }

            all_solutions.push_back(solutions);
        }

        // Try all combinations of solutions
        // For 2-way split: all_solutions[0].size() × all_solutions[1].size() combinations
        {
            for (size_t sol0_idx = 0; sol0_idx < all_solutions[0].size(); sol0_idx++)
            {
                for (size_t sol1_idx = 0; sol1_idx < all_solutions[1].size(); sol1_idx++)
                {
                    const auto& sol0 = all_solutions[0][sol0_idx];
                    const auto& sol1 = all_solutions[1][sol1_idx];

                    // Calculate total performance (sequential execution)
                    // Time = ops / GFLOPS
                    // Total time = time0 + time1

                    int64_t M0 = subproblem_dims[0].first;
                    int64_t N0 = subproblem_dims[0].second;
                    int64_t M1 = subproblem_dims[1].first;
                    int64_t N1 = subproblem_dims[1].second;

                    double ops0 = 2.0 * M0 * N0 * K;
                    double ops1 = 2.0 * M1 * N1 * K;

                    double time0_us = (ops0 / sol0.estimated_gflops) * 1e6;  // Convert to microseconds
                    double time1_us = (ops1 / sol1.estimated_gflops) * 1e6;

                    double total_time_us = time0_us + time1_us;

                    // Check if this is the best configuration so far
                    if (total_time_us < best_config.estimated_total_time_us)
                    {
                        best_config.split_sizes = split_sizes;
                        best_config.best_solution_indices = {(int)sol0_idx, (int)sol1_idx};
                        best_config.solutions = {sol0, sol1};
                        best_config.estimated_total_time_us = total_time_us;
                        best_config.estimated_total_gflops = (ops0 + ops1) / (total_time_us * 1e-6);
                        best_config.valid = true;
                    }
                }
            }
        }

next_ratio:
        continue;
    }

    if (best_config.valid)
    {
        std::cout << "Origami-Optimized Split: Best configuration found!" << std::endl;
        std::cout << "  Split sizes: [" << best_config.split_sizes[0] << ", "
                  << best_config.split_sizes[1] << "]" << std::endl;
        std::cout << "  Solution 0: " << best_config.solutions[0].solution_name
                  << " (waves=" << best_config.solutions[0].waves_count << ")" << std::endl;
        std::cout << "  Solution 1: " << best_config.solutions[1].solution_name
                  << " (waves=" << best_config.solutions[1].waves_count << ")" << std::endl;
        std::cout << "  Estimated total time: " << best_config.estimated_total_time_us << " us" << std::endl;
        std::cout << "  Estimated total GFLOPS: " << best_config.estimated_total_gflops << std::endl;
    }
    else
    {
        std::cerr << "Origami-Optimized Split: No valid configuration found!" << std::endl;
    }

    return best_config;
}

/**
 * @brief Wrapper function compatible with existing split API
 * Returns just the split sizes (solutions are stored separately)
 */
inline std::vector<int64_t> computeOrigamiOptimizedSplitsWithHandle(
    hipblasLtHandle_t handle,
    int64_t total_size,
    int num_splits,
    int macrotile_size,
    int64_t other_dim,
    int64_t K,
    bool is_m_split,
    hipblasOperation_t transA,
    hipblasOperation_t transB,
    hipDataType a_type,
    hipDataType b_type,
    hipDataType c_type,
    hipDataType d_type,
    hipblasComputeType_t compute_type)
{
    auto config = findOptimalOrigamiSplit(
        handle, total_size, other_dim, K,
        num_splits, macrotile_size, is_m_split,
        transA, transB,
        a_type, b_type, c_type, d_type,
        compute_type);

    if (config.valid)
    {
        return config.split_sizes;
    }
    else
    {
        // Fall back to uniform split
        std::vector<int64_t> uniform;
        int64_t base = (total_size / num_splits / macrotile_size) * macrotile_size;
        for (int i = 0; i < num_splits; i++)
            uniform.push_back(base);

        // Adjust last to cover remainder
        int64_t total_assigned = base * num_splits;
        if (total_assigned < total_size)
            uniform[num_splits-1] += (total_size - total_assigned);

        return uniform;
    }
}
