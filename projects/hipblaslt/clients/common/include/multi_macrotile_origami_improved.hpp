/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Origami-based split search: generates candidate split configurations and
 * exposes them for empirical micro-benchmarking in the timing path.
 *
 *******************************************************************************/

#pragma once

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <regex>

struct OrigamiCandidate
{
    std::vector<int64_t> split_sizes;
    std::string label;
};

inline std::vector<OrigamiCandidate>& getOrigamiCandidates()
{
    static thread_local std::vector<OrigamiCandidate> candidates;
    return candidates;
}

// Parse MacroTile dimensions from a Tensile solution name (e.g. "...MT256x96x16...")
inline bool parseMacroTileFromName(const std::string& name,
                                   size_t& mt_m, size_t& mt_n, size_t& mt_k)
{
    std::regex re("MT(\\d+)x(\\d+)x(\\d+)");
    std::smatch m;
    if (std::regex_search(name, m, re) && m.size() == 4)
    {
        mt_m = std::stoull(m[1].str());
        mt_n = std::stoull(m[2].str());
        mt_k = std::stoull(m[3].str());
        return true;
    }
    return false;
}

// Check whether splitting preserves the baseline MacroTile (within 75%).
inline bool isMacroTilePreserved(
    hipblasLtHandle_t handle,
    int64_t M, int64_t N, int64_t K,
    int num_splits, bool is_m_split,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    hipblasComputeType_t compute_type)
{
    auto query = [&](int64_t m, int64_t n) -> std::string {
        std::vector<hipblasLtMatmulHeuristicResult_t> algos;
        hipblaslt_ext::getAllAlgos(handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                   transA, transB, a_type, b_type, c_type, d_type,
                                   compute_type, algos);
        if (!algos.empty() && algos[0].state == HIPBLAS_STATUS_SUCCESS)
            return hipblaslt_ext::getSolutionNameFromAlgo(handle, algos[0].algo);
        return "";
    };

    std::string bl_name = query(M, N);
    size_t bl_m, bl_n, bl_k;
    if (!parseMacroTileFromName(bl_name, bl_m, bl_n, bl_k))
        return false;

    int64_t Ms = is_m_split ? (M / num_splits) : M;
    int64_t Ns = is_m_split ? N : (N / num_splits);

    std::string sp_name = query(Ms, Ns);
    size_t sp_m, sp_n, sp_k;
    if (!parseMacroTileFromName(sp_name, sp_m, sp_n, sp_k))
        return false;

    return sp_m >= bl_m * 0.75 && sp_n >= bl_n * 0.75;
}

// Generate candidate split configurations for empirical testing.
// Candidates include ratio-based + exhaustive power-of-2 splits.
// Minimum sub-problem size enforced to avoid pathological tiny splits.
inline std::vector<OrigamiCandidate> generateOrigamiCandidates(
    int64_t total_size,
    int macrotile_size)
{
    std::vector<OrigamiCandidate> candidates;
    int mt = std::max(macrotile_size, 1);

    int64_t min_sub = std::max((int64_t)2048, total_size * 15 / 100);
    min_sub = (min_sub / mt) * mt;
    min_sub = std::max(min_sub, (int64_t)mt);

    auto add = [&](int64_t s1, const std::string& label) {
        s1 = std::max(s1, min_sub);
        s1 = std::min(s1, total_size - min_sub);
        int64_t s2 = total_size - s1;
        if (s2 >= min_sub && s1 >= min_sub)
            candidates.push_back({{s1, s2}, label});
    };

    add((total_size / 2 / mt) * mt, "uniform-50/50");

    for (double r : {0.60, 0.40, 0.70, 0.30})
        add(((int64_t)(total_size * r) / mt) * mt,
            "asym-" + std::to_string((int)(r*100)) + "/" + std::to_string(100-(int)(r*100)));

    for (int64_t p = 2048; p < total_size; p *= 2)
    {
        int64_t rem = total_size - p;
        if (rem >= min_sub && p >= min_sub)
            add(p, "pow2-" + std::to_string(p/1024) + "k");
    }

    // Deduplicate by s1
    std::vector<OrigamiCandidate> unique;
    for (auto& c : candidates)
    {
        bool dup = false;
        for (auto& u : unique)
            if (u.split_sizes[0] == c.split_sizes[0]) { dup = true; break; }
        if (!dup)
            unique.push_back(c);
    }
    return unique;
}

// Entry point called by splitGemmProblem for S17/S18.
// Returns the default (uniform) split sizes; the real selection happens via
// empirical micro-benchmark in testing_matmul.hpp using getOrigamiCandidates().
inline std::vector<int64_t> computeOrigamiOptimizedSplitsWithHandle(
    hipblasLtHandle_t handle,
    int64_t total_size,
    int num_splits,
    int macrotile_size,
    int64_t other_dim,
    int64_t K,
    bool is_m_split,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    hipblasComputeType_t compute_type)
{
    int64_t M = is_m_split ? total_size : other_dim;
    int64_t N = is_m_split ? other_dim : total_size;

    if (!isMacroTilePreserved(handle, M, N, K, num_splits, is_m_split,
                               transA, transB, a_type, b_type, c_type, d_type, compute_type))
        return {};

    auto cands = generateOrigamiCandidates(total_size, macrotile_size);
    getOrigamiCandidates() = cands;

    if (!cands.empty())
        return cands[0].split_sizes;
    return {};
}
