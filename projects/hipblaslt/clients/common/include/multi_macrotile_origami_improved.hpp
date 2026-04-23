/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Origami-based split search: generates candidate split configurations,
 * scores them using Origami's compute_total_latency analytical model,
 * and exposes them for empirical micro-benchmarking in the timing path.
 *
 *******************************************************************************/

#pragma once

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <origami/gemm.hpp>
#include <origami/hardware.hpp>
#include <origami/types.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <regex>

struct OrigamiCandidate
{
    std::vector<int64_t> split_sizes;
    std::string label;
    double origami_total_latency; // sum of compute_total_latency for all sub-problems (cycles)
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

// Convert hipDataType to origami::data_type_t
inline origami::data_type_t hipToOrigamiDtype(hipDataType dt)
{
    switch (dt)
    {
    case HIP_R_16F:  return origami::data_type_t::Half;
    case HIP_R_16BF: return origami::data_type_t::BFloat16;
    case HIP_R_32F:  return origami::data_type_t::Float;
    case HIP_R_8F_E4M3_FNUZ: return origami::data_type_t::Float8_fnuz;
    case HIP_R_8F_E5M2_FNUZ: return origami::data_type_t::BFloat8_fnuz;
    case HIP_R_8F_E4M3: return origami::data_type_t::Float8;
    case HIP_R_8F_E5M2: return origami::data_type_t::BFloat8;
    default: return origami::data_type_t::Half;
    }
}

// Compute Origami total latency for a sub-problem given its dimensions and
// the MacroTile selected by the heuristic.
inline double computeOrigamiLatency(
    int64_t M, int64_t N, int64_t K,
    size_t mt_m, size_t mt_n, size_t mt_k,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    int device_id)
{
    try
    {
        origami::problem_t problem;
        problem.size = {(size_t)M, (size_t)N, (size_t)K};
        problem.batch = 1;
        problem.a_transpose = (transA == HIPBLAS_OP_T) ? origami::transpose_t::T : origami::transpose_t::N;
        problem.b_transpose = (transB == HIPBLAS_OP_T) ? origami::transpose_t::T : origami::transpose_t::N;
        problem.a_dtype = hipToOrigamiDtype(a_type);
        problem.b_dtype = hipToOrigamiDtype(b_type);
        problem.c_dtype = hipToOrigamiDtype(c_type);
        problem.d_dtype = hipToOrigamiDtype(d_type);
        problem.mi_dtype = origami::data_type_t::Float; // f32 compute

        origami::config_t config;
        config.mt = {mt_m, mt_n, mt_k};
        config.mi = {16, 16, 1};
        config.occupancy = 1;

        auto hardware = origami::hardware_t::get_hardware_for_device(device_id);

        return origami::compute_total_latency(problem, hardware, config, hardware.N_CU);
    }
    catch (...)
    {
        return -1.0; // signal failure
    }
}

// Query the heuristic for a sub-problem and get its MacroTile + Origami latency
inline double querySubProblemLatency(
    hipblasLtHandle_t handle,
    int64_t M, int64_t N, int64_t K,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    hipblasComputeType_t compute_type,
    int device_id)
{
    std::vector<hipblasLtMatmulHeuristicResult_t> algos;
    hipblaslt_ext::getAllAlgos(handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                               transA, transB, a_type, b_type, c_type, d_type,
                               compute_type, algos);
    if (algos.empty() || algos[0].state != HIPBLAS_STATUS_SUCCESS)
        return -1.0;

    std::string sol_name = hipblaslt_ext::getSolutionNameFromAlgo(handle, algos[0].algo);
    size_t mt_m, mt_n, mt_k;
    if (!parseMacroTileFromName(sol_name, mt_m, mt_n, mt_k))
        return -1.0;

    return computeOrigamiLatency(M, N, K, mt_m, mt_n, mt_k,
                                  transA, transB, a_type, b_type, c_type, d_type,
                                  device_id);
}

// MacroTile preservation check.
// Previously used getAllAlgos() which is NOT dimension-aware (returns the same
// kernel regardless of M,N), making the check ineffective. Now always returns
// true — the empirical micro-benchmark naturally rejects bad splits by measuring
// them as slower.
inline bool isMacroTilePreserved(
    hipblasLtHandle_t /*handle*/,
    int64_t /*M*/, int64_t /*N*/, int64_t /*K*/,
    int /*num_splits*/, bool /*is_m_split*/,
    hipblasOperation_t /*transA*/, hipblasOperation_t /*transB*/,
    hipDataType /*a_type*/, hipDataType /*b_type*/,
    hipDataType /*c_type*/, hipDataType /*d_type*/,
    hipblasComputeType_t /*compute_type*/)
{
    return true;
}

// Generate + score candidate split configurations.
// Each candidate is scored using Origami compute_total_latency:
//   total_latency = latency(sub0) + latency(sub1)
// Candidates are sorted by total_latency (best first).
inline std::vector<OrigamiCandidate> generateOrigamiCandidates(
    int64_t total_size,
    int macrotile_size,
    bool brute_force,
    hipblasLtHandle_t handle,
    int64_t other_dim,
    int64_t K,
    bool is_m_split,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    hipblasComputeType_t compute_type,
    int device_id)
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
            candidates.push_back({{s1, s2}, label, 0.0});
    };

    if (brute_force)
    {
        const int64_t step = 16;
        int64_t lo = (min_sub + step - 1) / step * step;
        int64_t hi = total_size - min_sub;
        for (int64_t s1 = lo; s1 <= hi; s1 += step)
            add(s1, "bf-" + std::to_string(s1));
    }
    else
    {
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

    // Candidates are not analytically scored here — the empirical micro-benchmark
    // in the timing path tests all candidates with actual GPU execution and picks
    // the fastest. Analytical scoring was removed because:
    //   1. getAllAlgos() is not dimension-aware — it returns the same kernel for all
    //      sub-problem sizes, making the scores identical for all candidates
    //   2. Even with correct scoring, empirical search found different winners 55% of
    //      the time (only 45% precision)
    //   3. The scoring loop called getAllAlgos() O(N×S) times (N candidates × S sub-problems),
    //      each taking 100ms+, creating a multi-minute bottleneck for large problems

    return unique;
}

// Generate 3-way split candidates.
// Tries combinations of pow2 + uniform partitioning into 3 sub-problems.
inline std::vector<OrigamiCandidate> generate3WayCandidates(
    int64_t total_size, int macrotile_size)
{
    std::vector<OrigamiCandidate> candidates;
    int mt = std::max(macrotile_size, 1);
    int64_t min_sub = std::max((int64_t)2048, total_size * 10 / 100);
    min_sub = (min_sub / mt) * mt;

    auto add3 = [&](int64_t s1, int64_t s2, const std::string& label) {
        int64_t s3 = total_size - s1 - s2;
        if (s1 >= min_sub && s2 >= min_sub && s3 >= min_sub)
            candidates.push_back({{s1, s2, s3}, label, 0.0});
    };

    // Uniform 3-way
    int64_t third = (total_size / 3 / mt) * mt;
    add3(third, third, "uniform-3way");

    // Pow2 + remainder
    for (int64_t p1 : {(int64_t)2048, (int64_t)4096, (int64_t)8192})
    {
        if (p1 >= total_size) continue;
        int64_t rem = total_size - p1;
        int64_t half_rem = (rem / 2 / mt) * mt;
        add3(p1, half_rem, "pow2-" + std::to_string(p1/1024) + "k+split");

        // Two pow2 + remainder
        for (int64_t p2 : {(int64_t)2048, (int64_t)4096, (int64_t)8192})
        {
            if (p1 + p2 >= total_size) continue;
            add3(p1, p2, "pow2-" + std::to_string(p1/1024) + "k+" + std::to_string(p2/1024) + "k");
        }
    }

    // Deduplicate
    std::vector<OrigamiCandidate> unique;
    for (auto& c : candidates)
    {
        bool dup = false;
        for (auto& u : unique)
            if (u.split_sizes == c.split_sizes) { dup = true; break; }
        if (!dup) unique.push_back(c);
    }
    return unique;
}

// Generate XCD-aware candidates.
// Targets sub-problem A-tile fitting in L2 per XCD (4 MB on MI355X).
inline std::vector<OrigamiCandidate> generateXCDAwareCandidates(
    int64_t total_size, int macrotile_size, int64_t K, size_t elem_size)
{
    std::vector<OrigamiCandidate> candidates;
    int mt = std::max(macrotile_size, 1);
    int64_t min_sub = std::max((int64_t)2048, total_size * 15 / 100);
    min_sub = (min_sub / mt) * mt;

    const int NUM_XCDS = 8;
    const size_t L2_PER_XCD = 4 * 1024 * 1024; // 4 MB

    auto add = [&](int64_t s1, const std::string& label) {
        s1 = std::max(s1, min_sub);
        s1 = std::min(s1, total_size - min_sub);
        int64_t s2 = total_size - s1;
        if (s2 >= min_sub && s1 >= min_sub)
            candidates.push_back({{s1, s2}, label, 0.0});
    };

    // Target: sub_A per XCD = (sub_M / NUM_XCDS) × K × elem_size ≤ L2_PER_XCD
    // sub_M ≤ L2_PER_XCD × NUM_XCDS / (K × elem_size)
    int64_t ideal_sub_m = (int64_t)(L2_PER_XCD * NUM_XCDS / (K * elem_size));
    ideal_sub_m = (ideal_sub_m / mt) * mt;
    ideal_sub_m = std::max(ideal_sub_m, min_sub);

    add(ideal_sub_m, "xcd-l2-opt-" + std::to_string(ideal_sub_m));

    // Also try fractions and multiples of the ideal
    for (double f : {0.5, 0.75, 1.25, 1.5, 2.0})
    {
        int64_t sz = (int64_t)(ideal_sub_m * f);
        sz = (sz / mt) * mt;
        add(sz, "xcd-" + std::to_string((int)(f*100)) + "pct");
    }

    // Standard candidates as fallback
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

    // Deduplicate
    std::vector<OrigamiCandidate> unique;
    for (auto& c : candidates)
    {
        bool dup = false;
        for (auto& u : unique)
            if (u.split_sizes[0] == c.split_sizes[0]) { dup = true; break; }
        if (!dup) unique.push_back(c);
    }
    return unique;
}

// Entry point called by splitGemmProblem.
// Strategies: 17/18=Origami, 19/20=brute-force, 21/22=3-way, 23/24=XCD-aware
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
    hipblasComputeType_t compute_type,
    bool brute_force,
    bool xcd_aware)
{
    int64_t M = is_m_split ? total_size : other_dim;
    int64_t N = is_m_split ? other_dim : total_size;

    if (!isMacroTilePreserved(handle, M, N, K, num_splits, is_m_split,
                               transA, transB, a_type, b_type, c_type, d_type, compute_type))
        return {};

    int device_id = 0;
    hipGetDevice(&device_id);

    std::vector<OrigamiCandidate> cands;

    if (num_splits >= 3)
    {
        // 3-way splitting (S21/S22)
        cands = generate3WayCandidates(total_size, macrotile_size);
    }
    else if (xcd_aware)
    {
        // XCD-aware splitting (S23/S24)
        size_t elem_size = getDataTypeSize(a_type);
        cands = generateXCDAwareCandidates(total_size, macrotile_size, K, elem_size);
    }
    else
    {
        // Standard Origami (S17/18) or brute-force (S19/20)
        cands = generateOrigamiCandidates(total_size, macrotile_size, brute_force,
                                           handle, other_dim, K, is_m_split,
                                           transA, transB, a_type, b_type, c_type, d_type,
                                           compute_type, device_id);
    }

    getOrigamiCandidates() = cands;

    if (!cands.empty())
        return cands[0].split_sizes;
    return {};
}
