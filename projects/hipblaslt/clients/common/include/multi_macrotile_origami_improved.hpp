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
#include <origami/origami.hpp>   // select_workgroup_mapping
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

// Query the dimension-aware heuristic for a sub-problem, extract its
// MacroTile from the solution name, and return the Origami analytical latency.
// Uses hipblasLtMatmulAlgoGetHeuristic (dimension-aware, fast) instead of
// getAllAlgos (dimension-blind, slow O(N) library scan).
inline double querySubProblemLatency(
    hipblasLtHandle_t handle,
    int64_t M, int64_t N, int64_t K,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    hipblasComputeType_t compute_type,
    int device_id)
{
    int64_t A_rows = (transA == HIPBLAS_OP_N) ? M : K;
    int64_t A_cols = (transA == HIPBLAS_OP_N) ? K : M;
    int64_t B_rows = (transB == HIPBLAS_OP_N) ? K : N;
    int64_t B_cols = (transB == HIPBLAS_OP_N) ? N : K;

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    hipblasLtMatrixLayoutCreate(&matA, a_type, A_rows, A_cols, A_rows);
    hipblasLtMatrixLayoutCreate(&matB, b_type, B_rows, B_cols, B_rows);
    hipblasLtMatrixLayoutCreate(&matC, c_type, M, N, M);
    hipblasLtMatrixLayoutCreate(&matD, d_type, M, N, M);

    hipblasLtMatmulDesc_t desc;
    hipblasLtMatmulDescCreate(&desc, compute_type, HIP_R_32F);
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int32_t));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int32_t));

    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);

    hipblasLtMatmulHeuristicResult_t heur;
    int ret = 0;
    hipblasLtMatmulAlgoGetHeuristic(handle, desc, matA, matB, matC, matD, pref, 1, &heur, &ret);

    double latency = -1.0;
    if (ret > 0)
    {
        try {
            std::string sol_name = hipblaslt_ext::getSolutionNameFromAlgo(handle, heur.algo);
            size_t mt_m, mt_n, mt_k;
            if (parseMacroTileFromName(sol_name, mt_m, mt_n, mt_k))
            {
                latency = computeOrigamiLatency(M, N, K, mt_m, mt_n, mt_k,
                                                transA, transB, a_type, b_type,
                                                c_type, d_type, device_id);
            }
        } catch (...) {}
    }

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatmulDescDestroy(desc);
    hipblasLtMatrixLayoutDestroy(matD);
    hipblasLtMatrixLayoutDestroy(matC);
    hipblasLtMatrixLayoutDestroy(matB);
    hipblasLtMatrixLayoutDestroy(matA);

    return latency;
}

// Full Origami WGM triple: wgm + wgmxcc + wgmxccchunk.
struct OrigamiWgmTriple
{
    int16_t wgm         = 0;
    int16_t wgmxcc      = 0;
    int16_t wgmxccchunk = 0;
    bool    valid       = false;
};

// Query Origami's select_workgroup_mapping for the optimal (wgm, wgmxcc, wgmxccchunk)
// triple of a sub-problem. On any internal failure, returns {0,0,0,false}.
inline OrigamiWgmTriple selectOrigamiWgmTriple(
    int64_t M, int64_t N, int64_t K,
    size_t mt_m, size_t mt_n, size_t mt_k,
    size_t mi_m, size_t mi_n, size_t mi_k,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    int device_id)
{
    OrigamiWgmTriple out;
    if (mt_m == 0 || mt_n == 0) return out;
    if (mi_m == 0 || mi_n == 0) { mi_m = 16; mi_n = 16; mi_k = 16; }
    try
    {
        origami::problem_t problem;
        problem.size = {(size_t)M, (size_t)N, (size_t)K};
        problem.batch = 1;
        problem.a_transpose = (transA == HIPBLAS_OP_T)
                              ? origami::transpose_t::T : origami::transpose_t::N;
        problem.b_transpose = (transB == HIPBLAS_OP_T)
                              ? origami::transpose_t::T : origami::transpose_t::N;
        problem.a_dtype = hipToOrigamiDtype(a_type);
        problem.b_dtype = hipToOrigamiDtype(b_type);
        problem.c_dtype = hipToOrigamiDtype(c_type);
        problem.d_dtype = hipToOrigamiDtype(d_type);
        problem.mi_dtype = origami::data_type_t::Float;

        origami::config_t config;
        config.mt = {mt_m, mt_n, mt_k};
        config.mi = {mi_m, mi_n, mi_k};
        config.occupancy = 1;

        auto hardware = origami::hardware_t::get_hardware_for_device(device_id);

        size_t n_tiles_m = (M + mt_m - 1) / mt_m;
        size_t n_tiles_n = (N + mt_n - 1) / mt_n;
        size_t skGrid    = n_tiles_m * n_tiles_n;

        auto mapping = origami::select_workgroup_mapping(problem, hardware, config, skGrid);

        auto clamp16 = [](long v) -> int16_t {
            if (v >  32767) return  32767;
            if (v < -32768) return -32768;
            return (int16_t)v;
        };
        out.wgm         = clamp16((long)mapping.wgm);
        out.wgmxcc      = clamp16((long)mapping.wgmxcc);
        out.wgmxccchunk = clamp16((long)mapping.wgmxccchunk);
        out.valid = true;
        return out;
    }
    catch (...)
    {
        return out;
    }
}

// Backwards-compatible wrapper: returns just the wgm scalar.
inline int16_t selectOrigamiWgm(
    int64_t M, int64_t N, int64_t K,
    size_t mt_m, size_t mt_n, size_t mt_k,
    size_t mi_m, size_t mi_n, size_t mi_k,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    int device_id)
{
    auto t = selectOrigamiWgmTriple(M, N, K, mt_m, mt_n, mt_k,
                                    mi_m, mi_n, mi_k,
                                    transA, transB, a_type, b_type,
                                    c_type, d_type, device_id);
    return t.valid ? t.wgm : (int16_t)0;
}

// ── Multi-MT-aware WGM selection ────────────────────────────────────────────
//
// Goal: when a GEMM is split into N sub-problems sharing one of A or B, schedule
// sub-problem i's workgroups on the same XCD that processed the matching tile
// (n-tile column for M-split, m-tile row for N-split) in sub-problem i-1.
// The shared matrix data already in that XCD's L2 cache is then re-used.
//
// Mechanism (MI350X, NUM_XCD=8, numCUsPerXCD=32):
//   * Tensile maps physical WG-id → XCD via a round-robin (wg_id % NUM_XCD),
//     with the (wgm, wgmxcc, wgmxccchunk) triple controlling how the logical
//     (m_tile, n_tile) → wg_id permutation is built.
//   * `GemmTuning` now exposes setWgm + setWgmXcc + setWgmXccChunk so we can
//     propagate ALL THREE values from sub[0] to sub[1+].
//
// Layout-gating: propagating sub[0]'s WGM triple is only physically meaningful
// when both sub-problems share the matrix that determines XCD-locality.
//
//   M-split (split along M, B is shared):
//     transB = N → B is K×N column-major.  WGs at the same n_tile column read
//                  the same B columns.  Same WGM triple → same column→XCD
//                  pattern.  HELPS.
//     transB = T → B is N×K, row-major.  Different access pattern; propagation
//                  was empirically harmful.  Skip.
//
//   N-split (split along N, A is shared):
//     transA = N → A is M×K column-major.  WGs at the same m_tile row read
//                  the same A rows.  Propagation HELPS.
//     transA = T → A is K×M.  Skip.
//
// `prev` carries sub-problem (i-1)'s applied triple; if invalid, falls back to
// plain Origami picks.
struct MultiMTAwareWgm
{
    OrigamiWgmTriple triple;     // chosen (wgm, wgmxcc, wgmxccchunk)
    bool overrode = false;       // did we override Origami's pick?
};

inline MultiMTAwareWgm selectMultiMTAwareWgmTriple(
    int sub_idx,
    int64_t M, int64_t N, int64_t K,
    size_t mt_m, size_t mt_n, size_t mt_k,
    size_t mi_m, size_t mi_n, size_t mi_k,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    int device_id,
    bool is_m_split,
    const OrigamiWgmTriple& prev)
{
    MultiMTAwareWgm out;

    // First sub-problem: use Origami's triple unconditionally.
    auto origami_t = selectOrigamiWgmTriple(M, N, K, mt_m, mt_n, mt_k,
                                            mi_m, mi_n, mi_k,
                                            transA, transB, a_type, b_type,
                                            c_type, d_type, device_id);
    if (sub_idx == 0 || !prev.valid)
    {
        out.triple = origami_t;
        return out;
    }

    // Layout gate: empirically (see MMT_AWARE_WGM_RESULTS.md), propagation
    // helps only when BOTH transposes are 'N' (NN layout) — i.e., A column-major
    // and B column-major.  TN (transA=T) was found to consistently regress and
    // NT (transB=T) was mostly neutral.  We require both `N`s here.
    bool layout_eligible = (transA == HIPBLAS_OP_N) && (transB == HIPBLAS_OP_N);

    if (!layout_eligible)
    {
        // Other layouts: keep per-sub Origami pick (preserve baseline behavior).
        out.triple = origami_t;
        return out;
    }
    (void)is_m_split;

    // Eligible: inherit sub[0]'s entire (wgm, wgmxcc, wgmxccchunk) triple.
    out.triple    = prev;
    out.overrode  = (origami_t.valid &&
                     (origami_t.wgm         != prev.wgm
                   || origami_t.wgmxcc      != prev.wgmxcc
                   || origami_t.wgmxccchunk != prev.wgmxccchunk));
    return out;
}

// Backwards-compatible wrapper: returns just the wgm scalar (older callers).
inline int16_t selectMultiMTAwareWgm(
    int sub_idx,
    int64_t M, int64_t N, int64_t K,
    size_t mt_m, size_t mt_n, size_t mt_k,
    size_t mi_m, size_t mi_n, size_t mi_k,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    int device_id,
    bool is_m_split,
    int16_t prev_wgm)
{
    OrigamiWgmTriple prev;
    if (prev_wgm != 0) { prev.wgm = prev_wgm; prev.valid = true; }
    auto r = selectMultiMTAwareWgmTriple(sub_idx, M, N, K,
                                         mt_m, mt_n, mt_k, mi_m, mi_n, mi_k,
                                         transA, transB, a_type, b_type,
                                         c_type, d_type, device_id,
                                         is_m_split, prev);
    return r.triple.valid ? r.triple.wgm : (int16_t)0;
}


// Parse "MI<m>x<n>x<k>" out of a Tensile solution name.
// Returns false if the substring is absent.
inline bool parseMatrixInstructionFromName(const std::string& name,
                                           size_t& mi_m, size_t& mi_n, size_t& mi_k)
{
    std::regex re("MI(\\d+)x(\\d+)x(\\d+)");
    std::smatch m;
    if (std::regex_search(name, m, re) && m.size() == 4)
    {
        mi_m = std::stoull(m[1].str());
        mi_n = std::stoull(m[2].str());
        mi_k = std::stoull(m[3].str());
        return true;
    }
    return false;
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
        const int NUM_CUS = 256;

        add((total_size / 2 / mt) * mt, "uniform-50/50");

        // Power-of-2 first piece
        for (int64_t p = 2048; p < total_size; p *= 2)
        {
            int64_t rem = total_size - p;
            if (rem >= min_sub && p >= min_sub)
                add(p, "pow2-" + std::to_string(p/1024) + "k");
        }

        // One asymmetric candidate
        add(((int64_t)(total_size * 0.40) / mt) * mt, "asym-40/60");

        // Wave-aligned candidates: find split points where BOTH sub-problems
        // produce workgroup counts that divide evenly into 256 CUs (zero tail waste).
        // This eliminates dispatch wave tail effects for both sub-problems.
        //
        // For M-split with MT_M=256, MT_N varies per sub-problem but we use
        // mt (the passed macrotile_size) as approximation.
        // WG count for sub = ceil(sub_size/mt) * ceil(other_dim/mt_other).
        // We want: ceil(s1/MT_M) * N_tiles mod NUM_CUS == 0 for both pieces.
        {
            int64_t mt_m = 256; // dominant MT_M on gfx950
            // N-tiles is constant across M-split sub-problems
            int64_t n_tiles = (other_dim + mt - 1) / mt;

            if (n_tiles > 0)
            {
                // We need ceil(s1/mt_m) * n_tiles mod 256 == 0
                // → ceil(s1/mt_m) mod (256 / gcd(n_tiles, 256)) == 0
                int64_t g = std::__gcd(n_tiles, (int64_t)NUM_CUS);
                int64_t m_tiles_step = NUM_CUS / g; // ceil(s1/mt_m) must be multiple of this

                // Generate candidates at each wave-aligned boundary
                for (int64_t m_tiles = m_tiles_step; m_tiles * mt_m < total_size; m_tiles += m_tiles_step)
                {
                    int64_t s1 = m_tiles * mt_m;
                    if (s1 >= min_sub && (total_size - s1) >= min_sub)
                        add(s1, "wave-" + std::to_string(m_tiles) + "mt");
                }
            }
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

    // Score any unscored candidates (3-way and XCD generators don't score
    // during generation; standard/brute-force already scored above).
    for (auto& cand : cands)
    {
        if (cand.origami_total_latency != 0.0)
            continue;
        double total = 0.0;
        bool   valid = true;
        for (size_t i = 0; i < cand.split_sizes.size(); i++)
        {
            int64_t M_sub = is_m_split ? cand.split_sizes[i] : M;
            int64_t N_sub = is_m_split ? N : cand.split_sizes[i];

            double lat = querySubProblemLatency(handle, M_sub, N_sub, K,
                                                transA, transB, a_type, b_type,
                                                c_type, d_type, compute_type,
                                                device_id);
            if (lat <= 0) { valid = false; break; }
            total += lat;
        }
        cand.origami_total_latency = valid ? total : 1e18;
    }

    // Sort all candidates by total analytical latency (best first).
    std::sort(cands.begin(), cands.end(),
              [](const OrigamiCandidate& a, const OrigamiCandidate& b) {
                  return a.origami_total_latency < b.origami_total_latency;
              });

    if (!cands.empty())
    {
        hipblaslt_cout << "Origami Analytical Ranking (" << cands.size() << " candidates, all paths):" << std::endl;
        for (size_t i = 0; i < std::min(cands.size(), (size_t)5); i++)
        {
            hipblaslt_cout << "  #" << (i+1) << " " << cands[i].label << " [";
            for (size_t j = 0; j < cands[i].split_sizes.size(); j++)
            {
                if (j) hipblaslt_cout << ",";
                hipblaslt_cout << cands[i].split_sizes[j];
            }
            hipblaslt_cout << "] latency=" << std::fixed << std::setprecision(0)
                          << cands[i].origami_total_latency << " cycles" << std::endl;
        }
    }

    getOrigamiCandidates() = cands;

    if (!cands.empty())
        return cands[0].split_sizes;
    return {};
}
