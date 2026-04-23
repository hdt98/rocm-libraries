/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Multi-MacroTile: splits a GEMM into sub-problems solved with per-subproblem
 * kernel selection.  Only the Origami-based empirical split search (S17/S18)
 * is retained; all legacy uniform / heuristic strategies have been removed.
 *
 *******************************************************************************/

#pragma once

#include <hipblaslt/hipblaslt.h>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

// Forward declaration -- implementation in multi_macrotile_origami_improved.hpp
std::vector<int64_t> computeOrigamiOptimizedSplitsWithHandle(
    hipblasLtHandle_t handle, int64_t total_size, int num_splits,
    int macrotile_size, int64_t other_dim, int64_t K, bool is_m_split,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    hipblasComputeType_t compute_type,
    bool brute_force = false,
    bool xcd_aware = false);

// ── Sub-problem descriptor ──────────────────────────────────────────────────

struct GemmSubProblem
{
    int64_t m_size, n_size, k_size;
    int64_t m_offset, n_offset;
    size_t offset_A_bytes, offset_B_bytes;
    size_t offset_C_bytes, offset_D_bytes;
    size_t offset_E_bytes, offset_bias_bytes;
    int expected_workgroups;
    int macrotile_m, macrotile_n;
};

// Pre-created context used by the timing hot-loop (avoids per-iteration
// layout create / destroy / heuristic query overhead).
struct SubProblemContext
{
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    hipblasLtMatmulDesc_t   matmul_desc; // per-subproblem descriptor for correct kernel selection
    hipblasLtMatmulAlgo_t   algo;
    void *A_ptr, *B_ptr, *C_ptr, *D_ptr;
    bool valid;
    bool owns_matmul_desc; // true if we created matmul_desc and need to destroy it
};

// ── Helpers ─────────────────────────────────────────────────────────────────

inline size_t getDataTypeSize(hipDataType dt)
{
    switch (dt)
    {
    case HIP_R_64F:                      return 8;
    case HIP_R_32F: case HIP_R_32I:     return 4;
    case HIP_R_16F: case HIP_R_16BF:    return 2;
    case HIP_R_8I:  case HIP_R_8U:
    case HIP_R_8F_E4M3: case HIP_R_8F_E5M2:
    case HIP_R_8F_E4M3_FNUZ: case HIP_R_8F_E5M2_FNUZ:
                                         return 1;
    default:                             return 2;
    }
}

inline size_t calculateOffsetA(int64_t m_off, int64_t k_off, int64_t lda,
                                hipblasOperation_t transA, hipDataType dt)
{
    size_t e = getDataTypeSize(dt);
    // transA=N: A is [M x K] col-major, lda >= M.  Element (m, k) at m + k*lda.
    //           Skipping m_off rows: offset = m_off + k_off * lda.
    // transA=T: A is [K x M] col-major, lda >= K.  Element (k, m) at k + m*lda.
    //           Skipping m_off columns: offset = k_off + m_off * lda.
    return (transA == HIPBLAS_OP_N)
        ? (m_off + k_off * lda) * e
        : (k_off + m_off * lda) * e;
}

inline size_t calculateOffsetB(int64_t n_off, int64_t k_off, int64_t ldb,
                                hipblasOperation_t transB, hipDataType dt)
{
    size_t e = getDataTypeSize(dt);
    return (transB == HIPBLAS_OP_N) ? n_off * ldb * e : n_off * e;
}

inline size_t calculateOffsetCD(int64_t m_off, int64_t n_off, int64_t ld,
                                 hipDataType dt)
{
    return (m_off + n_off * ld) * getDataTypeSize(dt);
}

inline size_t calculateOffsetBias(int64_t m_off, hipDataType dt)
{
    return m_off * getDataTypeSize(dt);
}

inline int estimateWorkgroups(int64_t M, int64_t N, int mt_m, int mt_n)
{
    if (mt_m <= 0 || mt_n <= 0) return 0;
    return (int)(((M + mt_m - 1) / mt_m) * ((N + mt_n - 1) / mt_n));
}

// ── Guard: should we attempt multi-MacroTile for this problem? ──────────────
// Derived from 716-point FP16 + 651-point BF16 benchmark analysis on MI350X.
// Rejects problem shapes where splitting consistently causes regressions.

inline bool shouldUseMultiMacroTile(int64_t M, int64_t N, int64_t K)
{
    int64_t minMN = std::min(M, N);
    int64_t maxMN = std::max(M, N);

    // min(M,N) <= 1024 with M-split creates tiny sub-problems along the
    // non-split axis → massive CU underutilization (avg -28.9% in benchmarks).
    if (minMN <= 1024)
        return false;

    // Both M,N <= 4096: single kernel already achieves ~100% CU utilization
    // in one dispatch wave (256 WGs). Splitting halves parallelism per
    // sub-problem and adds launch overhead (avg -12.1%).
    if (maxMN <= 4096)
        return false;

    // Very small K doesn't amortize the overhead of two kernel launches;
    // the compute-to-launch-cost ratio is too low (avg -3.2%).
    if (K <= 1024)
        return false;

    // Safe zone: M >= 5120 AND N >= 2048 AND K >= 2048.
    // Under these conditions the benchmark win rate exceeds 75% with
    // effectively zero catastrophic regressions.
    return true;
}

// ── Main entry point ────────────────────────────────────────────────────────
// Splits a GEMM along M (strategy 17) or N (strategy 18) using the Origami
// empirical search.  Returns a single-element vector when splitting is
// disabled or Origami declines.

inline std::vector<GemmSubProblem> splitGemmProblem(
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc, int64_t ldd, int64_t lde,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    hipDataType aux_type, hipDataType bias_type,
    int split_strategy,
    int num_splits,
    int macrotile_m = 128, int macrotile_n = 128,
    hipblasLtHandle_t handle = nullptr,
    hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F)
{
    std::vector<GemmSubProblem> subs;

    // Single-problem fallback
    auto makeSingle = [&]() {
        GemmSubProblem s{};
        s.m_size = M; s.n_size = N; s.k_size = K;
        s.expected_workgroups = estimateWorkgroups(M, N, macrotile_m, macrotile_n);
        s.macrotile_m = macrotile_m; s.macrotile_n = macrotile_n;
        subs.push_back(s);
        return subs;
    };

    // Strategy mapping:
    // 17/19 = M-split (19 = brute-force)
    // 18/20 = N-split (20 = brute-force)
    // 21 = 3-way M-split, 22 = 3-way N-split
    // 23 = XCD-aware M-split, 24 = XCD-aware N-split
    bool is_m_split = (split_strategy == 17 || split_strategy == 19 || split_strategy == 21 || split_strategy == 23);
    bool brute_force = (split_strategy == 19 || split_strategy == 20);
    bool three_way = (split_strategy == 21 || split_strategy == 22);
    bool xcd_aware = (split_strategy == 23 || split_strategy == 24);

    if (three_way && num_splits <= 0) num_splits = 3;
    else if (num_splits <= 0) num_splits = 2;
    num_splits = std::min(num_splits, 16);

    std::vector<int64_t> split_sizes;
    if (handle)
    {
        int64_t dim   = is_m_split ? M : N;
        int64_t other = is_m_split ? N : M;
        int     mt    = is_m_split ? macrotile_m : macrotile_n;

        split_sizes = computeOrigamiOptimizedSplitsWithHandle(
            handle, dim, num_splits, mt, other, K, is_m_split,
            transA, transB, a_type, b_type, c_type, d_type, compute_type,
            brute_force, xcd_aware);
    }

    if (split_sizes.empty())
        return makeSingle();

    num_splits = (int)split_sizes.size();

    // Build sub-problems from split sizes
    int64_t offset = 0;
    for (int i = 0; i < num_splits; i++)
    {
        GemmSubProblem s{};
        int64_t sz = split_sizes[i];

        if (is_m_split)
        {
            s.m_size = sz;        s.n_size = N;   s.k_size = K;
            s.m_offset = offset;  s.n_offset = 0;
            s.offset_A_bytes    = calculateOffsetA(offset, 0, lda, transA, a_type);
            s.offset_B_bytes    = 0;
            s.offset_C_bytes    = calculateOffsetCD(offset, 0, ldc, c_type);
            s.offset_D_bytes    = calculateOffsetCD(offset, 0, ldd, d_type);
            s.offset_E_bytes    = calculateOffsetCD(offset, 0, lde, aux_type);
            s.offset_bias_bytes = calculateOffsetBias(offset, bias_type);
        }
        else
        {
            s.m_size = M;         s.n_size = sz;  s.k_size = K;
            s.m_offset = 0;       s.n_offset = offset;
            s.offset_A_bytes    = 0;
            s.offset_B_bytes    = calculateOffsetB(offset, 0, ldb, transB, b_type);
            s.offset_C_bytes    = calculateOffsetCD(0, offset, ldc, c_type);
            s.offset_D_bytes    = calculateOffsetCD(0, offset, ldd, d_type);
            s.offset_E_bytes    = calculateOffsetCD(0, offset, lde, aux_type);
            s.offset_bias_bytes = 0;
        }

        s.expected_workgroups = estimateWorkgroups(s.m_size, s.n_size, macrotile_m, macrotile_n);
        s.macrotile_m = macrotile_m;
        s.macrotile_n = macrotile_n;

        subs.push_back(s);
        offset += sz;
    }

    return subs;
}
