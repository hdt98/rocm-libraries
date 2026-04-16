/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <hipblaslt/hipblaslt.h>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <iostream>

// Split strategies
enum class SplitStrategy
{
    Auto = 0,       // Automatically choose best strategy
    Workgroup = 1,  // Optimize for workgroup alignment
    Memory = 2,     // Split based on memory constraints
    MOnly = 3,      // Force split along M dimension only
    NOnly = 4,      // Force split along N dimension only
    TwoD = 5        // 2D tiling (split both M and N)
};

// Structure to describe a sub-problem for multi-MacroTile GEMM
struct GemmSubProblem
{
    // Dimensions of this sub-problem
    int64_t m_size;
    int64_t n_size;
    int64_t k_size; // K is constant across all sub-problems

    // Starting offsets in the output matrix D
    int64_t m_offset;
    int64_t n_offset;

    // Byte offsets for matrix pointers
    size_t offset_A_bytes;
    size_t offset_B_bytes;
    size_t offset_C_bytes;
    size_t offset_D_bytes;
    size_t offset_E_bytes;
    size_t offset_bias_bytes;

    // Workgroup info (optional, for analysis)
    int expected_workgroups;
    int macrotile_m;
    int macrotile_n;
};

// Calculate data type size in bytes
inline size_t getDataTypeSize(hipDataType dataType)
{
    switch(dataType)
    {
    case HIP_R_32F:
    case HIP_R_32I:
        return 4;
    case HIP_R_16F:
    case HIP_R_16BF:
        return 2;
    case HIP_R_8I:
    case HIP_R_8F_E4M3:
    case HIP_R_8F_E5M2:
    case HIP_R_8F_E4M3_FNUZ:
    case HIP_R_8F_E5M2_FNUZ:
    case HIP_R_8U:
        return 1;
    case HIP_R_64F:
        return 8;
    default:
        return 2; // default to 2 bytes
    }
}

// Calculate byte offset for matrix A
inline size_t calculateOffsetA(int64_t m_offset,
                               int64_t k_offset,
                               int64_t lda,
                               hipblasOperation_t transA,
                               hipDataType dataType)
{
    size_t elem_size = getDataTypeSize(dataType);

    if(transA == HIPBLAS_OP_N)
    {
        // A is M x K in column-major, element A[i,j] is at position i + j*lda
        // Offset of m_offset rows means we start at element m_offset
        return m_offset * elem_size;
    }
    else
    {
        // A^T means we're accessing K x M, stored as M x K with K-offset in columns
        // k_offset would move in column direction, m_offset moves in row direction
        return (m_offset + k_offset * lda) * elem_size;
    }
}

// Calculate byte offset for matrix B
inline size_t calculateOffsetB(int64_t n_offset,
                               int64_t k_offset,
                               int64_t ldb,
                               hipblasOperation_t transB,
                               hipDataType dataType)
{
    size_t elem_size = getDataTypeSize(dataType);

    if(transB == HIPBLAS_OP_N)
    {
        // B is K x N in column-major, element B[i,j] is at position i + j*ldb
        // Offset of n_offset columns means we move n_offset columns
        return n_offset * ldb * elem_size;
    }
    else
    {
        // B^T means physical matrix is N x K, offset n_offset rows
        return n_offset * elem_size;
    }
}

// Calculate byte offset for matrices C/D/E (all have same layout)
inline size_t calculateOffsetCD(int64_t m_offset,
                               int64_t n_offset,
                               int64_t ld,
                               hipDataType dataType)
{
    size_t elem_size = getDataTypeSize(dataType);

    // C/D/E are M x N, column-major
    // offset = m_offset + n_offset * ld
    return (m_offset + n_offset * ld) * elem_size;
}

// Calculate byte offset for bias vector
inline size_t calculateOffsetBias(int64_t m_offset,
                                  hipDataType dataType)
{
    size_t elem_size = getDataTypeSize(dataType);
    // Bias is a vector along M dimension
    return m_offset * elem_size;
}

// Calculate memory requirements for a GEMM problem
inline size_t calculateMemoryRequirement(int64_t M, int64_t N, int64_t K,
                                         hipDataType a_type, hipDataType b_type,
                                         hipDataType c_type, hipDataType d_type,
                                         size_t workspace_size)
{
    size_t a_size = M * K * getDataTypeSize(a_type);
    size_t b_size = K * N * getDataTypeSize(b_type);
    size_t c_size = M * N * getDataTypeSize(c_type);
    size_t d_size = M * N * getDataTypeSize(d_type);

    // Total = input matrices + output matrix + workspace
    return a_size + b_size + c_size + d_size + workspace_size;
}

// Score how well a number aligns with target multiples
inline double scoreAlignment(int value, const std::vector<int>& targets)
{
    double best_score = 0.0;

    for(int target : targets)
    {
        if(target == 0) continue;

        int remainder = value % target;
        double score = 1.0 - (double)remainder / target;

        // Exact match is best
        if(remainder == 0)
            score = 2.0;

        best_score = std::max(best_score, score);
    }

    return best_score;
}

// Estimate workgroups for a given problem size and MacroTile
inline int estimateWorkgroups(int64_t M, int64_t N, int mt_m, int mt_n)
{
    if(mt_m <= 0 || mt_n <= 0)
        return 0;

    int wg_m = (M + mt_m - 1) / mt_m;
    int wg_n = (N + mt_n - 1) / mt_n;

    return wg_m * wg_n;
}

// Find optimal number of splits for workgroup alignment
inline int findOptimalSplitsForWorkgroups(int64_t M, int64_t N,
                                          int mt_m, int mt_n,
                                          int target_wgs,
                                          int num_CUs,
                                          bool split_m = true)
{
    int base_wgs = estimateWorkgroups(M, N, mt_m, mt_n);

    // If already well-aligned, don't split
    if(base_wgs % target_wgs == 0 || base_wgs % num_CUs == 0)
        return 1;

    int best_splits = 1;
    double best_score = scoreAlignment(base_wgs, {target_wgs, num_CUs, num_CUs * 2});

    // Try different split counts
    int max_splits = split_m ? std::min(16, (int)(M / mt_m)) : std::min(16, (int)(N / mt_n));

    for(int splits = 2; splits <= max_splits; splits++)
    {
        int64_t dim_per_split = split_m ? (M / splits) : (N / splits);
        int wgs_per_split = split_m ?
            estimateWorkgroups(dim_per_split, N, mt_m, mt_n) :
            estimateWorkgroups(M, dim_per_split, mt_m, mt_n);

        // Score this configuration
        double alignment_score = scoreAlignment(wgs_per_split, {target_wgs, num_CUs});

        // Prefer fewer splits if scores are close
        double overhead_penalty = 1.0 / splits;
        double total_score = alignment_score * 0.7 + overhead_penalty * 0.3;

        if(total_score > best_score)
        {
            best_score = total_score;
            best_splits = splits;
        }
    }

    return best_splits;
}

// Find optimal splits based on memory constraints
inline int findOptimalSplitsForMemory(int64_t M, int64_t N, int64_t K,
                                      hipDataType a_type, hipDataType b_type,
                                      hipDataType c_type, hipDataType d_type,
                                      size_t available_memory,
                                      size_t workspace_size)
{
    size_t total_memory = calculateMemoryRequirement(M, N, K, a_type, b_type,
                                                      c_type, d_type, workspace_size);

    if(total_memory <= available_memory)
        return 1;

    // Calculate how many splits needed to fit in memory
    // Split along M dimension (most common case)
    int splits = 2;
    while(splits <= 16)
    {
        int64_t M_per_split = M / splits;
        size_t memory_per_split = calculateMemoryRequirement(M_per_split, N, K,
                                                              a_type, b_type,
                                                              c_type, d_type,
                                                              workspace_size);

        if(memory_per_split <= available_memory)
            return splits;

        splits++;
    }

    return 16; // Max splits if still doesn't fit
}

// Main splitting function with intelligent strategy selection
inline std::vector<GemmSubProblem> splitGemmProblem(
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc, int64_t ldd, int64_t lde,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    hipDataType aux_type, hipDataType bias_type,
    int split_strategy = 0,      // SplitStrategy enum
    int manual_num_splits = 0,    // 0 = auto
    int target_wgs = 256,
    int macrotile_m = 128,        // Estimated MacroTile size (if known)
    int macrotile_n = 128,
    int num_CUs = 256,            // MI355X has 256 CUs
    size_t available_memory = 512ULL * 1024 * 1024,  // 512 MiB default
    size_t workspace_size = 128ULL * 1024 * 1024)
{
    std::vector<GemmSubProblem> subProblems;

    int num_splits = 1;
    bool split_along_m = true;
    bool split_along_n = false;

    // Determine number of splits based on strategy
    if(manual_num_splits > 0)
    {
        // Manual override
        num_splits = std::min(manual_num_splits, 16);
    }
    else
    {
        switch(static_cast<SplitStrategy>(split_strategy))
        {
        case SplitStrategy::Auto:
        {
            // Try workgroup-based splitting first
            int wg_splits = findOptimalSplitsForWorkgroups(M, N, macrotile_m, macrotile_n,
                                                           target_wgs, num_CUs, true);

            // Check if memory constrained
            int mem_splits = findOptimalSplitsForMemory(M, N, K, a_type, b_type,
                                                         c_type, d_type,
                                                         available_memory, workspace_size);

            // Take the larger of the two
            num_splits = std::max(wg_splits, mem_splits);
            break;
        }
        case SplitStrategy::Workgroup:
            num_splits = findOptimalSplitsForWorkgroups(M, N, macrotile_m, macrotile_n,
                                                        target_wgs, num_CUs, true);
            break;
        case SplitStrategy::Memory:
            num_splits = findOptimalSplitsForMemory(M, N, K, a_type, b_type,
                                                    c_type, d_type,
                                                    available_memory, workspace_size);
            break;
        case SplitStrategy::MOnly:
            num_splits = 2;  // Simple 2-way split along M
            split_along_m = true;
            split_along_n = false;
            break;
        case SplitStrategy::NOnly:
            num_splits = 2;  // Simple 2-way split along N
            split_along_m = false;
            split_along_n = true;
            break;
        case SplitStrategy::TwoD:
            // 2D tiling: split both dimensions
            num_splits = 4;  // 2x2 grid
            split_along_m = true;
            split_along_n = true;
            break;
        }
    }

    // Clamp to reasonable range
    num_splits = std::max(1, std::min(num_splits, 16));

    // If only 1 split, return single problem
    if(num_splits == 1)
    {
        GemmSubProblem sub;
        sub.m_size = M;
        sub.n_size = N;
        sub.k_size = K;
        sub.m_offset = 0;
        sub.n_offset = 0;
        sub.offset_A_bytes = 0;
        sub.offset_B_bytes = 0;
        sub.offset_C_bytes = 0;
        sub.offset_D_bytes = 0;
        sub.offset_E_bytes = 0;
        sub.offset_bias_bytes = 0;
        sub.expected_workgroups = estimateWorkgroups(M, N, macrotile_m, macrotile_n);
        sub.macrotile_m = macrotile_m;
        sub.macrotile_n = macrotile_n;
        subProblems.push_back(sub);
        return subProblems;
    }

    // Generate sub-problems based on split configuration
    if(split_along_m && !split_along_n)
    {
        // Split along M dimension only
        for(int i = 0; i < num_splits; i++)
        {
            int64_t m_start = (M * i) / num_splits;
            int64_t m_end = (M * (i + 1)) / num_splits;
            int64_t m_size = m_end - m_start;

            GemmSubProblem sub;
            sub.m_size = m_size;
            sub.n_size = N;
            sub.k_size = K;
            sub.m_offset = m_start;
            sub.n_offset = 0;
            sub.offset_A_bytes = calculateOffsetA(m_start, 0, lda, transA, a_type);
            sub.offset_B_bytes = 0;  // B is shared across all M splits
            sub.offset_C_bytes = calculateOffsetCD(m_start, 0, ldc, c_type);
            sub.offset_D_bytes = calculateOffsetCD(m_start, 0, ldd, d_type);
            sub.offset_E_bytes = calculateOffsetCD(m_start, 0, lde, aux_type);
            sub.offset_bias_bytes = calculateOffsetBias(m_start, bias_type);
            sub.expected_workgroups = estimateWorkgroups(m_size, N, macrotile_m, macrotile_n);
            sub.macrotile_m = macrotile_m;
            sub.macrotile_n = macrotile_n;

            subProblems.push_back(sub);
        }
    }
    else if(!split_along_m && split_along_n)
    {
        // Split along N dimension only
        for(int i = 0; i < num_splits; i++)
        {
            int64_t n_start = (N * i) / num_splits;
            int64_t n_end = (N * (i + 1)) / num_splits;
            int64_t n_size = n_end - n_start;

            GemmSubProblem sub;
            sub.m_size = M;
            sub.n_size = n_size;
            sub.k_size = K;
            sub.m_offset = 0;
            sub.n_offset = n_start;
            sub.offset_A_bytes = 0;  // A is shared across all N splits
            sub.offset_B_bytes = calculateOffsetB(n_start, 0, ldb, transB, b_type);
            sub.offset_C_bytes = calculateOffsetCD(0, n_start, ldc, c_type);
            sub.offset_D_bytes = calculateOffsetCD(0, n_start, ldd, d_type);
            sub.offset_E_bytes = calculateOffsetCD(0, n_start, lde, aux_type);
            sub.offset_bias_bytes = 0;  // Bias along M, not affected by N split
            sub.expected_workgroups = estimateWorkgroups(M, n_size, macrotile_m, macrotile_n);
            sub.macrotile_m = macrotile_m;
            sub.macrotile_n = macrotile_n;

            subProblems.push_back(sub);
        }
    }
    else if(split_along_m && split_along_n)
    {
        // 2D tiling: split both M and N
        int splits_m = (int)sqrt(num_splits);
        int splits_n = num_splits / splits_m;

        for(int i = 0; i < splits_m; i++)
        {
            int64_t m_start = (M * i) / splits_m;
            int64_t m_end = (M * (i + 1)) / splits_m;
            int64_t m_size = m_end - m_start;

            for(int j = 0; j < splits_n; j++)
            {
                int64_t n_start = (N * j) / splits_n;
                int64_t n_end = (N * (j + 1)) / splits_n;
                int64_t n_size = n_end - n_start;

                GemmSubProblem sub;
                sub.m_size = m_size;
                sub.n_size = n_size;
                sub.k_size = K;
                sub.m_offset = m_start;
                sub.n_offset = n_start;
                sub.offset_A_bytes = calculateOffsetA(m_start, 0, lda, transA, a_type);
                sub.offset_B_bytes = calculateOffsetB(n_start, 0, ldb, transB, b_type);
                sub.offset_C_bytes = calculateOffsetCD(m_start, n_start, ldc, c_type);
                sub.offset_D_bytes = calculateOffsetCD(m_start, n_start, ldd, d_type);
                sub.offset_E_bytes = calculateOffsetCD(m_start, n_start, lde, aux_type);
                sub.offset_bias_bytes = calculateOffsetBias(m_start, bias_type);
                sub.expected_workgroups = estimateWorkgroups(m_size, n_size, macrotile_m, macrotile_n);
                sub.macrotile_m = macrotile_m;
                sub.macrotile_n = macrotile_n;

                subProblems.push_back(sub);
            }
        }
    }

    return subProblems;
}
