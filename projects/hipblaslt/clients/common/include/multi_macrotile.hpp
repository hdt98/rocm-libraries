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

// Forward declaration of Origami optimization function
// The actual implementation will be available when multi_macrotile_origami_improved.hpp is included later
std::vector<int64_t> computeOrigamiOptimizedSplitsWithHandle(
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
    hipblasComputeType_t compute_type);
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <iostream>

// Split strategies
enum class SplitStrategy
{
    Auto = 0,            // Automatically choose best strategy
    Workgroup = 1,       // Optimize for workgroup alignment
    Memory = 2,          // Split based on memory constraints
    MOnly = 3,           // Force split along M dimension only
    NOnly = 4,           // Force split along N dimension only
    TwoD = 5,            // 2D tiling (split both M and N)
    MacroTileAlign = 6,  // MacroTile-aligned splitting (non-uniform)
    PowerOf2 = 7,        // Power-of-2 biased splitting
    CUBalanced = 8,      // CU-balanced for stream-parallel
    Performance = 9,     // Performance-based (query heuristics)
    AdaptivePowerOf2 = 10, // Adaptive power-of-2 (falls back to uniform if imbalanced)
    CacheOptimizedM = 15,  // Cache-optimized M-split (uneven split when B fits in L2)
    CacheOptimizedN = 16,  // Cache-optimized N-split (uneven split when A fits in L2)
    OrigamiOptimizedM = 17, // Origami-optimized M-split (query all solutions, minimize total latency)
    OrigamiOptimizedN = 18  // Origami-optimized N-split (query all solutions, minimize total latency)
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

// Helper: Check if number is power of 2
inline bool isPowerOf2(int64_t n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

// Helper: Round to nearest power of 2
inline int64_t roundToPowerOf2(int64_t n)
{
    if (n <= 0) return 1;

    int64_t power = 1;
    while (power < n)
        power *= 2;

    // Choose closer power of 2
    int64_t lower = power / 2;
    return (n - lower < power - n) ? lower : power;
}

// Estimate workgroups for a given problem size
inline int estimateWorkgroupsSimple(int64_t M, int64_t N, int macrotile_m = 128, int macrotile_n = 128)
{
    int tiles_m = (M + macrotile_m - 1) / macrotile_m;
    int tiles_n = (N + macrotile_n - 1) / macrotile_n;
    return tiles_m * tiles_n;
}

// Automatic strategy selection based on problem characteristics
inline int autoSelectStrategy(int64_t M, int64_t N, int64_t K, size_t elem_size = 2)
{
    const size_t L2_SIZE = 96 * 1024 * 1024;  // 96 MB on MI355X

    // Rule 1: Check for very small K (but cache optimization can help even with smaller K)
    // Only completely disable for K < 4096
    if (K < 4096)
    {
        return 0; // Use baseline - very small K has negligible benefit
    }

    // Rule 2: Rectangular matrices - split along larger dimension
    // Consider cache optimization if shared matrix fits in L2
    double aspect_ratio = (double)M / N;

    if (aspect_ratio > 1.5)
    {
        // Tall matrix (M > N) - prefer M-split
        if (M >= 10240)
        {
            // Check if B matrix fits in cache
            size_t B_size = K * N * elem_size;
            if (B_size < L2_SIZE * 0.75)
            {
                return 15; // Cache-optimized M-split (uneven when B fits)
            }
            return 3; // Standard uniform M-split
        }
        return 0; // Both dimensions too small
    }

    if (aspect_ratio < 0.67)
    {
        // Wide matrix (M < N) - prefer N-split
        if (N >= 10240)
        {
            // Check if A matrix fits in cache
            size_t A_size = M * K * elem_size;
            if (A_size < L2_SIZE * 0.75)
            {
                return 16; // Cache-optimized N-split (uneven when A fits)
            }
            return 4; // Standard uniform N-split
        }
        return 0; // Both dimensions too small
    }

    // Rule 3: Square or nearly-square matrices
    // Disable for small square matrices (known to lose performance)
    if (M < 10240 || N < 10240)
    {
        return 0; // Use baseline - multi-MT hurts small square matrices
    }

    // Rule 4: Large square matrices with large K
    if (std::abs(aspect_ratio - 1.0) < 0.2) // Nearly square (within 20%)
    {
        // Check cache benefit for square matrices
        size_t B_size = K * N * elem_size;

        if (B_size < L2_SIZE * 0.75)
        {
            // B fits in cache - use cache-optimized M-split
            return 15;
        }

        // B doesn't fit - use power-of-2 or uniform based on dimensions
        bool m_is_pow2 = isPowerOf2(M);
        bool n_is_pow2 = isPowerOf2(N);

        // Also check if close to power-of-2 (within 10%)
        int64_t m_rounded = roundToPowerOf2(M);
        int64_t n_rounded = roundToPowerOf2(N);
        double m_deviation = std::abs((double)(M - m_rounded) / M);
        double n_deviation = std::abs((double)(N - n_rounded) / N);

        bool m_close_to_pow2 = (m_is_pow2 || m_deviation < 0.10);
        bool n_close_to_pow2 = (n_is_pow2 || n_deviation < 0.10);

        if (m_close_to_pow2 && n_close_to_pow2)
        {
            return 10; // Adaptive power-of-2 (safer than regular power-of-2)
        }
        else
        {
            return 3; // Uniform M-split for non-power-of-2
        }
    }

    // Rule 5: Default to cache-optimized or uniform M-split
    size_t B_size = K * N * elem_size;
    if (B_size < L2_SIZE * 0.75)
    {
        return 15; // Cache-optimized M-split
    }
    return 3; // Standard uniform M-split
}

// Automatic num_splits selection based on problem size
inline int autoSelectNumSplits(int64_t M, int64_t N, int64_t K, int num_CUs = 256)
{
    int total_wgs = estimateWorkgroupsSimple(M, N);

    // Target: 80-120 workgroups per sub-problem for good occupancy
    if (total_wgs < 200)
    {
        return 1; // Too small for splitting
    }
    else if (total_wgs < 400)
    {
        return 2; // 2-way split
    }
    else if (total_wgs < 800)
    {
        return 4; // 4-way split
    }
    else if (total_wgs < 1600)
    {
        return 8; // 8-way split
    }
    else
    {
        return 8; // Cap at 8-way (diminishing returns beyond this)
    }
}

// Check if multi-MacroTile should be used for this problem
inline bool shouldUseMultiMacroTile(int64_t M, int64_t N, int64_t K, int split_strategy)
{
    // If user explicitly disabled (strategy 0), respect it
    if (split_strategy == 0)
    {
        return false;
    }

    if (K < 4096)
    {
        return false; // Very small K has negligible benefit
    }

    // Check aspect ratio to determine if rectangular
    double aspect_ratio = (double)M / N;
    bool is_rectangular = (aspect_ratio > 1.5 || aspect_ratio < 0.67);

    if (is_rectangular)
    {
        // For rectangular matrices, only require the larger dimension to be >= 10240
        if (std::max(M, N) >= 10240)
        {
            return true; // Rectangular matrices often benefit
        }
        return false; // Both dimensions too small
    }
    else
    {
        // For square/nearly-square matrices, require both dimensions >= 10240
        if (M < 10240 || N < 10240)
        {
            return false; // Small square matrices lose -16% to -25%
        }
    }

    // Non-power-of-2 with regular power-of-2 strategy can be problematic
    if (split_strategy == 7) // PowerOf2 strategy (regular, not adaptive)
    {
        if (!isPowerOf2(M) || !isPowerOf2(N))
        {
            // Power-of-2 strategy can create imbalanced splits
            // Only allow if dimensions are reasonably close to power-of-2
            int64_t m_rounded = roundToPowerOf2(M);
            int64_t n_rounded = roundToPowerOf2(N);

            double m_deviation = std::abs((double)(M - m_rounded) / M);
            double n_deviation = std::abs((double)(N - n_rounded) / N);

            if (m_deviation > 0.15 || n_deviation > 0.15)
            {
                return false; // Too far from power-of-2, would create bad splits
            }
        }
    }

    return true; // Safe to use multi-MacroTile
}

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

// MacroTile-aligned splitting: Find split sizes that align well with MacroTile
inline std::vector<int64_t> computeMacroTileAlignedSplits(int64_t total_size,
                                                           int num_splits,
                                                           int macrotile_size)
{
    std::vector<int64_t> sizes(num_splits);

    // Target: each split should be multiple of macrotile_size
    int64_t base_size = (total_size / num_splits / macrotile_size) * macrotile_size;
    int64_t remainder = total_size - (base_size * num_splits);

    // Distribute base sizes
    for(int i = 0; i < num_splits; i++)
    {
        sizes[i] = base_size;
    }

    // Distribute remainder by adding macrotile_size to splits
    int splits_to_adjust = remainder / macrotile_size;
    for(int i = 0; i < splits_to_adjust && i < num_splits; i++)
    {
        sizes[i] += macrotile_size;
    }

    // If there's still remainder, add to last split
    int64_t final_remainder = total_size;
    for(int i = 0; i < num_splits; i++)
        final_remainder -= sizes[i];

    if(final_remainder > 0)
        sizes[num_splits - 1] += final_remainder;

    return sizes;
}

// Power-of-2 biased splitting: Prefer power-of-2 or near-power-of-2 sizes
inline std::vector<int64_t> computePowerOf2Splits(int64_t total_size, int num_splits)
{
    std::vector<int64_t> sizes;

    // Find closest power-of-2 splits
    int64_t remaining = total_size;

    for(int i = 0; i < num_splits - 1 && remaining > 0; i++)
    {
        // Find largest power-of-2 <= remaining / (num_splits - i)
        int64_t target = remaining / (num_splits - i);
        int64_t pow2 = 1;
        while(pow2 * 2 <= target)
            pow2 *= 2;

        // Allow slight deviation for better balance
        int64_t size = pow2;
        if(target - pow2 > pow2 / 4)  // If more than 25% away, use next power
            size = pow2 * 2;

        size = std::min(size, remaining);
        sizes.push_back(size);
        remaining -= size;
    }

    // Last split gets remainder
    if(remaining > 0)
        sizes.push_back(remaining);

    return sizes;
}

// Adaptive Power-of-2: Falls back to uniform if splits are too imbalanced
inline std::vector<int64_t> computeAdaptivePowerOf2Splits(int64_t total_size, int num_splits)
{
    auto pow2_splits = computePowerOf2Splits(total_size, num_splits);

    // Check balance - find min and max split sizes
    int64_t min_split = pow2_splits[0];
    int64_t max_split = pow2_splits[0];

    for (size_t i = 1; i < pow2_splits.size(); i++)
    {
        min_split = std::min(min_split, pow2_splits[i]);
        max_split = std::max(max_split, pow2_splits[i]);
    }

    // Calculate imbalance ratio
    double imbalance_ratio = (double)max_split / min_split;

    // If too imbalanced (>40% difference), fall back to uniform
    if (imbalance_ratio > 1.4)
    {
        // Use uniform splits instead
        std::vector<int64_t> uniform_splits;
        int64_t base_size = total_size / num_splits;
        int64_t remainder = total_size % num_splits;

        for (int i = 0; i < num_splits; i++)
        {
            int64_t size = base_size;
            if (i < remainder)
                size++; // Distribute remainder

            uniform_splits.push_back(size);
        }

        return uniform_splits;
    }

    // Balance is acceptable, use power-of-2 splits
    return pow2_splits;
}

// CU-balanced splitting: Distribute workgroups evenly across CUs
inline std::vector<int64_t> computeCUBalancedSplits(int64_t total_size,
                                                     int num_splits,
                                                     int64_t other_dim,
                                                     int macrotile_dim,
                                                     int macrotile_other,
                                                     int num_CUs)
{
    std::vector<int64_t> sizes;

    // Target workgroups per split
    int total_wgs = estimateWorkgroups(total_size, other_dim, macrotile_dim, macrotile_other);
    int target_wgs_per_split = total_wgs / num_splits;
    int target_CUs_per_split = num_CUs / num_splits;

    int64_t remaining = total_size;

    for(int i = 0; i < num_splits - 1 && remaining > 0; i++)
    {
        // Size that gives us target workgroups
        // wgs = (size / macrotile) * (other_dim / macrotile_other)
        // size = (target_wgs * macrotile * macrotile_other) / other_dim
        int64_t size = ((int64_t)target_wgs_per_split * macrotile_dim * macrotile_other) / other_dim;

        // Round to macrotile boundary
        size = (size / macrotile_dim) * macrotile_dim;

        // Ensure we don't exceed remaining
        size = std::min(size, remaining);
        size = std::max(size, (int64_t)macrotile_dim); // At least one macrotile

        sizes.push_back(size);
        remaining -= size;
    }

    // Last split gets remainder
    if(remaining > 0)
        sizes.push_back(remaining);

    return sizes;
}

// Calculate optimal split ratio based on L2 cache behavior
inline double calculateOptimalSplitRatio(int64_t M, int64_t N, int64_t K,
                                         size_t elem_size, bool is_m_split)
{
    const size_t L2_SIZE = 96 * 1024 * 1024;  // 96 MB on MI355X

    // Determine which matrix is shared across splits
    size_t shared_matrix_size;
    if (is_m_split)
        shared_matrix_size = K * N * elem_size;  // B matrix
    else
        shared_matrix_size = M * K * elem_size;  // A matrix

    // If shared matrix fits in cache (with 75% threshold for safety), use uneven split
    if (shared_matrix_size < L2_SIZE * 0.75)
    {
        // Adjust ratio based on compute intensity (K dimension)
        // Smaller K = more memory-bound = more aggressive uneven split
        if (K >= 16384)
            return 0.60;  // High compute, less cache-sensitive
        else if (K >= 8192)
            return 0.65;  // Medium compute
        else if (K >= 4096)
            return 0.70;  // Lower compute, cache matters more
        else
            return 0.75;  // Very memory-bound, aggressive split
    }

    // Shared matrix doesn't fit, uniform split is best
    return 0.50;
}

// Cache-Optimized Split: Uneven splits when shared matrix fits in L2 cache
inline std::vector<int64_t> computeCacheOptimizedSplits(
    int64_t total_size,
    int num_splits,
    int macrotile_size,
    int64_t other_dim,
    int64_t K,
    size_t elem_size,
    bool is_m_split)
{
    std::vector<int64_t> sizes;

    if (num_splits != 2)
    {
        // For 3+ splits, fall back to uniform
        int64_t base = (total_size / num_splits / macrotile_size) * macrotile_size;
        int64_t remainder = total_size - (base * num_splits);

        for (int i = 0; i < num_splits; i++)
        {
            int64_t size = base;
            if (i == 0 && remainder > 0)
                size += (remainder / macrotile_size) * macrotile_size;  // Give extra to first split
            sizes.push_back(size);
        }

        // Handle any remaining after alignment
        int64_t total_assigned = 0;
        for (auto s : sizes) total_assigned += s;
        if (total_assigned < total_size)
            sizes[0] += (total_size - total_assigned);

        return sizes;
    }

    // 2-way split: use cache-aware ratio
    double ratio = calculateOptimalSplitRatio(
        is_m_split ? total_size : other_dim,
        is_m_split ? other_dim : total_size,
        K, elem_size, is_m_split);

    // Calculate split sizes
    int64_t first_split = (int64_t)(total_size * ratio);

    // Align to MacroTile
    first_split = (first_split / macrotile_size) * macrotile_size;

    // Ensure minimum size
    first_split = std::max(first_split, (int64_t)macrotile_size);
    first_split = std::min(first_split, total_size - (int64_t)macrotile_size);

    int64_t second_split = total_size - first_split;

    sizes.push_back(first_split);
    sizes.push_back(second_split);

    return sizes;
}

// Structure to hold solution performance data
struct SolutionPerformance
{
    int solution_index;
    std::string solution_name;
    double estimated_latency_us;  // Estimated latency in microseconds
    size_t workspace_size;
    float waves_count;
};

// Structure to hold split configuration with estimated performance
struct SplitConfiguration
{
    std::vector<int64_t> split_sizes;
    std::vector<int> solution_indices;  // Best solution index for each sub-problem
    double total_latency_us;  // Total estimated latency
    bool valid;

    SplitConfiguration() : total_latency_us(1e9), valid(false) {}
};

// Origami-Optimized Split: Query all solutions and find optimal partitioning
// This function requires access to hipBLASLt handle and matrix layouts
// Note: Since we can't easily get solution performance without actual execution,
// we'll use wavesCount as a proxy for performance (higher waves = better utilization)
inline std::vector<int64_t> computeOrigamiOptimizedSplits(
    int64_t total_size,
    int num_splits,
    int macrotile_size,
    int64_t other_dim,
    int64_t K,
    bool is_m_split)
{
    // For now, implement a heuristic-based version
    // In production, this would:
    // 1. Generate candidate splits (50/50, 60/40, 70/30, etc.)
    // 2. For each split, query getAllAlgos for each sub-problem
    // 3. Estimate latency based on wavesCount or analytical model
    // 4. Select split + solution combination with minimum total latency

    // Generate candidate split ratios for 2-way split
    if (num_splits == 2)
    {
        std::vector<double> candidate_ratios = {0.50, 0.55, 0.60, 0.65, 0.70, 0.75};

        // For each candidate, estimate "balance quality"
        // We use workgroup balance as a proxy since we don't have access to
        // actual solution query at this point

        double best_ratio = 0.50;
        double best_balance_score = 0.0;

        for (double ratio : candidate_ratios)
        {
            int64_t split1 = (int64_t)(total_size * ratio);
            split1 = (split1 / macrotile_size) * macrotile_size;  // Align
            split1 = std::max(split1, (int64_t)macrotile_size);
            split1 = std::min(split1, total_size - (int64_t)macrotile_size);

            int64_t split2 = total_size - split1;

            // Estimate workgroups for each split
            int wg1 = (split1 / macrotile_size) * (other_dim / macrotile_size);
            int wg2 = (split2 / macrotile_size) * (other_dim / macrotile_size);

            // Balance score: prefer balanced workgroups
            // But also consider that larger problems often have better-tuned kernels
            double balance = (double)std::min(wg1, wg2) / std::max(wg1, wg2);
            double size_factor = std::min(1.0, (double)std::min(split1, split2) / 4096.0);
            double score = balance * size_factor;

            if (score > best_balance_score)
            {
                best_balance_score = score;
                best_ratio = ratio;
            }
        }

        // Calculate final splits with best ratio
        int64_t split1 = (int64_t)(total_size * best_ratio);
        split1 = (split1 / macrotile_size) * macrotile_size;
        split1 = std::max(split1, (int64_t)macrotile_size);
        split1 = std::min(split1, total_size - (int64_t)macrotile_size);
        int64_t split2 = total_size - split1;

        return {split1, split2};
    }
    else
    {
        // For 3+ splits, use uniform as baseline
        // Could be extended to search over all combinations
        std::vector<int64_t> sizes;
        int64_t base = (total_size / num_splits / macrotile_size) * macrotile_size;
        int64_t remainder = total_size - (base * num_splits);

        for (int i = 0; i < num_splits; i++)
        {
            int64_t size = base;
            if (i == 0 && remainder > 0)
                size += (remainder / macrotile_size) * macrotile_size;
            sizes.push_back(size);
        }

        // Handle any remaining after alignment
        int64_t total_assigned = 0;
        for (auto s : sizes) total_assigned += s;
        if (total_assigned < total_size)
            sizes[0] += (total_size - total_assigned);

        return sizes;
    }
}

// Performance-based splitting: Would require heuristic queries (placeholder for now)
// This is complex as it needs hipblasLtMatmulAlgoGetHeuristic calls
// We'll implement a simplified version based on known good sizes
inline std::vector<int64_t> computePerformanceSplits(int64_t total_size, int num_splits)
{
    // Placeholder: prefer sizes that are multiples of common optimal sizes
    // Common optimal sizes: 2048, 4096, 5120, 6144, 8192, etc.
    std::vector<int> good_sizes = {8192, 6144, 5120, 4096, 3072, 2048, 1024};

    std::vector<int64_t> sizes;
    int64_t remaining = total_size;

    for(int i = 0; i < num_splits - 1 && remaining > 0; i++)
    {
        int64_t target = remaining / (num_splits - i);

        // Find closest "good" size
        int64_t best_size = target;
        int64_t best_diff = std::abs(target - best_size);

        for(int good : good_sizes)
        {
            if(good <= remaining)
            {
                int64_t diff = std::abs(target - good);
                if(diff < best_diff)
                {
                    best_diff = diff;
                    best_size = good;
                }
            }
        }

        best_size = std::min(best_size, remaining);
        sizes.push_back(best_size);
        remaining -= best_size;
    }

    // Last split gets remainder
    if(remaining > 0)
        sizes.push_back(remaining);

    return sizes;
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
    size_t workspace_size = 128ULL * 1024 * 1024,
    hipblasLtHandle_t handle = nullptr,  // Optional handle for Origami optimization
    hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F)  // Compute type for Origami
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
        case SplitStrategy::MacroTileAlign:
            num_splits = 2;  // Start with 2-way split
            split_along_m = true;
            split_along_n = false;
            break;
        case SplitStrategy::PowerOf2:
            num_splits = 2;  // Start with 2-way split
            split_along_m = true;
            split_along_n = false;
            break;
        case SplitStrategy::CUBalanced:
            num_splits = 2;  // Start with 2-way split
            split_along_m = true;
            split_along_n = false;
            break;
        case SplitStrategy::Performance:
            num_splits = 2;  // Start with 2-way split
            split_along_m = true;
            split_along_n = false;
            break;
        case SplitStrategy::AdaptivePowerOf2:
            num_splits = 2;  // Start with 2-way split
            split_along_m = true;
            split_along_n = false;
            break;
        case SplitStrategy::CacheOptimizedM:
            num_splits = 2;  // 2-way split optimized for cache
            split_along_m = true;
            split_along_n = false;
            break;
        case SplitStrategy::CacheOptimizedN:
            num_splits = 2;  // 2-way split optimized for cache
            split_along_m = false;
            split_along_n = true;
            break;
        case SplitStrategy::OrigamiOptimizedM:
            num_splits = 2;  // 2-way split optimized via Origami queries
            split_along_m = true;
            split_along_n = false;
            break;
        case SplitStrategy::OrigamiOptimizedN:
            num_splits = 2;  // 2-way split optimized via Origami queries
            split_along_m = false;
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

    // Compute split sizes based on strategy
    std::vector<int64_t> m_split_sizes;
    std::vector<int64_t> n_split_sizes;
    bool use_intelligent_splits = false;

    if(split_along_m && !split_along_n)
    {
        // Determine if we should use intelligent splitting
        SplitStrategy strat = static_cast<SplitStrategy>(split_strategy);
        if(strat == SplitStrategy::MacroTileAlign)
        {
            m_split_sizes = computeMacroTileAlignedSplits(M, num_splits, macrotile_m);
            use_intelligent_splits = true;
        }
        else if(strat == SplitStrategy::PowerOf2)
        {
            m_split_sizes = computePowerOf2Splits(M, num_splits);
            use_intelligent_splits = true;
        }
        else if(strat == SplitStrategy::CUBalanced)
        {
            m_split_sizes = computeCUBalancedSplits(M, num_splits, N, macrotile_m, macrotile_n, num_CUs);
            use_intelligent_splits = true;
        }
        else if(strat == SplitStrategy::Performance)
        {
            m_split_sizes = computePerformanceSplits(M, num_splits);
            use_intelligent_splits = true;
        }
        else if(strat == SplitStrategy::AdaptivePowerOf2)
        {
            m_split_sizes = computeAdaptivePowerOf2Splits(M, num_splits);
            use_intelligent_splits = true;
        }
        else if(strat == SplitStrategy::CacheOptimizedM)
        {
            size_t elem_size = getDataTypeSize(b_type);
            m_split_sizes = computeCacheOptimizedSplits(M, num_splits, macrotile_m,
                                                        N, K, elem_size, true);
            use_intelligent_splits = true;
        }
        else if(strat == SplitStrategy::OrigamiOptimizedM)
        {
            // Origami optimization - use WithHandle version if handle available
            if (handle != nullptr)
            {
                m_split_sizes = computeOrigamiOptimizedSplitsWithHandle(
                    handle, M, num_splits, macrotile_m, N, K, true,
                    transA, transB, a_type, b_type, c_type, d_type, compute_type);
            }
            else
            {
                // Fallback to heuristic version if no handle
                m_split_sizes = computeOrigamiOptimizedSplits(M, num_splits, macrotile_m,
                                                              N, K, true);
            }
            use_intelligent_splits = true;
        }
    }
    else if(!split_along_m && split_along_n)
    {
        SplitStrategy strat = static_cast<SplitStrategy>(split_strategy);
        if(strat == SplitStrategy::MacroTileAlign)
        {
            n_split_sizes = computeMacroTileAlignedSplits(N, num_splits, macrotile_n);
            use_intelligent_splits = true;
        }
        else if(strat == SplitStrategy::PowerOf2)
        {
            n_split_sizes = computePowerOf2Splits(N, num_splits);
            use_intelligent_splits = true;
        }
        else if(strat == SplitStrategy::CUBalanced)
        {
            n_split_sizes = computeCUBalancedSplits(N, num_splits, M, macrotile_n, macrotile_m, num_CUs);
            use_intelligent_splits = true;
        }
        else if(strat == SplitStrategy::Performance)
        {
            n_split_sizes = computePerformanceSplits(N, num_splits);
            use_intelligent_splits = true;
        }
        else if(strat == SplitStrategy::AdaptivePowerOf2)
        {
            n_split_sizes = computeAdaptivePowerOf2Splits(N, num_splits);
            use_intelligent_splits = true;
        }
        else if(strat == SplitStrategy::CacheOptimizedN)
        {
            size_t elem_size = getDataTypeSize(a_type);
            n_split_sizes = computeCacheOptimizedSplits(N, num_splits, macrotile_n,
                                                        M, K, elem_size, false);
            use_intelligent_splits = true;
        }
        else if(strat == SplitStrategy::OrigamiOptimizedN)
        {
            // Origami optimization - use WithHandle version if handle available
            if (handle != nullptr)
            {
                n_split_sizes = computeOrigamiOptimizedSplitsWithHandle(
                    handle, N, num_splits, macrotile_n, M, K, false,
                    transA, transB, a_type, b_type, c_type, d_type, compute_type);
            }
            else
            {
                // Fallback to heuristic version if no handle
                n_split_sizes = computeOrigamiOptimizedSplits(N, num_splits, macrotile_n,
                                                              M, K, false);
            }
            use_intelligent_splits = true;
        }
    }

    // Generate sub-problems based on split configuration
    if(split_along_m && !split_along_n)
    {
        // Split along M dimension only
        int64_t m_offset = 0;
        for(int i = 0; i < num_splits; i++)
        {
            int64_t m_size;
            if(use_intelligent_splits)
            {
                m_size = m_split_sizes[i];
            }
            else
            {
                // Uniform splitting
                int64_t m_start = (M * i) / num_splits;
                int64_t m_end = (M * (i + 1)) / num_splits;
                m_size = m_end - m_start;
                m_offset = m_start;
            }

            GemmSubProblem sub;
            sub.m_size = m_size;
            sub.n_size = N;
            sub.k_size = K;
            sub.m_offset = m_offset;
            sub.n_offset = 0;
            sub.offset_A_bytes = calculateOffsetA(m_offset, 0, lda, transA, a_type);
            sub.offset_B_bytes = 0;  // B is shared across all M splits
            sub.offset_C_bytes = calculateOffsetCD(m_offset, 0, ldc, c_type);
            sub.offset_D_bytes = calculateOffsetCD(m_offset, 0, ldd, d_type);
            sub.offset_E_bytes = calculateOffsetCD(m_offset, 0, lde, aux_type);
            sub.offset_bias_bytes = calculateOffsetBias(m_offset, bias_type);
            sub.expected_workgroups = estimateWorkgroups(m_size, N, macrotile_m, macrotile_n);
            sub.macrotile_m = macrotile_m;
            sub.macrotile_n = macrotile_n;

            subProblems.push_back(sub);
            m_offset += m_size;  // Update offset for next split
        }
    }
    else if(!split_along_m && split_along_n)
    {
        // Split along N dimension only
        int64_t n_offset = 0;
        for(int i = 0; i < num_splits; i++)
        {
            int64_t n_size;
            if(use_intelligent_splits)
            {
                n_size = n_split_sizes[i];
            }
            else
            {
                // Uniform splitting
                int64_t n_start = (N * i) / num_splits;
                int64_t n_end = (N * (i + 1)) / num_splits;
                n_size = n_end - n_start;
                n_offset = n_start;
            }

            GemmSubProblem sub;
            sub.m_size = M;
            sub.n_size = n_size;
            sub.k_size = K;
            sub.m_offset = 0;
            sub.n_offset = n_offset;
            sub.offset_A_bytes = 0;  // A is shared across all N splits
            sub.offset_B_bytes = calculateOffsetB(n_offset, 0, ldb, transB, b_type);
            sub.offset_C_bytes = calculateOffsetCD(0, n_offset, ldc, c_type);
            sub.offset_D_bytes = calculateOffsetCD(0, n_offset, ldd, d_type);
            sub.offset_E_bytes = calculateOffsetCD(0, n_offset, lde, aux_type);
            sub.offset_bias_bytes = 0;  // Bias along M, not affected by N split
            sub.expected_workgroups = estimateWorkgroups(M, n_size, macrotile_m, macrotile_n);
            sub.macrotile_m = macrotile_m;
            sub.macrotile_n = macrotile_n;

            subProblems.push_back(sub);
            n_offset += n_size;  // Update offset for next split
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
