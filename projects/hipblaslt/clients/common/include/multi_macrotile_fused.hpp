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

#ifndef MULTI_MACROTILE_FUSED_HPP
#define MULTI_MACROTILE_FUSED_HPP

#pragma once

#include "multi_macrotile.hpp"
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <vector>
#include <string>

// Maximum number of sub-problems supported in fused dispatch
#define MAX_FUSED_SUBPROBLEMS 16

/**
 * @brief Structure describing one sub-problem for fused kernel dispatch
 *
 * This structure is passed to the device and contains all information
 * needed for a workgroup to execute its portion of the GEMM.
 */
struct FusedSubProblemInfo
{
    // Sub-problem dimensions
    int64_t m_size;
    int64_t n_size;
    int64_t k_size;

    // Offsets in original matrices (in elements, not bytes)
    int64_t m_offset;
    int64_t n_offset;

    // Byte offsets for matrix pointers
    size_t offset_A_bytes;
    size_t offset_B_bytes;
    size_t offset_C_bytes;
    size_t offset_D_bytes;

    // Workgroup assignment for this sub-problem
    int workgroup_start;  // First WG ID for this sub-problem
    int workgroup_count;  // Number of WGs for this sub-problem

    // Algorithm/kernel ID
    int solution_index;   // Solution index from hipBLASLt
    int macrotile_m;      // MacroTile M dimension
    int macrotile_n;      // MacroTile N dimension

    // Padding for alignment
    char padding[32];
};

/**
 * @brief Parameters for fused multi-MacroTile kernel dispatch
 *
 * This structure contains all information needed to execute multiple
 * sub-problems in a single kernel launch, with workgroup-based dispatch.
 */
struct FusedMultiMacrotileParams
{
    // Number of sub-problems
    int num_subproblems;

    // Total workgroups across all sub-problems
    int total_workgroups;

    // Array of sub-problem descriptors
    FusedSubProblemInfo subproblems[MAX_FUSED_SUBPROBLEMS];

    // Global matrix pointers (base addresses)
    void* A;
    void* B;
    void* C;
    void* D;

    // GEMM parameters (shared across all sub-problems)
    float alpha;
    float beta;

    // Leading dimensions (shared)
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t ldd;

    // Transpose operations
    hipblasOperation_t transA;
    hipblasOperation_t transB;

    // Data types
    hipDataType a_type;
    hipDataType b_type;
    hipDataType c_type;
    hipDataType d_type;

    // Workspace
    void* workspace;
    size_t workspace_size;
};

/**
 * @brief Create fused dispatch parameters from sub-problems
 *
 * @param subProblems Vector of sub-problem descriptors
 * @param A Base pointer to matrix A
 * @param B Base pointer to matrix B
 * @param C Base pointer to matrix C
 * @param D Base pointer to matrix D
 * @param alpha Alpha scalar
 * @param beta Beta scalar
 * @param lda Leading dimension of A
 * @param ldb Leading dimension of B
 * @param ldc Leading dimension of C
 * @param ldd Leading dimension of D
 * @param transA Transpose operation for A
 * @param transB Transpose operation for B
 * @param a_type Data type of A
 * @param b_type Data type of B
 * @param c_type Data type of C
 * @param d_type Data type of D
 * @param workspace Workspace pointer
 * @param workspace_size Workspace size
 *
 * @return Fused dispatch parameters structure
 */
inline FusedMultiMacrotileParams createFusedParams(
    const std::vector<GemmSubProblem>& subProblems,
    void* A, void* B, void* C, void* D,
    float alpha, float beta,
    int64_t lda, int64_t ldb, int64_t ldc, int64_t ldd,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    void* workspace, size_t workspace_size)
{
    FusedMultiMacrotileParams params = {};

    params.num_subproblems = std::min((int)subProblems.size(), MAX_FUSED_SUBPROBLEMS);
    params.total_workgroups = 0;

    // Set global parameters
    params.A = A;
    params.B = B;
    params.C = C;
    params.D = D;
    params.alpha = alpha;
    params.beta = beta;
    params.lda = lda;
    params.ldb = ldb;
    params.ldc = ldc;
    params.ldd = ldd;
    params.transA = transA;
    params.transB = transB;
    params.a_type = a_type;
    params.b_type = b_type;
    params.c_type = c_type;
    params.d_type = d_type;
    params.workspace = workspace;
    params.workspace_size = workspace_size;

    // Fill in sub-problem info
    for (int i = 0; i < params.num_subproblems; i++)
    {
        const auto& sub = subProblems[i];
        auto& info = params.subproblems[i];

        info.m_size = sub.m_size;
        info.n_size = sub.n_size;
        info.k_size = sub.k_size;
        info.m_offset = sub.m_offset;
        info.n_offset = sub.n_offset;
        info.offset_A_bytes = sub.offset_A_bytes;
        info.offset_B_bytes = sub.offset_B_bytes;
        info.offset_C_bytes = sub.offset_C_bytes;
        info.offset_D_bytes = sub.offset_D_bytes;

        // Workgroup assignment
        info.workgroup_start = params.total_workgroups;
        info.workgroup_count = sub.expected_workgroups;
        params.total_workgroups += sub.expected_workgroups;

        // MacroTile info
        info.macrotile_m = sub.macrotile_m;
        info.macrotile_n = sub.macrotile_n;
        info.solution_index = -1;  // Will be filled in later
    }

    return params;
}

/**
 * @brief Determine which sub-problem a workgroup belongs to
 *
 * Device function that maps global workgroup ID to sub-problem index
 * and local workgroup ID within that sub-problem.
 *
 * @param global_wg_id Global workgroup ID
 * @param params Fused dispatch parameters
 * @param subproblem_idx [out] Sub-problem index
 * @param local_wg_id [out] Local workgroup ID within sub-problem
 *
 * @return true if mapping successful, false if workgroup ID out of range
 */
__device__ inline bool mapWorkgroupToSubproblem(
    int global_wg_id,
    const FusedMultiMacrotileParams& params,
    int& subproblem_idx,
    int& local_wg_id)
{
    // Linear search through sub-problems
    // Could optimize with binary search for large num_subproblems
    for (int i = 0; i < params.num_subproblems; i++)
    {
        const auto& sub = params.subproblems[i];
        int wg_end = sub.workgroup_start + sub.workgroup_count;

        if (global_wg_id >= sub.workgroup_start && global_wg_id < wg_end)
        {
            subproblem_idx = i;
            local_wg_id = global_wg_id - sub.workgroup_start;
            return true;
        }
    }

    return false;  // Workgroup ID out of range
}

/**
 * @brief Calculate workgroup tile coordinates for a sub-problem
 *
 * Maps local workgroup ID to (tile_m, tile_n) coordinates in the output matrix.
 *
 * @param local_wg_id Local workgroup ID within sub-problem
 * @param m_size Sub-problem M dimension
 * @param n_size Sub-problem N dimension
 * @param macrotile_m MacroTile M dimension
 * @param macrotile_n MacroTile N dimension
 * @param tile_m [out] Tile row index
 * @param tile_n [out] Tile column index
 */
__device__ inline void calculateTileCoordinates(
    int local_wg_id,
    int64_t m_size, int64_t n_size,
    int macrotile_m, int macrotile_n,
    int& tile_m, int& tile_n)
{
    // Calculate number of tiles in each dimension
    int tiles_m = (m_size + macrotile_m - 1) / macrotile_m;
    int tiles_n = (n_size + macrotile_n - 1) / macrotile_n;

    // Map 1D workgroup ID to 2D tile coordinates
    // Row-major layout: tile_m varies fastest
    tile_m = local_wg_id % tiles_m;
    tile_n = local_wg_id / tiles_m;
}

#endif // MULTI_MACROTILE_FUSED_HPP
