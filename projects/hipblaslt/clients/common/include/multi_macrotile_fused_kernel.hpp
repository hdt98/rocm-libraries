/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 *******************************************************************************/

#ifndef MULTI_MACROTILE_FUSED_KERNEL_HPP
#define MULTI_MACROTILE_FUSED_KERNEL_HPP

#pragma once

#include "multi_macrotile_fused.hpp"
#include <hip/hip_runtime.h>

/**
 * @file multi_macrotile_fused_kernel.hpp
 * @brief Fused kernel dispatch for multi-MacroTile GEMM
 *
 * This file implements a device-side dispatch mechanism that allows
 * multiple sub-problems with different MacroTiles to be executed in
 * a single kernel launch. Each workgroup determines which sub-problem
 * it belongs to and dispatches to the appropriate GEMM kernel.
 *
 * ARCHITECTURE:
 * =============
 *
 * Instead of launching multiple kernels sequentially:
 *   for each sub-problem:
 *       hipblasLtMatmul(sub-problem) // ~15-20 μs launch overhead each
 *
 * We launch a single fused dispatch kernel:
 *   fusedMultiMacrotileDispatch(all sub-problems) // ~15-20 μs overhead TOTAL
 *
 * Each workgroup:
 *   1. Reads its global workgroup ID
 *   2. Maps to (sub-problem index, local WG ID)
 *   3. Loads sub-problem parameters (offsets, dimensions)
 *   4. Dispatches to the appropriate GEMM kernel for that MacroTile
 *
 * KERNEL DISPATCH STRATEGIES:
 * ===========================
 *
 * We support multiple strategies for calling the actual GEMM kernels:
 *
 * Strategy 1: FUNCTION POINTER TABLE (Preferred)
 * -----------------------------------------------
 * - Each solution index maps to a device function pointer
 * - Dispatch via indirect call: kernel_table[solution_idx](params)
 * - Pros: Most flexible, supports any kernel
 * - Cons: Indirect call overhead (~few cycles)
 *
 * Strategy 2: TEMPLATE METAPROGRAMMING
 * -------------------------------------
 * - Compile-time switch statement over common MacroTile sizes
 * - switch(macrotile_id) { case MT_256x208: kernel_256x208(); ... }
 * - Pros: Direct call, no overhead
 * - Cons: Limited to pre-compiled MacroTile set
 *
 * Strategy 3: DYNAMIC CODE OBJECT LOADING
 * ----------------------------------------
 * - Load .co files at runtime and extract function pointers
 * - Most complex but supports arbitrary kernels
 * - Requires hipModuleLoad and hipModuleGetFunction
 *
 * IMPLEMENTATION NOTES:
 * ====================
 *
 * The actual GEMM kernel implementations come from hipBLASLt's
 * Tensile library. We cannot directly call these kernels without:
 *
 * 1. Extracting function pointers from the library
 * 2. OR creating wrapper stubs that call hipblasLtMatmul
 * 3. OR modifying Tensile to generate fused-dispatch kernels
 *
 * For this initial implementation, we use a HYBRID approach:
 * - The dispatch kernel PREPARES all parameters
 * - We use HIP's device-side launch capability (if available)
 * - OR fall back to persistent kernel with dynamic parallelism
 */

// Forward declaration for GEMM kernel function pointers
// These are extracted from hipBLASLt's Tensile library
//
// NOTE: Tensile kernels have complex signatures that vary by kernel.
// We use hipFunction_t (void*) and launch via hipModuleLaunchKernel
// instead of trying to match exact C++ signatures.
typedef hipFunction_t GemmKernelFunc;

// Maximum number of unique kernels we can dispatch to
#define MAX_KERNEL_VARIANTS 64

/**
 * @brief Kernel function pointer table
 *
 * Maps solution index to actual GEMM kernel function.
 * This table is populated at runtime when setting up the fused dispatch.
 */
struct KernelDispatchTable
{
    int num_kernels;
    int solution_indices[MAX_KERNEL_VARIANTS];
    GemmKernelFunc kernel_funcs[MAX_KERNEL_VARIANTS];

    __device__ GemmKernelFunc getKernel(int solution_index) const
    {
        for (int i = 0; i < num_kernels; i++)
        {
            if (solution_indices[i] == solution_index)
                return kernel_funcs[i];
        }
        return nullptr;
    }
};

/**
 * @brief Persistent worker kernel for fused dispatch
 *
 * APPROACH: Instead of launching Tensile kernels from device (not supported),
 * we use a persistent kernel approach where workgroups process their assigned
 * work and the actual Tensile kernel invocation happens via cooperative launch.
 *
 * This kernel MARKS which sub-problem each workgroup belongs to, then the
 * host can read this and perform a batched launch of the appropriate kernels.
 *
 * @param params Fused dispatch parameters
 * @param wg_assignments Output: workgroup → sub-problem index mapping
 */
__global__ void fusedMultiMacrotileAssignmentKernel(
    FusedMultiMacrotileParams params,
    int* wg_assignments)
{
    // Get global workgroup ID
    int global_wg_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    // Map to sub-problem
    int subproblem_idx = -1;
    int local_wg_id = -1;

    if (!mapWorkgroupToSubproblem(global_wg_id, params, subproblem_idx, local_wg_id))
    {
        wg_assignments[global_wg_id] = -1;  // Invalid
        return;
    }

    // Store assignment
    wg_assignments[global_wg_id] = subproblem_idx;
}

/**
 * @brief Alternative: Direct dispatch kernel using hipModuleLaunchKernel
 *
 * NOTE: This version attempts to launch Tensile kernels from device code.
 * This requires device-side dynamic parallelism which may not be available.
 *
 * Keeping this as reference for future HIP versions that may support it.
 */
__global__ void fusedMultiMacrotileDirectDispatchKernel(
    FusedMultiMacrotileParams params,
    KernelDispatchTable dispatch_table)
{
    // Get global workgroup ID
    int global_wg_id = blockIdx.x;

    // Map to sub-problem
    int subproblem_idx = -1;
    int local_wg_id = -1;

    if (!mapWorkgroupToSubproblem(global_wg_id, params, subproblem_idx, local_wg_id))
        return;

    // Get sub-problem info
    const auto& sub = params.subproblems[subproblem_idx];

    // Calculate tile coordinates
    int tile_m, tile_n;
    calculateTileCoordinates(local_wg_id,
                            sub.m_size, sub.n_size,
                            sub.macrotile_m, sub.macrotile_n,
                            tile_m, tile_n);

    // Calculate matrix pointers with offsets
    const void* A_ptr = (const char*)params.A + sub.offset_A_bytes;
    const void* B_ptr = (const char*)params.B + sub.offset_B_bytes;
    const void* C_ptr = (const char*)params.C + sub.offset_C_bytes;
    void* D_ptr = (char*)params.D + sub.offset_D_bytes;

    // NOTE: Calling Tensile kernel from device not yet implemented
    // Would require device-side kernel launch capability
    // For now, this is a placeholder showing the intended structure
}

/**
 * @brief Alternative: Wrapper-based dispatch kernel
 *
 * This version doesn't directly call GEMM kernels. Instead, it prepares
 * parameters and uses cooperative groups to coordinate execution.
 *
 * This is a FALLBACK for systems where we can't extract kernel function
 * pointers from hipBLASLt.
 *
 * @param params Fused dispatch parameters
 * @param work_queue Shared work queue for distributing sub-problems
 */
__global__ void fusedMultiMacrotileWrapperKernel(
    FusedMultiMacrotileParams params,
    volatile int* work_queue)
{
    // This is a simplified version that demonstrates the concept
    // In practice, this would need to call back to host or use
    // device-side hipblasLtMatmul (if available in future HIP versions)

    int global_wg_id = blockIdx.x;

    int subproblem_idx = -1;
    int local_wg_id = -1;

    if (!mapWorkgroupToSubproblem(global_wg_id, params, subproblem_idx, local_wg_id))
        return;

    // Store workgroup assignment for later processing
    // This approach requires a two-phase launch:
    // 1. Dispatch kernel assigns workgroups to sub-problems
    // 2. Host reads assignments and launches appropriate kernels

    if (threadIdx.x == 0)
    {
        work_queue[global_wg_id] = subproblem_idx;
    }
}

/**
 * @brief Host-side function to launch fused dispatch
 *
 * @param params Fused dispatch parameters
 * @param dispatch_table Kernel dispatch table
 * @param stream HIP stream for execution
 *
 * @return HIP error code
 */
inline hipError_t launchFusedMultiMacrotileDispatch(
    const FusedMultiMacrotileParams& params,
    const KernelDispatchTable& dispatch_table,
    hipStream_t stream)
{
    // Launch with total workgroups
    dim3 grid(params.total_workgroups, 1, 1);
    dim3 block(256, 1, 1);  // Standard workgroup size for GEMM

    // Copy params to device memory
    FusedMultiMacrotileParams* d_params;
    hipMalloc(&d_params, sizeof(FusedMultiMacrotileParams));
    hipMemcpy(d_params, &params, sizeof(FusedMultiMacrotileParams), hipMemcpyHostToDevice);

    KernelDispatchTable* d_dispatch_table;
    hipMalloc(&d_dispatch_table, sizeof(KernelDispatchTable));
    hipMemcpy(d_dispatch_table, &dispatch_table, sizeof(KernelDispatchTable), hipMemcpyHostToDevice);

    // Launch kernel
    hipLaunchKernelGGL(fusedMultiMacrotileDirectDispatchKernel,
                       grid, block,
                       0, stream,
                       *d_params, *d_dispatch_table);

    hipError_t err = hipGetLastError();

    // Cleanup
    hipFree(d_params);
    hipFree(d_dispatch_table);

    return err;
}

// Placeholder GEMM kernel stub for demonstration
// In a real implementation, these would be the actual Tensile kernels
__device__ void stubGemmKernel(
    const void* A, const void* B, void* C, void* D,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc, int64_t ldd,
    float alpha, float beta,
    int tile_m, int tile_n)
{
    // This is just a placeholder
    // Real implementation would do actual GEMM computation
    // or call into Tensile-generated kernel code
}

#endif // MULTI_MACROTILE_FUSED_KERNEL_HPP
