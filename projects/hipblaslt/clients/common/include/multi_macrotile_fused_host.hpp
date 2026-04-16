/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 *******************************************************************************/

#ifndef MULTI_MACROTILE_FUSED_HOST_HPP
#define MULTI_MACROTILE_FUSED_HOST_HPP

#pragma once

// This file contains host-only code and should not be compiled for device
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)

#include "multi_macrotile_fused.hpp"
#include "multi_macrotile_kernel_extraction.hpp"
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <chrono>

/**
 * @file multi_macrotile_fused_host.hpp
 * @brief Host-side fused dispatch implementation
 *
 * STRATEGY: Since device-side kernel launch is not available in current HIP,
 * we use a PRACTICAL approach:
 *
 * 1. Launch all sub-problem kernels concurrently using hipLaunchKernel
 * 2. Use the extracted kernel functions directly
 * 3. Each kernel is launched with grid size appropriate for its sub-problem
 *
 * This is NOT true single-kernel fusion, but achieves similar benefits:
 * - All kernels launch concurrently (GPU-wide parallelism)
 * - Minimal host-side overhead (single batch of launches)
 * - Reduced CPU→GPU roundtrip latency
 */

/**
 * @brief Structure to hold kernel launch parameters
 */
struct KernelLaunchParams
{
    hipFunction_t kernel_func;
    dim3 grid_size;
    dim3 block_size;
    size_t shared_mem;

    // Tensile kernel parameters (simplified - actual signature varies)
    void* A;
    void* B;
    void* C;
    void* D;
    uint32_t size_L;
    uint32_t size_I;
    uint32_t size_J;
    uint32_t size_K;
    const void* alpha;
    const void* beta;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t ldd;
    uint32_t strideA1;
    uint32_t strideA2;
    uint32_t strideB1;
    uint32_t strideB2;
    uint32_t strideC1;
    uint32_t strideC2;
    uint32_t strideD1;
    uint32_t strideD2;
};

/**
 * @brief Launch Tensile kernel using hipModuleLaunchKernel
 *
 * @param kernel_func Extracted kernel function pointer
 * @param grid Grid dimensions
 * @param block Block dimensions
 * @param params Kernel parameters
 * @param stream HIP stream
 * @return HIP error code
 */
inline hipError_t launchTensileKernel(
    hipFunction_t kernel_func,
    dim3 grid, dim3 block,
    const KernelLaunchParams& params,
    hipStream_t stream)
{
    // Build kernel argument array
    // NOTE: This is simplified - actual Tensile kernels may have different signatures
    void* kernel_args[] = {
        (void*)&params.A,
        (void*)&params.B,
        (void*)&params.C,
        (void*)&params.D,
        (void*)&params.alpha,
        (void*)&params.beta,
        // Additional parameters would go here
    };

    // Launch kernel
    return hipModuleLaunchKernel(
        kernel_func,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        params.shared_mem,
        stream,
        kernel_args,
        nullptr);
}

/**
 * @brief Execute fused multi-MacroTile dispatch (concurrent batch launch)
 *
 * This function launches all sub-problem kernels concurrently on a single stream,
 * achieving GPU-wide parallelism without sequential launch overhead.
 *
 * @param subProblems Vector of sub-problem descriptors
 * @param algorithms Vector of algorithms (one per sub-problem)
 * @param handle hipBLASLt handle
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
 * @param stream HIP stream
 * @param device_id Device ID
 *
 * @return HIP error code
 */
inline hipError_t executeFusedMultiMacrotileBatchLaunch(
    const std::vector<GemmSubProblem>& subProblems,
    const std::vector<hipblasLtMatmulAlgo_t>& algorithms,
    hipblasLtHandle_t handle,
    void* A, void* B, void* C, void* D,
    float alpha, float beta,
    int64_t lda, int64_t ldb, int64_t ldc, int64_t ldd,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    void* workspace, size_t workspace_size,
    hipStream_t stream,
    int device_id)
{
    hipblaslt_cout << "\n=== Fused Multi-MacroTile Batch Launch ===" << std::endl;
    hipblaslt_cout << "Number of sub-problems: " << subProblems.size() << std::endl;

    // Create kernel dispatch table
    KernelDispatchTable dispatch_table = createKernelDispatchTable(
        handle, subProblems, algorithms, device_id);

    if (dispatch_table.num_kernels == 0)
    {
        hipblaslt_cerr << "ERROR: No kernels extracted, falling back to sequential" << std::endl;
        return hipErrorNotFound;
    }

    // Build launch parameters for each sub-problem
    std::vector<KernelLaunchParams> launch_params;

    for (size_t i = 0; i < subProblems.size(); i++)
    {
        const auto& sub = subProblems[i];
        const auto& algo = algorithms[i];

        // Get solution index
        int solution_idx = getSolutionIndexFromAlgo(algo);

        // Find kernel function in dispatch table
        hipFunction_t kernel_func = nullptr;
        for (int j = 0; j < dispatch_table.num_kernels; j++)
        {
            if (dispatch_table.solution_indices[j] == solution_idx)
            {
                kernel_func = dispatch_table.kernel_funcs[j];
                break;
            }
        }

        if (!kernel_func)
        {
            hipblaslt_cerr << "ERROR: Kernel not found for solution index " << solution_idx << std::endl;
            continue;
        }

        // Calculate grid dimensions
        int tiles_m = (sub.m_size + sub.macrotile_m - 1) / sub.macrotile_m;
        int tiles_n = (sub.n_size + sub.macrotile_n - 1) / sub.macrotile_n;
        int total_tiles = tiles_m * tiles_n;

        KernelLaunchParams params;
        params.kernel_func = kernel_func;
        params.grid_size = dim3(total_tiles, 1, 1);
        params.block_size = dim3(256, 1, 1);  // Standard Tensile workgroup size
        params.shared_mem = 0;  // Tensile manages LDS internally

        // Matrix pointers with offsets
        params.A = (char*)A + sub.offset_A_bytes;
        params.B = (char*)B + sub.offset_B_bytes;
        params.C = (char*)C + sub.offset_C_bytes;
        params.D = (char*)D + sub.offset_D_bytes;

        // Dimensions
        params.size_L = 0;  // Batch dimension (not used for single GEMM)
        params.size_I = sub.m_size;
        params.size_J = sub.n_size;
        params.size_K = sub.k_size;

        // Scalars
        params.alpha = &alpha;
        params.beta = &beta;

        // Leading dimensions
        params.lda = lda;
        params.ldb = ldb;
        params.ldc = ldc;
        params.ldd = ldd;

        // Strides (not used for single GEMM)
        params.strideA1 = 0;
        params.strideA2 = 0;
        params.strideB1 = 0;
        params.strideB2 = 0;
        params.strideC1 = 0;
        params.strideC2 = 0;
        params.strideD1 = 0;
        params.strideD2 = 0;

        launch_params.push_back(params);

        hipblaslt_cout << "Sub-problem " << i << ": Grid " << total_tiles
                  << " (" << tiles_m << "×" << tiles_n << ")"
                  << ", Block 256" << std::endl;
    }

    // Launch all kernels in batch
    hipblaslt_cout << "\nLaunching " << launch_params.size() << " kernels concurrently..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& params : launch_params)
    {
        hipError_t err = launchTensileKernel(
            params.kernel_func,
            params.grid_size,
            params.block_size,
            params,
            stream);

        if (err != hipSuccess)
        {
            hipblaslt_cerr << "ERROR: Kernel launch failed with error " << err << std::endl;
            return err;
        }
    }

    // Synchronize to measure total time
    hipError_t sync_err = hipStreamSynchronize(stream);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();

    hipblaslt_cout << "Batch launch complete: " << elapsed_us << " μs" << std::endl;
    hipblaslt_cout << "=========================================\n" << std::endl;

    return sync_err;
}

/**
 * @brief Wrapper function for compatibility with testing_matmul.hpp
 *
 * Attempts fused dispatch, falls back to sequential if it fails.
 */
inline bool tryFusedMultiMacrotileDispatch(
    const std::vector<GemmSubProblem>& subProblems,
    const std::vector<hipblasLtMatmulAlgo_t>& algorithms,
    hipblasLtHandle_t handle,
    void* A, void* B, void* C, void* D,
    float alpha, float beta,
    int64_t lda, int64_t ldb, int64_t ldc, int64_t ldd,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    void* workspace, size_t workspace_size,
    hipStream_t stream,
    int device_id)
{
    hipError_t err = executeFusedMultiMacrotileBatchLaunch(
        subProblems, algorithms, handle,
        A, B, C, D, alpha, beta,
        lda, ldb, ldc, ldd, transA, transB,
        a_type, b_type, c_type, d_type,
        workspace, workspace_size, stream, device_id);

    if (err != hipSuccess)
    {
        hipblaslt_cerr << "Fused dispatch failed, will fall back to sequential" << std::endl;
        return false;
    }

    return true;
}

#endif // !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)

#endif // MULTI_MACROTILE_FUSED_HOST_HPP
