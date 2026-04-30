/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Multi-MacroTile Fused Dispatch: eliminates inter-kernel launch overhead
 * by fusing sequential sub-problem dispatches into a single HIP Graph launch.
 *
 * Two approaches:
 *   1. HIP Graph capture: captures hipblasLtMatmul calls into a graph,
 *      replays with single hipGraphLaunch (works with existing Tensile kernels)
 *   2. Workgroup-partitioned wrapper: custom kernel that dispatches different
 *      sub-problems based on workgroup ID (proof of concept)
 *
 *******************************************************************************/

#pragma once

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include "multi_macrotile.hpp"
#include <vector>
#include <chrono>

// ── Approach 1: HIP Graph Fused Dispatch ─────────────────────────────────────
//
// Captures the sequential hipblasLtMatmul calls for all sub-problems into a
// HIP graph during the first iteration, then replays the graph for all
// subsequent iterations. This eliminates:
//   - Per-iteration host-side hipblasLtMatmul call overhead (~5-10μs each)
//   - Inter-kernel command processor gap (~2-5μs between dispatches)
//   - API overhead for layout/descriptor lookups
//
// For 2 sub-problems with 1000 iterations:
//   Sequential: 1000 × (2 × hipblasLtMatmul + 2 × CP gap) = 1000 × ~20μs overhead
//   Graph:      1 × graph_create + 1000 × hipGraphLaunch = ~0.5μs overhead/iter

struct FusedGraphContext
{
    hipGraph_t graph = nullptr;
    hipGraphExec_t graphExec = nullptr;
    bool valid = false;

    ~FusedGraphContext()
    {
        if (graphExec) hipGraphExecDestroy(graphExec);
        if (graph) hipGraphDestroy(graph);
    }
};

// Capture a sequence of hipblasLtMatmul calls into a HIP graph.
// Call this once with one iteration's worth of matmul calls.
inline bool captureFusedGraph(
    FusedGraphContext& ctx,
    hipblasLtHandle_t handle,
    const std::vector<SubProblemContext>& spCtxs,
    const void* alpha,
    const void* beta,
    void* workspace,
    size_t workspace_size,
    hipStream_t stream)
{
    hipError_t err;

    // Begin stream capture
    err = hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    if (err != hipSuccess) return false;

    // Issue all sub-problem matmuls into the captured stream.
    // Use dispatchSubProblem so this path also honors --origami_wgm.
    for (size_t sp = 0; sp < spCtxs.size(); sp++)
    {
        if (!spCtxs[sp].valid) continue;
        dispatchSubProblem(handle, spCtxs[sp],
                           alpha, beta, workspace, workspace_size, stream);
    }

    // End capture → produces graph
    err = hipStreamEndCapture(stream, &ctx.graph);
    if (err != hipSuccess) return false;

    // Instantiate graph for execution
    err = hipGraphInstantiate(&ctx.graphExec, ctx.graph, nullptr, nullptr, 0);
    if (err != hipSuccess) return false;

    ctx.valid = true;
    return true;
}

// Execute the fused graph (replays all captured matmuls with single launch)
inline hipError_t launchFusedGraph(const FusedGraphContext& ctx, hipStream_t stream)
{
    if (!ctx.valid) return hipErrorInvalidValue;
    return hipGraphLaunch(ctx.graphExec, stream);
}

// ── Approach 2: Workgroup-Partitioned Wrapper Kernel (PoC) ───────────────────
//
// A single HIP kernel that receives:
//   - Pointers to A, B, C, D for each sub-problem
//   - Dimensions (M, N, K) for each sub-problem
//   - Workgroup count for each sub-problem
//   - Total workgroup count = sum of all sub-problem WGs
//
// Each workgroup checks its global ID against the cumulative WG counts to
// determine which sub-problem it belongs to, then computes the appropriate
// tile of the output matrix.
//
// NOTE: This is a PROOF OF CONCEPT. It uses naive GEMM computation, not
// optimized MFMA instructions. For production use, this would need to call
// into pre-compiled Tensile kernel code (which requires code object loading
// and is platform-specific).

struct FusedSubProblemArgs
{
    const void* A;
    const void* B;
    void* C;
    void* D;
    int64_t M, N, K;
    int64_t lda, ldb, ldc, ldd;
    int wg_start; // first WG index for this sub-problem
    int wg_count; // number of WGs for this sub-problem
    int mt_m, mt_n; // macrotile dimensions
};

// Simple FP16 GEMM tile computation (NOT optimized — PoC only)
// Each workgroup computes one MT_M × MT_N tile
__global__ void fusedMultiMacroTileKernel(
    FusedSubProblemArgs* subs,
    int num_subs,
    float alpha,
    float beta)
{
    int global_wg = blockIdx.x;

    // Find which sub-problem this WG belongs to
    int sub_idx = -1;
    for (int s = 0; s < num_subs; s++)
    {
        if (global_wg >= subs[s].wg_start &&
            global_wg < subs[s].wg_start + subs[s].wg_count)
        {
            sub_idx = s;
            break;
        }
    }
    if (sub_idx < 0) return;

    const auto& sub = subs[sub_idx];
    int local_wg = global_wg - sub.wg_start;

    // Compute tile coordinates
    int tiles_n = (sub.N + sub.mt_n - 1) / sub.mt_n;
    int tile_row = local_wg / tiles_n;
    int tile_col = local_wg % tiles_n;

    int m_start = tile_row * sub.mt_m;
    int n_start = tile_col * sub.mt_n;

    // Each thread in the workgroup computes one element
    // (simplified — real Tensile uses MFMA with complex tiling)
    int tid = threadIdx.x;
    int threads_m = sub.mt_m < 256 ? sub.mt_m : 256;
    int threads_n = blockDim.x / threads_m;
    if (threads_n < 1) threads_n = 1;

    int local_m = tid % threads_m;
    int local_n = tid / threads_m;

    int gm = m_start + local_m;
    int gn = n_start + local_n;

    if (gm < sub.M && gn < sub.N)
    {
        const __half* A = (const __half*)sub.A;
        const __half* B = (const __half*)sub.B;
        __half* D = (__half*)sub.D;

        float acc = 0.0f;
        for (int k = 0; k < sub.K; k++)
        {
            float a_val = __half2float(A[gm + k * sub.lda]);
            float b_val = __half2float(B[k + gn * sub.ldb]);
            acc += a_val * b_val;
        }

        D[gm + gn * sub.ldd] = __float2half(alpha * acc + beta * __half2float(D[gm + gn * sub.ldd]));
    }
}
