// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GPU-computed GEMM verification metrics.
//
// All reductions run on device (data is already there).  Accumulation
// is in FP64 for precision.  Two-phase reduction: per-block partials
// on GPU, finalization on host.
//
// Metrics computed:
//   max_abs_error   — max |C[i] - R[i]|
//   max_rel_error   — max |C[i] - R[i]| / |R[i]|  (nonzero R only)
//   mean_abs_error  — (1/n) Σ |C[i] - R[i]|
//   frob_error      — ||C - R||_F
//   frob_ref        — ||R||_F
//   rel_frob        — ||C - R||_F / ||R||_F  (standard GEMM quality metric)
//   mismatch_count  — #{i : |C[i]-R[i]| > tol·|R[i]| + tol}
//   max_ulp_error   — max ULP distance (dtype-aware, FP32)
//   mean_ulp_error  — mean ULP distance

#pragma once

#include <rocm_ck/hip_check.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace rocm_ck::test {

// ============================================================================
// Result struct
// ============================================================================

struct GemmMetrics
{
    double max_abs_error;  // max |C[i] - R[i]|
    double max_rel_error;  // max |C[i] - R[i]| / |R[i]|
    double mean_abs_error; // mean |C[i] - R[i]|
    double frob_error;     // ||C - R||_F
    double frob_ref;       // ||R||_F
    double rel_frob;       // ||C - R||_F / ||R||_F
    int64_t max_ulp_error; // max ULP distance between result and ref
    double mean_ulp_error; // mean ULP distance
    int mismatch_count;    // pointwise failures
    int count;             // total elements
};

/// Print a formatted metrics report.
inline void printGemmMetrics(const char* label, const GemmMetrics& m)
{
    std::printf("=== %s  (%d elements) ===\n", label, m.count);
    std::printf("  max abs error:    %e\n", m.max_abs_error);
    std::printf("  max rel error:    %e\n", m.max_rel_error);
    std::printf("  mean abs error:   %e\n", m.mean_abs_error);
    std::printf("  ||C-R||_F:        %e\n", m.frob_error);
    std::printf("  ||R||_F:          %e\n", m.frob_ref);
    std::printf("  rel Frobenius:    %e\n", m.rel_frob);
    std::printf("  max ULP error:    %ld\n", static_cast<long>(m.max_ulp_error));
    std::printf("  mean ULP error:   %.2f\n", m.mean_ulp_error);
    std::printf("  mismatches:       %d / %d\n", m.mismatch_count, m.count);
}

// ============================================================================
// Device helpers — per-element computations
// ============================================================================

namespace detail {

// ============================================================================
// ULP (Unit in the Last Place) distance
// ============================================================================

/// Compute ULP distance between two FP32 values.
///
/// IEEE 754 sign-magnitude floats are lexicographically ordered when
/// reinterpreted as integers (after mapping negatives into a linear
/// order).  The ULP distance is the number of representable floats
/// between a and b.
///
/// Special cases: if either value is NaN, returns INT64_MAX.
__device__ inline int64_t ulpDistance(float a, float b)
{
    if(isnan(a) || isnan(b))
        return INT64_MAX;

    // Reinterpret bits as signed 32-bit integers
    int32_t ia, ib;
    memcpy(&ia, &a, sizeof(float));
    memcpy(&ib, &b, sizeof(float));

    // Convert sign-magnitude to two's complement ordering.
    // Negative floats: 0x80000000 maps to 0, 0xFFFFFFFF maps to -0x7FFFFFFF
    // This makes the integer representation monotonically increasing
    // from -FLT_MAX to +FLT_MAX.
    if(ia < 0)
        ia = static_cast<int32_t>(0x80000000u) - ia;
    if(ib < 0)
        ib = static_cast<int32_t>(0x80000000u) - ib;

    int64_t diff = static_cast<int64_t>(ia) - static_cast<int64_t>(ib);
    return (diff < 0) ? -diff : diff;
}

// ============================================================================
// Partial accumulator
// ============================================================================

/// Per-thread partial accumulator (all FP64 for precision).
struct MetricsPartial
{
    double sum_sq_err;
    double sum_sq_ref;
    double sum_abs_err;
    double max_abs_err;
    double max_rel_err;
    double sum_ulp;     // sum of ULP distances (for mean)
    int64_t max_ulp;    // max ULP distance
    int mismatch;
};

/// Combine two partials (associative, for reduction).
__device__ inline MetricsPartial combine(const MetricsPartial& a, const MetricsPartial& b)
{
    return {a.sum_sq_err + b.sum_sq_err,
            a.sum_sq_ref + b.sum_sq_ref,
            a.sum_abs_err + b.sum_abs_err,
            fmax(a.max_abs_err, b.max_abs_err),
            fmax(a.max_rel_err, b.max_rel_err),
            a.sum_ulp + b.sum_ulp,
            (a.max_ulp > b.max_ulp) ? a.max_ulp : b.max_ulp,
            a.mismatch + b.mismatch};
}

/// Compute per-element contribution to metrics.
__device__ inline MetricsPartial elementMetrics(float result, float ref, float tolerance)
{
    double r    = static_cast<double>(result);
    double e    = static_cast<double>(ref);
    double diff = fabs(r - e);

    double rel = 0.0;
    if(fabs(e) > 1e-30) // avoid division by near-zero
        rel = diff / fabs(e);

    int64_t ulp = ulpDistance(result, ref);

    int mismatch = (diff > static_cast<double>(tolerance) * fabs(e)
                         + static_cast<double>(tolerance))
                       ? 1
                       : 0;

    return {diff * diff,
            e * e,
            diff,
            diff,
            rel,
            static_cast<double>(ulp),
            ulp,
            mismatch};
}

// ============================================================================
// Block reduction via shared memory
// ============================================================================

constexpr int kMetricsBlock = 256;

/// Reduce a double array in shared memory (sum).
__device__ inline void blockReduceSum(double* sdata, int tid)
{
    __syncthreads();
    for(int s = kMetricsBlock / 2; s > 0; s >>= 1)
    {
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
}

/// Reduce a double array in shared memory (max).
__device__ inline void blockReduceMax(double* sdata, int tid)
{
    __syncthreads();
    for(int s = kMetricsBlock / 2; s > 0; s >>= 1)
    {
        if(tid < s)
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
}

/// Reduce an int array in shared memory (sum).
__device__ inline void blockReduceSumInt(int* sdata, int tid)
{
    __syncthreads();
    for(int s = kMetricsBlock / 2; s > 0; s >>= 1)
    {
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
}

/// Reduce an int64_t array in shared memory (max).
__device__ inline void blockReduceMaxInt64(int64_t* sdata, int tid)
{
    __syncthreads();
    for(int s = kMetricsBlock / 2; s > 0; s >>= 1)
    {
        if(tid < s && sdata[tid + s] > sdata[tid])
            sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
}

// ============================================================================
// Reduction kernel
// ============================================================================

__global__ void gemmMetricsKernel(const float* __restrict__ result,
                                  const float* __restrict__ ref,
                                  int count,
                                  float tolerance,
                                  MetricsPartial* __restrict__ partials)
{
    int tid       = threadIdx.x;
    int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;

    // --- Grid-stride accumulation ---
    MetricsPartial local = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0};
    for(int i = globalTid; i < count; i += gridStride)
    {
        auto em = elementMetrics(result[i], ref[i], tolerance);
        local   = combine(local, em);
    }

    // --- Block reduction ---
    // Shared memory is reused for each field.  The union ensures
    // enough space for all types without wasting memory.
    __shared__ union
    {
        double d[kMetricsBlock];
        int64_t i64[kMetricsBlock];
        int i32[kMetricsBlock];
    } smem;

    // sum_sq_err
    smem.d[tid] = local.sum_sq_err;
    blockReduceSum(smem.d, tid);
    if(tid == 0)
        partials[blockIdx.x].sum_sq_err = smem.d[0];

    // sum_sq_ref
    smem.d[tid] = local.sum_sq_ref;
    blockReduceSum(smem.d, tid);
    if(tid == 0)
        partials[blockIdx.x].sum_sq_ref = smem.d[0];

    // sum_abs_err
    smem.d[tid] = local.sum_abs_err;
    blockReduceSum(smem.d, tid);
    if(tid == 0)
        partials[blockIdx.x].sum_abs_err = smem.d[0];

    // max_abs_err
    smem.d[tid] = local.max_abs_err;
    blockReduceMax(smem.d, tid);
    if(tid == 0)
        partials[blockIdx.x].max_abs_err = smem.d[0];

    // max_rel_err
    smem.d[tid] = local.max_rel_err;
    blockReduceMax(smem.d, tid);
    if(tid == 0)
        partials[blockIdx.x].max_rel_err = smem.d[0];

    // sum_ulp (double)
    smem.d[tid] = local.sum_ulp;
    blockReduceSum(smem.d, tid);
    if(tid == 0)
        partials[blockIdx.x].sum_ulp = smem.d[0];

    // max_ulp (int64_t)
    smem.i64[tid] = local.max_ulp;
    blockReduceMaxInt64(smem.i64, tid);
    if(tid == 0)
        partials[blockIdx.x].max_ulp = smem.i64[0];

    // mismatch (int)
    smem.i32[tid] = local.mismatch;
    blockReduceSumInt(smem.i32, tid);
    if(tid == 0)
        partials[blockIdx.x].mismatch = smem.i32[0];
}

} // namespace detail

// ============================================================================
// Host wrapper
// ============================================================================

/// Compute all GEMM metrics on GPU.
/// result and ref are device pointers to float arrays of length count.
/// tolerance is the pointwise threshold for mismatch counting.
inline GemmMetrics computeGemmMetrics(const float* d_result,
                                      const float* d_ref,
                                      int count,
                                      float tolerance)
{
    constexpr int kBlock   = detail::kMetricsBlock;
    int numBlocks          = std::min(256, (count + kBlock - 1) / kBlock);

    // Allocate partials on device
    detail::MetricsPartial* d_partials = nullptr;
    HIP_CHECK(hipMalloc(&d_partials, numBlocks * sizeof(detail::MetricsPartial)));

    // Launch reduction
    detail::gemmMetricsKernel<<<numBlocks, kBlock>>>(
        d_result, d_ref, count, tolerance, d_partials);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Download partials
    std::vector<detail::MetricsPartial> partials(numBlocks);
    HIP_CHECK(hipMemcpy(partials.data(), d_partials,
                         numBlocks * sizeof(detail::MetricsPartial),
                         hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_partials));

    // Finalize on host
    double sum_sq_err  = 0.0;
    double sum_sq_ref  = 0.0;
    double sum_abs_err = 0.0;
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    double sum_ulp     = 0.0;
    int64_t max_ulp    = 0;
    int mismatch       = 0;

    for(int i = 0; i < numBlocks; ++i)
    {
        sum_sq_err += partials[i].sum_sq_err;
        sum_sq_ref += partials[i].sum_sq_ref;
        sum_abs_err += partials[i].sum_abs_err;
        max_abs_err = std::fmax(max_abs_err, partials[i].max_abs_err);
        max_rel_err = std::fmax(max_rel_err, partials[i].max_rel_err);
        sum_ulp += partials[i].sum_ulp;
        max_ulp = std::max(max_ulp, partials[i].max_ulp);
        mismatch += partials[i].mismatch;
    }

    double frob_err = std::sqrt(sum_sq_err);
    double frob_ref = std::sqrt(sum_sq_ref);
    double rel_frob = (frob_ref > 0.0) ? frob_err / frob_ref : 0.0;

    return {max_abs_err,
            max_rel_err,
            sum_abs_err / static_cast<double>(count),
            frob_err,
            frob_ref,
            rel_frob,
            max_ulp,
            sum_ulp / static_cast<double>(count),
            mismatch,
            count};
}

} // namespace rocm_ck::test
