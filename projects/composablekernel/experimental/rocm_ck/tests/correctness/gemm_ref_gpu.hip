// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GPU reference GEMM implementations for correctness tests.
//
// Two references, selected at runtime based on hardware:
//
//   gpuGemmMultiword  — 4× FP32 sgemm with Veltkamp splitting.
//                       Near-FP64 accuracy (~2^-48), works on any GPU.
//                       Based on FHLM23: "Matrix Multiplication in Multiword
//                       Arithmetic", Fasi/Higham/Lopez/Mary, SISC 2023.
//
//   gpuGemmFP64       — Single dgemm via rocBLAS.  True FP64 accuracy.
//                       Faster on CDNA (MI250X/MI300X) where FP64 rate is
//                       1/2 to 1/4 of FP32.
//
//   gpuGemm           — Dispatches to FP64 if hasFastFP64(), else multiword.
//
// All functions take float* device arrays and read strides from
// rocm_ck::Args tensor slots via the spec.

#pragma once

#include "gemm_ref_backend.hpp"

#include <rocm_ck/args.hpp>
#include <rocm_ck/hip_check.hpp>
#include <rocm_ck/layout.hpp>

#include <hip/hip_runtime.h>

#include <cstring>

namespace rocm_ck::test {

// ============================================================================
// Hardware detection
// ============================================================================

/// Returns true if the current device has fast FP64 (CDNA architecture).
inline bool hasFastFP64()
{
    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, device));

    // CDNA architectures: gfx90a (MI250X), gfx940/941/942 (MI300X)
    // These have FP64 rate >= 1/4 of FP32.
    return std::strncmp(prop.gcnArchName, "gfx90a", 6) == 0
           || std::strncmp(prop.gcnArchName, "gfx94", 5) == 0;
}

// ============================================================================
// Device kernels
// ============================================================================

namespace detail {

/// Veltkamp/Dekker splitting: a = a_hi + a_lo (exact).
/// Each half has at most 12 significant bits, so a_hi * b_hi is
/// exactly representable in FP32 (24-bit significand).
///
/// Splitting constant = 2^s + 1, where s = ceil(p/2) = 12 for FP32 (p=24).
__global__ void veltkamp_split(const float* __restrict__ in,
                               float* __restrict__ hi,
                               float* __restrict__ lo,
                               int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count)
        return;

    constexpr float kFactor = 4097.0f; // 2^12 + 1

    float a    = in[i];
    float c    = a * kFactor;
    float a_hi = c - (c - a);
    float a_lo = a - a_hi;

    hi[i] = a_hi;
    lo[i] = a_lo;
}

/// Sum 4 partial product matrices in FP64, store as FP32.
/// P0 is the dominant term (A_hi*B_hi), P3 is the smallest (A_lo*B_lo).
__global__ void accumulate_4(float* __restrict__ out,
                             const float* __restrict__ p0,
                             const float* __restrict__ p1,
                             const float* __restrict__ p2,
                             const float* __restrict__ p3,
                             int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count)
        return;

    // Accumulate small-to-large for best FP64 precision
    double sum = static_cast<double>(p3[i]) + static_cast<double>(p2[i])
                 + static_cast<double>(p1[i]) + static_cast<double>(p0[i]);
    out[i] = static_cast<float>(sum);
}

/// Convert float array to double (for FP64 reference path).
__global__ void float_to_double(const float* __restrict__ in,
                                double* __restrict__ out,
                                int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count)
        return;
    out[i] = static_cast<double>(in[i]);
}

/// Convert double array to float (for FP64 reference path).
__global__ void double_to_float(const double* __restrict__ in,
                                float* __restrict__ out,
                                int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count)
        return;
    out[i] = static_cast<float>(in[i]);
}

inline void launchElementwise1D(int count, auto kernel, auto... args)
{
    constexpr int kBlock = 256;
    int grid             = (count + kBlock - 1) / kBlock;
    kernel<<<grid, kBlock>>>(args..., count);
    HIP_CHECK(hipGetLastError());
}

} // namespace detail

// ============================================================================
// Multiword reference: 4x sgemm with Veltkamp splitting
// ============================================================================

/// GPU reference GEMM using double-FP32 multiword arithmetic.
/// Near-FP64 accuracy (~2^-48 effective precision) from 4 FP32 GEMMs.
///
/// Inputs a, b and output c are float arrays on device.
/// Strides come from args.tensors[spec.lhs/rhs/output().args_slot].
template <typename Spec>
void gpuGemmMultiword(const Spec& spec,
                      float* c,
                      const float* a,
                      const float* b,
                      int M,
                      int N,
                      int K,
                      const rocm_ck::Args& args,
                      GemmBackend& backend)
{
    int64_t a_s0 = args.tensors[spec.lhs().args_slot].strides[0];
    int64_t a_s1 = args.tensors[spec.lhs().args_slot].strides[1];
    int64_t b_s0 = args.tensors[spec.rhs().args_slot].strides[0];
    int64_t b_s1 = args.tensors[spec.rhs().args_slot].strides[1];
    int64_t c_s0 = args.tensors[spec.output().args_slot].strides[0];
    int64_t c_s1 = args.tensors[spec.output().args_slot].strides[1];

    int countA = M * K;
    int countB = K * N;
    int countC = M * N;

    // --- Allocate temporaries ---
    float *a_hi, *a_lo, *b_hi, *b_lo;
    float *p0, *p1, *p2, *p3;
    HIP_CHECK(hipMalloc(&a_hi, countA * sizeof(float)));
    HIP_CHECK(hipMalloc(&a_lo, countA * sizeof(float)));
    HIP_CHECK(hipMalloc(&b_hi, countB * sizeof(float)));
    HIP_CHECK(hipMalloc(&b_lo, countB * sizeof(float)));
    HIP_CHECK(hipMalloc(&p0, countC * sizeof(float)));
    HIP_CHECK(hipMalloc(&p1, countC * sizeof(float)));
    HIP_CHECK(hipMalloc(&p2, countC * sizeof(float)));
    HIP_CHECK(hipMalloc(&p3, countC * sizeof(float)));

    // --- Split A and B (element-wise, layout-agnostic) ---
    detail::launchElementwise1D(countA, detail::veltkamp_split, a, a_hi, a_lo);
    detail::launchElementwise1D(countB, detail::veltkamp_split, b, b_hi, b_lo);

    // --- 4 sgemm calls (backend handles layout translation) ---
    float alpha = 1.0f;
    float beta  = 0.0f;
    backend.sgemm(M, N, K, alpha, a_hi, a_s0, a_s1, b_hi, b_s0, b_s1, beta, p0, c_s0, c_s1);
    backend.sgemm(M, N, K, alpha, a_hi, a_s0, a_s1, b_lo, b_s0, b_s1, beta, p1, c_s0, c_s1);
    backend.sgemm(M, N, K, alpha, a_lo, a_s0, a_s1, b_hi, b_s0, b_s1, beta, p2, c_s0, c_s1);
    backend.sgemm(M, N, K, alpha, a_lo, a_s0, a_s1, b_lo, b_s0, b_s1, beta, p3, c_s0, c_s1);

    // --- Accumulate 4 partial products in FP64 ---
    detail::launchElementwise1D(countC, detail::accumulate_4, c, p0, p1, p2, p3);
    HIP_CHECK(hipDeviceSynchronize());

    // --- Cleanup ---
    HIP_CHECK(hipFree(a_hi));
    HIP_CHECK(hipFree(a_lo));
    HIP_CHECK(hipFree(b_hi));
    HIP_CHECK(hipFree(b_lo));
    HIP_CHECK(hipFree(p0));
    HIP_CHECK(hipFree(p1));
    HIP_CHECK(hipFree(p2));
    HIP_CHECK(hipFree(p3));
}

// ============================================================================
// FP64 reference: single dgemm
// ============================================================================

/// GPU reference GEMM using FP64 arithmetic via rocblas_dgemm.
/// True FP64 accuracy.  Converts float inputs to double, calls dgemm,
/// converts output back to float.
///
/// Best on CDNA where FP64 throughput is 1/2 to 1/4 of FP32.
template <typename Spec>
void gpuGemmFP64(const Spec& spec,
                 float* c,
                 const float* a,
                 const float* b,
                 int M,
                 int N,
                 int K,
                 const rocm_ck::Args& args,
                 GemmBackend& backend)
{
    int64_t a_s0 = args.tensors[spec.lhs().args_slot].strides[0];
    int64_t a_s1 = args.tensors[spec.lhs().args_slot].strides[1];
    int64_t b_s0 = args.tensors[spec.rhs().args_slot].strides[0];
    int64_t b_s1 = args.tensors[spec.rhs().args_slot].strides[1];
    int64_t c_s0 = args.tensors[spec.output().args_slot].strides[0];
    int64_t c_s1 = args.tensors[spec.output().args_slot].strides[1];

    int countA = M * K;
    int countB = K * N;
    int countC = M * N;

    // --- Allocate FP64 temporaries ---
    double *a_d, *b_d, *c_d;
    HIP_CHECK(hipMalloc(&a_d, countA * sizeof(double)));
    HIP_CHECK(hipMalloc(&b_d, countB * sizeof(double)));
    HIP_CHECK(hipMalloc(&c_d, countC * sizeof(double)));

    // --- Convert float → double (element-wise, layout-agnostic) ---
    detail::launchElementwise1D(countA, detail::float_to_double, a, a_d);
    detail::launchElementwise1D(countB, detail::float_to_double, b, b_d);

    // --- dgemm (backend handles layout translation) ---
    double alpha = 1.0;
    double beta  = 0.0;
    backend.dgemm(M, N, K, alpha, a_d, a_s0, a_s1, b_d, b_s0, b_s1, beta, c_d, c_s0, c_s1);

    // --- Convert double → float ---
    detail::launchElementwise1D(countC, detail::double_to_float, c_d, c);
    HIP_CHECK(hipDeviceSynchronize());

    // --- Cleanup ---
    HIP_CHECK(hipFree(a_d));
    HIP_CHECK(hipFree(b_d));
    HIP_CHECK(hipFree(c_d));
}

// ============================================================================
// User-selected dispatch
// ============================================================================

/// GPU reference method — controls which reference to use in tests.
/// Use as a gtest parameter to run the same test with both references.
enum class GpuRefMethod
{
    Multiword, // 4× sgemm + Veltkamp split, works on any GPU
    FP64       // single dgemm, requires FP64-capable hardware
};

inline const char* gpuRefMethodName(GpuRefMethod m)
{
    switch(m)
    {
    case GpuRefMethod::Multiword: return "Multiword";
    case GpuRefMethod::FP64: return "FP64";
    }
    return "???";
}

/// GPU reference GEMM — dispatches to the method you choose.
template <typename Spec>
void gpuGemm(GpuRefMethod method,
             const Spec& spec,
             float* c,
             const float* a,
             const float* b,
             int M,
             int N,
             int K,
             const rocm_ck::Args& args,
             GemmBackend& backend)
{
    switch(method)
    {
    case GpuRefMethod::Multiword:
        gpuGemmMultiword(spec, c, a, b, M, N, K, args, backend);
        break;
    case GpuRefMethod::FP64:
        gpuGemmFP64(spec, c, a, b, M, N, K, args, backend);
        break;
    }
}

} // namespace rocm_ck::test
