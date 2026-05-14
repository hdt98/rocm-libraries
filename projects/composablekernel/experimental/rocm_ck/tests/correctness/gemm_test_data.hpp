// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GPU data generators for GEMM correctness tests.
//
// Architecture: a generic fill_2d<T, Gen> kernel walks the 2D index space
// using shape/strides from rocm_ck::Args tensor slots.  Each generator
// is a lightweight __device__-callable functor that computes the value
// for element (m, n).  Stride logic lives in one place; adding a new
// data pattern only requires a new POD struct with:
//
//     __device__ float operator()(int m, int n, int M, int N) const;
//
// The element type T is resolved from rocm_ck::DataType in the host
// launcher — callers pass the dtype from the spec, not a C++ type.
//
// Supported generators:
//   ConstantGen    — every element = value
//   IdentityGen    — 1 on diagonal, 0 elsewhere
//   SequentialGen  — (m * N + n) % modulus
//   UniformGen     — uniform random in [lo, hi) via xorshift32
//   RademacherGen  — +1 or -1 with equal probability
//   NormalGen      — normal distribution (Box-Muller) with mean, stddev, seed

#pragma once

#include <rocm_ck/args.hpp>
#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/hip_check.hpp>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include <cstdint>
#include <cstdio>

namespace rocm_ck::test {

// ============================================================================
// Generic 2D fill kernel (2D grid — no integer division)
// ============================================================================

/// Fill a 2D tensor using shape/strides from args.tensors[tensor_slot].
/// T is the element type, resolved by the host launcher from DataType.
/// The Gen functor computes the value for each (m, n) element.
template <typename T, typename Gen>
__global__ void fill_2d(rocm_ck::Args args,
                        int tensor_slot,
                        Gen gen)
{
    const auto& t = args.tensors[tensor_slot];
    int M         = t.lengths[0];
    int N         = t.lengths[1];

    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(m >= M || n >= N)
        return;

    int64_t offset = m * t.strides[0] + n * t.strides[1];
    float val      = gen(m, n, M, N);

    static_cast<T*>(const_cast<void*>(t.ptr))[offset] = static_cast<T>(val);
}

// ============================================================================
// Host launcher: DataType dispatch
// ============================================================================

namespace detail {

template <typename T, typename Gen>
void launchFill2D(rocm_ck::Args& args, int tensor_slot, int M, int N, Gen gen)
{
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    dim3 block(kBlockX, kBlockY);
    dim3 grid((N + kBlockX - 1) / kBlockX, (M + kBlockY - 1) / kBlockY);
    fill_2d<T><<<grid, block>>>(args, tensor_slot, gen);
    HIP_CHECK(hipGetLastError());
}

} // namespace detail

/// Host launcher for fill_2d.  Dispatches to the correct element type
/// based on rocm_ck::DataType — callers pass the dtype from the spec.
template <typename Gen>
void fill2D(rocm_ck::Args& args, int tensor_slot, rocm_ck::DataType dtype, Gen gen)
{
    int M = args.tensors[tensor_slot].lengths[0];
    int N = args.tensors[tensor_slot].lengths[1];

    switch(dtype)
    {
    case rocm_ck::DataType::FP64:
        detail::launchFill2D<double>(args, tensor_slot, M, N, gen);
        break;
    case rocm_ck::DataType::FP32:
        detail::launchFill2D<float>(args, tensor_slot, M, N, gen);
        break;
    case rocm_ck::DataType::FP16:
        detail::launchFill2D<__half>(args, tensor_slot, M, N, gen);
        break;
    case rocm_ck::DataType::BF16:
        detail::launchFill2D<hip_bfloat16>(args, tensor_slot, M, N, gen);
        break;
    case rocm_ck::DataType::I8:
        detail::launchFill2D<int8_t>(args, tensor_slot, M, N, gen);
        break;
    case rocm_ck::DataType::I32:
        detail::launchFill2D<int32_t>(args, tensor_slot, M, N, gen);
        break;
    default:
        std::fprintf(stderr,
                     "fill2D: unsupported DataType %s\n",
                     rocm_ck::dataTypeName(dtype));
        std::abort();
    }
}

// ============================================================================
// Generator: constant value
// ============================================================================

struct ConstantGen
{
    float value;
    __device__ float operator()(int, int, int, int) const { return value; }
};

/// Convenience wrapper: fill with a constant value.
inline void fillConstant2D(rocm_ck::Args& args,
                           int tensor_slot,
                           rocm_ck::DataType dtype,
                           float value)
{
    fill2D(args, tensor_slot, dtype, ConstantGen{value});
}

// ============================================================================
// Generator: identity matrix — 1.0 on diagonal, 0.0 elsewhere
// ============================================================================

struct IdentityGen
{
    __device__ float operator()(int m, int n, int, int) const
    {
        return (m == n) ? 1.0f : 0.0f;
    }
};

// ============================================================================
// Generator: sequential (modular) — value = float((m * N + n) % modulus)
// ============================================================================

struct SequentialGen
{
    int modulus;
    __device__ float operator()(int m, int n, int, int N) const
    {
        return static_cast<float>((m * N + n) % modulus);
    }
};

// ============================================================================
// Generator: uniform random via xorshift32
// ============================================================================

namespace detail {

__device__ inline uint32_t xorshift32(uint32_t state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

} // namespace detail

struct UniformGen
{
    float lo;
    float hi;
    uint32_t seed;

    __device__ float operator()(int m, int n, int, int N) const
    {
        // Unique state per element: hash of (seed, linear index)
        uint32_t s = seed ^ static_cast<uint32_t>(m * N + n + 1);
        s          = detail::xorshift32(s);
        s          = detail::xorshift32(s); // extra round for better mixing
        float t    = static_cast<float>(s) / static_cast<float>(0xFFFFFFFFu);
        return lo + (hi - lo) * t;
    }
};

// ============================================================================
// Generator: Rademacher — +1 or -1 with equal probability
// ============================================================================

struct RademacherGen
{
    uint32_t seed;

    __device__ float operator()(int m, int n, int, int N) const
    {
        uint32_t s = seed ^ static_cast<uint32_t>(m * N + n + 1);
        s          = detail::xorshift32(s);
        return (s & 1u) ? 1.0f : -1.0f;
    }
};

// ============================================================================
// Generator: Normal distribution via Box-Muller transform
// ============================================================================

struct NormalGen
{
    float mean;
    float stddev;
    uint32_t seed;

    __device__ float operator()(int m, int n, int, int N) const
    {
        // Two independent uniform samples from xorshift for Box-Muller
        uint32_t s = seed ^ static_cast<uint32_t>(m * N + n + 1);
        s          = detail::xorshift32(s);
        s          = detail::xorshift32(s);

        // u1 in (0, 1) — must exclude 0 for log
        float u1 = (static_cast<float>(s) + 1.0f) / 4294967298.0f; // (0xFFFFFFFF + 2)
        s        = detail::xorshift32(s);
        float u2 = static_cast<float>(s) / 4294967295.0f; // [0, 1]

        constexpr float kTwoPi = 6.283185307179586f;
        float z                = sqrtf(-2.0f * logf(u1)) * cosf(kTwoPi * u2);
        return mean + stddev * z;
    }
};

} // namespace rocm_ck::test
