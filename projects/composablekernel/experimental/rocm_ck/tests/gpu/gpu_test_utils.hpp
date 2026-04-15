// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: host+device — GPU test utilities (device-side fill/compare, HIP event timing).
//
// All data stays on the GPU — no host↔device transfers in the hot path.

#pragma once

#include <rocm_ck/hip_check.hpp>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <gtest/gtest.h>

#include <climits>
#include <cstdio>

namespace rocm_ck::test {

/// Skip the current test if no GPU is available.
inline void skipIfNoGpu()
{
    int count = 0;
    auto err  = hipGetDeviceCount(&count);
    if(err != hipSuccess || count == 0)
        GTEST_SKIP() << "No GPU available";
}

// ============================================================================
// Device-side data generation
// ============================================================================

/// Fill a device buffer with (i % range) cast to the element type.
/// elem_bytes: 2 = fp16 (__half), 4 = fp32 (float).
__global__ void fill_pattern(void* ptr, int count, int elem_bytes, int range)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count)
        return;

    float val = static_cast<float>(i % range);
    if(elem_bytes == 2)
        static_cast<__half*>(ptr)[i] = __float2half(val);
    else
        static_cast<float*>(ptr)[i] = val;
}

/// Host wrapper: launch fill_pattern on a device buffer.
inline void deviceFill(void* dev_ptr, int count, int elem_bytes, int range = 8)
{
    constexpr int kBlock = 256;
    int grid             = (count + kBlock - 1) / kBlock;
    fill_pattern<<<dim3(grid), dim3(kBlock)>>>(dev_ptr, count, elem_bytes, range);
    HIP_CHECK(hipGetLastError());
}

// ============================================================================
// Device-side comparison
// ============================================================================

/// Byte-wise comparison kernel. Writes the first mismatch offset to *result
/// via atomicMin. Initialize *result to INT_MAX before launch.
__global__ void compare_buffers(const char* a, const char* b, int num_bytes, int* result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= num_bytes)
        return;

    if(a[i] != b[i])
        atomicMin(result, i);
}

/// Compare two device buffers bitwise. Returns -1 if identical, else the
/// byte offset of the first mismatch. Only transfers 4 bytes to host.
inline int64_t deviceBitwiseCompare(const void* dev_a, const void* dev_b, size_t bytes)
{
    int* d_result;
    HIP_CHECK(hipMalloc(&d_result, sizeof(int)));
    int init = INT_MAX;
    HIP_CHECK(hipMemcpy(d_result, &init, sizeof(int), hipMemcpyHostToDevice));

    constexpr int kBlock = 256;
    int grid             = (static_cast<int>(bytes) + kBlock - 1) / kBlock;
    compare_buffers<<<dim3(grid), dim3(kBlock)>>>(static_cast<const char*>(dev_a),
                                                  static_cast<const char*>(dev_b),
                                                  static_cast<int>(bytes),
                                                  d_result);
    HIP_CHECK(hipGetLastError());

    int h_result;
    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_result));

    return (h_result == INT_MAX) ? -1 : static_cast<int64_t>(h_result);
}

// ============================================================================
// GpuTimer: HIP event-based kernel timing
// ============================================================================

struct GpuTimer
{
    GpuTimer()
    {
        HIP_CHECK(hipEventCreate(&start_));
        HIP_CHECK(hipEventCreate(&stop_));
    }

    ~GpuTimer()
    {
        (void)hipEventDestroy(start_);
        (void)hipEventDestroy(stop_);
    }

    GpuTimer(const GpuTimer&)            = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    GpuTimer(GpuTimer&& other) noexcept : start_(other.start_), stop_(other.stop_)
    {
        other.start_ = nullptr;
        other.stop_  = nullptr;
    }

    GpuTimer& operator=(GpuTimer&& other) noexcept
    {
        if(this != &other)
        {
            (void)hipEventDestroy(start_);
            (void)hipEventDestroy(stop_);
            start_       = other.start_;
            stop_        = other.stop_;
            other.start_ = nullptr;
            other.stop_  = nullptr;
        }
        return *this;
    }

    void start() { HIP_CHECK(hipEventRecord(start_)); }
    void stop() { HIP_CHECK(hipEventRecord(stop_)); }

    float elapsed_ms()
    {
        HIP_CHECK(hipEventSynchronize(stop_));
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

    private:
    hipEvent_t start_{};
    hipEvent_t stop_{};
};

// ============================================================================
// Performance reporting
// ============================================================================

/// Benchmark a kernel lambda. Returns average time in ms.
template <typename F>
float benchmark(const char* name, int warmup, int iters, F&& kernel_fn)
{
    for(int i = 0; i < warmup; ++i)
        kernel_fn();
    HIP_CHECK(hipDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    for(int i = 0; i < iters; ++i)
        kernel_fn();
    timer.stop();
    float total_ms = timer.elapsed_ms();
    float avg_ms   = total_ms / iters;

    std::printf("  [PERF] %-40s %8.3f ms\n", name, avg_ms);
    return avg_ms;
}

inline void reportOverhead(const char* label, float bare_ms, float rocmck_ms)
{
    float pct = (bare_ms > 0) ? (rocmck_ms - bare_ms) / bare_ms * 100.0f : 0.0f;
    std::printf("  [PERF] %-40s bare=%.3f ms, rocm_ck=%.3f ms, overhead=%.1f%%\n",
                label,
                bare_ms,
                rocmck_ms,
                pct);
}

} // namespace rocm_ck::test
