// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Demonstration of Duff's device implemented as a plain HIP kernel.
//
// Duff's device (Tom Duff, 1983) is a technique for unrolling loops by
// exploiting C's switch-statement fall-through inside a do-while loop.
// Each thread applies the idiom to its own stripe of elements, handling
// arbitrary element counts without a separate scalar clean-up loop.
//
// Usage: duffs_device [array_size]   (default: 10007)

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(call)                                                                   \
    do                                                                                    \
    {                                                                                     \
        hipError_t err_ = (call);                                                         \
        if(err_ != hipSuccess)                                                            \
        {                                                                                 \
            fprintf(stderr,                                                               \
                    "HIP error '%s' (%d) at %s:%d\n",                                    \
                    hipGetErrorString(err_),                                              \
                    static_cast<int>(err_),                                               \
                    __FILE__,                                                             \
                    __LINE__);                                                            \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while(0)

/// Unroll factor used in the kernel (must be a power of two, 1-based cases below).
static constexpr int UNROLL_FACTOR = 8;

// =============================================================================
// Kernel
// =============================================================================

/// @brief Copies @p n integers from @p src to @p dst using Duff's device.
///
/// Each thread owns every stride-th element starting at its thread ID.
/// Duff's device handles the remainder on the first pass so no element is
/// visited twice and no separate scalar epilogue is needed.
__global__ void duffs_device_copy(int* __restrict__ dst,
                                  const int* __restrict__ src,
                                  int n)
{
    const int tid    = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                       + static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x);

    // Count how many elements belong to this thread.
    int count = 0;
    for(int idx = tid; idx < n; idx += stride)
        ++count;

    if(count == 0)
        return;

    // Number of full-unroll iterations needed (ceiling division).
    int loops = (count + UNROLL_FACTOR - 1) / UNROLL_FACTOR;
    int i     = tid;

    // clang-format off
    // Duff's device: the switch jumps into the middle of the unrolled loop
    // body to consume the remainder (count % UNROLL_FACTOR) first, then
    // continues with groups of UNROLL_FACTOR until all elements are copied.
    switch(count % UNROLL_FACTOR)
    {
    case 0: do { dst[i] = src[i]; i += stride; // NOLINT(bugprone-branch-clone)
    case 7:      dst[i] = src[i]; i += stride;
    case 6:      dst[i] = src[i]; i += stride;
    case 5:      dst[i] = src[i]; i += stride;
    case 4:      dst[i] = src[i]; i += stride;
    case 3:      dst[i] = src[i]; i += stride;
    case 2:      dst[i] = src[i]; i += stride;
    case 1:      dst[i] = src[i]; i += stride;
              } while(--loops > 0);
    }
    // clang-format on
}

// =============================================================================
// Host driver
// =============================================================================

int main(int argc, char* argv[])
{
    const int n = (argc > 1) ? atoi(argv[1]) : 10007;
    if(n <= 0)
    {
        fprintf(stderr, "Usage: %s [array_size > 0]\n", argv[0]);
        return EXIT_FAILURE;
    }

    printf("Duff's device HIP kernel: copying %d integers\n", n);

    // --- Host arrays ---
    int* h_src = new int[n];
    int* h_dst = new int[n];
    for(int i = 0; i < n; ++i)
        h_src[i] = i;
    memset(h_dst, 0, static_cast<size_t>(n) * sizeof(int));

    // --- Device arrays ---
    int* d_src = nullptr;
    int* d_dst = nullptr;
    HIP_CHECK(hipMalloc(&d_src, static_cast<size_t>(n) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_dst, static_cast<size_t>(n) * sizeof(int)));

    HIP_CHECK(hipMemcpy(d_src, h_src, static_cast<size_t>(n) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_dst, 0, static_cast<size_t>(n) * sizeof(int)));

    // --- Launch ---
    constexpr int block_size = 256;
    const int     grid_size  = (n + block_size - 1) / block_size;
    duffs_device_copy<<<grid_size, block_size>>>(d_dst, d_src, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // --- Verify ---
    HIP_CHECK(hipMemcpy(h_dst, d_dst, static_cast<size_t>(n) * sizeof(int), hipMemcpyDeviceToHost));

    bool passed = true;
    for(int i = 0; i < n; ++i)
    {
        if(h_dst[i] != h_src[i])
        {
            fprintf(stderr, "Mismatch at index %d: got %d, expected %d\n", i, h_dst[i], h_src[i]);
            passed = false;
            break;
        }
    }
    printf("Result: %s\n", passed ? "PASS" : "FAIL");

    // --- Cleanup ---
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));
    delete[] h_src;
    delete[] h_dst;

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
