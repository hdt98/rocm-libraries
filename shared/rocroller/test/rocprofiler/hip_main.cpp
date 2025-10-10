// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
//
// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <rocprofiler-sdk-roctx/roctx.h>

#include "hip/hip_runtime.h"

// Two waves per SIMD on MI300
#define DATA_SIZE (304 * 64 * 4 * 2)
#define HIP_API_CALL(CALL)   \
    if((CALL) != hipSuccess) \
    {                        \
        abort();             \
    }

__global__ void divide_kernel(float* a, const float* b, const float* c, int /* unused */)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= DATA_SIZE)
        return;

    a[index] = (b[index] - c[index]) / abs(c[index] + b[index]) + 1;
}

int main(int /*argc*/, char** /*argv*/)
{
    // Allocate device memory
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    HIP_API_CALL(hipMalloc(&d_a, DATA_SIZE * sizeof(float)));
    HIP_API_CALL(hipMalloc(&d_b, DATA_SIZE * sizeof(float)));
    HIP_API_CALL(hipMalloc(&d_c, DATA_SIZE * sizeof(float)));

    // Initialize memory
    HIP_API_CALL(hipMemset(d_a, 0, DATA_SIZE * sizeof(float)));
    HIP_API_CALL(hipMemset(d_b, 1, DATA_SIZE * sizeof(float)));
    HIP_API_CALL(hipMemset(d_c, 2, DATA_SIZE * sizeof(float)));

    // Start profiling
    roctxProfilerResume(0);

    // Launch kernel
    dim3 blockSize(512);
    dim3 gridSize((DATA_SIZE + blockSize.x - 1) / blockSize.x);

    hipLaunchKernelGGL(divide_kernel, gridSize, blockSize, 0, 0, d_a, d_b, d_c, 0);
    HIP_API_CALL(hipGetLastError());

    // Wait for kernel to complete
    HIP_API_CALL(hipDeviceSynchronize());

    // Stop profiling
    roctxProfilerPause(0);

    // Free memory
    HIP_API_CALL(hipFree(d_a));
    HIP_API_CALL(hipFree(d_b));
    HIP_API_CALL(hipFree(d_c));

    return 0;
}
