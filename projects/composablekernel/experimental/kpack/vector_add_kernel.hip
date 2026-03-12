// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Simple vector addition kernel for kpack demo.
// Compiled as device-only code object (.hsaco) per architecture.

#include <hip/hip_runtime.h>

extern "C" __global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
