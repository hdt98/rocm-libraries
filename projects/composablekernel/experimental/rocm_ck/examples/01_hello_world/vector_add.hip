// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Device-only vector addition kernel.
// Compiled per-architecture into .hsaco code objects for the kpack archive.

#include <hip/hip_runtime.h>

extern "C" __global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}