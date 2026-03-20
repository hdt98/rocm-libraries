// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "cpu_ref.hpp"

void cpu_gemm(const float* a,
              const float* b,
              float* c,
              int M,
              int N,
              int K,
              int stride_A,
              int stride_B,
              int stride_C)
{
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for(int k = 0; k < K; ++k)
            {
                sum += a[i * stride_A + k] * b[j * stride_B + k];
            }
            c[i * stride_C + j] = sum;
        }
    }
}

void cpu_gemm_add(float* e, const float* c, const float* d0, int count)
{
    for(int i = 0; i < count; ++i)
        e[i] = c[i] + d0[i];
}
