// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// CPU reference calculations for the GEMM example.

#pragma once

/// Naive triple-loop CPU reference GEMM: C = A * B
/// A [M x K] RowMajor, B [K x N] ColumnMajor, C [M x N] RowMajor.
void cpu_gemm(const float* a,
              const float* b,
              float* c,
              int M,
              int N,
              int K,
              int stride_A,
              int stride_B,
              int stride_C);

/// Fused epilogue reference: e[i] = c[i] + d0[i]
void cpu_gemm_add(float* e, const float* c, const float* d0, int count);

/// Fused epilogue reference: e[i] = max(0, c[i] + d0[i])
void cpu_gemm_add_relu(float* e, const float* c, const float* d0, int count);
