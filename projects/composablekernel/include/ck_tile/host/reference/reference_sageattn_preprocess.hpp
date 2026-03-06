// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

// CPU float32 reference implementations for SageAttention V3 preprocessing.
//
// All functions use [B, H, seqlen, hdim] row-major layout with explicit stride arguments.
// Conventions:
//   src[b, h, n, d] = src_ptr[b * batch_stride + h * nhead_stride + n * seqlen_stride + d]

namespace ck_tile {
namespace reference {

// -------------------------------------------------------------------------
// Q preprocessing: per-Q-block column-mean subtraction.
// For each block of q_block_size rows, compute column mean, subtract, optionally store mean.
//
// Inputs:
//   Q:            [B, H, seqlen_q, hdim]  float32
//   B, H, seqlen_q, hdim, q_block_size:   dimensions
// Outputs:
//   q_mean_out:   [B, H, num_q_blocks, hdim] float32  (column mean per q-block)
//   q_smooth_out: [B, H, seqlen_q, hdim]     float32  (Q - q_mean broadcast)
// -------------------------------------------------------------------------
inline void reference_sageattn_q_smooth(const float* Q,
                                        float*       q_mean_out,
                                        float*       q_smooth_out,
                                        int B,
                                        int H,
                                        int seqlen_q,
                                        int hdim,
                                        int q_block_size)
{
    const int num_q_blocks = (seqlen_q + q_block_size - 1) / q_block_size;

    for(int b = 0; b < B; b++)
    {
        for(int h = 0; h < H; h++)
        {
            for(int qi = 0; qi < num_q_blocks; qi++)
            {
                const int row_start = qi * q_block_size;
                const int row_end   = std::min(row_start + q_block_size, seqlen_q);
                const int n_rows    = row_end - row_start;

                // Compute column mean over the block
                for(int d = 0; d < hdim; d++)
                {
                    float sum = 0.0f;
                    for(int n = row_start; n < row_end; n++)
                    {
                        sum += Q[b * H * seqlen_q * hdim + h * seqlen_q * hdim + n * hdim + d];
                    }
                    const float mean = sum / static_cast<float>(n_rows);

                    // Store q_mean
                    if(q_mean_out != nullptr)
                    {
                        q_mean_out[b * H * num_q_blocks * hdim + h * num_q_blocks * hdim +
                                   qi * hdim + d] = mean;
                    }

                    // Subtract mean from all rows in this block
                    if(q_smooth_out != nullptr)
                    {
                        for(int n = row_start; n < row_end; n++)
                        {
                            q_smooth_out[b * H * seqlen_q * hdim + h * seqlen_q * hdim +
                                         n * hdim + d] =
                                Q[b * H * seqlen_q * hdim + h * seqlen_q * hdim + n * hdim + d] -
                                mean;
                        }
                    }
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// K preprocessing: global channel-mean subtraction.
// k_mean[d] = mean_{b,h,n}(K[b,h,n,d]) over all batches, heads, and tokens.
//
// Inputs:
//   K:            [B, H, seqlen_k, hdim]  float32
// Outputs:
//   k_mean_out:   [hdim]                  float32  (global channel mean)
//   k_smooth_out: [B, H, seqlen_k, hdim] float32  (K - k_mean)
// -------------------------------------------------------------------------
inline void reference_sageattn_k_smooth(const float* K,
                                        float*       k_mean_out,
                                        float*       k_smooth_out,
                                        int B,
                                        int H,
                                        int seqlen_k,
                                        int hdim)
{
    const int total_tokens = B * H * seqlen_k;

    // Compute global channel mean
    for(int d = 0; d < hdim; d++)
    {
        float sum = 0.0f;
        for(int b = 0; b < B; b++)
            for(int h = 0; h < H; h++)
                for(int n = 0; n < seqlen_k; n++)
                    sum += K[b * H * seqlen_k * hdim + h * seqlen_k * hdim + n * hdim + d];
        const float mean = sum / static_cast<float>(total_tokens);
        if(k_mean_out != nullptr)
            k_mean_out[d] = mean;

        // Subtract mean
        if(k_smooth_out != nullptr)
        {
            for(int b = 0; b < B; b++)
                for(int h = 0; h < H; h++)
                    for(int n = 0; n < seqlen_k; n++)
                        k_smooth_out[b * H * seqlen_k * hdim + h * seqlen_k * hdim + n * hdim +
                                     d] =
                            K[b * H * seqlen_k * hdim + h * seqlen_k * hdim + n * hdim + d] -
                            mean;
        }
    }
}

// -------------------------------------------------------------------------
// delta_s = Q_mean @ K^T
//
// Inputs:
//   q_mean:   [B, H, num_q_blocks, hdim]  float32
//   K:        [B, H, seqlen_k, hdim]      float32  (original unsmoothed K)
// Output:
//   delta_s:  [B, H, num_q_blocks, seqlen_k]  float32
// -------------------------------------------------------------------------
inline void reference_sageattn_delta_s(const float* q_mean,
                                       const float* K,
                                       float*       delta_s,
                                       int B,
                                       int H,
                                       int num_q_blocks,
                                       int seqlen_k,
                                       int hdim)
{
    for(int b = 0; b < B; b++)
    {
        for(int h = 0; h < H; h++)
        {
            for(int qi = 0; qi < num_q_blocks; qi++)
            {
                for(int kj = 0; kj < seqlen_k; kj++)
                {
                    float dot = 0.0f;
                    for(int d = 0; d < hdim; d++)
                    {
                        const float qm =
                            q_mean[b * H * num_q_blocks * hdim + h * num_q_blocks * hdim +
                                   qi * hdim + d];
                        const float kv =
                            K[b * H * seqlen_k * hdim + h * seqlen_k * hdim + kj * hdim + d];
                        dot += qm * kv;
                    }
                    delta_s[b * H * num_q_blocks * seqlen_k + h * num_q_blocks * seqlen_k +
                            qi * seqlen_k + kj] = dot;
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// V preprocessing: transpose [B, H, seqlen_k, hdim_v] -> [B, H, hdim_v, seqlen_k]
//
// Output V_T is stored row-major: V_T[b, h, d, n] = V[b, h, n, d]
// -------------------------------------------------------------------------
inline void reference_sageattn_v_transpose(const float* V,
                                            float*       V_T_out,
                                            int B,
                                            int H,
                                            int seqlen_k,
                                            int hdim_v)
{
    for(int b = 0; b < B; b++)
        for(int h = 0; h < H; h++)
            for(int n = 0; n < seqlen_k; n++)
                for(int d = 0; d < hdim_v; d++)
                    V_T_out[b * H * hdim_v * seqlen_k + h * hdim_v * seqlen_k + d * seqlen_k +
                             n] = V[b * H * seqlen_k * hdim_v + h * seqlen_k * hdim_v + n * hdim_v +
                                    d];
}

} // namespace reference
} // namespace ck_tile
