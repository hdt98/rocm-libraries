// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "ck_tile/host/reference/reference_mxfp4.hpp"

// CPU float32 reference implementations for SageAttention V3 preprocessing.
//
// All functions use [B, H, seqlen, hdim] row-major layout.
// Layout convention:
//   tensor[b, h, n, d] = ptr[b * batch_stride + h * nhead_stride + n * seqlen_stride + d]

namespace ck_tile {
namespace reference {

// -------------------------------------------------------------------------
// Q preprocessing: per-Q-block column-mean subtraction + MXFP4 quantize.
//
// Inputs:
//   Q:  [B, H, seqlen_q, hdim]  float
// Outputs:
//   q_mean_out:  [B, H, num_q_blocks, hdim]  float  (column mean per block)
//   q_hat_out:   [B, H, seqlen_q, hdim/2]   uint8  (MXFP4 packed, lower nibble first)
//   q_scale_out: [B, H, seqlen_q, hdim/32]  uint8  (e8m0 scale bytes)
// -------------------------------------------------------------------------
inline void reference_sageattn_v3_q_preprocess(const float* Q,
                                             float*       q_mean_out,
                                             uint8_t*     q_hat_out,
                                             uint8_t*     q_scale_out,
                                             int          B,
                                             int          H,
                                             int          seqlen_q,
                                             int          hdim,
                                             int          q_block_size)
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

                // Compute column mean for this Q block
                std::vector<float> col_mean(hdim, 0.0f);
                for(int d = 0; d < hdim; d++)
                {
                    float sum = 0.0f;
                    for(int n = row_start; n < row_end; n++)
                        sum += Q[b * H * seqlen_q * hdim + h * seqlen_q * hdim + n * hdim + d];
                    col_mean[d] = sum / static_cast<float>(n_rows);
                }

                // Write q_mean
                if(q_mean_out != nullptr)
                {
                    for(int d = 0; d < hdim; d++)
                        q_mean_out[b * H * num_q_blocks * hdim + h * num_q_blocks * hdim +
                                   qi * hdim + d] = col_mean[d];
                }

                // Quantize each row: Q_smooth = Q - col_mean → MXFP4
                if(q_hat_out != nullptr && q_scale_out != nullptr)
                {
                    std::vector<float> row_smooth(hdim);
                    for(int n = row_start; n < row_end; n++)
                    {
                        for(int d = 0; d < hdim; d++)
                            row_smooth[d] =
                                Q[b * H * seqlen_q * hdim + h * seqlen_q * hdim + n * hdim + d] -
                                col_mean[d];

                        const int hat_row_off =
                            b * H * seqlen_q * (hdim / 2) + h * seqlen_q * (hdim / 2) +
                            n * (hdim / 2);
                        const int scale_row_off =
                            b * H * seqlen_q * (hdim / kMxfp4Group) +
                            h * seqlen_q * (hdim / kMxfp4Group) + n * (hdim / kMxfp4Group);

                        mxfp4_quantize(row_smooth.data(),
                                       hdim,
                                       q_hat_out + hat_row_off,
                                       q_scale_out + scale_row_off);
                    }
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// Compute per-head channel mean of K.
//
// k_mean_out[b, h, d] = mean over seqlen_k of K[b, h, :, d].
//
// Inputs:
//   K:           [B, H, seqlen_k, hdim]  float
// Output:
//   k_mean_out:  [B, H, hdim]            float
// -------------------------------------------------------------------------
inline void reference_sageattn_v3_k_smooth(const float* K,
                                        float*       k_mean_out,
                                        int          B,
                                        int          H,
                                        int          seqlen_k,
                                        int          hdim)
{
    for(int b = 0; b < B; b++)
    {
        for(int h = 0; h < H; h++)
        {
            for(int d = 0; d < hdim; d++)
            {
                float sum = 0.0f;
                for(int n = 0; n < seqlen_k; n++)
                    sum += K[b * H * seqlen_k * hdim + h * seqlen_k * hdim + n * hdim + d];
                k_mean_out[b * H * hdim + h * hdim + d] = sum / static_cast<float>(seqlen_k);
            }
        }
    }
}

// -------------------------------------------------------------------------
// K preprocessing: K_smooth = K - k_mean → MXFP4 quantize.
//
// Inputs:
//   K:       [B, H, seqlen_k, hdim]  float
//   k_mean:  [B, H, hdim]            float  (per-head channel mean)
// Outputs:
//   k_hat_out:   [B, H, seqlen_k, hdim/2]   uint8
//   k_scale_out: [B, H, seqlen_k, hdim/32]  uint8
// -------------------------------------------------------------------------
inline void reference_sageattn_v3_k_preprocess(const float* K,
                                             const float* k_mean,
                                             uint8_t*     k_hat_out,
                                             uint8_t*     k_scale_out,
                                             int          B,
                                             int          H,
                                             int          seqlen_k,
                                             int          hdim)
{
    std::vector<float> row_smooth(hdim);
    for(int b = 0; b < B; b++)
    {
        for(int h = 0; h < H; h++)
        {
            for(int n = 0; n < seqlen_k; n++)
            {
                for(int d = 0; d < hdim; d++)
                    row_smooth[d] =
                        K[b * H * seqlen_k * hdim + h * seqlen_k * hdim + n * hdim + d] -
                        k_mean[b * H * hdim + h * hdim + d];

                const int hat_row_off =
                    b * H * seqlen_k * (hdim / 2) + h * seqlen_k * (hdim / 2) + n * (hdim / 2);
                const int scale_row_off =
                    b * H * seqlen_k * (hdim / kMxfp4Group) +
                    h * seqlen_k * (hdim / kMxfp4Group) + n * (hdim / kMxfp4Group);

                mxfp4_quantize(
                    row_smooth.data(), hdim, k_hat_out + hat_row_off, k_scale_out + scale_row_off);
            }
        }
    }
}

// -------------------------------------------------------------------------
// delta_s = Q_mean @ K_smooth^T
//
// K_smooth[kj, d] = K[kj, d] - k_mean[d]  (per-head channel mean subtraction)
//
// Inputs:
//   q_mean:  [B, H, num_q_blocks, hdim]  float
//   K:       [B, H, seqlen_k, hdim]      float  (original unsmoothed K)
//   k_mean:  [B, H, hdim]                float  (per-head channel mean)
// Output:
//   delta_s: [B, H, num_q_blocks, seqlen_k]  float
// -------------------------------------------------------------------------
inline void reference_sageattn_v3_delta_s(const float* q_mean,
                                       const float* K,
                                       const float* k_mean,
                                       float*       delta_s,
                                       int          B,
                                       int          H,
                                       int          num_q_blocks,
                                       int          seqlen_k,
                                       int          hdim)
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
                        const float km = k_mean[b * H * hdim + h * hdim + d];
                        dot += qm * (kv - km);
                    }
                    delta_s[b * H * num_q_blocks * seqlen_k + h * num_q_blocks * seqlen_k +
                            qi * seqlen_k + kj] = dot;
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// V preprocessing: transpose [B, H, seqlen_k, hdim_v] → [B, H, hdim_v, seqlen_k]
// and MXFP4 quantize along seqlen_k dimension (groups of 32).
//
// Outputs:
//   v_hat_out:   [B, H, hdim_v, seqlen_k/2]         uint8 (MXFP4 packed)
//   v_scale_out: [B, H, hdim_v, seqlen_k/kMxfp4Group] uint8 (e8m0 scale bytes)
// -------------------------------------------------------------------------
inline void reference_sageattn_v3_v_preprocess(const float* V,
                                             uint8_t*     v_hat_out,
                                             uint8_t*     v_scale_out,
                                             int          B,
                                             int          H,
                                             int          seqlen_k,
                                             int          hdim_v)
{
    std::vector<float> col(seqlen_k); // one channel column across seqlen_k
    for(int b = 0; b < B; b++)
    {
        for(int h = 0; h < H; h++)
        {
            for(int d = 0; d < hdim_v; d++)
            {
                for(int n = 0; n < seqlen_k; n++)
                    col[n] = V[b * H * seqlen_k * hdim_v + h * seqlen_k * hdim_v + n * hdim_v + d];

                const int hat_ch_off =
                    b * H * hdim_v * (seqlen_k / 2) + h * hdim_v * (seqlen_k / 2) +
                    d * (seqlen_k / 2);
                const int scale_ch_off =
                    b * H * hdim_v * (seqlen_k / kMxfp4Group) +
                    h * hdim_v * (seqlen_k / kMxfp4Group) + d * (seqlen_k / kMxfp4Group);

                mxfp4_quantize(
                    col.data(), seqlen_k, v_hat_out + hat_ch_off, v_scale_out + scale_ch_off);
            }
        }
    }
}

// -------------------------------------------------------------------------
// Dequantize MXFP4 tensor [B, H, rows, cols] — thin wrapper over
// reference_dequant_mxfp4 from reference_mxfp4.hpp kept for name consistency.
//
//   hat   layout: [B, H, rows, cols/2]
//   scale layout: [B, H, rows, cols/kMxfp4Group]
//   out   layout: [B, H, rows, cols]
// -------------------------------------------------------------------------
inline void reference_sageattn_v3_dequant_mxfp4(const uint8_t* hat,
                                              const uint8_t* scale_bytes,
                                              float*         out,
                                              int            B,
                                              int            H,
                                              int            rows,
                                              int            cols)
{
    reference_dequant_mxfp4(hat, scale_bytes, out, B, H, rows, cols);
}

// -------------------------------------------------------------------------
// K pre-shuffle for contiguous async buffer_load_dwordx4 to LDS.
//
// The async K load uses warp-interleaved N ordering in LDS:
//   N_interleaved = lane_group * NumWarps + warp_id
// but contiguous DRAM access gives:
//   N_contiguous  = warp_id * LaneGroups + lane_group
//
// This function permutes K rows within each (LaneGroups * NumWarps)-row
// block so that contiguous loads produce the correct interleaved LDS layout.
//
//   K_shuffled[n][k] = K_original[(n % LaneGroups) * NumWarps + n / LaneGroups][k]
//
// Template parameters:
//   kLaneGroups: number of lane groups per warp (WarpSize / LanesPerK)
//   kNumWarps:   number of warps in the block
//
// Data layout: K[B, H, seqlen_k, hdim_packed] where hdim_packed is in
// packed-type units (e.g. pk_fp4_t bytes or fp16 half-words).
// -------------------------------------------------------------------------
template <int kLaneGroups, int kNumWarps, typename T>
inline void reference_sageattn_v3_k_shuffle(const T* k_in,
                                            T*       k_out,
                                            int      B,
                                            int      H,
                                            int      seqlen_k,
                                            int      hdim_packed)
{
    constexpr int kBlockRows = kLaneGroups * kNumWarps;
    const int     num_blocks = seqlen_k / kBlockRows;

    for(int b = 0; b < B; b++)
    {
        for(int h = 0; h < H; h++)
        {
            for(int blk = 0; blk < num_blocks; blk++)
            {
                const int base_row = blk * kBlockRows;
                for(int n = 0; n < kBlockRows; n++)
                {
                    const int src_n = (n % kLaneGroups) * kNumWarps + n / kLaneGroups;
                    const int dst_off =
                        ((b * H + h) * seqlen_k + base_row + n) * hdim_packed;
                    const int src_off =
                        ((b * H + h) * seqlen_k + base_row + src_n) * hdim_packed;
                    std::copy(k_in + src_off, k_in + src_off + hdim_packed, k_out + dst_off);
                }
            }
        }
    }
}

// V pre-shuffle for contiguous async buffer_load_dwordx4 to LDS.
//
// V after preprocessing is [B, Hkv, hdim_v, seqlen_k_packed] — same structure
// as K's [B, Hkv, seqlen_k, hdim_packed]. Both are [N, K]-major with K contiguous.
// The same (LaneGroups, NumWarps) transpose used for K applies to V:
//   V_shuffled[n] = V_original[(n % LaneGroups) * NumWarps + n / LaneGroups]
//
// Template parameters: same as K shuffle (kLaneGroups, kNumWarps).
// Data layout: V[B, H, hdim_v, seqlen_k_packed] (packed type units).
template <int kLaneGroups, int kNumWarps, typename T>
inline void reference_sageattn_v3_v_shuffle(const T* v_in,
                                            T*       v_out,
                                            int      B,
                                            int      H,
                                            int      hdim_v,
                                            int      seqlen_k_packed)
{
    constexpr int kBlockRows = kLaneGroups * kNumWarps;
    const int     num_blocks = hdim_v / kBlockRows;

    for(int b = 0; b < B; b++)
    {
        for(int h = 0; h < H; h++)
        {
            for(int blk = 0; blk < num_blocks; blk++)
            {
                const int base_row = blk * kBlockRows;
                for(int n = 0; n < kBlockRows; n++)
                {
                    const int src_n = (n % kLaneGroups) * kNumWarps + n / kLaneGroups;
                    const int dst_off =
                        ((b * H + h) * hdim_v + base_row + n) * seqlen_k_packed;
                    const int src_off =
                        ((b * H + h) * hdim_v + base_row + src_n) * seqlen_k_packed;
                    std::copy(v_in + src_off, v_in + src_off + seqlen_k_packed, v_out + dst_off);
                }
            }
        }
    }
}

} // namespace reference
} // namespace ck_tile
