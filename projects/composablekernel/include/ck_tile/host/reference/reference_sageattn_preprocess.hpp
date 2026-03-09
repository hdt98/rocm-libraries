// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// CPU float32 reference implementations for SageAttention V3 preprocessing.
//
// All functions use [B, H, seqlen, hdim] row-major layout.
// Layout convention:
//   tensor[b, h, n, d] = ptr[b * batch_stride + h * nhead_stride + n * seqlen_stride + d]
//
// MXFP4 (E2M1) representable values (positive half):
//   0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
//   Packing: one byte holds two FP4 values; lower nibble = element[even], upper = element[odd].
//   e8m0 scale:  float = bit_cast<float>(uint32_t(scale_byte) << 23)

namespace ck_tile {
namespace reference {

// -------------------------------------------------------------------------
// MXFP4 helpers
// -------------------------------------------------------------------------

// Decode e8m0 scale byte to float32.
inline float sageattn_e8m0_to_float(uint8_t scale_byte)
{
    uint32_t bits = static_cast<uint32_t>(scale_byte) << 23;
    float    f;
    __builtin_memcpy(&f, &bits, sizeof(float));
    return f;
}

// Decode a 4-bit FP4 E2M1 nibble (0-15) to float32 (without scale).
inline float sageattn_fp4_nibble_to_float(uint8_t nibble)
{
    static const float vals[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    return vals[nibble & 0xF];
}

// Round a float32 value to the nearest FP4 E2M1 value (without scale).
// Returns the 4-bit nibble code (0-15).
inline uint8_t sageattn_float_to_fp4_nibble(float x)
{
    static const float abs_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const bool negative = (x < 0.0f);
    const float x_abs   = std::abs(x);

    int   best     = 0;
    float best_err = std::abs(x_abs - abs_vals[0]);
    for(int i = 1; i < 8; i++)
    {
        float err = std::abs(x_abs - abs_vals[i]);
        if(err < best_err)
        {
            best_err = err;
            best     = i;
        }
    }
    return static_cast<uint8_t>(negative ? (best | 8) : best);
}

// Compute e8m0 scale for a group of 32 floats given max absolute value.
// Formula: smallest power-of-2 scale such that max_abs / scale <= 6.
inline uint8_t sageattn_compute_scale_byte(float max_abs)
{
    // scale = ceil_pow2(max_abs / 6.0f)
    constexpr float rcp_dst_max = 1.0f / 6.0f;
    const float     scaled      = max_abs * rcp_dst_max;

    // Extract exponent of scaled (round up mantissa to get ceiling power of 2)
    uint32_t bits_in;
    __builtin_memcpy(&bits_in, &scaled, sizeof(float));
    constexpr uint32_t mant_mask = 0x007FFFFFu;
    constexpr uint32_t head_mask = 0xFF800000u;
    const uint32_t     bits_out  = (bits_in + mant_mask) & head_mask;

    float scale;
    __builtin_memcpy(&scale, &bits_out, sizeof(float));
    // Scale byte = exponent bits [30:23]
    return static_cast<uint8_t>(bits_out >> 23);
}

// -------------------------------------------------------------------------
// Q preprocessing: per-Q-block column-mean subtraction.
//
// Inputs:
//   Q:  [B, H, seqlen_q, hdim]  InputT (float or fp16-as-float)
// Outputs:
//   q_mean_out:  [B, H, num_q_blocks, hdim]  float  (column mean per block)
//   q_hat_out:   [B, H, seqlen_q, hdim/2]   uint8  (MXFP4 packed, lower nibble first)
//   q_scale_out: [B, H, seqlen_q, hdim/32]  uint8  (e8m0 scale bytes)
// -------------------------------------------------------------------------
inline void reference_sageattn_q_preprocess(const float* Q,
                                             float*       q_mean_out,
                                             uint8_t*     q_hat_out,
                                             uint8_t*     q_scale_out,
                                             int B,
                                             int H,
                                             int seqlen_q,
                                             int hdim,
                                             int q_block_size)
{
    const int num_q_blocks = (seqlen_q + q_block_size - 1) / q_block_size;
    constexpr int kG = 32; // MXFP4 group size

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
                    for(int n = row_start; n < row_end; n++)
                    {
                        const int num_groups = hdim / kG;
                        for(int g = 0; g < num_groups; g++)
                        {
                            const int d_start = g * kG;

                            float max_abs = 0.0f;
                            for(int j = 0; j < kG; j++)
                            {
                                float val = Q[b * H * seqlen_q * hdim + h * seqlen_q * hdim +
                                              n * hdim + d_start + j] -
                                            col_mean[d_start + j];
                                max_abs = std::max(max_abs, std::abs(val));
                            }

                            const uint8_t scale_byte = sageattn_compute_scale_byte(max_abs);
                            const float   scale       = sageattn_e8m0_to_float(scale_byte);

                            const int scale_off = b * H * seqlen_q * (hdim / kG) +
                                                  h * seqlen_q * (hdim / kG) + n * (hdim / kG) + g;
                            q_scale_out[scale_off] = scale_byte;

                            for(int j = 0; j < kG; j += 2)
                            {
                                float v0 = Q[b * H * seqlen_q * hdim + h * seqlen_q * hdim +
                                              n * hdim + d_start + j] -
                                           col_mean[d_start + j];
                                float v1 = Q[b * H * seqlen_q * hdim + h * seqlen_q * hdim +
                                              n * hdim + d_start + j + 1] -
                                           col_mean[d_start + j + 1];
                                const uint8_t n0 = sageattn_float_to_fp4_nibble(v0 / scale);
                                const uint8_t n1 = sageattn_float_to_fp4_nibble(v1 / scale);
                                const int hat_off =
                                    b * H * seqlen_q * (hdim / 2) + h * seqlen_q * (hdim / 2) +
                                    n * (hdim / 2) + (d_start + j) / 2;
                                q_hat_out[hat_off] = static_cast<uint8_t>((n1 << 4) | (n0 & 0xF));
                            }
                        }
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
inline void reference_sageattn_k_smooth(const float* K,
                                        float*       k_mean_out,
                                        int B,
                                        int H,
                                        int seqlen_k,
                                        int hdim)
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

inline void reference_sageattn_k_preprocess(const float* K,
                                             const float* k_mean,
                                             uint8_t*     k_hat_out,
                                             uint8_t*     k_scale_out,
                                             int B,
                                             int H,
                                             int seqlen_k,
                                             int hdim)
{
    constexpr int kG = 32;
    for(int b = 0; b < B; b++)
    {
        for(int h = 0; h < H; h++)
        {
            for(int n = 0; n < seqlen_k; n++)
            {
                const int num_groups = hdim / kG;
                for(int g = 0; g < num_groups; g++)
                {
                    const int d_start = g * kG;
                    float     max_abs = 0.0f;
                    for(int j = 0; j < kG; j++)
                    {
                        float val = K[b * H * seqlen_k * hdim + h * seqlen_k * hdim + n * hdim +
                                      d_start + j] -
                                    k_mean[b * H * hdim + h * hdim + d_start + j];
                        max_abs = std::max(max_abs, std::abs(val));
                    }

                    const uint8_t scale_byte = sageattn_compute_scale_byte(max_abs);
                    const float   scale       = sageattn_e8m0_to_float(scale_byte);

                    const int scale_off = b * H * seqlen_k * (hdim / kG) +
                                          h * seqlen_k * (hdim / kG) + n * (hdim / kG) + g;
                    k_scale_out[scale_off] = scale_byte;

                    for(int j = 0; j < kG; j += 2)
                    {
                        float v0 = K[b * H * seqlen_k * hdim + h * seqlen_k * hdim + n * hdim +
                                     d_start + j] -
                                   k_mean[b * H * hdim + h * hdim + d_start + j];
                        float v1 = K[b * H * seqlen_k * hdim + h * seqlen_k * hdim + n * hdim +
                                     d_start + j + 1] -
                                   k_mean[b * H * hdim + h * hdim + d_start + j + 1];
                        const uint8_t n0      = sageattn_float_to_fp4_nibble(v0 / scale);
                        const uint8_t n1      = sageattn_float_to_fp4_nibble(v1 / scale);
                        const int     hat_off = b * H * seqlen_k * (hdim / 2) +
                                            h * seqlen_k * (hdim / 2) + n * (hdim / 2) +
                                            (d_start + j) / 2;
                        k_hat_out[hat_off] = static_cast<uint8_t>((n1 << 4) | (n0 & 0xF));
                    }
                }
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
inline void reference_sageattn_delta_s(const float* q_mean,
                                       const float* K,
                                       const float* k_mean,
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
//   v_hat_out:   [B, H, hdim_v, seqlen_k/2]   uint8 (MXFP4 packed)
//   v_scale_out: [B, H, hdim_v, seqlen_k/32]  uint8 (e8m0 scale bytes)
// -------------------------------------------------------------------------
inline void reference_sageattn_v_preprocess(const float* V,
                                             uint8_t*     v_hat_out,
                                             uint8_t*     v_scale_out,
                                             int B,
                                             int H,
                                             int seqlen_k,
                                             int hdim_v)
{
    constexpr int kG = 32;
    const int     num_groups = seqlen_k / kG;

    for(int b = 0; b < B; b++)
    {
        for(int h = 0; h < H; h++)
        {
            for(int d = 0; d < hdim_v; d++)
            {
                for(int g = 0; g < num_groups; g++)
                {
                    const int k_start = g * kG;
                    float     max_abs = 0.0f;

                    for(int j = 0; j < kG; j++)
                    {
                        float val = V[b * H * seqlen_k * hdim_v + h * seqlen_k * hdim_v +
                                      (k_start + j) * hdim_v + d];
                        max_abs   = std::max(max_abs, std::abs(val));
                    }

                    const uint8_t scale_byte = sageattn_compute_scale_byte(max_abs);
                    const float   scale       = sageattn_e8m0_to_float(scale_byte);

                    // v_scale layout: [B, H, hdim_v, seqlen_k/kG]
                    const int scale_off = b * H * hdim_v * num_groups +
                                          h * hdim_v * num_groups + d * num_groups + g;
                    v_scale_out[scale_off] = scale_byte;

                    // v_hat layout: [B, H, hdim_v, seqlen_k/2]
                    for(int j = 0; j < kG; j += 2)
                    {
                        float v0 = V[b * H * seqlen_k * hdim_v + h * seqlen_k * hdim_v +
                                     (k_start + j) * hdim_v + d];
                        float v1 = V[b * H * seqlen_k * hdim_v + h * seqlen_k * hdim_v +
                                     (k_start + j + 1) * hdim_v + d];
                        const uint8_t n0 = sageattn_float_to_fp4_nibble(v0 / scale);
                        const uint8_t n1 = sageattn_float_to_fp4_nibble(v1 / scale);
                        const int hat_off =
                            b * H * hdim_v * (seqlen_k / 2) + h * hdim_v * (seqlen_k / 2) +
                            d * (seqlen_k / 2) + (k_start + j) / 2;
                        v_hat_out[hat_off] = static_cast<uint8_t>((n1 << 4) | (n0 & 0xF));
                    }
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// Dequantize MXFP4 output back to float32 (for correctness verification).
//
// hat layout: [B, H, rows, cols/2]  (each byte = two nibbles, lower first)
// scale layout: [B, H, rows, cols/32]
// out layout: [B, H, rows, cols]
// -------------------------------------------------------------------------
inline void reference_sageattn_dequant_mxfp4(const uint8_t* hat,
                                              const uint8_t* scale_bytes,
                                              float*         out,
                                              int B,
                                              int H,
                                              int rows,
                                              int cols)
{
    constexpr int kG = 32;
    for(int b = 0; b < B; b++)
    {
        for(int h = 0; h < H; h++)
        {
            for(int r = 0; r < rows; r++)
            {
                const int num_groups = cols / kG;
                for(int g = 0; g < num_groups; g++)
                {
                    const int scale_off = b * H * rows * num_groups + h * rows * num_groups +
                                          r * num_groups + g;
                    const float scale = sageattn_e8m0_to_float(scale_bytes[scale_off]);

                    const int d_start = g * kG;
                    for(int j = 0; j < kG; j += 2)
                    {
                        const int hat_off = b * H * rows * (cols / 2) + h * rows * (cols / 2) +
                                            r * (cols / 2) + (d_start + j) / 2;
                        const uint8_t byte = hat[hat_off];
                        const uint8_t n0   = byte & 0xF;
                        const uint8_t n1   = (byte >> 4) & 0xF;
                        out[b * H * rows * cols + h * rows * cols + r * cols + d_start + j] =
                            sageattn_fp4_nibble_to_float(n0) * scale;
                        out[b * H * rows * cols + h * rows * cols + r * cols + d_start + j + 1] =
                            sageattn_fp4_nibble_to_float(n1) * scale;
                    }
                }
            }
        }
    }
}


} // namespace reference
} // namespace ck_tile
