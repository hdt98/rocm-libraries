// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// SageAttnPreprocessPipeline: per-CTA computation for one tile of Q, K, and V.
//
// Each CTA (identified by tile_x, head, batch) performs ALL of Q, K, V work
// for its tile index, in the following order:
//
//   1. RunQMean:      compute column mean of Q tile → store in smem + global q_mean
//   2. RunDeltaS:     delta_s[tile_x, kj] = dot(q_mean, K[kj]) for all kj  (float)
//   3. RunQQuantize:  Q_smooth = Q - q_mean (using smem) → MXFP4
//   4. RunKQuantize:  K_smooth = K - k_mean (inline) → MXFP4
//   5. RunV:          V[:, channel] transpose → MXFP4
//
// Template parameters:
//   InputT_:           input element type (float or fp16_t)
//   kRows_:            tile rows (kM0 == kN0, used for Q and K tiles)
//   kCols_:            tile cols (hdim)
//   kScaleGranularity: MXFP4 group size (must be 32)
//   kBlockSize_:       threads per CTA

template <typename InputT_,
          index_t kRows_,
          index_t kCols_,
          index_t kScaleGranularity_ = 32,
          index_t kBlockSize_        = 256>
struct SageAttnPreprocessPipeline
{
    using InputT = InputT_;

    static constexpr index_t kRows             = kRows_;
    static constexpr index_t kCols             = kCols_;
    static constexpr index_t kScaleGranularity = kScaleGranularity_;
    static constexpr index_t kBlockSize        = kBlockSize_;

    static_assert(kScaleGranularity == 32, "MXFP4 scale granularity must be 32");
    static_assert(kCols % kScaleGranularity == 0,
                  "kCols must be divisible by kScaleGranularity for MXFP4");

    // Shared memory: kCols floats for q_mean broadcast
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kCols * sizeof(float);
    }

    // -------------------------------------------------------------------------
    // Step 1: compute column mean of [kRows, kCols] Q tile.
    //   smem[0..kCols) receives q_mean (float).
    //   q_mean_ptr: global output [kCols] for this Q tile.
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunQMean(const InputT* __restrict__ src_ptr,
                                 float* __restrict__ q_mean_ptr,
                                 void* smem) const
    {
        float* smem_mean  = reinterpret_cast<float*>(smem);
        const index_t tid = get_thread_id();

        // Each thread owns columns [tid, tid+kBlockSize, ...] and sums all kRows rows.
        for(index_t d = tid; d < kCols; d += kBlockSize)
        {
            float sum = 0.0f;
            for(index_t r = 0; r < kRows; r++)
                sum += static_cast<float>(src_ptr[r * kCols + d]);
            smem_mean[d] = sum / static_cast<float>(kRows);
        }
        block_sync_lds();

        // Write q_mean to global memory
        for(index_t d = tid; d < kCols; d += kBlockSize)
            q_mean_ptr[d] = smem_mean[d];
        // Note: no sync after this; smem_mean remains valid for subsequent steps.
    }

    // -------------------------------------------------------------------------
    // Step 2: delta_s[kj] = dot(q_mean_smem, K_smooth[kj]) for all kj in seqlen_k.
    //   K_smooth[kj, d] = K[kj, d] - k_mean[d]  (computed inline, not stored).
    //   Reads q_mean from smem (set by RunQMean).
    //   k_base_ptr: base of this (batch, head)'s K: [seqlen_k, kCols] row-major.
    //   k_mean_ptr: per-head channel mean [kCols] for this (batch, head).
    //   delta_s_ptr: global output [seqlen_k] for this Q tile.
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunDeltaS(const InputT* __restrict__ k_base_ptr,
                                  index_t seqlen_k,
                                  index_t stride_k,
                                  const float* __restrict__ k_mean_ptr,
                                  float* __restrict__ delta_s_ptr,
                                  const void* smem) const
    {
        const float*  smem_mean = reinterpret_cast<const float*>(smem);
        const index_t tid       = get_thread_id();

        for(index_t kj = tid; kj < seqlen_k; kj += kBlockSize)
        {
            const InputT* k_row = k_base_ptr + kj * stride_k;
            float dot           = 0.0f;
            for(index_t d = 0; d < kCols; d++)
            {
                const float k_smooth = static_cast<float>(k_row[d]) - k_mean_ptr[d];
                dot += smem_mean[d] * k_smooth;
            }
            delta_s_ptr[kj] = dot;
        }
    }

    // -------------------------------------------------------------------------
    // Step 3: quantize Q_smooth = Q_tile - q_mean (smem) → MXFP4.
    //   Reads q_mean from smem.
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunQQuantize(const InputT* __restrict__ src_ptr,
                                     uint8_t* __restrict__ dst_hat_ptr,
                                     uint8_t* __restrict__ dst_scale_ptr,
                                     const void* smem) const
    {
        const float*  smem_mean = reinterpret_cast<const float*>(smem);
        const index_t tid       = get_thread_id();

        for(index_t r = tid; r < kRows; r += kBlockSize)
        {
            QuantizeRowSubMean(src_ptr + r * kCols,
                               smem_mean,
                               dst_hat_ptr + r * (kCols / 2),
                               dst_scale_ptr + r * (kCols / kScaleGranularity));
        }
    }

    // -------------------------------------------------------------------------
    // Step 4: quantize K_smooth = K_tile - k_mean_ptr (inline) → MXFP4.
    //   k_mean_ptr: [kCols] per-head channel mean for this (batch, head).
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunKQuantize(const InputT* __restrict__ src_ptr,
                                     const float* __restrict__ k_mean_ptr,
                                     uint8_t* __restrict__ dst_hat_ptr,
                                     uint8_t* __restrict__ dst_scale_ptr) const
    {
        const index_t tid = get_thread_id();
        for(index_t r = tid; r < kRows; r += kBlockSize)
        {
            QuantizeRowSubMean(src_ptr + r * kCols,
                               k_mean_ptr,
                               dst_hat_ptr + r * (kCols / 2),
                               dst_scale_ptr + r * (kCols / kScaleGranularity));
        }
    }

    // -------------------------------------------------------------------------
    // Step 5: one hdim_v channel of V → transpose + MXFP4 quantize.
    //   src_ptr:      V[0, hdim_ch] in [seqlen_k, hdim_v] layout (stride hdim_v per row)
    //   dst_hat_ptr:  V_hat[hdim_ch, 0] in [hdim_v, seqlen_k/2] layout
    //   dst_scale_ptr:V_scale[hdim_ch, 0] in [hdim_v, seqlen_k/32] layout
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunV(const InputT* __restrict__ src_ptr,
                             index_t seqlen_k,
                             index_t hdim_v,
                             uint8_t* __restrict__ dst_hat_ptr,
                             uint8_t* __restrict__ dst_scale_ptr) const
    {
        const index_t tid       = get_thread_id();
        const index_t num_groups = seqlen_k / kScaleGranularity;
        constexpr float rcp_dst_max = 1.0f / 6.0f;

        for(index_t g = tid; g < num_groups; g += kBlockSize)
        {
            const index_t k_start = g * kScaleGranularity;
            float group_data[kScaleGranularity];
            float max_abs = 0.0f;

            for(index_t j = 0; j < kScaleGranularity; j++)
            {
                group_data[j] = static_cast<float>(src_ptr[(k_start + j) * hdim_v]);
                max_abs       = max(max_abs, abs(group_data[j]));
            }

            const float scale = bit_cast<float>(
                (bit_cast<uint32_t>(max_abs * rcp_dst_max) + numeric_traits<float>::mant_mask) &
                numeric_traits<float>::head_mask);

            dst_scale_ptr[g] = static_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);

            uint8_t* hat_group = dst_hat_ptr + g * (kScaleGranularity / 2);
            PackFP4Group(group_data, hat_group, scale);
        }
    }

private:
    // Quantize one row of kCols InputT values: val = src[d] - mean[d], then MXFP4.
    CK_TILE_DEVICE void QuantizeRowSubMean(const InputT* __restrict__ src_row,
                                            const float* __restrict__ mean,
                                            uint8_t* __restrict__ dst_hat_row,
                                            uint8_t* __restrict__ dst_scale_row) const
    {
        constexpr float rcp_dst_max  = 1.0f / 6.0f;
        const index_t   num_groups   = kCols / kScaleGranularity;

        for(index_t g = 0; g < num_groups; g++)
        {
            const index_t d_start = g * kScaleGranularity;
            float group_data[kScaleGranularity];
            float max_abs = 0.0f;

            for(index_t j = 0; j < kScaleGranularity; j++)
            {
                const float val = static_cast<float>(src_row[d_start + j]) - mean[d_start + j];
                group_data[j]   = val;
                max_abs         = max(max_abs, abs(val));
            }

            const float scale = bit_cast<float>(
                (bit_cast<uint32_t>(max_abs * rcp_dst_max) + numeric_traits<float>::mant_mask) &
                numeric_traits<float>::head_mask);

            dst_scale_row[g] = static_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);

            uint8_t* hat_group = dst_hat_row + g * (kScaleGranularity / 2);
            PackFP4Group(group_data, hat_group, scale);
        }
    }

    // Pack 32 floats into 16 bytes (packed FP4) using AMD CVT intrinsic.
    CK_TILE_DEVICE void PackFP4Group(const float* __restrict__ group_data,
                                      uint8_t* __restrict__ hat_group,
                                      float scale) const
    {
        for(index_t j = 0; j < kScaleGranularity / 8; j++)
        {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
            uint32_t x;
            x = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
                x, group_data[8 * j + 0], group_data[8 * j + 1], scale, 0);
            x = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
                x, group_data[8 * j + 2], group_data[8 * j + 3], scale, 1);
            x = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
                x, group_data[8 * j + 4], group_data[8 * j + 5], scale, 2);
            x = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
                x, group_data[8 * j + 6], group_data[8 * j + 7], scale, 3);
#pragma clang diagnostic pop
            *reinterpret_cast<uint32_t*>(hat_group + j * 4) = x;
        }
    }
};

} // namespace ck_tile
