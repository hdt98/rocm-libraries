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
//   1. RunQMean:             compute column mean of Q tile → store in smem + global q_mean (InputT)
//   2. RunQQuantize:         Q_smooth = Q - q_mean (using smem) → MXFP4
//   3. RunKSmoothAndQuantize: K' = K - k_mean → write k_prime; K' → MXFP4
//   4. RunV:                 V[:, channel] transpose → MXFP4
//
// delta_s = q_mean @ K'^T is computed separately via a batched GEMM kernel.
//
// RunKMeanPartial: used by SageAttnKMeanKernel (separate launch, different grid).
//   Each CTA accumulates a partial column-sum of its K tile into k_mean_partial (float atomicAdd).
//   The last CTA to finish normalises and stores k_mean as InputT.
//
// Template parameters:
//   InputT_:           input element type (fp16_t or bf16_t)
//   kRows_:            tile rows (kM0 == kN0, used for Q and K tiles)
//   kCols_:            tile cols (hdim)
//   kScaleGranularity: MXFP4 group size (must be 32)
//   kBlockSize_:       threads per CTA (preprocess kernel)

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

    // Shared memory: kCols InputT elements for q_mean broadcast.
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kCols * sizeof(InputT);
    }

    // -------------------------------------------------------------------------
    // Step 1: compute column mean of [kRows, kCols] Q tile.
    //   Accumulates in InputT; smem[0..kCols) and q_mean_ptr receive the result.
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunQMean(const InputT* __restrict__ src_ptr,
                                 InputT* __restrict__ q_mean_ptr,
                                 void* smem) const
    {
        InputT*       smem_mean = reinterpret_cast<InputT*>(smem);
        const index_t tid       = get_thread_id();

        for(index_t d = tid; d < kCols; d += kBlockSize)
        {
            InputT sum = InputT(0);
            for(index_t r = 0; r < kRows; r++)
                sum += src_ptr[r * kCols + d];
            // Divide in float for precision, cast back to InputT.
            smem_mean[d] = static_cast<InputT>(static_cast<float>(sum) / static_cast<float>(kRows));
        }
        block_sync_lds();

        for(index_t d = tid; d < kCols; d += kBlockSize)
            q_mean_ptr[d] = smem_mean[d];
        // smem_mean remains valid for RunQQuantize (no sync needed after write-to-global).
    }

    // -------------------------------------------------------------------------
    // Step 2: quantize Q_smooth = Q_tile - q_mean (smem) → MXFP4.
    //   Reads q_mean from smem (set by RunQMean).
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunQQuantize(const InputT* __restrict__ src_ptr,
                                     uint8_t* __restrict__ dst_hat_ptr,
                                     uint8_t* __restrict__ dst_scale_ptr,
                                     const void* smem) const
    {
        const InputT* smem_mean = reinterpret_cast<const InputT*>(smem);
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
    // Step 3: compute K' = K_tile - k_mean, write to k_prime_ptr in natural
    //   row-major layout [kRows, kCols], then quantize K' → MXFP4.
    //   k_mean_ptr:   [kCols] InputT per-head channel mean for this (batch, head).
    //   k_prime_ptr:  pointer to K'[row_start, 0] in [seqlen_k, hdim] global layout.
    //   k_prime_stride: hdim (row stride of K').
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunKSmoothAndQuantize(const InputT* __restrict__ src_ptr,
                                              const InputT* __restrict__ k_mean_ptr,
                                              InputT* __restrict__ k_prime_ptr,
                                              index_t k_prime_stride,
                                              uint8_t* __restrict__ dst_hat_ptr,
                                              uint8_t* __restrict__ dst_scale_ptr) const
    {
        const index_t tid = get_thread_id();
        InputT kprime_row[kCols]; // kCols is constexpr — stack array.
        for(index_t r = tid; r < kRows; r += kBlockSize)
        {
            const InputT* src_row  = src_ptr + r * kCols;
            InputT*       dst_row  = k_prime_ptr + r * k_prime_stride;

            for(index_t d = 0; d < kCols; d++)
            {
                kprime_row[d] =
                    static_cast<InputT>(static_cast<float>(src_row[d]) -
                                        static_cast<float>(k_mean_ptr[d]));
                dst_row[d] = kprime_row[d];
            }

            QuantizeRow(kprime_row,
                        dst_hat_ptr + r * (kCols / 2),
                        dst_scale_ptr + r * (kCols / kScaleGranularity));
        }
    }

    // -------------------------------------------------------------------------
    // RunKMeanPartial: called from SageAttnKMeanKernel (BlockSize = kCols).
    //   Each thread d accumulates K[row_start..row_end, d] into k_mean_partial[d]
    //   via float atomicAdd.
    //
    //   k_tile_ptr:      K[row_start, 0] for this (batch, head, tile_x)
    //   n_rows:          actual rows in this tile (handles tail padding)
    //   stride_k:        hdim (columns per row)
    //   k_mean_partial:  [kCols] float accumulator for this (batch, head), pre-zeroed
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunKMeanPartial(const InputT* __restrict__ k_tile_ptr,
                                        index_t n_rows,
                                        index_t stride_k,
                                        float* __restrict__ k_mean_partial) const
    {
        // One thread per d-channel (BlockSize == kCols in SageAttnKMeanKernel).
        const index_t d = get_thread_id();

        float acc = 0.0f;
        for(index_t r = 0; r < n_rows; r++)
            acc += static_cast<float>(k_tile_ptr[r * stride_k + d]);

        atomicAdd(&k_mean_partial[d], acc);
    }

private:
    // -------------------------------------------------------------------------
    // Quantize one row of kCols InputT values after subtracting mean: MXFP4.
    //   val = src[d] - mean[d]
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void QuantizeRowSubMean(const InputT* __restrict__ src_row,
                                            const InputT* __restrict__ mean,
                                            uint8_t* __restrict__ dst_hat_row,
                                            uint8_t* __restrict__ dst_scale_row) const
    {
        constexpr float rcp_dst_max = 1.0f / 6.0f;
        const index_t   num_groups  = kCols / kScaleGranularity;

        for(index_t g = 0; g < num_groups; g++)
        {
            const index_t d_start = g * kScaleGranularity;
            float group_data[kScaleGranularity];
            float max_abs = 0.0f;

            for(index_t j = 0; j < kScaleGranularity; j++)
            {
                const float val = static_cast<float>(src_row[d_start + j]) -
                                  static_cast<float>(mean[d_start + j]);
                group_data[j] = val;
                max_abs       = max(max_abs, abs(val));
            }

            const float scale = bit_cast<float>(
                (bit_cast<uint32_t>(max_abs * rcp_dst_max) + numeric_traits<float>::mant_mask) &
                numeric_traits<float>::head_mask);

            dst_scale_row[g] = static_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);

            uint8_t* hat_group = dst_hat_row + g * (kScaleGranularity / 2);
            PackFP4Group(group_data, hat_group, scale);
        }
    }

    // -------------------------------------------------------------------------
    // Quantize one row of kCols InputT values (already smooth) → MXFP4.
    //   Used by RunKSmoothAndQuantize after K' has been computed.
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void QuantizeRow(const InputT* __restrict__ src_row,
                                     uint8_t* __restrict__ dst_hat_row,
                                     uint8_t* __restrict__ dst_scale_row) const
    {
        constexpr float rcp_dst_max = 1.0f / 6.0f;
        const index_t   num_groups  = kCols / kScaleGranularity;

        for(index_t g = 0; g < num_groups; g++)
        {
            const index_t d_start = g * kScaleGranularity;
            float group_data[kScaleGranularity];
            float max_abs = 0.0f;

            for(index_t j = 0; j < kScaleGranularity; j++)
            {
                group_data[j] = static_cast<float>(src_row[d_start + j]);
                max_abs       = max(max_abs, abs(group_data[j]));
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
