// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// Preprocessing mode enum for the unified SageAttention V3 preprocessing kernel.
enum class SageAttnPreprocessMode
{
    Q = 0, // Per-Q-block channel-mean subtraction + MXFP4 quantization + store q_mean
    K = 1, // Global channel-mean subtraction + MXFP4 quantization
    V = 2, // Transpose [seqlen_k, hdim_v] -> [hdim_v, seqlen_k] + MXFP4 quantization
};

// SageAttnPreprocessPipeline: thread-parallel preprocessing for one (B, H, tile) block.
//
// Grid mapping (set by the kernel layer):
//   mode=Q: gridDim = (num_q_tiles, nhead, batch), blockDim = kBlockSize
//           Each CTA processes one Q tile of kM0 rows, all hdim cols.
//   mode=K: gridDim = (num_k_tiles, nhead, batch), blockDim = kBlockSize
//           Each CTA processes one K tile of kN0 rows, all hdim cols.
//   mode=V: gridDim = (hdim_v, nhead, batch), blockDim = kBlockSize
//           Each CTA processes one row of V_T (one hdim_v channel across all seqlen_k).
//           Input tile: [1, seqlen_k] floats -> output: [1, seqlen_k/2] pk_fp4_t + [1, seqlen_k/32] e8m0_t.
//           But V transpose means we actually need to process [kN0, 1] input cols for each hdim row.
//           Reinterpret: gridDim = (hdim_v, nhead, batch), each CTA handles one hdim_v row,
//           looping over seqlen_k in chunks of kN0*32 (MXFP4 group-of-32 along seqlen_k).
//
// This pipeline uses raw pointer arithmetic and scalar/vector loads for maximum simplicity.
// MXFP4 quantization is done per group of 32 elements along the innermost quantization axis:
//   - Q/K: innermost axis is hdim (each row quantized in groups of 32 hdim elements)
//   - V:   innermost axis is seqlen_k (each hdim_v row quantized in groups of 32 seqlen_k elems)
//
// Template parameters:
//   kRows:             number of rows per tile (kM0 for Q, kN0 for K; 1 for V)
//   kCols:             number of cols per tile (hdim for Q/K; seqlen_k for V mode)
//   kScaleGranularity: MXFP4 group size, must be 32
//   kBlockSize:        threads per CTA

template <index_t kRows_,
          index_t kCols_,
          index_t kScaleGranularity_ = 32,
          index_t kBlockSize_        = 256>
struct SageAttnPreprocessPipeline
{
    static constexpr index_t kRows             = kRows_;
    static constexpr index_t kCols             = kCols_;
    static constexpr index_t kScaleGranularity = kScaleGranularity_;
    static constexpr index_t kBlockSize        = kBlockSize_;

    static_assert(kScaleGranularity == 32, "MXFP4 scale granularity must be 32");
    static_assert(kCols % kScaleGranularity == 0,
                  "kCols must be divisible by kScaleGranularity for MXFP4");

    // Total elements in this tile: kRows * kCols
    // Shared memory: kCols floats for broadcasting column mean (Q/K modes)
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kCols * sizeof(float);
    }

    // -------------------------------------------------------------------------
    // Q mode: compute column mean of [kRows, kCols] src tile, subtract, quantize.
    //
    // Parameters:
    //   src_ptr:         float32 pointer to [kRows, kCols] tile (row-major, stride=kCols)
    //   dst_hat_ptr:     pk_fp4_t pointer to [kRows, kCols/2] output (byte-packed fp4)
    //   dst_scale_ptr:   e8m0_t pointer to [kRows, kCols/32] output
    //   q_mean_ptr:      float32 pointer to [kCols] output (this block's column mean)
    //   smem:            shared memory (kCols floats)
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunQ(const float* __restrict__ src_ptr,
                             uint8_t* __restrict__ dst_hat_ptr,
                             uint8_t* __restrict__ dst_scale_ptr,
                             float* __restrict__ q_mean_ptr,
                             void* smem) const
    {
        float* smem_col_mean = reinterpret_cast<float*>(smem);
        const index_t tid    = get_thread_id();

        // Step 1: Zero shared column accumulators
        for(index_t d = tid; d < kCols; d += kBlockSize)
            smem_col_mean[d] = 0.0f;
        block_sync_lds();

        // Step 2: Each thread sums its rows into smem (atomic float add)
        // Thread tid owns rows [tid*kRows/kBlockSize .. (tid+1)*kRows/kBlockSize) (if evenly split)
        // Simpler: each thread iterates over all kRows for its assigned cols
        // Thread tid handles cols: tid, tid+kBlockSize, tid+2*kBlockSize, ...
        for(index_t d = tid; d < kCols; d += kBlockSize)
        {
            float sum = 0.0f;
            for(index_t r = 0; r < kRows; r++)
                sum += src_ptr[r * kCols + d];
            smem_col_mean[d] = sum / static_cast<float>(kRows);
        }
        block_sync_lds();

        // Step 3: Store column mean to output
        for(index_t d = tid; d < kCols; d += kBlockSize)
            q_mean_ptr[d] = smem_col_mean[d];

        // Step 4: For each row, subtract mean and MXFP4-quantize
        // Each thread handles one or more complete rows
        // We partition rows across threads: thread tid handles rows where (row % kBlockSize == tid)
        // This is less efficient but correct for any kRows and kBlockSize.
        // For performance, kRows should be small (e.g., 64 or 128) and kBlockSize should be chosen
        // so each thread handles exactly kRows/kBlockSize rows.
        for(index_t r = tid; r < kRows; r += kBlockSize)
        {
            QuantizeRowMXFP4(src_ptr + r * kCols,
                             smem_col_mean,
                             dst_hat_ptr + r * (kCols / 2),
                             dst_scale_ptr + r * (kCols / kScaleGranularity),
                             kCols);
        }
    }

    // -------------------------------------------------------------------------
    // K mode: subtract provided k_mean, then MXFP4-quantize.
    //
    // Parameters:
    //   src_ptr:         float32 [kRows, kCols] tile
    //   dst_hat_ptr:     pk_fp4_t [kRows, kCols/2] output
    //   dst_scale_ptr:   e8m0_t [kRows, kCols/32] output
    //   k_mean_ptr:      float32 [kCols] global channel mean
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunK(const float* __restrict__ src_ptr,
                             uint8_t* __restrict__ dst_hat_ptr,
                             uint8_t* __restrict__ dst_scale_ptr,
                             const float* __restrict__ k_mean_ptr) const
    {
        const index_t tid = get_thread_id();
        for(index_t r = tid; r < kRows; r += kBlockSize)
        {
            QuantizeRowMXFP4(src_ptr + r * kCols,
                             k_mean_ptr,
                             dst_hat_ptr + r * (kCols / 2),
                             dst_scale_ptr + r * (kCols / kScaleGranularity),
                             kCols);
        }
    }

    // -------------------------------------------------------------------------
    // V mode: process one hdim_v channel (row index = blockIdx.x in the grid).
    // Input is column-major [hdim_v, seqlen_k] logically, but stored as row-major
    // [seqlen_k, hdim_v] in memory.
    //
    // This CTA handles ONE hdim_v row of the transposed V tensor, i.e., all seqlen_k
    // elements V[:, hdim_channel]. Quantize in groups of 32 along seqlen_k.
    //
    // Parameters:
    //   src_ptr:        float32 pointer to V[0, hdim_channel] in [seqlen_k, hdim_v] layout
    //   seqlen_k:       number of tokens
    //   hdim_v:         V head dim
    //   dst_hat_ptr:    pk_fp4_t pointer to V_hat[hdim_channel, 0] in [hdim_v, seqlen_k/2] layout
    //   dst_scale_ptr:  e8m0_t pointer to V_scale[hdim_channel, 0] in [hdim_v, seqlen_k/32] layout
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunV(const float* __restrict__ src_ptr,
                             index_t seqlen_k,
                             index_t hdim_v,
                             uint8_t* __restrict__ dst_hat_ptr,
                             uint8_t* __restrict__ dst_scale_ptr) const
    {
        const index_t tid = get_thread_id();

        // Each thread processes groups of kScaleGranularity seqlen_k elements.
        // Total groups = seqlen_k / 32; distribute groups across threads.
        const index_t num_groups = seqlen_k / kScaleGranularity;
        constexpr float rcp_dst_max = 1.0f / 6.0f;

        for(index_t g = tid; g < num_groups; g += kBlockSize)
        {
            const index_t k_start = g * kScaleGranularity;

            // Load 32 consecutive seqlen_k elements for this hdim_v channel
            float group_data[kScaleGranularity];
            float max_abs = 0.0f;
            for(index_t j = 0; j < kScaleGranularity; j++)
            {
                // V is stored as [seqlen_k, hdim_v] row-major; accessing column hdim_channel
                // src_ptr already points to V[0, hdim_channel], stride hdim_v between rows
                group_data[j] = src_ptr[(k_start + j) * hdim_v];
                max_abs       = max(max_abs, abs(group_data[j]));
            }

            // e8m0 scale: exp2(ceil(log2(max_abs/6)))
            const float scale = bit_cast<float>(
                (bit_cast<uint32_t>(max_abs * rcp_dst_max) + numeric_traits<float>::mant_mask) &
                numeric_traits<float>::head_mask);

            // Store e8m0 scale byte
            dst_scale_ptr[g] = bit_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);

            // Pack 32 floats -> 16 bytes (pairs of FP4 into bytes) of pk_fp4_t
            uint8_t* hat_group = dst_hat_ptr + g * (kScaleGranularity / 2);
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
                // Write 4 bytes (8 FP4 values) to dst_hat
                *reinterpret_cast<uint32_t*>(hat_group + j * 4) = x;
            }
        }
    }

private:
    // Quantize one row of `ncols` float32 values using per-group-of-32 MXFP4.
    // col_mean[d] is subtracted before quantization (pass nullptr for no subtraction).
    // Output:
    //   dst_hat_row:   ncols/2 bytes (packed FP4 pairs)
    //   dst_scale_row: ncols/32 e8m0 scale bytes
    CK_TILE_DEVICE void QuantizeRowMXFP4(const float* __restrict__ src_row,
                                          const float* __restrict__ col_mean,
                                          uint8_t* __restrict__ dst_hat_row,
                                          uint8_t* __restrict__ dst_scale_row,
                                          index_t ncols) const
    {
        constexpr float rcp_dst_max = 1.0f / 6.0f;

        const index_t num_groups = ncols / kScaleGranularity;
        for(index_t g = 0; g < num_groups; g++)
        {
            const index_t d_start = g * kScaleGranularity;
            float group_data[kScaleGranularity];
            float max_abs = 0.0f;

            for(index_t j = 0; j < kScaleGranularity; j++)
            {
                const float val = src_row[d_start + j] - col_mean[d_start + j];
                group_data[j]   = val;
                max_abs         = max(max_abs, abs(val));
            }

            const float scale = bit_cast<float>(
                (bit_cast<uint32_t>(max_abs * rcp_dst_max) + numeric_traits<float>::mant_mask) &
                numeric_traits<float>::head_mask);

            // Store e8m0: upper 8 bits of the float scale exponent+sign
            dst_scale_row[g] = static_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);

            uint8_t* hat_group = dst_hat_row + g * (kScaleGranularity / 2);
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
    }
};

} // namespace ck_tile
