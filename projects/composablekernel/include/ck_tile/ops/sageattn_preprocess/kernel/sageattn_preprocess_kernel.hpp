// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/sageattn_preprocess/pipeline/sageattn_preprocess_pipeline.hpp"

namespace ck_tile {

// Host-side args for SageAttnPreprocessKernel
struct SageAttnPreprocessHostArgs
{
    SageAttnPreprocessMode mode;

    // Input tensor: float32
    //   mode=Q,K: [batch, nhead, seqlen, hdim] (i_perm=true layout)
    //   mode=V:   [batch, nhead, seqlen_k, hdim_v]
    const float* src_ptr;
    index_t      batch;
    index_t      nhead;
    index_t      seqlen;       // seqlen_q for Q, seqlen_k for K/V
    index_t      hdim;         // head dim (hdim for Q/K, hdim_v for V)

    index_t stride_src;        // row stride of src (= hdim for Q/K/V)
    index_t nhead_stride_src;  // stride between heads (= seqlen * hdim)
    index_t batch_stride_src;  // stride between batches (= nhead * seqlen * hdim)

    // Output hat tensor: pk_fp4_t (stored as uint8_t, 2 FP4 per byte)
    //   mode=Q,K: [batch, nhead, seqlen, hdim/2]
    //   mode=V:   [batch, nhead, hdim_v, seqlen/2]
    uint8_t* dst_hat_ptr;
    index_t  stride_dst_hat;        // innermost stride of dst_hat (in bytes)
    index_t  nhead_stride_dst_hat;
    index_t  batch_stride_dst_hat;

    // Output scale tensor: e8m0_t (stored as uint8_t)
    //   mode=Q,K: [batch, nhead, seqlen, hdim/32]
    //   mode=V:   [batch, nhead, hdim_v, seqlen/32]
    uint8_t* dst_scale_ptr;
    index_t  stride_dst_scale;
    index_t  nhead_stride_dst_scale;
    index_t  batch_stride_dst_scale;

    // mode=Q only: output q_mean [batch, nhead, num_q_blocks, hdim] float32
    float*   q_mean_ptr;
    index_t  q_block_size;         // = kM0 (Q tile size, must match FMHA pipeline)
    index_t  stride_q_mean;        // hdim (innermost)
    index_t  nhead_stride_q_mean;
    index_t  batch_stride_q_mean;
    index_t  q_block_stride_q_mean; // stride between q_blocks

    // mode=K only: input k_mean [hdim] float32 (global channel mean)
    const float* k_mean_ptr;
};

// Kernel argument struct (device-side)
struct SageAttnPreprocessKargs
{
    SageAttnPreprocessMode mode;

    const float* src_ptr;
    index_t      seqlen;
    index_t      hdim;
    index_t      stride_src;
    index_t      nhead_stride_src;
    index_t      batch_stride_src;

    uint8_t* dst_hat_ptr;
    index_t  stride_dst_hat;
    index_t  nhead_stride_dst_hat;
    index_t  batch_stride_dst_hat;

    uint8_t* dst_scale_ptr;
    index_t  stride_dst_scale;
    index_t  nhead_stride_dst_scale;
    index_t  batch_stride_dst_scale;

    // Q-mode specific
    float*   q_mean_ptr;
    index_t  q_block_size;
    index_t  stride_q_mean;
    index_t  nhead_stride_q_mean;
    index_t  batch_stride_q_mean;
    index_t  q_block_stride_q_mean;

    // K-mode specific
    const float* k_mean_ptr;
};

// SageAttnPreprocessKernel: unified Q/K/V preprocessing for SageAttention V3.
//
// Grid:
//   mode=Q: (num_q_tiles, nhead, batch)  — one CTA per [kM0, hdim] Q tile
//   mode=K: (num_k_tiles, nhead, batch)  — one CTA per [kN0, hdim] K tile
//   mode=V: (hdim_v, nhead, batch)       — one CTA per hdim_v channel row
//
// Template parameter kRows is the tile size along seqlen (kM0 for Q, kN0 for K).
// For V mode kRows is unused (single hdim_v channel per CTA, iterate over seqlen_k).

template <index_t kRows_, index_t kCols_, index_t kBlockSize_ = 256>
struct SageAttnPreprocessKernel
{
    static constexpr index_t kRows      = kRows_;
    static constexpr index_t kCols      = kCols_;
    static constexpr index_t kBlockSize = kBlockSize_;

    using Pipeline =
        SageAttnPreprocessPipeline<kRows, kCols, /*kScaleGranularity=*/32, kBlockSize>;

    using Kargs = SageAttnPreprocessKargs;
    using Hargs = SageAttnPreprocessHostArgs;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const Hargs& h)
    {
        Kargs k{};
        k.mode                  = h.mode;
        k.src_ptr               = h.src_ptr;
        k.seqlen                = h.seqlen;
        k.hdim                  = h.hdim;
        k.stride_src            = h.stride_src;
        k.nhead_stride_src      = h.nhead_stride_src;
        k.batch_stride_src      = h.batch_stride_src;
        k.dst_hat_ptr           = h.dst_hat_ptr;
        k.stride_dst_hat        = h.stride_dst_hat;
        k.nhead_stride_dst_hat  = h.nhead_stride_dst_hat;
        k.batch_stride_dst_hat  = h.batch_stride_dst_hat;
        k.dst_scale_ptr         = h.dst_scale_ptr;
        k.stride_dst_scale      = h.stride_dst_scale;
        k.nhead_stride_dst_scale = h.nhead_stride_dst_scale;
        k.batch_stride_dst_scale = h.batch_stride_dst_scale;
        k.q_mean_ptr            = h.q_mean_ptr;
        k.q_block_size          = h.q_block_size;
        k.stride_q_mean         = h.stride_q_mean;
        k.nhead_stride_q_mean   = h.nhead_stride_q_mean;
        k.batch_stride_q_mean   = h.batch_stride_q_mean;
        k.q_block_stride_q_mean = h.q_block_stride_q_mean;
        k.k_mean_ptr            = h.k_mean_ptr;
        return k;
    }

    // Grid size helpers for each mode
    CK_TILE_HOST static dim3 GridSizeQ(const Hargs& h)
    {
        return dim3(integer_divide_ceil(h.seqlen, h.q_block_size), h.nhead, h.batch);
    }
    CK_TILE_HOST static dim3 GridSizeK(const Hargs& h)
    {
        return dim3(integer_divide_ceil(h.seqlen, kRows), h.nhead, h.batch);
    }
    CK_TILE_HOST static dim3 GridSizeV(const Hargs& h)
    {
        return dim3(h.hdim, h.nhead, h.batch);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Pipeline::GetSmemSize();
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t tile_idx  = get_block_id();   // blockIdx.x
        const index_t head_idx  = blockIdx.y;
        const index_t batch_idx = blockIdx.z;

        __shared__ char smem[GetSmemSize()];

        Pipeline pipeline{};

        if(kargs.mode == SageAttnPreprocessMode::Q)
        {
            // Q tile: [kM0, hdim], starting at seqlen row = tile_idx * kM0
            const index_t row_start = tile_idx * kargs.q_block_size;
            if(row_start >= kargs.seqlen)
                return;

            const float* src = kargs.src_ptr + batch_idx * kargs.batch_stride_src +
                               head_idx * kargs.nhead_stride_src + row_start * kargs.stride_src;

            uint8_t* dst_hat = kargs.dst_hat_ptr + batch_idx * kargs.batch_stride_dst_hat +
                               head_idx * kargs.nhead_stride_dst_hat +
                               row_start * kargs.stride_dst_hat;

            uint8_t* dst_scale = kargs.dst_scale_ptr + batch_idx * kargs.batch_stride_dst_scale +
                                 head_idx * kargs.nhead_stride_dst_scale +
                                 row_start * kargs.stride_dst_scale;

            float* q_mean = kargs.q_mean_ptr + batch_idx * kargs.batch_stride_q_mean +
                            head_idx * kargs.nhead_stride_q_mean +
                            tile_idx * kargs.q_block_stride_q_mean;

            pipeline.RunQ(src, dst_hat, dst_scale, q_mean, smem);
        }
        else if(kargs.mode == SageAttnPreprocessMode::K)
        {
            const index_t row_start = tile_idx * kRows;
            if(row_start >= kargs.seqlen)
                return;

            const float* src = kargs.src_ptr + batch_idx * kargs.batch_stride_src +
                               head_idx * kargs.nhead_stride_src + row_start * kargs.stride_src;

            uint8_t* dst_hat = kargs.dst_hat_ptr + batch_idx * kargs.batch_stride_dst_hat +
                               head_idx * kargs.nhead_stride_dst_hat +
                               row_start * kargs.stride_dst_hat;

            uint8_t* dst_scale = kargs.dst_scale_ptr + batch_idx * kargs.batch_stride_dst_scale +
                                 head_idx * kargs.nhead_stride_dst_scale +
                                 row_start * kargs.stride_dst_scale;

            pipeline.RunK(src, dst_hat, dst_scale, kargs.k_mean_ptr);
        }
        else // mode == V
        {
            // tile_idx = hdim_v channel index
            const index_t hdim_ch = tile_idx;
            if(hdim_ch >= kargs.hdim)
                return;

            // V source: [seqlen_k, hdim_v] row-major.
            // Access V[:, hdim_ch]: stride hdim_v between consecutive seqlen_k rows.
            const float* src = kargs.src_ptr + batch_idx * kargs.batch_stride_src +
                               head_idx * kargs.nhead_stride_src + hdim_ch;

            // V_hat output: [hdim_v, seqlen_k/2] row-major.
            // Row hdim_ch, stride_dst_hat = seqlen_k/2 bytes.
            uint8_t* dst_hat = kargs.dst_hat_ptr + batch_idx * kargs.batch_stride_dst_hat +
                               head_idx * kargs.nhead_stride_dst_hat +
                               hdim_ch * kargs.stride_dst_hat;

            // V_scale output: [hdim_v, seqlen_k/32] row-major.
            uint8_t* dst_scale = kargs.dst_scale_ptr + batch_idx * kargs.batch_stride_dst_scale +
                                 head_idx * kargs.nhead_stride_dst_scale +
                                 hdim_ch * kargs.stride_dst_scale;

            pipeline.RunV(src, kargs.seqlen, kargs.hdim, dst_hat, dst_scale);
        }
    }
};

} // namespace ck_tile
