// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/sageattn_preprocess/pipeline/sageattn_preprocess_pipeline.hpp"

namespace ck_tile {

// Host-side args for SageAttnPreprocessKernel.
//
// Grid: (num_tiles, nhead, batch)
//   num_tiles = max(num_q_tiles, num_k_tiles, hdim)
//
// Each CTA at tile_x processes (with bounds checking):
//   - Q tile[tile_x]:      Q smooth → delta_s → Q quantize
//   - K tile[tile_x]:      K smooth + quantize (inline)
//   - V channel[tile_x]:   V transpose + quantize
//
// Computation order within each CTA:
//   1. Q mean (smem)  2. delta_s (float)  3. Q quantize
//   4. K quantize     5. V quantize
struct SageAttnPreprocessHostArgs
{
    // --- Q: [batch, nhead, seqlen_q, hdim] InputT ---
    const void* q_ptr;         // InputT elements
    index_t     seqlen_q;
    index_t     hdim;          // == hdim_v (shared head dim)
    index_t     stride_q;      // = hdim
    index_t     nhead_stride_q;
    index_t     batch_stride_q;

    // Q hat: [batch, nhead, seqlen_q, hdim/2] uint8
    uint8_t* q_hat_ptr;
    index_t  stride_q_hat;
    index_t  nhead_stride_q_hat;
    index_t  batch_stride_q_hat;

    // Q scale: [batch, nhead, seqlen_q, hdim/32] uint8
    uint8_t* q_scale_ptr;
    index_t  stride_q_scale;
    index_t  nhead_stride_q_scale;
    index_t  batch_stride_q_scale;

    // Q mean: [batch, nhead, num_q_tiles, hdim] float
    float*   q_mean_ptr;
    index_t  q_tile_size;          // kM0
    index_t  stride_q_mean;        // = hdim
    index_t  nhead_stride_q_mean;  // = num_q_tiles * hdim
    index_t  batch_stride_q_mean;

    // --- K: [batch, nhead, seqlen_k, hdim] InputT ---
    const void* k_ptr;
    index_t     seqlen_k;
    index_t     stride_k;
    index_t     nhead_stride_k;
    index_t     batch_stride_k;

    // K hat: [batch, nhead, seqlen_k, hdim/2] uint8
    uint8_t* k_hat_ptr;
    index_t  stride_k_hat;
    index_t  nhead_stride_k_hat;
    index_t  batch_stride_k_hat;

    // K scale: [batch, nhead, seqlen_k, hdim/32] uint8
    uint8_t* k_scale_ptr;
    index_t  stride_k_scale;
    index_t  nhead_stride_k_scale;
    index_t  batch_stride_k_scale;

    // K mean: [batch, nhead, hdim] float — per-head mean over seqlen_k
    const float* k_mean_ptr;
    index_t      nhead_stride_k_mean; // = hdim
    index_t      batch_stride_k_mean; // = nhead * hdim

    // --- delta_s: [batch, nhead, num_q_tiles, seqlen_k] float ---
    float*   delta_s_ptr;
    index_t  stride_delta_s;       // = seqlen_k
    index_t  nhead_stride_delta_s; // = num_q_tiles * seqlen_k
    index_t  batch_stride_delta_s;

    // --- V: [batch, nhead, seqlen_k, hdim] InputT ---
    const void* v_ptr;
    index_t     nhead_stride_v;  // = seqlen_k * hdim
    index_t     batch_stride_v;

    // V hat: [batch, nhead, hdim, seqlen_k/2] uint8 (transposed)
    uint8_t* v_hat_ptr;
    index_t  stride_v_hat;       // = seqlen_k/2 (row stride in transposed layout)
    index_t  nhead_stride_v_hat;
    index_t  batch_stride_v_hat;

    // V scale: [batch, nhead, hdim, seqlen_k/32] uint8
    uint8_t* v_scale_ptr;
    index_t  stride_v_scale;
    index_t  nhead_stride_v_scale;
    index_t  batch_stride_v_scale;

    // --- Dimensions ---
    index_t batch;
    index_t nhead;
    index_t num_q_tiles; // = ceil(seqlen_q / q_tile_size)
    index_t num_k_tiles; // = ceil(seqlen_k / kRows)
    // num_tiles = max(num_q_tiles, num_k_tiles, hdim) — used for GridSize
};

// Device-side args (mirrors Hargs, typed pointers resolved by InputT)
template <typename InputT>
struct SageAttnPreprocessKargs
{
    const InputT* q_ptr;
    index_t       seqlen_q;
    index_t       hdim;
    index_t       stride_q;
    index_t       nhead_stride_q;
    index_t       batch_stride_q;

    uint8_t* q_hat_ptr;
    index_t  stride_q_hat;
    index_t  nhead_stride_q_hat;
    index_t  batch_stride_q_hat;

    uint8_t* q_scale_ptr;
    index_t  stride_q_scale;
    index_t  nhead_stride_q_scale;
    index_t  batch_stride_q_scale;

    float*   q_mean_ptr;
    index_t  q_tile_size;
    index_t  stride_q_mean;
    index_t  nhead_stride_q_mean;
    index_t  batch_stride_q_mean;

    const InputT* k_ptr;
    index_t       seqlen_k;
    index_t       stride_k;
    index_t       nhead_stride_k;
    index_t       batch_stride_k;

    uint8_t* k_hat_ptr;
    index_t  stride_k_hat;
    index_t  nhead_stride_k_hat;
    index_t  batch_stride_k_hat;

    uint8_t* k_scale_ptr;
    index_t  stride_k_scale;
    index_t  nhead_stride_k_scale;
    index_t  batch_stride_k_scale;

    const float* k_mean_ptr;
    index_t      nhead_stride_k_mean;
    index_t      batch_stride_k_mean;

    float*   delta_s_ptr;
    index_t  stride_delta_s;
    index_t  nhead_stride_delta_s;
    index_t  batch_stride_delta_s;

    const InputT* v_ptr;
    index_t       nhead_stride_v;
    index_t       batch_stride_v;

    uint8_t* v_hat_ptr;
    index_t  stride_v_hat;
    index_t  nhead_stride_v_hat;
    index_t  batch_stride_v_hat;

    uint8_t* v_scale_ptr;
    index_t  stride_v_scale;
    index_t  nhead_stride_v_scale;
    index_t  batch_stride_v_scale;

    index_t num_q_tiles;
    index_t num_k_tiles;
    // hdim is used as the V channel count (num_v_channels = hdim)
};

// SageAttnPreprocessKernel
//
// Template parameters:
//   InputT_:    input element type (float or fp16_t)
//   kRows_:     tile rows (kM0 == kN0)
//   kCols_:     hdim (tile columns)
//   kBlockSize_:threads per CTA
template <typename InputT_, index_t kRows_, index_t kCols_, index_t kBlockSize_ = 256>
struct SageAttnPreprocessKernel
{
    using InputT = InputT_;

    static constexpr index_t kRows      = kRows_;
    static constexpr index_t kCols      = kCols_;
    static constexpr index_t kBlockSize = kBlockSize_;

    using Pipeline =
        SageAttnPreprocessPipeline<InputT, kRows, kCols, /*kScaleGranularity=*/32, kBlockSize>;

    using Kargs = SageAttnPreprocessKargs<InputT>;
    using Hargs = SageAttnPreprocessHostArgs;

    CK_TILE_HOST static Kargs MakeKargs(const Hargs& h)
    {
        Kargs k{};
        k.q_ptr                = static_cast<const InputT*>(h.q_ptr);
        k.seqlen_q             = h.seqlen_q;
        k.hdim                 = h.hdim;
        k.stride_q             = h.stride_q;
        k.nhead_stride_q       = h.nhead_stride_q;
        k.batch_stride_q       = h.batch_stride_q;
        k.q_hat_ptr            = h.q_hat_ptr;
        k.stride_q_hat         = h.stride_q_hat;
        k.nhead_stride_q_hat   = h.nhead_stride_q_hat;
        k.batch_stride_q_hat   = h.batch_stride_q_hat;
        k.q_scale_ptr          = h.q_scale_ptr;
        k.stride_q_scale       = h.stride_q_scale;
        k.nhead_stride_q_scale = h.nhead_stride_q_scale;
        k.batch_stride_q_scale = h.batch_stride_q_scale;
        k.q_mean_ptr           = h.q_mean_ptr;
        k.q_tile_size          = h.q_tile_size;
        k.stride_q_mean        = h.stride_q_mean;
        k.nhead_stride_q_mean  = h.nhead_stride_q_mean;
        k.batch_stride_q_mean  = h.batch_stride_q_mean;

        k.k_ptr                = static_cast<const InputT*>(h.k_ptr);
        k.seqlen_k             = h.seqlen_k;
        k.stride_k             = h.stride_k;
        k.nhead_stride_k       = h.nhead_stride_k;
        k.batch_stride_k       = h.batch_stride_k;
        k.k_hat_ptr            = h.k_hat_ptr;
        k.stride_k_hat         = h.stride_k_hat;
        k.nhead_stride_k_hat   = h.nhead_stride_k_hat;
        k.batch_stride_k_hat   = h.batch_stride_k_hat;
        k.k_scale_ptr          = h.k_scale_ptr;
        k.stride_k_scale       = h.stride_k_scale;
        k.nhead_stride_k_scale = h.nhead_stride_k_scale;
        k.batch_stride_k_scale = h.batch_stride_k_scale;
        k.k_mean_ptr           = h.k_mean_ptr;
        k.nhead_stride_k_mean  = h.nhead_stride_k_mean;
        k.batch_stride_k_mean  = h.batch_stride_k_mean;

        k.delta_s_ptr          = h.delta_s_ptr;
        k.stride_delta_s       = h.stride_delta_s;
        k.nhead_stride_delta_s = h.nhead_stride_delta_s;
        k.batch_stride_delta_s = h.batch_stride_delta_s;

        k.v_ptr                = static_cast<const InputT*>(h.v_ptr);
        k.nhead_stride_v       = h.nhead_stride_v;
        k.batch_stride_v       = h.batch_stride_v;
        k.v_hat_ptr            = h.v_hat_ptr;
        k.stride_v_hat         = h.stride_v_hat;
        k.nhead_stride_v_hat   = h.nhead_stride_v_hat;
        k.batch_stride_v_hat   = h.batch_stride_v_hat;
        k.v_scale_ptr          = h.v_scale_ptr;
        k.stride_v_scale       = h.stride_v_scale;
        k.nhead_stride_v_scale = h.nhead_stride_v_scale;
        k.batch_stride_v_scale = h.batch_stride_v_scale;

        k.num_q_tiles = h.num_q_tiles;
        k.num_k_tiles = h.num_k_tiles;
        return k;
    }

    // Grid: (num_tiles, nhead, batch)
    // num_tiles = max(num_q_tiles, num_k_tiles, hdim)
    CK_TILE_HOST static dim3 GridSize(const Hargs& h)
    {
        const index_t num_tiles = max(max(h.num_q_tiles, h.num_k_tiles), h.hdim);
        return dim3(num_tiles, h.nhead, h.batch);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Pipeline::GetSmemSize();
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t tile_x    = get_block_id();
        const index_t head_idx  = blockIdx.y;
        const index_t batch_idx = blockIdx.z;

        const bool do_q = tile_x < kargs.num_q_tiles;
        const bool do_k = tile_x < kargs.num_k_tiles;
        const bool do_v = tile_x < kargs.hdim;

        __shared__ char smem[GetSmemSize()];

        Pipeline pipeline{};

        // ---- Step 1: Q mean ------------------------------------------------
        if(do_q)
        {
            const index_t row_start = tile_x * kargs.q_tile_size;
            if(row_start < kargs.seqlen_q)
            {
                const InputT* src_q =
                    kargs.q_ptr + batch_idx * kargs.batch_stride_q +
                    head_idx * kargs.nhead_stride_q + row_start * kargs.stride_q;

                float* q_mean =
                    kargs.q_mean_ptr + batch_idx * kargs.batch_stride_q_mean +
                    head_idx * kargs.nhead_stride_q_mean + tile_x * kargs.stride_q_mean;

                pipeline.RunQMean(src_q, q_mean, smem);
            }
        }
        block_sync_lds(); // smem[q_mean] visible to all steps below

        // ---- Step 2: delta_s -----------------------------------------------
        if(do_q)
        {
            const index_t row_start = tile_x * kargs.q_tile_size;
            if(row_start < kargs.seqlen_q)
            {
                const InputT* k_base =
                    kargs.k_ptr + batch_idx * kargs.batch_stride_k +
                    head_idx * kargs.nhead_stride_k;

                const float* k_mean =
                    kargs.k_mean_ptr + batch_idx * kargs.batch_stride_k_mean +
                    head_idx * kargs.nhead_stride_k_mean;

                float* delta_s =
                    kargs.delta_s_ptr + batch_idx * kargs.batch_stride_delta_s +
                    head_idx * kargs.nhead_stride_delta_s + tile_x * kargs.stride_delta_s;

                pipeline.RunDeltaS(k_base, kargs.seqlen_k, kargs.stride_k, k_mean, delta_s, smem);
            }
        }

        // ---- Step 3: Q quantize --------------------------------------------
        if(do_q)
        {
            const index_t row_start = tile_x * kargs.q_tile_size;
            if(row_start < kargs.seqlen_q)
            {
                const InputT* src_q =
                    kargs.q_ptr + batch_idx * kargs.batch_stride_q +
                    head_idx * kargs.nhead_stride_q + row_start * kargs.stride_q;

                uint8_t* dst_q_hat =
                    kargs.q_hat_ptr + batch_idx * kargs.batch_stride_q_hat +
                    head_idx * kargs.nhead_stride_q_hat + row_start * kargs.stride_q_hat;

                uint8_t* dst_q_scale =
                    kargs.q_scale_ptr + batch_idx * kargs.batch_stride_q_scale +
                    head_idx * kargs.nhead_stride_q_scale + row_start * kargs.stride_q_scale;

                pipeline.RunQQuantize(src_q, dst_q_hat, dst_q_scale, smem);
            }
        }

        // ---- Step 4: K quantize (K smooth inline) --------------------------
        if(do_k)
        {
            const index_t row_start = tile_x * kRows;
            if(row_start < kargs.seqlen_k)
            {
                const InputT* src_k =
                    kargs.k_ptr + batch_idx * kargs.batch_stride_k +
                    head_idx * kargs.nhead_stride_k + row_start * kargs.stride_k;

                const float* k_mean =
                    kargs.k_mean_ptr + batch_idx * kargs.batch_stride_k_mean +
                    head_idx * kargs.nhead_stride_k_mean;

                uint8_t* dst_k_hat =
                    kargs.k_hat_ptr + batch_idx * kargs.batch_stride_k_hat +
                    head_idx * kargs.nhead_stride_k_hat + row_start * kargs.stride_k_hat;

                uint8_t* dst_k_scale =
                    kargs.k_scale_ptr + batch_idx * kargs.batch_stride_k_scale +
                    head_idx * kargs.nhead_stride_k_scale + row_start * kargs.stride_k_scale;

                pipeline.RunKQuantize(src_k, k_mean, dst_k_hat, dst_k_scale);
            }
        }

        // ---- Step 5: V transpose + quantize --------------------------------
        if(do_v)
        {
            const index_t hdim_ch = tile_x;

            const InputT* src_v =
                kargs.v_ptr + batch_idx * kargs.batch_stride_v +
                head_idx * kargs.nhead_stride_v + hdim_ch;

            uint8_t* dst_v_hat =
                kargs.v_hat_ptr + batch_idx * kargs.batch_stride_v_hat +
                head_idx * kargs.nhead_stride_v_hat + hdim_ch * kargs.stride_v_hat;

            uint8_t* dst_v_scale =
                kargs.v_scale_ptr + batch_idx * kargs.batch_stride_v_scale +
                head_idx * kargs.nhead_stride_v_scale + hdim_ch * kargs.stride_v_scale;

            pipeline.RunV(src_v, kargs.seqlen_k, kargs.hdim, dst_v_hat, dst_v_scale);
        }
    }
};

} // namespace ck_tile
