// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/sageattn_preprocess/pipeline/sageattn_preprocess_pipeline.hpp"

namespace ck_tile {

// ============================================================================
// SageAttnPreprocessKernel
//
// Grid: (num_tiles, nhead, batch)
//   num_tiles = max(num_q_tiles, num_k_tiles, hdim)
//
// Each CTA at tile_x processes (with bounds checking):
//   - Q tile[tile_x]:      Q mean → smem → Q quantize
//   - K tile[tile_x]:      K' = K - k_mean → k_prime; K' quantize
//   - V channel[tile_x]:   V transpose + quantize
//
// delta_s is produced by a separate batched GEMM (q_mean @ K'^T).
// k_mean must already be available (written by SageAttnKMeanKernel).
// ============================================================================

struct SageAttnPreprocessHostArgs
{
    // --- Q: [batch, nhead, seqlen_q, hdim] InputT ---
    const void* q_ptr;
    index_t     seqlen_q;
    index_t     hdim;
    index_t     stride_q;
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

    // Q mean: [batch, nhead, num_q_tiles, hdim] InputT
    void*   q_mean_ptr;           // InputT*
    index_t q_tile_size;          // kM0
    index_t stride_q_mean;        // = hdim
    index_t nhead_stride_q_mean;  // = num_q_tiles * hdim
    index_t batch_stride_q_mean;

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

    // K mean: [batch, nhead, hdim] InputT — produced by SageAttnKMeanKernel
    const void* k_mean_ptr;       // InputT*
    index_t     nhead_stride_k_mean;
    index_t     batch_stride_k_mean;

    // K prime: [batch, nhead, seqlen_k, hdim] InputT row-major — K' = K - k_mean
    void*   k_prime_ptr;
    index_t stride_k_prime;
    index_t nhead_stride_k_prime;
    index_t batch_stride_k_prime;

    // --- V: [batch, nhead, seqlen_k, hdim] InputT ---
    const void* v_ptr;
    index_t     nhead_stride_v;
    index_t     batch_stride_v;

    // V hat: [batch, nhead, hdim, seqlen_k/2] uint8 (transposed)
    uint8_t* v_hat_ptr;
    index_t  stride_v_hat;
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
    index_t num_q_tiles;
    index_t num_k_tiles;
};

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

    InputT* q_mean_ptr;
    index_t q_tile_size;
    index_t stride_q_mean;
    index_t nhead_stride_q_mean;
    index_t batch_stride_q_mean;

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

    const InputT* k_mean_ptr;
    index_t       nhead_stride_k_mean;
    index_t       batch_stride_k_mean;

    InputT* k_prime_ptr;
    index_t stride_k_prime;
    index_t nhead_stride_k_prime;
    index_t batch_stride_k_prime;

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
};

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
        k.q_mean_ptr           = static_cast<InputT*>(h.q_mean_ptr);
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
        k.k_mean_ptr           = static_cast<const InputT*>(h.k_mean_ptr);
        k.nhead_stride_k_mean  = h.nhead_stride_k_mean;
        k.batch_stride_k_mean  = h.batch_stride_k_mean;
        k.k_prime_ptr          = static_cast<InputT*>(h.k_prime_ptr);
        k.stride_k_prime       = h.stride_k_prime;
        k.nhead_stride_k_prime = h.nhead_stride_k_prime;
        k.batch_stride_k_prime = h.batch_stride_k_prime;

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
    //   num_tiles = max(num_q_tiles, num_k_tiles); V is handled by SageAttnVPreprocessKernel.
    CK_TILE_HOST static dim3 GridSize(const Hargs& h)
    {
        const index_t num_tiles = max(h.num_q_tiles, h.num_k_tiles);
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

                InputT* q_mean =
                    kargs.q_mean_ptr + batch_idx * kargs.batch_stride_q_mean +
                    head_idx * kargs.nhead_stride_q_mean + tile_x * kargs.stride_q_mean;

                pipeline.RunQMean(src_q, q_mean, smem);
            }
        }
        block_sync_lds(); // smem q_mean visible for step 2

        // ---- Step 2: Q quantize --------------------------------------------
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

        // ---- Step 3: K smooth + quantize -----------------------------------
        if(do_k)
        {
            const index_t row_start = tile_x * kRows;
            if(row_start < kargs.seqlen_k)
            {
                const InputT* src_k =
                    kargs.k_ptr + batch_idx * kargs.batch_stride_k +
                    head_idx * kargs.nhead_stride_k + row_start * kargs.stride_k;

                const InputT* k_mean =
                    kargs.k_mean_ptr + batch_idx * kargs.batch_stride_k_mean +
                    head_idx * kargs.nhead_stride_k_mean;

                // k_prime stored in natural layout: [batch, nhead, seqlen_k, hdim] row-major.
                InputT* k_prime =
                    kargs.k_prime_ptr + batch_idx * kargs.batch_stride_k_prime +
                    head_idx * kargs.nhead_stride_k_prime + row_start * kargs.stride_k_prime;

                uint8_t* dst_k_hat =
                    kargs.k_hat_ptr + batch_idx * kargs.batch_stride_k_hat +
                    head_idx * kargs.nhead_stride_k_hat + row_start * kargs.stride_k_hat;

                uint8_t* dst_k_scale =
                    kargs.k_scale_ptr + batch_idx * kargs.batch_stride_k_scale +
                    head_idx * kargs.nhead_stride_k_scale + row_start * kargs.stride_k_scale;

                pipeline.RunKSmoothAndQuantize(
                    src_k, k_mean, k_prime, kargs.stride_k_prime, dst_k_hat, dst_k_scale);
            }
        }

    }
};

// ============================================================================
// SageAttnKMeanKernel
//
// Computes k_mean[batch, nhead, hdim] = mean over seqlen_k of K[batch, nhead, :, hdim].
//
// Grid:      (num_k_tiles, nhead, batch)
// BlockSize: kCols  (one thread per d-channel)
//
// Algorithm: each CTA atomicAdds its partial column-sum into k_mean_partial (float).
// The last CTA to finish (detected via completion counter) normalises and casts to InputT.
//
// Caller must zero-initialise k_mean_partial_ptr and counter_ptr before launch.
// ============================================================================

struct SageAttnKMeanHostArgs
{
    const void* k_ptr;
    index_t     seqlen_k;
    index_t     hdim;
    index_t     stride_k;          // = hdim
    index_t     nhead_stride_k;
    index_t     batch_stride_k;

    float*   k_mean_partial_ptr;   // [batch, nhead, hdim] float, pre-zeroed
    void*    k_mean_ptr;           // [batch, nhead, hdim] InputT, output
    int32_t* counter_ptr;          // [batch, nhead] int32, pre-zeroed

    index_t nhead_stride_kmean;    // = hdim
    index_t batch_stride_kmean;    // = nhead * hdim

    index_t num_k_tiles;
    index_t nhead;
    index_t batch;
};

template <typename InputT>
struct SageAttnKMeanKargs
{
    const InputT* k_ptr;
    index_t       seqlen_k;
    index_t       hdim;
    index_t       stride_k;
    index_t       nhead_stride_k;
    index_t       batch_stride_k;

    float*        k_mean_partial_ptr;
    InputT*       k_mean_ptr;
    int32_t*      counter_ptr;

    index_t nhead_stride_kmean;
    index_t batch_stride_kmean;

    index_t num_k_tiles;
    index_t nhead;
};

template <typename InputT_, index_t kRows_, index_t kCols_>
struct SageAttnKMeanKernel
{
    using InputT = InputT_;

    static constexpr index_t kRows      = kRows_;
    static constexpr index_t kCols      = kCols_;
    static constexpr index_t kBlockSize = kCols; // one thread per d-channel

    // Pipeline reused only for RunKMeanPartial.
    using Pipeline = SageAttnPreprocessPipeline<InputT, kRows, kCols>;

    using Kargs = SageAttnKMeanKargs<InputT>;
    using Hargs = SageAttnKMeanHostArgs;

    CK_TILE_HOST static Kargs MakeKargs(const Hargs& h)
    {
        Kargs k{};
        k.k_ptr                = static_cast<const InputT*>(h.k_ptr);
        k.seqlen_k             = h.seqlen_k;
        k.hdim                 = h.hdim;
        k.stride_k             = h.stride_k;
        k.nhead_stride_k       = h.nhead_stride_k;
        k.batch_stride_k       = h.batch_stride_k;
        k.k_mean_partial_ptr   = h.k_mean_partial_ptr;
        k.k_mean_ptr           = static_cast<InputT*>(h.k_mean_ptr);
        k.counter_ptr          = h.counter_ptr;
        k.nhead_stride_kmean   = h.nhead_stride_kmean;
        k.batch_stride_kmean   = h.batch_stride_kmean;
        k.num_k_tiles          = h.num_k_tiles;
        k.nhead                = h.nhead;
        return k;
    }

    CK_TILE_HOST static dim3 GridSize(const Hargs& h)
    {
        return dim3(h.num_k_tiles, h.nhead, h.batch);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    // Smem: one int32 flag to broadcast "is_last" to all threads.
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return sizeof(int32_t);
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t tile_x    = get_block_id();
        const index_t head_idx  = blockIdx.y;
        const index_t batch_idx = blockIdx.z;

        // Tile row range (handles tail).
        const index_t row_start = tile_x * kRows;
        const index_t row_end   = min(row_start + kRows, kargs.seqlen_k);
        const index_t n_rows    = row_end - row_start;

        const InputT* k_tile =
            kargs.k_ptr + batch_idx * kargs.batch_stride_k +
            head_idx * kargs.nhead_stride_k + row_start * kargs.stride_k;

        float* partial =
            kargs.k_mean_partial_ptr + batch_idx * kargs.batch_stride_kmean +
            head_idx * kargs.nhead_stride_kmean;

        // --- Step 1: accumulate partial column-sum via atomicAdd ---
        Pipeline pipeline{};
        pipeline.RunKMeanPartial(k_tile, n_rows, kargs.stride_k, partial);

        // Ensure our atomicAdds are globally visible before incrementing counter.
        __threadfence();

        // --- Step 2: completion counter (thread 0 only) ---
        __shared__ int32_t smem_is_last;
        if(get_thread_id() == 0)
        {
            int32_t* cnt  = kargs.counter_ptr + batch_idx * kargs.nhead + head_idx;
            int32_t  prev = atomicAdd(cnt, 1);
            smem_is_last  = (prev == kargs.num_k_tiles - 1) ? 1 : 0;
        }
        block_sync_lds();

        // --- Step 3: last CTA normalises and stores k_mean as InputT ---
        if(smem_is_last)
        {
            const index_t d = get_thread_id();
            InputT* k_mean  =
                kargs.k_mean_ptr + batch_idx * kargs.batch_stride_kmean +
                head_idx * kargs.nhead_stride_kmean;

            const float mean_f =
                partial[d] / static_cast<float>(kargs.seqlen_k);
            k_mean[d] = static_cast<InputT>(mean_f);
        }
    }
};

// ============================================================================
// SageAttnVPreprocessKernel
//
// Quantises V (transposed layout) using an LDS-based 2-D tile transpose that
// avoids the non-coalesced column reads of the old per-channel approach.
//
// Grid:      (seqlen_k / kVGroup, hdim / kVHdimTile, batch * nhead)
// BlockSize: kVGroup  (one thread per seqlen row in the group)
//
// Algorithm per CTA (g_idx, d_idx, bh_idx):
//   1. Each thread tid loads one row of V:
//        V[g_idx*kVGroup + tid, d_idx*kVHdimTile .. +kVHdimTile]
//      into LDS as float32 → coalesced HBM reads.
//   2. After __syncthreads, thread tid owns channel d_local=tid from LDS and
//      reads the 32-row column smem[0..kVGroup-1][d_local] → no HBM access.
//   3. Computes MXFP4 scale and packs the group, then writes to V_hat /
//      V_scale in transposed layout [hdim, seqlen_k/2|seqlen_k/32].
//
// LDS bank conflict analysis (gfx950: 64 banks × 4 B each):
//   smem[kVGroup][kVHdimTile + kLDSPad] as float32, kLDSPad = 1.
//   bank(j, d) = (j*(kVHdimTile+1) + d) % 64.
//   gcd(kVHdimTile+1, 64) = gcd(33, 64) = 1 → all 32 rows in a column
//   access 32 distinct banks → zero bank conflicts for both row and column
//   access patterns.
//
// Caller requirements:
//   seqlen_k % kVGroup    == 0
//   hdim     % kVHdimTile == 0
// ============================================================================

struct SageAttnVPreprocessHostArgs
{
    // V source: [batch, nhead, seqlen_k, hdim] InputT row-major
    const void* v_ptr;
    index_t     seqlen_k;
    index_t     hdim;              // also the row stride of V
    index_t     nhead_stride_v;    // = seqlen_k * hdim
    index_t     batch_stride_v;    // = nhead * seqlen_k * hdim

    // V hat: [batch, nhead, hdim, seqlen_k/2] uint8 (transposed)
    uint8_t* v_hat_ptr;
    index_t  stride_v_hat;        // = seqlen_k / 2
    index_t  nhead_stride_v_hat;
    index_t  batch_stride_v_hat;

    // V scale: [batch, nhead, hdim, seqlen_k/32] uint8
    uint8_t* v_scale_ptr;
    index_t  stride_v_scale;      // = seqlen_k / kScaleGranularity
    index_t  nhead_stride_v_scale;
    index_t  batch_stride_v_scale;

    index_t nhead;
    index_t batch;
};

template <typename InputT>
struct SageAttnVPreprocessKargs
{
    const InputT* v_ptr;
    index_t       seqlen_k;
    index_t       hdim;
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

    index_t nhead;
};

template <typename InputT_, index_t kVGroup_ = 32, index_t kVHdimTile_ = 32>
struct SageAttnVPreprocessKernel
{
    using InputT = InputT_;

    static constexpr index_t kVGroup         = kVGroup_;
    static constexpr index_t kVHdimTile      = kVHdimTile_;
    static constexpr index_t kScaleGranularity = 32;
    static constexpr index_t kBlockSize      = kVGroup;
    // Pad LDS rows by 1 float32 to make column stride coprime with bank count.
    //   gcd(kVHdimTile + 1, 64) = gcd(33, 64) = 1 → zero bank conflicts.
    static constexpr index_t kLDSPad        = 1;

    static_assert(kVGroup == kScaleGranularity,
                  "kVGroup must equal kScaleGranularity (32) for MXFP4");
    static_assert(kVHdimTile % kVGroup == 0 || kVGroup % kVHdimTile == 0,
                  "kVHdimTile and kVGroup must be multiples of each other");

    using Kargs = SageAttnVPreprocessKargs<InputT>;
    using Hargs = SageAttnVPreprocessHostArgs;

    CK_TILE_HOST static Kargs MakeKargs(const Hargs& h)
    {
        Kargs k{};
        k.v_ptr                = static_cast<const InputT*>(h.v_ptr);
        k.seqlen_k             = h.seqlen_k;
        k.hdim                 = h.hdim;
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
        k.nhead                = h.nhead;
        return k;
    }

    // Grid: (seqlen_k/kVGroup, hdim/kVHdimTile, batch*nhead)
    CK_TILE_HOST static dim3 GridSize(const Hargs& h)
    {
        return dim3(h.seqlen_k / kVGroup, h.hdim / kVHdimTile, h.batch * h.nhead);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    // Static LDS: kVGroup × (kVHdimTile + kLDSPad) float32.
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kVGroup * (kVHdimTile + kLDSPad) * sizeof(float);
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t g_idx    = blockIdx.x;           // seqlen group index
        const index_t d_idx    = blockIdx.y;           // hdim tile index
        const index_t bh_idx   = blockIdx.z;           // batch * nhead (linear)
        const index_t head_idx  = bh_idx % kargs.nhead;
        const index_t batch_idx = bh_idx / kargs.nhead;
        const index_t tid       = get_thread_id();     // 0 .. kVGroup-1

        // bank(j, d) = (j*(kVHdimTile+kLDSPad) + d) % 64
        //   gcd(kVHdimTile+kLDSPad, 64) = gcd(33, 64) = 1 → zero bank conflicts.
        __shared__ float smem[kVGroup][kVHdimTile + kLDSPad];

        // ---- Step 1: coalesced load [kVGroup, kVHdimTile] from V → float LDS ----
        const index_t row       = g_idx * kVGroup + tid;
        const index_t col_start = d_idx * kVHdimTile;

        const InputT* v_base =
            kargs.v_ptr + batch_idx * kargs.batch_stride_v +
            head_idx * kargs.nhead_stride_v;

        if(row < kargs.seqlen_k)
        {
            // Each thread reads kVHdimTile consecutive elements — coalesced.
            const InputT* v_row = v_base + row * kargs.hdim + col_start;
            for(index_t d = 0; d < kVHdimTile; d++)
                smem[tid][d] = static_cast<float>(v_row[d]);
        }
        else
        {
            for(index_t d = 0; d < kVHdimTile; d++)
                smem[tid][d] = 0.0f;
        }
        block_sync_lds();

        // ---- Step 2: quantize per hdim channel from LDS ----
        // With kVHdimTile == kVGroup each thread handles exactly one channel.
        constexpr float rcp_dst_max = 1.0f / 6.0f;

        uint8_t* v_hat_base =
            kargs.v_hat_ptr + batch_idx * kargs.batch_stride_v_hat +
            head_idx * kargs.nhead_stride_v_hat;

        uint8_t* v_scale_base =
            kargs.v_scale_ptr + batch_idx * kargs.batch_stride_v_scale +
            head_idx * kargs.nhead_stride_v_scale;

        for(index_t d_local = tid; d_local < kVHdimTile; d_local += kVGroup)
        {
            const index_t d_global = col_start + d_local;

            // Read one column of the LDS tile → kVGroup float values.
            // bank(j, d_local) distinct for j=0..kVGroup-1 (no conflict).
            float group_data[kVGroup];
            float max_abs = 0.0f;
            for(index_t j = 0; j < kVGroup; j++)
            {
                group_data[j] = smem[j][d_local];
                max_abs       = max(max_abs, abs(group_data[j]));
            }

            // MXFP4 scale (e8m0, round-up to next power-of-two exponent).
            const float scale = bit_cast<float>(
                (bit_cast<uint32_t>(max_abs * rcp_dst_max) + numeric_traits<float>::mant_mask) &
                numeric_traits<float>::head_mask);

            // V_scale[d_global, g_idx]  (transposed layout)
            v_scale_base[d_global * kargs.stride_v_scale + g_idx] =
                static_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);

            // V_hat[d_global, g_idx*(kVGroup/2) ..]
            uint8_t* hat_ptr =
                v_hat_base + d_global * kargs.stride_v_hat + g_idx * (kVGroup / 2);
            PackFP4Group(group_data, hat_ptr, scale);
        }
    }

private:
    CK_TILE_DEVICE void PackFP4Group(const float* __restrict__ group_data,
                                      uint8_t* __restrict__ hat_group,
                                      float scale) const
    {
        for(index_t j = 0; j < kVGroup / 8; j++)
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
