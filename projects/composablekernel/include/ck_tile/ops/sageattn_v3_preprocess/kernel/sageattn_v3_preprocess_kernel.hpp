// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/pipeline/sageattn_v3_preprocess_pipeline.hpp"

namespace ck_tile {

// Q preprocess kernel: grid=(num_q_tiles, nhead_q, batch), blocksize=kQMeanGroupSize*(kHeadDim/32).
template <typename InputT>
struct SageAttnV3QPreprocessKargs
{
    const InputT* q_ptr;
    index_t seqlen_q;
    index_t stride_q;
    index_t nhead_stride_q;
    index_t batch_stride_q;

    uint8_t* q_hat_ptr;
    index_t stride_q_hat;
    index_t nhead_stride_q_hat;
    index_t batch_stride_q_hat;

    uint8_t* q_scale_ptr;
    index_t stride_q_scale;
    index_t nhead_stride_q_scale;
    index_t batch_stride_q_scale;

    // q_mean: [batch, nhead, num_q_tiles, hdim] InputT
    InputT* q_mean_ptr;
    index_t stride_q_mean;
    index_t nhead_stride_q_mean;
    index_t batch_stride_q_mean;

    index_t batch;
    index_t nhead;
    index_t num_q_tiles;
};

template <typename InputT_,
          index_t kQMeanGroupSize_,
          index_t kHeadDim_,
          index_t kBlockSize_ = kQMeanGroupSize_ * (kHeadDim_ / 32)>
struct SageAttnV3QPreprocessKernel
{
    using InputT = InputT_;

    static constexpr index_t kQMeanGroupSize      = kQMeanGroupSize_;
    static constexpr index_t kHeadDim      = kHeadDim_;
    static constexpr index_t kBlockSize = kBlockSize_;

    using Problem  = SA3QKPrepProblem<InputT, kQMeanGroupSize, kHeadDim>;
    using Pipeline = SA3QKPrepPipeline<Problem>;

    using Kargs = SageAttnV3QPreprocessKargs<InputT>;

    CK_TILE_HOST static dim3 GridSize(const Kargs& k)
    {
        return dim3(k.num_q_tiles, k.nhead, k.batch);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Pipeline::GetSmemSize(); }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t tile_x    = get_block_id();
        const index_t head_idx  = blockIdx.y;
        const index_t batch_idx = blockIdx.z;

        __shared__ char smem[GetSmemSize()];

        Pipeline pipeline{};

        const index_t row_start = tile_x * static_cast<index_t>(kQMeanGroupSize);
        const index_t n_rows    = min(static_cast<index_t>(kQMeanGroupSize), kargs.seqlen_q - row_start);

        const InputT* src_q = kargs.q_ptr + batch_idx * kargs.batch_stride_q +
                              head_idx * kargs.nhead_stride_q + row_start * kargs.stride_q;

        InputT* q_mean = kargs.q_mean_ptr + batch_idx * kargs.batch_stride_q_mean +
                         head_idx * kargs.nhead_stride_q_mean + tile_x * kargs.stride_q_mean;

        uint8_t* dst_q_hat = kargs.q_hat_ptr + batch_idx * kargs.batch_stride_q_hat +
                             head_idx * kargs.nhead_stride_q_hat + row_start * kargs.stride_q_hat;

        uint8_t* dst_q_scale = kargs.q_scale_ptr + batch_idx * kargs.batch_stride_q_scale +
                               head_idx * kargs.nhead_stride_q_scale +
                               row_start * kargs.stride_q_scale;

        pipeline.RunLoadQTile(src_q, smem);
        block_sync_lds();
        pipeline.RunQMean(smem, q_mean, n_rows);
        block_sync_lds();
        // mean is cached in smem_mean (LDS); RunQQuantize reads it from there.
        pipeline.RunQQuantize(smem, dst_q_hat, dst_q_scale, n_rows);
    }
};

// K preprocess kernel: K' = K - k_mean, write K', MXFP4 quantize.
// Grid: (num_k_tiles, nhead, batch), smem: kHeadDim*sizeof(float) for k_mean cache.
template <typename InputT>
struct SageAttnV3KPreprocessKargs
{
    const InputT* k_ptr;
    index_t seqlen_k;
    index_t stride_k;
    index_t nhead_stride_k;
    index_t batch_stride_k;

    uint8_t* k_hat_ptr;
    index_t stride_k_hat;
    index_t nhead_stride_k_hat;
    index_t batch_stride_k_hat;

    uint8_t* k_scale_ptr;
    index_t stride_k_scale;
    index_t nhead_stride_k_scale;
    index_t batch_stride_k_scale;

    // k_mean_float: [batch, nhead, hdim] float atomic partial sum (produced by KMean kernel)
    const float* k_mean_float;
    index_t nhead_stride_k_mean;
    index_t batch_stride_k_mean;

    InputT* k_prime_ptr;
    index_t stride_k_prime;
    index_t nhead_stride_k_prime;
    index_t batch_stride_k_prime;

    index_t batch;
    index_t nhead;
    index_t num_k_tiles;
};

template <typename InputT_,
          index_t kQMeanGroupSize_,
          index_t kHeadDim_,
          index_t kBlockSize_ = kQMeanGroupSize_ * (kHeadDim_ / 32)>
struct SageAttnV3KPreprocessKernel
{
    using InputT = InputT_;

    static constexpr index_t kQMeanGroupSize      = kQMeanGroupSize_;
    static constexpr index_t kHeadDim      = kHeadDim_;
    static constexpr index_t kBlockSize = kBlockSize_;

    using Problem  = SA3QKPrepProblem<InputT, kQMeanGroupSize, kHeadDim>;
    using Pipeline = SA3QKPrepPipeline<Problem>;

    using Kargs = SageAttnV3KPreprocessKargs<InputT>;

    CK_TILE_HOST static dim3 GridSize(const Kargs& k)
    {
        return dim3(k.num_k_tiles, k.nhead, k.batch);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Pipeline::GetKSmemSize(); }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t tile_x    = get_block_id();
        const index_t head_idx  = blockIdx.y;
        const index_t batch_idx = blockIdx.z;

        Pipeline pipeline{};

        const index_t row_start = tile_x * static_cast<index_t>(kQMeanGroupSize);
        const index_t n_rows    = min(static_cast<index_t>(kQMeanGroupSize), kargs.seqlen_k - row_start);

        const InputT* src_k = kargs.k_ptr + batch_idx * kargs.batch_stride_k +
                              head_idx * kargs.nhead_stride_k + row_start * kargs.stride_k;

        const float* k_mean_float = kargs.k_mean_float + batch_idx * kargs.batch_stride_k_mean +
                                    head_idx * kargs.nhead_stride_k_mean;

        InputT* k_prime = kargs.k_prime_ptr + batch_idx * kargs.batch_stride_k_prime +
                          head_idx * kargs.nhead_stride_k_prime + row_start * kargs.stride_k_prime;

        uint8_t* dst_k_hat = kargs.k_hat_ptr + batch_idx * kargs.batch_stride_k_hat +
                             head_idx * kargs.nhead_stride_k_hat + row_start * kargs.stride_k_hat;

        uint8_t* dst_k_scale = kargs.k_scale_ptr + batch_idx * kargs.batch_stride_k_scale +
                               head_idx * kargs.nhead_stride_k_scale +
                               row_start * kargs.stride_k_scale;

        __shared__ char smem[GetSmemSize()];
        const float seqlen_k_inv = 1.0f / static_cast<float>(kargs.seqlen_k);
        pipeline.RunKSmoothAndQuantize(src_k,
                                       k_mean_float,
                                       seqlen_k_inv,
                                       k_prime,
                                       kargs.stride_k_prime,
                                       dst_k_hat,
                                       dst_k_scale,
                                       smem,
                                       n_rows);
    }
};

// KMean kernel: per-head column mean of K via register accumulation (zero LDS).
// Grid: (ceil(seqlen_k/kChunkRows), nhead, batch), blocksize=kWarps*64=256.
template <typename InputT>
struct SageAttnV3KMeanKargs
{
    const InputT* k_ptr;
    index_t seqlen_k;
    index_t stride_k;
    index_t nhead_stride_k;
    index_t batch_stride_k;

    // float atomic partial sum: [batch, nhead, hdim]
    float* k_mean_float;
    index_t nhead_stride_kmean;
    index_t batch_stride_kmean;

    index_t batch;
    index_t nhead;
};

template <typename InputT_, index_t kQMeanGroupSize_, index_t kHeadDim_>
struct SageAttnV3KMeanKernel
{
    using InputT = InputT_;

    // kQMeanGroupSize_ for template-param consistency; KMean uses kChunkRows.
    static constexpr index_t kQMeanGroupSize = kQMeanGroupSize_;
    static constexpr index_t kHeadDim = kHeadDim_;

    using Problem  = SA3KMeanProblem<InputT, kHeadDim>;
    using Pipeline = SA3KMeanPipeline<Problem>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;
    static constexpr index_t kChunkRows = Problem::kChunkRows;

    using Kargs = SageAttnV3KMeanKargs<InputT>;

    CK_TILE_HOST static dim3 GridSize(const Kargs& k)
    {
        const index_t n_chunks = (k.seqlen_k + kChunkRows - 1) / kChunkRows;
        return dim3(n_chunks, k.nhead, k.batch);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Pipeline::GetSmemSize(); }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t chunk_idx = get_block_id();
        const index_t head_idx  = blockIdx.y;
        const index_t batch_idx = blockIdx.z;

        const index_t row_start = chunk_idx * kChunkRows;
        const index_t row_end   = min(row_start + kChunkRows, kargs.seqlen_k);
        const index_t num_rows  = row_end - row_start;

        const InputT* k_head =
            kargs.k_ptr + batch_idx * kargs.batch_stride_k + head_idx * kargs.nhead_stride_k;
        const InputT* k_chunk = k_head + row_start * kargs.stride_k;

        float* k_mean_out = kargs.k_mean_float + batch_idx * kargs.batch_stride_kmean +
                            head_idx * kargs.nhead_stride_kmean;

        Pipeline pipeline{};
        pipeline.RunKMean(k_chunk, kargs.stride_k, num_rows, k_mean_out);
    }
};

// V preprocess kernel: quantize V transposed layout via 2-phase LDS pipeline.
// Grid: (seqlen_k/(kScaleGranularity*kScaleGroupsPerTile), 1, batch*nhead),
// blocksize=kScaleGroupsPerTile*kVHdimTile.
template <typename InputT>
struct SageAttnV3VPreprocessKargs
{
    const InputT* v_ptr;
    index_t seqlen_k;      // padded (GridSize)
    index_t seqlen_k_real; // unpadded (bounds check)
    index_t hdim;
    index_t nhead_stride_v;
    index_t batch_stride_v;

    uint8_t* v_hat_ptr;
    index_t stride_v_hat;
    index_t nhead_stride_v_hat;
    index_t batch_stride_v_hat;

    uint8_t* v_scale_ptr;
    index_t stride_v_scale;
    index_t nhead_stride_v_scale;
    index_t batch_stride_v_scale;

    index_t nhead;
    index_t batch;
};

template <typename InputT_,
          index_t kVHdimTile_ = 128> // full hdim; set per-dispatch
struct SageAttnV3VPreprocessKernel
{
    using InputT = InputT_;

    using Problem  = SA3VPrepProblem<InputT, kVHdimTile_>;
    using Pipeline = SA3VPrepPipeline<Problem>;

    static constexpr index_t kScaleGranularity          = Problem::kScaleGranularity;
    static constexpr index_t kVHdimTile       = Problem::kVHdimTile;
    static constexpr index_t kScaleGroupsPerTile = Problem::kScaleGroupsPerTile;
    static constexpr index_t kBlockSize       = Problem::kBlockSize;

    using Kargs = SageAttnV3VPreprocessKargs<InputT>;

    CK_TILE_HOST static dim3 GridSize(const Kargs& k)
    {
        return dim3(k.seqlen_k / (kScaleGranularity * kScaleGroupsPerTile), 1, k.batch * k.nhead);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Pipeline::GetSmemSize(); }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t bh_idx     = blockIdx.z;
        const index_t head_idx   = bh_idx % kargs.nhead;
        const index_t batch_idx  = bh_idx / kargs.nhead;
        const index_t g_abs_base = blockIdx.x * kScaleGroupsPerTile;

        const InputT* v_base =
            kargs.v_ptr + batch_idx * kargs.batch_stride_v + head_idx * kargs.nhead_stride_v;

        __shared__ char smem_raw[GetSmemSize()];
        InputT* smem_v = reinterpret_cast<InputT*>(smem_raw);

        Pipeline pipeline{};

        pipeline.RunLoadToSmem(
            v_base, kargs.hdim, kargs.seqlen_k, kargs.seqlen_k_real, g_abs_base, smem_v);
        block_sync_lds();

        uint8_t* v_hat_base = kargs.v_hat_ptr + batch_idx * kargs.batch_stride_v_hat +
                              head_idx * kargs.nhead_stride_v_hat;
        uint8_t* v_scale_base = kargs.v_scale_ptr + batch_idx * kargs.batch_stride_v_scale +
                                head_idx * kargs.nhead_stride_v_scale;

        pipeline.RunQuantize(
            smem_v, g_abs_base, v_hat_base, v_scale_base, kargs.stride_v_hat, kargs.stride_v_scale);
    }
};

// ============================================================================
// Fused K+V scale packing kernel: e8m0 uint8 → packed int32 for MFMA OPSEL.
//
// Each thread packs one K scale int32 AND one V scale int32 (if in range).
// Grid covers max(k_total, v_total) per head; threads beyond either total
// skip that half.
//
// K input:  k_scale [seqlen_k_padded, hdim/32] uint8  (row-major, shuffled)
// K output: k_scale_packed [k_total_per_head] int32
// V input:  v_scale [hdim, seqlen_k_padded/32] uint8  (col-major, shuffled)
// V output: v_scale_packed [v_total_per_head] int32
// ============================================================================
struct SageAttnV3ScalePackKargs
{
    const uint8_t* k_scale_ptr;
    int32_t* k_scale_packed_ptr;
    index_t stride_k_scale;
    index_t nhead_stride_k_scale;
    index_t batch_stride_k_scale;
    index_t nhead_stride_k_scale_packed;
    index_t batch_stride_k_scale_packed;

    const uint8_t* v_scale_ptr;
    int32_t* v_scale_packed_ptr;
    index_t stride_v_scale;
    index_t nhead_stride_v_scale;
    index_t batch_stride_v_scale;
    index_t nhead_stride_v_scale_packed;
    index_t batch_stride_v_scale_packed;

    index_t hdim;
    index_t seqlen_k_padded;
    index_t nhead;
    index_t batch;
};

template <index_t kK0_>
struct SageAttnV3ScalePackKernel
{
    static constexpr index_t kK0              = kK0_;
    static constexpr index_t kK1              = kK0_;
    static constexpr index_t kN0              = 128;
    static constexpr index_t kScaleGran       = 32;
    static constexpr index_t kScaleRepeatB    = 4;
    static constexpr index_t kCNLane          = (kK0 <= 64) ? 32 : 16;
    static constexpr index_t kABScaleKLane    = kK0 / kScaleGran;
    static constexpr index_t kNIterPerWarp    = kN0 / kCNLane;
    static constexpr index_t kNumScaleGroupsK = kNIterPerWarp / kScaleRepeatB;
    static constexpr index_t kBlockSize       = 256;

    using Kargs = SageAttnV3ScalePackKargs;

    CK_TILE_HOST_DEVICE static index_t KPackedPerHead(index_t seqlen_k_padded, index_t hdim)
    {
        const index_t num_blocks     = seqlen_k_padded / kN0;
        const index_t k0_loops       = hdim / kK0;
        const index_t kScalePerBlock = k0_loops * kNumScaleGroupsK;
        return num_blocks * kABScaleKLane * kScalePerBlock;
    }

    CK_TILE_HOST_DEVICE static index_t VPackedPerHead(index_t seqlen_k_padded, index_t hdim)
    {
        const index_t num_blocks       = seqlen_k_padded / kN0;
        const index_t k1_loops         = kN0 / kK1;
        const index_t kNIterPerWarp_V  = hdim / kCNLane;
        const index_t kNumScaleGroupsV = kNIterPerWarp_V / kScaleRepeatB;
        const index_t kScalePerBlock_V = k1_loops * kNumScaleGroupsV;
        return num_blocks * kABScaleKLane * kScalePerBlock_V;
    }

    CK_TILE_HOST static dim3 GridSize(const Kargs& k)
    {
        const index_t k_total = KPackedPerHead(k.seqlen_k_padded, k.hdim);
        const index_t v_total = VPackedPerHead(k.seqlen_k_padded, k.hdim);
        const index_t total   = (k_total > v_total) ? k_total : v_total;
        return dim3((total + kBlockSize - 1) / kBlockSize, k.nhead, k.batch);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t head_idx  = blockIdx.y;
        const index_t batch_idx = blockIdx.z;
        const index_t flat_id   = blockIdx.x * kBlockSize + threadIdx.x;

        const index_t k0_loops = kargs.hdim / kK0;
        const index_t k1_loops = kN0 / kK1;

        // ---- K scale pack ----
        const index_t k_total = KPackedPerHead(kargs.seqlen_k_padded, kargs.hdim);
        if(flat_id < k_total)
        {
            index_t rem      = flat_id;
            const index_t sg = rem % kNumScaleGroupsK;
            rem /= kNumScaleGroupsK;
            const index_t ki = rem % k0_loops;
            rem /= k0_loops;
            const index_t kg = rem % kABScaleKLane;
            rem /= kABScaleKLane;
            const index_t block = rem;

            const index_t col    = ki * kABScaleKLane + kg;
            const uint8_t* k_src = kargs.k_scale_ptr + batch_idx * kargs.batch_stride_k_scale +
                                   head_idx * kargs.nhead_stride_k_scale;

            int32_t packed = 0;
            for(index_t si = 0; si < kScaleRepeatB; si++)
            {
                const index_t row = block * kN0 + (sg * kScaleRepeatB + si) * kCNLane;
                uint8_t byte_val  = 0;
                if(row < kargs.seqlen_k_padded)
                    byte_val = k_src[row * kargs.stride_k_scale + col];
                packed |= static_cast<int32_t>(byte_val) << (si * 8);
            }

            int32_t* k_dst = kargs.k_scale_packed_ptr +
                             batch_idx * kargs.batch_stride_k_scale_packed +
                             head_idx * kargs.nhead_stride_k_scale_packed;
            k_dst[flat_id] = packed;
        }

        // ---- V scale pack ----
        const index_t v_total = VPackedPerHead(kargs.seqlen_k_padded, kargs.hdim);
        if(flat_id < v_total)
        {
            const index_t kNIterPerWarp_V  = kargs.hdim / kCNLane;
            const index_t kNumScaleGroupsV = kNIterPerWarp_V / kScaleRepeatB;

            index_t rem      = flat_id;
            const index_t sg = rem % kNumScaleGroupsV;
            rem /= kNumScaleGroupsV;
            const index_t ki = rem % k1_loops;
            rem /= k1_loops;
            const index_t kg = rem % kABScaleKLane;
            rem /= kABScaleKLane;
            const index_t block = rem;

            const index_t v_col  = block * (kK1 / kScaleGran) + ki * kABScaleKLane + kg;
            const uint8_t* v_src = kargs.v_scale_ptr + batch_idx * kargs.batch_stride_v_scale +
                                   head_idx * kargs.nhead_stride_v_scale;

            int32_t packed = 0;
            for(index_t si = 0; si < kScaleRepeatB; si++)
            {
                const index_t v_row = (sg * kScaleRepeatB + si) * kCNLane;
                uint8_t byte_val    = 0;
                if(v_row < kargs.hdim)
                    byte_val = v_src[v_row * kargs.stride_v_scale + v_col];
                packed |= static_cast<int32_t>(byte_val) << (si * 8);
            }

            int32_t* v_dst = kargs.v_scale_packed_ptr +
                             batch_idx * kargs.batch_stride_v_scale_packed +
                             head_idx * kargs.nhead_stride_v_scale_packed;
            v_dst[flat_id] = packed;
        }
    }
};

} // namespace ck_tile
