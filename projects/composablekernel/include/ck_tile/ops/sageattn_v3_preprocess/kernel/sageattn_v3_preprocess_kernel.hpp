// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/pipeline/sageattn_v3_preprocess_pipeline.hpp"

namespace ck_tile {

// Q preprocess kernel: grid=(num_q_tiles, nhead_q, batch), blocksize=kRows*(kCols/32).
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
          index_t kRows_,
          index_t kCols_,
          index_t kBlockSize_ = kRows_ * (kCols_ / 32)>
struct SageAttnV3QPreprocessKernel
{
    using InputT = InputT_;

    static constexpr index_t kRows      = kRows_;
    static constexpr index_t kCols      = kCols_;
    static constexpr index_t kBlockSize = kBlockSize_;

    using Problem  = SA3QKPrepProblem<InputT, kRows, kCols>;
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

        const index_t row_start = tile_x * static_cast<index_t>(kRows);
        const index_t n_rows    = min(static_cast<index_t>(kRows), kargs.seqlen_q - row_start);

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
// Grid: (num_k_tiles, nhead, batch), smem: kCols*sizeof(float) for k_mean cache.
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
          index_t kRows_,
          index_t kCols_,
          index_t kBlockSize_ = kRows_ * (kCols_ / 32)>
struct SageAttnV3KPreprocessKernel
{
    using InputT = InputT_;

    static constexpr index_t kRows      = kRows_;
    static constexpr index_t kCols      = kCols_;
    static constexpr index_t kBlockSize = kBlockSize_;

    using Problem  = SA3QKPrepProblem<InputT, kRows, kCols>;
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

        const index_t row_start = tile_x * static_cast<index_t>(kRows);
        const index_t n_rows    = min(static_cast<index_t>(kRows), kargs.seqlen_k - row_start);

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
        pipeline.RunKSmoothAndQuantize(
            src_k, k_mean_float, seqlen_k_inv,
            k_prime, kargs.stride_k_prime, dst_k_hat, dst_k_scale, smem, n_rows);
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

template <typename InputT_, index_t kRows_, index_t kCols_>
struct SageAttnV3KMeanKernel
{
    using InputT = InputT_;

    // kRows_ for template-param consistency; KMean uses kChunkRows.
    static constexpr index_t kRows  = kRows_;
    static constexpr index_t kCols  = kCols_;

    using Problem  = SA3KMeanProblem<InputT, kCols>;
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
// Grid: (seqlen_k/(kVGroup*kVGroupsPerBlock), 1, batch*nhead), blocksize=kVGroupsPerBlock*kVHdimTile.
template <typename InputT>
struct SageAttnV3VPreprocessKargs
{
    const InputT* v_ptr;
    index_t seqlen_k;       // padded (GridSize)
    index_t seqlen_k_real;  // unpadded (bounds check)
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
          index_t kVGroup_    = 32,
          index_t kVHdimTile_ = 128>   // full hdim; set per-dispatch
struct SageAttnV3VPreprocessKernel
{
    using InputT = InputT_;

    using Problem  = SA3VPrepProblem<InputT, kVHdimTile_>;
    using Pipeline = SA3VPrepPipeline<Problem>;

    static constexpr index_t kVGroup          = Problem::kVGroup;
    static constexpr index_t kVHdimTile       = Problem::kVHdimTile;
    static constexpr index_t kVGroupsPerBlock = Problem::kVGroupsPerBlock;
    static constexpr index_t kBlockSize       = Problem::kBlockSize;

    using Kargs = SageAttnV3VPreprocessKargs<InputT>;

    CK_TILE_HOST static dim3 GridSize(const Kargs& k)
    {
        return dim3(k.seqlen_k / (kVGroup * kVGroupsPerBlock), 1, k.batch * k.nhead);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Pipeline::GetSmemSize(); }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t bh_idx    = blockIdx.z;
        const index_t head_idx  = bh_idx % kargs.nhead;
        const index_t batch_idx = bh_idx / kargs.nhead;
        const index_t g_abs_base = blockIdx.x * kVGroupsPerBlock;

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
            smem_v, g_abs_base, v_hat_base, v_scale_base,
            kargs.stride_v_hat, kargs.stride_v_scale);
    }
};

} // namespace ck_tile
