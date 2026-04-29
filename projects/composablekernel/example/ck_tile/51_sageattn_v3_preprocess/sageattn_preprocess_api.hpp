// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/sageattn_v3_preprocess/pipeline/sageattn_v3_preprocess_pipeline.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/kernel/sageattn_v3_preprocess_kernel.hpp"

#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/hip_check_error.hpp"

namespace ck_tile {

// User-facing args for SageAttnV3Preprocess::run().
// Fill only input strides and output pointers; all output strides and
// padded dimensions are derived internally from (seqlen_q, seqlen_k, hdim).
//
// GQA support: nhead_q may differ from nhead_kv.
//   Q layout: [batch, nhead_q,  seqlen_q, hdim]
//   K layout: [batch, nhead_kv, seqlen_k, hdim]
//   V layout: [batch, nhead_kv, seqlen_k, hdim]
// When nhead_q == nhead_kv this is equivalent to standard MHA.
template <typename InputT>
struct SageAttnV3PreprocessArgs
{
    // --- Q: [batch, nhead_q, seqlen_q, hdim] InputT ---
    const InputT* q_ptr;
    index_t seqlen_q;
    index_t hdim;
    index_t stride_q;       // = hdim (row-major Q)
    index_t nhead_stride_q; // = seqlen_q * hdim
    index_t batch_stride_q; // = nhead_q * seqlen_q * hdim

    // Q outputs (pointers only; strides derived from padded dims)
    uint8_t* q_hat_ptr;   // [batch, nhead_q, seqlen_q_padded, hdim/2] uint8
    uint8_t* q_scale_ptr; // [batch, nhead_q, seqlen_q_padded, hdim/32] uint8
    InputT* q_mean_ptr;   // [batch, nhead_q, num_q_tiles, hdim] InputT

    // --- K: [batch, nhead_kv, seqlen_k, hdim] InputT ---
    const InputT* k_ptr;
    index_t seqlen_k;
    index_t stride_k;       // = hdim (row-major K)
    index_t nhead_stride_k; // = seqlen_k * hdim
    index_t batch_stride_k; // = nhead_kv * seqlen_k * hdim

    // K outputs (pointers only)
    uint8_t* k_hat_ptr;   // [batch, nhead_kv, seqlen_k_padded, hdim/2] uint8
    uint8_t* k_scale_ptr; // [batch, nhead_kv, seqlen_k_padded, hdim/32] uint8

    // --- V: [batch, nhead_kv, seqlen_k, hdim] InputT ---
    const InputT* v_ptr;
    index_t nhead_stride_v; // = seqlen_k * hdim
    index_t batch_stride_v; // = nhead_kv * seqlen_k * hdim

    // V outputs (pointers only; transposed layout [batch, nhead_kv, hdim, seqlen_k_padded/...])
    uint8_t* v_hat_ptr;   // [batch, nhead_kv, hdim, seqlen_k_padded/2] uint8
    uint8_t* v_scale_ptr; // [batch, nhead_kv, hdim, seqlen_k_padded/32] uint8

    // Packed scale outputs for MFMA OPSEL (optional, nullptr to skip)
    int32_t* k_scale_packed_ptr; // [batch, nhead_kv, packed_size_per_head_k] int32
    int32_t* v_scale_packed_ptr; // [batch, nhead_kv, packed_size_per_head_v] int32

    // --- Common dimensions ---
    index_t batch;
    index_t nhead_q;  // number of Q heads
    index_t nhead_kv; // number of KV heads (may differ from nhead_q for GQA)
};

// Stage bitmask for sageattn_v3_preprocess_run.
// Each bit enables one kernel launch; default (kSA3StageAll) runs all.
static constexpr uint32_t kSA3StageKMean       = 1u << 0; // Launch 0: KMean
static constexpr uint32_t kSA3StageQPreprocess = 1u << 1; // Launch 1: Q preprocess
static constexpr uint32_t kSA3StageKPreprocess = 1u << 2; // Launch 2: K preprocess
static constexpr uint32_t kSA3StageVPreprocess = 1u << 3; // Launch 3: V preprocess
static constexpr uint32_t kSA3StageDeltaS      = 1u << 4; // Launch 4: delta_s GEMM
static constexpr uint32_t kSA3StageScalePack   = 1u << 5; // Launch 5: fused K+V scale pack
static constexpr uint32_t kSA3StageAll         = 0x3Fu;   // All stages (bits 0-5)

// ============================================================================
// sageattn_v3_preprocess_run
//
// Host-side wrapper that launches the five kernels required for SA3
// preprocessing on a single HIP stream. Supports GQA (nhead_q != nhead_kv).
//
//   Launch 0: SageAttnV3KMeanKernel
//     Grid: (num_k_tiles, nhead_kv, batch).
//     Atomic-adds partial K column sums into k_mean_float (float scratch).
//
//   Launch 1: SageAttnV3QPreprocessKernel
//     Grid: (num_q_tiles, nhead_q, batch).
//     Load Q tile -> LDS, compute per-tile column mean, MXFP4 quantize.
//     -> q_hat / q_scale / q_mean.
//
//   Launch 2: SageAttnV3KPreprocessKernel
//     Grid: (num_k_tiles, nhead_kv, batch).
//     K' = K - k_mean (reads k_mean_float, divides by seqlen_k in registers),
//     MXFP4 quantize K' -> k_hat / k_scale / k_prime.
//
//   Launch 3: SageAttnV3VPreprocessKernel
//     Grid: (seqlen_k/(R*kVGroup), 1, batch*nhead_kv). BlockSize: R*hdim.
//     V: fp16 LDS tile load + direct VGPR->global MXFP4 write.
//
//   Launch 4: BatchedGemmKernel
//     delta_s = q_mean @ K'^T
//     A: q_mean  [batch*nhead_q,  num_q_tiles, hdim] InputT  RowMajor
//     B: K'      [batch*nhead_kv, seqlen_k,    hdim] InputT  ColMajor
//     C: delta_s [batch*nhead_q,  num_q_tiles, seqlen_k] float RowMajor
//     Note: for GQA, q_mean is broadcast over KV heads inside the GEMM via
//     batch_stride_A = 0 when nhead_q < nhead_kv (one q_mean per KV group).
//     Currently requires nhead_q == nhead_kv (delta_s per Q head independently).
//
// Caller-allocated device buffers:
//   k_mean_buf:  float  [batch, nhead_kv, hdim]  (float scratch)
//   k_prime_buf: InputT [batch, nhead_kv, seqlen_k_padded, hdim]
// ============================================================================

// Buffer size descriptor returned by get_buffer_sizes.
// All *_bytes fields are byte counts for device allocation.
struct SageAttnV3PreprocessBufferSizes
{
    // ---- outputs produced by sageattn_v3_preprocess_run ----
    size_t delta_s_bytes;        // float  [B, H, num_q_tiles, seqlen_k_padded]
    size_t k_prime_bytes;        // InputT [B, H, seqlen_k_padded, hdim]
    size_t q_hat_bytes;          // uint8  [B, H, seqlen_q_padded, hdim/2]
    size_t q_scale_bytes;        // uint8  [B, H, seqlen_q_padded, hdim/32]
    size_t q_mean_bytes;         // InputT [B, H, num_q_tiles, hdim]
    size_t k_hat_bytes;          // uint8  [B, H, seqlen_k_padded, hdim/2]
    size_t k_scale_bytes;        // uint8  [B, H, seqlen_k_padded, hdim/32]
    size_t v_hat_bytes;          // uint8  [B, H, hdim, seqlen_k_padded/2]
    size_t v_scale_bytes;        // uint8  [B, H, hdim, seqlen_k_padded/32]
    size_t k_scale_packed_bytes; // int32  [B, H_kv, packed_k_per_head]
    size_t v_scale_packed_bytes; // int32  [B, H_kv, packed_v_per_head]
    // ---- caller-allocated scratch (unchanged by padding) ----
    size_t k_mean_bytes; // InputT [B, H, hdim]
    // ---- padded dims ----
    index_t seqlen_q_padded;
    index_t seqlen_k_padded;
    index_t num_q_tiles;
    index_t num_k_tiles;
};

// GEMM tile configuration for delta_s = q_mean @ K'^T.
//   M = num_q_tiles (small, typically 1-64)
//   N = seqlen_k    (large)
//   K = hdim        (128 or 256)
//
// BlockTile<32,64,32>, BlockWarps<1,2,1>: MWarp=1, NWarp=2 -> better tile_A reuse.
template <typename InputT>
using DeltaSV3GemmShape = std::conditional_t<
    std::is_same_v<InputT, float>,
    TileGemmShape<sequence<32, 64, 32>, sequence<1, 2, 1>, sequence<32, 32, 8>>,
    TileGemmShape<sequence<32, 64, 32>, sequence<1, 2, 1>, sequence<32, 32, 16>>>;

template <typename InputT, index_t kQMeanGroupSize, index_t kHeadDim>
struct SageAttnV3Preprocess
{
    using Args        = SageAttnV3PreprocessArgs<InputT>;
    using BufferSizes = SageAttnV3PreprocessBufferSizes;

    static constexpr index_t kFmhaTileN0       = 128;
    static constexpr index_t kScaleGranularity  = 32;

    static_assert(kFmhaTileN0 % kScaleGranularity == 0,
                  "kFmhaTileN0 must be divisible by kScaleGranularity");
    static_assert(kFmhaTileN0 % kQMeanGroupSize == 0 ||
                  kQMeanGroupSize % kFmhaTileN0 == 0,
                  "kQMeanGroupSize and kFmhaTileN0 must be multiples of each other");
    static_assert(kHeadDim % kScaleGranularity == 0,
                  "kHeadDim must be divisible by kScaleGranularity");

    static BufferSizes get_buffer_sizes(index_t batch,
                                        index_t nhead_q,
                                        index_t nhead_kv,
                                        index_t seqlen_q,
                                        index_t seqlen_k,
                                        index_t hdim)
    {
        constexpr index_t kG = 32; // MXFP4 scale granularity

        constexpr index_t kQPad = kQMeanGroupSize;
        constexpr index_t kKPad = (kFmhaTileN0 > kQMeanGroupSize) ? kFmhaTileN0
                                                                    : kQMeanGroupSize;

        const index_t sq_pad = ((seqlen_q + kQPad - 1) / kQPad) * kQPad;
        const index_t sk_pad = ((seqlen_k + kKPad - 1) / kKPad) * kKPad;
        const index_t nqt    = sq_pad / kQMeanGroupSize;
        const index_t nkt    = sk_pad / kQMeanGroupSize;

        SageAttnV3PreprocessBufferSizes s{};
        s.seqlen_q_padded = sq_pad;
        s.seqlen_k_padded = sk_pad;
        s.num_q_tiles     = nqt;
        s.num_k_tiles     = nkt;

        s.delta_s_bytes = static_cast<size_t>(batch * nhead_q * nqt * sk_pad) * sizeof(float);
        s.k_prime_bytes = static_cast<size_t>(batch * nhead_kv * sk_pad * hdim) * sizeof(InputT);
        s.q_hat_bytes   = static_cast<size_t>(batch * nhead_q * sq_pad * (hdim / 2));
        s.q_scale_bytes = static_cast<size_t>(batch * nhead_q * sq_pad * (hdim / kG));
        s.q_mean_bytes  = static_cast<size_t>(batch * nhead_q * nqt * hdim) * sizeof(InputT);
        s.k_hat_bytes   = static_cast<size_t>(batch * nhead_kv * sk_pad * (hdim / 2));
        s.k_scale_bytes = static_cast<size_t>(batch * nhead_kv * sk_pad * (hdim / kG));
        s.v_hat_bytes   = static_cast<size_t>(batch * nhead_kv * hdim * (sk_pad / 2));
        s.v_scale_bytes = static_cast<size_t>(batch * nhead_kv * hdim * (sk_pad / kG));
        // Packed scale sizes (depends on MFMA tile shape → kK0)
        {
            constexpr index_t kK0_          = (kHeadDim <= 128) ? 64 : 128;
            using PackKernel                = SageAttnV3ScalePackKernel<kK0_>;
            const index_t k_packed_per_head = PackKernel::KPackedPerHead(sk_pad, hdim);
            const index_t v_packed_per_head = PackKernel::VPackedPerHead(sk_pad, hdim);
            s.k_scale_packed_bytes =
                static_cast<size_t>(batch * nhead_kv * k_packed_per_head) * sizeof(int32_t);
            s.v_scale_packed_bytes =
                static_cast<size_t>(batch * nhead_kv * v_packed_per_head) * sizeof(int32_t);
        }
        // k_mean_buf: float scratch for KMean atomic accumulation, [batch, nhead_kv, hdim] float.
        s.k_mean_bytes = static_cast<size_t>(batch * nhead_kv * hdim) * sizeof(float);
        return s;
    }

    static void
    run(const SageAttnV3PreprocessArgs<InputT>& args,
        float* delta_s_ptr,  // [batch, nhead_q,  num_q_tiles, seqlen_k_padded] float
        float* k_mean_buf,   // [batch, nhead_kv, hdim]                         float scratch
        InputT* k_prime_buf, // [batch, nhead_kv, seqlen_k_padded, hdim]        InputT
        hipStream_t stream,
        uint32_t stages = kSA3StageAll)
    {
        const index_t batch    = args.batch;
        const index_t nhead_q  = args.nhead_q;
        const index_t nhead_kv = args.nhead_kv;
        const index_t hdim     = args.hdim;
        const index_t seqlen_q = args.seqlen_q;
        const index_t seqlen_k = args.seqlen_k;
        constexpr index_t kG   = 32; // MXFP4 scale granularity

        const auto bsz       = get_buffer_sizes(batch, nhead_q, nhead_kv, seqlen_q, seqlen_k, hdim);
        const index_t sq_pad = bsz.seqlen_q_padded;
        const index_t sk_pad = bsz.seqlen_k_padded;
        const index_t num_q_tiles = bsz.num_q_tiles;
        const index_t num_k_tiles = bsz.num_k_tiles;

        float* k_mean_float = k_mean_buf;

        // ------------------------------------------------------------------ //
        // Launch 0: KMean -- accumulate K column sums into float scratch.
        // ------------------------------------------------------------------ //
        if(stages & kSA3StageKMean)
        {
            using KMeanKernel = SageAttnV3KMeanKernel<InputT, kQMeanGroupSize, kHeadDim>;
            using KMeanKargs  = typename KMeanKernel::Kargs;

            const std::size_t k_mean_float_bytes =
                static_cast<std::size_t>(batch * nhead_kv * hdim) * sizeof(float);
            HIP_CHECK_ERROR(hipMemsetAsync(k_mean_float, 0, k_mean_float_bytes, stream));

            KMeanKargs kargs{};
            kargs.k_ptr              = args.k_ptr;
            kargs.seqlen_k           = seqlen_k;
            kargs.stride_k           = args.stride_k;
            kargs.nhead_stride_k     = args.nhead_stride_k;
            kargs.batch_stride_k     = args.batch_stride_k;
            kargs.k_mean_float       = k_mean_float;
            kargs.nhead_stride_kmean = hdim;
            kargs.batch_stride_kmean = nhead_kv * hdim;
            kargs.batch              = batch;
            kargs.nhead              = nhead_kv;

            const dim3 grids   = KMeanKernel::GridSize(kargs);
            const dim3 blocks  = KMeanKernel::BlockSize();
            const index_t smem = KMeanKernel::GetSmemSize();
            stream_config sc{stream};
            launch_and_check(sc, make_kernel(KMeanKernel{}, grids, blocks, smem, kargs));
        }

        // ------------------------------------------------------------------ //
        // Launch 1: Q preprocess -- load Q tile -> LDS, mean, MXFP4 quantize.
        // ------------------------------------------------------------------ //
        if(stages & kSA3StageQPreprocess)
        {
            using QKernel = SageAttnV3QPreprocessKernel<InputT, kQMeanGroupSize, kHeadDim>;
            using QKargs  = typename QKernel::Kargs;

            QKargs kargs{};
            kargs.q_ptr                = args.q_ptr;
            kargs.seqlen_q             = seqlen_q;
            kargs.stride_q             = args.stride_q;
            kargs.nhead_stride_q       = args.nhead_stride_q;
            kargs.batch_stride_q       = args.batch_stride_q;
            kargs.q_hat_ptr            = args.q_hat_ptr;
            kargs.stride_q_hat         = hdim / 2;
            kargs.nhead_stride_q_hat   = sq_pad * (hdim / 2);
            kargs.batch_stride_q_hat   = nhead_q * sq_pad * (hdim / 2);
            kargs.q_scale_ptr          = args.q_scale_ptr;
            kargs.stride_q_scale       = hdim / kG;
            kargs.nhead_stride_q_scale = sq_pad * (hdim / kG);
            kargs.batch_stride_q_scale = nhead_q * sq_pad * (hdim / kG);
            kargs.q_mean_ptr           = args.q_mean_ptr;
            kargs.stride_q_mean        = hdim;
            kargs.nhead_stride_q_mean  = num_q_tiles * hdim;
            kargs.batch_stride_q_mean  = nhead_q * num_q_tiles * hdim;
            kargs.batch                = batch;
            kargs.nhead                = nhead_q;
            kargs.num_q_tiles          = num_q_tiles;

            const dim3 grids   = QKernel::GridSize(kargs);
            const dim3 blocks  = QKernel::BlockSize();
            const index_t smem = QKernel::GetSmemSize();
            stream_config sc{stream};
            launch_and_check(sc, make_kernel(QKernel{}, grids, blocks, smem, kargs));
        }

        // ------------------------------------------------------------------ //
        // Launch 2: K preprocess -- K' = K - k_mean, MXFP4 quantize.
        // ------------------------------------------------------------------ //
        if(stages & kSA3StageKPreprocess)
        {
            using KKernel = SageAttnV3KPreprocessKernel<InputT, kQMeanGroupSize, kHeadDim>;
            using KKargs  = typename KKernel::Kargs;

            KKargs kargs{};
            kargs.k_ptr                = args.k_ptr;
            kargs.seqlen_k             = seqlen_k;
            kargs.stride_k             = args.stride_k;
            kargs.nhead_stride_k       = args.nhead_stride_k;
            kargs.batch_stride_k       = args.batch_stride_k;
            kargs.k_hat_ptr            = args.k_hat_ptr;
            kargs.stride_k_hat         = hdim / 2;
            kargs.nhead_stride_k_hat   = sk_pad * (hdim / 2);
            kargs.batch_stride_k_hat   = nhead_kv * sk_pad * (hdim / 2);
            kargs.k_scale_ptr          = args.k_scale_ptr;
            kargs.stride_k_scale       = hdim / kG;
            kargs.nhead_stride_k_scale = sk_pad * (hdim / kG);
            kargs.batch_stride_k_scale = nhead_kv * sk_pad * (hdim / kG);
            kargs.k_mean_float         = k_mean_float;
            kargs.nhead_stride_k_mean  = hdim;
            kargs.batch_stride_k_mean  = nhead_kv * hdim;
            kargs.k_prime_ptr          = k_prime_buf;
            kargs.stride_k_prime       = hdim;
            kargs.nhead_stride_k_prime = sk_pad * hdim;
            kargs.batch_stride_k_prime = nhead_kv * sk_pad * hdim;
            kargs.batch                = batch;
            kargs.nhead                = nhead_kv;
            kargs.num_k_tiles          = num_k_tiles;

            const dim3 grids   = KKernel::GridSize(kargs);
            const dim3 blocks  = KKernel::BlockSize();
            const index_t smem = KKernel::GetSmemSize();
            stream_config sc{stream};
            launch_and_check(sc, make_kernel(KKernel{}, grids, blocks, smem, kargs));
        }

        // ------------------------------------------------------------------ //
        // Launch 1c: V preprocess -- fp16 LDS tile + direct VGPR->global MXFP4 write
        // ------------------------------------------------------------------ //
        if(stages & kSA3StageVPreprocess)
        {
            // Dispatch on hdim: kVHdimTile=hdim so each CTA covers the full hdim (y=1).
            // kBlockSize = R * kVGroup * kVHdimTile / kLoadVec (= 4*hdim for fp16).
            auto dispatch_v = [&](auto kVHdimTile_c) {
                using VKernel = SageAttnV3VPreprocessKernel<InputT, kVHdimTile_c.value>;
                using VKargs  = typename VKernel::Kargs;

                VKargs kargs{};
                kargs.v_ptr                = args.v_ptr;
                kargs.seqlen_k             = sk_pad;   // used for GridSize (must be padded)
                kargs.seqlen_k_real        = seqlen_k; // used for input bounds check
                kargs.hdim                 = hdim;
                kargs.nhead_stride_v       = args.nhead_stride_v;
                kargs.batch_stride_v       = args.batch_stride_v;
                kargs.v_hat_ptr            = args.v_hat_ptr;
                kargs.stride_v_hat         = sk_pad / 2;
                kargs.nhead_stride_v_hat   = hdim * (sk_pad / 2);
                kargs.batch_stride_v_hat   = nhead_kv * hdim * (sk_pad / 2);
                kargs.v_scale_ptr          = args.v_scale_ptr;
                kargs.stride_v_scale       = sk_pad / kG;
                kargs.nhead_stride_v_scale = hdim * (sk_pad / kG);
                kargs.batch_stride_v_scale = nhead_kv * hdim * (sk_pad / kG);
                kargs.nhead                = nhead_kv;
                kargs.batch                = batch;

                const dim3 grids   = VKernel::GridSize(kargs);
                const dim3 blocks  = VKernel::BlockSize();
                const index_t smem = VKernel::GetSmemSize();

                stream_config sc{stream};
                launch_and_check(sc, make_kernel(VKernel{}, grids, blocks, smem, kargs));
            };

            if(hdim == 64)
                dispatch_v(number<64>{});
            else if(hdim == 128)
                dispatch_v(number<128>{});
            else if(hdim == 256)
                dispatch_v(number<256>{});
        } // end if(stages & kSA3StageVPreprocess)

        // ------------------------------------------------------------------ //
        // Launch 2: batched GEMM  delta_s = q_mean @ K'^T
        // ------------------------------------------------------------------ //
        if(stages & kSA3StageDeltaS)
        {
            using ALayout = tensor_layout::gemm::RowMajor;
            // K' stored naturally as [seqlen_k=N, hdim=K] row-major -> ColMajor B[K, N] -> K'^T
            using BLayout = tensor_layout::gemm::ColumnMajor;
            using CLayout = tensor_layout::gemm::RowMajor;

            constexpr bool kPadM         = true;  // num_q_tiles may not be multiple of M_Tile
            constexpr bool kPadN         = true;  // seqlen_k may not be multiple of N_Tile
            constexpr bool kPadK         = false; // hdim is always multiple of K_Tile
            constexpr bool DoubleSmemBuf = false;
            constexpr bool TransposeC    = false;

            using GemmTraits = TileGemmUniversalTraits<kPadM,
                                                       kPadN,
                                                       kPadK,
                                                       DoubleSmemBuf,
                                                       ALayout,
                                                       BLayout,
                                                       CLayout,
                                                       TransposeC>;

            using Shape = DeltaSV3GemmShape<InputT>;

            using GemmProblem = UniversalGemmPipelineProblem<InputT, // ADataType
                                                             InputT, // BDataType
                                                             float,  // AccDataType
                                                             Shape,
                                                             GemmTraits,
                                                             GemmPipelineScheduler::Intrawave>;

            using GemmPipeline = GemmPipelineAgBgCrCompV3<GemmProblem>;

            using GemmEpilogue = CShuffleEpilogue<CShuffleEpilogueProblem<
                InputT,  // ADataType
                InputT,  // BDataType
                tuple<>, // DsDataType (no D)
                float,   // AccDataType
                float,   // CDataType (delta_s is float)
                tuple<>, // DsLayout
                CLayout,
                element_wise::PassThrough,
                Shape::kM,
                Shape::kN,
                /*MWarp=*/1,
                /*NWarp=*/2,
                /*MWarpTile=*/32,
                /*NWarpTile=*/32,
                Shape::WarpTile::at(number<2>{}), // 16 for fp16/bf16, 8 for float
                TransposeC>>;

            using GemmTilePartitioner = GemmSpatiallyLocalTilePartitioner<Shape,
                                                                          /*GroupNum=*/8,
                                                                          /*M01=*/4>;
            using GemmKernel = BatchedGemmKernel<GemmTilePartitioner, GemmPipeline, GemmEpilogue>;

            const index_t M = num_q_tiles;
            const index_t N = sk_pad;
            const index_t K = hdim;

            // A: q_mean  [batch, nhead_q,  num_q_tiles, hdim]  row-major
            // B: K'      [batch, nhead_kv, sk_pad, hdim]  col-major (leading dim = K)
            // C: delta_s [batch, nhead_q,  num_q_tiles, sk_pad]  row-major
            stream_config sc{stream};

            if(nhead_q == nhead_kv)
            {
                // MHA path: single batched GEMM over all batch*nhead_q heads.
                const index_t batch_count = batch * nhead_q;
                BatchedGemmHostArgs gemm_hargs(args.q_mean_ptr, // A ptr
                                               k_prime_buf,     // B ptr (K')
                                               delta_s_ptr,     // C ptr
                                               /*k_batch=*/1,
                                               M,
                                               N,
                                               K,
                                               /*stride_A=*/K,
                                               /*stride_B=*/K,
                                               /*stride_C=*/N,
                                               /*batch_stride_A=*/M * K, // num_q_tiles * hdim
                                               /*batch_stride_B=*/N * K, // sk_pad * hdim
                                               /*batch_stride_C=*/M * N, // num_q_tiles * sk_pad
                                               batch_count);
                auto kargs       = GemmKernel::MakeKernelArgs(gemm_hargs);
                const dim3 grids = GemmKernel::GridSize(M, N, /*k_batch=*/1, batch_count);
                const dim3 blks  = GemmKernel::BlockSize();
                launch_and_check(sc, make_kernel(GemmKernel{}, grids, blks, 0, kargs));
            }
            else
            {
                // GQA path: run nhead_q separate batched GEMMs (batch_count=batch each).
                // Q head i_q uses KV head i_kv = i_q / ratio.
                // batch_stride for A and C skips all nhead_q heads; for B all nhead_kv heads.
                const index_t ratio          = nhead_q / nhead_kv;
                const index_t batch_stride_A = nhead_q * M * K;
                const index_t batch_stride_B = nhead_kv * N * K;
                const index_t batch_stride_C = nhead_q * M * N;
                for(index_t i_q = 0; i_q < nhead_q; i_q++)
                {
                    const index_t i_kv = i_q / ratio;
                    BatchedGemmHostArgs gemm_hargs(
                        args.q_mean_ptr + i_q * M * K, // A: q_mean head i_q, batch=0
                        k_prime_buf + i_kv * N * K,    // B: K'     head i_kv, batch=0
                        delta_s_ptr + i_q * M * N,     // C: delta_s head i_q, batch=0
                        /*k_batch=*/1,
                        M,
                        N,
                        K,
                        /*stride_A=*/K,
                        /*stride_B=*/K,
                        /*stride_C=*/N,
                        batch_stride_A,
                        batch_stride_B,
                        batch_stride_C,
                        batch);
                    auto kargs       = GemmKernel::MakeKernelArgs(gemm_hargs);
                    const dim3 grids = GemmKernel::GridSize(M, N, /*k_batch=*/1, batch);
                    const dim3 blks  = GemmKernel::BlockSize();
                    launch_and_check(sc, make_kernel(GemmKernel{}, grids, blks, 0, kargs));
                }
            }
        }

        // ------------------------------------------------------------------ //
        // Launch 5: fused K+V scale packing — e8m0 uint8 → packed int32
        // ------------------------------------------------------------------ //
        if((stages & kSA3StageScalePack) && args.k_scale_packed_ptr != nullptr &&
           args.v_scale_packed_ptr != nullptr)
        {
            constexpr index_t kK0_ = (kHeadDim <= 128) ? 64 : 128;
            using PackKernel       = SageAttnV3ScalePackKernel<kK0_>;
            using PackKargs        = typename PackKernel::Kargs;

            const index_t k_packed_per_head = PackKernel::KPackedPerHead(sk_pad, hdim);
            const index_t v_packed_per_head = PackKernel::VPackedPerHead(sk_pad, hdim);

            PackKargs kargs{};
            kargs.k_scale_ptr                 = args.k_scale_ptr;
            kargs.k_scale_packed_ptr          = args.k_scale_packed_ptr;
            kargs.stride_k_scale              = hdim / kG;
            kargs.nhead_stride_k_scale        = sk_pad * (hdim / kG);
            kargs.batch_stride_k_scale        = nhead_kv * sk_pad * (hdim / kG);
            kargs.nhead_stride_k_scale_packed = k_packed_per_head;
            kargs.batch_stride_k_scale_packed = nhead_kv * k_packed_per_head;
            kargs.v_scale_ptr                 = args.v_scale_ptr;
            kargs.v_scale_packed_ptr          = args.v_scale_packed_ptr;
            kargs.stride_v_scale              = sk_pad / kG;
            kargs.nhead_stride_v_scale        = hdim * (sk_pad / kG);
            kargs.batch_stride_v_scale        = nhead_kv * hdim * (sk_pad / kG);
            kargs.nhead_stride_v_scale_packed = v_packed_per_head;
            kargs.batch_stride_v_scale_packed = nhead_kv * v_packed_per_head;
            kargs.hdim                        = hdim;
            kargs.seqlen_k_padded             = sk_pad;
            kargs.nhead                       = nhead_kv;
            kargs.batch                       = batch;

            const dim3 grids  = PackKernel::GridSize(kargs);
            const dim3 blocks = PackKernel::BlockSize();
            stream_config sc{stream};
            launch_and_check(sc, make_kernel(PackKernel{}, grids, blocks, 0, kargs));
        }
    }
};

} // namespace ck_tile
