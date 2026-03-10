// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/sageattn_preprocess/pipeline/sageattn_preprocess_pipeline.hpp"
#include "ck_tile/ops/sageattn_preprocess/kernel/sageattn_preprocess_kernel.hpp"

#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile {

// ============================================================================
// sageattn_preprocess_run
//
// Host-side wrapper that launches the three kernels required for SA3
// preprocessing on a single HIP stream:
//
//   Launch 0: SageAttnKMeanKernel
//     Computes k_mean[b, h, d] = mean_n(K[b,h,n,d]), stored as InputT.
//     Grid: (num_k_tiles, nhead, batch), BlockSize: kCols.
//     Requires k_mean_partial_buf (float) and counter_buf (int32) to be
//     zero-initialised before launch (done here via hipMemsetAsync).
//
//   Launch 1: SageAttnPreprocessKernel
//     Q: mean → quantize (MXFP4, stores q_mean as InputT)
//     K: smooth (K' = K - k_mean) → k_prime_buf; quantize K' → MXFP4
//     V: transpose + quantize → MXFP4
//     Grid: (max(num_q_tiles, num_k_tiles, hdim), nhead, batch).
//
//   Launch 2: BatchedGemmKernel
//     delta_s = q_mean @ K'^T
//     A: q_mean  [batch*nhead, num_q_tiles, hdim] InputT  RowMajor
//     B: K'      [batch*nhead, seqlen_k,    hdim] InputT  ColMajor (= K'^T)
//     C: delta_s [batch*nhead, num_q_tiles, seqlen_k] float RowMajor
//
// Template parameters:
//   InputT:  fp16_t or bf16_t (matches Q / K / V element type)
//   kRows:   tile rows for Q and K (kM0 == kN0)
//   kCols:   hdim
//
// delta_s strides are derived from M=num_q_tiles, N=seqlen_k (no stride params needed).
//
// Caller-allocated device buffers (all on the same device as Q/K/V):
//   k_mean_buf:         InputT [batch, nhead, hdim]
//   k_prime_buf:        InputT [batch, nhead, seqlen_k, hdim]
//   k_mean_partial_buf: float  [batch, nhead, hdim]   (scratch, not consumed after run)
//   counter_buf:        int32  [batch, nhead]          (scratch, not consumed after run)
// ============================================================================

// GEMM tile configuration for delta_s = q_mean @ K'^T.
//   M = num_q_tiles (small, typically 1–64)
//   N = seqlen_k    (large)
//   K = hdim        (128 or 256)
//
// BlockTile<32,64,32>, BlockWarps<1,2,1>: MWarp=1, NWarp=2 → 128 threads.
// WarpTile K dimension differs by input type:
//   fp16/bf16: WarpTile<32,32,16> → mfma_f32_32x32x8f16 × 2 iters  (kK=16)
//   float:     WarpTile<32,32, 8> → mfma_f32_32x32x2f32 × 4 iters  (kK= 8)
// kPadM=true to handle num_q_tiles not divisible by 32.
template <typename InputT>
using DeltaSGemmShape = std::conditional_t<
    std::is_same_v<InputT, float>,
    TileGemmShape<sequence<32, 64, 32>, sequence<1, 2, 1>, sequence<32, 32,  8>>,
    TileGemmShape<sequence<32, 64, 32>, sequence<1, 2, 1>, sequence<32, 32, 16>>>;

template <typename InputT, index_t kRows, index_t kCols>
void sageattn_preprocess_run(
    // InputT: fp16_t, bf16_t, or float.
    // ---- preprocess args ----
    const SageAttnPreprocessHostArgs& prep_args,
    // ---- delta_s output ----
    float*      delta_s_ptr,   // [batch, nhead, num_q_tiles, seqlen_k]
    // ---- caller-allocated scratch / output buffers ----
    InputT*  k_mean_buf,            // [batch, nhead, hdim]          InputT
    InputT*  k_prime_buf,           // [batch, nhead, seqlen_k, hdim] InputT
    float*   k_mean_partial_buf,    // [batch, nhead, hdim]          float  (scratch)
    int32_t* counter_buf,           // [batch, nhead]                int32  (scratch)
    // ---- stream ----
    hipStream_t stream)
{
    const index_t batch  = prep_args.batch;
    const index_t nhead  = prep_args.nhead;
    const index_t hdim   = prep_args.hdim;
    const index_t seqlen_k    = prep_args.seqlen_k;
    const index_t num_q_tiles = prep_args.num_q_tiles;
    const index_t num_k_tiles = prep_args.num_k_tiles;

    // ------------------------------------------------------------------ //
    // Launch 0: k_mean kernel
    // ------------------------------------------------------------------ //
    (void)hipMemsetAsync(k_mean_partial_buf, 0, batch * nhead * hdim * sizeof(float), stream);
    (void)hipMemsetAsync(counter_buf,        0, batch * nhead * sizeof(int32_t),       stream);

    {
        using KMeanKernel = SageAttnKMeanKernel<InputT, kRows, kCols>;
        SageAttnKMeanHostArgs kmean_hargs{};
        kmean_hargs.k_ptr               = prep_args.k_ptr;
        kmean_hargs.seqlen_k            = seqlen_k;
        kmean_hargs.hdim                = hdim;
        kmean_hargs.stride_k            = prep_args.stride_k;
        kmean_hargs.nhead_stride_k      = prep_args.nhead_stride_k;
        kmean_hargs.batch_stride_k      = prep_args.batch_stride_k;
        kmean_hargs.k_mean_partial_ptr  = k_mean_partial_buf;
        kmean_hargs.k_mean_ptr          = k_mean_buf;
        kmean_hargs.counter_ptr         = counter_buf;
        kmean_hargs.nhead_stride_kmean  = hdim;
        kmean_hargs.batch_stride_kmean  = nhead * hdim;
        kmean_hargs.num_k_tiles         = num_k_tiles;
        kmean_hargs.nhead               = nhead;
        kmean_hargs.batch               = batch;

        auto kargs         = KMeanKernel::MakeKargs(kmean_hargs);
        const dim3 grids   = KMeanKernel::GridSize(kmean_hargs);
        const dim3 blocks  = KMeanKernel::BlockSize();
        const index_t smem = KMeanKernel::GetSmemSize();

        stream_config sc{stream};
        launch_and_check(sc, make_kernel(KMeanKernel{}, grids, blocks, smem, kargs));
    }

    // ------------------------------------------------------------------ //
    // Launch 1: preprocess kernel (Q + K' + V)
    // ------------------------------------------------------------------ //
    {
        // Wire k_mean and k_prime into the preprocess host args.
        // K' is stored in natural layout: [batch, nhead, seqlen_k, hdim] row-major.
        // stride_k_prime = hdim (row stride within the [seqlen_k, hdim] matrix).
        SageAttnPreprocessHostArgs prep = prep_args;
        prep.k_mean_ptr           = k_mean_buf;
        prep.k_prime_ptr          = k_prime_buf;
        prep.stride_k_prime       = hdim;
        prep.nhead_stride_k_prime = seqlen_k * hdim;
        prep.batch_stride_k_prime = nhead * seqlen_k * hdim;

        using PrepKernel = SageAttnPreprocessKernel<InputT, kRows, kCols>;
        auto kargs         = PrepKernel::MakeKargs(prep);
        const dim3 grids   = PrepKernel::GridSize(prep);
        const dim3 blocks  = PrepKernel::BlockSize();
        const index_t smem = PrepKernel::GetSmemSize();

        stream_config sc{stream};
        launch_and_check(sc, make_kernel(PrepKernel{}, grids, blocks, smem, kargs));
    }

    // ------------------------------------------------------------------ //
    // Launch 2: batched GEMM  delta_s = q_mean @ K'^T
    // ------------------------------------------------------------------ //
    {
        using ALayout = tensor_layout::gemm::RowMajor;
        // K' stored naturally as [seqlen_k=N, hdim=K] row-major → ColMajor B[K, N] → K'^T ✓
        using BLayout = tensor_layout::gemm::ColumnMajor;
        using CLayout = tensor_layout::gemm::RowMajor;

        constexpr bool kPadM           = true;  // num_q_tiles may not be multiple of M_Tile
        constexpr bool kPadN           = true;  // seqlen_k may not be multiple of N_Tile
        constexpr bool kPadK           = false; // hdim is always multiple of K_Tile
        constexpr bool DoubleSmemBuf   = false;
        constexpr bool TransposeC      = false;

        using GemmTraits = TileGemmUniversalTraits<kPadM, kPadN, kPadK,
                                                   DoubleSmemBuf,
                                                   ALayout, BLayout, CLayout,
                                                   TransposeC>;

        using Shape = DeltaSGemmShape<InputT>;

        using GemmProblem = UniversalGemmPipelineProblem<InputT,  // ADataType
                                                         InputT,  // BDataType
                                                         float,   // AccDataType
                                                         Shape,
                                                         GemmTraits,
                                                         GemmPipelineScheduler::Intrawave>;

        using GemmPipeline = GemmPipelineAgBgCrCompV3<GemmProblem>;

        using GemmEpilogue = CShuffleEpilogue<
            CShuffleEpilogueProblem<InputT,   // ADataType
                                    InputT,   // BDataType
                                    tuple<>,  // DsDataType (no D)
                                    float,    // AccDataType
                                    float,    // CDataType (delta_s is float)
                                    tuple<>,  // DsLayout
                                    CLayout,
                                    element_wise::PassThrough,
                                    Shape::kM,
                                    Shape::kN,
                                    /*MWarp=*/1,
                                    /*NWarp=*/2,
                                    /*MWarpTile=*/32,
                                    /*NWarpTile=*/32,
                                    Shape::WarpTile::at(number<2>{}),  // 16 for fp16/bf16, 8 for float
                                    TransposeC>>;

        using GemmTilePartitioner = GemmSpatiallyLocalTilePartitioner<Shape,
                                                                       /*GroupNum=*/8,
                                                                       /*M01=*/4>;
        using GemmKernel = BatchedGemmKernel<GemmTilePartitioner, GemmPipeline, GemmEpilogue>;

        // M = num_q_tiles, N = seqlen_k, K = hdim, batch_count = batch * nhead
        const index_t M            = num_q_tiles;
        const index_t N            = seqlen_k;
        const index_t K            = hdim;
        const index_t batch_count  = batch * nhead;

        // A: q_mean  [batch*nhead, num_q_tiles, hdim]  row-major, stride=hdim
        // B: K'      [batch*nhead, seqlen_k,   hdim]  col-major (leading dim = hdim = K)
        //   ColMajor B[K,N] where K'[n,k] = B[k,n] → GEMM computes q_mean @ K'^T ✓
        // C: delta_s [batch*nhead, num_q_tiles, seqlen_k]  row-major, stride=seqlen_k
        BatchedGemmHostArgs gemm_hargs(
            prep_args.q_mean_ptr,                     // A ptr
            k_prime_buf,                              // B ptr (K')
            delta_s_ptr,                              // C ptr
            /*k_batch=*/1,
            M, N, K,
            /*stride_A=*/K,                           // q_mean row stride = hdim
            /*stride_B=*/K,                           // ColMajor leading dim = hdim (= K)
            /*stride_C=*/N,                           // delta_s row stride = seqlen_k
            /*batch_stride_A=*/M * K,                 // num_q_tiles * hdim
            /*batch_stride_B=*/N * K,                 // seqlen_k * hdim
            /*batch_stride_C=*/M * N,                 // num_q_tiles * seqlen_k
            batch_count);

        auto kargs       = GemmKernel::MakeKernelArgs(gemm_hargs);
        const dim3 grids = GemmKernel::GridSize(M, N, /*k_batch=*/1, batch_count);
        const dim3 blocks = GemmKernel::BlockSize();
        // BatchedGemmKernel uses statically-declared __shared__ arrays; dynamic smem = 0.
        stream_config sc{stream};
        launch_and_check(sc, make_kernel(GemmKernel{}, grids, blocks, 0, kargs));
    }
}

} // namespace ck_tile
