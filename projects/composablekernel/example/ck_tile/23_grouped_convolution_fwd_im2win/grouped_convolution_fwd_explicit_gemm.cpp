// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ═══════════════════════════════════════════════════════════════════════
// grouped_convolution_fwd_explicit_gemm.cpp
//
// Three-stage explicit im2col + pure GEMM convolution:
//
//   Stage 1 — ImageToColumn kernel (existing CK Tile op):
//     Reads I[N, Hi, Wi, G, C] (NHWGC) and writes fully materialised
//     A[G, M=N×Ho×Wo, K_gemm=C×Y×X] (packed row-major, Y×X innermost).
//     Each element A[g, m, k] is the explicit unrolled input value.
//     Size: G × N×Ho×Wo × C×Y×X  (can be large for big spatial dims)
//
//   Stage 2 — BatchedGemmKernel (pure GEMM, no descriptor transforms):
//     A[G, M, K_gemm]  (RowMajor)   ← materialized im2col
//     B[G, K_out, K_gemm] (ColMajor) ← weight W[G, K, C, Y, X] reshaped
//     C[G, M, K_out]   (RowMajor)   ← output O (as GEMM result)
//
//     Because both A and B are flat packed matrices, the GPU loads use
//     unit-stride reads with no index calculation overhead during the
//     GEMM loop — zero VALU ops for address computation.
//
// Hypothesis: the Stage-1 cost (write 2× memory) might be offset for
// large problems by eliminating all im2col address computation in the
// GEMM. This driver benchmarks the combined cost and compares.
//
// Weight B layout: W[G, K, C, Y, X] stored GKCYX → per group W[K, C×Y×X]
//   - Treated as ColumnMajor (N×K): stride_B = K_gemm (inner dim)
//   - batch_stride_B = K_out × K_gemm
//
// Output C: O[G, M, K_out] RowMajor, stride_C = K_out
//   - Note: this is NOT the usual NHWGK layout; G is the batch dim here.
//   - For comparison with reference, we verify against CPU conv output.
//
// GEMM tile config is selected via EXPLICIT_GEMM_CONFIG (0-5):
//   0: M128N32K64  Memory pipeline (baseline)
//   1: M64N64K64   CV3
//   2: M128N64K64  CV3
//   3: M64N64K64   CV3 Occ2
//   4: M128N64K64  CV3 Occ2
//   5: M64N128K64  CV3
// ═══════════════════════════════════════════════════════════════════════

#include <hip/hip_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/image_to_column.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"

// Arg-parsing helpers from example 20
#include "../20_grouped_convolution/grouped_convolution_utils.hpp"

// Weight transpose: GKCYX → GKYXC (to match ImageToColumn K-ordering [Y,X,C])
#include "ck_tile/ops/batched_transpose.hpp"

template <typename DataType>
static void transpose_gkcyx_to_gkyxc(const ck_tile::DeviceMem& d_src,
                                      ck_tile::DeviceMem&       d_dst,
                                      int G, int K, int C, int Y, int X)
{
    using BlockTile  = ck_tile::sequence<64, 64>;
    using WarpLayout = ck_tile::sequence<1, 1>;
    using Problem    = ck_tile::BatchedTransposeProblem<DataType, BlockTile, WarpLayout,
                                                        /*kPadM=*/true, /*kPadN=*/true>;
    using Kernel = ck_tile::BatchedTransposeKernel<ck_tile::BatchedTransposePipeline<Problem>>;

    const ck_tile::BatchedTransposeHostArgs hargs{
        d_src.GetDeviceBuffer(), d_dst.GetDeviceBuffer(),
        G * K, C, Y * X, C * Y * X,
        BlockTile::at(ck_tile::number<0>{}), BlockTile::at(ck_tile::number<1>{})};

    auto tkargs = Kernel::MakeKargs(hargs);
    ck_tile::launch_kernel(ck_tile::stream_config{},
        ck_tile::make_kernel<1>(Kernel{}, Kernel::GridSize(hargs), Kernel::BlockSize(), 0, tkargs));
}

// ── GEMM tile configs ─────────────────────────────────────────────────────────
// Mirrors the Stage-2 configs in grouped_convolution_fwd_gnchw_im2win_2stage.cpp.
// A is RowMajor [M, K_gemm], B is ColMajor [K_out, K_gemm], C is RowMajor [M, K_out].

#ifndef EXPLICIT_GEMM_CONFIG
#define EXPLICIT_GEMM_CONFIG 3   // default: CV3_M64N64K64_Occ2
#endif

#if EXPLICIT_GEMM_CONFIG == 0
struct GemmConfig {
    static constexpr ck_tile::index_t M_Tile = 128, N_Tile = 32,  K_Tile = 64;
    static constexpr ck_tile::index_t M_Warp = 4,   N_Warp = 1,  K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32, N_Warp_Tile = 32, K_Warp_Tile = 16;
    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr const char* name = "Mem_M128N32K64";
};
#elif EXPLICIT_GEMM_CONFIG == 1
struct GemmConfig {
    static constexpr ck_tile::index_t M_Tile = 64,  N_Tile = 64,  K_Tile = 64;
    static constexpr ck_tile::index_t M_Warp = 2,   N_Warp = 2,  K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32, N_Warp_Tile = 32, K_Warp_Tile = 16;
    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr const char* name = "CV3_M64N64K64";
};
#elif EXPLICIT_GEMM_CONFIG == 2
struct GemmConfig {
    static constexpr ck_tile::index_t M_Tile = 128, N_Tile = 64,  K_Tile = 64;
    static constexpr ck_tile::index_t M_Warp = 4,   N_Warp = 2,  K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32, N_Warp_Tile = 32, K_Warp_Tile = 16;
    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr const char* name = "CV3_M128N64K64";
};
#elif EXPLICIT_GEMM_CONFIG == 3
struct GemmConfig {
    static constexpr ck_tile::index_t M_Tile = 64,  N_Tile = 64,  K_Tile = 64;
    static constexpr ck_tile::index_t M_Warp = 2,   N_Warp = 2,  K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32, N_Warp_Tile = 32, K_Warp_Tile = 16;
    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr const char* name = "CV3_M64N64K64_Occ2";
};
#elif EXPLICIT_GEMM_CONFIG == 4
struct GemmConfig {
    static constexpr ck_tile::index_t M_Tile = 128, N_Tile = 64,  K_Tile = 64;
    static constexpr ck_tile::index_t M_Warp = 4,   N_Warp = 2,  K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32, N_Warp_Tile = 32, K_Warp_Tile = 16;
    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr const char* name = "CV3_M128N64K64_Occ2";
};
#elif EXPLICIT_GEMM_CONFIG == 5
struct GemmConfig {
    static constexpr ck_tile::index_t M_Tile = 64,  N_Tile = 128, K_Tile = 64;
    static constexpr ck_tile::index_t M_Warp = 2,   N_Warp = 4,  K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32, N_Warp_Tile = 32, K_Warp_Tile = 16;
    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr const char* name = "CV3_M64N128K64";
};
#else
#error "Unknown EXPLICIT_GEMM_CONFIG — valid range: 0..5"
#endif

// ── PipelineTypeTraits for BatchedGemmKernel (mirrors example 16) ─────────────
template <ck_tile::GemmPipeline PipelineId>
struct GemmPipelineTraits;

template <>
struct GemmPipelineTraits<ck_tile::GemmPipeline::MEMORY>
{
    template <typename P>
    using Pipeline = ck_tile::GemmPipelineAgBgCrMem<P>;
};

template <>
struct GemmPipelineTraits<ck_tile::GemmPipeline::COMPUTE_V3>
{
    template <typename P>
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompV3<P>;
};

// ── Stage-1: ImageToColumn launcher ──────────────────────────────────────────
// Materializes A[G, M=N×Ho×Wo, K_gemm=C×Y×X] from I[N,Hi,Wi,G,C] (NHWGC).
// Uses kPadM=false, kPadN=false (adjust if M or K_gemm is not tile-aligned).
template <typename DataType>
float launch_stage1_im2col(
    const void*                       in_ptr,      // I[N,Hi,Wi,G,C] (NHWGC)
    void*                             out_ptr,     // A[G, M, K_padded] (row-padded)
    const ck_tile::conv::ConvParam&   p,
    ck_tile::index_t                  stride_A,    // padded row stride (K_padded)
    int n_warmup, int n_repeat)
{
    // ImageToColumn block shape: 256 M rows × 32 K columns per block
    using BlockTile = ck_tile::sequence<256, 32>;
    using Problem   = ck_tile::BlockImageToColumnProblem<DataType, DataType,
                           ck_tile::TileImageToColumnShape<
                               ck_tile::sequence<4, 4>,
                               ck_tile::sequence<64, 4>,
                               BlockTile>,
                           2,   // NDimSpatial
                           8,   // AlignIn  (C divisible by 8)
                           8>;  // AlignOut

    using Kernel = ck_tile::ImageToColumn<Problem>;

    const ck_tile::index_t G  = static_cast<ck_tile::index_t>(p.G_);
    const ck_tile::index_t N  = static_cast<ck_tile::index_t>(p.N_);
    const ck_tile::index_t C  = static_cast<ck_tile::index_t>(p.C_);
    const ck_tile::index_t Hi = static_cast<ck_tile::index_t>(p.input_spatial_lengths_[0]);
    const ck_tile::index_t Wi = static_cast<ck_tile::index_t>(p.input_spatial_lengths_[1]);
    const ck_tile::index_t Ho = static_cast<ck_tile::index_t>(p.output_spatial_lengths_[0]);
    const ck_tile::index_t Wo = static_cast<ck_tile::index_t>(p.output_spatial_lengths_[1]);
    const ck_tile::index_t Y  = static_cast<ck_tile::index_t>(p.filter_spatial_lengths_[0]);
    const ck_tile::index_t X  = static_cast<ck_tile::index_t>(p.filter_spatial_lengths_[1]);

    const ck_tile::index_t GemmM    = N * Ho * Wo;
    const ck_tile::index_t GemmK    = C * Y * X;

    // NHWGC strides (per-group view uses [N, H, W, C] with strides including G)
    const ck_tile::long_index_t NStride  = Hi * Wi * G * C;
    const ck_tile::long_index_t HiStride = Wi * G * C;
    const ck_tile::long_index_t WiStride = G * C;
    const ck_tile::long_index_t CStride  = 1;
    const ck_tile::long_index_t GStride  = C;  // per-group pointer offset

    // image_g_n_c_wis_strides: [G_stride, N_stride, C_stride, Hi_stride, Wi_stride]
    // (G_stride is the per-group byte offset; ImageToColumn multiplies by iBatch)
    const ck_tile::array<ck_tile::long_index_t, 5> img_strides = {
        GStride, NStride, CStride, HiStride, WiStride};

    // Output A[G, M, K_padded] with padded row stride for vector alignment.
    //   batch_stride_A = M * stride_A,  stride_A (M stride) = K_padded,  K stride = 1
    const ck_tile::array<ck_tile::long_index_t, 3> gemm_strides = {
        static_cast<ck_tile::long_index_t>(GemmM) * stride_A,   // G stride
        static_cast<ck_tile::long_index_t>(stride_A),            // M stride (padded row)
        static_cast<ck_tile::long_index_t>(1)};                   // K stride (innermost)

    auto kargs = Kernel::MakeKargs(
        in_ptr,
        out_ptr,
        G, N, C,
        {Hi, Wi},
        {Y, X},
        {Ho, Wo},
        img_strides,
        gemm_strides,
        {static_cast<ck_tile::long_index_t>(p.conv_filter_strides_[0]),
         static_cast<ck_tile::long_index_t>(p.conv_filter_strides_[1])},
        {static_cast<ck_tile::long_index_t>(p.conv_filter_dilations_[0]),
         static_cast<ck_tile::long_index_t>(p.conv_filter_dilations_[1])},
        {static_cast<ck_tile::long_index_t>(p.input_left_pads_[0]),
         static_cast<ck_tile::long_index_t>(p.input_left_pads_[1])},
        {static_cast<ck_tile::long_index_t>(p.input_right_pads_[0]),
         static_cast<ck_tile::long_index_t>(p.input_right_pads_[1])});

    const dim3 grids  = Kernel::GridSize(GemmM, GemmK, G);
    const dim3 blocks = Kernel::BlockSize();

    return ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat},
        ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));
}

// ── Stage-2: BatchedGemmKernel launcher ──────────────────────────────────────
// Pure GEMM: no descriptor transforms during the GEMM loop.
//
// A[G, M, K_gemm]   RowMajor    stride_A = K_gemm
// B[G, N=K_out, K_gemm] ColMajor  stride_B = K_gemm  (B^T is [K_gemm, K_out])
// C[G, M, N=K_out]  RowMajor    stride_C = K_out
//
// B is the GKCYX weight viewed as [G, K_out, K_gemm] with K_gemm = C*Y*X innermost.
// In GKCYX format: W[g, k, c, y, x] at offset g*K*K_gemm + k*K_gemm + c*Y*X + y*X + x.
// So B is naturally ColMajor in the sense that B[k, k_gemm] has stride_B=K_gemm.
template <typename DataType>
float launch_stage2_pure_gemm(
    const void*                     a_ptr,      // A[G, M, K_padded] RowMajor (from Stage-1)
    const void*                     b_ptr,      // W[G, K_out, K_gemm] ColMajor = GKCYX
    void*                           c_ptr,      // C[G, M, K_out] RowMajor
    ck_tile::index_t G,
    ck_tile::index_t M,
    ck_tile::index_t N,         // = K_out (output channels)
    ck_tile::index_t K_gemm,    // = C*Y*X (actual reduction dimension)
    ck_tile::index_t stride_A,  // padded row stride for A (≥ K_gemm, multiple of VecAlign)
    int n_warmup, int n_repeat)
{
    // GEMM: C[M, N=K_out] = Σ_k A[M, k] × W[N=K_out, k]
    //
    // A: RowMajor [M, K_gemm]  — im2col matrix, K_gemm innermost, stride_A=K_padded
    //
    // B: ColumnMajor, represents W_gkyxc[K_out, K_gemm] (GKYXC weight, K_gemm innermost)
    //    ColMajor B[K_gemm, K_out]: B[k,n] = ptr[k + n*K_gemm] = W_gkyxc[n, k]  ✓
    //    stride_B = K_gemm (leading dimension for ColMajor)
    //    GEMM computes: C[m,n] = Σ_k A[m,k] × B[k,n] = Σ_k A[m,k] × W_gkyxc[n,k]  ✓
    //    K ordering: ImageToColumn merge([Y,X,C]) → k=y*X*C+x*C+c
    //                GKYXC W[K_out, Y, X, C] → k_gemm = y*X*C+x*C+c  ✓ (same ordering)
    //
    // C: RowMajor [M, N=K_out], stride_C=N
    using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;

    // Pad M and N to handle non-tile-multiples
    constexpr bool kPadM = true;
    constexpr bool kPadN = true;
    constexpr bool kPadK = true;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<
        kPadM, kPadN, kPadK,
        GemmConfig::DoubleSmemBuffer,
        ALayout, BLayout, CLayout>;

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
        DataType, DataType, float,
        GemmShape,
        GemmUniversalTraits,
        GemmConfig::Scheduler>;

    using GemmPipeline = typename GemmPipelineTraits<GemmConfig::Pipeline>::template Pipeline<
        UniversalGemmProblem>;

    // VectorSizeC=4: works for both small K_out=4 and large K_out=256 (both % 4 == 0).
    using GemmEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
        DataType, DataType,
        ck_tile::tuple<>,
        float,
        DataType,
        ck_tile::tuple<>,
        CLayout,
        ck_tile::element_wise::PassThrough,
        TilePartitioner::MPerBlock,
        TilePartitioner::NPerBlock,
        GemmConfig::M_Warp,
        GemmConfig::N_Warp,
        GemmConfig::M_Warp_Tile,
        GemmConfig::N_Warp_Tile,
        GemmConfig::K_Warp_Tile,
        false,   // TransposeC
        1,       // NumWaveGroups
        true,    // FixedVectorSize
        4>>;     // VectorSizeC

    using Kernel = ck_tile::BatchedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

    // Strides:
    //   A[G, M, K_gemm] RowMajor with padded row:   stride_A (padded), batch_stride_A = M*stride_A
    //   B[G, K_out, K_gemm] ColMajor: stride_B = K_gemm (leading dim), batch_stride_B = N*K_gemm
    //   C[G, M, K_out] RowMajor:    stride_C = N=K_out,  batch_stride_C = M*N
    const ck_tile::index_t stride_B = K_gemm;   // ColMajor B: leading dim = K_gemm (rows in ColMajor [K_gemm, K_out])
    const ck_tile::index_t stride_C = N;         // RowMajor
    const ck_tile::index_t batch_stride_A = M * stride_A;
    const ck_tile::index_t batch_stride_B = N * K_gemm;
    const ck_tile::index_t batch_stride_C = M * N;

    ck_tile::BatchedGemmHostArgs hargs(
        a_ptr, b_ptr, c_ptr,
        /*k_batch=*/1,
        M, N, K_gemm,
        stride_A, stride_B, stride_C,
        batch_stride_A, batch_stride_B, batch_stride_C,
        /*batch_count=*/G);
    // Note: stride_A = K_padded (padded row stride), not K_gemm.
    // The GEMM uses K_gemm for computation but reads with stride_A for pointer arithmetic.

    auto kargs = Kernel::MakeKernelArgs(hargs);

    const dim3 grids  = Kernel::GridSize(M, N, /*KBatch=*/1, G);
    const dim3 blocks = Kernel::BlockSize();

    if(!Kernel::IsSupportedArgument(kargs))
        throw std::runtime_error("Stage-2 pure GEMM: arguments not supported.\n");

    return ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat},
        ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    try
    {
        auto [result, arg_parser] = create_args(argc, argv);
        if(!result) return -1;

        const std::string data_type = arg_parser.get_str("prec");
        if(data_type != "fp16" && data_type != "bf16")
        {
            std::cerr << "Unsupported precision: " << data_type << "\n";
            return 1;
        }
        const bool use_fp16 = (data_type == "fp16");

        std::vector<ck_tile::index_t> filter_sp, image_sp, strides, dilations, lpads, rpads;
        const ck_tile::index_t num_dim_sp =
            fill_spatial_dimensions(filter_sp, image_sp, strides, dilations, lpads, rpads, arg_parser);

        ck_tile::conv::ConvParam p{num_dim_sp,
            arg_parser.get_int("g"), arg_parser.get_int("n"),
            arg_parser.get_int("k"), arg_parser.get_int("c"),
            filter_sp, image_sp, strides, dilations, lpads, rpads};

        const int n_warmup = arg_parser.get_int("warmup");
        const int n_repeat = arg_parser.get_int("repeat");
        const int verify   = arg_parser.get_int("v");

        const ck_tile::index_t G      = static_cast<ck_tile::index_t>(p.G_);
        const ck_tile::index_t N      = static_cast<ck_tile::index_t>(p.N_);
        const ck_tile::index_t C      = static_cast<ck_tile::index_t>(p.C_);
        const ck_tile::index_t K      = static_cast<ck_tile::index_t>(p.K_);
        const ck_tile::index_t Hi     = static_cast<ck_tile::index_t>(p.input_spatial_lengths_[0]);
        const ck_tile::index_t Wi     = static_cast<ck_tile::index_t>(p.input_spatial_lengths_[1]);
        const ck_tile::index_t Ho     = static_cast<ck_tile::index_t>(p.output_spatial_lengths_[0]);
        const ck_tile::index_t Wo     = static_cast<ck_tile::index_t>(p.output_spatial_lengths_[1]);
        const ck_tile::index_t Y      = static_cast<ck_tile::index_t>(p.filter_spatial_lengths_[0]);
        const ck_tile::index_t X      = static_cast<ck_tile::index_t>(p.filter_spatial_lengths_[1]);

        const ck_tile::index_t GemmM  = N * Ho * Wo;
        const ck_tile::index_t GemmN  = K;          // output channels
        const ck_tile::index_t GemmK  = C * Y * X;  // reduction dimension

        // Pad GemmK to the nearest multiple of 8 (fp16 vector load size) to prevent
        // out-of-bounds reads in the last vector load of each A row. The GEMM kernel
        // is launched with the actual GemmK (unpadded) but the A matrix has extra zero
        // columns appended so each row is vector-aligned. Padding zeros do not affect
        // the sum because kPadK=true masks them in LDS.
        constexpr ck_tile::index_t VecAlignK = 8;
        const ck_tile::index_t GemmK_padded  = ((GemmK + VecAlignK - 1) / VecAlignK) * VecAlignK;
        const ck_tile::index_t stride_A_padded = GemmK_padded; // row stride for A matrix

        const size_t elem_bytes  = use_fp16 ? sizeof(ck_tile::half_t) : sizeof(ck_tile::bf16_t);
        const size_t in_size     = static_cast<size_t>(N) * Hi * Wi * G * C;   // NHWGC
        // A allocated with padded stride so each row is 8-element aligned.
        const size_t a_size      = static_cast<size_t>(G) * GemmM * GemmK_padded;
        const size_t wei_size    = static_cast<size_t>(G) * K * C * Y * X;     // GKCYX
        const size_t out_size    = static_cast<size_t>(G) * GemmM * GemmN;     // C[G,M,N]

        const double a_gb = static_cast<double>(a_size) * elem_bytes / 1e9;
        std::cout << "Explicit GEMM fwd conv\n"
                  << "  G=" << G << " N=" << N << " C=" << C << " K=" << K
                  << " Hi=" << Hi << " Wi=" << Wi << " Ho=" << Ho << " Wo=" << Wo
                  << " Y=" << Y << " X=" << X << "\n"
                  << "  GEMM: M=" << GemmM << " N=" << GemmN << " K=" << GemmK << "\n"
                  << "  A (im2col) size: " << G << "×" << GemmM << "×" << GemmK
                  << " = " << a_gb << " GB fp" << (use_fp16 ? 16 : 16) << "\n"
                  << "  GEMM config: " << GemmConfig::name << "\n";

        if(a_gb > 8.0)
        {
            std::cerr << "  WARNING: im2col A matrix is very large (" << a_gb << " GB).\n";
        }

        using half_t = ck_tile::half_t;
        using bf16_t = ck_tile::bf16_t;

        // Host tensors (NHWGC input, GKCYX weight, GNKHW output for reference)
        const auto in_nhwgc_desc = ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<
            ck_tile::tensor_layout::convolution::NHWGC>(p);
        const auto wei_gkcyx_desc = ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
            ck_tile::tensor_layout::convolution::GKCYX>(p);
        // For NHWGK output (reference), convert from GKCYX weight
        const auto wei_gkyxc_desc = ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
            ck_tile::tensor_layout::convolution::GKYXC>(p);
        const auto out_nhwgk_desc = ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<
            ck_tile::tensor_layout::convolution::NHWGK>(p);

        ck_tile::HostTensor<half_t> input_f16(in_nhwgc_desc),  weight_f16(wei_gkcyx_desc);
        ck_tile::HostTensor<bf16_t> input_bf16(in_nhwgc_desc), weight_bf16(wei_gkcyx_desc);

        if(use_fp16)
        {
            ck_tile::FillUniformDistribution<half_t>{-2.f, 2.f}(input_f16);
            ck_tile::FillUniformDistribution<half_t>{-2.f, 2.f}(weight_f16);
        }
        else
        {
            ck_tile::FillUniformDistribution<bf16_t>{-2.f, 2.f}(input_bf16);
            ck_tile::FillUniformDistribution<bf16_t>{-2.f, 2.f}(weight_bf16);
        }

        ck_tile::DeviceMem input_buf(in_size   * elem_bytes);
        ck_tile::DeviceMem a_buf(a_size         * elem_bytes);  // im2col A matrix
        ck_tile::DeviceMem weight_buf(wei_size  * elem_bytes);  // GKCYX (original)
        ck_tile::DeviceMem weight_gkyxc_buf(wei_size * elem_bytes); // GKYXC (transposed)
        ck_tile::DeviceMem output_buf(out_size  * elem_bytes);  // C[G, M, K_out]

        if(use_fp16) { input_buf.ToDevice(input_f16.data()); weight_buf.ToDevice(weight_f16.data()); }
        else         { input_buf.ToDevice(input_bf16.data()); weight_buf.ToDevice(weight_bf16.data()); }
        a_buf.SetZero();
        output_buf.SetZero();

        // Transpose GKCYX → GKYXC to match ImageToColumn K-ordering [Y, X, C].
        // ImageToColumn produces K = merge([Y, X, C]), so the weight must have the same K ordering.
        if(use_fp16)
            transpose_gkcyx_to_gkyxc<ck_tile::half_t>(weight_buf, weight_gkyxc_buf, G, K, C, Y, X);
        else
            transpose_gkcyx_to_gkyxc<ck_tile::bf16_t>(weight_buf, weight_gkyxc_buf, G, K, C, Y, X);

        // ── Stage 1: I → A (full im2col materialization) ─────────────────────
        float t1;
        if(use_fp16)
            t1 = launch_stage1_im2col<half_t>(
                input_buf.GetDeviceBuffer(), a_buf.GetDeviceBuffer(), p,
                stride_A_padded, n_warmup, n_repeat);
        else
            t1 = launch_stage1_im2col<bf16_t>(
                input_buf.GetDeviceBuffer(), a_buf.GetDeviceBuffer(), p,
                stride_A_padded, n_warmup, n_repeat);

        const size_t s1_bytes = (in_size + a_size) * elem_bytes;
        std::cout << "  Stage 1 (I→A): " << t1 << " ms  "
                  << s1_bytes / 1.e6f / t1 << " GB/s\n";

        // ── Stage 2: pure GEMM A×B^T → C ─────────────────────────────────────
        output_buf.SetZero();
        float t2;
        // Stage-2 uses GKYXC weight (transposed) so K-ordering matches ImageToColumn [Y,X,C].
        if(use_fp16)
            t2 = launch_stage2_pure_gemm<half_t>(
                a_buf.GetDeviceBuffer(),
                weight_gkyxc_buf.GetDeviceBuffer(),   // GKYXC: K=[Y,X,C] ✓
                output_buf.GetDeviceBuffer(),
                G, GemmM, GemmN, GemmK,
                stride_A_padded, n_warmup, n_repeat);
        else
            t2 = launch_stage2_pure_gemm<bf16_t>(
                a_buf.GetDeviceBuffer(),
                weight_gkyxc_buf.GetDeviceBuffer(),
                output_buf.GetDeviceBuffer(),
                G, GemmM, GemmN, GemmK,
                stride_A_padded, n_warmup, n_repeat);

        // Compute FLOPs: 2 * G * M * N * K_gemm
        const size_t flop = 2ULL * G * GemmM * GemmN * GemmK;
        const size_t s2bytes = (a_size + wei_size + out_size) * elem_bytes;
        std::cout << "  Stage 2 (A×B→C): " << t2 << " ms  "
                  << static_cast<float>(flop) / 1.e9f / t2 << " TFlops  "
                  << s2bytes / 1.e6f / t2 << " GB/s\n"
                  << "  Combined:        " << t1 + t2 << " ms\n";

        // ── Verification ─────────────────────────────────────────────────────
        // The output C[G, M, K_out] RowMajor is NOT the standard conv output layout.
        // For verification we compare against the CPU reference on the same problem.
        // The output C[g, m, k] = C[g, n*Ho*Wo+ho*Wo+wo, k_out] which corresponds
        // to O[n, ho, wo, g, k_out] in NHWGK format (just reindexed).
        if(verify != 0)
        {
            std::cout << "  Running CPU reference (NHWGC/GKYXC → NHWGK)...\n";

            // Need GKYXC weight for reference kernel (CPU ref uses GKCYX)
            // For CPU ref, we use reference_grouped_conv_fwd which expects GNCHW + GKCYX
            // But our input is NHWGC — need to reorder for reference
            // Simplest: use reference on GNCHW (reorder NHWGC → GNCHW on host)

            // Reorder NHWGC → GNCHW
            const auto in_gnchw_desc = ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<
                ck_tile::tensor_layout::convolution::GNCHW>(p);
            ck_tile::HostTensor<half_t> input_gnchw_f16(in_gnchw_desc);
            ck_tile::HostTensor<bf16_t> input_gnchw_bf16(in_gnchw_desc);

            // Manual NHWGC → GNCHW reorder
            if(use_fp16)
            {
                for(int g=0; g<G; g++)
                    for(int n=0; n<N; n++)
                        for(int c=0; c<C; c++)
                            for(int h=0; h<Hi; h++)
                                for(int w=0; w<Wi; w++)
                                {
                                    int si = n*(Hi*Wi*G*C) + h*(Wi*G*C) + w*(G*C) + g*C + c;
                                    int di = g*(N*C*Hi*Wi) + n*(C*Hi*Wi) + c*(Hi*Wi) + h*Wi + w;
                                    input_gnchw_f16.mData[di] = input_f16.mData[si];
                                }
            }

            const auto out_nhwgk_desc_ref = ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<
                ck_tile::tensor_layout::convolution::NHWGK>(p);

            ck_tile::HostTensor<half_t> oref_f16(out_nhwgk_desc_ref);
            ck_tile::HostTensor<bf16_t> oref_bf16(out_nhwgk_desc_ref);
            oref_f16.SetZero();

            if(use_fp16)
                ck_tile::reference_grouped_conv_fwd<2, half_t, half_t, half_t>(
                    input_gnchw_f16, weight_f16, oref_f16,
                    p.conv_filter_strides_, p.conv_filter_dilations_,
                    p.input_left_pads_, p.input_right_pads_);

            // The Stage-2 output C[G, M=N×Ho×Wo, K_out] RowMajor should match
            // O_ref[n, ho, wo, g, k] (NHWGK) when properly indexed.
            // For comparison: C[g, m=n*Ho*Wo+ho*Wo+wo, k] = O_ref[n, ho, wo, g, k]
            // Both have size G × M × K_out — just check element-wise after reindexing.

            if(use_fp16)
            {
                ck_tile::HostTensor<half_t> odev_gemm({static_cast<std::size_t>(G),
                                                        static_cast<std::size_t>(GemmM),
                                                        static_cast<std::size_t>(GemmN)});
                output_buf.FromDevice(odev_gemm.data());

                // Convert C[G,M,K_out] → NHWGK ordering for comparison with reference
                ck_tile::HostTensor<half_t> odev_nhwgk(out_nhwgk_desc_ref);
                for(int g=0; g<G; g++)
                    for(int n=0; n<N; n++)
                        for(int ho=0; ho<Ho; ho++)
                            for(int wo=0; wo<Wo; wo++)
                                for(int k=0; k<K; k++)
                                {
                                    int m  = n*(Ho*Wo) + ho*Wo + wo;
                                    int gi = g*(GemmM*GemmN) + m*GemmN + k; // C[g,m,k]
                                    int oi = n*(Ho*Wo*G*K) + ho*(Wo*G*K) + wo*(G*K) + g*K + k; // NHWGK
                                    odev_nhwgk.mData[oi] = odev_gemm.mData[gi];
                                }

                const float max_val = *std::max_element(oref_f16.mData.begin(), oref_f16.mData.end());
                const ck_tile::index_t GemmK_idx = static_cast<ck_tile::index_t>(C * Y * X);
                const auto ta = calculate_rtol_atol<half_t, half_t, float, half_t>(GemmK_idx, 1, max_val);
                const bool pass = ck_tile::check_err(odev_nhwgk, oref_f16, "Error: incorrect results!",
                    ta.at(ck_tile::number<0>{}), ta.at(ck_tile::number<1>{}));
                std::cout << "  verification: " << (pass ? "PASSED" : "FAILED") << "\n";
                return pass ? 0 : 1;
            }
        }

        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
