// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ═══════════════════════════════════════════════════════════════════════
// grouped_convolution_fwd_gnchw_im2win_2stage.cpp
//
// Two-stage im2win forward convolution:
//
//   Stage 1 — ImageToIm2win kernel:
//     GNCHW: I[G,N,C,Hi,Wi] → I'[G,N,C,Ho,Wi_pad,Y]  (packed, Y innermost)
//     NHWGC: I[N,Hi,Wi,G,C] → I'[G,N,C,Ho,Wi_pad,Y]  (same output format)
//     Formula: I'[n,c,ho,wi_pad,y] = I[n,c, ho·Sy+y·Dy-LPH, wi_pad-LPW]
//
//   Stage 2 — Standard im2col forward conv kernel:
//     A = I' via GNCHW_Im2win descriptor (X window applied lazily)
//     B = W[G,K,C,Y,X] (GKCYX)
//     C = O  (NHWGK, K unit-stride)
//
// Supported layouts:
//   -in_layout GNCHW  -wei_layout GKCYX  -out_layout NHWGK
//   -in_layout NHWGC  -wei_layout GKYXC  -out_layout NHWGK
//
// Stage-2 tile is selected at compile time via IM2WIN_STAGE2_CONFIG (0-5):
//   0: M128N32K64   Memory pipeline, 4×1 warps, 32×32×16 MFMA  (small-K baseline)
//   1: M64N64K64    CV3,    2×2 warps, 32×32×16 MFMA            (best for large K)
//   2: M128N64K64   CV3,    2×2 warps, 32×32×16 MFMA
//   3: M64N64K64    CV3,    2×2 warps, 32×32×16 MFMA, Occ2=2   (higher occupancy)
//   4: M128N64K64   CV3,    2×2 warps, 32×32×16 MFMA, Occ2=2
//   5: M64N128K64   CV3,    2×4 warps, 32×32×16 MFMA            (N_Tile=128 for K=256)
// ═══════════════════════════════════════════════════════════════════════

#include <hip/hip_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ck_tile/host.hpp"

// Stage-1 kernel
#include "im2win_two_stage_configs.hpp"

// Stage-2: use the full example-20 invoker pattern (includes all deps)
#include "../20_grouped_convolution/grouped_convolution_utils.hpp"
#include "../20_grouped_convolution/grouped_convolution_forward_invoker.hpp"

// ── Stage-2 config structs ────────────────────────────────────────────────────

struct Stage2ConfigBase : public ConvConfigBase
{
    // VectorSizeA=4 works for C*Y divisible by 4 (both C=4 and C=256 with Y=3: 12 and 768).
    // VectorSizeB=8 for GKCYX weight (C innermost, C≥8 for large problems; 4 for small).
    // VectorSizeC=8 for NHWGK output (K innermost, K≥8 for large problems; 4 for small).
    // Subclasses can override if needed.
    static constexpr ck_tile::index_t VectorSizeA = 4;
    static constexpr ck_tile::index_t VectorSizeB = 4;
    static constexpr ck_tile::index_t VectorSizeC = 4;
    static constexpr bool DoubleSmemBuffer         = false;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
};

// Config 0: memory pipeline, 128×32, 4×1 warps — good for small K/C problems
template <typename PrecType>
struct Im2winStage2Config_Mem_M128N32K64 : public Stage2ConfigBase
{
    static constexpr ck_tile::index_t VectorSizeA = 4;  // for small C
    static constexpr ck_tile::index_t VectorSizeB = 4;
    static constexpr ck_tile::index_t VectorSizeC = 4;

    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);
    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Intrawave;
};

// Config 1: CV3, 64×64, 2×2 warps, 32×32×16 MFMA — matches best LK config
template <typename PrecType>
struct Im2winStage2Config_CV3_M64N64K64 : public Stage2ConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 64;
    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
};

// Config 2: CV3, 128×64, 4×2 warps (but only 2+2 fits for CV3)
// Use 2×2 warps: M_Tile=128 = 2 M-warps × 32×32 M_WT = 2×32=64? No — need 4 warps for 128.
// M_Tile=128 with 32×32×16: M_Warp=4, N_Warp=2 → BlockSize=4*2*64=512, use 32×32×16 MFMA:
// M_Tile = M_Warp * M_Warp_Tile = 4*32 = 128  ✓
// N_Tile = N_Warp * N_Warp_Tile = 2*32 = 64   ✓
template <typename PrecType>
struct Im2winStage2Config_CV3_M128N64K64 : public Stage2ConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 64;
    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
};

// Config 3: CV3, 64×64, 2×2 warps, kBlockPerCu=2 (higher occupancy)
template <typename PrecType>
struct Im2winStage2Config_CV3_M64N64K64_Occ2 : public Stage2ConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 64;
    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
    static constexpr int kBlockPerCu = 2;
};

// Config 4: CV3, 128×64, 4×2 warps, kBlockPerCu=2
template <typename PrecType>
struct Im2winStage2Config_CV3_M128N64K64_Occ2 : public Stage2ConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 64;
    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
    static constexpr int kBlockPerCu = 2;
};

// Config 5: CV3, 64×128, 2×4 warps — large N tile for K=256 (N_Tile=128 = K/2)
template <typename PrecType>
struct Im2winStage2Config_CV3_M64N128K64 : public Stage2ConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64;
    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
};

// ── VS8 variants for large-C problems (C=256, K=256) — VectorSize=8 ──────────
// These configs use VectorSizeA/B/C=8 which requires C%8==0 and K%8==0.
// For C=256, K=256: all vector sizes work. For C=4, K=4: use VS4 configs above.

template <typename PrecType>
struct Im2winStage2Config_CV3_M64N64K64_VS8 : public Im2winStage2Config_CV3_M64N64K64<PrecType>
{
    static constexpr ck_tile::index_t VectorSizeA = 8;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;
};

template <typename PrecType>
struct Im2winStage2Config_CV3_M128N64K64_VS8 : public Im2winStage2Config_CV3_M128N64K64<PrecType>
{
    static constexpr ck_tile::index_t VectorSizeA = 8;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;
};

template <typename PrecType>
struct Im2winStage2Config_CV3_M64N64K64_Occ2_VS8 : public Im2winStage2Config_CV3_M64N64K64_Occ2<PrecType>
{
    static constexpr ck_tile::index_t VectorSizeA = 8;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;
};

template <typename PrecType>
struct Im2winStage2Config_CV3_M128N64K64_Occ2_VS8 : public Im2winStage2Config_CV3_M128N64K64_Occ2<PrecType>
{
    static constexpr ck_tile::index_t VectorSizeA = 8;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;
};

template <typename PrecType>
struct Im2winStage2Config_CV3_M64N128K64_VS8 : public Im2winStage2Config_CV3_M64N128K64<PrecType>
{
    static constexpr ck_tile::index_t VectorSizeA = 8;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;
};

// ── Active Stage-2 config (compile-time) ─────────────────────────────────────
// IDs 0-5:  VectorSize=4 (works for small C=4 and large C=256)
// IDs 6-10: VectorSize=8 (requires C%8==0, K%8==0 — faster for large C=256)
#ifndef IM2WIN_STAGE2_CONFIG
#define IM2WIN_STAGE2_CONFIG 3   // default: CV3_M64N64K64_Occ2 VS4
#endif

template <typename PrecType>
#if IM2WIN_STAGE2_CONFIG == 0
using ActiveStage2Config = Im2winStage2Config_Mem_M128N32K64<PrecType>;
#elif IM2WIN_STAGE2_CONFIG == 1
using ActiveStage2Config = Im2winStage2Config_CV3_M64N64K64<PrecType>;
#elif IM2WIN_STAGE2_CONFIG == 2
using ActiveStage2Config = Im2winStage2Config_CV3_M128N64K64<PrecType>;
#elif IM2WIN_STAGE2_CONFIG == 3
using ActiveStage2Config = Im2winStage2Config_CV3_M64N64K64_Occ2<PrecType>;
#elif IM2WIN_STAGE2_CONFIG == 4
using ActiveStage2Config = Im2winStage2Config_CV3_M128N64K64_Occ2<PrecType>;
#elif IM2WIN_STAGE2_CONFIG == 5
using ActiveStage2Config = Im2winStage2Config_CV3_M64N128K64<PrecType>;
// ── VS8 variants (large C=256, K=256) ───────────────────────────────────────
#elif IM2WIN_STAGE2_CONFIG == 6
using ActiveStage2Config = Im2winStage2Config_CV3_M64N64K64_VS8<PrecType>;
#elif IM2WIN_STAGE2_CONFIG == 7
using ActiveStage2Config = Im2winStage2Config_CV3_M128N64K64_VS8<PrecType>;
#elif IM2WIN_STAGE2_CONFIG == 8
using ActiveStage2Config = Im2winStage2Config_CV3_M64N64K64_Occ2_VS8<PrecType>;
#elif IM2WIN_STAGE2_CONFIG == 9
using ActiveStage2Config = Im2winStage2Config_CV3_M128N64K64_Occ2_VS8<PrecType>;
#elif IM2WIN_STAGE2_CONFIG == 10
using ActiveStage2Config = Im2winStage2Config_CV3_M64N128K64_VS8<PrecType>;
#else
#error "Unknown IM2WIN_STAGE2_CONFIG — valid range: 0..10"
#endif

// ── Stage-2 launcher ──────────────────────────────────────────────────────────
// Dispatches with GNCHW_Im2win A descriptor + GKCYX weight + NHWGK output.
template <typename DataType>
float launch_stage2_conv(const ck_tile::GroupedConvFwdHostArgs<>& args,
                         int n_warmup, int n_repeat)
{
    using GNCHW_Im2win = ck_tile::tensor_layout::convolution::GNCHW_Im2win;
    using GKCYX        = ck_tile::tensor_layout::convolution::GKCYX;
    using NHWGK        = ck_tile::tensor_layout::convolution::NHWGK;
    using Config       = ActiveStage2Config<DataType>;

    return GroupedConvolutionForwardInvoker::grouped_conv_fwd<2,
                                                              Config,
                                                              DataType,
                                                              DataType,
                                                              float,
                                                              DataType,
                                                              GNCHW_Im2win,
                                                              GKCYX,
                                                              NHWGK>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});
}

// ── Stage-1 launcher ──────────────────────────────────────────────────────────
template <typename DataType>
float launch_stage1_im2win(const void* in_ptr,
                            void*       out_ptr,
                            const ck_tile::conv::ConvParam& p,
                            bool is_nhwgc,
                            int n_warmup, int n_repeat)
{
    using Kernel = ActiveIm2winTransformKernel<DataType>;

    auto kargs = Kernel::MakeKargs(
        in_ptr, out_ptr,
        static_cast<int>(p.G_),
        static_cast<int>(p.N_),
        static_cast<int>(p.C_),
        static_cast<int>(p.input_spatial_lengths_[0]),
        static_cast<int>(p.input_spatial_lengths_[1]),
        static_cast<int>(p.output_spatial_lengths_[0]),
        static_cast<int>(p.filter_spatial_lengths_[0]),
        static_cast<int>(p.conv_filter_strides_[0]),
        static_cast<int>(p.conv_filter_strides_[1]),
        static_cast<int>(p.conv_filter_dilations_[0]),
        static_cast<int>(p.conv_filter_dilations_[1]),
        static_cast<int>(p.input_left_pads_[0]),
        static_cast<int>(p.input_left_pads_[1]),
        static_cast<int>(p.input_right_pads_[0]),
        static_cast<int>(p.input_right_pads_[1]),
        is_nhwgc);

    const dim3 grids  = Kernel::GridSize(kargs);
    const dim3 blocks = Kernel::BlockSize();

    return ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat},
        ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));
}

// ── Helpers ───────────────────────────────────────────────────────────────────
// Host-side reorder: GNCHW → NHWGC (for reference comparison when input is NHWGC)
template <typename DataType>
static ck_tile::HostTensor<DataType>
reorder_gnchw_to_nhwgc_simple(const ck_tile::HostTensor<DataType>& src,
                                int G, int N, int C, int H, int W)
{
    ck_tile::HostTensor<DataType> dst({static_cast<std::size_t>(N),
                                       static_cast<std::size_t>(H),
                                       static_cast<std::size_t>(W),
                                       static_cast<std::size_t>(G),
                                       static_cast<std::size_t>(C)});
    for(int g = 0; g < G; ++g)
        for(int n = 0; n < N; ++n)
            for(int h = 0; h < H; ++h)
                for(int w = 0; w < W; ++w)
                    for(int c = 0; c < C; ++c)
                    {
                        int si = g*(N*C*H*W) + n*(C*H*W) + c*(H*W) + h*W + w;
                        int di = n*(H*W*G*C) + h*(W*G*C) + w*(G*C) + g*C + c;
                        dst.mData[di] = src.mData[si];
                    }
    return dst;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    try
    {
        auto [result, arg_parser] = create_args(argc, argv);
        if(!result)
            return -1;

        const std::string data_type  = arg_parser.get_str("prec");
        const std::string in_layout  = arg_parser.get_str("in_layout");
        const std::string wei_layout = arg_parser.get_str("wei_layout");
        const std::string out_layout = arg_parser.get_str("out_layout");

        if(data_type != "fp16" && data_type != "bf16")
        {
            std::cerr << "Unsupported precision: " << data_type << "\n";
            return 1;
        }
        const bool use_fp16 = (data_type == "fp16");

        // Determine input layout
        const bool input_is_nhwgc =
            (in_layout == "NHWGC" || in_layout.empty());  // default: GNCHW
        const bool input_is_gnchw = (in_layout == "GNCHW" || in_layout.empty());
        if(!input_is_nhwgc && !input_is_gnchw)
        {
            std::cerr << "Unsupported in_layout: " << in_layout
                      << " (supported: GNCHW, NHWGC)\n";
            return 1;
        }
        // When not explicitly specified, default to GNCHW
        const bool use_nhwgc = input_is_nhwgc && !input_is_gnchw;

        std::vector<ck_tile::index_t> filter_spatial_lengths, image_spatial_lengths;
        std::vector<ck_tile::index_t> strides, dilations, lpads, rpads;

        const ck_tile::index_t num_dim_sp = fill_spatial_dimensions(
            filter_spatial_lengths, image_spatial_lengths,
            strides, dilations, lpads, rpads, arg_parser);

        ck_tile::conv::ConvParam p{num_dim_sp,
            arg_parser.get_int("g"), arg_parser.get_int("n"),
            arg_parser.get_int("k"), arg_parser.get_int("c"),
            filter_spatial_lengths, image_spatial_lengths,
            strides, dilations, lpads, rpads};

        const int n_warmup = arg_parser.get_int("warmup");
        const int n_repeat = arg_parser.get_int("repeat");
        const int verify   = arg_parser.get_int("v");

        const int G    = static_cast<int>(p.G_);
        const int N    = static_cast<int>(p.N_);
        const int C    = static_cast<int>(p.C_);
        const int K    = static_cast<int>(p.K_);
        const int Hi   = static_cast<int>(p.input_spatial_lengths_[0]);
        const int Wi   = static_cast<int>(p.input_spatial_lengths_[1]);
        const int Ho   = static_cast<int>(p.output_spatial_lengths_[0]);
        const int Wo   = static_cast<int>(p.output_spatial_lengths_[1]);
        const int Y    = static_cast<int>(p.filter_spatial_lengths_[0]);
        const int X    = static_cast<int>(p.filter_spatial_lengths_[1]);
        const int LPW  = static_cast<int>(p.input_left_pads_[1]);
        const int RPW  = static_cast<int>(p.input_right_pads_[1]);
        const int Wi_pad = Wi + LPW + RPW;

        const size_t elem_bytes  = use_fp16 ? sizeof(ck_tile::half_t) : sizeof(ck_tile::bf16_t);
        // Input sizes differ by layout
        const size_t in_size_gnchw  = static_cast<size_t>(G) * N * C * Hi * Wi;
        const size_t in_size_nhwgc  = static_cast<size_t>(N) * Hi * Wi * G * C;
        const size_t in_size        = use_nhwgc ? in_size_nhwgc : in_size_gnchw;
        const size_t iprime_size = static_cast<size_t>(G) * N * C * Ho * Wi_pad * Y;
        const size_t wei_size    = static_cast<size_t>(G) * K * C * Y * X;
        const size_t out_size    = static_cast<size_t>(N) * Ho * Wo * G * K; // NHWGK

        using half_t = ck_tile::half_t;
        using bf16_t = ck_tile::bf16_t;

        // Host tensors — always allocate as GNCHW internally for reference
        const auto in_gnchw_desc = ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<
            ck_tile::tensor_layout::convolution::GNCHW>(p);
        const auto wei_gnchw_desc = ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
            ck_tile::tensor_layout::convolution::GKCYX>(p);
        const auto out_desc = ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<
            ck_tile::tensor_layout::convolution::NHWGK>(p);

        ck_tile::HostTensor<half_t> input_f16(in_gnchw_desc),  weight_f16(wei_gnchw_desc);
        ck_tile::HostTensor<bf16_t> input_bf16(in_gnchw_desc), weight_bf16(wei_gnchw_desc);

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

        ck_tile::DeviceMem input_buf(in_size     * elem_bytes);
        ck_tile::DeviceMem iprime_buf(iprime_size * elem_bytes);
        ck_tile::DeviceMem weight_buf(wei_size    * elem_bytes);
        ck_tile::DeviceMem output_buf(out_size    * elem_bytes);

        if(use_nhwgc)
        {
            // Reorder GNCHW → NHWGC for input
            if(use_fp16)
            {
                auto inp_nhwgc = reorder_gnchw_to_nhwgc_simple(input_f16, G, N, C, Hi, Wi);
                input_buf.ToDevice(inp_nhwgc.data());
            }
            else
            {
                auto inp_nhwgc = reorder_gnchw_to_nhwgc_simple(input_bf16, G, N, C, Hi, Wi);
                input_buf.ToDevice(inp_nhwgc.data());
            }
        }
        else
        {
            if(use_fp16) input_buf.ToDevice(input_f16.data());
            else         input_buf.ToDevice(input_bf16.data());
        }

        if(use_fp16) weight_buf.ToDevice(weight_f16.data());
        else         weight_buf.ToDevice(weight_bf16.data());
        iprime_buf.SetZero();
        output_buf.SetZero();

        // Host args for Stage-2 (in_ptr = I', weight and output from real buffers)
        ck_tile::GroupedConvFwdHostArgs<> args(p,
            iprime_buf.GetDeviceBuffer(),
            weight_buf.GetDeviceBuffer(),
            {}, output_buf.GetDeviceBuffer(), 1);

        const std::string layout_str = use_nhwgc ? "NHWGC/GKCYX" : "GNCHW/GKCYX";
        std::cout << "Two-stage im2win fwd conv (" << layout_str << ")\n"
                  << "  G=" << G << " N=" << N << " C=" << C << " K=" << K
                  << " Hi=" << Hi << " Wi=" << Wi << " Ho=" << Ho << " Wo=" << Wo
                  << " Y=" << Y << " X=" << X << "\n"
                  << "  I' size: [" << G << "×" << N << "×" << C << "×"
                  << Ho << "×" << Wi_pad << "×" << Y << "] = "
                  << iprime_size * elem_bytes / 1e6 << " MB\n";

        // Stage 1
        float t1;
        if(use_fp16)
            t1 = launch_stage1_im2win<half_t>(input_buf.GetDeviceBuffer(),
                    iprime_buf.GetDeviceBuffer(), p, use_nhwgc, n_warmup, n_repeat);
        else
            t1 = launch_stage1_im2win<bf16_t>(input_buf.GetDeviceBuffer(),
                    iprime_buf.GetDeviceBuffer(), p, use_nhwgc, n_warmup, n_repeat);

        const size_t s1_bytes = (in_size + iprime_size) * elem_bytes;
        std::cout << "  Stage 1 (I→I'):   " << t1 << " ms  "
                  << s1_bytes / 1.e6f / t1 << " GB/s\n";

        // Stage 2
        output_buf.SetZero();
        float t2;
        if(use_fp16) t2 = launch_stage2_conv<half_t>(args, n_warmup, n_repeat);
        else         t2 = launch_stage2_conv<bf16_t>(args, n_warmup, n_repeat);

        const size_t flop   = args.GetFlops();
        const size_t s2bytes = use_fp16 ? args.GetByte<half_t, half_t, half_t>()
                                        : args.GetByte<bf16_t, bf16_t, bf16_t>();
        std::cout << "  Stage 2 (I'+W→O): " << t2 << " ms  "
                  << static_cast<float>(flop) / 1.e9f / t2 << " TFlops  "
                  << s2bytes / 1.e6f / t2 << " GB/s\n"
                  << "  Combined:         " << t1 + t2 << " ms\n";

        // Verification
        if(verify != 0)
        {
            std::cout << "  Running CPU reference...\n";
            ck_tile::HostTensor<half_t> oref_f16(out_desc);
            ck_tile::HostTensor<bf16_t> oref_bf16(out_desc);
            oref_f16.SetZero();
            oref_bf16.SetZero();

            if(use_fp16)
                ck_tile::reference_grouped_conv_fwd<2, half_t, half_t, half_t>(
                    input_f16, weight_f16, oref_f16,
                    p.conv_filter_strides_, p.conv_filter_dilations_,
                    p.input_left_pads_, p.input_right_pads_);
            else
                ck_tile::reference_grouped_conv_fwd<2, bf16_t, bf16_t, bf16_t>(
                    input_bf16, weight_bf16, oref_bf16,
                    p.conv_filter_strides_, p.conv_filter_dilations_,
                    p.input_left_pads_, p.input_right_pads_);

            const ck_tile::index_t GemmK = static_cast<ck_tile::index_t>(C) * Y * X;
            bool pass = false;
            if(use_fp16)
            {
                ck_tile::HostTensor<half_t> odev(out_desc);
                output_buf.FromDevice(odev.data());
                const float mv = *std::max_element(oref_f16.mData.begin(), oref_f16.mData.end());
                const auto ta = calculate_rtol_atol<half_t, half_t, float, half_t>(GemmK, 1, mv);
                pass = ck_tile::check_err(odev, oref_f16, "Error: incorrect results!",
                    ta.at(ck_tile::number<0>{}), ta.at(ck_tile::number<1>{}));
            }
            else
            {
                ck_tile::HostTensor<bf16_t> odev(out_desc);
                output_buf.FromDevice(odev.data());
                const float mv = *std::max_element(oref_bf16.mData.begin(), oref_bf16.mData.end());
                const auto ta = calculate_rtol_atol<bf16_t, bf16_t, float, bf16_t>(GemmK, 1, mv);
                pass = ck_tile::check_err(odev, oref_bf16, "Error: incorrect results!",
                    ta.at(ck_tile::number<0>{}), ta.at(ck_tile::number<1>{}));
            }

            std::cout << "  verification: " << (pass ? "PASSED" : "FAILED") << "\n";
            return pass ? 0 : 1;
        }

        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
