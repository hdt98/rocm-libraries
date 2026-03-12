// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ═══════════════════════════════════════════════════════════════════════
// Integration test for GroupedConvolutionForwardIm2winKernel
//
// Strategy
// --------
// For each test case:
//   1. Fill input (GNCHW) and weight (GKCYX) host tensors with random data.
//   2. Upload to device.
//   3. Run im2win kernel → output (NHWGK) on device.
//   4. Prepare the GPU reference inputs:
//        • Weight GKCYX→GKYXC: use BatchedTransposeKernel on device
//          (batch = G×K, height = C, width = Y×X).
//        • Input  GNCHW→NHWGC: use a strided HostTensor view on host to
//          avoid the nested G/N dimension interleaving that would require
//          two separate GPU transpose passes.
//   5. Run the naive GPU reference → NHWGK output.
//   6. Compare im2win and reference outputs element-wise.
//
// The weight transpose from GKCYX to GKYXC maps cleanly to a 2D batched
// transpose (swap the C row-dimension with the Y×X column-dimension) and
// is therefore done entirely on the GPU using BatchedTransposeKernel.
//
// The input GNCHW→NHWGC requires two separate dim-group swaps (G↔N and
// C↔H*W) that cannot be expressed as a single 2D batched transpose because
// G and N are interleaved in the target layout.  We handle it on host via
// a strided HostTensor view, which copies elements with the correct strides
// without any explicit index arithmetic.
//
// Test matrix
// -----------
//   • fp16 and bf16 data types
//   • G=1, G=4, G=32 (primary target)
//   • C=K ∈ {4, 8, 16}
//   • 3×3 filter, unit stride, same-pad (Ho=Hi, Wo=Wi)
//   • Stride=2 (valid, no-pad)
//   • N=1 and N=2 (multi-batch)
// ═══════════════════════════════════════════════════════════════════════

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ref/naive_grouped_conv_fwd_gpu.hpp"
#include "ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_im2win.hpp"
#include "ck_tile/ops/batched_transpose.hpp"

#include "grouped_conv_fwd_im2win_invoker.hpp"

// ── GPU transpose helper ──────────────────────────────────────────────────────
// Launches BatchedTransposeKernel to transpose a flat [batch, height, width]
// device buffer to [batch, width, height] in-place (src and dst are separate).
//
// Uses the Universal pipeline with padding enabled so arbitrary shapes work.
template <typename DataType>
static void gpu_batched_transpose(const ck_tile::DeviceMem& d_src,
                                  ck_tile::DeviceMem& d_dst,
                                  int batch,
                                  int height,
                                  int width)
{
    // Block tile 64×64, single warp, padding enabled for arbitrary shapes.
    using BlockTile  = ck_tile::sequence<64, 64>;
    using WarpLayout = ck_tile::sequence<1, 1>;
    using Problem    = ck_tile::BatchedTransposeProblem<DataType,
                                                        BlockTile,
                                                        WarpLayout,
                                                        /*kPadM=*/true,
                                                        /*kPadN=*/true>;
    using Pipeline   = ck_tile::BatchedTransposePipeline<Problem>;
    using Kernel     = ck_tile::BatchedTransposeKernel<Pipeline>;

    const ck_tile::BatchedTransposeHostArgs hargs{
        d_src.GetDeviceBuffer(),
        d_dst.GetDeviceBuffer(),
        batch,
        height,
        width,
        /*dim_stride=*/height * width,
        /*dim_block_h=*/BlockTile::at(ck_tile::number<0>{}),
        /*dim_block_w=*/BlockTile::at(ck_tile::number<1>{})};

    auto kargs            = Kernel::MakeKargs(hargs);
    const dim3 grid_size  = Kernel::GridSize(hargs);
    const dim3 block_size = Kernel::BlockSize();

    ck_tile::launch_kernel(ck_tile::stream_config{},
                           ck_tile::make_kernel<1>(Kernel{}, grid_size, block_size, 0, kargs));
}

// ── Host layout reorder: GNCHW → NHWGC ───────────────────────────────────────
// The naive GPU reference requires input in NHWGC = [N, Hi, Wi, G, C].
// Our input is in GNCHW = [G, N, C, Hi, Wi].
//
// This permutation requires swapping (G,N) and (C,Hi,Wi) dimension groups,
// which cannot be expressed as a single 2D batched transpose because G and N
// are interleaved in the target layout.  We use a strided HostTensor view to
// let CK's tensor infrastructure compute the correct element offsets.
template <typename DataType>
static ck_tile::HostTensor<DataType>
host_gnchw_to_nhwgc(const ck_tile::HostTensor<DataType>& src, int G, int N, int C, int H, int W)
{
    // src physical layout: [G, N, C, H, W] with packed strides
    //   stride(G) = N*C*H*W,  stride(N) = C*H*W,  stride(C) = H*W,
    //   stride(H) = W,        stride(W) = 1
    //
    // dst logical layout:  [N, H, W, G, C]
    //   stride(N) = H*W*G*C,  stride(H) = W*G*C,  stride(W) = G*C,
    //   stride(G) = C,        stride(C) = 1
    //
    // We build dst with these explicit strides so that element (n,h,w,g,c)
    // in the dst HostTensor maps to the correct flat index, then copy from src.

    const int nhwgc_total = N * H * W * G * C;

    // dst tensor with NHWGC element order (packed)
    ck_tile::HostTensor<DataType> dst({static_cast<std::size_t>(N),
                                       static_cast<std::size_t>(H),
                                       static_cast<std::size_t>(W),
                                       static_cast<std::size_t>(G),
                                       static_cast<std::size_t>(C)});
    dst.mData.resize(nhwgc_total);

    // Use explicit index arithmetic matching the GNCHW src strides.
    const int sN = C * H * W; // src stride of N within one G slice
    const int sC = H * W;
    const int sH = W;
    const int sW = 1;
    const int sG = N * C * H * W; // src stride of G (outermost)

    for(int n = 0; n < N; ++n)
        for(int h = 0; h < H; ++h)
            for(int w = 0; w < W; ++w)
                for(int g = 0; g < G; ++g)
                    for(int c = 0; c < C; ++c)
                    {
                        // GNCHW src flat index
                        int src_idx = g * sG + n * sN + c * sC + h * sH + w * sW;
                        // NHWGC dst flat index
                        int dst_idx = n * (H * W * G * C) + h * (W * G * C) +
                                      w * (G * C) + g * C + c;
                        dst.mData[dst_idx] = src.mData[src_idx];
                    }
    return dst;
}

// ── Test runner ───────────────────────────────────────────────────────────────

struct ConvProblem
{
    int G, N, C, K;
    int Hi, Wi, Y, X;
    int Sy, Sx;   // stride
    int LPH, LPW; // left pad
    std::string label;
};

template <typename DataType>
static bool run_im2win_vs_ref(const ConvProblem& p)
{
    using AccDataType = float;

    const int Ho = (p.Hi + 2 * p.LPH - p.Y) / p.Sy + 1;
    const int Wo = (p.Wi + 2 * p.LPW - p.X) / p.Sx + 1;

    // ── Layout choice ─────────────────────────────────────────────────
    // Input:  GNCHW (G, N, C, Hi, Wi) — channels-first
    // Weight: GKCYX (G, K, C,  Y,  X) — channels-first
    // Output: NHWGK (N, Ho, Wo, G, K) — channels-last, K innermost
    //
    // NHWGK is required because the CShuffleEpilogue writes vectorised
    // stores along the N_gemm=K dimension, which must be contiguous.
    using InLayout  = ck_tile::tensor_layout::convolution::GNCHW;
    using WeiLayout = ck_tile::tensor_layout::convolution::GKCYX;
    using OutLayout = ck_tile::tensor_layout::convolution::NHWGK;
    using Config    = Im2winConvTestConfig<DataType>;

    // ── Host tensors ──────────────────────────────────────────────────
    ck_tile::HostTensor<DataType> h_input({static_cast<std::size_t>(p.G),
                                           static_cast<std::size_t>(p.N),
                                           static_cast<std::size_t>(p.C),
                                           static_cast<std::size_t>(p.Hi),
                                           static_cast<std::size_t>(p.Wi)});
    ck_tile::HostTensor<DataType> h_weight({static_cast<std::size_t>(p.G),
                                            static_cast<std::size_t>(p.K),
                                            static_cast<std::size_t>(p.C),
                                            static_cast<std::size_t>(p.Y),
                                            static_cast<std::size_t>(p.X)});

    ck_tile::FillUniformDistribution<DataType>{-2.f, 2.f}(h_input);
    ck_tile::FillUniformDistribution<DataType>{-2.f, 2.f}(h_weight);

    // ── Device buffers for im2win kernel (channels-first layout) ──────
    const std::size_t out_elems =
        static_cast<std::size_t>(p.N) * Ho * Wo * p.G * p.K;
    ck_tile::DeviceMem d_input(h_input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_weight(h_weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_output(out_elems * sizeof(DataType));

    d_input.ToDevice(h_input.data());
    d_weight.ToDevice(h_weight.data());
    d_output.SetZero();

    // ── Run im2win kernel ─────────────────────────────────────────────
    ck_tile::conv::ConvParam conv_param{
        2 /*NDimSpatial*/,
        p.G, p.N, p.K, p.C,
        {p.Y,  p.X},
        {p.Hi, p.Wi},
        {p.Sy, p.Sx},
        {1,    1},        // dilation
        {p.LPH, p.LPW},
        {p.LPH, p.LPW}}; // symmetric padding

    ck_tile::GroupedConvFwdHostArgs<> args(conv_param,
                                           d_input.GetDeviceBuffer(),
                                           d_weight.GetDeviceBuffer(),
                                           {},
                                           d_output.GetDeviceBuffer(),
                                           /*kbatch=*/1);

    GroupedConvFwdIm2winInvoker::grouped_conv_fwd<2,
                                                   Config,
                                                   DataType,
                                                   DataType,
                                                   AccDataType,
                                                   DataType,
                                                   InLayout,
                                                   WeiLayout,
                                                   OutLayout>(args,
                                                              ck_tile::stream_config{nullptr,
                                                                                     false});

    // Copy im2win output (NHWGK) back to host.
    ck_tile::HostTensor<DataType> h_output_im2win({static_cast<std::size_t>(p.N),
                                                   static_cast<std::size_t>(Ho),
                                                   static_cast<std::size_t>(Wo),
                                                   static_cast<std::size_t>(p.G),
                                                   static_cast<std::size_t>(p.K)});
    d_output.FromDevice(h_output_im2win.data());

    // ── Prepare reference inputs ──────────────────────────────────────
    //
    // Weight: GKCYX → GKYXC via GPU BatchedTransposeKernel
    // ─────────────────────────────────────────────────────
    // View GKCYX as [G*K, C, Y*X].  Transposing height=C ↔ width=Y*X
    // gives [G*K, Y*X, C] = GKYXC.
    const int wei_batch  = p.G * p.K;
    const int wei_height = p.C;
    const int wei_width  = p.Y * p.X;

    ck_tile::DeviceMem d_weight_gkyxc(h_weight.get_element_space_size_in_bytes());
    gpu_batched_transpose<DataType>(d_weight, d_weight_gkyxc, wei_batch, wei_height, wei_width);

    //
    // Input: GNCHW → NHWGC via strided host copy
    // ────────────────────────────────────────────
    // The target NHWGC layout interleaves the G and N outer dimensions
    // (dst[n,h,w,g,c] comes from src[g,n,c,h,w]), which cannot be
    // expressed as a single 2D batched transpose.  We use an explicit
    // host-side copy with the correct index arithmetic.
    auto h_input_nhwgc = host_gnchw_to_nhwgc(h_input, p.G, p.N, p.C, p.Hi, p.Wi);

    ck_tile::DeviceMem d_input_ref(h_input_nhwgc.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_output_ref(out_elems * sizeof(DataType));

    d_input_ref.ToDevice(h_input_nhwgc.data());
    d_output_ref.SetZero();

    // ── Run naive GPU reference → NHWGK ───────────────────────────────
    ck_tile::naive_grouped_conv_fwd<2, DataType, DataType, DataType>(
        reinterpret_cast<const DataType*>(d_input_ref.GetDeviceBuffer()),
        reinterpret_cast<const DataType*>(d_weight_gkyxc.GetDeviceBuffer()),
        reinterpret_cast<DataType*>(d_output_ref.GetDeviceBuffer()),
        p.G, p.N, p.K, p.C,
        {static_cast<ck_tile::long_index_t>(p.Hi), static_cast<ck_tile::long_index_t>(p.Wi)},
        {static_cast<ck_tile::long_index_t>(p.Y),  static_cast<ck_tile::long_index_t>(p.X)},
        {static_cast<ck_tile::long_index_t>(Ho),   static_cast<ck_tile::long_index_t>(Wo)},
        {static_cast<ck_tile::long_index_t>(p.Sy), static_cast<ck_tile::long_index_t>(p.Sx)},
        {1LL, 1LL},
        {static_cast<ck_tile::long_index_t>(p.LPH),
         static_cast<ck_tile::long_index_t>(p.LPW)});

    // Copy reference output (NHWGK) to host.
    ck_tile::HostTensor<DataType> h_output_ref({static_cast<std::size_t>(p.N),
                                                static_cast<std::size_t>(Ho),
                                                static_cast<std::size_t>(Wo),
                                                static_cast<std::size_t>(p.G),
                                                static_cast<std::size_t>(p.K)});
    d_output_ref.FromDevice(h_output_ref.data());

    // ── Compute tolerance and compare ─────────────────────────────────
    const int GemmK = p.C * p.Y * p.X;
    const float max_val = *std::max_element(h_output_ref.mData.begin(),
                                            h_output_ref.mData.end());
    const auto rtol = ck_tile::get_relative_threshold<DataType, DataType, AccDataType>(GemmK);
    const auto atol = ck_tile::get_absolute_threshold<DataType, DataType, AccDataType>(
        max_val, GemmK);

    bool pass = ck_tile::check_err(
        h_output_im2win, h_output_ref, "im2win vs GPU reference", rtol, atol);

    if(!pass)
        std::cout << "[FAIL] " << p.label << "  rtol=" << rtol << " atol=" << atol << "\n";

    return pass;
}

// ── Test runner for NHWGC/GKYXC layout (channels-last) ───────────────────────
//
// Channels-last is the layout that enables group merging (G adjacent to C).
// The GPU reference already uses NHWGC/GKYXC, so no transpose is needed.
template <typename DataType>
static bool run_im2win_vs_ref_nhwgc(const ConvProblem& p)
{
    using AccDataType = float;

    const int Ho = (p.Hi + 2 * p.LPH - p.Y) / p.Sy + 1;
    const int Wo = (p.Wi + 2 * p.LPW - p.X) / p.Sx + 1;

    using InLayout  = ck_tile::tensor_layout::convolution::NHWGC;
    using WeiLayout = ck_tile::tensor_layout::convolution::GKYXC;
    using OutLayout = ck_tile::tensor_layout::convolution::NHWGK;
    using Config    = Im2winConvTestConfig<DataType>;

    // ── Host tensors in NHWGC / GKYXC / NHWGK ────────────────────────
    // NHWGC = [N, Hi, Wi, G, C]
    ck_tile::HostTensor<DataType> h_input({static_cast<std::size_t>(p.N),
                                           static_cast<std::size_t>(p.Hi),
                                           static_cast<std::size_t>(p.Wi),
                                           static_cast<std::size_t>(p.G),
                                           static_cast<std::size_t>(p.C)});
    // GKYXC = [G, K, Y, X, C]
    ck_tile::HostTensor<DataType> h_weight({static_cast<std::size_t>(p.G),
                                            static_cast<std::size_t>(p.K),
                                            static_cast<std::size_t>(p.Y),
                                            static_cast<std::size_t>(p.X),
                                            static_cast<std::size_t>(p.C)});

    ck_tile::FillUniformDistribution<DataType>{-2.f, 2.f}(h_input);
    ck_tile::FillUniformDistribution<DataType>{-2.f, 2.f}(h_weight);

    const std::size_t out_elems =
        static_cast<std::size_t>(p.N) * Ho * Wo * p.G * p.K;
    ck_tile::DeviceMem d_input(h_input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_weight(h_weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_output(out_elems * sizeof(DataType));

    d_input.ToDevice(h_input.data());
    d_weight.ToDevice(h_weight.data());
    d_output.SetZero();

    // ── Run im2win kernel (NHWGC/GKYXC → NHWGK) ──────────────────────
    ck_tile::conv::ConvParam conv_param{
        2, p.G, p.N, p.K, p.C,
        {p.Y, p.X}, {p.Hi, p.Wi}, {p.Sy, p.Sx},
        {1, 1}, {p.LPH, p.LPW}, {p.LPH, p.LPW}};

    ck_tile::GroupedConvFwdHostArgs<> args(conv_param,
                                           d_input.GetDeviceBuffer(),
                                           d_weight.GetDeviceBuffer(),
                                           {}, d_output.GetDeviceBuffer(), 1);

    GroupedConvFwdIm2winInvoker::grouped_conv_fwd<2, Config,
                                                   DataType, DataType, AccDataType, DataType,
                                                   InLayout, WeiLayout, OutLayout>(
        args, ck_tile::stream_config{nullptr, false});

    ck_tile::HostTensor<DataType> h_output_im2win({static_cast<std::size_t>(p.N),
                                                   static_cast<std::size_t>(Ho),
                                                   static_cast<std::size_t>(Wo),
                                                   static_cast<std::size_t>(p.G),
                                                   static_cast<std::size_t>(p.K)});
    d_output.FromDevice(h_output_im2win.data());

    // ── GPU reference (same NHWGC/GKYXC, no transpose needed) ─────────
    ck_tile::DeviceMem d_output_ref(out_elems * sizeof(DataType));
    d_output_ref.SetZero();

    ck_tile::naive_grouped_conv_fwd<2, DataType, DataType, DataType>(
        reinterpret_cast<const DataType*>(d_input.GetDeviceBuffer()),
        reinterpret_cast<const DataType*>(d_weight.GetDeviceBuffer()),
        reinterpret_cast<DataType*>(d_output_ref.GetDeviceBuffer()),
        p.G, p.N, p.K, p.C,
        {static_cast<ck_tile::long_index_t>(p.Hi), static_cast<ck_tile::long_index_t>(p.Wi)},
        {static_cast<ck_tile::long_index_t>(p.Y),  static_cast<ck_tile::long_index_t>(p.X)},
        {static_cast<ck_tile::long_index_t>(Ho),   static_cast<ck_tile::long_index_t>(Wo)},
        {static_cast<ck_tile::long_index_t>(p.Sy), static_cast<ck_tile::long_index_t>(p.Sx)},
        {1LL, 1LL},
        {static_cast<ck_tile::long_index_t>(p.LPH),
         static_cast<ck_tile::long_index_t>(p.LPW)});

    ck_tile::HostTensor<DataType> h_output_ref({static_cast<std::size_t>(p.N),
                                                static_cast<std::size_t>(Ho),
                                                static_cast<std::size_t>(Wo),
                                                static_cast<std::size_t>(p.G),
                                                static_cast<std::size_t>(p.K)});
    d_output_ref.FromDevice(h_output_ref.data());

    // ── Compare ───────────────────────────────────────────────────────
    const int GemmK = p.C * p.Y * p.X;
    const float max_val = *std::max_element(h_output_ref.mData.begin(),
                                            h_output_ref.mData.end());
    const auto rtol = ck_tile::get_relative_threshold<DataType, DataType, AccDataType>(GemmK);
    const auto atol = ck_tile::get_absolute_threshold<DataType, DataType, AccDataType>(
        max_val, GemmK);

    bool pass = ck_tile::check_err(
        h_output_im2win, h_output_ref, "im2win(NHWGC) vs GPU reference", rtol, atol);

    if(!pass)
        std::cout << "[FAIL] " << p.label << " (NHWGC)  rtol=" << rtol << " atol=" << atol << "\n";

    return pass;
}

// ── GTest parametrised fixture ────────────────────────────────────────────────

template <typename DataType>
class GroupedConvFwdIm2winTest : public ::testing::Test
{
};

using TestTypes = ::testing::Types<ck_tile::half_t, ck_tile::bf16_t>;
TYPED_TEST_SUITE(GroupedConvFwdIm2winTest, TestTypes);

// Helper macro — avoids repeating run_im2win_vs_ref<TypeParam> everywhere.
#define RUN(problem) EXPECT_TRUE((run_im2win_vs_ref<TypeParam>(problem)))

// ── Test cases ────────────────────────────────────────────────────────────────

// 3×3, same-pad, unit stride, G=1
TYPED_TEST(GroupedConvFwdIm2winTest, G1_C8_K8_3x3_pad1)
{
    RUN((ConvProblem{1, 1, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1, "G1_C8_K8_3x3_pad1"}));
}

// 3×3, same-pad, unit stride, G=4  (C=K=8, primary interest)
TYPED_TEST(GroupedConvFwdIm2winTest, G4_C8_K8_3x3_pad1)
{
    RUN((ConvProblem{4, 1, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1, "G4_C8_K8_3x3_pad1"}));
}

// 3×3, same-pad, unit stride, G=32 — the primary use case from CLAUDE.local.md
TYPED_TEST(GroupedConvFwdIm2winTest, G32_C8_K8_3x3_pad1)
{
    RUN((ConvProblem{32, 1, 8, 8, 16, 16, 3, 3, 1, 1, 1, 1, "G32_C8_K8_3x3_pad1"}));
}

// C=K=4 (smallest target channel count)
TYPED_TEST(GroupedConvFwdIm2winTest, G32_C4_K4_3x3_pad1)
{
    RUN((ConvProblem{32, 1, 4, 4, 16, 16, 3, 3, 1, 1, 1, 1, "G32_C4_K4_3x3_pad1"}));
}

// C=K=16
TYPED_TEST(GroupedConvFwdIm2winTest, G32_C16_K16_3x3_pad1)
{
    RUN((ConvProblem{32, 1, 16, 16, 16, 16, 3, 3, 1, 1, 1, 1, "G32_C16_K16_3x3_pad1"}));
}

// Stride=2 (valid padding, larger input)
TYPED_TEST(GroupedConvFwdIm2winTest, G4_C8_K8_3x3_stride2)
{
    // Hi=Wi=9, Y=X=3, stride=2, no pad → Ho=Wo=4
    RUN((ConvProblem{4, 1, 8, 8, 9, 9, 3, 3, 2, 2, 0, 0, "G4_C8_K8_3x3_stride2"}));
}

// Multi-batch N=2
TYPED_TEST(GroupedConvFwdIm2winTest, G4_N2_C8_K8_3x3_pad1)
{
    RUN((ConvProblem{4, 2, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1, "G4_N2_C8_K8_3x3_pad1"}));
}

// Larger spatial, G=32 — realistic depthwise-style workload
TYPED_TEST(GroupedConvFwdIm2winTest, G32_C8_K8_3x3_Hi32Wi32)
{
    RUN((ConvProblem{32, 1, 8, 8, 32, 32, 3, 3, 1, 1, 1, 1, "G32_C8_K8_3x3_Hi32Wi32"}));
}

#undef RUN

// ── Channels-last (NHWGC/GKYXC) test suite ───────────────────────────────────
// Tests the same problems but with the channels-last layout pair that enables
// group merging.  The GPU reference uses NHWGC/GKYXC directly so no transpose
// is needed, making these tests simpler and faster.
template <typename DataType>
class GroupedConvFwdIm2winTestNhwgc : public ::testing::Test {};

TYPED_TEST_SUITE(GroupedConvFwdIm2winTestNhwgc, TestTypes);

#define RUN_CL(problem) EXPECT_TRUE((run_im2win_vs_ref_nhwgc<TypeParam>(problem)))

TYPED_TEST(GroupedConvFwdIm2winTestNhwgc, G1_C8_K8_3x3_pad1)
{
    RUN_CL((ConvProblem{1, 1, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1, "G1_C8_K8_3x3_pad1"}));
}

TYPED_TEST(GroupedConvFwdIm2winTestNhwgc, G4_C8_K8_3x3_pad1)
{
    RUN_CL((ConvProblem{4, 1, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1, "G4_C8_K8_3x3_pad1"}));
}

// Primary target: G=32, C=K=8
TYPED_TEST(GroupedConvFwdIm2winTestNhwgc, G32_C8_K8_3x3_pad1)
{
    RUN_CL((ConvProblem{32, 1, 8, 8, 16, 16, 3, 3, 1, 1, 1, 1, "G32_C8_K8_3x3_pad1"}));
}

// C=K=4 (smallest target channel count)
TYPED_TEST(GroupedConvFwdIm2winTestNhwgc, G32_C4_K4_3x3_pad1)
{
    RUN_CL((ConvProblem{32, 1, 4, 4, 16, 16, 3, 3, 1, 1, 1, 1, "G32_C4_K4_3x3_pad1"}));
}

// Stride=2, no pad
TYPED_TEST(GroupedConvFwdIm2winTestNhwgc, G4_C8_K8_3x3_stride2)
{
    RUN_CL((ConvProblem{4, 1, 8, 8, 9, 9, 3, 3, 2, 2, 0, 0, "G4_C8_K8_3x3_stride2"}));
}

// Multi-batch N=2
TYPED_TEST(GroupedConvFwdIm2winTestNhwgc, G4_N2_C8_K8_3x3_pad1)
{
    RUN_CL((ConvProblem{4, 2, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1, "G4_N2_C8_K8_3x3_pad1"}));
}

// Larger spatial
TYPED_TEST(GroupedConvFwdIm2winTestNhwgc, G32_C8_K8_3x3_Hi32Wi32)
{
    RUN_CL((ConvProblem{32, 1, 8, 8, 32, 32, 3, 3, 1, 1, 1, 1, "G32_C8_K8_3x3_Hi32Wi32"}));
}

#undef RUN_CL
