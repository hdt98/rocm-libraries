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
//   3. Run im2win kernel → output (GNKHW) on device.
//   4. Run naive GPU reference kernel (NHWGC/GKYXC/NHWGK layout) on
//      separately-layout-transposed device buffers.
//   5. Transpose the reference output back to GNKHW and compare
//      element-wise with the im2win result.
//
// The naive GPU reference is the same kernel used by the existing
// grouped conv forward example (ck_tile/ref/naive_grouped_conv_fwd_gpu.hpp)
// and provides an independent correctness oracle.
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

#include "grouped_conv_fwd_im2win_invoker.hpp"

// ── Layout helpers ────────────────────────────────────────────────────────────
// The naive GPU ref operates on NHWGC / GKYXC / NHWGK (channels-last).
// Our im2win kernel operates on GNCHW / GKCYX / GNKHW (channels-first).
// We perform layout transpositions on host before/after GPU reference.

// GNCHW → NHWGC  (input reorder for ref)
template <typename T>
static ck_tile::HostTensor<T>
gnchw_to_nhwgc(const ck_tile::HostTensor<T>& src, int G, int N, int C, int H, int W)
{
    // src shape: [G, N, C, H, W], dst shape: [N, H, W, G, C]
    ck_tile::HostTensor<T> dst({static_cast<std::size_t>(N),
                                static_cast<std::size_t>(H),
                                static_cast<std::size_t>(W),
                                static_cast<std::size_t>(G),
                                static_cast<std::size_t>(C)});
    for(int g = 0; g < G; ++g)
        for(int n = 0; n < N; ++n)
            for(int c = 0; c < C; ++c)
                for(int h = 0; h < H; ++h)
                    for(int w = 0; w < W; ++w)
                    {
                        // src: g*N*C*H*W + n*C*H*W + c*H*W + h*W + w
                        std::size_t si = g * (N * C * H * W) + n * (C * H * W) + c * (H * W) +
                                         h * W + w;
                        // dst: n*H*W*G*C + h*W*G*C + w*G*C + g*C + c
                        std::size_t di = n * (H * W * G * C) + h * (W * G * C) + w * (G * C) +
                                         g * C + c;
                        dst.mData[di] = src.mData[si];
                    }
    return dst;
}

// GKCYX → GKYXC  (weight reorder for ref)
template <typename T>
static ck_tile::HostTensor<T>
gkcyx_to_gkyxc(const ck_tile::HostTensor<T>& src, int G, int K, int C, int Y, int X)
{
    // src shape: [G, K, C, Y, X], dst shape: [G, K, Y, X, C]
    ck_tile::HostTensor<T> dst({static_cast<std::size_t>(G),
                                static_cast<std::size_t>(K),
                                static_cast<std::size_t>(Y),
                                static_cast<std::size_t>(X),
                                static_cast<std::size_t>(C)});
    for(int g = 0; g < G; ++g)
        for(int k = 0; k < K; ++k)
            for(int c = 0; c < C; ++c)
                for(int y = 0; y < Y; ++y)
                    for(int x = 0; x < X; ++x)
                    {
                        std::size_t si = g * (K * C * Y * X) + k * (C * Y * X) + c * (Y * X) +
                                         y * X + x;
                        std::size_t di = g * (K * Y * X * C) + k * (Y * X * C) + y * (X * C) +
                                         x * C + c;
                        dst.mData[di] = src.mData[si];
                    }
    return dst;
}

// NHWGK → GNKHW  (output reorder: ref → im2win layout)
template <typename T>
static ck_tile::HostTensor<T>
nhwgk_to_gnkhw(const ck_tile::HostTensor<T>& src, int G, int N, int K, int H, int W)
{
    // src shape: [N, H, W, G, K], dst shape: [G, N, K, H, W]
    ck_tile::HostTensor<T> dst({static_cast<std::size_t>(G),
                                static_cast<std::size_t>(N),
                                static_cast<std::size_t>(K),
                                static_cast<std::size_t>(H),
                                static_cast<std::size_t>(W)});
    for(int g = 0; g < G; ++g)
        for(int n = 0; n < N; ++n)
            for(int k = 0; k < K; ++k)
                for(int h = 0; h < H; ++h)
                    for(int w = 0; w < W; ++w)
                    {
                        // src: n*H*W*G*K + h*W*G*K + w*G*K + g*K + k
                        std::size_t si = n * (H * W * G * K) + h * (W * G * K) + w * (G * K) +
                                         g * K + k;
                        // dst: g*N*K*H*W + n*K*H*W + k*H*W + h*W + w
                        std::size_t di = g * (N * K * H * W) + n * (K * H * W) + k * (H * W) +
                                         h * W + w;
                        dst.mData[di] = src.mData[si];
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
    // We use NHWGK for the output because:
    //   1. The CShuffleEpilogue writes M×N tiles where N=K.  K must be
    //      contiguous (stride 1) in global memory for vectorised stores.
    //   2. GNKHW has stride(K) = Ho×Wo — not contiguous → wrong writes.
    //   3. The naive GPU reference also outputs NHWGK, so no transpose
    //      is needed for comparison.
    using InLayout  = ck_tile::tensor_layout::convolution::GNCHW;
    using WeiLayout = ck_tile::tensor_layout::convolution::GKCYX;
    using OutLayout = ck_tile::tensor_layout::convolution::NHWGK;
    using Config    = Im2winConvTestConfig<DataType>;

    // ── Host tensors ──────────────────────────────────────────────────
    // Input: GNCHW = [G, N, C, Hi, Wi]
    ck_tile::HostTensor<DataType> h_input({static_cast<std::size_t>(p.G),
                                           static_cast<std::size_t>(p.N),
                                           static_cast<std::size_t>(p.C),
                                           static_cast<std::size_t>(p.Hi),
                                           static_cast<std::size_t>(p.Wi)});
    // Weight: GKCYX = [G, K, C, Y, X] (channels-first)
    // The naive GPU reference uses GKYXC; we'll reorder below.
    ck_tile::HostTensor<DataType> h_weight({static_cast<std::size_t>(p.G),
                                            static_cast<std::size_t>(p.K),
                                            static_cast<std::size_t>(p.C),
                                            static_cast<std::size_t>(p.Y),
                                            static_cast<std::size_t>(p.X)});

    ck_tile::FillUniformDistribution<DataType>{-2.f, 2.f}(h_input);
    ck_tile::FillUniformDistribution<DataType>{-2.f, 2.f}(h_weight);

    // ── Device buffers ────────────────────────────────────────────────
    const std::size_t out_elems =
        static_cast<std::size_t>(p.N) * Ho * Wo * p.G * p.K;
    ck_tile::DeviceMem d_input(h_input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_weight(h_weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_output(out_elems * sizeof(DataType));

    d_input.ToDevice(h_input.data());
    d_weight.ToDevice(h_weight.data());
    d_output.SetZero();

    // ── Build ConvParam and run im2win kernel ─────────────────────────
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

    // ── GPU reference (also NHWGK) ─────────────────────────────────────
    // The naive GPU ref uses NHWGC input and GKYXC weight; reorder our
    // channels-first host tensors to channels-last for it.
    auto h_input_nhwgc  = gnchw_to_nhwgc(h_input,  p.G, p.N, p.C, p.Hi, p.Wi);
    auto h_weight_gkyxc = gkcyx_to_gkyxc(h_weight, p.G, p.K, p.C, p.Y,  p.X);

    ck_tile::DeviceMem d_input_ref(h_input_nhwgc.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_weight_ref(h_weight_gkyxc.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_output_ref(out_elems * sizeof(DataType));

    d_input_ref.ToDevice(h_input_nhwgc.data());
    d_weight_ref.ToDevice(h_weight_gkyxc.data());
    d_output_ref.SetZero();

    ck_tile::naive_grouped_conv_fwd<2, DataType, DataType, DataType>(
        reinterpret_cast<const DataType*>(d_input_ref.GetDeviceBuffer()),
        reinterpret_cast<const DataType*>(d_weight_ref.GetDeviceBuffer()),
        reinterpret_cast<DataType*>(d_output_ref.GetDeviceBuffer()),
        p.G, p.N, p.K, p.C,
        {static_cast<ck_tile::long_index_t>(p.Hi), static_cast<ck_tile::long_index_t>(p.Wi)},
        {static_cast<ck_tile::long_index_t>(p.Y),  static_cast<ck_tile::long_index_t>(p.X)},
        {static_cast<ck_tile::long_index_t>(Ho),   static_cast<ck_tile::long_index_t>(Wo)},
        {static_cast<ck_tile::long_index_t>(p.Sy), static_cast<ck_tile::long_index_t>(p.Sx)},
        {1LL, 1LL},
        {static_cast<ck_tile::long_index_t>(p.LPH),
         static_cast<ck_tile::long_index_t>(p.LPW)});

    // Copy GPU reference output (NHWGK) to host.
    ck_tile::HostTensor<DataType> h_output_ref({static_cast<std::size_t>(p.N),
                                                static_cast<std::size_t>(Ho),
                                                static_cast<std::size_t>(Wo),
                                                static_cast<std::size_t>(p.G),
                                                static_cast<std::size_t>(p.K)});
    d_output_ref.FromDevice(h_output_ref.data());

    // ── Compute tolerance ─────────────────────────────────────────────
    const int GemmK = p.C * p.Y * p.X;
    const float max_val = *std::max_element(h_output_ref.mData.begin(),
                                            h_output_ref.mData.end());
    const auto rtol = ck_tile::get_relative_threshold<DataType, DataType, AccDataType>(GemmK);
    const auto atol = ck_tile::get_absolute_threshold<DataType, DataType, AccDataType>(
        max_val, GemmK);

    // ── Compare ───────────────────────────────────────────────────────
    bool pass = ck_tile::check_err(
        h_output_im2win, h_output_ref, "im2win vs GPU reference", rtol, atol);

    if(!pass)
        std::cout << "[FAIL] " << p.label << "  rtol=" << rtol << " atol=" << atol << "\n";

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
