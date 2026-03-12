// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// ═══════════════════════════════════════════════════════════════════════
// run_grouped_conv_fwd_im2win_example.inc
//
// Shared test/benchmark harness for the im2win forward convolution example.
//
// Layout contract:
//   Input:  GNCHW   (channels-first)
//   Weight: GKCYX   (channels-first)
//   Output: NHWGK   (channels-last, K innermost — required by epilogue)
//
// For verification the naive GPU reference (NHWGC/GKYXC/NHWGK) is used.
// The GKCYX→GKYXC weight transpose is performed on-device using
// BatchedTransposeKernel; the GNCHW→NHWGC input reorder uses an explicit
// strided HostTensor copy (the G/N interleaving makes a single 2D batched
// transpose insufficient).
// ═══════════════════════════════════════════════════════════════════════

#include "ck_tile/ref/naive_grouped_conv_fwd_gpu.hpp"
#include "ck_tile/ops/batched_transpose.hpp"
#include "../20_grouped_convolution/grouped_convolution_utils.hpp"

// ── On-device weight transpose: GKCYX → GKYXC ────────────────────────────────
// Views GKCYX as [G*K, C, Y*X] and transposes to [G*K, Y*X, C] = GKYXC.
template <typename DataType>
static void transpose_gkcyx_to_gkyxc(const ck_tile::DeviceMem& d_src,
                                      ck_tile::DeviceMem& d_dst,
                                      int G,
                                      int K,
                                      int C,
                                      int Y,
                                      int X)
{
    using BlockTile  = ck_tile::sequence<64, 64>;
    using WarpLayout = ck_tile::sequence<1, 1>;
    using Problem    = ck_tile::BatchedTransposeProblem<DataType,
                                                        BlockTile,
                                                        WarpLayout,
                                                        /*kPadM=*/true,
                                                        /*kPadN=*/true>;
    using Kernel = ck_tile::BatchedTransposeKernel<ck_tile::BatchedTransposePipeline<Problem>>;

    const ck_tile::BatchedTransposeHostArgs hargs{
        d_src.GetDeviceBuffer(),
        d_dst.GetDeviceBuffer(),
        G * K,   // batch
        C,       // height  (dim to become the new "width")
        Y * X,   // width   (dim to become the new "height")
        C * Y * X,
        BlockTile::at(ck_tile::number<0>{}),
        BlockTile::at(ck_tile::number<1>{})};

    auto kargs = Kernel::MakeKargs(hargs);
    ck_tile::launch_kernel(ck_tile::stream_config{},
                           ck_tile::make_kernel<1>(
                               Kernel{}, Kernel::GridSize(hargs), Kernel::BlockSize(), 0, kargs));
}

// ── Host-side input reorder: GNCHW → NHWGC ───────────────────────────────────
// The NHWGC target interleaves G and N dimensions (dst[n,h,w,g,c] = src[g,n,c,h,w])
// which cannot be expressed as a single 2D batched transpose.
template <typename DataType>
static ck_tile::HostTensor<DataType> reorder_gnchw_to_nhwgc(
    const ck_tile::HostTensor<DataType>& src, int G, int N, int C, int H, int W)
{
    ck_tile::HostTensor<DataType> dst({static_cast<std::size_t>(N),
                                       static_cast<std::size_t>(H),
                                       static_cast<std::size_t>(W),
                                       static_cast<std::size_t>(G),
                                       static_cast<std::size_t>(C)});
    const int sG = N * C * H * W;
    const int sN = C * H * W;
    const int sC = H * W;
    for(int g = 0; g < G; ++g)
        for(int n = 0; n < N; ++n)
            for(int h = 0; h < H; ++h)
                for(int w = 0; w < W; ++w)
                    for(int c = 0; c < C; ++c)
                    {
                        int si = g * sG + n * sN + c * sC + h * W + w;
                        int di = n * (H * W * G * C) + h * (W * G * C) + w * (G * C) + g * C + c;
                        dst.mData[di] = src.mData[si];
                    }
    return dst;
}

// ── invoke_grouped_conv_fwd_im2win — kernel launch + perf metrics ─────────────
template <ck_tile::index_t NDimSpatial,
          typename ConvConfig,
          typename Invoker,
          typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
float invoke_grouped_conv_fwd_im2win(const ck_tile::GroupedConvFwdHostArgs<>& args,
                                     int n_warmup,
                                     int n_repeat)
{
    float ave_time = Invoker::template grouped_conv_fwd<NDimSpatial,
                                                        ConvConfig,
                                                        InDataType,
                                                        WeiDataType,
                                                        AccDataType,
                                                        OutDataType,
                                                        InLayout,
                                                        WeiLayout,
                                                        OutLayout>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});

    std::size_t flop     = args.GetFlops();
    std::size_t num_byte = args.GetByte<InDataType, WeiDataType, OutDataType>();
    float tflops         = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec     = num_byte / 1.E6 / ave_time;

    std::cout << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    return ave_time;
}

// ── run_grouped_conv_fwd_im2win_example_with_layouts ─────────────────────────
template <ck_tile::index_t NDimSpatial,
          typename ConvConfig,
          typename Invoker,
          typename InDataType,
          typename WeiDataType = InDataType,
          typename OutDataType = InDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
int run_grouped_conv_fwd_im2win_example_with_layouts(
    int argc, char* argv[], const InLayout, const WeiLayout, const OutLayout)
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using AccDataType = float;

    std::vector<ck_tile::index_t> filter_spatial_lengths;
    std::vector<ck_tile::index_t> image_spatial_lengths;
    std::vector<ck_tile::index_t> strides;
    std::vector<ck_tile::index_t> dilations;
    std::vector<ck_tile::index_t> lpads;
    std::vector<ck_tile::index_t> rpads;

    const ck_tile::index_t num_dim_sp = fill_spatial_dimensions(filter_spatial_lengths,
                                                                 image_spatial_lengths,
                                                                 strides,
                                                                 dilations,
                                                                 lpads,
                                                                 rpads,
                                                                 arg_parser);

    ck_tile::conv::ConvParam conv_param{num_dim_sp,
                                        arg_parser.get_int("g"),
                                        arg_parser.get_int("n"),
                                        arg_parser.get_int("k"),
                                        arg_parser.get_int("c"),
                                        filter_spatial_lengths,
                                        image_spatial_lengths,
                                        strides,
                                        dilations,
                                        lpads,
                                        rpads};

    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");

    // ── Build host tensors with GNCHW / GKCYX / NHWGK descriptors ────────────
    const auto in_g_n_c_wis_desc =
        ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
    const auto wei_g_k_c_xs_desc =
        ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
    const auto out_g_n_k_wos_desc =
        ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    ck_tile::HostTensor<InDataType>  input(in_g_n_c_wis_desc);
    ck_tile::HostTensor<WeiDataType> weight(wei_g_k_c_xs_desc);
    ck_tile::HostTensor<OutDataType> output(out_g_n_k_wos_desc);

    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(input);
        ck_tile::FillUniformDistribution<WeiDataType>{-5.f, 5.f}(weight);
    }
    else if(init_method == 1)
    {
        ck_tile::FillMonotonicSeq<InDataType>{}(input);
        ck_tile::FillMonotonicSeq<WeiDataType>{}(weight);
    }
    else if(init_method == 2)
    {
        ck_tile::FillUniformDistribution<InDataType>{1.f, 1.f}(input);
        ck_tile::FillUniformDistribution<WeiDataType>{1.f, 1.f}(weight);
    }
    else
    {
        input.SetZero();
        weight.SetZero();
    }

    ck_tile::DeviceMem input_buf(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem weight_buf(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem output_buf(output.get_element_space_size_in_bytes());

    input_buf.ToDevice(input.data());
    weight_buf.ToDevice(weight.data());
    output_buf.SetZero();

    ck_tile::GroupedConvFwdHostArgs<> args(conv_param,
                                           input_buf.GetDeviceBuffer(),
                                           weight_buf.GetDeviceBuffer(),
                                           {},
                                           output_buf.GetDeviceBuffer(),
                                           kbatch);

    std::cout << "Run im2win Grouped Conv Fwd" << std::endl;
    std::cout << "  input:  " << input.mDesc  << std::endl;
    std::cout << "  weight: " << weight.mDesc << std::endl;
    std::cout << "  output: " << output.mDesc << std::endl;

    invoke_grouped_conv_fwd_im2win<NDimSpatial,
                                   ConvConfig,
                                   Invoker,
                                   InDataType,
                                   WeiDataType,
                                   AccDataType,
                                   OutDataType,
                                   InLayout,
                                   WeiLayout,
                                   OutLayout>(args, n_warmup, n_repeat);

    output_buf.FromDevice(output.data());
    bool pass = true;

    const int verify = arg_parser.get_int("v");

    if(verify == 1)
    {
        // ── CPU reference ─────────────────────────────────────────────────────
        // The CPU reference works in-place on host tensors with arbitrary layout,
        // so it can directly consume the GNCHW input and GKCYX weight and compare
        // against the NHWGK output.
        ck_tile::HostTensor<OutDataType> output_ref(out_g_n_k_wos_desc);
        output_ref.SetZero();

        ck_tile::reference_grouped_conv_fwd<NDimSpatial, InDataType, WeiDataType, OutDataType>(
            input,
            weight,
            output_ref,
            conv_param.conv_filter_strides_,
            conv_param.conv_filter_dilations_,
            conv_param.input_left_pads_,
            conv_param.input_right_pads_);

        const ck_tile::index_t GemmK =
            weight.get_element_size() / (conv_param.G_ * conv_param.K_);
        const float max_val =
            *std::max_element(output_ref.mData.begin(), output_ref.mData.end());
        const auto rtol_atol =
            calculate_rtol_atol<InDataType, WeiDataType, AccDataType, OutDataType>(
                GemmK, kbatch, max_val);

        pass = ck_tile::check_err(output,
                                  output_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "  rtol=" << rtol_atol.at(ck_tile::number<0>{})
                  << " atol=" << rtol_atol.at(ck_tile::number<1>{}) << std::endl;
        std::cout << "CPU verification: " << (pass ? "PASSED" : "FAILED") << std::endl;
    }
    else if(verify == 2)
    {
        // ── GPU reference ─────────────────────────────────────────────────────
        // The naive GPU reference kernel uses NHWGC input and GKYXC weight.
        //   • Weight GKCYX→GKYXC: on-device BatchedTransposeKernel
        //   • Input  GNCHW→NHWGC: host-side strided copy (G/N interleaving)

        // Weight transpose on device: [G*K, C, Y*X] → [G*K, Y*X, C]
        ck_tile::DeviceMem weight_gkyxc_buf(weight.get_element_space_size_in_bytes());
        transpose_gkcyx_to_gkyxc<WeiDataType>(weight_buf,
                                              weight_gkyxc_buf,
                                              static_cast<int>(conv_param.G_),
                                              static_cast<int>(conv_param.K_),
                                              static_cast<int>(conv_param.C_),
                                              static_cast<int>(conv_param.filter_spatial_lengths_[0]),
                                              static_cast<int>(conv_param.filter_spatial_lengths_[1]));

        // Input reorder on host: GNCHW → NHWGC
        auto input_nhwgc = reorder_gnchw_to_nhwgc(input,
                                                   static_cast<int>(conv_param.G_),
                                                   static_cast<int>(conv_param.N_),
                                                   static_cast<int>(conv_param.C_),
                                                   static_cast<int>(conv_param.input_spatial_lengths_[0]),
                                                   static_cast<int>(conv_param.input_spatial_lengths_[1]));

        ck_tile::DeviceMem input_ref_buf(input_nhwgc.get_element_space_size_in_bytes());
        ck_tile::DeviceMem output_ref_buf(output.get_element_space_size_in_bytes());
        input_ref_buf.ToDevice(input_nhwgc.data());
        output_ref_buf.SetZero();

        std::cout << "  Running GPU reference kernel..." << std::endl;
        ck_tile::naive_grouped_conv_fwd<NDimSpatial, InDataType, WeiDataType, OutDataType>(
            reinterpret_cast<const InDataType*>(input_ref_buf.GetDeviceBuffer()),
            reinterpret_cast<const WeiDataType*>(weight_gkyxc_buf.GetDeviceBuffer()),
            reinterpret_cast<OutDataType*>(output_ref_buf.GetDeviceBuffer()),
            conv_param.G_,
            conv_param.N_,
            conv_param.K_,
            conv_param.C_,
            conv_param.input_spatial_lengths_,
            conv_param.filter_spatial_lengths_,
            conv_param.output_spatial_lengths_,
            conv_param.conv_filter_strides_,
            conv_param.conv_filter_dilations_,
            conv_param.input_left_pads_);

        ck_tile::HostTensor<OutDataType> output_ref(out_g_n_k_wos_desc);
        output_ref_buf.FromDevice(output_ref.data());

        const ck_tile::index_t GemmK =
            weight.get_element_size() / (conv_param.G_ * conv_param.K_);
        const float max_val =
            *std::max_element(output_ref.mData.begin(), output_ref.mData.end());
        const auto rtol_atol =
            calculate_rtol_atol<InDataType, WeiDataType, AccDataType, OutDataType>(
                GemmK, kbatch, max_val);

        pass = ck_tile::check_err(output,
                                  output_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "  rtol=" << rtol_atol.at(ck_tile::number<0>{})
                  << " atol=" << rtol_atol.at(ck_tile::number<1>{}) << std::endl;
        std::cout << "GPU verification: " << (pass ? "PASSED" : "FAILED") << std::endl;
    }

    return pass ? 0 : 1;
}

// ── run_grouped_conv_fwd_im2win_example_prec_type ─────────────────────────────
// Dispatches to the correct NDimSpatial based on layout strings.
// Only 2D is supported by the im2win kernel (GNCHW/GKCYX/NHWGK).
template <typename Invoker,
          typename ConvConfig,
          typename InPrecType,
          typename WeiPrecType = InPrecType,
          typename OutPrecType = InPrecType>
int run_grouped_conv_fwd_im2win_example_prec_type(
    std::string in_layout, std::string wei_layout, std::string out_layout, int argc, char* argv[])
{
    using GNCHW = ck_tile::tensor_layout::convolution::GNCHW;
    using GKCYX = ck_tile::tensor_layout::convolution::GKCYX;
    using NHWGK = ck_tile::tensor_layout::convolution::NHWGK;

    if(in_layout == "GNCHW" && wei_layout == "GKCYX" && out_layout == "NHWGK")
    {
        return run_grouped_conv_fwd_im2win_example_with_layouts<ck_tile::number<2>{},
                                                                 ConvConfig,
                                                                 Invoker,
                                                                 InPrecType,
                                                                 WeiPrecType,
                                                                 OutPrecType>(
            argc, argv, GNCHW{}, GKCYX{}, NHWGK{});
    }
    else
    {
        throw std::runtime_error(
            "im2win example only supports GNCHW/GKCYX/NHWGK layout combination.\n"
            "Pass --in_layout GNCHW --wei_layout GKCYX --out_layout NHWGK\n");
    }
}
