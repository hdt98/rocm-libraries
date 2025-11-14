#include "ck/ck.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/utility/amd_xdlops.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/utility/reduction_operator.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

#define DEBUG_DATA_TYPE float
#define DEBUG_MAX_ENTRY 32
#define DEBUG_LINE_SIZE 64
#include "debug_utils.h"
#undef DEBUG_DATA_TYPE
#undef DEBUG_MAX_ENTRY
#undef DEBUG_LINE_SIZE

#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_multiblock.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

using InpDataType      = ck::half_t;
using WeiDataType      = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using OutDataType      = ck::half_t;

#include "mio_conv_args.h"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InpElementOp = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::UnaryConvert;

static constexpr ck::index_t NDimSpatial = 2;

// using InpLayout = ck::tensor_layout::convolution::GNHWC;
// using WeiLayout = ck::tensor_layout::convolution::GKYXC;
// using OutLayout = ck::tensor_layout::convolution::GNHWK;

// using InpLayout = ck::tensor_layout::convolution::NHWGC;
// using WeiLayout = ck::tensor_layout::convolution::GKYXC;
// using OutLayout = ck::tensor_layout::convolution::NHWGK;

using InpLayout = ck::tensor_layout::convolution::NGCHW;
using WeiLayout = ck::tensor_layout::convolution::GKYXC;
// using WeiLayout = ck::tensor_layout::convolution::GKCYX;
using OutLayout = ck::tensor_layout::convolution::NGKHW;
// using OutLayout = ck::tensor_layout::convolution::GNHWK;

using DeviceGroupedConvNDFwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
        NDimSpatial,
        InpLayout,
        WeiLayout,
        ck::Tuple<>,
        OutLayout,
        InpDataType,
        WeiDataType,
        AccDataType,
        CShuffleDataType,
        ck::Tuple<>,
        OutDataType,
        InpElementOp,
        WeiElementOp,
        OutElementOp,
        ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
        // ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter3x3,
        ck::tensor_operation::device::GemmSpecialization::MNKPadding,

        // NGCHW config 5 - test 5
        // 1,              // NumGemmKPrefetchStage
        // 128,            // BlockSize
        // 64,             // MPerBlock
        // 64,             // NPerBlock
        // 16,             // KPerBlock
        // 4,              // AK1
        // 4,              // BK1
        // 16,             // MPerXdl
        // 16,             // NPerXdl
        // 2,              // MXdlPerWave
        // 1,              // NXdlPerWave
        // S<4, 32, 1>,    // ABlockTransferThreadClusterLengths_AK0_M_AK1
        // S<0, 2, 1>,     // ABlockTransferThreadClusterArrangeOrder
        // S<0, 2, 1>,     // ABlockTransferSrcAccessOrder
        // 1,              // ABlockTransferSrcVectorDim
        // 8,              // ABlockTransferSrcScalarPerVector
        // 4,              // ABlockTransferDstScalarPerVector_AK1
        // 1,              // ABlockLdsExtraM
        // S<4, 32, 1>,    // BBlockTransferThreadClusterLengths_BK0_N_BK1
        // S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
        // S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
        // 2,              // BBlockTransferSrcVectorDim
        // 4,              // BBlockTransferSrcScalarPerVector
        // 4,              // BBlockTransferDstScalarPerVector_BK1
        // 1,              // BBlockLdsExtraN
        // 1,              // CShuffleMXdlPerWavePerShuffle
        // 1,              // CShuffleNXdlPerWavePerShuffle
        // S<1, 16, 1, 4>, // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        // 8               // CDEBlockTransferScalarPerVector_NPerBlock

        // NGCHW config 4 - 4.2
        1,              // NumGemmKPrefetchStage
        64,             // BlockSize
        128,            // MPerBlock
        32,             // NPerBlock
        16,             // KPerBlock
        4,              // AK1
        8,              // BK1
        32,             // MPerXdl
        32,             // NPerXdl
        4,              // MXdlPerWave
        1,              // NXdlPerWave
        S<4, 16, 1>,    // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<0, 2, 1>,     // ABlockTransferThreadClusterArrangeOrder
        S<0, 2, 1>,     // ABlockTransferSrcAccessOrder
        1,              // ABlockTransferSrcVectorDim
        8,              // ABlockTransferSrcScalarPerVector
        4,              // ABlockTransferDstScalarPerVector_AK1
        1,              // ABlockLdsExtraM
        S<2, 32, 1>,    // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
        2,              // BBlockTransferSrcVectorDim
        8,              // BBlockTransferSrcScalarPerVector
        8,              // BBlockTransferDstScalarPerVector_BK1
        1,              // BBlockLdsExtraN
        1,              // CShuffleMXdlPerWavePerShuffle
        1,              // CShuffleNXdlPerWavePerShuffle
        S<1, 16, 1, 4>, // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8               // CDEBlockTransferScalarPerVector_NPerBlock

        // NGCHW config 3 - 5.4
        // 1,              // NumGemmKPrefetchStage
        // 128,            // BlockSize
        // 128,            // MPerBlock
        // 32,             // NPerBlock
        // 32,             // KPerBlock
        // 4,              // AK1
        // 8,              // BK1
        // 32,             // MPerXdl
        // 32,             // NPerXdl
        // 2,              // MXdlPerWave
        // 1,              // NXdlPerWave
        // S<8, 16, 1>,    // ABlockTransferThreadClusterLengths_AK0_M_AK1
        // S<0, 2, 1>,     // ABlockTransferThreadClusterArrangeOrder
        // S<0, 2, 1>,     // ABlockTransferSrcAccessOrder
        // 1,              // ABlockTransferSrcVectorDim
        // 8,              // ABlockTransferSrcScalarPerVector
        // 4,              // ABlockTransferDstScalarPerVector_AK1
        // 1,              // ABlockLdsExtraM
        // S<4, 32, 1>,    // BBlockTransferThreadClusterLengths_BK0_N_BK1
        // S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
        // S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
        // 2,              // BBlockTransferSrcVectorDim
        // 8,              // BBlockTransferSrcScalarPerVector
        // 8,              // BBlockTransferDstScalarPerVector_BK1
        // 1,              // BBlockLdsExtraN
        // 1,              // CShuffleMXdlPerWavePerShuffle
        // 1,              // CShuffleNXdlPerWavePerShuffle
        // S<1, 32, 1, 4>, // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        // 8               // CDEBlockTransferScalarPerVector_NPerBlock

        // NGCHW config 2 - 5.6
        // 1,              // NumGemmKPrefetchStage
        // 128,            // BlockSize
        // 128,            // MPerBlock
        // 32,             // NPerBlock
        // 32,             // KPerBlock
        // 4,              // AK1
        // 8,              // BK1
        // 32,             // MPerXdl
        // 32,             // NPerXdl
        // 2,              // MXdlPerWave
        // 1,              // NXdlPerWave
        // S<8, 16, 1>,    // ABlockTransferThreadClusterLengths_AK0_M_AK1
        // S<0, 2, 1>,     // ABlockTransferThreadClusterArrangeOrder
        // S<0, 2, 1>,     // ABlockTransferSrcAccessOrder
        // 1,              // ABlockTransferSrcVectorDim
        // 8,              // ABlockTransferSrcScalarPerVector
        // 4,              // ABlockTransferDstScalarPerVector_AK1
        // 1,              // ABlockLdsExtraM
        // S<4, 32, 1>,    // BBlockTransferThreadClusterLengths_BK0_N_BK1
        // S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
        // S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
        // 2,              // BBlockTransferSrcVectorDim
        // 8,              // BBlockTransferSrcScalarPerVector
        // 8,              // BBlockTransferDstScalarPerVector_BK1
        // 1,              // BBlockLdsExtraN
        // 1,              // CShuffleMXdlPerWavePerShuffle
        // 1,              // CShuffleNXdlPerWavePerShuffle
        // S<1, 32, 1, 4>, // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        // 8               // CDEBlockTransferScalarPerVector_NPerBlock

        // NGCHW config 1 - 6.9
        // 1,              // NumGemmKPrefetchStage
        // 64,             // BlockSize
        // 128,            // MPerBlock
        // 32,             // NPerBlock
        // 32,             // KPerBlock
        // 8,              // AK1
        // 8,              // BK1
        // 32,             // MPerXdl
        // 32,             // NPerXdl
        // 4,              // MXdlPerWave
        // 1,              // NXdlPerWave
        // S<4, 16, 1>,    // ABlockTransferThreadClusterLengths_AK0_M_AK1
        // S<0, 2, 1>,     // ABlockTransferThreadClusterArrangeOrder
        // S<0, 2, 1>,     // ABlockTransferSrcAccessOrder
        // 1,              // ABlockTransferSrcVectorDim
        // 8,              // ABlockTransferSrcScalarPerVector
        // 8,              // ABlockTransferDstScalarPerVector_AK1
        // 1,              // ABlockLdsExtraM
        // S<4, 16, 1>,    // BBlockTransferThreadClusterLengths_BK0_N_BK1
        // S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
        // S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
        // 2,              // BBlockTransferSrcVectorDim
        // 8,              // BBlockTransferSrcScalarPerVector
        // 8,              // BBlockTransferDstScalarPerVector_BK1
        // 1,              // BBlockLdsExtraN
        // 1,              // CShuffleMXdlPerWavePerShuffle
        // 1,              // CShuffleNXdlPerWavePerShuffle
        // S<1, 16, 1, 4>, // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        // 8               // CDEBlockTransferScalarPerVector_NPerBlock

        // NGCHW config 0 - 10.7
        // 1,              // NumGemmKPrefetchStage
        // 64,             // BlockSize
        // 64,             // MPerBlock
        // 64,             // NPerBlock
        // 32,             // KPerBlock
        // 8,              // AK1
        // 8,              // BK1
        // 32,             // MPerXdl
        // 32,             // NPerXdl
        // 2,              // MXdlPerWave
        // 2,              // NXdlPerWave
        // S<4, 16, 1>,    // ABlockTransferThreadClusterLengths_AK0_M_AK1
        // S<0, 2, 1>,     // ABlockTransferThreadClusterArrangeOrder
        // S<0, 2, 1>,     // ABlockTransferSrcAccessOrder
        // 1,              // ABlockTransferSrcVectorDim
        // 4,              // ABlockTransferSrcScalarPerVector
        // 8,              // ABlockTransferDstScalarPerVector_AK1
        // 1,              // ABlockLdsExtraM
        // S<4, 16, 1>,    // BBlockTransferThreadClusterLengths_BK0_N_BK1
        // S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
        // S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
        // 2,              // BBlockTransferSrcVectorDim
        // 8,              // BBlockTransferSrcScalarPerVector
        // 8,              // BBlockTransferDstScalarPerVector_BK1
        // 1,              // BBlockLdsExtraN
        // 1,              // CShuffleMXdlPerWavePerShuffle
        // 1,              // CShuffleNXdlPerWavePerShuffle
        // S<1, 16, 1, 4>, // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        // 8               // CDEBlockTransferScalarPerVector_NPerBlock

        >;

int main(int argc, char* argv[])
{
    // args made compatible with MIOpenDriver short args
    ParseHostArgs(argc, argv);
    HIP_CHECK_ERROR(hipSetDevice(1));

    ck::utils::conv::ConvParam conv_param{NDimSpatial,
                                          G,
                                          N,
                                          K / G,
                                          C / G,
                                          {Y, X},
                                          {H, W},
                                          {conv_stride_h, conv_stride_w},
                                          {dilation_h, dilation_w},
                                          {pad_h, pad_w},
                                          {pad_h, pad_w}};

    const auto inp_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InpLayout>(conv_param);
    const auto wei_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
    const auto out_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    Tensor<InpDataType> inp(inp_desc);
    Tensor<WeiDataType> wei(wei_desc);
    Tensor<OutDataType> out(out_desc);

    std::cout << "Host tensors: " << std::endl;
    std::cout << "inp: " << inp.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out.mDesc << std::endl;

    constexpr int modv = 10;
    inp.GenerateTensorValue([&](auto... is) { return inp.GetOffsetFromMultiIndex(is...) % modv; });

    // stay consistant with our pytorch debug reference
    {
        HostTensorDescriptor whack =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
                ck::tensor_layout::convolution::GKCYX>(conv_param);
        wei.GenerateTensorValue(
            [&](auto... is) { return whack.GetOffsetFromMultiIndex(is...) % modv; });
    }

    // inp.GenerateTensorValue(GeneratorTensor_2<InpDataType>{-5, 5});
    // wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});

    out.SetZero();

    DeviceMem inp_device_buf(sizeof(InpDataType) * inp.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());

    inp_device_buf.ToDevice(inp.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    std::array<ck::index_t, NDimSpatial + 3> inp_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> inp_strides{};
    std::array<ck::index_t, NDimSpatial + 3> wei_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> wei_strides{};
    std::array<ck::index_t, NDimSpatial + 3> out_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> out_strides{};

    auto copy = [](auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(inp_desc.GetLengths(), inp_lengths);
    copy(inp_desc.GetStrides(), inp_strides);
    copy(wei_desc.GetLengths(), wei_lengths);
    copy(wei_desc.GetStrides(), wei_strides);
    copy(out_desc.GetLengths(), out_lengths);
    copy(out_desc.GetStrides(), out_strides);

    auto conv     = DeviceGroupedConvNDFwdInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(inp_device_buf.GetDeviceBuffer(),
                                      wei_device_buf.GetDeviceBuffer(),
                                      std::array<const void*, 0>{},
                                      out_device_buf.GetDeviceBuffer(),
                                      inp_lengths,
                                      inp_strides,
                                      wei_lengths,
                                      wei_strides,
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{{}},
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{{}},
                                      out_lengths,
                                      out_strides,
                                      {conv_stride_h, conv_stride_w}, // conv_filter_strides
                                      {dilation_h, dilation_w},       // conv_filter_dilations
                                      {pad_h, pad_w},                 // input_left_pads
                                      {pad_h, pad_w},                 // input_right_pads
                                      InpElementOp{},
                                      WeiElementOp{},
                                      OutElementOp{});

    if(!conv.IsSupportedArgument(argument))
    {
        std::cout << "Unsupported arguments" << std::endl;
        return 0;
    }

    std::cout << "Convert conv to gemm: " << std::endl;
    argument.Print();

    auto stream_config = StreamConfig{nullptr, // stream_id
                                      true,    // time_kernel
                                      0,       // log_level

                                      1, // cold_niters
                                      5, // nrepeat

                                      //   5,  // cold_niters
                                      //   20, // nrepeat

                                      true, // flush_cache
                                      1};   // rotating_count

    float avg_time = invoker.Run(argument, stream_config);

    std::cout << "Perf: " << avg_time << " ms\n";

    out_device_buf.FromDevice(out.mData.data());

    // PrintTensor(inp);
    // PrintTensor(wei);
    // PrintTensor(out);

    // PrintDebugBuf();

    constexpr bool do_verification = true;
    if(do_verification)
    {
        Tensor<OutDataType> out_host(out_desc);

        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<
            NDimSpatial,
            InpDataType,
            WeiDataType,
            OutDataType,
            InpElementOp,
            WeiElementOp,
            ck::tensor_operation::element_wise::PassThrough>();

        auto ref_invoker = ref_conv.MakeInvoker();
        auto ref_argument =
            ref_conv.MakeArgument(inp,
                                  wei,
                                  out_host,
                                  conv_param.conv_filter_strides_,
                                  conv_param.conv_filter_dilations_,
                                  conv_param.input_left_pads_,
                                  conv_param.input_right_pads_,
                                  InpElementOp{},
                                  WeiElementOp{},
                                  ck::tensor_operation::element_wise::PassThrough{});

        ref_invoker.Run(ref_argument);
        // cde_elementwise
        out_host.ForEach([&](auto&, auto idx) { OutElementOp{}(out_host(idx), out_host(idx)); });

        // PrintTensor(out_host);

        auto valid = ck::utils::check_err(
            out.mData, out_host.mData, "Error: incorrect results!", 1e-5f, 1e-4f);

        if(valid)
            std::cout << "Test pass\n";

        return 0;
    }
}