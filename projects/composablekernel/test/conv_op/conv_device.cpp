// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

// #define CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4 1

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_conv_fwd_wconv.hpp"

//#include "windows.h"

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::UnaryConvert;

#define DEFAULT_H 64
#define DEFAULT_W 64
#define DEFAULT_C 16
#define DEFAULT_K 16

#define DEFAULT_H_PERWAVE 8
#define DEFAULT_W_PERWAVE 8
#define DEFAULT_H_PERBLOCK 16
#define DEFAULT_W_PERBLOCK 16
#define DEFAULT_C_PERBLOCK 16
#define DEFAULT_K_PERBLOCK 16
#define DEFAULT_BLOCKSIZE 128

enum ShapeType
{
    Shape_4X2,
    Shape_4X4,
    Shape_8X4,
};

enum FilterType
{
    Filter_1X1,
    Filter_3X3
};

template <typename T>
struct Debug;


using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <typename InputLay, typename WeightLay, typename OutputLay>
struct CommonLayoutSetting
{
    using InputLayout  = InputLay;
    using WeightLayout = WeightLay;
    using OutputLayout = OutputLay;
};

template <ck::index_t NDimSpatial>
struct CommonLayoutSettingSelector;

namespace ctl = ck::tensor_layout::convolution;

template <>
struct CommonLayoutSettingSelector<1> final
    : CommonLayoutSetting<ctl::G_NW_C, ctl::G_K_X_C, ctl::G_NW_K>
{
};

template <>
struct CommonLayoutSettingSelector<2> final
    : CommonLayoutSetting<ctl::G_NHW_C, ctl::G_K_YX_C, ctl::G_NHW_K>
{
};

template <>
struct CommonLayoutSettingSelector<3> final
    : CommonLayoutSetting<ctl::G_NDHW_C, ctl::G_K_ZYX_C, ctl::G_NDHW_K>
{
};

template <ck::index_t NDimSpatial>
using InputLayout = typename CommonLayoutSettingSelector<NDimSpatial>::InputLayout;

template <ck::index_t NDimSpatial>
using WeightLayout = typename CommonLayoutSettingSelector<NDimSpatial>::WeightLayout;

template <ck::index_t NDimSpatial>
using OutputLayout = typename CommonLayoutSettingSelector<NDimSpatial>::OutputLayout;

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
    int h                = DEFAULT_H;
    int w                = DEFAULT_W;
    int c                = DEFAULT_C;
    int k                = DEFAULT_K;
    uint32_t test_mask   = 0xffffffff;
};

ExecutionConfig config;

template <typename DataType>
void DumpTensor(const Tensor<DataType>& tensor, const char* str)
{
    assert(tensor.GetNumOfDimension() == 5);
    auto lengths = tensor.GetLengths();
    std::cout << str << "  [ " << std::endl;
    for(uint32_t i0 = 0; i0 < lengths[0]; i0++)
    {
        if(lengths[1] > 1)
        {
            std::cout << "  [";
        }
        for(uint32_t i1 = 0; i1 < lengths[1]; i1++)
        {
            if(lengths[2] > 1)
            {
                std::cout << "  [";
            }
            for(uint32_t i2 = 0; i2 < lengths[2]; i2++)
            {
                if(lengths[3] > 1)
                {
                    std::cout << "  [";
                }
                for(uint32_t i3 = 0; i3 < lengths[3]; i3++)
                {
                    if(lengths[4] > 1)
                    {
                        std::cout << "  [";
                    }
                    for(uint32_t i4 = 0; i4 < lengths[4]; i4++)
                    {
                        std::vector<std::size_t> idx({i0, i1, i2, i3, i4});
                        std::cout << ck::type_convert<float>(tensor(idx)) << ", ";
                    }
                    if(lengths[4] > 1)
                    {
                        std::cout << "]" << std::endl;
                    }
                }
                if(lengths[3] > 1)
                {
                    std::cout << "]" << std::endl;
                }
            }
            if(lengths[2] > 1)
            {
                std::cout << "]" << std::endl;
            }
        }
        if(lengths[1] > 1)
        {
            std::cout << "]" << std::endl;
        }
    }
    std::cout << "]" << std::endl;
}

template <typename DataType>
inline constexpr double get_rtol()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 1e-1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 1.5e-1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

template <typename DataType>
inline constexpr double get_atol()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 16.1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 8192.1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

template <typename InDataType,
          typename WeiDataType,
          typename GPUAccType,
          typename CPUAccType,
          ShapeType Shape,
          FilterType Filter,
          bool Dilation,
          bool EnableLds,
          int32_t TestMask>
bool run_test()
{
    if ((config.test_mask & TestMask) == 0)
    {
        return true;
    }
    constexpr ck::index_t FilterSize = (Filter == Filter_1X1) ? 1 : 3;
    constexpr ck::index_t HPerWconv  = (Shape == Shape_8X4) ? 8 : 4;
    constexpr ck::index_t WPerWconv  = (Shape == Shape_4X2) ? 2 : 4;

    const ck::index_t Width          = config.w;
    const ck::index_t Height         = config.h;
    const ck::index_t InputChannels  = config.c;
    const ck::index_t OutputChannels = config.k;

    constexpr ck::index_t HPerBlock  = DEFAULT_H_PERBLOCK;
    constexpr ck::index_t WPerBlock  = DEFAULT_W_PERBLOCK;
    constexpr ck::index_t CPerBlock  = DEFAULT_C_PERBLOCK;
    constexpr ck::index_t KPerBlock  = DEFAULT_K_PERBLOCK;
    constexpr ck::index_t HRepeat    = DEFAULT_H_PERWAVE / HPerWconv;
    constexpr ck::index_t WRepeat    = DEFAULT_W_PERWAVE / WPerWconv;

    constexpr ck::index_t n_dim          = 2;
    constexpr ck::index_t group_count    = 1;
    constexpr ck::index_t n_batch        = 1;
    const ck::index_t n_out_channels = OutputChannels;
    const ck::index_t n_in_channels  = InputChannels;

    constexpr ck::index_t DilationSize = Dilation ? 2 : 1;

    const std::vector<ck::index_t> filters_1x1{1, 1};
    const std::vector<ck::index_t> filters_3x3{3, 3};
    const std::vector<ck::index_t> dilations_1{1, 1};
    const std::vector<ck::index_t> dilations_2{2, 2};
    const std::vector<ck::index_t> pads_0{0, 0};
    const std::vector<ck::index_t> pads_1{1, 1};
    const std::vector<ck::index_t> pads_2{2, 2};

    const std::vector<ck::index_t>& filters_len =
        (Filter == Filter_1X1) ? filters_1x1 : filters_3x3;
    const std::vector<ck::index_t> input_len = {Height, Width};
    const std::vector<ck::index_t> strides{1, 1};
    const std::vector<ck::index_t>& dilations  = Dilation ? dilations_2 : dilations_1;
    const std::vector<ck::index_t>& left_pads  = (Filter == Filter_1X1) ? pads_0
                                                 : Dilation             ? pads_2
                                                                        : pads_1;
    const std::vector<ck::index_t>& right_pads = (Filter == Filter_1X1) ? pads_0
                                                 : Dilation             ? pads_2
                                                                        : pads_1;

    ck::utils::conv::ConvParam conv_param{n_dim,
                                          group_count,
                                          n_batch,
                                          n_out_channels,
                                          n_in_channels,
                                          filters_len,
                                          input_len,
                                          strides,
                                          dilations,
                                          left_pads,
                                          right_pads};

    constexpr auto NDimSpatial = ck::Number<n_dim>{};
    const auto in_element_op   = InElementOp{};
    const auto wei_element_op  = WeiElementOp{};
    const auto out_element_op  = OutElementOp{};

    namespace ctc   = ck::tensor_layout::convolution;
    auto in_layout  = ctc::GNHWC{};
    auto wei_layout = ctc::GKYXC{};
    auto out_layout = ctc::GNHWK{};
    using InLayout  = decltype(in_layout);
    using WeiLayout = decltype(wei_layout);
    using OutLayout = decltype(out_layout);

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<CPUAccType> out_host(out_g_n_k_wos_desc);
    Tensor<GPUAccType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        break;
    }

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                 InDataType,
                                                                 WeiDataType,
                                                                 CPUAccType,
                                                                 InElementOp,
                                                                 WeiElementOp,
                                                                 OutElementOp>();

    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(in,
                                              wei,
                                              out_host,
                                              conv_param.conv_filter_strides_,
                                              conv_param.conv_filter_dilations_,
                                              conv_param.input_left_pads_,
                                              conv_param.input_right_pads_,
                                              in_element_op,
                                              wei_element_op,
                                              out_element_op);

    if(config.do_verification)
    {
        ref_invoker.Run(ref_argument);
    }

    DumpTensor(in, "Input");
    DumpTensor(wei, "Weight");
    DumpTensor(out_host, "Accum");

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(GPUAccType) * out_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    // do Conv
    static constexpr auto ConvSpec =
        FilterSize == 1
            ? ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0
            : ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter3x3Stride1Pad0;

    constexpr ck::index_t InBlockTransferScalarPerVector = sizeof(uint32_t) / sizeof(InDataType);
    constexpr ck::index_t Cluster_In_C = CPerBlock / InBlockTransferScalarPerVector;
    constexpr ck::index_t Cluster_In_W = 4;

    constexpr ck::index_t Cluster_In_H = DEFAULT_BLOCKSIZE / Cluster_In_C / Cluster_In_W;
    using InBlockTransferThreadClusterLengths =
        ck::Sequence<Cluster_In_H, Cluster_In_W, Cluster_In_C>;
    constexpr bool InBlockLdsAddExtraM = true;

    constexpr ck::index_t WeiBlockTransferScalarPerVector = sizeof(uint32_t) / sizeof(WeiDataType);
    constexpr ck::index_t Cluster_Wei_C        = CPerBlock / WeiBlockTransferScalarPerVector;
    constexpr ck::index_t Cluster_Wei_K        = (DEFAULT_BLOCKSIZE / Cluster_Wei_C) > KPerBlock
                                                     ? KPerBlock
                                                     : (DEFAULT_BLOCKSIZE / Cluster_Wei_C);
    using WeiBlockTransferThreadClusterLengths = ck::Sequence<Cluster_Wei_K, 1, Cluster_Wei_C>;
    constexpr ck::index_t WeiBlockLdsAddExtraM = true;

    constexpr ck::index_t AccBlockTransferScalarPerVector = sizeof(uint32_t) / sizeof(GPUAccType);
    constexpr ck::index_t Cluster_Acc_K = KPerBlock / AccBlockTransferScalarPerVector;
    constexpr ck::index_t Cluster_Acc_W = 4;
    constexpr ck::index_t Cluster_Acc_H = DEFAULT_BLOCKSIZE / Cluster_Acc_K / Cluster_Acc_W;
    using AccBlockTransferClusterLengths =
        ck::Sequence<Cluster_Acc_H, Cluster_Acc_W, Cluster_Acc_K>;
    float avg_time = 0;
    using DeviceConvFwdInstance =
        ck::tensor_operation::device::DeviceConvWconv<NDimSpatial,
                                                      InputLayout<NDimSpatial>,
                                                      WeightLayout<NDimSpatial>,
                                                      OutputLayout<NDimSpatial>,
                                                      InDataType,
                                                      WeiDataType,
                                                      GPUAccType,
                                                      InElementOp,
                                                      WeiElementOp,
                                                      OutElementOp,
                                                      ConvSpec,
                                                      1,
                                                      DEFAULT_BLOCKSIZE,
                                                      HPerBlock,
                                                      WPerBlock,
                                                      CPerBlock,
                                                      KPerBlock,
                                                      HRepeat,
                                                      WRepeat,
                                                      HPerWconv,
                                                      WPerWconv,
                                                      FilterSize,
                                                      DilationSize,
                                                      DilationSize,
                                                      InBlockTransferThreadClusterLengths,
                                                      InBlockTransferScalarPerVector,
                                                      InBlockTransferScalarPerVector,
                                                      EnableLds,
                                                      InBlockLdsAddExtraM,
                                                      WeiBlockTransferThreadClusterLengths,
                                                      WeiBlockTransferScalarPerVector,
                                                      WeiBlockTransferScalarPerVector,
                                                      EnableLds,
                                                      WeiBlockLdsAddExtraM,
                                                      AccBlockTransferClusterLengths,
                                                      AccBlockTransferScalarPerVector,
                                                      EnableLds>;

    auto conv    = DeviceConvFwdInstance{};
    auto invoker = conv.MakeInvoker();
    auto argument =
        conv.MakeArgument(in_device_buf.GetDeviceBuffer(),
                          wei_device_buf.GetDeviceBuffer(),
                          out_device_buf.GetDeviceBuffer(),
                          a_g_n_c_wis_lengths,
                          a_g_n_c_wis_strides,
                          b_g_k_c_xs_lengths,
                          b_g_k_c_xs_strides,
                          e_g_n_k_wos_lengths,
                          e_g_n_k_wos_strides,
                          conv_filter_strides,
                          conv_filter_dilations,
                          input_left_pads,
                          input_right_pads,
                          InElementOp{},
                          WeiElementOp{},
                          OutElementOp{});

    if(!conv.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    avg_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});
    out_device_buf.FromDevice(out_device.mData.data());

    DumpTensor(out_device, "Accum_Device");
    std::cout << "Test <" << HPerWconv << "x" << WPerWconv << ", F:" << FilterSize
              << ", Src:" << sizeof(InDataType) << ", Dst:" << sizeof(GPUAccType) << ">: ";

    if (config.time_kernel)
    {
        std::cout << "Execute Time: " << avg_time << " ";
    }

    if (config.do_verification)
    {
        if constexpr(std::is_same<GPUAccType, ck::bhalf_t>::value)
        {
            // check_err doesn't support bhalf_t
            std::cout << "Ignored\n";
            return true;
        }
        else
        {
            bool ret = ck::utils::check_err(out_device,
                                            out_host,
                                            "Error: incorrect results!",
                                            get_rtol<GPUAccType>(),
                                            get_atol<GPUAccType>());
            if(ret)
            {
                std::cout << "Passed\n";
            }
            else
            {
                std::cout << "Failed\n";
            }

            return ret;
        }
    }
    else
    {
        return true;
    }
}

template <typename SrcType, typename GPUAccType, typename CPUAccType, int32_t TestMask>
bool run_test_fmt()
{
    if ((config.test_mask & TestMask) == 0)
    {
        return true;
    }
    bool pass = true;
    // clang-format off
    //                                                           |ShapeType |FilterType |Dilation |Iter4 |TestMask
    if constexpr(std::is_same<GPUAccType, float>::value || std::is_same<GPUAccType, int32_t>::value)
    {
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_4X2, Filter_1X1, false, false, 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_4X2, Filter_3X3, false, false, 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_4X2, Filter_3X3, true,  false, 0x40000>();

        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_4X2, Filter_1X1, false, true, 0x80000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_4X2, Filter_3X3, false, true, 0x100000>();
    }
    else
    {
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_4X2, Filter_1X1, false, false, 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_4X4, Filter_1X1, false, false, 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_8X4, Filter_1X1, false, false, 0x40000>();

        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_4X2, Filter_3X3, false, false, 0x80000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_4X4, Filter_3X3, false, false, 0x100000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_8X4, Filter_3X3, false, false, 0x200000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_4X2, Filter_3X3, true,  false, 0x400000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_4X4, Filter_3X3, true,  false, 0x800000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_8X4, Filter_3X3, true,  false, 0x1000000>();

        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_4X2, Filter_1X1, false,  true, 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_4X4, Filter_1X1, false,  true, 0x4000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_8X4, Filter_1X1, false,  true, 0x8000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_4X2, Filter_3X3, false,  true, 0x10000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_4X4, Filter_3X3, false,  true, 0x20000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, Shape_8X4, Filter_3X3, false,  true, 0x40000000>();
    }
    // clang-format on
    return pass;
}

inline void print_help_msg()
{
    std::cerr << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << "arg4: test mask (hex)\n"
              << "arg5-8: tensor size {H x W x C x K}" << std::endl;
}

inline bool parse_cmd_args(int argc,
                           char* argv[],
                           ExecutionConfig& config)
{
    if(argc == 1)
    {
        // use default
    }
    else if(argc == 9)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
        config.test_mask       = std::stoul(argv[4], nullptr, 0);
        config.h               = std::stoi(argv[5]);
        config.w               = std::stoi(argv[6]);
        config.k               = std::stoi(argv[7]);
        config.c               = std::stoi(argv[8]);
    }
    else
    {
        print_help_msg();
        return false;
    }

    return true;
}

int main(int argc, char* argv[])
{
    bool pass = true;
    //MessageBoxA(NULL, "", "", MB_OK);

    if (parse_cmd_args(argc, argv, config) == false)
    {
        return -1;
    }

    // clang-format off
    //                  |SrcType     |GPUAccType  |CPUAccType|TestMask
    pass &= run_test_fmt<ck::half_t,  float,       float,     0x1>();
    //pass &= run_test_fmt<ck::bhalf_t, float,       float,     0x2>();
    pass &= run_test_fmt<ck::f8_t,    float,       float,     0x4>();
    pass &= run_test_fmt<ck::bf8_t,   float,       float,     0x8>();
    pass &= run_test_fmt<int8_t,      float,       float,     0x10>();
    pass &= run_test_fmt<int8_t,      int32_t,     int32_t,   0x20>();
    
    pass &= run_test_fmt<ck::half_t,  ck::half_t,  ck::half_t, 0x40>();
    //pass &= run_test_fmt<ck::bhalf_t, ck::bhalf_t, ck::half_t, 0x80>();
    pass &= run_test_fmt<ck::f8_t,    ck::half_t,  ck::half_t, 0x100>();
    pass &= run_test_fmt<ck::bf8_t,   ck::bhalf_t, ck::half_t, 0x200>();
    pass &= run_test_fmt<int8_t,      ck::half_t,  ck::half_t, 0x400>();

#if CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    //pass &= run_test_fmt<ck::int4_t,  float,       float,     0x800>();
    //pass &= run_test_fmt<ck::int4_t,  int32_t,     int32_t,    0x1000>();
    //pass &= run_test_fmt<ck::int4_t,  ck::half_t,  ck::half_t, 0x2000>();
#endif
    // clang-format on

    std::cout << "TestConv ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
