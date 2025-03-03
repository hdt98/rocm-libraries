// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

//#define ENABLE_WAVEGROUP 1
//#define ENABLE_CONST_LAYOUT 1

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
#define DEFAULT_WAVEGROUP_BLOCKSIZE 256

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_layout {
namespace convolution {

// input tensor
struct CONST_GNHWC : public BaseTensorLayout
{
    static constexpr const char* name = "CONST_GNHWC";
    static constexpr auto N           = Number<1>{};
    static constexpr auto H           = Number<DEFAULT_H>{};
    static constexpr auto W           = Number<DEFAULT_W>{};
    static constexpr auto C           = Number<DEFAULT_C>{};
};

// weight tensor
template <index_t FilterSize>
struct CONST_GKYXC : public BaseTensorLayout
{
    static constexpr const char* name = "CONST_GKYXC";
    static constexpr auto K           = Number<DEFAULT_K>{};
    static constexpr auto X           = Number<FilterSize>{};
    static constexpr auto Y           = Number<FilterSize>{};
    static constexpr auto C           = Number<DEFAULT_C>{};
};

// output tensor
struct CONST_GNHWK : public BaseTensorLayout
{
    static constexpr const char* name = "CONST_GNHWK";
    static constexpr auto N           = Number<1>{};
    static constexpr auto H           = Number<DEFAULT_H>{};
    static constexpr auto W           = Number<DEFAULT_W>{};
    static constexpr auto K           = Number<DEFAULT_K>{};
};

} // namespace convolution
} // namespace tensor_layout
} // namespace ck

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_convsuba_fwd_wconvsuba.hpp"

//#include "windows.h"

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::UnaryConvert;

enum ShapeType
{
    Shape_4X2,
    Shape_4X4,
    Shape_8X4,
};

enum FilterType
{
    Filter_1X1,
    Filter_3X3,
    Filter_2X2,
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

template <ck::index_t FilterSize>
using ConstLayoutSetting =
    CommonLayoutSetting<ctl::CONST_GNHWC, ctl::CONST_GKYXC<FilterSize>, ctl::CONST_GNHWK>;

template <ck::index_t FilterSize>
using ConstInputLayout = typename ConstLayoutSetting<FilterSize>::InputLayout;

template <ck::index_t FilterSize>
using ConstWeightLayout = typename ConstLayoutSetting<FilterSize>::WeightLayout;

template <ck::index_t FilterSize>
using ConstOutputLayout = typename ConstLayoutSetting<FilterSize>::OutputLayout;

struct ExecutionConfig final
{
    uint32_t test_mask   = 0xffffffff;
    int init_method      = 1;
    bool do_verification = true;
    bool dump_tensor     = true;
    bool time_kernel     = false;
    int h                = DEFAULT_H;
    int w                = DEFAULT_W;
    int c                = DEFAULT_C;
    int k                = DEFAULT_K;
};

extern ExecutionConfig config;

template <typename DataType>
void DumpTensor(const Tensor<DataType>& tensor, const char* str)
{
    if(config.dump_tensor == false)
        return;
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
                        std::cout << "]";
                    }
                    if(lengths[4] > 3)
                    {
                        std::cout << std::endl;
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

template <typename Type>
const char* get_string()
{
    if constexpr(std::is_same<Type, ck::half_t>::value)
    {
        return "half_t";
    }

    if constexpr(std::is_same<Type, float>::value)
    {
        return "float";
    }

    if constexpr(std::is_same<Type, ck::bhalf_t>::value)
    {
        return "bhalf_t";
    }

    if constexpr(std::is_same<Type, ck::f8_t>::value)
    {
        return "f8_t";
    }

    if constexpr(std::is_same<Type, ck::bf8_t>::value)
    {
        return "bf8_t";
    }

    if constexpr(std::is_same<Type, int8_t>::value)
    {
        return "int8_t";
    }

    if constexpr(std::is_same<Type, int32_t>::value)
    {
        return "int32_t";
    }

    if constexpr(std::is_same<Type, uint8_t>::value)
    {
        return "uint8_t";
    }

    if constexpr(std::is_same<Type, uint32_t>::value)
    {
        return "uint32_t";
    }

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    if constexpr(std::is_same<Type, ck::int4_t>::value)
    {
        return "int4_t";
    }

    if constexpr(std::is_same<Type, ck::uint4_t>::value)
    {
        return "uint4_t";
    }
#endif
}

inline const char* get_string(ShapeType type)
{
    switch(type)
    {
    case Shape_4X2: return "Shape_4x2";
    case Shape_4X4: return "Shape_4x4";
    case Shape_8X4: return "Shape_8x4";
    }
}

inline const char* get_string(FilterType filter)
{
    switch(filter)
    {
    case Filter_1X1: return "Filter_1X1";
    case Filter_3X3: return "Filter_3X3";
    case Filter_2X2: return "Filter_2X2";
    }
}

template <typename InDataType,
          typename WeiDataType,
          typename GPUAccType,
          ShapeType Shape,
          FilterType Filter,
          bool Dilation,
          int LdsMode,
          bool EnableWaveGroup,
          int32_t TestMask>
bool run_test()
{
    if((config.test_mask & 0xFFFF0000 & TestMask) == 0)
    {
        return true;
    }
    constexpr ck::index_t FilterSize =
        (Filter == Filter_1X1) ? 1 : ((Filter == Filter_2X2) ? 2 : 3);
    constexpr ck::index_t HPerWconv = (Shape == Shape_8X4) ? 8 : 4;
    constexpr ck::index_t WPerWconv = (Shape == Shape_4X2) ? 2 : 4;

    const ck::index_t Width          = config.w;
    const ck::index_t Height         = config.h;
    const ck::index_t InputChannels  = config.c;
    const ck::index_t OutputChannels = config.k;
    constexpr ck::index_t BlockSize =
        EnableWaveGroup ? DEFAULT_WAVEGROUP_BLOCKSIZE : DEFAULT_BLOCKSIZE;

    // on gfx13, the wavegroup count is fixed to 4. so, HPerWave/WPerWave is fixed too.
    constexpr ck::index_t HPerWave = EnableWaveGroup ? DEFAULT_H_PERBLOCK / 2 : DEFAULT_H_PERWAVE;
    constexpr ck::index_t WPerWave = EnableWaveGroup ? DEFAULT_W_PERBLOCK / 2 : DEFAULT_W_PERWAVE;

    constexpr ck::index_t DefaultHScale =
        (Filter == Filter_2X2 && Shape == Shape_8X4 && HPerWave < 16) ? 2 : 1;

    constexpr ck::index_t HPerBlock = DEFAULT_H_PERBLOCK * DefaultHScale;
    constexpr ck::index_t WPerBlock = DEFAULT_W_PERBLOCK;
    constexpr ck::index_t CPerBlock = DEFAULT_C_PERBLOCK;
    constexpr ck::index_t KPerBlock = DEFAULT_K_PERBLOCK;
    constexpr ck::index_t HRepeat   = HPerWave * DefaultHScale / HPerWconv;
    constexpr ck::index_t WRepeat   = WPerWave / WPerWconv;

    constexpr ck::index_t n_dim       = 2;
    constexpr ck::index_t group_count = 1;
    constexpr ck::index_t n_batch     = 1;
    const ck::index_t n_out_channels  = OutputChannels;
    const ck::index_t n_in_channels   = InputChannels;

    constexpr ck::index_t DilationSize = Dilation ? 2 : 1;

    const std::vector<ck::index_t> filters_1x1{1, 1};
    const std::vector<ck::index_t> filters_2x2{2, 2};
    const std::vector<ck::index_t> filters_3x3{3, 3};
    const std::vector<ck::index_t> dilations_1{1, 1};
    const std::vector<ck::index_t> dilations_2{2, 2};
    const std::vector<ck::index_t> pads_0{0, 0};
    const std::vector<ck::index_t> pads_1{1, 1};
    const std::vector<ck::index_t> pads_2{2, 2};

    const std::vector<ck::index_t>& filters_len = (Filter == Filter_1X1)   ? filters_1x1
                                                  : (Filter == Filter_2X2) ? filters_2x2
                                                                           : filters_3x3;
    const std::vector<ck::index_t> input_len    = {Height, Width};
    const std::vector<ck::index_t> strides_1{1, 1};
    const std::vector<ck::index_t> strides_2{2, 2};
    const std::vector<ck::index_t>& strides   = (Filter == Filter_2X2) ? strides_2 : strides_1;
    const std::vector<ck::index_t>& dilations = Dilation ? dilations_2 : dilations_1;
    const std::vector<ck::index_t>& left_pads =
        (Filter == Filter_3X3) ? (Dilation ? pads_2 : pads_1) : pads_0;
    const std::vector<ck::index_t>& right_pads =
        (Filter == Filter_3X3) ? (Dilation ? pads_2 : pads_1) : pads_0;

    constexpr bool InEnableLds  = LdsMode & 1 ? true : false;
    constexpr bool WeiEnableLds = LdsMode & 2 ? true : false;
    constexpr bool AccEnableLds = LdsMode & 4 ? true : false;
    constexpr bool EnableAsync  = LdsMode & 8 ? true : false;

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
    Tensor<GPUAccType> out_host(out_g_n_k_wos_desc);
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

    constexpr ck::index_t CPerWConv_Bhalf = (Filter == Filter_3X3 && Shape == Shape_4X2) ? 8 : 4;

    constexpr ck::long_index_t Acc_Convert_Interval =
        std::is_same<GPUAccType, ck::bhalf_t>::value ? CPerWConv_Bhalf : CPerBlock;
    auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                 InDataType,
                                                                 WeiDataType,
                                                                 GPUAccType,
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
                                              out_element_op,
                                              {},
                                              {},
                                              {},
                                              Acc_Convert_Interval,
                                              true);

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
        (FilterSize == 1)
            ? ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0
            : ((FilterSize == 2) ? ck::tensor_operation::device::ConvolutionForwardSpecialization::
                                       Filter2x2Stride2Pad0
                                 : ck::tensor_operation::device::ConvolutionForwardSpecialization::
                                       Filter3x3Stride1Pad0);
    //? ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter2x2Stride2OddHWPad0
    //: ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0;

    constexpr ck::index_t InBlockTransferScalarPerVector = sizeof(uint32_t) / sizeof(InDataType);
    constexpr ck::index_t Cluster_In_C = CPerBlock / InBlockTransferScalarPerVector;
    constexpr ck::index_t Cluster_In_W = 4;

    constexpr ck::index_t ActiveBlockSize = EnableWaveGroup ? 128 : BlockSize;
    constexpr ck::index_t Cluster_In_H    = ActiveBlockSize / Cluster_In_C / Cluster_In_W;
    using InBlockTransferThreadClusterLengths =
        ck::Sequence<Cluster_In_H, Cluster_In_W, Cluster_In_C>;
    constexpr bool InBlockLdsAddExtraM = true;

    constexpr ck::index_t WeiBlockTransferScalarPerVector = sizeof(uint32_t) / sizeof(WeiDataType);
    constexpr ck::index_t Cluster_Wei_C        = CPerBlock / WeiBlockTransferScalarPerVector;
    constexpr ck::index_t Cluster_Wei_K        = (ActiveBlockSize / Cluster_Wei_C) > KPerBlock
                                                     ? KPerBlock
                                                     : (ActiveBlockSize / Cluster_Wei_C);
    using WeiBlockTransferThreadClusterLengths = ck::Sequence<Cluster_Wei_K, 1, Cluster_Wei_C>;
    constexpr ck::index_t WeiBlockLdsAddExtraM = true;

    constexpr ck::index_t AccBlockTransferScalarPerVector = sizeof(uint32_t) / sizeof(GPUAccType);
    constexpr ck::index_t Cluster_Acc_K = KPerBlock / AccBlockTransferScalarPerVector;
    constexpr ck::index_t Cluster_Acc_W = 4;
    constexpr ck::index_t Cluster_Acc_H = ActiveBlockSize / Cluster_Acc_K / Cluster_Acc_W;
    using AccBlockTransferClusterLengths =
        ck::Sequence<Cluster_Acc_H, Cluster_Acc_W, Cluster_Acc_K>;
    using EmptyTuple = ck::Tuple<>;
    float avg_time   = 0;
    using DeviceConvFwdInstance =
        ck::tensor_operation::device::DeviceConvSubaWconv<NDimSpatial,
#ifdef ENABLE_CONST_LAYOUT
                                                          ConstInputLayout<FilterSize>,
                                                          ConstWeightLayout<FilterSize>,
                                                          EmptyTuple,
                                                          ConstOutputLayout<FilterSize>,
#else
                                                          InputLayout<NDimSpatial>,
                                                          WeightLayout<NDimSpatial>,
                                                          EmptyTuple,
                                                          OutputLayout<NDimSpatial>,
#endif
                                                          InDataType,
                                                          WeiDataType,
                                                          EmptyTuple,
                                                          GPUAccType,
                                                          GPUAccType,
                                                          InElementOp,
                                                          WeiElementOp,
                                                          OutElementOp,
                                                          ConvSpec,
                                                          1,
                                                          BlockSize,
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
                                                          InEnableLds,
                                                          InBlockLdsAddExtraM,
                                                          WeiBlockTransferThreadClusterLengths,
                                                          WeiBlockTransferScalarPerVector,
                                                          WeiBlockTransferScalarPerVector,
                                                          WeiEnableLds,
                                                          WeiBlockLdsAddExtraM,
                                                          EmptyTuple,
                                                          ck::Sequence<>,
                                                          ck::Sequence<>,
                                                          false,
                                                          true,
                                                          AccBlockTransferClusterLengths,
                                                          AccBlockTransferScalarPerVector,
                                                          AccEnableLds,
                                                          EnableAsync,
                                                          EnableWaveGroup>;

    auto conv     = DeviceConvFwdInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(in_device_buf.GetDeviceBuffer(),
                                      wei_device_buf.GetDeviceBuffer(),
                                      std::array<const void*, 0>{},
                                      out_device_buf.GetDeviceBuffer(),
                                      a_g_n_c_wis_lengths,
                                      a_g_n_c_wis_strides,
                                      b_g_k_c_xs_lengths,
                                      b_g_k_c_xs_strides,
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
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
        std::cout << "wrong! device_conv with the specified compilation parameters does "
                     "not support this Conv problem"
                  << std::endl;
        return false;
    }
    avg_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});
    out_device_buf.FromDevice(out_device.mData.data());

    DumpTensor(out_device, "Accum_Device");

    std::cout <<
#ifdef ENABLE_WAVEGROUP
        "conv_device_wavegroup<In/Wei:"
#else
        "conv_device<In/Wei:"
#endif
              << get_string<InDataType>() << ", Out:" << get_string<GPUAccType>() << ", "
              << get_string(Shape) << ", " << get_string(Filter) << ", Dilation:" << DilationSize
              << ", LdsMod:" << LdsMode << ", WaveGroup:" << EnableWaveGroup << ", Id : 0x"
              << std::hex << TestMask << " Size: { " << config.h << "x" << config.w << "x"
              << config.c << "x" << config.k << " }>: Status: ";

    if(config.time_kernel)
    {
        std::cout << "Execute Time: " << avg_time << " ";
    }

    if(config.do_verification)
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
    else
    {
        return true;
    }
}

template <typename SrcType, typename GPUAccType, int LdsMode, int32_t TestMask>
bool run_test_fmt()
{
    if((config.test_mask & TestMask) == 0)
    {
        return true;
    }
    bool pass = true;

#ifdef ENABLE_WAVEGROUP
    constexpr bool WaveGroup = true;
#else
    constexpr bool WaveGroup = false;
#endif
    // clang-format off
    {
    //                                                           |ShapeType  |FilterType |Dilation |Lds |WaveGroup |TestMask
    if constexpr(std::is_same<GPUAccType, float>::value || std::is_same<GPUAccType, int32_t>::value)
    {
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_2X2, false, 0,       WaveGroup, TestMask | 0x10000>();

        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_2X2, false, LdsMode, WaveGroup, TestMask | 0x80000>();
    }
    else
    {
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, TestMask | 0x10000>();
        // TODO: fix it.
        bool fail_case = WaveGroup && (TestMask == 0x40) && (config.c == 0x40);
        if (fail_case == false)
        {
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_1X1, false, 0,       WaveGroup, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_1X1, false, 0,       WaveGroup, TestMask | 0x40000>();
        }
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_3X3, false, 0,       WaveGroup, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_3X3, false, 0,       WaveGroup, TestMask | 0x200000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, TestMask | 0x400000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_3X3, true,  0,       WaveGroup, TestMask | 0x800000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_3X3, true,  0,       WaveGroup, TestMask | 0x1000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_2X2, false, 0,       WaveGroup, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_2X2, false, 0,       WaveGroup, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_2X2, false, 0,       WaveGroup, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_1X1, false, LdsMode, WaveGroup, TestMask | 0x4000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_1X1, false, LdsMode, WaveGroup, TestMask | 0x8000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, TestMask | 0x10000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_3X3, false, LdsMode, WaveGroup, TestMask | 0x20000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_3X3, false, LdsMode, WaveGroup, TestMask | 0x40000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_2X2, false, LdsMode, WaveGroup, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_2X2, false, LdsMode, WaveGroup, TestMask | 0x4000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_2X2, false, LdsMode, WaveGroup, TestMask | 0x8000000>();
    }

    }
    // clang-format on
    return pass;
}

inline void print_help_msg()
{
    std::cerr << "arg1: test mask (hex)\n"
              << "arg2: verification (0=no, 1=yes)\n"
              << "arg3: dump tensor (0=no, 1=yes)\n"
              << "arg4: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg5: time kernel (0=no, 1=yes)\n"
              << "arg6-9: tensor size {H x W x C x K}" << std::endl;
}

inline bool parse_cmd_args(int argc, char* argv[], ExecutionConfig& cfg)
{
    if(argc == 1)
    {
        // use default
    }
    else if(argc <= 10)
    {
        if(argc > 1)
        {
            cfg.test_mask = std::stoul(argv[1], nullptr, 0);
        }
        if(argc > 2)
        {
            cfg.do_verification = std::stoi(argv[2]);
        }
        if(argc > 3)
        {
            cfg.dump_tensor = std::stoi(argv[3]);
        }
        if(argc > 4)
        {
            cfg.init_method = std::stoi(argv[4]);
        }
        if(argc > 5)
        {
            cfg.time_kernel = std::stoi(argv[5]);
        }
        if(argc > 6)
        {
            config.h = std::stoi(argv[6]);
        }
        if(argc > 7)
        {
            cfg.w = std::stoi(argv[7]);
        }
        if(argc > 8)
        {
            cfg.c = std::stoi(argv[8]);
        }
        if(argc > 9)
        {
            cfg.k = std::stoi(argv[9]);
        }
    }
    else
    {
        print_help_msg();
        return false;
    }

    return true;
}

using half_t  = ck::half_t;
using bhalf_t = ck::bhalf_t;
using f8_t    = ck::f8_t;
using bf8_t   = ck::bf8_t;

#define Extern_Test_Func(SrcType, GpuAccType, LdsMode, TestMask) \
    extern bool run_test_fmt_##SrcType##_##GpuAccType##_##LdsMode##_##TestMask();

#define Call_Test_Func(SrcType, GpuAccType, LdsMode, TestMask) \
    run_test_fmt_##SrcType##_##GpuAccType##_##LdsMode##_##TestMask()

#define Def_Test_Func(SrcType, GpuAccType, LdsMode, TestMask)             \
    bool run_test_fmt_##SrcType##_##GpuAccType##_##LdsMode##_##TestMask() \
    {                                                                     \
        return run_test_fmt<SrcType, GpuAccType, LdsMode, TestMask>();    \
    }
