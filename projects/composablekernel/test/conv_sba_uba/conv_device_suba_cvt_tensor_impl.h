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
#include "ck/tensor_operation/gpu/device/impl/device_convsuba_fwd_wconvsuba.hpp"

using InElementOp         = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp        = ck::tensor_operation::element_wise::PassThrough;
using PassThroughOp       = ck::tensor_operation::element_wise::PassThrough;
using MultiplyAdd         = ck::tensor_operation::element_wise::MultiplyAdd;
using MultiplyAddRelu     = ck::tensor_operation::element_wise::MultiplyAddRelu<>;
using MultiplyAddHardTanh = ck::tensor_operation::element_wise::MultiplyAddHardTanh;
using ScaleAdd            = ck::tensor_operation::element_wise::ScaleAdd;
using ScaleAddRelu        = ck::tensor_operation::element_wise::ScaleAddRelu<>;
using ScaleAddHardTanh    = ck::tensor_operation::element_wise::ScaleAddHardTanh;
using Scale               = ck::tensor_operation::element_wise::Scale;
using ScaleRelu           = ck::tensor_operation::element_wise::ScaleRelu<>;
using ScaleHardTanh       = ck::tensor_operation::element_wise::ScaleHardTanh;

using MultiplyAddClamp     = ck::tensor_operation::element_wise::MultiplyAddClamp;
using MultiplyAddClampRelu = ck::tensor_operation::element_wise::MultiplyAddRelu<true>;
using ScaleAddClamp        = ck::tensor_operation::element_wise::ScaleAddClamp;
using ScaleAddClampRelu    = ck::tensor_operation::element_wise::ScaleAddRelu<true>;
using ScaleClamp           = ck::tensor_operation::element_wise::ScaleClamp;
using ScaleClampRelu       = ck::tensor_operation::element_wise::ScaleRelu<true>;

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
#define DEFAULT_WAVEGROUP_BLOCKSIZE 512

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

template <typename InputLay,
          typename WeightLay,
          typename BiasLay,
          typename ScaleLay,
          typename OutputLay>
struct CommonLayoutSetting
{
    using InputLayout  = InputLay;
    using WeightLayout = WeightLay;
    using BiasLayout   = BiasLay;
    using ScaleLayout  = ScaleLay;
    using OutputLayout = OutputLay;
};

template <ck::index_t NDimSpatial>
struct CommonLayoutSettingSelector;

namespace ctl = ck::tensor_layout::convolution;

template <>
struct CommonLayoutSettingSelector<1> final
    : CommonLayoutSetting<ctl::G_NW_C, ctl::G_K_X_C, ctl::G_K, ctl::G_K, ctl::G_NW_K>
{
};

template <>
struct CommonLayoutSettingSelector<2> final
    : CommonLayoutSetting<ctl::G_NHW_C, ctl::G_K_YX_C, ctl::G_K, ctl::G_K, ctl::G_NHW_K>
{
};

template <>
struct CommonLayoutSettingSelector<3> final
    : CommonLayoutSetting<ctl::G_NDHW_C, ctl::G_K_ZYX_C, ctl::G_K, ctl::G_K, ctl::G_NDHW_K>
{
};

template <ck::index_t NDimSpatial>
using InputLayout = typename CommonLayoutSettingSelector<NDimSpatial>::InputLayout;

template <ck::index_t NDimSpatial>
using WeightLayout = typename CommonLayoutSettingSelector<NDimSpatial>::WeightLayout;

template <ck::index_t NDimSpatial>
using OutputLayout = typename CommonLayoutSettingSelector<NDimSpatial>::OutputLayout;

template <ck::index_t NDimSpatial>
using BiasLayout = typename CommonLayoutSettingSelector<NDimSpatial>::BiasLayout;

template <ck::index_t NDimSpatial>
using ScaleLayout = typename CommonLayoutSettingSelector<NDimSpatial>::ScaleLayout;

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
    auto strides = tensor.GetStrides();
    std::cout << str << "  [ " << std::endl;
    for(uint32_t i0 = 0; i0 < lengths[0]; i0++)
    {
        if(lengths[1] > 1)
        {
            std::cout << "  [";
        }
        if(i0 > 0 && strides[0] == 0)
        {
            continue;
        }
        for(uint32_t i1 = 0; i1 < lengths[1]; i1++)
        {
            if(lengths[2] > 1)
            {
                std::cout << "  [";
            }
            if(i1 > 0 && strides[1] == 0)
            {
                continue;
            }
            for(uint32_t i2 = 0; i2 < lengths[2]; i2++)
            {
                if(lengths[3] > 1)
                {
                    std::cout << "  [";
                }
                if(i2 > 0 && strides[2] == 0)
                {
                    continue;
                }
                for(uint32_t i3 = 0; i3 < lengths[3]; i3++)
                {
                    if(lengths[4] > 1)
                    {
                        std::cout << "  [";
                    }
                    if(i3 > 0 && strides[3] == 0)
                    {
                        continue;
                    }
                    for(uint32_t i4 = 0; i4 < lengths[4]; i4++)
                    {
                        if(i4 > 0 && strides[4] == 0)
                        {
                            continue;
                        }
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
        // Shape_4x2 + Filter_2X2 fail with 5e-2
        return 1e-1;
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
          typename EDataType,
          ShapeType Shape,
          FilterType Filter,
          bool Dilation,
          int LdsMode,
          bool EnableWaveGroup,
          int SbaMode,
          int ActiveFun,
          bool CvtToTensor,
          uint32_t TestMask>
bool run_test()
{
    if((config.test_mask & 0xFFFFFF00 & TestMask) == 0)
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

    using GPUOutType = typename std::conditional<CvtToTensor, EDataType, GPUAccType>::type;
    using CPUOutType = typename std::conditional<CvtToTensor, EDataType, GPUAccType>::type;

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
    constexpr bool DsEnableLds  = LdsMode & 16 ? true : false;

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
    const auto pass_through_op = PassThroughOp{};

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

    const auto bias_g_n_k_wos_desc =
        ck::utils::conv::make_scalebias_host_tensor_descriptor<OutLayout>(conv_param,
                                                                          OutputChannels);
    const auto scale_g_n_k_wos_desc =
        ck::utils::conv::make_scalebias_host_tensor_descriptor<OutLayout>(conv_param,
                                                                          OutputChannels);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<GPUAccType> bias(bias_g_n_k_wos_desc);
    Tensor<GPUAccType> scale(scale_g_n_k_wos_desc);
    Tensor<CPUOutType> out_host(out_g_n_k_wos_desc);
    Tensor<GPUOutType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "bias: " << bias.mDesc << std::endl;
    std::cout << "scale: " << scale.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    float scale_value = 0;
    switch(config.init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});

        if constexpr(SbaMode == 2) // ScaleBiasPacked
        {
            bias.GenerateTensorValue(GeneratorTensor_2<GPUAccType>{-10, 10});
            scale.GenerateTensorValue(GeneratorTensor_2<GPUAccType>{-3, 3});
        }
        else
        {
            scale_value = 3.0f;
            if constexpr(SbaMode == 1) // uniform scale + v bias
            {
                bias.GenerateTensorValue(GeneratorTensor_2<GPUAccType>{-10, 10});
            }
        }
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});

        if constexpr(SbaMode == 2) // ScaleBiasPacked
        {
            bias.GenerateTensorValue(GeneratorTensor_3<GPUAccType>{-0.3, 0.3});
            scale.GenerateTensorValue(GeneratorTensor_2<GPUAccType>{-3, 3});
        }
        else
        {
            scale_value = 3.0f;
            if constexpr(SbaMode == 1) // uniform scale + v bias
            {
                bias.GenerateTensorValue(GeneratorTensor_3<GPUAccType>{-0.3, 0.3});
            }
        }
        break;
    }

    constexpr bool Clamp =
        ck::is_same_v<GPUOutType, int8_t> || ck::is_same_v<GPUOutType, ck::f8_ocp_t>;

    const auto out_element_op = [&]() {
        if constexpr(SbaMode == 0)
        {
            if constexpr(ActiveFun == 0)
            {
                if constexpr(Clamp)
                {
                    return ScaleClamp{scale_value};
                }
                else
                {
                    return Scale{scale_value};
                }
            }
            else if constexpr(ActiveFun == 1)
            {
                if constexpr(Clamp)
                {
                    return ScaleClampRelu{scale_value};
                }
                else
                {
                    return ScaleRelu{scale_value};
                }
            }
            else if constexpr(ActiveFun == 2)
            {
                return ScaleHardTanh{scale_value};
            }
        }
        else if constexpr(SbaMode == 1)
        {
            if constexpr(ActiveFun == 0)
            {
                if constexpr(Clamp)
                {
                    return ScaleAddClamp{scale_value};
                }
                else
                {
                    return ScaleAdd{scale_value};
                }
            }
            else if constexpr(ActiveFun == 1)
            {
                if constexpr(Clamp)
                {
                    return ScaleAddClampRelu{scale_value};
                }
                else
                {
                    return ScaleAddRelu{scale_value};
                }
            }
            else if constexpr(ActiveFun == 2)
            {
                return ScaleAddHardTanh{scale_value};
            }
        }
        else if constexpr(SbaMode == 2)
        {
            if constexpr(ActiveFun == 0)
            {
                if constexpr(Clamp)
                {
                    return MultiplyAddClamp{};
                }
                else
                {
                    return MultiplyAdd{};
                }
            }
            else if constexpr(ActiveFun == 1)
            {
                if constexpr(Clamp)
                {
                    return MultiplyAddClampRelu{};
                }
                else
                {
                    return MultiplyAddRelu{};
                }
            }
            else if constexpr(ActiveFun == 2)
            {
                return MultiplyAddHardTanh{};
            }
        }
        else
        {
            static_assert(0);
        }
    }();

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> scale_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> scale_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial + 3> bias_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> bias_g_n_k_wos_strides{};
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
    copy(scale_g_n_k_wos_desc.GetLengths(), scale_g_n_k_wos_lengths);
    copy(scale_g_n_k_wos_desc.GetStrides(), scale_g_n_k_wos_strides);
    copy(bias_g_n_k_wos_desc.GetLengths(), bias_g_n_k_wos_lengths);
    copy(bias_g_n_k_wos_desc.GetStrides(), bias_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);

    Tensor<GPUAccType> c_host(out_g_n_k_wos_desc);

    constexpr ck::index_t CPerWConv_Bhalf = (Filter == Filter_3X3 && Shape == Shape_4X2) ? 8 : 4;

    constexpr ck::long_index_t Acc_Convert_Interval =
        std::is_same<GPUAccType, ck::bhalf_t>::value ? CPerWConv_Bhalf : CPerBlock;
    auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                 InDataType,
                                                                 WeiDataType,
                                                                 GPUAccType,
                                                                 InElementOp,
                                                                 WeiElementOp,
                                                                 PassThroughOp>();

    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(in,
                                              wei,
                                              c_host,
                                              conv_param.conv_filter_strides_,
                                              conv_param.conv_filter_dilations_,
                                              conv_param.input_left_pads_,
                                              conv_param.input_right_pads_,
                                              in_element_op,
                                              wei_element_op,
                                              pass_through_op,
                                              {},
                                              {},
                                              {},
                                              Acc_Convert_Interval,
                                              true);

    if(config.do_verification)
    {
        ref_invoker.Run(ref_argument);
        out_host.ForEach([&](auto&, auto idx) {
            if constexpr(SbaMode == 0)
            {
                out_element_op(out_host(idx), c_host(idx));
            }
            else if constexpr(SbaMode == 1)
            {
                out_element_op(out_host(idx), c_host(idx), bias(idx));
            }
            else if constexpr(SbaMode == 2)
            {
                out_element_op(out_host(idx), c_host(idx), scale(idx), bias(idx));
            }
        });
    }

    DumpTensor(in, "Input");
    DumpTensor(wei, "Weight");
    DumpTensor(bias, "Bias");
    DumpTensor(scale, "Scale");
    DumpTensor(c_host, "ConvOut");
    DumpTensor(out_host, "Output");

    DeviceMem in_device_buf(sizeof(InDataType) *
                            in.mDesc.GetElementSpaceSize()); // ISSUE HAPPENS in latest CSIM
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem bias_device_buf(sizeof(GPUAccType) * bias.mDesc.GetElementSpaceSize());
    DeviceMem scale_device_buf(sizeof(GPUAccType) * scale.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(GPUOutType) * out_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());
    bias_device_buf.ToDevice(bias.mData.data());
    scale_device_buf.ToDevice(scale.mData.data());

    // do Conv
    static constexpr auto ConvSpec =
        FilterSize == 1
            ? ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0
        : FilterSize == 2
            ? ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter2x2Stride2Pad0
            : ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter3x3Stride1Pad0;

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

    constexpr ck::index_t AccBlockTransferScalarPerVector = sizeof(uint32_t) / sizeof(GPUOutType);
    constexpr ck::index_t Cluster_Acc_K = KPerBlock / AccBlockTransferScalarPerVector;
    constexpr ck::index_t Cluster_Acc_W = 4;
    constexpr ck::index_t Cluster_Acc_H = ActiveBlockSize / Cluster_Acc_K / Cluster_Acc_W;
    using AccBlockTransferClusterLengths =
        ck::Sequence<Cluster_Acc_H, Cluster_Acc_W, Cluster_Acc_K>;

    constexpr ck::index_t DBlockTransferScalarPerVector = sizeof(uint32_t) / sizeof(GPUAccType);
    constexpr ck::index_t Cluster_Ds_K                  = KPerBlock / DBlockTransferScalarPerVector;
    constexpr ck::index_t DsBlockLdsAddExtraM           = true;

    float avg_time    = 0;
    using AccDataType = GPUAccType;

    constexpr auto DsDataTypeInfo = [&]() {
        if constexpr(SbaMode == 0)
        {
            using DsDataType                          = ck::Tuple<>;
            using DsDataLayout                        = ck::Tuple<>;
            using DsBlockTransferThreadClusterLengths = ck::Tuple<>;
            using DsBlockTransferScalarPerVector      = ck::Sequence<>;
            return make_tuple(DsDataType{},
                              DsDataLayout{},
                              DsBlockTransferThreadClusterLengths{},
                              DsBlockTransferScalarPerVector{});
        }
        else if constexpr(SbaMode == 1)
        {
            using DsDataType                          = ck::Tuple<GPUAccType>;
            using DsDataLayout                        = ck::Tuple<BiasLayout<NDimSpatial>>;
            using DsBlockTransferThreadClusterLengths = ck::Tuple<ck::Sequence<Cluster_Ds_K>>;
            using DsBlockTransferScalarPerVector      = ck::Sequence<DBlockTransferScalarPerVector>;
            return make_tuple(DsDataType{},
                              DsDataLayout{},
                              DsBlockTransferThreadClusterLengths{},
                              DsBlockTransferScalarPerVector{});
        }
        else if constexpr(SbaMode == 2)
        {
            using DsDataType   = ck::Tuple<GPUAccType, GPUAccType>;
            using DsDataLayout = ck::Tuple<ScaleLayout<NDimSpatial>, BiasLayout<NDimSpatial>>;
            using DsBlockTransferThreadClusterLengths =
                ck::Tuple<ck::Sequence<Cluster_Ds_K>, ck::Sequence<Cluster_Ds_K>>;
            using DsBlockTransferScalarPerVector =
                ck::Sequence<DBlockTransferScalarPerVector, DBlockTransferScalarPerVector>;
            return make_tuple(DsDataType{},
                              DsDataLayout{},
                              DsBlockTransferThreadClusterLengths{},
                              DsBlockTransferScalarPerVector{});
        }
    }();

    using DsDataType   = ck::remove_cvref_t<ck::tuple_element_t<0, decltype(DsDataTypeInfo)>>;
    using DsDataLayout = ck::remove_cvref_t<ck::tuple_element_t<1, decltype(DsDataTypeInfo)>>;
    using DsBlockTransferThreadClusterLengths =
        ck::remove_cvref_t<ck::tuple_element_t<2, decltype(DsDataTypeInfo)>>;
    using DsBlockTransferScalarPerVector =
        ck::remove_cvref_t<ck::tuple_element_t<3, decltype(DsDataTypeInfo)>>;
    using DeviceSubaCvtConvFwdInstance =
        ck::tensor_operation::device::DeviceConvSubaWconv<NDimSpatial,
                                                          InputLayout<NDimSpatial>,
                                                          WeightLayout<NDimSpatial>,
                                                          DsDataLayout,
                                                          OutputLayout<NDimSpatial>,
                                                          InDataType,
                                                          WeiDataType,
                                                          DsDataType,
                                                          AccDataType,
                                                          GPUOutType,
                                                          InElementOp,
                                                          WeiElementOp,
                                                          decltype(out_element_op),
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
                                                          DsBlockTransferThreadClusterLengths,
                                                          DsBlockTransferScalarPerVector,
                                                          DsBlockTransferScalarPerVector,
                                                          DsEnableLds,
                                                          DsBlockLdsAddExtraM,
                                                          AccBlockTransferClusterLengths,
                                                          AccBlockTransferScalarPerVector,
                                                          AccEnableLds,
                                                          EnableAsync,
                                                          EnableWaveGroup,
                                                          false,  // shuffleOnLoad
                                                          false>; // transpose>;

    std::array<const void*, SbaMode> ds_bufs;
    std::array<std::array<ck::index_t, NDimSpatial + 3>, SbaMode> ds_lengths;
    std::array<std::array<ck::index_t, NDimSpatial + 3>, SbaMode> ds_strides;
    if constexpr(SbaMode == 0) {}
    else if constexpr(SbaMode == 1)
    {
        ds_bufs[0]    = bias_device_buf.GetDeviceBuffer();
        ds_lengths[0] = bias_g_n_k_wos_lengths;
        ds_strides[0] = bias_g_n_k_wos_strides;
    }
    else if constexpr(SbaMode == 2)
    {
        ds_bufs[0]    = scale_device_buf.GetDeviceBuffer();
        ds_bufs[1]    = bias_device_buf.GetDeviceBuffer();
        ds_lengths[0] = scale_g_n_k_wos_lengths;
        ds_lengths[1] = bias_g_n_k_wos_lengths;
        ds_strides[0] = scale_g_n_k_wos_strides;
        ds_strides[1] = bias_g_n_k_wos_strides;
    }
    auto conv     = DeviceSubaCvtConvFwdInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(in_device_buf.GetDeviceBuffer(),
                                      wei_device_buf.GetDeviceBuffer(),
                                      ds_bufs,
                                      out_device_buf.GetDeviceBuffer(),
                                      a_g_n_c_wis_lengths,
                                      a_g_n_c_wis_strides,
                                      b_g_k_c_xs_lengths,
                                      b_g_k_c_xs_strides,
                                      ds_lengths,
                                      ds_strides,
                                      e_g_n_k_wos_lengths,
                                      e_g_n_k_wos_strides,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads,
                                      InElementOp{},
                                      WeiElementOp{},
                                      out_element_op);

    if(!conv.IsSupportedArgument(argument))
    {
        std::cout << "wrong! device_conv with the specified compilation parameters does "
                     "not support this Conv problem"
                  << std::endl;
        return false;
    }
    avg_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

    out_device_buf.FromDevice(out_device.mData.data());
    DumpTensor(out_device, "Out_Device");

    std::cout <<
#ifdef ENABLE_WAVEGROUP
        "conv_sba_uba_device_wavegroup<In/Wei:"
#else
        "conv_sba_uba_device<In/Wei:"
#endif
              << get_string<InDataType>() << ", Out:" << get_string<GPUAccType>() << ", "
              << get_string(Shape) << ", " << get_string(Filter) << ", Dilation:" << DilationSize
              << ", LdsMod:" << LdsMode << ", SbaMode:" << SbaMode << ", ActiveFun:" << ActiveFun
              << ", CvtToTensor:" << CvtToTensor << ", WaveGroup:" << EnableWaveGroup << ", Id : 0x"
              << std::hex << TestMask << " Size: { " << config.h << "x" << config.w << "x"
              << config.c << "x" << config.k << " }>: Status: ";

    if(config.time_kernel)
    {
        std::cout << "Execute Time: " << avg_time << " ";
    }

    if(config.do_verification)
    {
        bool ret = false;
        ret      = ck::utils::check_err(out_device,
                                   out_host,
                                   "Error: incorrect results!",
                                   get_rtol<GPUOutType>(),
                                   get_atol<GPUOutType>());

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
            cfg.h = std::stoi(argv[6]);
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
