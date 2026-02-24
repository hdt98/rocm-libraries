// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <miopen/ck_builder/instance_data/xdl_bwd_weight.hpp>
#include <miopen/ck_builder/factories/grouped_conv_bwd_weight/common_aliases.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_bwd_weight {

using namespace instance;

// =====================================================================================
// F32 generic instances (no compute types / no transpose transfer params)
// 1 instance
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_xdl_c_shuffle_f32_generic_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    ckb::ConvSpecialization conv_spec)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeight_Xdl_CShuffle_Generic<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,64,4,4,32,32,2,2,{1,4,16,1},{0,3,1,2},{0,2,1,3},2,1,4,true,{1,4,16,1},{0,3,1,2},{0,2,1,3},2,1,4,true,1,1,{1,16,1,4},1)
        // clang-format on
    };
}

// =====================================================================================
// F32 instances (with compute types and transpose transfer params)
// 20 instances
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_xdl_c_shuffle_f32_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    ckb::ConvSpecialization conv_spec)
{
    return std::array{
        // clang-format off
        // generic instance
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,64,4,4,32,32,2,2,{1,4,16,1},{0,3,1,2},{0,2,1,3},2,1,4,true,{1,4,16,1},{0,3,1,2},{0,2,1,3},2,1,4,true,1,1,{1,16,1,4},1,F32,F32),
        // instances for small conv.K and conv.C
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,32,4,4,32,32,2,1,{1,4,32,1},{0,3,1,2},{0,2,1,3},2,4,4,true,{1,4,8,4},{0,3,1,2},{0,2,1,3},2,1,1,true,1,1,{1,32,1,4},1,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,32,64,4,4,32,32,1,2,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,1,2,true,{1,4,16,1},{0,3,1,2},{0,2,1,3},2,4,4,true,1,1,{1,16,1,4},4,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,256,128,4,4,32,32,4,2,{1,4,64,1},{0,3,1,2},{0,2,1,3},2,4,4,true,{1,4,32,2},{0,3,1,2},{0,2,1,3},2,4,2,true,1,1,{1,32,1,8},4,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,128,256,4,4,32,32,2,4,{1,4,32,2},{0,3,1,2},{0,2,1,3},2,4,2,true,{1,4,64,1},{0,3,1,2},{0,2,1,3},2,4,4,true,1,1,{1,32,1,8},4,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,128,4,4,32,32,4,2,{1,4,32,1},{0,3,1,2},{0,2,1,3},2,4,4,true,{1,4,32,1},{0,3,1,2},{0,2,1,3},2,4,4,true,1,1,{1,32,1,4},4,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,128,128,4,4,32,32,2,2,{1,4,32,2},{0,3,1,2},{0,2,1,3},2,4,2,true,{1,4,32,2},{0,3,1,2},{0,2,1,3},2,4,2,true,1,1,{1,32,1,4},4,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,64,4,4,32,32,2,2,{1,4,32,1},{0,3,1,2},{0,2,1,3},2,4,4,true,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,4,2,true,1,1,{1,32,1,4},4,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,64,128,4,4,32,32,2,2,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,4,2,true,{1,4,32,1},{0,3,1,2},{0,2,1,3},2,4,4,true,1,1,{1,32,1,4},4,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,64,4,4,32,32,2,2,{1,4,16,1},{0,3,1,2},{0,2,1,3},2,4,4,true,{1,4,16,1},{0,3,1,2},{0,2,1,3},2,4,4,true,1,1,{1,16,1,4},4,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,128,64,4,4,32,32,2,1,{1,4,32,2},{0,3,1,2},{0,2,1,3},2,4,2,true,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,4,1,true,1,1,{1,32,1,4},4,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,64,128,4,4,32,32,1,2,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,4,1,true,{1,4,32,2},{0,3,1,2},{0,2,1,3},2,4,2,true,1,1,{1,32,1,4},4,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,32,4,4,32,32,2,1,{1,4,32,1},{0,3,1,2},{0,2,1,3},2,4,4,true,{1,4,8,4},{0,3,1,2},{0,2,1,3},2,4,1,true,1,1,{1,32,1,4},4,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,32,128,4,4,32,32,1,2,{1,4,8,4},{0,3,1,2},{0,2,1,3},2,4,1,true,{1,4,32,1},{0,3,1,2},{0,2,1,3},2,4,4,true,1,1,{1,32,1,4},4,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,32,4,4,32,32,2,1,{1,4,16,1},{0,3,1,2},{0,2,1,3},2,4,4,true,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,4,2,true,1,1,{1,16,1,4},4,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,32,64,4,4,32,32,1,2,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,4,2,true,{1,4,16,1},{0,3,1,2},{0,2,1,3},2,4,4,true,1,1,{1,16,1,4},4,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,64,64,8,8,32,32,1,1,{1,8,16,1},{0,3,1,2},{0,3,1,2},2,4,4,true,{1,8,16,1},{0,3,1,2},{0,3,1,2},2,4,4,true,1,1,{1,16,1,16},4,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,64,64,8,8,32,32,1,1,{1,8,16,1},{0,3,1,2},{0,3,1,2},2,4,4,true,{1,8,32,1},{0,3,1,2},{0,3,1,2},2,1,4,true,1,1,{1,4,1,64},1,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,64,64,8,8,32,32,1,1,{1,8,32,1},{0,3,1,2},{0,3,1,2},2,1,4,true,{1,8,16,1},{0,3,1,2},{0,3,1,2},2,4,4,true,1,1,{1,16,1,16},4,F32,F32),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,64,64,8,8,32,32,1,1,{1,8,32,1},{0,3,1,2},{0,3,1,2},2,1,4,true,{1,8,32,1},{0,3,1,2},{0,3,1,2},2,1,4,true,1,1,{1,4,1,64},1,F32,F32)
        // clang-format on
    };
}

// =====================================================================================
// F32 TF32 generic instances (no compute types / no transpose transfer params)
// 1 instance
// Note: TF32 compute type used internally by CK
// TODO: TF32 compute type is not yet supported in ckb::DataType enum
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_xdl_c_shuffle_f32_tf32_generic_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    ckb::ConvSpecialization conv_spec)
{
    // TODO: TF32 compute type is not yet supported in ckb::DataType enum
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvBwdWeight_Xdl_CShuffle_Generic<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,64,4,4,32,32,2,2,{1,4,16,1},{0,3,1,2},{0,2,1,3},2,1,4,true,{1,4,16,1},{0,3,1,2},{0,2,1,3},2,1,4,true,1,1,{1,16,1,4},1)
    //     // clang-format on
    // };
    return std::array<XdlBwdWeightInstance<>, 0>{};
}

// =====================================================================================
// F32 TF32 instances (with compute types and transpose transfer params)
// 20 instances
// Note: TF32 compute type used internally by CK
// TODO: TF32 compute type is not yet supported in ckb::DataType enum
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_xdl_c_shuffle_f32_tf32_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    ckb::ConvSpecialization conv_spec)
{
    // TODO: TF32 compute type is not yet supported in ckb::DataType enum
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,64,4,4,32,32,2,2,{1,4,16,1},{0,3,1,2},{0,2,1,3},2,1,4,true,{1,4,16,1},{0,3,1,2},{0,2,1,3},2,1,4,true,1,1,{1,16,1,4},1,TF32,TF32),
    //     DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,32,4,4,32,32,2,1,{1,4,32,1},{0,3,1,2},{0,2,1,3},2,4,4,true,{1,4,8,4},{0,3,1,2},{0,2,1,3},2,1,1,true,1,1,{1,32,1,4},1,TF32,TF32),
    //     ... (20 instances total)
    //     // clang-format on
    // };
    return std::array<XdlBwdWeightInstance<>, 0>{};
}

// =====================================================================================
// F16 generic instances (no compute types / no transpose transfer params)
// 1 instance
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_xdl_c_shuffle_f16_generic_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    ckb::ConvSpecialization conv_spec)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeight_Xdl_CShuffle_Generic<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,64,4,8,32,32,2,2,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,2,4,true,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,2,4,true,1,1,{1,16,1,4},2)
        // clang-format on
    };
}

// =====================================================================================
// F16 instances (with compute types and transpose transfer params)
// 16 instances
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_xdl_c_shuffle_f16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    ckb::ConvSpecialization conv_spec)
{
    return std::array{
        // clang-format off
        // generic instance
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,64,4,8,32,32,2,2,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,2,4,true,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,2,4,true,1,1,{1,16,1,4},2,F16,F16),
        // instance for small conv.K (conv.K and conv.C must be divisible by 2)
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,32,4,8,32,32,2,1,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,4,8},{0,3,1,2},{0,2,1,3},2,2,1,true,1,1,{1,32,1,4},2,F16,F16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,32,64,4,8,32,32,1,2,{1,4,4,4},{0,3,1,2},{0,2,1,3},2,2,2,true,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,16,1,4},8,F16,F16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,256,128,4,8,32,32,4,2,{1,4,32,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,1,1,{1,32,1,8},8,F16,F16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,128,256,4,8,32,32,2,4,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,{1,4,32,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,32,1,8},8,F16,F16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,128,4,8,32,32,4,2,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,32,1,4},8,F16,F16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,128,128,4,8,32,32,2,2,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,1,1,{1,32,1,4},8,F16,F16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,64,4,8,32,32,2,2,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,8,4},{0,3,1,2},{0,2,1,3},2,8,2,true,1,1,{1,32,1,4},8,F16,F16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,64,128,4,8,32,32,2,2,{1,4,8,4},{0,3,1,2},{0,2,1,3},2,8,2,true,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,32,1,4},8,F16,F16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,64,4,8,32,32,2,2,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,16,1,4},8,F16,F16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,128,64,4,8,32,32,2,1,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,{1,4,8,8},{0,3,1,2},{0,2,1,3},2,8,1,true,1,1,{1,32,1,4},8,F16,F16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,64,128,4,8,32,32,1,2,{1,4,8,8},{0,3,1,2},{0,2,1,3},2,8,1,true,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,1,1,{1,32,1,4},8,F16,F16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,32,4,8,32,32,2,1,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,4,8},{0,3,1,2},{0,2,1,3},2,8,1,true,1,1,{1,32,1,4},8,F16,F16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,32,128,4,8,32,32,1,2,{1,4,4,8},{0,3,1,2},{0,2,1,3},2,8,1,true,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,32,1,4},8,F16,F16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,32,4,8,32,32,2,1,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,4,4},{0,3,1,2},{0,2,1,3},2,8,2,true,1,1,{1,16,1,4},8,F16,F16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,32,64,4,8,32,32,1,2,{1,4,4,4},{0,3,1,2},{0,2,1,3},2,8,2,true,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,16,1,4},8,F16,F16)
        // clang-format on
    };
}

// =====================================================================================
// BF16 (in=BF16, wei=F32, out=BF16) generic instances
// 1 instance
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_xdl_c_shuffle_bf16_f32_bf16_generic_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    ckb::ConvSpecialization conv_spec)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeight_Xdl_CShuffle_Generic<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,64,4,8,32,32,2,2,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,1,4,true,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,1,4,true,1,1,{1,16,1,4},1)
        // clang-format on
    };
}

// =====================================================================================
// BF16 (in=BF16, wei=F32, out=BF16) instances (with compute types and transpose transfer)
// 16 instances
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_xdl_c_shuffle_bf16_f32_bf16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    ckb::ConvSpecialization conv_spec)
{
    return std::array{
        // clang-format off
        // generic instance
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,64,4,8,32,32,2,2,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,1,4,true,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,1,4,true,1,1,{1,16,1,4},1,BF16,BF16),
        // instance for small conv.K
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,32,4,8,32,32,2,1,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,4,8},{0,3,1,2},{0,2,1,3},2,1,1,true,1,1,{1,32,1,4},1,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,32,64,4,8,32,32,1,2,{1,4,4,4},{0,3,1,2},{0,2,1,3},2,1,2,true,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,16,1,4},4,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,256,128,4,8,32,32,4,2,{1,4,32,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,1,1,{1,32,1,8},4,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,128,256,4,8,32,32,2,4,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,{1,4,32,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,32,1,8},4,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,128,4,8,32,32,4,2,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,32,1,4},4,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,128,128,4,8,32,32,2,2,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,1,1,{1,32,1,4},4,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,64,4,8,32,32,2,2,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,8,4},{0,3,1,2},{0,2,1,3},2,8,2,true,1,1,{1,32,1,4},4,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,64,128,4,8,32,32,2,2,{1,4,8,4},{0,3,1,2},{0,2,1,3},2,8,2,true,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,32,1,4},4,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,64,4,8,32,32,2,2,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,16,1,4},4,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,128,64,4,8,32,32,2,1,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,{1,4,8,8},{0,3,1,2},{0,2,1,3},2,8,1,true,1,1,{1,32,1,4},4,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,64,128,4,8,32,32,1,2,{1,4,8,8},{0,3,1,2},{0,2,1,3},2,8,1,true,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,1,1,{1,32,1,4},4,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,32,4,8,32,32,2,1,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,4,8},{0,3,1,2},{0,2,1,3},2,8,1,true,1,1,{1,32,1,4},4,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,32,128,4,8,32,32,1,2,{1,4,4,8},{0,3,1,2},{0,2,1,3},2,8,1,true,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,32,1,4},4,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,32,4,8,32,32,2,1,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,4,4},{0,3,1,2},{0,2,1,3},2,8,2,true,1,1,{1,16,1,4},4,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,32,64,4,8,32,32,1,2,{1,4,4,4},{0,3,1,2},{0,2,1,3},2,8,2,true,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,16,1,4},4,BF16,BF16)
        // clang-format on
    };
}

// =====================================================================================
// BF16 (in=BF16, wei=BF16, out=BF16) instances (with compute types and transpose transfer)
// 16 instances
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_xdl_c_shuffle_bf16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    ckb::ConvSpecialization conv_spec)
{
    return std::array{
        // clang-format off
        // generic instance
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,64,4,8,32,32,2,2,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,2,4,true,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,2,4,true,1,1,{1,16,1,4},2,BF16,BF16),
        // instance for small conv.K (conv.K and conv.C must be divisible by 2)
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,32,4,8,32,32,2,1,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,4,8},{0,3,1,2},{0,2,1,3},2,2,1,true,1,1,{1,32,1,4},2,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,32,64,4,8,32,32,1,2,{1,4,4,4},{0,3,1,2},{0,2,1,3},2,2,2,true,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,16,1,4},8,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,256,128,4,8,32,32,4,2,{1,4,32,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,1,1,{1,32,1,8},8,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,128,256,4,8,32,32,2,4,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,{1,4,32,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,32,1,8},8,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,128,4,8,32,32,4,2,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,32,1,4},8,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,128,128,4,8,32,32,2,2,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,1,1,{1,32,1,4},8,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,64,4,8,32,32,2,2,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,8,4},{0,3,1,2},{0,2,1,3},2,8,2,true,1,1,{1,32,1,4},8,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,64,128,4,8,32,32,2,2,{1,4,8,4},{0,3,1,2},{0,2,1,3},2,8,2,true,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,32,1,4},8,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,64,4,8,32,32,2,2,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,16,1,4},8,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,128,64,4,8,32,32,2,1,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,{1,4,8,8},{0,3,1,2},{0,2,1,3},2,8,1,true,1,1,{1,32,1,4},8,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,64,128,4,8,32,32,1,2,{1,4,8,8},{0,3,1,2},{0,2,1,3},2,8,1,true,{1,4,16,4},{0,3,1,2},{0,2,1,3},2,8,2,true,1,1,{1,32,1,4},8,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,128,32,4,8,32,32,2,1,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,4,8},{0,3,1,2},{0,2,1,3},2,8,1,true,1,1,{1,32,1,4},8,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,128,32,128,4,8,32,32,1,2,{1,4,4,8},{0,3,1,2},{0,2,1,3},2,8,1,true,{1,4,16,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,32,1,4},8,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,64,32,4,8,32,32,2,1,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,8,4,true,{1,4,4,4},{0,3,1,2},{0,2,1,3},2,8,2,true,1,1,{1,16,1,4},8,BF16,BF16),
        DeviceGroupedConvBwdWeight_Xdl_CShuffle<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,64,32,64,4,8,32,32,1,2,{1,4,4,4},{0,3,1,2},{0,2,1,3},2,8,2,true,{1,4,8,2},{0,3,1,2},{0,2,1,3},2,8,4,true,1,1,{1,16,1,4},8,BF16,BF16)
        // clang-format on
    };
}

// =====================================================================================
// F16 with BF8/F8 compute type instances
// Skipped for now: requires CK_ENABLE_FP8 && CK_ENABLE_BF8
// See device_grouped_conv_bwd_weight_xdl_c_shuffle_f16_comp_bf8_f8_instances in CK source.
// =====================================================================================

} // namespace grouped_conv_bwd_weight
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
