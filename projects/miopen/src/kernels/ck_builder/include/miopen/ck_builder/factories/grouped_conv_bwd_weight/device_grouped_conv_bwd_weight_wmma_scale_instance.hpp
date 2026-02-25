// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <miopen/ck_builder/instance_data/wmma_multi_d_bwd_weight.hpp>
#include <miopen/ck_builder/factories/grouped_conv_bwd_weight/common_aliases.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_bwd_weight {

using namespace instance;

// =====================================================================================
// F16 WMMA scale backward weight instances (F16/F16/F16/F32, NumDTensor=0)
// 10 instances
// Mapped from CK: device_grouped_conv_bwd_weight_wmma_c_shuffle_f16_scale_instances
//
// Scale variant uses:
//   - NumDTensor=0, DsLayout=Empty_Tuple, DsDataType=Empty_Tuple
//   - WeightElementwiseOp = Scale (instead of PassThrough)
//   - InputElementwiseOp = PassThrough, OutputElementwiseOp = PassThrough
//
// Note: The scale WMMA variant uses the same CK device class as bilinear
// (DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3) with transfer order S<2,0,1>.
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_wmma_c_shuffle_f16_scale_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    ckb::ConvSpecialization conv_spec)
{
    // TODO: Enable support for Wmma kernels on appropriate platforms later
    // return std::array{
    //     // clang-format off
    //     // generic instance
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,Scale,PassThrough,conv_spec,64,64,64,32,8,16,16,4,2,{4,8,1},{2,0,1},{1,0,2},1,2,4,true,{4,8,1},{2,0,1},{1,0,2},1,2,4,true,1,1,{1,16,1,4},2,1,Intrawave,PipeV1),
    //     // for fp16 conv.K and conv.C must be divisible by 2
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,Scale,PassThrough,conv_spec,64,32,32,32,8,16,16,2,1,{4,8,1},{2,0,1},{1,0,2},1,2,2,false,{4,16,1},{2,0,1},{1,0,2},1,2,2,false,1,1,{1,8,1,8},2,1,Intrawave,PipeV1),
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,Scale,PassThrough,conv_spec,128,128,128,32,8,16,16,8,2,{4,32,1},{2,0,1},{1,0,2},1,4,8,false,{4,32,1},{2,0,1},{1,0,2},1,4,8,false,1,1,{1,16,1,8},8,1,Intrawave,PipeV1),
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,Scale,PassThrough,conv_spec,128,128,128,32,8,16,16,8,2,{4,32,1},{2,0,1},{1,0,2},1,4,8,true,{4,32,1},{2,0,1},{1,0,2},1,4,8,false,1,1,{1,16,1,8},8,1,Intrawave,PipeV1),
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,Scale,PassThrough,conv_spec,64,64,64,64,8,16,16,4,2,{8,8,1},{2,0,1},{1,0,2},1,8,8,false,{8,8,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1,16,1,4},8,1,Intrawave,PipeV1),
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,Scale,PassThrough,conv_spec,256,128,256,64,8,16,16,8,2,{8,32,1},{2,0,1},{1,0,2},1,4,8,true,{8,32,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1,16,1,16},8,1,Intrawave,PipeV1),
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,Scale,PassThrough,conv_spec,128,48,64,128,8,16,16,3,1,{16,8,1},{2,0,1},{1,0,2},1,6,8,true,{16,8,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1,16,1,8},8,1,Intrawave,PipeV1),
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,Scale,PassThrough,conv_spec,128,96,128,64,8,16,16,6,2,{8,16,1},{2,0,1},{1,0,2},1,6,8,true,{8,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1,16,1,8},8,1,Intrawave,PipeV1),
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,Scale,PassThrough,conv_spec,128,64,64,128,8,16,16,4,1,{16,8,1},{2,0,1},{1,0,2},1,8,8,false,{16,8,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1,16,1,8},8,1,Intrawave,PipeV1),
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,Scale,PassThrough,conv_spec,256,96,128,128,8,16,16,6,1,{16,16,1},{2,0,1},{1,0,2},1,6,8,true,{16,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1,16,1,16},8,1,Intrawave,PipeV1)
    //     // clang-format on
    // };
    return std::array<WmmaMultiDBwdWeightInstance<0>, 0>{};
}

// =====================================================================================
// BF16 WMMA scale backward weight instances (BF16/BF16/BF16/F32, NumDTensor=0)
// 11 instances
// Mapped from CK: device_grouped_conv_bwd_weight_wmma_c_shuffle_bf16_scale_instances
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_wmma_c_shuffle_bf16_scale_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    ckb::ConvSpecialization conv_spec)
{
    // TODO: Enable support for Wmma kernels on appropriate platforms later
    // return std::array{
    //     // clang-format off
    //     // generic instance
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,Scale,PassThrough,conv_spec,64,64,64,32,8,16,16,4,2,{4,8,1},{2,0,1},{1,0,2},1,1,4,true,{4,8,1},{2,0,1},{1,0,2},1,1,4,true,1,1,{1,16,1,4},1,1,Intrawave,PipeV1),
    //     // other instances
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,Scale,PassThrough,conv_spec,64,32,32,32,8,16,16,2,1,{4,8,1},{2,0,1},{1,0,2},1,2,2,false,{4,16,1},{2,0,1},{1,0,2},1,2,2,false,1,1,{1,8,1,8},2,1,Intrawave,PipeV1),
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,Scale,PassThrough,conv_spec,128,128,128,32,8,16,16,8,2,{4,32,1},{2,0,1},{1,0,2},1,4,8,false,{4,32,1},{2,0,1},{1,0,2},1,4,8,false,1,1,{1,16,1,8},8,1,Intrawave,PipeV1),
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,Scale,PassThrough,conv_spec,128,128,128,32,8,16,16,8,2,{4,32,1},{2,0,1},{1,0,2},1,4,8,true,{4,32,1},{2,0,1},{1,0,2},1,4,8,false,1,1,{1,16,1,8},8,1,Intrawave,PipeV1),
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,Scale,PassThrough,conv_spec,64,64,64,64,8,16,16,4,2,{8,8,1},{2,0,1},{1,0,2},1,8,8,false,{8,8,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1,16,1,4},8,1,Intrawave,PipeV1),
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,Scale,PassThrough,conv_spec,256,128,256,64,8,16,16,8,2,{8,32,1},{2,0,1},{1,0,2},1,4,8,true,{8,32,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1,16,1,16},8,1,Intrawave,PipeV1),
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,Scale,PassThrough,conv_spec,128,48,64,128,8,16,16,3,1,{16,8,1},{2,0,1},{1,0,2},1,6,8,true,{16,8,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1,16,1,8},8,1,Intrawave,PipeV1),
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,Scale,PassThrough,conv_spec,128,96,128,64,8,16,16,6,2,{8,16,1},{2,0,1},{1,0,2},1,6,8,true,{8,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1,16,1,8},8,1,Intrawave,PipeV1),
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,Scale,PassThrough,conv_spec,128,64,64,128,8,16,16,4,1,{16,8,1},{2,0,1},{1,0,2},1,8,8,false,{16,8,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1,16,1,8},8,1,Intrawave,PipeV1),
    //     DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,Scale,PassThrough,conv_spec,256,96,128,128,8,16,16,6,1,{16,16,1},{2,0,1},{1,0,2},1,6,8,true,{16,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1,16,1,16},8,1,Intrawave,PipeV1)
    //     // clang-format on
    // };
    return std::array<WmmaMultiDBwdWeightInstance<0>, 0>{};
}

} // namespace grouped_conv_bwd_weight
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
