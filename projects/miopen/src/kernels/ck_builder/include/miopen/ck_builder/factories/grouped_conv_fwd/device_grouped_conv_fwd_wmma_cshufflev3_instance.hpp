// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <miopen/ck_builder/instance_data/wmma_v3.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/common_aliases.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_fwd {

using namespace instance;

// Note: DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3 is UNSUPPORTED by CK Builder.
// All functions return empty arrays as stubs.
// Commented-out instances below document what the original CK source provides.

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_cshufflev3_bf16_generic_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // UNSUPPORTED: DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,64,64,64,32,8,8,16,16,4,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,Intrawave,PipeV1)
    //     // clang-format on
    // };
    return std::array<int, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_cshufflev3_bf16_instances_part1(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // UNSUPPORTED: DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,64,64,64,32,8,8,16,16,4,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,128,32,8,8,16,16,4,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,32,1,8},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,64,64,64,32,8,8,16,16,4,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,64,64,32,32,8,8,16,16,4,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,256,32,8,8,16,16,4,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,Interwave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,128,64,8,8,16,16,4,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,48,64,8,8,16,16,2,3,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,64,1,2},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,48,64,8,8,16,16,2,3,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,64,1,2},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,64,64,64,8,8,16,16,2,1,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,64,64,64,8,8,16,16,2,1,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},1,Intrawave,PipeV1)
    //     // clang-format on
    // };
    return std::array<int, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_cshufflev3_bf16_instances_part2(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // UNSUPPORTED: DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,128,32,8,8,16,16,4,4,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,64,32,64,8,8,16,16,2,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,64,64,64,8,8,16,16,2,2,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,64,64,64,8,8,16,16,2,2,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,64,64,64,8,8,16,16,2,2,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,64,96,64,8,8,16,16,2,3,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,64,96,64,8,8,16,16,2,3,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,96,32,8,8,16,16,4,3,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,96,32,8,8,16,16,4,3,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1)
    //     // clang-format on
    // };
    return std::array<int, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_cshufflev3_bf16_instances_part3(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // UNSUPPORTED: DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,96,64,8,8,16,16,4,3,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,256,64,8,8,16,16,4,4,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,96,64,8,8,16,16,2,3,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,64,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,96,64,8,8,16,16,2,3,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,64,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,96,64,8,8,16,16,2,3,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,64,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,64,32,8,8,16,16,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,64,32,8,8,16,16,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,64,32,8,8,16,16,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,32,32,8,8,16,16,4,1,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,Intrawave,PipeV1)
    //     // clang-format on
    // };
    return std::array<int, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_cshufflev3_bf16_instances_part4(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // UNSUPPORTED: DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,32,32,8,8,16,16,4,1,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,32,32,8,8,16,16,4,1,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,256,64,64,8,8,16,16,4,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,64,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,256,32,8,8,16,16,4,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,128,32,8,8,16,16,8,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,128,32,8,8,16,16,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,64,128,32,8,8,16,16,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,64,64,32,8,8,16,16,1,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,2,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,64,64,32,8,8,16,16,1,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,Intrawave,PipeV1)
    //     // clang-format on
    // };
    return std::array<int, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_cshufflev3_f16_generic_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // UNSUPPORTED: DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,64,64,64,32,8,8,16,16,4,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,Intrawave,PipeV1)
    //     // clang-format on
    // };
    return std::array<int, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_cshufflev3_f16_instances_part1(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // UNSUPPORTED: DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,64,64,64,32,8,8,16,16,4,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,128,32,8,8,16,16,4,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,32,1,8},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,64,64,64,32,8,8,16,16,4,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,64,64,32,32,8,8,16,16,4,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,256,32,8,8,16,16,4,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,Interwave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,128,64,8,8,16,16,4,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,48,64,8,8,16,16,2,3,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,64,1,2},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,48,64,8,8,16,16,2,3,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,64,1,2},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,64,64,64,8,8,16,16,2,1,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},1,Intrawave,PipeV1)
    //     // clang-format on
    // };
    return std::array<int, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_cshufflev3_f16_instances_part2(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // UNSUPPORTED: DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,64,64,64,8,8,16,16,2,1,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,128,32,8,8,16,16,4,4,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,64,32,64,8,8,16,16,2,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,64,64,64,8,8,16,16,2,2,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,64,64,64,8,8,16,16,2,2,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,64,64,64,8,8,16,16,2,2,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,64,96,64,8,8,16,16,2,3,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,64,96,64,8,8,16,16,2,3,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,96,32,8,8,16,16,4,3,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,96,32,8,8,16,16,4,3,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1)
    //     // clang-format on
    // };
    return std::array<int, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_cshufflev3_f16_instances_part3(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // UNSUPPORTED: DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,96,64,8,8,16,16,4,3,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,{8,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,256,64,8,8,16,16,4,4,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,96,64,8,8,16,16,2,3,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,64,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,96,64,8,8,16,16,2,3,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,64,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,96,64,8,8,16,16,2,3,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,64,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,64,32,8,8,16,16,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,64,32,8,8,16,16,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,64,32,8,8,16,16,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,32,32,8,8,16,16,4,1,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,Intrawave,PipeV1)
    //     // clang-format on
    // };
    return std::array<int, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_cshufflev3_f16_instances_part4(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // UNSUPPORTED: DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,32,32,8,8,16,16,4,1,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,32,32,8,8,16,16,4,1,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,256,64,64,8,8,16,16,4,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,{8,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,64,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,256,32,8,8,16,16,4,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,128,128,32,8,8,16,16,8,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,128,128,32,8,8,16,16,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,128,64,128,32,8,8,16,16,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,64,64,32,8,8,16,16,1,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,2,8,true,1,1,{1,32,1,4},1,Intrawave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,256,64,64,32,8,8,16,16,1,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,Intrawave,PipeV1)
    //     // clang-format on
    // };
    return std::array<int, 0>{};
}

} // namespace grouped_conv_fwd
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
