// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <miopen/ck_builder/instance_data/dl_bwd_weight.hpp>
#include <miopen/ck_builder/factories/grouped_conv_bwd_weight/common_aliases.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_bwd_weight {

using namespace instance;

// =====================================================================================
// F32 DL instances
// 1 instance: InData=F32, WeiData=F32, OutData=F32, AccData=F32
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_dl_f32_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    ckb::ConvSpecialization conv_spec)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightDl<0>(spatial_dim,in_layout,wei_layout,out_layout,F32,F32,F32,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,128,128,16,1,4,4,1,{8,2},{8,2},{1,8,1,1,1},{1,2,1,128,1},{0,2,3,1,4},{0,2,3,1,4},{1,1,1,1,1},{0,2,3,1,4},{1,1,1,1,1},{1,1,1,8,1},{1,16,1,16,1},{0,1,4,2,3},{0,1,4,2,3},{1,1,1,1,1},{0,1,4,2,3},{1,1,1,1,1},{0,1,2,3,4,5},5,1)
        // clang-format on
    };
}

// =====================================================================================
// F16 DL instances
// 1 instance: InData=F16, WeiData=F16, OutData=F16, AccData=F32
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_dl_f16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    ckb::ConvSpecialization conv_spec)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightDl<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,128,128,16,1,4,4,1,{8,2},{8,2},{1,8,1,1,1},{1,2,1,128,1},{0,2,3,1,4},{0,2,3,1,4},{1,1,1,1,1},{0,2,3,1,4},{1,1,1,1,1},{1,1,1,8,1},{1,16,1,16,1},{0,1,4,2,3},{0,1,4,2,3},{1,1,1,1,1},{0,1,4,2,3},{1,1,1,1,1},{0,1,2,3,4,5},5,1)
        // clang-format on
    };
}

// =====================================================================================
// BF16 DL instances (InData=BF16, WeiData=F32, OutData=BF16, AccData=F32)
// 1 instance
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_dl_bf16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    ckb::ConvSpecialization conv_spec)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightDl<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,F32,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,256,128,128,16,1,4,4,1,{8,2},{8,2},{1,8,1,1,1},{1,2,1,128,1},{0,2,3,1,4},{0,2,3,1,4},{1,1,1,1,1},{0,2,3,1,4},{1,1,1,1,1},{1,1,1,8,1},{1,16,1,16,1},{0,1,4,2,3},{0,1,4,2,3},{1,1,1,1,1},{0,1,4,2,3},{1,1,1,1,1},{0,1,2,3,4,5},5,1)
        // clang-format on
    };
}

} // namespace grouped_conv_bwd_weight
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
