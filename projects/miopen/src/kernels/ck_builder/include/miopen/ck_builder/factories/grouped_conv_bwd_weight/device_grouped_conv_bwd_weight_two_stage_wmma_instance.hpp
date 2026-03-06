// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <miopen/ck_builder/instance_data/wmma_two_stage_bwd_weight.hpp>
#include <miopen/ck_builder/factories/grouped_conv_bwd_weight/common_aliases.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_bwd_weight {

using namespace instance;

// =====================================================================================
// F16 two-stage WMMA backward weight instances (F16/F16/F16/F32)
// 1 active instance from CK source (others are commented out in CK)
// Mapped from CK: device_grouped_conv_bwd_weight_two_stage_nhwgc_wmma_c_shuffle_f16_instances
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_two_stage_nhwgc_wmma_c_shuffle_f16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    ckb::ConvSpecialization conv_spec,
    ckb::PipelineScheduler scheduler = Intrawave,
    ckb::PipelineVersion pipe_ver    = PipeV1)
{
    // TODO: Enable support for Wmma kernels on appropriate platforms later
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,conv_spec,32,16,16,32,8,16,16,1,1,{4,8,1},{2,0,1},{1,0,2},1,1,4,false,{4,8,1},{2,0,1},{1,0,2},1,1,4,false,1,1,{1,4,1,8},1,scheduler,pipe_ver,1)
    //     // clang-format on
    // };

    return std::array<WmmaTwoStageBwdWeightInstance<0>, 0>{};
}

// =====================================================================================
// BF16 two-stage WMMA backward weight instances (BF16/BF16/BF16/F32)
// 1 active instance from CK source (others are commented out in CK)
// Mapped from CK: device_grouped_conv_bwd_weight_two_stage_nhwgc_wmma_c_shuffle_bf16_instances
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_two_stage_nhwgc_wmma_c_shuffle_bf16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    ckb::ConvSpecialization conv_spec,
    ckb::PipelineScheduler scheduler = Intrawave,
    ckb::PipelineVersion pipe_ver    = PipeV1)
{
    // TODO: Enable support for Wmma kernels on appropriate platforms later
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3<0>(spatial_dim,in_layout,wei_layout,out_layout,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,conv_spec,32,16,16,32,8,16,16,1,1,{4,8,1},{2,0,1},{1,0,2},1,1,4,false,{4,8,1},{2,0,1},{1,0,2},1,1,4,false,1,1,{1,4,1,8},1,scheduler,pipe_ver,1)
    //     // clang-format on
    // };

    return std::array<WmmaTwoStageBwdWeightInstance<0>, 0>{};
}

} // namespace grouped_conv_bwd_weight
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
