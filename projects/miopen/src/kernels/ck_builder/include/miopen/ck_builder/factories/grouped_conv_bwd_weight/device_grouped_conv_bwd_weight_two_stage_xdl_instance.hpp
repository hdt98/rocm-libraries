// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <miopen/ck_builder/instance_data/xdl_two_stage_bwd_weight.hpp>
#include <miopen/ck_builder/factories/grouped_conv_bwd_weight/common_aliases.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_bwd_weight {

using namespace instance;

// =====================================================================================
// NHWGC F16 generic (1 instance)
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_two_stage_nhwgc_xdl_c_shuffle_f16_generic_instances(
    std::size_t sd,
    ckb::TensorLayout il,
    ckb::TensorLayout wl,
    ckb::TensorLayout ol,
    ckb::ConvSpecialization cs,
    ckb::PipelineScheduler sched,
    ckb::PipelineVersion pv)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs,64,16,16,32,8,16,16,1,1,{4,8,1},{2,0,1},{1,0,2},1,1,4,false,{4,8,1},{2,0,1},{1,0,2},1,1,4,false,1,1,{1,8,1,8},1,sched,pv,1,F16,F16)
        // clang-format on
    };
}

// =====================================================================================
// NHWGC F16 main (8 instances)
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_two_stage_nhwgc_xdl_c_shuffle_f16_instances(
    std::size_t sd,
    ckb::TensorLayout il,
    ckb::TensorLayout wl,
    ckb::TensorLayout ol,
    ckb::ConvSpecialization cs,
    ckb::PipelineScheduler sched,
    ckb::PipelineVersion pv)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,16, 32,8,16,16,1,1,{4, 8,1},{2,0,1},{1,0,2},1,1,4,false,{4, 8,1},{2,0,1},{1,0,2},1,1,4,false,1,1,{1, 8,1, 8},1,sched,pv,1,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,64, 32,8,32,32,1,2,{4, 8,1},{2,0,1},{1,0,2},1,2,2,false,{4,16,1},{2,0,1},{1,0,2},1,2,2,false,1,1,{1, 8,1, 8},1,sched,pv,2,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,32, 32,8,32,32,1,1,{4, 8,1},{2,0,1},{1,0,2},1,2,2,false,{4,16,1},{2,0,1},{1,0,2},1,2,2,false,1,1,{1, 8,1, 8},1,sched,pv,2,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,64, 32,8,32,32,1,2,{4, 8,1},{2,0,1},{1,0,2},1,4,4,false,{4,16,1},{2,0,1},{1,0,2},1,4,4,false,1,1,{1, 8,1, 8},1,sched,pv,4,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,128,32,8,32,32,1,4,{4, 4,1},{2,0,1},{1,0,2},1,8,8,false,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 8,1, 8},1,sched,pv,8,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,32, 32,8,32,32,1,1,{4,16,1},{2,0,1},{1,0,2},1,2,2,false,{4, 8,1},{2,0,1},{1,0,2},1,2,2,false,1,1,{1, 8,1, 8},1,sched,pv,2,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,64,32, 32,8,32,32,2,1,{4,16,1},{2,0,1},{1,0,2},1,4,4,false,{4, 8,1},{2,0,1},{1,0,2},1,4,4,false,1,1,{1, 8,1, 8},1,sched,pv,4,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,128,32,32,8,32,32,4,1,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,{4, 4,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 8,1, 8},1,sched,pv,8,F16,F16)
        // clang-format on
    };
}

// =====================================================================================
// NHWGC F16 part2 (7 instances)
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_two_stage_nhwgc_xdl_c_shuffle_f16_part2_instances(
    std::size_t sd,
    ckb::TensorLayout il,
    ckb::TensorLayout wl,
    ckb::TensorLayout ol,
    ckb::ConvSpecialization cs,
    ckb::PipelineScheduler sched,
    ckb::PipelineVersion pv)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs,256,64,64,64,8,32,32,1,1,{8, 8,1},{2,0,1},{2,0,1},1,8,8,false,{8, 8,1},{2,0,1},{2,0,1},1,8,8,false,1,1,{1,16,1,16},4,sched,pv,1,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs,256,64,64,64,8,32,32,1,1,{8,16,1},{2,0,1},{2,0,1},1,4,8,false,{8,32,1},{2,0,1},{2,0,1},1,1,8,false,1,1,{1, 4,1,64},1,sched,pv,1,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs,256,64,64,64,8,32,32,1,1,{8,32,1},{2,0,1},{2,0,1},1,1,8,false,{8,16,1},{2,0,1},{2,0,1},1,4,8,false,1,1,{1,16,1,16},4,sched,pv,1,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs,256,64,64,64,8,32,32,1,1,{8,32,1},{2,0,1},{2,0,1},1,2,8,false,{8,32,1},{2,0,1},{2,0,1},1,2,8,false,1,1,{1, 8,1,32},2,sched,pv,1,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs,256,64,64,64,8,32,32,1,1,{8,32,1},{2,0,1},{2,0,1},1,1,8,false,{8,32,1},{2,0,1},{2,0,1},1,1,8,false,1,1,{1, 4,1,64},1,sched,pv,1,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,256,32,8,16,16,1,16,{4, 2,1},{2,0,1},{1,0,2},1,8,8,false,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 4,1,16},1,sched,pv,8,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,128,32,8,16,16,1, 8,{4, 4,1},{2,0,1},{1,0,2},1,4,8,false,{4,16,1},{2,0,1},{1,0,2},1,4,8,false,1,1,{1, 4,1,16},1,sched,pv,4,F16,F16)
        // clang-format on
    };
}

// =====================================================================================
// NHWGC F16 irregular (5 instances)
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_two_stage_nhwgc_xdl_c_shuffle_f16_irregular_instances(
    std::size_t sd,
    ckb::TensorLayout il,
    ckb::TensorLayout wl,
    ckb::TensorLayout ol,
    ckb::ConvSpecialization cs,
    ckb::PipelineScheduler sched,
    ckb::PipelineVersion pv)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs,64, 48, 64,32,8,16,16,3,4,{4,16,1},{2,0,1},{2,0,1},1, 3, 4,false,{4,16,1},{2,0,1},{2,0,1},1, 4, 4,false,1,1,{1,8,1,8},1,sched,pv,1,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs,64, 64, 48,32,8,16,16,4,3,{4,16,1},{2,0,1},{2,0,1},1, 4, 4,false,{4,16,1},{2,0,1},{2,0,1},1, 3, 4,false,1,1,{1,8,1,8},1,sched,pv,1,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs,64, 64, 80,32,8,16,16,4,5,{4,16,1},{2,0,1},{2,0,1},1, 4, 4,false,{4,16,1},{2,0,1},{2,0,1},1, 5, 4,false,1,1,{1,8,1,8},1,sched,pv,1,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs,64, 64,112,32,8,16,16,4,7,{4,16,1},{2,0,1},{2,0,1},1, 4, 4,false,{4,16,1},{2,0,1},{2,0,1},1, 7, 4,false,1,1,{1,8,1,8},1,sched,pv,1,F16,F16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs,64, 64,208,32,8,16,16,4,13,{4,16,1},{2,0,1},{2,0,1},1,4, 4,false,{4,16,1},{2,0,1},{2,0,1},1,13, 4,false,1,1,{1,8,1,8},1,sched,pv,1,F16,F16)
        // clang-format on
    };
}

// =====================================================================================
// NHWGC BF16 generic (1 instance)
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_two_stage_nhwgc_xdl_c_shuffle_bf16_generic_instances(
    std::size_t sd,
    ckb::TensorLayout il,
    ckb::TensorLayout wl,
    ckb::TensorLayout ol,
    ckb::ConvSpecialization cs,
    ckb::PipelineScheduler sched,
    ckb::PipelineVersion pv)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs,64,16,16,32,8,16,16,1,1,{4,8,1},{2,0,1},{1,0,2},1,1,4,false,{4,8,1},{2,0,1},{1,0,2},1,1,4,false,1,1,{1,8,1,8},1,sched,pv,1,BF16,BF16)
        // clang-format on
    };
}

// =====================================================================================
// NHWGC BF16 main (8 instances)
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_two_stage_nhwgc_xdl_c_shuffle_bf16_instances(
    std::size_t sd,
    ckb::TensorLayout il,
    ckb::TensorLayout wl,
    ckb::TensorLayout ol,
    ckb::ConvSpecialization cs,
    ckb::PipelineScheduler sched,
    ckb::PipelineVersion pv)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,16, 32,8,16,16,1,1,{4, 8,1},{2,0,1},{1,0,2},1,1,4,false,{4, 8,1},{2,0,1},{1,0,2},1,1,4,false,1,1,{1, 8,1, 8},1,sched,pv,1,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,64, 32,8,32,32,1,2,{4, 8,1},{2,0,1},{1,0,2},1,2,2,false,{4,16,1},{2,0,1},{1,0,2},1,2,2,false,1,1,{1, 8,1, 8},1,sched,pv,2,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,32, 32,8,32,32,1,1,{4, 8,1},{2,0,1},{1,0,2},1,2,2,false,{4,16,1},{2,0,1},{1,0,2},1,2,2,false,1,1,{1, 8,1, 8},1,sched,pv,2,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,64, 32,8,32,32,1,2,{4, 8,1},{2,0,1},{1,0,2},1,4,4,false,{4,16,1},{2,0,1},{1,0,2},1,4,4,false,1,1,{1, 8,1, 8},1,sched,pv,4,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,128,32,8,32,32,1,4,{4, 4,1},{2,0,1},{1,0,2},1,8,8,false,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 8,1, 8},1,sched,pv,8,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,32, 32,8,32,32,1,1,{4,16,1},{2,0,1},{1,0,2},1,2,2,false,{4, 8,1},{2,0,1},{1,0,2},1,2,2,false,1,1,{1, 8,1, 8},1,sched,pv,2,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,64,32, 32,8,32,32,2,1,{4,16,1},{2,0,1},{1,0,2},1,4,4,false,{4, 8,1},{2,0,1},{1,0,2},1,4,4,false,1,1,{1, 8,1, 8},1,sched,pv,4,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,128,32,32,8,32,32,4,1,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,{4, 4,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 8,1, 8},1,sched,pv,8,BF16,BF16)
        // clang-format on
    };
}

// =====================================================================================
// NHWGC BF16 part2 (7 instances)
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_two_stage_nhwgc_xdl_c_shuffle_bf16_part2_instances(
    std::size_t sd,
    ckb::TensorLayout il,
    ckb::TensorLayout wl,
    ckb::TensorLayout ol,
    ckb::ConvSpecialization cs,
    ckb::PipelineScheduler sched,
    ckb::PipelineVersion pv)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs,256,64,64,64,8,32,32,1,1,{8, 4,1},{2,0,1},{2,0,1},1,8,8,false,{8, 4,1},{2,0,1},{2,0,1},1,8,8,false,1,1,{1,16,1,16},4,sched,pv,1,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs,256,64,64,64,8,32,32,1,1,{8,16,1},{2,0,1},{2,0,1},1,4,8,false,{8,32,1},{2,0,1},{2,0,1},1,1,8,false,1,1,{1, 4,1,64},1,sched,pv,1,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs,256,64,64,64,8,32,32,1,1,{8,32,1},{2,0,1},{2,0,1},1,1,8,false,{8,16,1},{2,0,1},{2,0,1},1,4,8,false,1,1,{1,16,1,16},4,sched,pv,1,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs,256,64,64,64,8,32,32,1,1,{8,32,1},{2,0,1},{2,0,1},1,2,8,false,{8,32,1},{2,0,1},{2,0,1},1,2,8,false,1,1,{1, 8,1,32},2,sched,pv,1,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs,256,64,64,64,8,32,32,1,1,{8,32,1},{2,0,1},{2,0,1},1,1,8,false,{8,32,1},{2,0,1},{2,0,1},1,1,8,false,1,1,{1, 4,1,64},1,sched,pv,1,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,256,32,8,16,16,1,16,{4, 2,1},{2,0,1},{1,0,2},1,8,8,false,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 4,1,16},1,sched,pv,8,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,128,32,8,16,16,1, 8,{4, 4,1},{2,0,1},{1,0,2},1,4,8,false,{4,16,1},{2,0,1},{1,0,2},1,4,8,false,1,1,{1, 4,1,16},1,sched,pv,4,BF16,BF16)
        // clang-format on
    };
}

// =====================================================================================
// NHWGC BF16 irregular (5 instances)
// =====================================================================================
constexpr auto
device_grouped_conv_bwd_weight_two_stage_nhwgc_xdl_c_shuffle_bf16_irregular_instances(
    std::size_t sd,
    ckb::TensorLayout il,
    ckb::TensorLayout wl,
    ckb::TensorLayout ol,
    ckb::ConvSpecialization cs,
    ckb::PipelineScheduler sched,
    ckb::PipelineVersion pv)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs,64, 48, 64,32,8,16,16,3,4,{4,16,1},{2,0,1},{2,0,1},1, 3, 4,false,{4,16,1},{2,0,1},{2,0,1},1, 4, 4,false,1,1,{1,8,1,8},1,sched,pv,1,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs,64, 64, 48,32,8,16,16,4,3,{4,16,1},{2,0,1},{2,0,1},1, 4, 4,false,{4,16,1},{2,0,1},{2,0,1},1, 3, 4,false,1,1,{1,8,1,8},1,sched,pv,1,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs,64, 64, 80,32,8,16,16,4,5,{4,16,1},{2,0,1},{2,0,1},1, 4, 4,false,{4,16,1},{2,0,1},{2,0,1},1, 5, 4,false,1,1,{1,8,1,8},1,sched,pv,1,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs,64, 64,112,32,8,16,16,4,7,{4,16,1},{2,0,1},{2,0,1},1, 4, 4,false,{4,16,1},{2,0,1},{2,0,1},1, 7, 4,false,1,1,{1,8,1,8},1,sched,pv,1,BF16,BF16),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs,64, 64,208,32,8,16,16,4,13,{4,16,1},{2,0,1},{2,0,1},1,4, 4,false,{4,16,1},{2,0,1},{2,0,1},1,13, 4,false,1,1,{1,8,1,8},1,sched,pv,1,BF16,BF16)
        // clang-format on
    };
}

// =====================================================================================
// NGCHW F16 generic (1 instance) — with compute types and transpose transfer
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_two_stage_ngchw_xdl_c_shuffle_f16_generic_instances(
    std::size_t sd,
    ckb::TensorLayout il,
    ckb::TensorLayout wl,
    ckb::TensorLayout ol,
    ckb::ConvSpecialization cs,
    ckb::PipelineScheduler sched,
    ckb::PipelineVersion pv)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs,64,16,16,32,8,16,16,1,1,{4,8,1},{2,0,1},{1,0,2},1,1,4,false,{4,8,1},{2,0,1},{1,0,2},1,1,4,false,1,1,{1,8,1,8},1,sched,pv,1,F16,F16,1,1)
        // clang-format on
    };
}

// =====================================================================================
// NGCHW F16 main (18 instances)
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_two_stage_ngchw_xdl_c_shuffle_f16_instances(
    std::size_t sd,
    ckb::TensorLayout il,
    ckb::TensorLayout wl,
    ckb::TensorLayout ol,
    ckb::ConvSpecialization cs,
    ckb::PipelineScheduler sched,
    ckb::PipelineVersion pv)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,16, 32,8,16,16,1,1,{4, 8,1},{2,0,1},{1,0,2},1,1,4,false,{4, 8,1},{2,0,1},{1,0,2},1,1,4,false,1,1,{1, 8,1, 8},1,sched,pv,1,F16,F16,1,1),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,32, 32,8,32,32,1,1,{4, 8,1},{2,0,1},{1,0,2},1,2,2,false,{4,16,1},{2,0,1},{1,0,2},1,2,2,false,1,1,{1, 8,1, 8},1,sched,pv,2,F16,F16,2,2),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,64, 32,8,32,32,1,2,{4, 8,1},{2,0,1},{1,0,2},1,4,4,false,{4,16,1},{2,0,1},{1,0,2},1,4,4,false,1,1,{1, 8,1, 8},1,sched,pv,4,F16,F16,4,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,128,32,8,32,32,1,4,{4, 4,1},{2,0,1},{1,0,2},1,8,8,false,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 4,1, 8},1,sched,pv,8,F16,F16,8,8),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,32, 32,8,32,32,1,1,{4,16,1},{2,0,1},{1,0,2},1,2,2,false,{4, 8,1},{2,0,1},{1,0,2},1,2,2,false,1,1,{1, 8,1, 8},1,sched,pv,2,F16,F16,2,2),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,64,32, 32,8,32,32,2,1,{4,16,1},{2,0,1},{1,0,2},1,4,4,false,{4, 8,1},{2,0,1},{1,0,2},1,4,4,false,1,1,{1, 8,1, 8},1,sched,pv,4,F16,F16,4,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,128,32,32,8,32,32,4,1,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,{4, 4,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 8,1, 4},1,sched,pv,8,F16,F16,8,8),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,32, 32,8,32,32,1,1,{4, 8,1},{2,0,1},{1,0,2},1,2,2,false,{4,16,1},{2,0,1},{1,0,2},1,2,2,false,1,1,{1, 8,1, 8},1,sched,pv,2,F16,F16,1,2),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,64, 32,8,32,32,1,2,{4, 8,1},{2,0,1},{1,0,2},1,4,4,false,{4,16,1},{2,0,1},{1,0,2},1,4,4,false,1,1,{1, 8,1, 8},1,sched,pv,4,F16,F16,1,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,128,32,8,32,32,1,4,{4, 4,1},{2,0,1},{1,0,2},1,8,8,false,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 4,1, 8},1,sched,pv,8,F16,F16,1,8),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,64,32, 32,8,32,32,2,1,{4,16,1},{2,0,1},{1,0,2},1,4,4,false,{4, 8,1},{2,0,1},{1,0,2},1,4,4,false,1,1,{1, 8,1, 8},1,sched,pv,4,F16,F16,1,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,128,32,32,8,32,32,4,1,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,{4, 4,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 8,1, 4},1,sched,pv,8,F16,F16,1,8),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,32, 32,8,32,32,1,1,{4, 8,1},{2,0,1},{1,0,2},1,2,2,false,{4,16,1},{2,0,1},{1,0,2},1,2,2,false,1,1,{1, 8,1, 8},1,sched,pv,2,F16,F16,2,1),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,64, 32,8,32,32,1,2,{4, 8,1},{2,0,1},{1,0,2},1,4,4,false,{4,16,1},{2,0,1},{1,0,2},1,4,4,false,1,1,{1, 8,1, 8},1,sched,pv,4,F16,F16,4,1),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,128,32,8,32,32,1,4,{4, 4,1},{2,0,1},{1,0,2},1,8,8,false,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 4,1, 8},1,sched,pv,8,F16,F16,8,1),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,64,32, 32,8,32,32,2,1,{4,16,1},{2,0,1},{1,0,2},1,4,4,false,{4, 8,1},{2,0,1},{1,0,2},1,4,4,false,1,1,{1, 8,1, 8},1,sched,pv,4,F16,F16,4,1),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,128,32,32,8,32,32,4,1,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,{4, 4,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 8,1, 4},1,sched,pv,8,F16,F16,8,1)
        // clang-format on
    };
}

// =====================================================================================
// NGCHW F16 part2 (9 instances)
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_two_stage_ngchw_xdl_c_shuffle_f16_part2_instances(
    std::size_t sd,
    ckb::TensorLayout il,
    ckb::TensorLayout wl,
    ckb::TensorLayout ol,
    ckb::ConvSpecialization cs,
    ckb::PipelineScheduler sched,
    ckb::PipelineVersion pv)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs,256,64,64,64,8,32,32,1,1,{8, 8,1},{2,0,1},{2,0,1},1,8,8,false,{8, 8,1},{2,0,1},{2,0,1},1,8,8,false,1,1,{1,16,1,16},4,sched,pv,1,F16,F16,4,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs,256,64,64,64,8,32,32,1,1,{8,32,1},{2,0,1},{2,0,1},1,2,8,false,{8,32,1},{2,0,1},{2,0,1},1,2,8,false,1,1,{1, 8,1,32},2,sched,pv,1,F16,F16,2,2),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs,256,64,64,64,8,32,32,1,1,{8,32,1},{2,0,1},{2,0,1},1,1,8,false,{8,32,1},{2,0,1},{2,0,1},1,1,8,false,1,1,{1, 4,1,64},1,sched,pv,1,F16,F16,1,1),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,256,32,8,16,16,1,16,{4, 2,1},{2,0,1},{1,0,2},1,8,8,false,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 4,1,16},1,sched,pv,8,F16,F16,4,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,256,32,8,16,16,1,16,{4, 2,1},{2,0,1},{1,0,2},1,8,8,false,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 4,1,16},1,sched,pv,8,F16,F16,2,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,256,32,8,16,16,1,16,{4, 2,1},{2,0,1},{1,0,2},1,8,8,false,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 4,1,16},1,sched,pv,8,F16,F16,1,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,128,32,8,16,16,1, 8,{4, 4,1},{2,0,1},{1,0,2},1,4,8,false,{4,16,1},{2,0,1},{1,0,2},1,4,8,false,1,1,{1, 4,1,16},1,sched,pv,4,F16,F16,4,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,128,32,8,16,16,1, 8,{4, 4,1},{2,0,1},{1,0,2},1,4,8,false,{4,16,1},{2,0,1},{1,0,2},1,4,8,false,1,1,{1, 4,1,16},1,sched,pv,4,F16,F16,2,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,F16,F16,F16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,128,32,8,16,16,1, 8,{4, 4,1},{2,0,1},{1,0,2},1,4,8,false,{4,16,1},{2,0,1},{1,0,2},1,4,8,false,1,1,{1, 4,1,16},1,sched,pv,4,F16,F16,1,4)
        // clang-format on
    };
}

// =====================================================================================
// NGCHW BF16 generic (1 instance)
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_two_stage_ngchw_xdl_c_shuffle_bf16_generic_instances(
    std::size_t sd,
    ckb::TensorLayout il,
    ckb::TensorLayout wl,
    ckb::TensorLayout ol,
    ckb::ConvSpecialization cs,
    ckb::PipelineScheduler sched,
    ckb::PipelineVersion pv)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs,64,16,16,32,8,16,16,1,1,{4,8,1},{2,0,1},{1,0,2},1,1,4,false,{4,8,1},{2,0,1},{1,0,2},1,1,4,false,1,1,{1,8,1,8},1,sched,pv,1,BF16,BF16,1,1)
        // clang-format on
    };
}

// =====================================================================================
// NGCHW BF16 main (18 instances)
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_two_stage_ngchw_xdl_c_shuffle_bf16_instances(
    std::size_t sd,
    ckb::TensorLayout il,
    ckb::TensorLayout wl,
    ckb::TensorLayout ol,
    ckb::ConvSpecialization cs,
    ckb::PipelineScheduler sched,
    ckb::PipelineVersion pv)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,16, 32,8,16,16,1,1,{4, 8,1},{2,0,1},{1,0,2},1,1,4,false,{4, 8,1},{2,0,1},{1,0,2},1,1,4,false,1,1,{1, 8,1, 8},1,sched,pv,1,BF16,BF16,1,1),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,32, 32,8,32,32,1,1,{4, 8,1},{2,0,1},{1,0,2},1,2,2,false,{4,16,1},{2,0,1},{1,0,2},1,2,2,false,1,1,{1, 8,1, 8},1,sched,pv,2,BF16,BF16,2,2),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,64, 32,8,32,32,1,2,{4, 8,1},{2,0,1},{1,0,2},1,4,4,false,{4,16,1},{2,0,1},{1,0,2},1,4,4,false,1,1,{1, 8,1, 8},1,sched,pv,4,BF16,BF16,4,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,128,32,8,32,32,1,4,{4, 4,1},{2,0,1},{1,0,2},1,8,8,false,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 4,1, 8},1,sched,pv,8,BF16,BF16,8,8),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,32, 32,8,32,32,1,1,{4,16,1},{2,0,1},{1,0,2},1,2,2,false,{4, 8,1},{2,0,1},{1,0,2},1,2,2,false,1,1,{1, 8,1, 8},1,sched,pv,2,BF16,BF16,2,2),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,64,32, 32,8,32,32,2,1,{4,16,1},{2,0,1},{1,0,2},1,4,4,false,{4, 8,1},{2,0,1},{1,0,2},1,4,4,false,1,1,{1, 8,1, 8},1,sched,pv,4,BF16,BF16,4,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,128,32,32,8,32,32,4,1,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,{4, 4,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 8,1, 4},1,sched,pv,8,BF16,BF16,8,8),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,32, 32,8,32,32,1,1,{4, 8,1},{2,0,1},{1,0,2},1,2,2,false,{4,16,1},{2,0,1},{1,0,2},1,2,2,false,1,1,{1, 8,1, 8},1,sched,pv,2,BF16,BF16,1,2),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,64, 32,8,32,32,1,2,{4, 8,1},{2,0,1},{1,0,2},1,4,4,false,{4,16,1},{2,0,1},{1,0,2},1,4,4,false,1,1,{1, 8,1, 8},1,sched,pv,4,BF16,BF16,1,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,128,32,8,32,32,1,4,{4, 4,1},{2,0,1},{1,0,2},1,8,8,false,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 4,1, 8},1,sched,pv,8,BF16,BF16,1,8),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,64,32, 32,8,32,32,2,1,{4,16,1},{2,0,1},{1,0,2},1,4,4,false,{4, 8,1},{2,0,1},{1,0,2},1,4,4,false,1,1,{1, 8,1, 8},1,sched,pv,4,BF16,BF16,1,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,128,32,32,8,32,32,4,1,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,{4, 4,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 8,1, 4},1,sched,pv,8,BF16,BF16,1,8),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,32, 32,8,32,32,1,1,{4, 8,1},{2,0,1},{1,0,2},1,2,2,false,{4,16,1},{2,0,1},{1,0,2},1,2,2,false,1,1,{1, 8,1, 8},1,sched,pv,2,BF16,BF16,2,1),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,64, 32,8,32,32,1,2,{4, 8,1},{2,0,1},{1,0,2},1,4,4,false,{4,16,1},{2,0,1},{1,0,2},1,4,4,false,1,1,{1, 8,1, 8},1,sched,pv,4,BF16,BF16,4,1),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,32,128,32,8,32,32,1,4,{4, 4,1},{2,0,1},{1,0,2},1,8,8,false,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 4,1, 8},1,sched,pv,8,BF16,BF16,8,1),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,64,32, 32,8,32,32,2,1,{4,16,1},{2,0,1},{1,0,2},1,4,4,false,{4, 8,1},{2,0,1},{1,0,2},1,4,4,false,1,1,{1, 8,1, 8},1,sched,pv,4,BF16,BF16,4,1),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,128,32,32,8,32,32,4,1,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,{4, 4,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 8,1, 4},1,sched,pv,8,BF16,BF16,8,1)
        // clang-format on
    };
}

// =====================================================================================
// NGCHW BF16 part2 (9 instances)
// =====================================================================================
constexpr auto device_grouped_conv_bwd_weight_two_stage_ngchw_xdl_c_shuffle_bf16_part2_instances(
    std::size_t sd,
    ckb::TensorLayout il,
    ckb::TensorLayout wl,
    ckb::TensorLayout ol,
    ckb::ConvSpecialization cs,
    ckb::PipelineScheduler sched,
    ckb::PipelineVersion pv)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs,256,64,64,64,8,32,32,1,1,{8, 8,1},{2,0,1},{2,0,1},1,8,8,false,{8, 8,1},{2,0,1},{2,0,1},1,8,8,false,1,1,{1,16,1,16},4,sched,pv,1,BF16,BF16,4,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs,256,64,64,64,8,32,32,1,1,{8,32,1},{2,0,1},{2,0,1},1,2,8,false,{8,32,1},{2,0,1},{2,0,1},1,2,8,false,1,1,{1, 8,1,32},2,sched,pv,1,BF16,BF16,2,2),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs,256,64,64,64,8,32,32,1,1,{8,32,1},{2,0,1},{2,0,1},1,1,8,false,{8,32,1},{2,0,1},{2,0,1},1,1,8,false,1,1,{1, 4,1,64},1,sched,pv,1,BF16,BF16,1,1),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,256,32,8,16,16,1,16,{4, 2,1},{2,0,1},{1,0,2},1,8,8,false,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 4,1,16},1,sched,pv,8,BF16,BF16,4,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,256,32,8,16,16,1,16,{4, 2,1},{2,0,1},{1,0,2},1,8,8,false,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 4,1,16},1,sched,pv,8,BF16,BF16,2,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,256,32,8,16,16,1,16,{4, 2,1},{2,0,1},{1,0,2},1,8,8,false,{4,16,1},{2,0,1},{1,0,2},1,8,8,false,1,1,{1, 4,1,16},1,sched,pv,8,BF16,BF16,1,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,128,32,8,16,16,1, 8,{4, 4,1},{2,0,1},{1,0,2},1,4,8,false,{4,16,1},{2,0,1},{1,0,2},1,4,8,false,1,1,{1, 4,1,16},1,sched,pv,4,BF16,BF16,4,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,128,32,8,16,16,1, 8,{4, 4,1},{2,0,1},{1,0,2},1,4,8,false,{4,16,1},{2,0,1},{1,0,2},1,4,8,false,1,1,{1, 4,1,16},1,sched,pv,4,BF16,BF16,2,4),
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<0>(sd,il,wl,ol,BF16,BF16,BF16,F32,PassThrough,PassThrough,PassThrough,cs, 64,16,128,32,8,16,16,1, 8,{4, 4,1},{2,0,1},{1,0,2},1,4,8,false,{4,16,1},{2,0,1},{1,0,2},1,4,8,false,1,1,{1, 4,1,16},1,sched,pv,4,BF16,BF16,1,4)
        // clang-format on
    };
}

} // namespace grouped_conv_bwd_weight
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
