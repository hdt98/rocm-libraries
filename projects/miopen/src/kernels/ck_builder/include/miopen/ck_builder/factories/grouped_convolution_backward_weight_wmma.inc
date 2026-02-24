// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

// conv2d backward weight
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_weight_wmma_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvBwdWeight<2,
                                                               NHWGC,
                                                               GKYXC,
                                                               NHWGK,
                                                               F16,
                                                               F16,
                                                               F16,
                                                               PassThrough,
                                                               PassThrough,
                                                               PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_weight_two_stage_wmma_nhwgc_gkyxc_nhwgk_f16_pipev1_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvBwdWeight<2,
                                                               NHWGC,
                                                               GKYXC,
                                                               NHWGK,
                                                               F16,
                                                               F16,
                                                               F16,
                                                               PassThrough,
                                                               PassThrough,
                                                               PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_weight_wmma_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvBwdWeight<2,
                                                               NHWGC,
                                                               GKYXC,
                                                               NHWGK,
                                                               BF16,
                                                               BF16,
                                                               BF16,
                                                               PassThrough,
                                                               PassThrough,
                                                               PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_weight_two_stage_wmma_nhwgc_gkyxc_nhwgk_bf16_pipev1_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvBwdWeight<2,
                                                               NHWGC,
                                                               GKYXC,
                                                               NHWGK,
                                                               BF16,
                                                               BF16,
                                                               BF16,
                                                               PassThrough,
                                                               PassThrough,
                                                               PassThrough>>>& instances);
#endif

// conv3d backward weight
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_weight_wmma_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvBwdWeight<3,
                                                               NDHWGC,
                                                               GKZYXC,
                                                               NDHWGK,
                                                               F16,
                                                               F16,
                                                               F16,
                                                               PassThrough,
                                                               PassThrough,
                                                               PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_weight_two_stage_wmma_ndhwgc_gkzyxc_ndhwgk_f16_pipev1_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvBwdWeight<3,
                                                               NDHWGC,
                                                               GKZYXC,
                                                               NDHWGK,
                                                               F16,
                                                               F16,
                                                               F16,
                                                               PassThrough,
                                                               PassThrough,
                                                               PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_weight_wmma_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvBwdWeight<3,
                                                               NDHWGC,
                                                               GKZYXC,
                                                               NDHWGK,
                                                               BF16,
                                                               BF16,
                                                               BF16,
                                                               PassThrough,
                                                               PassThrough,
                                                               PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_weight_two_stage_wmma_ndhwgc_gkzyxc_ndhwgk_bf16_pipev1_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvBwdWeight<3,
                                                               NDHWGC,
                                                               GKZYXC,
                                                               NDHWGK,
                                                               BF16,
                                                               BF16,
                                                               BF16,
                                                               PassThrough,
                                                               PassThrough,
                                                               PassThrough>>>& instances);
#endif

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
