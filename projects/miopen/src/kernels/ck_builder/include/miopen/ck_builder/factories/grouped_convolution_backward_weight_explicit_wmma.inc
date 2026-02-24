// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {
// 2D
#ifdef CK_ENABLE_BF16

void add_device_grouped_convnd_bwd_weight_wmma_bf16_bf16_bf16_exp_comp_default_instances(
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

void add_device_grouped_convnd_bwd_weight_wmma_bf16_bf16_bf16_exp_odd_mn_instances(
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

void add_device_grouped_convnd_bwd_weight_wmma_bf16_bf16_bf16_exp_comp_mnkpadding_instances(
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

#ifdef CK_ENABLE_FP16

void add_device_grouped_convnd_bwd_weight_wmma_f16_f16_f16_exp_comp_default_instances(
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

void add_device_grouped_convnd_bwd_weight_wmma_f16_f16_f16_exp_comp_mnkpadding_instances(
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

void add_device_grouped_convnd_bwd_weight_wmma_f16_f16_f16_exp_odd_mn_instances(
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

// 3D
#ifdef CK_ENABLE_BF16

void add_device_grouped_convnd_bwd_weight_wmma_bf16_bf16_bf16_exp_comp_default_instances(
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

void add_device_grouped_convnd_bwd_weight_wmma_bf16_bf16_bf16_exp_odd_mn_instances(
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

void add_device_grouped_convnd_bwd_weight_wmma_bf16_bf16_bf16_exp_comp_mnkpadding_instances(
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

#ifdef CK_ENABLE_FP16

void add_device_grouped_convnd_bwd_weight_wmma_f16_f16_f16_exp_comp_default_instances(
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

void add_device_grouped_convnd_bwd_weight_wmma_f16_f16_f16_exp_comp_mnkpadding_instances(
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

void add_device_grouped_convnd_bwd_weight_wmma_f16_f16_f16_exp_odd_mn_instances(
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

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
