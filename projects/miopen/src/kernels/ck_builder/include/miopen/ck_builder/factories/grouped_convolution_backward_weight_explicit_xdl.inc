// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {
// 2D
#ifdef CK_ENABLE_BF16

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_comp_default_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_comp_mnkpadding_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_mem_v1_default_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_mem_v1_mnkpadding_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_mem_v2_default_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_mem_v2_mnkpadding_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_odd_mn_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_odd_m_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_odd_n_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_comp_default_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_comp_mnkpadding_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_mem_v1_default_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_mem_v1_mnkpadding_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_mem_v2_default_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_mem_v2_mnkpadding_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_odd_mn_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_odd_m_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_odd_n_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_comp_default_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_comp_mnkpadding_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_mem_v1_default_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_mem_v1_mnkpadding_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_mem_v2_default_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_mem_v2_mnkpadding_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_odd_mn_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_odd_m_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_odd_n_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_comp_default_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_comp_mnkpadding_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_mem_v1_default_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_mem_v1_mnkpadding_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_mem_v2_default_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_mem_v2_mnkpadding_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_odd_mn_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_odd_m_instances(
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

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_odd_n_instances(
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
