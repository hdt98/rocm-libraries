// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<3,
                                                               NDHWGC,
                                                               GKZYXC,
                                                               NDHWGK,
                                                               Empty_Tuple,
                                                               F32,
                                                               F32,
                                                               F32,
                                                               Empty_Tuple,
                                                               PassThrough,
                                                               Scale,
                                                               PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<3,
                                                               NDHWGC,
                                                               GKZYXC,
                                                               NDHWGK,
                                                               Empty_Tuple,
                                                               F16,
                                                               F16,
                                                               F16,
                                                               Empty_Tuple,
                                                               PassThrough,
                                                               Scale,
                                                               PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<3,
                                                               NDHWGC,
                                                               GKZYXC,
                                                               NDHWGK,
                                                               Empty_Tuple,
                                                               BF16,
                                                               F32,
                                                               BF16,
                                                               Empty_Tuple,
                                                               PassThrough,
                                                               Scale,
                                                               PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_TF32
void add_device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<3,
                                                               NDHWGC,
                                                               GKZYXC,
                                                               NDHWGK,
                                                               Empty_Tuple,
                                                               F32,
                                                               F32,
                                                               F32,
                                                               Empty_Tuple,
                                                               PassThrough,
                                                               Scale,
                                                               PassThrough,
                                                               TF32,
                                                               TF32>>>& instances);
#endif

#if defined(CK_ENABLE_FP16) && defined(CK_ENABLE_FP8) && defined(CK_ENABLE_BF8)
void add_device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f16_comp_bf8_f8_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<3,
                                                               NDHWGC,
                                                               GKZYXC,
                                                               NDHWGK,
                                                               Empty_Tuple,
                                                               F16,
                                                               F16,
                                                               F16,
                                                               Empty_Tuple,
                                                               PassThrough,
                                                               Scale,
                                                               PassThrough,
                                                               BF8,
                                                               F8>>>& instances);
#endif

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
