// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_weight_wmma_bilinear_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<3,
                                                               NDHWGC,
                                                               GKZYXC,
                                                               NDHWGK,
                                                               ck::Tuple<GKZYXC>,
                                                               F16,
                                                               F16,
                                                               F16,
                                                               ck::Tuple<F16>,
                                                               PassThrough,
                                                               Bilinear,
                                                               PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_weight_wmma_bilinear_ndhwgc_gkzyxc_ndhwgk_bf16_bf16_bf16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<3,
                                                               NDHWGC,
                                                               GKZYXC,
                                                               NDHWGK,
                                                               ck::Tuple<GKZYXC>,
                                                               BF16,
                                                               BF16,
                                                               BF16,
                                                               ck::Tuple<BF16>,
                                                               PassThrough,
                                                               Bilinear,
                                                               PassThrough>>>& instances);
#endif

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
