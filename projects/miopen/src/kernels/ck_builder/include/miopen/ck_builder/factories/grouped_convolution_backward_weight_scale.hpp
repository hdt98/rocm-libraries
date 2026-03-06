// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight_multiple_d.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>

#include "device_operation_instance_factory.hpp"

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

#ifdef CK_USE_XDL
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<
        ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<3,
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
                                                                          PassThrough>>>&
        instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<
        ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<3,
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
                                                                          PassThrough>>>&
        instances);
#endif

#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<
        ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<3,
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
                                                                          PassThrough>>>&
        instances);
#endif

#ifdef CK_ENABLE_TF32
void add_device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<
        ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<3,
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
    std::vector<std::unique_ptr<
        ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<3,
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
#endif // CK_USE_XDL

#ifdef CK_USE_WMMA
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_weight_wmma_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<
        ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<3,
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
                                                                          PassThrough>>>&
        instances);
#endif

#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_weight_wmma_scale_ndhwgc_gkzyxc_ndhwgk_bf16_bf16_bf16_instances(
    std::vector<std::unique_ptr<
        ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<3,
                                                                          NDHWGC,
                                                                          GKZYXC,
                                                                          NDHWGK,
                                                                          Empty_Tuple,
                                                                          BF16,
                                                                          BF16,
                                                                          BF16,
                                                                          Empty_Tuple,
                                                                          PassThrough,
                                                                          Scale,
                                                                          PassThrough>>>&
        instances);
#endif
#endif // CK_USE_WMMA

template <ck::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename DsLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename DsDataType,
          typename ComputeTypeA,
          typename ComputeTypeB>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<
        NumDimSpatial,
        InLayout,
        WeiLayout,
        OutLayout,
        DsLayout,
        InDataType,
        WeiDataType,
        OutDataType,
        DsDataType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::Scale,
        ck::tensor_operation::element_wise::PassThrough,
        ComputeTypeA,
        ComputeTypeB>>
{
    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<
        NumDimSpatial,
        InLayout,
        WeiLayout,
        OutLayout,
        DsLayout,
        InDataType,
        WeiDataType,
        OutDataType,
        DsDataType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::Scale,
        ck::tensor_operation::element_wise::PassThrough,
        ComputeTypeA,
        ComputeTypeB>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_USE_XDL
        if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, NDHWGC> &&
                     is_same_v<WeiLayout, GKZYXC> && is_same_v<OutLayout, NDHWGK> &&
                     DsLayout::Size() == 0)
        {
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float>)
            {
#ifdef CK_ENABLE_TF32
                if constexpr(is_same_v<ComputeTypeA, ck::tf32_t> &&
                             is_same_v<ComputeTypeB, ck::tf32_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<ComputeTypeA, float> && is_same_v<ComputeTypeB, float>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_instances(
                        op_ptrs);
                }
#endif
            }
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, ck::half_t> && is_same_v<WeiDataType, ck::half_t> &&
                         is_same_v<OutDataType, ck::half_t>)
            {
#if defined(CK_ENABLE_FP8) && defined(CK_ENABLE_BF8)
                if constexpr(is_same_v<ComputeTypeA, ck::bf8_t> &&
                             is_same_v<ComputeTypeB, ck::f8_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f16_comp_bf8_f8_instances(
                        op_ptrs);
                }
#endif
                if constexpr(is_same_v<ComputeTypeA, ck::half_t> &&
                             is_same_v<ComputeTypeB, ck::half_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances(
                        op_ptrs);
                }
            }
#endif
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, ck::bhalf_t> &&
                         is_same_v<ComputeTypeA, ck::bhalf_t> &&
                         is_same_v<ComputeTypeB, ck::bhalf_t>)
            {
                add_device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_bf16_f32_bf16_instances(
                    op_ptrs);
            }
#endif
        }
#endif

#ifdef CK_USE_WMMA
        if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, NDHWGC> &&
                     is_same_v<WeiLayout, GKZYXC> && is_same_v<OutLayout, NDHWGK> &&
                     DsLayout::Size() == 0)
        {
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, ck::half_t> && is_same_v<WeiDataType, ck::half_t> &&
                         is_same_v<OutDataType, ck::half_t> &&
                         is_same_v<ComputeTypeA, ck::half_t> && is_same_v<ComputeTypeB, ck::half_t>)
            {
                add_device_grouped_conv3d_bwd_weight_wmma_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances(
                    op_ptrs);
            }
#endif
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                         is_same_v<WeiDataType, ck::bhalf_t> &&
                         is_same_v<OutDataType, ck::bhalf_t> &&
                         is_same_v<ComputeTypeA, ck::bhalf_t> &&
                         is_same_v<ComputeTypeB, ck::bhalf_t>)
            {
                add_device_grouped_conv3d_bwd_weight_wmma_scale_ndhwgc_gkzyxc_ndhwgk_bf16_bf16_bf16_instances(
                    op_ptrs);
            }
#endif
        }
#endif

        return op_ptrs;
    }
};

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
