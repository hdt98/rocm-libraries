// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>

#include "device_operation_instance_factory.hpp"

#ifdef CK_USE_XDL
#include "grouped_convolution_forward_scale_xdl.inc"
#endif

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

template <ck::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename DLayouts,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename DDataTypes,
          typename OutDataType,
          typename ComputeType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    NumDimSpatial,
    InLayout,
    WeiLayout,
    DLayouts,
    OutLayout,
    InDataType,
    WeiDataType,
    DDataTypes,
    OutDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::Scale,
    ComputeType>>
{
    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
        NumDimSpatial,
        InLayout,
        WeiLayout,
        DLayouts,
        OutLayout,
        InDataType,
        WeiDataType,
        DDataTypes,
        OutDataType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::Scale,
        ComputeType>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, NDHWGC> &&
                     is_same_v<WeiLayout, GKZYXC> && is_same_v<OutLayout, NDHWGK> &&
                     DLayouts::Size() == 0)
        {
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float>)
            {
#ifdef CK_ENABLE_TF32
                if constexpr(is_same_v<ComputeType, TF32>)
                {
                    add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<ComputeType, float>)
                {
                    add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_instances(
                        op_ptrs);
                }
#endif
            }
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, ck::half_t> && is_same_v<WeiDataType, ck::half_t> &&
                         is_same_v<OutDataType, ck::half_t> && is_same_v<ComputeType, ck::half_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                         is_same_v<WeiDataType, ck::bhalf_t> && is_same_v<OutDataType, ck::bhalf_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
                    op_ptrs);
            }
#endif
#ifdef CK_ENABLE_INT8
            if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                         is_same_v<OutDataType, int8_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_int8_instances(
                    op_ptrs);
            }
#endif
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
