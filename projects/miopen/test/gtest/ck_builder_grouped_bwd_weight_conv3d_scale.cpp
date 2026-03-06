// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_builder_shared.hpp"

#include <miopen/ck_builder/factories/grouped_convolution_backward_weight_scale.hpp>

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>

#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight_scale.hpp"

using InLayout                             = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout                            = ck::tensor_layout::convolution::GKZYXC;
using OutLayout                            = ck::tensor_layout::convolution::NDHWGK;
using PassThrough                          = ck::tensor_operation::element_wise::PassThrough;
using Scale                                = ck::tensor_operation::element_wise::Scale;
static constexpr ck::index_t NumDimSpatial = 3;

template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename ComputeTypeA = InDataType,
          typename ComputeTypeB = InDataType>
using DeviceOpGBwdWeightScale =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<NumDimSpatial,
                                                                      InLayout,
                                                                      WeiLayout,
                                                                      OutLayout,
                                                                      ck::Tuple<>,
                                                                      InDataType,
                                                                      WeiDataType,
                                                                      OutDataType,
                                                                      ck::Tuple<>,
                                                                      PassThrough,
                                                                      Scale,
                                                                      PassThrough,
                                                                      ComputeTypeA,
                                                                      ComputeTypeB>;

template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename ComputeTypeA = InDataType,
          typename ComputeTypeB = InDataType>
using DeviceOpGBwdWeightScaleCKPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdWeightScale<InDataType, WeiDataType, OutDataType, ComputeTypeA, ComputeTypeB>>;

template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename ComputeTypeA = InDataType,
          typename ComputeTypeB = InDataType>
using DeviceOpGBwdWeightScaleBuilderPtrs =
    miopen::conv::ck_builder::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdWeightScale<InDataType, WeiDataType, OutDataType, ComputeTypeA, ComputeTypeB>>;

namespace {
template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename ComputeTypeA = InDataType,
          typename ComputeTypeB = InDataType>
void CompareInstanceLists()
{
    auto ckFactoryInstances      = DeviceOpGBwdWeightScaleCKPtrs<InDataType,
                                                                 WeiDataType,
                                                                 OutDataType,
                                                                 ComputeTypeA,
                                                                 ComputeTypeB>::GetInstances();
    auto builderFactoryInstances = DeviceOpGBwdWeightScaleBuilderPtrs<InDataType,
                                                                      WeiDataType,
                                                                      OutDataType,
                                                                      ComputeTypeA,
                                                                      ComputeTypeB>::GetInstances();

    compare_instance_vectors(ckFactoryInstances, builderFactoryInstances);
}
} // namespace

TEST(CPU_CKBuilderGroupedBwdWeightConv3DScale_FP32, CompareInstanceListsFloat)
{
    CompareInstanceLists<float, float, float>();
}

TEST(CPU_CKBuilderGroupedBwdWeightConv3DScale_FP16, CompareInstanceListsHalf)
{
    CompareInstanceLists<ck::half_t, ck::half_t, ck::half_t>();
}

TEST(CPU_CKBuilderGroupedBwdWeightConv3DScale_BFP16, CompareInstanceListsBHalf)
{
    CompareInstanceLists<ck::bhalf_t, float, ck::bhalf_t>();
}
