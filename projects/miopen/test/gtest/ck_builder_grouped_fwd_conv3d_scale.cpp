// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_builder_shared.hpp"

#include <miopen/ck_builder/factories/grouped_convolution_forward_scale.hpp>

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>

#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp"

using InLayout                             = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout                            = ck::tensor_layout::convolution::GKZYXC;
using OutLayout                            = ck::tensor_layout::convolution::NDHWGK;
using PassThrough                          = ck::tensor_operation::element_wise::PassThrough;
using Scale                                = ck::tensor_operation::element_wise::Scale;
static constexpr ck::index_t NumDimSpatial = 3;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGFwdScale =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  ck::Tuple<>,
                                                                  OutLayout,
                                                                  DataType,
                                                                  DataType,
                                                                  ck::Tuple<>,
                                                                  DataType,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  Scale>;
template <typename DataType, typename ComputeType = DataType>
using DeviceOpGFwdScalePtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGFwdScale<DataType, ComputeType>>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGFwdScaleBuilderPtrs =
    miopen::conv::ck_builder::instance::DeviceOperationInstanceFactory<
        DeviceOpGFwdScale<DataType, ComputeType>>;

template <typename DataType>
void CompareInstanceLists()
{
    auto ckFactoryInstances      = DeviceOpGFwdScalePtrs<DataType>::GetInstances();
    auto builderFactoryInstances = DeviceOpGFwdScaleBuilderPtrs<DataType>::GetInstances();

    compare_instance_vectors(ckFactoryInstances, builderFactoryInstances);
}

TEST(CPU_CKBuilderGroupedFwdConv3DScale_FP32, CompareInstanceListsFloat)
{
    CompareInstanceLists<float>();
}

TEST(CPU_CKBuilderGroupedFwdConv3DScale_FP16, CompareInstanceListsHalf)
{
    CompareInstanceLists<ck::half_t>();
}

TEST(CPU_CKBuilderGroupedFwdConv3DScale_BFP16, CompareInstanceListsBHalf)
{
    CompareInstanceLists<ck::bhalf_t>();
}

TEST(CPU_CKBuilderGroupedFwdConv3DScale_I8, CompareInstanceListsInt8)
{
    CompareInstanceLists<int8_t>();
}
