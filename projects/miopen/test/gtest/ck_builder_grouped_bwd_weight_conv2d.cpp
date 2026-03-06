// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_builder_shared.hpp"

#include <miopen/ck_builder/factories/grouped_convolution_backward_weight.hpp>

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>

#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp"

using InLayout                             = ck::tensor_layout::convolution::NHWGC;
using WeiLayout                            = ck::tensor_layout::convolution::GKYXC;
using OutLayout                            = ck::tensor_layout::convolution::NHWGK;
using PassThrough                          = ck::tensor_operation::element_wise::PassThrough;
static constexpr ck::index_t NumDimSpatial = 2;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdWeight = ck::tensor_operation::device::DeviceGroupedConvBwdWeight<NumDimSpatial,
                                                                                    InLayout,
                                                                                    WeiLayout,
                                                                                    OutLayout,
                                                                                    DataType,
                                                                                    DataType,
                                                                                    DataType,
                                                                                    PassThrough,
                                                                                    PassThrough,
                                                                                    PassThrough,
                                                                                    ComputeType,
                                                                                    ComputeType>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdWeightCKPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdWeight<DataType, ComputeType>>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdWeightBuilderPtrs =
    miopen::conv::ck_builder::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdWeight<DataType, ComputeType>>;

namespace {
template <typename DataType>
void CompareInstanceLists()
{
    auto ckFactoryInstances      = DeviceOpGBwdWeightCKPtrs<DataType>::GetInstances();
    auto builderFactoryInstances = DeviceOpGBwdWeightBuilderPtrs<DataType>::GetInstances();

    compare_instance_vectors(ckFactoryInstances, builderFactoryInstances);
}
} // namespace

TEST(CPU_CKBuilderGroupedBwdWeightConv2D_FP32, CompareInstanceListsFloat)
{
    CompareInstanceLists<float>();
}

TEST(CPU_CKBuilderGroupedBwdWeightConv2D_FP16, CompareInstanceListsHalf)
{
    CompareInstanceLists<ck::half_t>();
}

TEST(CPU_CKBuilderGroupedBwdWeightConv2D_BFP16, CompareInstanceListsBHalf)
{
    CompareInstanceLists<ck::bhalf_t>();
}
