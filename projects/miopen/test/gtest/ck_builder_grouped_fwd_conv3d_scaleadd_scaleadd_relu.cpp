// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_builder_shared.hpp"

#include <miopen/ck_builder/factories/grouped_convolution_forward_scaleadd_scaleadd_relu.hpp>

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>

#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scaleadd_scaleadd_relu.hpp"

using InLayout             = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout            = ck::tensor_layout::convolution::GKZYXC;
using OutLayout            = ck::tensor_layout::convolution::NDHWGK;
using G_K                  = ck::tensor_layout::convolution::G_K;
using PassThrough          = ck::tensor_operation::element_wise::PassThrough;
using ScaleAddScaleAddRelu = ck::tensor_operation::element_wise::ScaleAddScaleAddRelu;
static constexpr ck::index_t NumDimSpatial = 3;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGFwdSAR =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  ck::Tuple<OutLayout, G_K>,
                                                                  OutLayout,
                                                                  DataType,
                                                                  DataType,
                                                                  ck::Tuple<DataType, DataType>,
                                                                  DataType,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  ScaleAddScaleAddRelu,
                                                                  ComputeType>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGFwdSARPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGFwdSAR<DataType, ComputeType>>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGFwdSARBuilderPtrs =
    miopen::conv::ck_builder::instance::DeviceOperationInstanceFactory<
        DeviceOpGFwdSAR<DataType, ComputeType>>;

template <typename DataType>
void CompareInstanceLists()
{
    auto ckFactoryInstances      = DeviceOpGFwdSARPtrs<DataType>::GetInstances();
    auto builderFactoryInstances = DeviceOpGFwdSARBuilderPtrs<DataType>::GetInstances();

    compare_instance_vectors(ckFactoryInstances, builderFactoryInstances);
}

TEST(CPU_CKBuilderGroupedFwdConv3DScaleAddScaleAddRelu_FP32, CompareInstanceListsFloat)
{
    CompareInstanceLists<float>();
}

TEST(CPU_CKBuilderGroupedFwdConv3DScaleAddScaleAddRelu_FP16, CompareInstanceListsHalf)
{
    CompareInstanceLists<ck::half_t>();
}

TEST(CPU_CKBuilderGroupedFwdConv3DScaleAddScaleAddRelu_BFP16, CompareInstanceListsBHalf)
{
    CompareInstanceLists<ck::bhalf_t>();
}

TEST(CPU_CKBuilderGroupedFwdConv3DScaleAddScaleAddRelu_I8, CompareInstanceListsInt8)
{
    CompareInstanceLists<int8_t>();
}
