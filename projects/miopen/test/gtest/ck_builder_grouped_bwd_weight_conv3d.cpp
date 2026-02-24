// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_builder_shared.hpp"

#include <miopen/ck_builder/factories/grouped_convolution_backward_weight.hpp>

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>

#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp"

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

// ===== Default (PassThrough) =====

using InLayout3D                             = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout3D                            = ck::tensor_layout::convolution::GKZYXC;
using OutLayout3D                            = ck::tensor_layout::convolution::NDHWGK;
static constexpr ck::index_t NumDimSpatial3D = 3;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdWeight3D =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeight<NumDimSpatial3D,
                                                             InLayout3D,
                                                             WeiLayout3D,
                                                             OutLayout3D,
                                                             DataType,
                                                             DataType,
                                                             DataType,
                                                             PassThrough,
                                                             PassThrough,
                                                             PassThrough,
                                                             ComputeType,
                                                             ComputeType>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdWeight3DCKPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdWeight3D<DataType, ComputeType>>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdWeight3DBuilderPtrs =
    miopen::conv::ck_builder::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdWeight3D<DataType, ComputeType>>;

template <typename DataType>
void CompareInstanceLists()
{
    auto ckFactoryInstances      = DeviceOpGBwdWeight3DCKPtrs<DataType>::GetInstances();
    auto builderFactoryInstances = DeviceOpGBwdWeight3DBuilderPtrs<DataType>::GetInstances();

    compare_instance_vectors(ckFactoryInstances, builderFactoryInstances);
}

TEST(CPU_CKBuilderGroupedBwdWeightConv3D_Default_FP32, CompareInstanceListsFloat)
{
    CompareInstanceLists<float>();
}

TEST(CPU_CKBuilderGroupedBwdWeightConv3D_Default_FP16, CompareInstanceListsHalf)
{
    CompareInstanceLists<ck::half_t>();
}

TEST(CPU_CKBuilderGroupedBwdWeightConv3D_Default_BFP16, CompareInstanceListsBHalf)
{
    CompareInstanceLists<ck::bhalf_t>();
}
