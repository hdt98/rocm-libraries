// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <miopen/ck_builder/generate_kernels.hpp>

namespace ckb                              = ck_tile::builder;
using BaseOperator                         = ck::tensor_operation::device::BaseOperator;
using InLayout                             = ck::tensor_layout::convolution::NGCHW;
using WeiLayout                            = ck::tensor_layout::convolution::GKCYX;
using OutLayout                            = ck::tensor_layout::convolution::NGKHW;
using PassThrough                          = ck::tensor_operation::element_wise::PassThrough;
using EmptyTuple                           = ck::Tuple<>;
static constexpr ck::index_t NumDimSpatial = 2;

template <typename DataType>
using DeviceOpGFwdDefault =
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
                                                                  PassThrough>;
template <typename DataType>
using CK_DeviceOpGFwdDefaultPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGFwdDefault<DataType>>;

template <typename DataType>
using MIOpen_DeviceOpGFwdDefaultPtrs =
    miopen::kernels::ck_builder::DeviceOperationInstanceFactory<DeviceOpGFwdDefault<DataType>>;

TEST(CK_Builder, Factory_Kernel_Diff)
{
    auto ckKernelInstances     = CK_DeviceOpGFwdDefaultPtrs<float>::GetInstances();
    auto miopenKernelInstances = MIOpen_DeviceOpGFwdDefaultPtrs<float>::GetInstances();

    std::cout << ckKernelInstances.size() << " " << miopenKernelInstances.size() << std::endl;
}
