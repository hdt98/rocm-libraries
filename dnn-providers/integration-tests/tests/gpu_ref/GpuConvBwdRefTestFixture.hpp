// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn_gpu_ref/GpuFpReferenceConvolution.hpp>

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

// ============================================================================
// Shared infrastructure for backward-data (dgrad) GPU-vs-CPU reference tests.
// Included by both the fast (unit) and slow (integration) test binaries.
// ============================================================================

namespace gpu_conv_bwd_ref_test
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_gpu_ref;

// Validates that two tensors are element-wise close using the standard allClose validator.
template <typename T>
void assertAllClose(TensorBase<T>& expected, TensorBase<T>& actual, float tolerance)
{
    auto validator = CpuFpReferenceValidation<T>(tolerance, 0.0f);
    ASSERT_TRUE(validator.allClose(expected, actual));
}

// Core helper: fills tensors, runs GPU and CPU dgrad, compares results.
template <typename DxDataType, typename WDataType, typename DyDataType, typename ComputeDataType>
void compareGpuVsCpuConvBwd(Tensor<DxDataType>& dxCpu,
                            Tensor<DxDataType>& dxGpu,
                            Tensor<WDataType>& wTensor,
                            Tensor<DyDataType>& dyTensor,
                            const std::vector<int64_t>& strides,
                            const std::vector<int64_t>& dilations,
                            const std::vector<int64_t>& prePadding,
                            const std::vector<int64_t>& postPadding,
                            float tolerance,
                            float fillRange)
{
    const unsigned int seed = 42;
    dyTensor.fillWithRandomValues(
        static_cast<DyDataType>(-fillRange), static_cast<DyDataType>(fillRange), seed);
    wTensor.fillWithRandomValues(
        static_cast<WDataType>(-fillRange), static_cast<WDataType>(fillRange), seed + 1);

    CpuFpReferenceConvolution::dgrad<DxDataType, WDataType, DyDataType, ComputeDataType>(
        dxCpu, wTensor, dyTensor, strides, dilations, prePadding, postPadding);

    GpuFpReferenceConvolution::dgrad<DxDataType, WDataType, DyDataType, ComputeDataType>(
        dxGpu, wTensor, dyTensor, strides, dilations, prePadding, postPadding);

    assertAllClose(dxCpu, dxGpu, tolerance);
}

// Asymmetric padding with optional layout.
template <typename DataType, typename ComputeDataType = double>
void runGpuVsCpuConvBwd(const std::vector<int64_t>& xDims,
                        const std::vector<int64_t>& wDims,
                        const std::vector<int64_t>& yDims,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& dilations,
                        const std::vector<int64_t>& prePadding,
                        const std::vector<int64_t>& postPadding,
                        float tolerance,
                        const TensorLayout* layout = nullptr,
                        float fillRange = 1.0f)
{
    auto dxCpu = layout != nullptr ? Tensor<DataType>(xDims, *layout) : Tensor<DataType>(xDims);
    auto dxGpu = layout != nullptr ? Tensor<DataType>(xDims, *layout) : Tensor<DataType>(xDims);
    auto wTensor = Tensor<DataType>(wDims);
    auto dyTensor = layout != nullptr ? Tensor<DataType>(yDims, *layout) : Tensor<DataType>(yDims);

    compareGpuVsCpuConvBwd<DataType, DataType, DataType, ComputeDataType>(dxCpu,
                                                                          dxGpu,
                                                                          wTensor,
                                                                          dyTensor,
                                                                          strides,
                                                                          dilations,
                                                                          prePadding,
                                                                          postPadding,
                                                                          tolerance,
                                                                          fillRange);
}

// ============================================================================
// ConvBwdShapeCase — shape parameters for parameterized dgrad tests
// ============================================================================

struct ConvBwdShapeCase
{
    std::vector<int64_t> xDims;
    std::vector<int64_t> wDims;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    std::vector<int64_t> padding;
    int64_t groups = 1;
    std::string tag;

    const TensorLayout* layout = nullptr;

    std::vector<int64_t> computeOutputDims() const
    {
        auto numSpatialDims = xDims.size() - 2;
        std::vector<int64_t> yDims = {xDims[0], wDims[0]};
        for(size_t i = 0; i < numSpatialDims; ++i)
        {
            auto outputSize
                = (xDims[2 + i] + 2 * padding[i] - dilations[i] * (wDims[2 + i] - 1) - 1)
                      / strides[i]
                  + 1;
            yDims.push_back(outputSize);
        }
        return yDims;
    }

    friend std::ostream& operator<<(std::ostream& os, const ConvBwdShapeCase& tc)
    {
        return os << tc.tag;
    }
};

// ============================================================================
// ConvBwdShapeSuite — parameterized fixture for shape-based GPU-vs-CPU tests
// ============================================================================

template <typename DataType>
class ConvBwdShapeSuite : public ::testing::TestWithParam<ConvBwdShapeCase>
{
protected:
    static float tolerance(const ConvBwdShapeCase& tc)
    {
        constexpr double FILL_RANGE = 1.0;
        return hipdnn_test_sdk::utilities::conv::
            calculateConvDgradTolerance<DataType, DataType, double>(
                -FILL_RANGE, FILL_RANGE, -FILL_RANGE, FILL_RANGE, tc.wDims);
    }

    void runConvBwdShapeTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        auto yDims = tc.computeOutputDims();
        runGpuVsCpuConvBwd<DataType>(tc.xDims,
                                     tc.wDims,
                                     yDims,
                                     tc.strides,
                                     tc.dilations,
                                     tc.padding,
                                     tc.padding,
                                     tolerance(tc),
                                     tc.layout);
    }
};

using TestGpuConvBwdRefShapesFp32 = ConvBwdShapeSuite<float>;
using TestGpuConvBwdRefShapesFp16 = ConvBwdShapeSuite<half>;
using TestGpuConvBwdRefShapesBfp16 = ConvBwdShapeSuite<bfloat16>;

} // namespace gpu_conv_bwd_ref_test
