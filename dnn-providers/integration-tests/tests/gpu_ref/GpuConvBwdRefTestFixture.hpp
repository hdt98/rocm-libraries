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

// Returns copies of the given cases with channel-last layout set.
inline std::vector<ConvBwdShapeCase> withChannelLastLayout(std::vector<ConvBwdShapeCase> cases)
{
    for(auto& tc : cases)
    {
        if(tc.xDims.size() == 5)
        {
            tc.layout = &TensorLayout::NDHWC;
        }
        else if(tc.xDims.size() == 4)
        {
            tc.layout = &TensorLayout::NHWC;
        }
        else
        {
            tc.layout = &TensorLayout::NLC;
        }
    }
    return cases;
}

// ============================================================================
// Shape Catalog — centralized convolution shapes for dgrad tests.
// Each function is kept separate for easy future splitting into tiers.
// ============================================================================

// Small 2D shapes: output < 1K elements, suitable for all types
inline std::vector<ConvBwdShapeCase> getSmall2dDgradCases()
{
    return {
        {{1, 1, 8, 8}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 1, "Basic3x3"},
        {{1, 3, 8, 8}, {6, 3, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, "MultiChanPad"},
        {{2, 4, 8, 8}, {4, 2, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 2, "Grouped2Batch2"},
        {{1, 1, 8, 8}, {2, 1, 3, 3}, {2, 2}, {1, 1}, {0, 0}, 1, "Stride2"},
        {{1, 1, 12, 12}, {1, 1, 3, 3}, {1, 1}, {2, 2}, {0, 0}, 1, "Dilation2"},
        {{1, 3, 8, 8}, {3, 1, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 3, "Depthwise3Chan"},
        {{1, 8, 4, 4}, {16, 8, 1, 1}, {1, 1}, {1, 1}, {0, 0}, 1, "Pointwise1x1"},
        {{1, 7, 8, 8}, {7, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 7, "DepthwiseOdd7"},
        {{1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 1, "SingleElement"},
    };
}

// Medium 2D shapes: ResNet/ResNeXt-like, suitable for fp32 + fp16
inline std::vector<ConvBwdShapeCase> getMedium2dDgradCases()
{
    return {
        {{8, 64, 28, 28}, {128, 32, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 2, "ResNeXt2Group"},
        {{8, 128, 14, 14}, {256, 4, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, "ResNeXt32x4d"},
        {{4, 64, 56, 56}, {64, 64, 1, 1}, {1, 1}, {1, 1}, {0, 0}, 1, "ResNet1x1Reduce"},
        {{8, 3, 28, 28}, {64, 3, 7, 7}, {2, 2}, {1, 1}, {3, 3}, 1, "ResNetStem7x7"},
        {{8, 64, 14, 14}, {64, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 8, "Grouped8"},
        {{4, 16, 48, 48}, {16, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 16, "MobileNetDW16"},
        {{8, 3, 108, 108}, {63, 1, 3, 3}, {2, 2}, {1, 1}, {1, 1}, 3, "RGB3GroupStride2"},
        {{4, 32, 28, 28}, {32, 16, 5, 5}, {1, 1}, {1, 1}, {2, 2}, 2, "Grouped2Kernel5x5"},
        {{8, 128, 28, 28}, {128, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 8, "Grouped8MidRes"},
        {{2, 256, 14, 14}, {256, 256, 1, 1}, {1, 1}, {1, 1}, {0, 0}, 1, "Bottleneck1x1Expand"},
        {{4, 4, 48, 48}, {16, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 4, "Depthwise4Chan"},
        {{8, 7, 14, 14}, {63, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 7, "OddChanGrouped7"},
    };
}

// Large 2D shapes: stress tests, fp32 only
inline std::vector<ConvBwdShapeCase> getLarge2dDgradCases()
{
    return {
        {{16, 128, 56, 56}, {256, 4, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, "ResNeXt32x4dHiRes"},
        {{16, 512, 14, 14}, {1024, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, "ResNeXtDeep32Group"},
        {{16, 256, 28, 28}, {512, 8, 3, 3}, {2, 2}, {1, 1}, {1, 1}, 32, "ResNeXtStride2Down"},
        {{16, 3, 224, 224}, {63, 1, 7, 7}, {2, 2}, {1, 1}, {3, 3}, 3, "LargeStem7x7"},
        {{8, 128, 56, 56}, {128, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 8, "MidRes8Group56x56"},
        {{16, 192, 28, 28}, {32, 12, 5, 5}, {1, 1}, {1, 1}, {2, 2}, 16, "Inception5x5x16Group"},
    };
}

// Small 1D shapes
inline std::vector<ConvBwdShapeCase> getSmall1dDgradCases()
{
    return {
        // TODO - ALMIOPEN-1854: Bring these back
        // {{1, 1, 8}, {1, 1, 3}, {1}, {1}, {0}, 1, "Basic1d"},
        // {{1, 1, 6}, {1, 1, 3}, {1}, {1}, {1}, 1, "Padded1d"},
        // {{1, 1, 10}, {1, 1, 3}, {2}, {1}, {0}, 1, "Stride2x1d"},
        // {{1, 1, 9}, {1, 1, 3}, {1}, {2}, {0}, 1, "Dilation2x1d"},
        // {{1, 3, 8}, {2, 3, 3}, {1}, {1}, {0}, 1, "MultiChan1d"},
        // {{2, 1, 8}, {1, 1, 3}, {1}, {1}, {0}, 1, "MultiBatch1d"},
        // {{1, 4, 8}, {4, 2, 3}, {1}, {1}, {0}, 2, "Grouped2x1d"},
        // {{1, 3, 8}, {2, 3, 1}, {1}, {1}, {0}, 1, "Pointwise1d"},
    };
}

// Medium 1D shapes
inline std::vector<ConvBwdShapeCase> getMedium1dDgradCases()
{
    return {
        {{8, 64, 128}, {128, 64, 3}, {1}, {1}, {1}, 1, "WaveNet64Ch"},
        {{4, 32, 256}, {32, 32, 5}, {1}, {1}, {2}, 1, "Kernel5Pad2"},
        {{8, 128, 64}, {128, 16, 3}, {1}, {1}, {1}, 8, "Grouped8x1d"},
        {{4, 16, 512}, {16, 16, 7}, {2}, {1}, {3}, 1, "Stride2Kernel7"},
        {{8, 32, 128}, {32, 1, 3}, {1}, {1}, {1}, 32, "Depthwise32x1d"},
        {{4, 64, 64}, {128, 64, 1}, {1}, {1}, {0}, 1, "Pointwise64Ch"},
    };
}

// Small 3D shapes
inline std::vector<ConvBwdShapeCase> getSmall3dDgradCases()
{
    return {
        {{1, 1, 4, 4, 4}, {1, 1, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 1, "Basic3d"},
        {{1, 1, 6, 6, 6}, {1, 1, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, "Padded3d"},
        {{2, 4, 4, 4, 4}, {8, 2, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 2, "Grouped2x3d"},
        {{1, 1, 5, 5, 5}, {1, 1, 3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, 1, "Stride2x3d"},
        {{1, 1, 7, 7, 7}, {1, 1, 3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {0, 0, 0}, 1, "Dilation2x3d"},
        {{1, 3, 4, 4, 4}, {2, 3, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 1, "MultiChan3d"},
    };
}

// Medium 3D shapes
inline std::vector<ConvBwdShapeCase> getMedium3dDgradCases()
{
    return {
        {{2, 16, 8, 8, 8}, {32, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, "Standard16Ch3d"},
        {{1, 16, 4, 14, 14}, {16, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, "NonCube3d"},
        {{2, 16, 8, 8, 8}, {32, 16, 5, 5, 5}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 1, "Kernel5x5x5"},
    };
}

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
