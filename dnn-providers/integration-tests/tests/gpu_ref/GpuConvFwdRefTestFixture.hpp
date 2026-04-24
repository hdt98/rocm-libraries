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
// Shared infrastructure for forward convolution GPU-vs-CPU reference tests.
// Included by both the fast (unit) and slow (integration) test binaries.
// ============================================================================

namespace gpu_conv_fwd_ref_test
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_gpu_ref;

// Validates that two tensors are element-wise close using the standard allClose validator.
// Handles NaN/Inf detection, stride-aware indexing, and parallel comparison.
template <typename T>
void assertAllClose(TensorBase<T>& expected, TensorBase<T>& actual, float tolerance)
{
    auto validator = CpuFpReferenceValidation<T>(tolerance, 0.0f);
    ASSERT_TRUE(validator.allClose(expected, actual));
}

// Core helper: fills tensors, runs GPU and CPU convolution, compares results.
// Separate template params support mixed input/weight types (WDataType != XDataType).
template <typename XDataType, typename WDataType, typename YDataType, typename ComputeDataType>
void compareGpuVsCpuConvFwd(Tensor<XDataType>& xTensor,
                            Tensor<WDataType>& wTensor,
                            Tensor<YDataType>& yCpu,
                            Tensor<YDataType>& yGpu,
                            const std::vector<int64_t>& strides,
                            const std::vector<int64_t>& dilations,
                            const std::vector<int64_t>& prePadding,
                            const std::vector<int64_t>& postPadding,
                            float tolerance,
                            float fillRange)
{
    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(
        static_cast<XDataType>(-fillRange), static_cast<XDataType>(fillRange), seed);
    wTensor.fillWithRandomValues(
        static_cast<WDataType>(-fillRange), static_cast<WDataType>(fillRange), seed + 1);

    CpuFpReferenceConvolution::fprop<XDataType, WDataType, YDataType, ComputeDataType>(
        xTensor, wTensor, yCpu, strides, dilations, prePadding, postPadding);

    GpuFpReferenceConvolution::fprop<XDataType, WDataType, YDataType, ComputeDataType>(
        xTensor, wTensor, yGpu, strides, dilations, prePadding, postPadding);

    assertAllClose(yCpu, yGpu, tolerance);
}

// --- Forward convolution helper overloads ---
// fillRange controls the magnitude of random fill values [-fillRange, +fillRange].
// For small output types (e.g. fp8), reduce fillRange to prevent overflow:
// each output element accumulates cPerGroup * Kh * Kw products, so
// max output ~ numMACs * fillRange^2. Keep numMACs * fillRange^2 < type max.

// Asymmetric padding with optional layout.
// When layout is non-null, input/output tensors use channel-last strides (e.g. NHWC, NDHWC)
// generated via Tensor(dims, layout). Weights always use default packed (KCRS) strides.
// When layout is null, all tensors use default packed strides (NCHW/NCDHW).
template <typename DataType, typename ComputeDataType = double>
void runGpuVsCpuConvFwd(const std::vector<int64_t>& xDims,
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
    auto xTensor = layout != nullptr ? Tensor<DataType>(xDims, *layout) : Tensor<DataType>(xDims);
    auto wTensor = Tensor<DataType>(wDims);
    auto yCpu = layout != nullptr ? Tensor<DataType>(yDims, *layout) : Tensor<DataType>(yDims);
    auto yGpu = layout != nullptr ? Tensor<DataType>(yDims, *layout) : Tensor<DataType>(yDims);

    compareGpuVsCpuConvFwd<DataType, DataType, DataType, ComputeDataType>(xTensor,
                                                                          wTensor,
                                                                          yCpu,
                                                                          yGpu,
                                                                          strides,
                                                                          dilations,
                                                                          prePadding,
                                                                          postPadding,
                                                                          tolerance,
                                                                          fillRange);
}

// ============================================================================
// ConvFwdShapeCase — shape parameters for parameterized convolution tests
// ============================================================================

struct ConvFwdShapeCase
{
    std::vector<int64_t> xDims;
    std::vector<int64_t> wDims;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    std::vector<int64_t> padding;
    int64_t groups = 1;
    std::string tag;

    // When non-null, input/output tensors use this channel-last layout (NHWC/NDHWC).
    // Weights always use default packed (KCRS) strides regardless.
    // When null, all tensors use default packed strides (NCHW/NCDHW).
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

    friend std::ostream& operator<<(std::ostream& os, const ConvFwdShapeCase& tc)
    {
        return os << tc.tag;
    }
};

// Returns copies of the given cases with channel-last layout set.
// Uses NHWC for 4D (2D conv) and NDHWC for 5D (3D conv).
// Points to the static TensorLayout constants which have program lifetime.
inline std::vector<ConvFwdShapeCase> withChannelLastLayout(std::vector<ConvFwdShapeCase> cases)
{
    for(auto& tc : cases)
    {
        tc.layout = (tc.xDims.size() == 5) ? &TensorLayout::NDHWC : &TensorLayout::NHWC;
    }
    return cases;
}

// ============================================================================
// Shape Catalog — centralized convolution shapes, categorized by size.
// Each function is kept separate for easy future splitting into tiers.
// ============================================================================

// Small 2D shapes: output < 1K elements, suitable for all types
inline std::vector<ConvFwdShapeCase> getSmall2dConvCases()
{
    return {
        // Basic single-channel 3x3 convolution, no padding
        {{1, 1, 8, 8}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 1, "Basic3x3"},
        // Multiple input/output channels with padding
        {{1, 3, 8, 8}, {6, 3, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, "MultiChanPad"},
        // 2-group convolution with multi-batch
        {{2, 4, 8, 8}, {4, 2, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 2, "Grouped2Batch2"},
        // Stride=2 downsampling
        {{1, 1, 8, 8}, {2, 1, 3, 3}, {2, 2}, {1, 1}, {0, 0}, 1, "Stride2"},
        // Dilation=2 (expanded receptive field)
        {{1, 1, 12, 12}, {1, 1, 3, 3}, {1, 1}, {2, 2}, {0, 0}, 1, "Dilation2"},
        // Depthwise convolution (groups == input channels)
        {{1, 3, 8, 8}, {3, 1, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 3, "Depthwise3Chan"},
        // 1x1 pointwise convolution (channel mixing only)
        {{1, 8, 4, 4}, {16, 8, 1, 1}, {1, 1}, {1, 1}, {0, 0}, 1, "Pointwise1x1"},
        // Depthwise with odd group count
        {{1, 7, 8, 8}, {7, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 7, "DepthwiseOdd7"},
        // Minimum output: single element (3x3 input, 3x3 kernel)
        {{1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 1, "SingleElement"},
    };
}

// Medium 2D shapes: ResNet/ResNeXt/Inception-like, suitable for fp32 + fp16
inline std::vector<ConvFwdShapeCase> getMedium2dConvCases()
{
    return {
        // ResNeXt-like 2-group block
        {{8, 64, 28, 28}, {128, 32, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 2, "ResNeXt2Group"},
        // ResNeXt-32x4d bottleneck (32 groups, 4 channels/group)
        {{8, 128, 14, 14}, {256, 4, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, "ResNeXt32x4d"},
        // ResNet 1x1 pointwise reduction
        {{4, 64, 56, 56}, {64, 64, 1, 1}, {1, 1}, {1, 1}, {0, 0}, 1, "ResNet1x1Reduce"},
        // ResNet stem layer: 7x7 kernel, stride=2
        {{8, 3, 28, 28}, {64, 3, 7, 7}, {2, 2}, {1, 1}, {3, 3}, 1, "ResNetStem7x7"},
        // 8-group convolution
        {{8, 64, 14, 14}, {64, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 8, "Grouped8"},
        // MobileNet-style depthwise (16 channels)
        {{4, 16, 48, 48}, {16, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 16, "MobileNetDW16"},
        // RGB 3-group with stride-2 downsampling
        {{8, 3, 108, 108}, {63, 1, 3, 3}, {2, 2}, {1, 1}, {1, 1}, 3, "RGB3GroupStride2"},
        // 2-group with 5x5 kernel
        {{4, 32, 28, 28}, {32, 16, 5, 5}, {1, 1}, {1, 1}, {2, 2}, 2, "Grouped2Kernel5x5"},
        // 8-group mid-resolution
        {{8, 128, 28, 28}, {128, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 8, "Grouped8MidRes"},
        // Bottleneck 1x1 channel expansion
        {{2, 256, 14, 14}, {256, 256, 1, 1}, {1, 1}, {1, 1}, {0, 0}, 1, "Bottleneck1x1Expand"},
        // Small depthwise (4 channels)
        {{4, 4, 48, 48}, {16, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 4, "Depthwise4Chan"},
        // Odd channel count grouped (7 groups)
        {{8, 7, 14, 14}, {63, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 7, "OddChanGrouped7"},
    };
}

// Large 2D shapes: stress tests matching real workloads, fp32 only
inline std::vector<ConvFwdShapeCase> getLarge2dConvCases()
{
    return {
        // ResNeXt-32x4d high-resolution block
        {{16, 128, 56, 56}, {256, 4, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, "ResNeXt32x4dHiRes"},
        // ResNeXt deep 32-group (512->1024 channels)
        {{16, 512, 14, 14}, {1024, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, "ResNeXtDeep32Group"},
        // ResNeXt stride-2 downsample (256->512)
        {{16, 256, 28, 28}, {512, 8, 3, 3}, {2, 2}, {1, 1}, {1, 1}, 32, "ResNeXtStride2Down"},
        // Large stem: 3-group 7x7 on 224x224 input
        {{16, 3, 224, 224}, {63, 1, 7, 7}, {2, 2}, {1, 1}, {3, 3}, 3, "LargeStem7x7"},
        // Mid-resolution 8-group on 56x56
        {{8, 128, 56, 56}, {128, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 8, "MidRes8Group56x56"},
        // Inception-like 5x5 kernel, 16-group
        {{16, 192, 28, 28}, {32, 12, 5, 5}, {1, 1}, {1, 1}, {2, 2}, 16, "Inception5x5x16Group"},
        // DeepSpeech-like non-square spatial (161x700)
        {{4, 4, 161, 700}, {32, 1, 5, 20}, {2, 2}, {1, 1}, {0, 0}, 4, "DeepSpeechNonSquare"},
        // Non-square spatial with 2-group (79x341)
        {{8, 32, 79, 341}, {32, 16, 5, 10}, {2, 2}, {1, 1}, {0, 0}, 2, "NonSquareGrouped2"},
    };
}

// Small 1D shapes: basic NCW convolution tests
inline std::vector<ConvFwdShapeCase> getSmall1dConvCases()
{
    return {
        // Basic 1D: single-channel, kernel=3
        {{1, 1, 8}, {1, 1, 3}, {1}, {1}, {0}, 1, "Basic1d"},
        // 1D with padding
        {{1, 1, 6}, {1, 1, 3}, {1}, {1}, {1}, 1, "Padded1d"},
        // 1D with stride=2
        {{1, 1, 10}, {1, 1, 3}, {2}, {1}, {0}, 1, "Stride2x1d"},
        // 1D with dilation=2
        {{1, 1, 9}, {1, 1, 3}, {1}, {2}, {0}, 1, "Dilation2x1d"},
        // 1D multi-channel (3 in, 2 out)
        {{1, 3, 8}, {2, 3, 3}, {1}, {1}, {0}, 1, "MultiChan1d"},
        // 1D multi-batch
        {{2, 1, 8}, {1, 1, 3}, {1}, {1}, {0}, 1, "MultiBatch1d"},
        // 1D grouped (2 groups)
        {{1, 4, 8}, {4, 2, 3}, {1}, {1}, {0}, 2, "Grouped2x1d"},
        // 1D pointwise (kernel=1)
        {{1, 3, 8}, {2, 3, 1}, {1}, {1}, {0}, 1, "Pointwise1d"},
    };
}

// Small 3D shapes: basic 3D convolution tests
inline std::vector<ConvFwdShapeCase> getSmall3dConvCases()
{
    return {
        // Basic 3D: single-channel 3x3x3
        {{1, 1, 4, 4, 4}, {1, 1, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 1, "Basic3d"},
        // 3D with padding
        {{1, 1, 6, 6, 6}, {1, 1, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, "Padded3d"},
        // 3D grouped (2 groups)
        {{2, 4, 4, 4, 4}, {8, 2, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 2, "Grouped2x3d"},
        // 3D with stride=2
        {{1, 1, 5, 5, 5}, {1, 1, 3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, 1, "Stride2x3d"},
        // 3D with dilation=2
        {{1, 1, 7, 7, 7}, {1, 1, 3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {0, 0, 0}, 1, "Dilation2x3d"},
        // 3D multi-channel (3 in, 2 out)
        {{1, 3, 4, 4, 4}, {2, 3, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 1, "MultiChan3d"},
    };
}

// Medium 3D shapes: larger 3D convolutions
inline std::vector<ConvFwdShapeCase> getMedium3dConvCases()
{
    return {
        // Standard 3D with 16 input channels and padding
        {{2, 16, 8, 8, 8}, {32, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, "Standard16Ch3d"},
        // Non-cube spatial dimensions (4x14x14)
        {{1, 16, 4, 14, 14}, {16, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, "NonCube3d"},
        // Large 5x5x5 kernel
        {{2, 16, 8, 8, 8}, {32, 16, 5, 5, 5}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 1, "Kernel5x5x5"},
    };
}

// ============================================================================
// ConvFwdShapeSuite — parameterized fixture for shape-based GPU-vs-CPU tests
// ============================================================================

template <typename DataType>
class ConvFwdShapeSuite : public ::testing::TestWithParam<ConvFwdShapeCase>
{
protected:
    static float tolerance(const ConvFwdShapeCase& tc)
    {
        constexpr double FILL_RANGE = 1.0;
        return hipdnn_test_sdk::utilities::conv::
            calculateConvFpropTolerance<DataType, DataType, double>(
                -FILL_RANGE, FILL_RANGE, -FILL_RANGE, FILL_RANGE, tc.wDims);
    }

    void runConvFwdShapeTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        auto yDims = tc.computeOutputDims();
        runGpuVsCpuConvFwd<DataType>(tc.xDims,
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

using TestGpuConvFwdRefShapesFp32 = ConvFwdShapeSuite<float>;
using TestGpuConvFwdRefShapesFp16 = ConvFwdShapeSuite<half>;
using TestGpuConvFwdRefShapesBfp16 = ConvFwdShapeSuite<bfloat16>;

} // namespace gpu_conv_fwd_ref_test
