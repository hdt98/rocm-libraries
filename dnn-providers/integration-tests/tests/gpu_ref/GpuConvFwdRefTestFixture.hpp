// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "ConvShapeCase.hpp"
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

using gpu_conv_ref_test::ConvShapeCase;
using ConvFwdShapeCase = ConvShapeCase;

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

// One-liner subclasses — each creates a distinct GTest-visible type so that
// INSTANTIATE_TEST_SUITE_P can use clean tier-only prefixes (Smoke, Medium, Full)
// while the suite name itself carries dimensionality and layout information.

// Default layout (NCL / NCHW / NCDHW)
class TestGpuConvFwdRef1dFp32 : public ConvFwdShapeSuite<float>
{
};
class TestGpuConvFwdRef2dFp32 : public ConvFwdShapeSuite<float>
{
};
class TestGpuConvFwdRef3dFp32 : public ConvFwdShapeSuite<float>
{
};
class TestGpuConvFwdRef1dFp16 : public ConvFwdShapeSuite<half>
{
};
class TestGpuConvFwdRef2dFp16 : public ConvFwdShapeSuite<half>
{
};
class TestGpuConvFwdRef3dFp16 : public ConvFwdShapeSuite<half>
{
};
class TestGpuConvFwdRef1dBfp16 : public ConvFwdShapeSuite<bfloat16>
{
};
class TestGpuConvFwdRef2dBfp16 : public ConvFwdShapeSuite<bfloat16>
{
};
class TestGpuConvFwdRef3dBfp16 : public ConvFwdShapeSuite<bfloat16>
{
};

// Channel-last layout (NLC / NHWC / NDHWC)
class TestGpuConvFwdRefNlc1dFp32 : public ConvFwdShapeSuite<float>
{
};
class TestGpuConvFwdRefNhwc2dFp32 : public ConvFwdShapeSuite<float>
{
};
class TestGpuConvFwdRefNdhwc3dFp32 : public ConvFwdShapeSuite<float>
{
};
class TestGpuConvFwdRefNlc1dFp16 : public ConvFwdShapeSuite<half>
{
};
class TestGpuConvFwdRefNhwc2dFp16 : public ConvFwdShapeSuite<half>
{
};
class TestGpuConvFwdRefNdhwc3dFp16 : public ConvFwdShapeSuite<half>
{
};
class TestGpuConvFwdRefNlc1dBfp16 : public ConvFwdShapeSuite<bfloat16>
{
};
class TestGpuConvFwdRefNhwc2dBfp16 : public ConvFwdShapeSuite<bfloat16>
{
};
class TestGpuConvFwdRefNdhwc3dBfp16 : public ConvFwdShapeSuite<bfloat16>
{
};

} // namespace gpu_conv_fwd_ref_test
