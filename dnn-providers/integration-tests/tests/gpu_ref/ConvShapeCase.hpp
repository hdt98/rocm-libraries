// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

// ============================================================================
// ConvShapeCase — unified shape parameters for parameterized convolution tests.
// Shared across forward, dgrad, and wgrad directions.
// ============================================================================

namespace gpu_conv_ref_test
{

using hipdnn_data_sdk::utilities::TensorLayout;

struct ConvShapeCase
{
    std::vector<int64_t> xDims;
    std::vector<int64_t> wDims;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    std::vector<int64_t> padding;
    int64_t groups = 1;
    std::string tag;

    // When non-null, input/output tensors use this channel-last layout (NHWC/NDHWC/NLC).
    // Weights always use default packed (KCRS) strides regardless.
    // When null, all tensors use default packed strides (NCHW/NCDHW/NCL).
    const TensorLayout* layout = nullptr;

    // Computes the forward output dimensions (= dy dims for dgrad/wgrad).
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

    friend std::ostream& operator<<(std::ostream& os, const ConvShapeCase& tc)
    {
        return os << tc.tag;
    }
};

// Name generator for parameterized tests — extracts the tag from ConvShapeCase.
// Usage: INSTANTIATE_TEST_SUITE_P(Smoke, Suite, ::testing::ValuesIn(cases), byTag());
inline auto byTag()
{
    return [](const auto& info) { return info.param.tag; };
}

} // namespace gpu_conv_ref_test
