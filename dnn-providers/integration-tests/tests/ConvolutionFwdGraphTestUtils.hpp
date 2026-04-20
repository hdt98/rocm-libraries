// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>

namespace hipdnn_integration_tests::test_utils
{

inline std::vector<int64_t> computePackedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size());
    int64_t stride = 1;
    for(auto i = static_cast<int>(dims.size()) - 1; i >= 0; --i)
    {
        strides[static_cast<size_t>(i)] = stride;
        stride *= dims[static_cast<size_t>(i)];
    }
    return strides;
}

// Creates a minimal graph with a ConvolutionFwd node.
// Uses symmetric padding (prePadding == postPadding == padding).
inline flatbuffers::FlatBufferBuilder
    createConvFwdGraph(int64_t xUid,
                       int64_t wUid,
                       int64_t yUid,
                       const std::vector<int64_t>& xDims,
                       const std::vector<int64_t>& wDims,
                       const std::vector<int64_t>& yDims,
                       const std::vector<int64_t>& xStrides,
                       const std::vector<int64_t>& wStrides,
                       const std::vector<int64_t>& yStrides,
                       const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& convStride,
                       const std::vector<int64_t>& dilation,
                       hipdnn_data_sdk::data_objects::DataType dataType)
{
    using namespace hipdnn_data_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, xUid, "x", dataType, &xStrides, &xDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, wUid, "w", dataType, &wStrides, &wDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, yUid, "y", dataType, &yStrides, &yDims));

    auto convAttrs = CreateConvolutionFwdAttributesDirect(builder,
                                                          xUid,
                                                          wUid,
                                                          yUid,
                                                          &padding,
                                                          &padding,
                                                          &convStride,
                                                          &dilation,
                                                          ConvMode::CROSS_CORRELATION);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "conv_fwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::ConvolutionFwdAttributes,
                                     convAttrs.Union()));

    auto graph = CreateGraphDirect(
        builder, "ConvFwdTestGraph", dataType, dataType, DataType::FLOAT, &tensors, &nodes);

    builder.Finish(graph);
    return builder;
}

} // namespace hipdnn_integration_tests::test_utils
