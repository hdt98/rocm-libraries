// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/gpu_graph_executor/detail/GpuConvolutionFwdPlan.hpp"

namespace
{

using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

std::vector<int64_t> computePackedStrides(const std::vector<int64_t>& dims)
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

flatbuffers::FlatBufferBuilder createConvFwdGraph(int64_t xUid,
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
                                                  DataType dataType)
{
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

} // namespace

TEST(TestGpuConvolutionFwdPlanBuilder, PlanConstruction)
{
    constexpr int64_t X_UID = 10;
    constexpr int64_t W_UID = 11;
    constexpr int64_t Y_UID = 12;

    const std::vector<int64_t> xDims = {1, 1, 2, 2};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> yDims = {1, 1, 2, 2};

    auto graphBuilder = createConvFwdGraph(X_UID,
                                           W_UID,
                                           Y_UID,
                                           xDims,
                                           wDims,
                                           yDims,
                                           computePackedStrides(xDims),
                                           computePackedStrides(wDims),
                                           computePackedStrides(yDims),
                                           {0, 0},
                                           {1, 1},
                                           {1, 1},
                                           DataType::FLOAT);

    auto graphWrap = hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionFwdPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<GpuConvolutionFwdPlan<float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuConvolutionFwdPlanBuilder, IsApplicable)
{
    constexpr int64_t X_UID = 10;
    constexpr int64_t W_UID = 11;
    constexpr int64_t Y_UID = 12;

    const std::vector<int64_t> xDims = {1, 1, 2, 2};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> yDims = {1, 1, 2, 2};

    auto graphBuilder = createConvFwdGraph(X_UID,
                                           W_UID,
                                           Y_UID,
                                           xDims,
                                           wDims,
                                           yDims,
                                           computePackedStrides(xDims),
                                           computePackedStrides(wDims),
                                           computePackedStrides(yDims),
                                           {0, 0},
                                           {1, 1},
                                           {1, 1},
                                           DataType::FLOAT);

    auto graphWrap = hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionFwdPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Half builder should not be applicable for a float graph
    const GpuConvolutionFwdPlanBuilder<DataType::HALF,
                                       DataType::HALF,
                                       DataType::HALF,
                                       DataType::FLOAT>
        halfPlanBuilder;

    EXPECT_FALSE(halfPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Missing tensor should return false
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(W_UID);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

TEST(TestGpuConvolutionFwdPlan, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    constexpr int64_t X_UID = 1;
    constexpr int64_t W_UID = 2;
    constexpr int64_t Y_UID = 3;

    const std::vector<int64_t> xDims = {1, 1, 4, 4};
    const std::vector<int64_t> wDims = {1, 1, 3, 3};
    const std::vector<int64_t> yDims = {1, 1, 2, 2};

    auto xStrides = computePackedStrides(xDims);
    auto wStrides = computePackedStrides(wDims);
    auto yStrides = computePackedStrides(yDims);

    const std::vector<int64_t> stride = {1, 1};
    const std::vector<int64_t> dilation = {1, 1};
    const std::vector<int64_t> padding = {0, 0};

    auto graphBuilder = createConvFwdGraph(X_UID,
                                           W_UID,
                                           Y_UID,
                                           xDims,
                                           wDims,
                                           yDims,
                                           xStrides,
                                           wStrides,
                                           yStrides,
                                           padding,
                                           stride,
                                           dilation,
                                           DataType::FLOAT);

    auto graphWrap = hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const auto* nodeAttributes = graphWrap.getNode(0).attributes_as_ConvolutionFwdAttributes();
    const auto& tensorMap = graphWrap.getTensorMap();

    GpuConvolutionFwdParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                   *tensorMap.at(nodeAttributes->w_tensor_uid()),
                                   *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                   padding,
                                   padding,
                                   stride,
                                   dilation,
                                   ConvMode::CROSS_CORRELATION);

    GpuConvolutionFwdPlan<float, float, float, float> patient(std::move(params));

    // Compute element counts
    size_t xCount = 1;
    for(auto d : xDims)
    {
        xCount *= static_cast<size_t>(d);
    }
    size_t wCount = 1;
    for(auto d : wDims)
    {
        wCount *= static_cast<size_t>(d);
    }
    size_t yCount = 1;
    for(auto d : yDims)
    {
        yCount *= static_cast<size_t>(d);
    }

    // Allocate and fill input data
    std::vector<float> xData(xCount);
    std::vector<float> wData(wCount);
    std::vector<float> gpuYData(yCount, 0.0f);

    for(size_t i = 0; i < xCount; ++i)
    {
        xData[i] = static_cast<float>(i + 1);
    }
    for(size_t i = 0; i < wCount; ++i)
    {
        wData[i] = 1.0f;
    }

    // Execute the plan
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[X_UID] = xData.data();
    variantPack[W_UID] = wData.data();
    variantPack[Y_UID] = gpuYData.data();

    patient.execute(variantPack);

    // Run CPU reference for comparison
    hipdnn_data_sdk::utilities::Tensor<float> cpuX(xDims, xStrides);
    hipdnn_data_sdk::utilities::Tensor<float> cpuW(wDims, wStrides);
    hipdnn_data_sdk::utilities::Tensor<float> cpuY(yDims, yStrides);

    std::memcpy(cpuX.rawHostData(), xData.data(), xCount * sizeof(float));
    std::memcpy(cpuW.rawHostData(), wData.data(), wCount * sizeof(float));

    hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::fprop<float, float, float, float>(
        cpuX, cpuW, cpuY, stride, dilation, padding, padding);

    // Compare GPU plan output against CPU reference
    const auto* cpuResult = static_cast<const float*>(cpuY.rawHostData());
    for(size_t i = 0; i < yCount; ++i)
    {
        EXPECT_NEAR(gpuYData[i], cpuResult[i], 1e-5) << "Mismatch at index " << i;
    }
}
