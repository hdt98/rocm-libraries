// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "ConvolutionFwdGraphTestUtils.hpp"
#include "harness/gpu_graph_executor/detail/GpuConvolutionFwdPlan.hpp"

using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

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

    // Prepare host input data
    std::vector<float> xData(xCount);
    std::vector<float> wData(wCount);

    for(size_t i = 0; i < xCount; ++i)
    {
        xData[i] = static_cast<float>(i + 1);
    }
    for(size_t i = 0; i < wCount; ++i)
    {
        wData[i] = 1.0f;
    }

    // Allocate device buffers and copy inputs to device
    void* dX = nullptr;
    void* dW = nullptr;
    void* dY = nullptr;
    ASSERT_EQ(hipMalloc(&dX, xCount * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dW, wCount * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dY, yCount * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemset(dY, 0, yCount * sizeof(float)), hipSuccess);

    ASSERT_EQ(
        hipMemcpy(dX, xData.data(), xCount * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(
        hipMemcpy(dW, wData.data(), wCount * sizeof(float), hipMemcpyHostToDevice), hipSuccess);

    // Execute the plan with device pointers
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[X_UID] = dX;
    variantPack[W_UID] = dW;
    variantPack[Y_UID] = dY;

    patient.execute(variantPack);

    // Copy GPU result back to host
    std::vector<float> gpuYData(yCount);
    ASSERT_EQ(
        hipMemcpy(gpuYData.data(), dY, yCount * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

    static_cast<void>(hipFree(dX));
    static_cast<void>(hipFree(dW));
    static_cast<void>(hipFree(dY));

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
