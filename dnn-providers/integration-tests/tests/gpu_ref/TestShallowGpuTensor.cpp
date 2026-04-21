// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_gpu_ref/ShallowGpuTensor.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

using hipdnn_data_sdk::utilities::MemoryLocation;
using hipdnn_data_sdk::utilities::Workspace;
using hipdnn_gpu_ref::ShallowGpuTensor;

TEST(TestShallowGpuTensor, PackedTensorReportsCorrectShape)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {1, 3, 2, 2};
    const std::vector<int64_t> strides = {12, 4, 2, 1};

    constexpr size_t ELEMENT_COUNT = 12; // 1*3*2*2
    const Workspace workspace(ELEMENT_COUNT * sizeof(float));

    const ShallowGpuTensor<float> tensor(workspace.get(), dims, strides);

    EXPECT_EQ(tensor.dims(), dims);
    EXPECT_EQ(tensor.strides(), strides);
    EXPECT_TRUE(tensor.isPacked());
    EXPECT_EQ(tensor.elementCount(), ELEMENT_COUNT);
    EXPECT_EQ(tensor.elementSpace(), ELEMENT_COUNT);
}

TEST(TestShallowGpuTensor, NonPackedTensorReportsCorrectElementSpace)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8, 16};

    // elementSpace = (2-1)*2 + (2-1)*4 + (2-1)*8 + (2-1)*16 + 1 = 31
    constexpr size_t ELEMENT_SPACE = 31;
    const Workspace workspace(ELEMENT_SPACE * sizeof(float));

    const ShallowGpuTensor<float> tensor(workspace.get(), dims, strides);

    EXPECT_EQ(tensor.elementCount(), 16u); // 2*2*2*2
    EXPECT_EQ(tensor.elementSpace(), ELEMENT_SPACE);
    EXPECT_FALSE(tensor.isPacked());
}

TEST(TestShallowGpuTensor, MemoryExposesDevicePointer)
{
    SKIP_IF_NO_DEVICES();

    const Workspace workspace(4 * sizeof(float));

    ShallowGpuTensor<float> tensor(workspace.get(), {1, 1, 2, 2}, {4, 4, 2, 1});

    EXPECT_EQ(tensor.memory().deviceData(), workspace.get());
    EXPECT_EQ(tensor.memory().location(), MemoryLocation::DEVICE);
}

TEST(TestShallowGpuTensor, HostFillOperationsThrow)
{
    SKIP_IF_NO_DEVICES();

    const Workspace workspace(4 * sizeof(float));

    ShallowGpuTensor<float> tensor(workspace.get(), {1, 1, 2, 2}, {4, 4, 2, 1});

    EXPECT_THROW(tensor.fillWithValue(1.0f), std::runtime_error);
    EXPECT_THROW(tensor.fillWithRandomValues(-1.0f, 1.0f, 42), std::runtime_error);

    std::array<float, 4> hostData = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_THROW(tensor.fillWithData(hostData.data(), sizeof(hostData)), std::runtime_error);
}

TEST(TestShallowGpuTensor, HostMemoryAccessThrows)
{
    SKIP_IF_NO_DEVICES();

    const Workspace workspace(4 * sizeof(float));

    ShallowGpuTensor<float> tensor(workspace.get(), {1, 1, 2, 2}, {4, 4, 2, 1});

    auto& mem = tensor.memory();
    EXPECT_THROW(mem.hostData(), std::runtime_error);
    EXPECT_THROW(mem.hostDataAsync(), std::runtime_error);
    EXPECT_THROW(mem.markHostModified(), std::runtime_error);
}

TEST(TestShallowGpuTensor, MoveConstruction)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {1, 1, 3, 3};
    const std::vector<int64_t> strides = {9, 9, 3, 1};

    const Workspace workspace(9 * sizeof(float));

    ShallowGpuTensor<float> source(workspace.get(), dims, strides);
    ShallowGpuTensor<float> dest(std::move(source));

    EXPECT_EQ(dest.dims(), dims);
    EXPECT_EQ(dest.strides(), strides);
    EXPECT_EQ(dest.memory().deviceData(), workspace.get());
    EXPECT_EQ(dest.elementCount(), 9u);

    // Source memory is emptied — intentionally inspecting moved-from state
    EXPECT_EQ(source.memory().deviceData(), nullptr); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(source.memory().count(), 0u);
    EXPECT_TRUE(source.memory().empty());
}
