// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_gpu_ref/ShallowGpuTensor.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace hipdnn_gpu_ref;
using namespace hipdnn_data_sdk::utilities;

TEST(TestShallowGpuTensor, ConstructionAndShape)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {1, 3, 2, 2};
    auto strides = std::vector<int64_t>{12, 4, 2, 1};

    constexpr size_t ELEMENT_COUNT = 12; // 1*3*2*2
    void* dPtr = nullptr;
    ASSERT_EQ(hipMalloc(&dPtr, ELEMENT_COUNT * sizeof(float)), hipSuccess);

    const ShallowGpuTensor<float> tensor(dPtr, dims, strides);

    EXPECT_EQ(tensor.dims(), dims);
    EXPECT_EQ(tensor.strides(), strides);
    EXPECT_TRUE(tensor.isPacked());
    EXPECT_EQ(tensor.elementCount(), ELEMENT_COUNT);
    EXPECT_EQ(tensor.elementSpace(), ELEMENT_COUNT);

    static_cast<void>(hipFree(dPtr));
}

TEST(TestShallowGpuTensor, MemoryAccessDeviceOnly)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {1, 1, 2, 2};
    const std::vector<int64_t> strides = {4, 4, 2, 1};

    void* dPtr = nullptr;
    ASSERT_EQ(hipMalloc(&dPtr, 4 * sizeof(float)), hipSuccess);

    ShallowGpuTensor<float> tensor(dPtr, dims, strides);

    EXPECT_EQ(tensor.memory().deviceData(), dPtr);
    EXPECT_EQ(tensor.memory().location(), MemoryLocation::DEVICE);

    static_cast<void>(hipFree(dPtr));
}

TEST(TestShallowGpuTensor, FillWithValueThrows)
{
    SKIP_IF_NO_DEVICES();

    void* dPtr = nullptr;
    ASSERT_EQ(hipMalloc(&dPtr, 4 * sizeof(float)), hipSuccess);

    ShallowGpuTensor<float> tensor(dPtr, {1, 1, 2, 2}, {4, 4, 2, 1});

    EXPECT_THROW(tensor.fillWithValue(1.0f), std::runtime_error);

    static_cast<void>(hipFree(dPtr));
}

TEST(TestShallowGpuTensor, FillWithRandomValuesThrows)
{
    SKIP_IF_NO_DEVICES();

    void* dPtr = nullptr;
    ASSERT_EQ(hipMalloc(&dPtr, 4 * sizeof(float)), hipSuccess);

    ShallowGpuTensor<float> tensor(dPtr, {1, 1, 2, 2}, {4, 4, 2, 1});

    EXPECT_THROW(tensor.fillWithRandomValues(-1.0f, 1.0f, 42), std::runtime_error);

    static_cast<void>(hipFree(dPtr));
}

TEST(TestShallowGpuTensor, FillWithDataThrows)
{
    SKIP_IF_NO_DEVICES();

    void* dPtr = nullptr;
    ASSERT_EQ(hipMalloc(&dPtr, 4 * sizeof(float)), hipSuccess);

    ShallowGpuTensor<float> tensor(dPtr, {1, 1, 2, 2}, {4, 4, 2, 1});

    std::array<float, 4> hostData = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_THROW(tensor.fillWithData(hostData.data(), sizeof(hostData)), std::runtime_error);

    static_cast<void>(hipFree(dPtr));
}

TEST(TestShallowGpuTensor, HostAccessThrows)
{
    SKIP_IF_NO_DEVICES();

    void* dPtr = nullptr;
    ASSERT_EQ(hipMalloc(&dPtr, 4 * sizeof(float)), hipSuccess);

    ShallowGpuTensor<float> tensor(dPtr, {1, 1, 2, 2}, {4, 4, 2, 1});

    auto& mem = tensor.memory();
    EXPECT_THROW(mem.hostData(), std::runtime_error);
    EXPECT_THROW(mem.hostDataAsync(), std::runtime_error);
    EXPECT_THROW(mem.markHostModified(), std::runtime_error);

    static_cast<void>(hipFree(dPtr));
}

TEST(TestShallowGpuTensor, NonPackedTensor)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8, 16};

    // elementSpace = (2-1)*2 + (2-1)*4 + (2-1)*8 + (2-1)*16 + 1 = 31
    constexpr size_t ELEMENT_SPACE = 31;
    void* dPtr = nullptr;
    ASSERT_EQ(hipMalloc(&dPtr, ELEMENT_SPACE * sizeof(float)), hipSuccess);

    const ShallowGpuTensor<float> tensor(dPtr, dims, strides);

    EXPECT_EQ(tensor.elementCount(), 16u); // 2*2*2*2
    EXPECT_EQ(tensor.elementSpace(), ELEMENT_SPACE);
    EXPECT_FALSE(tensor.isPacked());

    static_cast<void>(hipFree(dPtr));
}

TEST(TestShallowGpuTensor, MoveSemantics)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {1, 1, 3, 3};
    const std::vector<int64_t> strides = {9, 9, 3, 1};

    void* dPtr = nullptr;
    ASSERT_EQ(hipMalloc(&dPtr, 9 * sizeof(float)), hipSuccess);

    ShallowGpuTensor<float> source(dPtr, dims, strides);
    ShallowGpuTensor<float> dest(std::move(source));

    // Destination preserves dims and pointer
    EXPECT_EQ(dest.dims(), dims);
    EXPECT_EQ(dest.strides(), strides);
    EXPECT_EQ(dest.memory().deviceData(), dPtr);
    EXPECT_EQ(dest.elementCount(), 9u);

    // Source memory is emptied — intentionally inspecting moved-from state
    EXPECT_EQ(source.memory().deviceData(), nullptr); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(source.memory().count(), 0u);
    EXPECT_TRUE(source.memory().empty());

    static_cast<void>(hipFree(dPtr));
}
