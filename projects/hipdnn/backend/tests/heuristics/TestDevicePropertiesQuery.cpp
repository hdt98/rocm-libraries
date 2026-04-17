// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestDevicePropertiesQuery.cpp
 * @brief Additional tests for DeviceProperties query functionality
 *
 * These tests cover the queryDeviceProperties() function which interacts with HIP.
 */

#include "heuristics/DeviceProperties.hpp"
#include <gtest/gtest.h>

using namespace hipdnn_backend::heuristics;

class TestDevicePropertiesQuery : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ========== Query Tests ==========

TEST_F(TestDevicePropertiesQuery, QueryDevicePropertiesSucceeds)
{
    // This test requires a HIP device to be available
    // If no device is available, queryDeviceProperties should return defaults without throwing
    EXPECT_NO_THROW({
        const DeviceProperties props = queryDeviceProperties();

        // Device ID should be set (either valid device or -1/0 for default)
        EXPECT_GE(props.deviceId, -1);

        // Multi-processor count should be non-negative
        EXPECT_GE(props.multiProcessorCount, 0);

        // Total global memory should be non-negative
        EXPECT_GE(props.totalGlobalMem, 0u);
    });
}

TEST_F(TestDevicePropertiesQuery, QueryDevicePropertiesIsConsistent)
{
    // Query twice and verify results are consistent
    const DeviceProperties props1 = queryDeviceProperties();
    const DeviceProperties props2 = queryDeviceProperties();

    // Properties should be consistent across calls (same device)
    EXPECT_EQ(props1.deviceId, props2.deviceId);
    EXPECT_EQ(props1.multiProcessorCount, props2.multiProcessorCount);
    EXPECT_EQ(props1.totalGlobalMem, props2.totalGlobalMem);
}

TEST_F(TestDevicePropertiesQuery, QueryDevicePropertiesReturnsReasonableValues)
{
    const DeviceProperties props = queryDeviceProperties();

    // If a device is available, multiProcessorCount should be positive
    // If no device, it should be 0 (default)
    if(props.deviceId >= 0)
    {
        // Device ID is valid - we might have a real device
        // Don't assert on specific values as they're hardware-dependent
        EXPECT_TRUE(props.multiProcessorCount >= 0);
        EXPECT_TRUE(props.totalGlobalMem >= 0);
    }
    else
    {
        // No device or error case - defaults should be returned
        EXPECT_EQ(props.deviceId, 0); // Default value from struct initialization
        EXPECT_EQ(props.multiProcessorCount, 0);
        EXPECT_EQ(props.totalGlobalMem, 0u);
    }
}

// ========== Round-trip with Query ==========

TEST_F(TestDevicePropertiesQuery, QueryAndSerializeRoundTrip)
{
    // Query actual device properties
    const DeviceProperties queried = queryDeviceProperties();

    // Serialize
    const auto serialized = serializeDeviceProperties(queried);
    ASSERT_FALSE(serialized.empty());

    // Deserialize
    const DeviceProperties deserialized
        = deserializeDeviceProperties(serialized.data(), serialized.size());

    // Verify fields match
    EXPECT_EQ(deserialized.deviceId, queried.deviceId);
    EXPECT_EQ(deserialized.multiProcessorCount, queried.multiProcessorCount);
    EXPECT_EQ(deserialized.totalGlobalMem, queried.totalGlobalMem);
}

TEST_F(TestDevicePropertiesQuery, QueryResultCanBeWrapped)
{
    const DeviceProperties props = queryDeviceProperties();
    const auto serialized = serializeDeviceProperties(props);

    const hipdnnPluginConstData_t wrapper = wrapSerializedDeviceProperties(serialized);

    // Verify wrapper is valid
    EXPECT_NE(wrapper.ptr, nullptr);
    EXPECT_GT(wrapper.size, 0u);
    EXPECT_EQ(wrapper.size, serialized.size());
}

// ========== Multiple Query Tests ==========

TEST_F(TestDevicePropertiesQuery, MultipleQueriesSucceed)
{
    for(int i = 0; i < 10; ++i)
    {
        EXPECT_NO_THROW({
            const DeviceProperties props = queryDeviceProperties();
            EXPECT_GE(props.deviceId, -1);
        });
    }
}

TEST_F(TestDevicePropertiesQuery, QueryAndSerializeMultipleTimes)
{
    for(int i = 0; i < 5; ++i)
    {
        const DeviceProperties props = queryDeviceProperties();
        const auto serialized = serializeDeviceProperties(props);
        const auto deserialized = deserializeDeviceProperties(serialized.data(), serialized.size());

        EXPECT_EQ(deserialized.deviceId, props.deviceId);
        EXPECT_EQ(deserialized.multiProcessorCount, props.multiProcessorCount);
        EXPECT_EQ(deserialized.totalGlobalMem, props.totalGlobalMem);
    }
}
