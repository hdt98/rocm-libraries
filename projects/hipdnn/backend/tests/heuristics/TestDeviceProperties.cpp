// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "heuristics/DeviceProperties.hpp"
#include <gtest/gtest.h>
#include <limits>

using namespace hipdnn_backend::heuristics;

class TestDeviceProperties : public ::testing::Test
{
protected:
    DeviceProperties createTestProperties()
    {
        DeviceProperties props;
        props.deviceId            = 0;
        props.multiProcessorCount = 120;
        props.totalGlobalMem      = 16ULL * 1024 * 1024 * 1024; // 16 GB
        return props;
    }
};

// Test basic round-trip serialization
TEST_F(TestDeviceProperties, SerializeDeserializeRoundTrip)
{
    DeviceProperties original = createTestProperties();

    // Serialize
    auto serialized = serializeDeviceProperties(original);
    ASSERT_FALSE(serialized.empty());

    // Deserialize
    DeviceProperties deserialized =
        deserializeDeviceProperties(serialized.data(), serialized.size());

    // Verify fields match
    EXPECT_EQ(deserialized.deviceId, original.deviceId);
    EXPECT_EQ(deserialized.multiProcessorCount, original.multiProcessorCount);
    EXPECT_EQ(deserialized.totalGlobalMem, original.totalGlobalMem);
}

// Test default/zero values
TEST_F(TestDeviceProperties, SerializeDefaultValues)
{
    DeviceProperties props; // All zeros/defaults

    auto serialized   = serializeDeviceProperties(props);
    auto deserialized = deserializeDeviceProperties(serialized.data(), serialized.size());

    EXPECT_EQ(deserialized.deviceId, -1); // Default from struct
    EXPECT_EQ(deserialized.multiProcessorCount, 0);
    EXPECT_EQ(deserialized.totalGlobalMem, 0u);
}

// Test with maximum values
TEST_F(TestDeviceProperties, SerializeMaxValues)
{
    DeviceProperties props;
    props.deviceId            = std::numeric_limits<int>::max();
    props.multiProcessorCount = std::numeric_limits<int>::max();
    props.totalGlobalMem      = std::numeric_limits<size_t>::max();

    auto serialized   = serializeDeviceProperties(props);
    auto deserialized = deserializeDeviceProperties(serialized.data(), serialized.size());

    EXPECT_EQ(deserialized.deviceId, props.deviceId);
    EXPECT_EQ(deserialized.multiProcessorCount, props.multiProcessorCount);
    EXPECT_EQ(deserialized.totalGlobalMem, props.totalGlobalMem);
}

// Test wrapper function
TEST_F(TestDeviceProperties, WrapSerializedDeviceProperties)
{
    DeviceProperties props = createTestProperties();
    auto serialized        = serializeDeviceProperties(props);

    // Create wrapper
    hipdnnPluginConstData_t wrapper = wrapSerializedDeviceProperties(serialized);

    // Verify wrapper points to serialized data
    EXPECT_EQ(wrapper.ptr, serialized.data());
    EXPECT_EQ(wrapper.size, serialized.size());

    // Verify we can deserialize through wrapper
    auto deserialized =
        deserializeDeviceProperties(static_cast<const uint8_t*>(wrapper.ptr), wrapper.size);
    EXPECT_EQ(deserialized.deviceId, props.deviceId);
}

// Test error handling - null buffer
TEST_F(TestDeviceProperties, DeserializeNullBuffer)
{
    EXPECT_THROW(deserializeDeviceProperties(nullptr, 100), hipdnn_backend::HipdnnException);
}

// Test error handling - zero size
TEST_F(TestDeviceProperties, DeserializeZeroSize)
{
    uint8_t dummy[10] = {};
    EXPECT_THROW(deserializeDeviceProperties(dummy, 0), hipdnn_backend::HipdnnException);
}

// Test error handling - invalid FlatBuffer data
TEST_F(TestDeviceProperties, DeserializeInvalidData)
{
    // Create garbage data
    std::vector<uint8_t> garbage(100, 0xFF);
    EXPECT_THROW(deserializeDeviceProperties(garbage.data(), garbage.size()),
                 hipdnn_backend::HipdnnException);
}

// Test error handling - truncated buffer
TEST_F(TestDeviceProperties, DeserializeTruncatedBuffer)
{
    DeviceProperties props = createTestProperties();
    auto serialized        = serializeDeviceProperties(props);

    // Try to deserialize with truncated size
    if(serialized.size() > 4)
    {
        EXPECT_THROW(deserializeDeviceProperties(serialized.data(), serialized.size() / 2),
                     hipdnn_backend::HipdnnException);
    }
}

// Test multiple serialize/deserialize cycles
TEST_F(TestDeviceProperties, MultipleRoundTrips)
{
    DeviceProperties props = createTestProperties();

    for(int i = 0; i < 5; ++i)
    {
        auto serialized   = serializeDeviceProperties(props);
        auto deserialized = deserializeDeviceProperties(serialized.data(), serialized.size());

        EXPECT_EQ(deserialized.deviceId, props.deviceId);
        EXPECT_EQ(deserialized.multiProcessorCount, props.multiProcessorCount);
        EXPECT_EQ(deserialized.totalGlobalMem, props.totalGlobalMem);

        // Use deserialized as input for next iteration
        props = deserialized;
    }
}

// Test that different properties produce different serializations
TEST_F(TestDeviceProperties, DifferentPropertiesProduceDifferentSerializations)
{
    DeviceProperties props1;
    props1.deviceId            = 0;
    props1.multiProcessorCount = 120;
    props1.totalGlobalMem      = 16ULL * 1024 * 1024 * 1024;

    DeviceProperties props2;
    props2.deviceId            = 1; // Different device ID
    props2.multiProcessorCount = 120;
    props2.totalGlobalMem      = 16ULL * 1024 * 1024 * 1024;

    auto serialized1 = serializeDeviceProperties(props1);
    auto serialized2 = serializeDeviceProperties(props2);

    // Serializations should differ
    EXPECT_NE(serialized1, serialized2);
}

// Test serialization is deterministic
TEST_F(TestDeviceProperties, SerializationIsDeterministic)
{
    DeviceProperties props = createTestProperties();

    auto serialized1 = serializeDeviceProperties(props);
    auto serialized2 = serializeDeviceProperties(props);

    // Same input should produce identical serialization
    EXPECT_EQ(serialized1, serialized2);
}
