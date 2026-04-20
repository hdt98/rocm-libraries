// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipdnnException.hpp"
#include "heuristics/DeviceProperties.hpp"
#include <array>
#include <gtest/gtest.h>
#include <limits>

using namespace hipdnn_backend::heuristics;

class TestDeviceProperties : public ::testing::Test
{
protected:
    static DeviceProperties createTestProperties()
    {
        DeviceProperties props;
        props.deviceId = 0;
        props.multiProcessorCount = 120;
        props.totalGlobalMem = 16ULL * 1024 * 1024 * 1024; // 16 GB
        props.architectureName = "gfx90a";
        return props;
    }
};

// Test basic round-trip serialization
TEST_F(TestDeviceProperties, SerializeDeserializeRoundTrip)
{
    const DeviceProperties original = createTestProperties();

    // Serialize
    auto serialized = serializeDeviceProperties(original);
    ASSERT_FALSE(serialized.empty());

    // Deserialize
    const DeviceProperties deserialized
        = deserializeDeviceProperties(serialized.data(), serialized.size());

    // Verify fields match
    EXPECT_EQ(deserialized.deviceId, original.deviceId);
    EXPECT_EQ(deserialized.multiProcessorCount, original.multiProcessorCount);
    EXPECT_EQ(deserialized.totalGlobalMem, original.totalGlobalMem);
    EXPECT_EQ(deserialized.architectureName, original.architectureName);
}

// Test default/zero values
TEST_F(TestDeviceProperties, SerializeDefaultValues)
{
    const DeviceProperties props; // All zeros/defaults

    auto serialized = serializeDeviceProperties(props);
    auto deserialized = deserializeDeviceProperties(serialized.data(), serialized.size());

    EXPECT_EQ(deserialized.deviceId, -1); // Default from struct
    EXPECT_EQ(deserialized.multiProcessorCount, 0);
    EXPECT_EQ(deserialized.totalGlobalMem, 0u);
    EXPECT_TRUE(deserialized.architectureName.empty()); // Default is empty string
}

// Test with maximum values
TEST_F(TestDeviceProperties, SerializeMaxValues)
{
    DeviceProperties props;
    props.deviceId = std::numeric_limits<int>::max();
    props.multiProcessorCount = std::numeric_limits<int>::max();
    props.totalGlobalMem = std::numeric_limits<size_t>::max();
    props.architectureName = "gfx942"; // Different architecture

    auto serialized = serializeDeviceProperties(props);
    auto deserialized = deserializeDeviceProperties(serialized.data(), serialized.size());

    EXPECT_EQ(deserialized.deviceId, props.deviceId);
    EXPECT_EQ(deserialized.multiProcessorCount, props.multiProcessorCount);
    EXPECT_EQ(deserialized.totalGlobalMem, props.totalGlobalMem);
    EXPECT_EQ(deserialized.architectureName, props.architectureName);
}

// Test wrapper function
TEST_F(TestDeviceProperties, WrapSerializedDeviceProperties)
{
    const DeviceProperties props = createTestProperties();
    auto serialized = serializeDeviceProperties(props);

    // Create wrapper
    const hipdnnPluginConstData_t wrapper = wrapSerializedDeviceProperties(serialized);

    // Verify wrapper points to serialized data
    EXPECT_EQ(wrapper.ptr, serialized.data());
    EXPECT_EQ(wrapper.size, serialized.size());

    // Verify we can deserialize through wrapper
    auto deserialized
        = deserializeDeviceProperties(static_cast<const uint8_t*>(wrapper.ptr), wrapper.size);
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
    const std::array<uint8_t, 10> dummy = {};
    EXPECT_THROW(deserializeDeviceProperties(dummy.data(), 0), hipdnn_backend::HipdnnException);
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
    const DeviceProperties props = createTestProperties();
    auto serialized = serializeDeviceProperties(props);

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
        auto serialized = serializeDeviceProperties(props);
        auto deserialized = deserializeDeviceProperties(serialized.data(), serialized.size());

        EXPECT_EQ(deserialized.deviceId, props.deviceId);
        EXPECT_EQ(deserialized.multiProcessorCount, props.multiProcessorCount);
        EXPECT_EQ(deserialized.totalGlobalMem, props.totalGlobalMem);
        EXPECT_EQ(deserialized.architectureName, props.architectureName);

        // Use deserialized as input for next iteration
        props = deserialized;
    }
}

// Test that different properties produce different serializations
TEST_F(TestDeviceProperties, DifferentPropertiesProduceDifferentSerializations)
{
    DeviceProperties props1;
    props1.deviceId = 0;
    props1.multiProcessorCount = 120;
    props1.totalGlobalMem = 16ULL * 1024 * 1024 * 1024;
    props1.architectureName = "gfx90a";

    DeviceProperties props2;
    props2.deviceId = 1; // Different device ID
    props2.multiProcessorCount = 120;
    props2.totalGlobalMem = 16ULL * 1024 * 1024 * 1024;
    props2.architectureName = "gfx90a";

    auto serialized1 = serializeDeviceProperties(props1);
    auto serialized2 = serializeDeviceProperties(props2);

    // Serializations should differ
    EXPECT_NE(serialized1, serialized2);
}

// Test serialization is deterministic
TEST_F(TestDeviceProperties, SerializationIsDeterministic)
{
    const DeviceProperties props = createTestProperties();

    auto serialized1 = serializeDeviceProperties(props);
    auto serialized2 = serializeDeviceProperties(props);

    // Same input should produce identical serialization
    EXPECT_EQ(serialized1, serialized2);
}

// Test architecture name serialization with different architectures
TEST_F(TestDeviceProperties, ArchitectureNameRoundTrip)
{
    // Test gfx90a
    DeviceProperties props1;
    props1.deviceId = 0;
    props1.multiProcessorCount = 110;
    props1.totalGlobalMem = 32ULL * 1024 * 1024 * 1024;
    props1.architectureName = "gfx90a";

    auto serialized1 = serializeDeviceProperties(props1);
    auto deserialized1 = deserializeDeviceProperties(serialized1.data(), serialized1.size());
    EXPECT_EQ(deserialized1.architectureName, "gfx90a");

    // Test gfx942
    DeviceProperties props2;
    props2.deviceId = 1;
    props2.multiProcessorCount = 304;
    props2.totalGlobalMem = 192ULL * 1024 * 1024 * 1024;
    props2.architectureName = "gfx942";

    auto serialized2 = serializeDeviceProperties(props2);
    auto deserialized2 = deserializeDeviceProperties(serialized2.data(), serialized2.size());
    EXPECT_EQ(deserialized2.architectureName, "gfx942");

    // Test gfx1100 (RDNA3)
    DeviceProperties props3;
    props3.deviceId = 2;
    props3.multiProcessorCount = 96;
    props3.totalGlobalMem = 16ULL * 1024 * 1024 * 1024;
    props3.architectureName = "gfx1100";

    auto serialized3 = serializeDeviceProperties(props3);
    auto deserialized3 = deserializeDeviceProperties(serialized3.data(), serialized3.size());
    EXPECT_EQ(deserialized3.architectureName, "gfx1100");
}

// Test that architecture name affects serialization
TEST_F(TestDeviceProperties, DifferentArchitecturesProduceDifferentSerializations)
{
    DeviceProperties props1;
    props1.deviceId = 0;
    props1.multiProcessorCount = 120;
    props1.totalGlobalMem = 16ULL * 1024 * 1024 * 1024;
    props1.architectureName = "gfx90a";

    DeviceProperties props2 = props1;
    props2.architectureName = "gfx942"; // Only architecture differs

    auto serialized1 = serializeDeviceProperties(props1);
    auto serialized2 = serializeDeviceProperties(props2);

    // Different architectures should produce different serializations
    EXPECT_NE(serialized1, serialized2);
}

// Test empty architecture name
TEST_F(TestDeviceProperties, EmptyArchitectureNameRoundTrip)
{
    DeviceProperties props;
    props.deviceId = 0;
    props.multiProcessorCount = 120;
    props.totalGlobalMem = 16ULL * 1024 * 1024 * 1024;
    props.architectureName = ""; // Explicitly empty

    auto serialized = serializeDeviceProperties(props);
    auto deserialized = deserializeDeviceProperties(serialized.data(), serialized.size());

    EXPECT_TRUE(deserialized.architectureName.empty());
}

TEST_F(TestDeviceProperties, QueryDevicePropertiesSucceeds)
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

TEST_F(TestDeviceProperties, QueryDevicePropertiesIsConsistent)
{
    // Query twice and verify results are consistent
    const DeviceProperties props1 = queryDeviceProperties();
    const DeviceProperties props2 = queryDeviceProperties();

    // Properties should be consistent across calls (same device)
    EXPECT_EQ(props1.deviceId, props2.deviceId);
    EXPECT_EQ(props1.multiProcessorCount, props2.multiProcessorCount);
    EXPECT_EQ(props1.totalGlobalMem, props2.totalGlobalMem);
    EXPECT_EQ(props1.architectureName, props2.architectureName);
}

TEST_F(TestDeviceProperties, QueryDevicePropertiesReturnsReasonableValues)
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
        // Architecture name should be populated if device is valid
        // (may be empty if hipGetDeviceProperties fails, but that's okay)
    }
    else
    {
        // No device or error case - defaults should be returned
        EXPECT_EQ(props.deviceId, 0); // Default value from struct initialization
        EXPECT_EQ(props.multiProcessorCount, 0);
        EXPECT_EQ(props.totalGlobalMem, 0u);
        // Architecture name may be empty in error cases
    }
}

TEST_F(TestDeviceProperties, QueryDevicePropertiesIncludesArchitectureName)
{
    const DeviceProperties props = queryDeviceProperties();

    // If device query succeeded, architecture name should be available
    if(props.deviceId >= 0 && props.multiProcessorCount > 0)
    {
        // Architecture name should be populated for a valid device
        // Common AMD GPU architectures start with "gfx"
        // This is hardware-dependent, so we just check it's not empty
        EXPECT_FALSE(props.architectureName.empty())
            << "Expected architecture name for valid device with ID " << props.deviceId;
    }
    // If no device or query failed, architecture name may be empty (no assertion)
}

// ========== Round-trip with Query ==========

TEST_F(TestDeviceProperties, QueryAndSerializeRoundTrip)
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
    EXPECT_EQ(deserialized.architectureName, queried.architectureName);
}

TEST_F(TestDeviceProperties, QueryResultCanBeWrapped)
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

TEST_F(TestDeviceProperties, MultipleQueriesSucceed)
{
    for(int i = 0; i < 10; ++i)
    {
        EXPECT_NO_THROW({
            const DeviceProperties props = queryDeviceProperties();
            EXPECT_GE(props.deviceId, -1);
        });
    }
}

TEST_F(TestDeviceProperties, QueryAndSerializeMultipleTimes)
{
    for(int i = 0; i < 5; ++i)
    {
        const DeviceProperties props = queryDeviceProperties();
        const auto serialized = serializeDeviceProperties(props);
        const auto deserialized = deserializeDeviceProperties(serialized.data(), serialized.size());

        EXPECT_EQ(deserialized.deviceId, props.deviceId);
        EXPECT_EQ(deserialized.multiProcessorCount, props.multiProcessorCount);
        EXPECT_EQ(deserialized.totalGlobalMem, props.totalGlobalMem);
        EXPECT_EQ(deserialized.architectureName, props.architectureName);
    }
}
