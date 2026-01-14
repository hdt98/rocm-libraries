// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>

#include "../TestPluginConstants.hpp"
#include "descriptors/KnobInfoDescriptor.hpp"
#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>

namespace hipdnn_backend
{

class TestKnobInfoDescriptor : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _descriptor = std::make_shared<KnobInfoDescriptor>();
    }

    // Helper method to create a test knob
    static flatbuffers::DetachedBuffer createTestKnob(int64_t knobId,
                                                      const std::string& description)
    {
        flatbuffers::FlatBufferBuilder builder;

        // Create knob ID string
        auto knobIdStr = builder.CreateString("KNOB_" + std::to_string(knobId));
        auto descStr = builder.CreateString(description);

        // Create default value (using IntValue as example)
        auto defaultValue = hipdnn_data_sdk::data_objects::CreateIntValue(builder, 100);

        // Create the knob
        auto knob = hipdnn_data_sdk::data_objects::CreateKnob(
            builder,
            knobId,
            knobIdStr,
            descStr,
            hipdnn_data_sdk::data_objects::KnobValue::IntValue,
            defaultValue.Union(),
            hipdnn_data_sdk::data_objects::KnobConstraint::NONE,
            0, // no constraints
            false // not deprecated
        );

        builder.Finish(knob);
        return builder.Release();
    }

    std::shared_ptr<KnobInfoDescriptor> _descriptor;
};

TEST_F(TestKnobInfoDescriptor, InitialState)
{
    EXPECT_FALSE(_descriptor->isFinalized());
}

TEST_F(TestKnobInfoDescriptor, GetStaticType)
{
    EXPECT_EQ(KnobInfoDescriptor::getStaticType(), HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR);
}

TEST_F(TestKnobInfoDescriptor, FinalizeWithoutKnobDataFails)
{
    EXPECT_THROW(_descriptor->finalize(), HipdnnException);
}

TEST_F(TestKnobInfoDescriptor, SetKnobDataAndFinalize)
{
    auto knobBuffer = createTestKnob(42, "Test knob");
    auto knob = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Knob>(knobBuffer.data());

    EXPECT_NO_THROW(_descriptor->setKnobData(knob));
    EXPECT_NO_THROW(_descriptor->finalize());
    EXPECT_TRUE(_descriptor->isFinalized());
}

TEST_F(TestKnobInfoDescriptor, SetKnobDataAfterFinalizeFails)
{
    auto knobBuffer = createTestKnob(42, "Test knob");
    auto knob = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Knob>(knobBuffer.data());

    _descriptor->setKnobData(knob);
    _descriptor->finalize();

    EXPECT_THROW(_descriptor->setKnobData(knob), HipdnnException);
}

TEST_F(TestKnobInfoDescriptor, SetKnobDataWithNullFails)
{
    EXPECT_THROW(_descriptor->setKnobData(nullptr), HipdnnException);
}

TEST_F(TestKnobInfoDescriptor, GetSerializedKnobValue)
{
    auto knobBuffer = createTestKnob(42, "Test knob");
    auto knob = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Knob>(knobBuffer.data());

    _descriptor->setKnobData(knob);
    _descriptor->finalize();

    void* serializedData = nullptr;
    int64_t elementCount = 0;

    EXPECT_NO_THROW(_descriptor->getAttribute(HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                              HIPDNN_TYPE_VOID_PTR,
                                              1,
                                              &elementCount,
                                              &serializedData));

    EXPECT_EQ(elementCount, 1);
    EXPECT_NE(serializedData, nullptr);

    // Clean up allocated memory
    delete[] static_cast<uint8_t*>(serializedData);
}

TEST_F(TestKnobInfoDescriptor, GetSerializedKnobValueBeforeFinalizeFails)
{
    void* serializedData = nullptr;

    EXPECT_THROW(_descriptor->getAttribute(HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                           HIPDNN_TYPE_VOID_PTR,
                                           1,
                                           nullptr,
                                           &serializedData),
                 HipdnnException);
}

TEST_F(TestKnobInfoDescriptor, GetSerializedKnobValueWrongTypeFails)
{
    auto knobBuffer = createTestKnob(42, "Test knob");
    auto knob = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Knob>(knobBuffer.data());

    _descriptor->setKnobData(knob);
    _descriptor->finalize();

    void* serializedData = nullptr;

    EXPECT_THROW(_descriptor->getAttribute(HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                           HIPDNN_TYPE_INT64, // Wrong type
                                           1,
                                           nullptr,
                                           &serializedData),
                 HipdnnException);
}

TEST_F(TestKnobInfoDescriptor, GetSerializedKnobValueWrongElementCountFails)
{
    auto knobBuffer = createTestKnob(42, "Test knob");
    auto knob = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Knob>(knobBuffer.data());

    _descriptor->setKnobData(knob);
    _descriptor->finalize();

    void* serializedData = nullptr;

    EXPECT_THROW(_descriptor->getAttribute(HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                           HIPDNN_TYPE_VOID_PTR,
                                           2, // Wrong element count
                                           nullptr,
                                           &serializedData),
                 HipdnnException);
}

TEST_F(TestKnobInfoDescriptor, GetSerializedKnobValueNullPointerFails)
{
    auto knobBuffer = createTestKnob(42, "Test knob");
    auto knob = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Knob>(knobBuffer.data());

    _descriptor->setKnobData(knob);
    _descriptor->finalize();

    EXPECT_THROW(
        _descriptor->getAttribute(
            HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT, HIPDNN_TYPE_VOID_PTR, 1, nullptr, nullptr),
        HipdnnException);
}

TEST_F(TestKnobInfoDescriptor, UnsupportedGetAttributeThrows)
{
    auto knobBuffer = createTestKnob(42, "Test knob");
    auto knob = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Knob>(knobBuffer.data());

    _descriptor->setKnobData(knob);
    _descriptor->finalize();

    int64_t value = 0;

    EXPECT_THROW(_descriptor->getAttribute(
                     HIPDNN_ATTR_KNOB_INFO_TYPE, HIPDNN_TYPE_INT64, 1, nullptr, &value),
                 HipdnnException);

    EXPECT_THROW(_descriptor->getAttribute(
                     HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, nullptr, &value),
                 HipdnnException);
}

TEST_F(TestKnobInfoDescriptor, SetAttributeNotSupported)
{
    int64_t value = 42;

    EXPECT_THROW(
        _descriptor->setAttribute(HIPDNN_ATTR_KNOB_INFO_TYPE, HIPDNN_TYPE_INT64, 1, &value),
        HipdnnException);
}

TEST_F(TestKnobInfoDescriptor, ToString)
{
    auto str1 = _descriptor->toString();
    EXPECT_NE(str1.find("KnobInfoDescriptor"), std::string::npos);
    EXPECT_NE(str1.find("knobDataSet=false"), std::string::npos);

    auto knobBuffer = createTestKnob(42, "Test knob");
    auto knob = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Knob>(knobBuffer.data());

    _descriptor->setKnobData(knob);
    auto str2 = _descriptor->toString();
    EXPECT_NE(str2.find("knobDataSet=true"), std::string::npos);
    EXPECT_NE(str2.find("bufferSize="), std::string::npos);
}

TEST_F(TestKnobInfoDescriptor, MultipleKnobs)
{
    // Test setting different knob data
    auto knobBuffer1 = createTestKnob(42, "First knob");
    auto knob1 = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Knob>(knobBuffer1.data());

    _descriptor->setKnobData(knob1);

    // Create a new knob with different data
    auto knobBuffer2 = createTestKnob(100, "Second knob");
    auto knob2 = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Knob>(knobBuffer2.data());

    // Should be able to set new knob data before finalize
    EXPECT_NO_THROW(_descriptor->setKnobData(knob2));

    _descriptor->finalize();

    // Verify the final knob data is from the second knob
    void* serializedData = nullptr;
    int64_t elementCount = 0;

    _descriptor->getAttribute(HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                              HIPDNN_TYPE_VOID_PTR,
                              1,
                              &elementCount,
                              &serializedData);

    // Verify we can deserialize it back
    auto deserializedKnob
        = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Knob>(serializedData);
    EXPECT_EQ(deserializedKnob->knob_id(), 100);

    delete[] static_cast<uint8_t*>(serializedData);
}

} // namespace hipdnn_backend
