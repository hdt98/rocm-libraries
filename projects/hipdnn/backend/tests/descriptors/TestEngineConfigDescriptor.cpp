// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/EngineConfigDescriptor.hpp"
#include "descriptors/EngineDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockDescriptor.hpp"
#include "mocks/MockEnginePluginResourceManager.hpp"
#include "mocks/MockHandle.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace hipdnn_backend;
using namespace plugin;
using namespace hipdnn_backend::test_utilities;
using namespace ::testing;

using ::testing::Return;

class TestEngineConfigDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<EngineConfigDescriptor> getEngineConfigDescriptor() const
    {
        return _engineConfigWrapper->asDescriptor<EngineConfigDescriptor>();
    }

    std::shared_ptr<MockEngineDescriptor> getMockEngine() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockEngineDescriptor>(
            _mockEngineWrapper.get());
    }

    std::shared_ptr<MockEngineDescriptor> getMockEngineBadType() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockEngineDescriptor>(
            _mockEngineBadTypeWrapper.get());
    }

    std::shared_ptr<MockGraphDescriptor> getMockGraphDescriptor() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockGraphDescriptor>(
            _mockGraphWrapper.get());
    }

    void setEngine() const
    {
        EXPECT_CALL(*getMockEngine(), isFinalized()).WillOnce(Return(true));
        ASSERT_NO_THROW(getEngineConfigDescriptor()->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));
    }

    void makeEngineConfigFinalized() const
    {
        // TODO: These expectations being hidden in here are dangerous.
        // It's easy to forget they exist, and if any of these functions are called prior to this
        // call it is undefined behavior. Similarly, any expectations on functions called within
        // "finalize" or "setup" made after calling this function are UB
        EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
        EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));
        EXPECT_CALL(*getMockEngine(), getGraph()).WillRepeatedly(Return(getMockGraphDescriptor()));
        EXPECT_CALL(*getMockGraphDescriptor(), getHandle()).WillOnce(Return(_mockHandle.get()));
        EXPECT_CALL(*_mockHandle, getPluginResourceManager())
            .WillOnce(Return(_mockEnginePluginResourceManager));
        EXPECT_CALL(*_mockEnginePluginResourceManager, getWorkspaceSize(_, _, _))
            .WillOnce(Return(1024));

        setEngine();
        ASSERT_NO_THROW(getEngineConfigDescriptor()->finalize());
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _engineConfigWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockEngineWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockEngineBadTypeWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockWrongTypeWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockGraphWrapper = nullptr;
    std::unique_ptr<MockHandle> _mockHandle = nullptr;
    std::shared_ptr<MockEnginePluginResourceManager> _mockEnginePluginResourceManager = nullptr;

    void SetUp() override
    {
        _engineConfigWrapper = createDescriptor<EngineConfigDescriptor>();
        _mockEngineWrapper = createDescriptor<MockEngineDescriptor>();
        _mockEngineBadTypeWrapper = createDescriptor<MockEngineDescriptor>();
        _mockWrongTypeWrapper = createDescriptor<MockDescriptor<EngineConfigDescriptor>>();
        _mockGraphWrapper = createDescriptor<MockGraphDescriptor>();
        _mockHandle = std::make_unique<MockHandle>();
        _mockEnginePluginResourceManager = std::make_shared<MockEnginePluginResourceManager>();
    }
};

TEST_F(TestEngineConfigDescriptor, CreateEngineConfigDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    ASSERT_NE(engineConfig, nullptr);
    ASSERT_FALSE(engineConfig->isFinalized());
    ASSERT_EQ(engineConfig->getType(), HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
}

TEST_F(TestEngineConfigDescriptor, SetEngineConfigDescriptorEngine)
{
    auto engineConfig = getEngineConfigDescriptor();

    EXPECT_CALL(*getMockEngineBadType(), isFinalized()).Times(1);
    EXPECT_CALL(*getMockEngine(), getEngineId()).Times(AnyNumber());
    EXPECT_CALL(*getMockEngine(), isFinalized()).WillOnce(Return(false)).WillOnce(Return(true));

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_NO_THROW(engineConfig->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, &_mockEngineWrapper),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_mockEngineWrapper),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t engine = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          &_mockEngineBadTypeWrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          &_mockWrongTypeWrapper),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, SetAttrOnFinalizedEngineConfigDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    makeEngineConfigFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestEngineConfigDescriptor, FinalizeEngineConfigDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(engineConfig->finalize(), HIPDNN_STATUS_BAD_PARAM);

    makeEngineConfigFinalized();
}

TEST_F(TestEngineConfigDescriptor, GetAttrOnUnfinalizedEngineConfigDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    hipdnnBackendDescriptor_t dummyEngine = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &dummyEngine),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestEngineConfigDescriptor, GetEngineConfigDescriptorUnsupportedAttr)
{
    auto engineConfig = getEngineConfigDescriptor();
    hipdnnBackendDescriptor_t dummy = nullptr;

    makeEngineConfigFinalized();

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->getAttribute(HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          nullptr,
                                                          &dummy),
                               HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestEngineConfigDescriptor, GetEngineConfigDescriptorEngine)
{
    auto engineConfig = getEngineConfigDescriptor();
    ScopedDescriptor engine;
    ScopedDescriptor engine2;
    makeEngineConfigFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, nullptr, engine.getPtr()),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->getAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          2,
                                                          nullptr,
                                                          engine.getPtr()),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(engineConfig->getAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, engine.getPtr()));
    ASSERT_EQ(*engine.get(), *(_mockEngineWrapper.get()));

    int64_t count;
    ASSERT_NO_THROW(engineConfig->getAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, engine2.getPtr()));
    ASSERT_EQ(count, 1);
}

TEST_F(TestEngineConfigDescriptor, GetEngineThrowsIfNotFinalized)
{
    auto engineConfig = getEngineConfigDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(engineConfig->getEngine(), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestEngineConfigDescriptor, GetEngineReturnsPointerIfFinalized)
{
    auto engineConfig = getEngineConfigDescriptor();
    makeEngineConfigFinalized();
    auto enginePtr = engineConfig->getEngine();
    ASSERT_NE(enginePtr, nullptr);
    ASSERT_EQ(static_cast<const IBackendDescriptor*>(enginePtr.get()),
              static_cast<const IBackendDescriptor*>(getMockEngine().get()));
}

TEST_F(TestEngineConfigDescriptor, GetEngineDescriptorMaxWorkspaceSize)
{
    auto engineConfig = getEngineConfigDescriptor();
    int64_t workspaceSize = 0;

    makeEngineConfigFinalized();

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->getAttribute(HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          nullptr,
                                                          &workspaceSize),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 2, nullptr, &workspaceSize),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(engineConfig->getAttribute(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &workspaceSize));
    ASSERT_EQ(workspaceSize, 1024);

    int64_t count;
    ASSERT_NO_THROW(engineConfig->getAttribute(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &count, &workspaceSize));
    ASSERT_EQ(count, 1);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceWrongAttributeType)
{
    auto engineConfig = getEngineConfigDescriptor();
    hipdnnPluginConstData_t knobData = {nullptr, 0};

    // Wrong attribute type - expecting HIPDNN_TYPE_FLATBUFFER_PTR_EXT
    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT, HIPDNN_TYPE_INT64, 1, &knobData),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                   1,
                                   &knobData),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceWrongElementCount)
{
    auto engineConfig = getEngineConfigDescriptor();
    hipdnnPluginConstData_t knobData = {nullptr, 0};

    // Wrong element count - expecting 1
    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                   HIPDNN_TYPE_FLATBUFFER_PTR_EXT,
                                   2,
                                   &knobData),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceNullPointer)
{
    auto engineConfig = getEngineConfigDescriptor();

    // Null pointer for array of elements
    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                   HIPDNN_TYPE_FLATBUFFER_PTR_EXT,
                                   1,
                                   nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceNullData)
{
    auto engineConfig = getEngineConfigDescriptor();
    hipdnnPluginConstData_t knobData = {nullptr, 10}; // Null ptr but non-zero size

    // Serialized data is null
    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                   HIPDNN_TYPE_FLATBUFFER_PTR_EXT,
                                   1,
                                   &knobData),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceZeroSize)
{
    auto engineConfig = getEngineConfigDescriptor();
    uint8_t dummyData = 0;
    hipdnnPluginConstData_t knobData = {&dummyData, 0}; // Valid ptr but zero size

    // Serialized data size is zero
    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                   HIPDNN_TYPE_FLATBUFFER_PTR_EXT,
                                   1,
                                   &knobData),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceInvalidBuffer)
{
    auto engineConfig = getEngineConfigDescriptor();
    std::array<uint8_t, 10> invalidData
        = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    hipdnnPluginConstData_t knobData = {invalidData.data(), invalidData.size()};

    // Invalid KnobSetting flatbuffer
    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                   HIPDNN_TYPE_FLATBUFFER_PTR_EXT,
                                   1,
                                   &knobData),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceOnFinalizedDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    makeEngineConfigFinalized();

    // Create a valid knob setting buffer (even though we can't use it on finalized descriptor)
    flatbuffers::FlatBufferBuilder builder;
    auto intValue = hipdnn_data_sdk::data_objects::CreateIntValue(builder, 100);
    hipdnn_data_sdk::data_objects::KnobSettingBuilder settingBuilder(builder);
    settingBuilder.add_knob_id(42);
    settingBuilder.add_value_type(hipdnn_data_sdk::data_objects::KnobValue::IntValue);
    settingBuilder.add_value(intValue.Union());
    auto setting = settingBuilder.Finish();
    builder.Finish(setting);
    auto buffer = builder.Release();

    hipdnnPluginConstData_t knobData = {buffer.data(), buffer.size()};

    // Cannot set attributes on finalized descriptor
    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                   HIPDNN_TYPE_FLATBUFFER_PTR_EXT,
                                   1,
                                   &knobData),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceValid)
{
    auto engineConfig = getEngineConfigDescriptor();

    // Create a valid knob setting
    flatbuffers::FlatBufferBuilder builder;
    auto intValue = hipdnn_data_sdk::data_objects::CreateIntValue(builder, 100);
    hipdnn_data_sdk::data_objects::KnobSettingBuilder settingBuilder(builder);
    settingBuilder.add_knob_id(42);
    settingBuilder.add_value_type(hipdnn_data_sdk::data_objects::KnobValue::IntValue);
    settingBuilder.add_value(intValue.Union());
    auto setting = settingBuilder.Finish();
    builder.Finish(setting);
    auto buffer = builder.Release();

    hipdnnPluginConstData_t knobData = {buffer.data(), buffer.size()};

    // Should successfully set the knob choice
    ASSERT_NO_THROW(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                               HIPDNN_TYPE_FLATBUFFER_PTR_EXT,
                                               1,
                                               &knobData));

    // We should still be able to finalize after setting a knob choice
    setEngine();
    EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));
    EXPECT_CALL(*getMockEngine(), getGraph()).WillRepeatedly(Return(getMockGraphDescriptor()));
    EXPECT_CALL(*getMockGraphDescriptor(), getHandle()).WillOnce(Return(_mockHandle.get()));
    EXPECT_CALL(*_mockHandle, getPluginResourceManager())
        .WillOnce(Return(_mockEnginePluginResourceManager));
    EXPECT_CALL(*_mockEnginePluginResourceManager, getWorkspaceSize(_, _, _))
        .WillOnce(Return(1024));

    ASSERT_NO_THROW(engineConfig->finalize());
}

TEST_F(TestEngineConfigDescriptor, SetMultipleKnobChoices)
{
    auto engineConfig = getEngineConfigDescriptor();

    // Create first knob setting with knob_id 42
    flatbuffers::FlatBufferBuilder builder1;
    auto intValue1 = hipdnn_data_sdk::data_objects::CreateIntValue(builder1, 100);
    hipdnn_data_sdk::data_objects::KnobSettingBuilder settingBuilder1(builder1);
    settingBuilder1.add_knob_id(42);
    settingBuilder1.add_value_type(hipdnn_data_sdk::data_objects::KnobValue::IntValue);
    settingBuilder1.add_value(intValue1.Union());
    auto setting1 = settingBuilder1.Finish();
    builder1.Finish(setting1);
    auto buffer1 = builder1.Release();
    hipdnnPluginConstData_t knobData1 = {buffer1.data(), buffer1.size()};

    // Create second knob setting with knob_id 99
    flatbuffers::FlatBufferBuilder builder2;
    auto intValue2 = hipdnn_data_sdk::data_objects::CreateIntValue(builder2, 200);
    hipdnn_data_sdk::data_objects::KnobSettingBuilder settingBuilder2(builder2);
    settingBuilder2.add_knob_id(99);
    settingBuilder2.add_value_type(hipdnn_data_sdk::data_objects::KnobValue::IntValue);
    settingBuilder2.add_value(intValue2.Union());
    auto setting2 = settingBuilder2.Finish();
    builder2.Finish(setting2);
    auto buffer2 = builder2.Release();
    hipdnnPluginConstData_t knobData2 = {buffer2.data(), buffer2.size()};

    // Set both knob choices
    ASSERT_NO_THROW(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                               HIPDNN_TYPE_FLATBUFFER_PTR_EXT,
                                               1,
                                               &knobData1));
    ASSERT_NO_THROW(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                               HIPDNN_TYPE_FLATBUFFER_PTR_EXT,
                                               1,
                                               &knobData2));

    // Update the same knob_id (42) with a new value
    flatbuffers::FlatBufferBuilder builder3;
    auto intValue3 = hipdnn_data_sdk::data_objects::CreateIntValue(builder3, 300);
    hipdnn_data_sdk::data_objects::KnobSettingBuilder settingBuilder3(builder3);
    settingBuilder3.add_knob_id(42); // Same ID as first one
    settingBuilder3.add_value_type(hipdnn_data_sdk::data_objects::KnobValue::IntValue);
    settingBuilder3.add_value(intValue3.Union());
    auto setting3 = settingBuilder3.Finish();
    builder3.Finish(setting3);
    auto buffer3 = builder3.Release();
    hipdnnPluginConstData_t knobData3 = {buffer3.data(), buffer3.size()};

    // Should replace the existing knob setting with knob_id 42
    ASSERT_NO_THROW(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                               HIPDNN_TYPE_FLATBUFFER_PTR_EXT,
                                               1,
                                               &knobData3));
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceWithDifferentValueTypes)
{
    auto engineConfig = getEngineConfigDescriptor();

    // Create knob setting with string value
    flatbuffers::FlatBufferBuilder builder1;
    auto strOffset = builder1.CreateString("test_value");
    auto stringValue = hipdnn_data_sdk::data_objects::CreateStringValue(builder1, strOffset);
    hipdnn_data_sdk::data_objects::KnobSettingBuilder settingBuilder1(builder1);
    settingBuilder1.add_knob_id(1);
    settingBuilder1.add_value_type(hipdnn_data_sdk::data_objects::KnobValue::StringValue);
    settingBuilder1.add_value(stringValue.Union());
    auto setting1 = settingBuilder1.Finish();
    builder1.Finish(setting1);
    auto buffer1 = builder1.Release();
    hipdnnPluginConstData_t knobData1 = {buffer1.data(), buffer1.size()};

    // Create knob setting with float value
    flatbuffers::FlatBufferBuilder builder2;
    auto floatValue = hipdnn_data_sdk::data_objects::CreateFloatValue(builder2, 3.14159);
    hipdnn_data_sdk::data_objects::KnobSettingBuilder settingBuilder2(builder2);
    settingBuilder2.add_knob_id(2);
    settingBuilder2.add_value_type(hipdnn_data_sdk::data_objects::KnobValue::FloatValue);
    settingBuilder2.add_value(floatValue.Union());
    auto setting2 = settingBuilder2.Finish();
    builder2.Finish(setting2);
    auto buffer2 = builder2.Release();
    hipdnnPluginConstData_t knobData2 = {buffer2.data(), buffer2.size()};

    // Should handle different value types
    ASSERT_NO_THROW(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                               HIPDNN_TYPE_FLATBUFFER_PTR_EXT,
                                               1,
                                               &knobData1));
    ASSERT_NO_THROW(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                               HIPDNN_TYPE_FLATBUFFER_PTR_EXT,
                                               1,
                                               &knobData2));
}
