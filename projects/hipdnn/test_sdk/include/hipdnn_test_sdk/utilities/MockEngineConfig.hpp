// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_data_sdk/flatbuffer_utilities/KnobSettingWrapper.hpp>

namespace hipdnn_test_sdk::utilities
{

class MockEngineConfig : public hipdnn_plugin_sdk::IEngineConfig
{
public:
    MOCK_METHOD(const hipdnn_data_sdk::data_objects::EngineConfig&,
                getEngineConfig,
                (),
                (const, override));
    MOCK_METHOD(bool, isValid, (), (const, override));
    MOCK_METHOD(int64_t, engineId, (), (const, override));
    MOCK_METHOD(uint32_t, knobSettingCount, (), (const, override));
    MOCK_METHOD(const hipdnn_data_sdk::data_objects::KnobSetting&,
                getKnobSetting,
                (uint32_t),
                (const, override));
    MOCK_METHOD(const hipdnn_plugin_sdk::IKnobSetting&,
                getKnobSettingWrapper,
                (uint32_t),
                (const, override));
    MOCK_METHOD((const std::vector<std::unique_ptr<hipdnn_plugin_sdk::IKnobSetting>>&),
                knobSettingWrappers,
                (),
                (const, override));
};

}
