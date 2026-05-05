// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestStaticOrderingPlugin.cpp
 * @brief Unit tests for the StaticOrdering heuristic plugin C ABI (RFC 0007)
 *
 * Links the plugin's _private STATIC archive directly so the C ABI entry points
 * can be exercised without dlopen'ing the SHARED plugin. Covers handle and
 * descriptor lifecycle, BAD_PARAM branches on every entry point, the
 * applied=0 empty-list path, and the GetSortedEngineIds two-call query and
 * clipping behaviors.
 *
 * The ordering algorithm itself (utilities::sortEngineIds) is exercised by
 * backend/tests/utilities/TestEngineOrdering.cpp; this file focuses on the
 * plugin's wrapper logic.
 */

// HIPDNN_PLUGIN_STATIC_DEFINE / HIPDNN_HEURISTIC_PLUGIN_STATIC_DEFINE come
// from target_compile_definitions in this test's CMakeLists.txt.

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

namespace
{

const int64_t MIOPEN_ENGINE_ID = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE");
const int64_t MIOPEN_DETERMINISTIC_ID
    = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE_DETERMINISTIC");
const int64_t CUSTOM_ENGINE_ID
    = hipdnn_data_sdk::utilities::engineNameToId("Plugin1::CustomEngine");

class TestStaticOrderingPlugin : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hipdnnHeuristicHandleCreate(&_handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
        ASSERT_NE(_handle, nullptr);
        ASSERT_EQ(hipdnnHeuristicPolicyDescriptorCreate(_handle, &_desc),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
        ASSERT_NE(_desc, nullptr);
    }

    void TearDown() override
    {
        if(_desc != nullptr)
        {
            EXPECT_EQ(hipdnnHeuristicPolicyDescriptorDestroy(_desc), HIPDNN_PLUGIN_STATUS_SUCCESS);
        }
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnHeuristicHandleDestroy(_handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
        }
    }

    hipdnnHeuristicHandle_t _handle = nullptr;
    hipdnnHeuristicPolicyDescriptor_t _desc = nullptr;
};

} // namespace

// ========== Plugin Metadata ==========

TEST(TestStaticOrderingPluginMetadata, GetNameReturnsExpectedPolicyName)
{
    const char* name = nullptr;
    EXPECT_EQ(hipdnnPluginGetName(&name), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(name, "SelectionHeuristic::StaticOrdering");
}

TEST(TestStaticOrderingPluginMetadata, GetNameRejectsNull)
{
    EXPECT_EQ(hipdnnPluginGetName(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestStaticOrderingPluginMetadata, GetVersionReturnsNonEmpty)
{
    const char* version = nullptr;
    EXPECT_EQ(hipdnnPluginGetVersion(&version), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(version, nullptr);
    EXPECT_GT(std::string_view(version).size(), 0u);
}

TEST(TestStaticOrderingPluginMetadata, GetVersionRejectsNull)
{
    EXPECT_EQ(hipdnnPluginGetVersion(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestStaticOrderingPluginMetadata, GetApiVersionReturnsNonEmpty)
{
    const char* version = nullptr;
    EXPECT_EQ(hipdnnPluginGetApiVersion(&version), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(version, nullptr);
    EXPECT_GT(std::string_view(version).size(), 0u);
}

TEST(TestStaticOrderingPluginMetadata, GetApiVersionRejectsNull)
{
    EXPECT_EQ(hipdnnPluginGetApiVersion(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestStaticOrderingPluginMetadata, GetTypeReturnsHeuristic)
{
    hipdnnPluginType_t type = HIPDNN_PLUGIN_TYPE_UNSPECIFIED;
    EXPECT_EQ(hipdnnPluginGetType(&type), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(type, HIPDNN_PLUGIN_TYPE_HEURISTIC);
}

TEST(TestStaticOrderingPluginMetadata, GetTypeRejectsNull)
{
    EXPECT_EQ(hipdnnPluginGetType(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== Logging ==========

TEST(TestStaticOrderingPluginLogging, SetLoggingCallbackSucceedsWithNullCallback)
{
    EXPECT_EQ(hipdnnPluginSetLoggingCallback(nullptr), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestStaticOrderingPluginLogging, SetLogLevelSucceeds)
{
    EXPECT_EQ(hipdnnPluginSetLogLevel(HIPDNN_SEV_WARN), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestStaticOrderingPluginLogging, GetLastErrorStringWithNullDoesNotCrash)
{
    EXPECT_NO_FATAL_FAILURE(hipdnnPluginGetLastErrorString(nullptr));
}

TEST(TestStaticOrderingPluginLogging, GetLastErrorStringReturnsNonNullMessage)
{
    const char* msg = nullptr;
    hipdnnPluginGetLastErrorString(&msg);
    EXPECT_NE(msg, nullptr);
}

// ========== Handle Lifecycle ==========

TEST(TestStaticOrderingPluginHandle, CreateAndDestroySucceed)
{
    hipdnnHeuristicHandle_t handle = nullptr;
    EXPECT_EQ(hipdnnHeuristicHandleCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_NE(handle, nullptr);
    EXPECT_EQ(hipdnnHeuristicHandleDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestStaticOrderingPluginHandle, CreateRejectsNullOutPointer)
{
    EXPECT_EQ(hipdnnHeuristicHandleCreate(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestStaticOrderingPluginHandle, DestroyRejectsNullHandle)
{
    EXPECT_EQ(hipdnnHeuristicHandleDestroy(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingPlugin, SetDevicePropertiesAcceptsNonEmptyBuffer)
{
    const std::array<uint8_t, 4> buffer{0xDE, 0xAD, 0xBE, 0xEF};
    const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
    EXPECT_EQ(hipdnnHeuristicHandleSetDeviceProperties(_handle, &data),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST_F(TestStaticOrderingPlugin, SetDevicePropertiesRejectsNullHandle)
{
    const std::array<uint8_t, 1> buffer{0x01};
    const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
    EXPECT_EQ(hipdnnHeuristicHandleSetDeviceProperties(nullptr, &data),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingPlugin, SetDevicePropertiesRejectsNullBufferStruct)
{
    EXPECT_EQ(hipdnnHeuristicHandleSetDeviceProperties(_handle, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingPlugin, SetDevicePropertiesRejectsNullBufferPointer)
{
    const hipdnnPluginConstData_t data{nullptr, 4};
    EXPECT_EQ(hipdnnHeuristicHandleSetDeviceProperties(_handle, &data),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingPlugin, SetDevicePropertiesRejectsZeroSize)
{
    const std::array<uint8_t, 1> buffer{0x01};
    const hipdnnPluginConstData_t data{buffer.data(), 0};
    EXPECT_EQ(hipdnnHeuristicHandleSetDeviceProperties(_handle, &data),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== Policy Descriptor Lifecycle ==========

TEST_F(TestStaticOrderingPlugin, DescriptorCreateRejectsNullHandle)
{
    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    EXPECT_EQ(hipdnnHeuristicPolicyDescriptorCreate(nullptr, &desc),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(desc, nullptr);
}

TEST_F(TestStaticOrderingPlugin, DescriptorCreateRejectsNullOutPointer)
{
    EXPECT_EQ(hipdnnHeuristicPolicyDescriptorCreate(_handle, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestStaticOrderingPluginDescriptor, DestroyRejectsNullDescriptor)
{
    EXPECT_EQ(hipdnnHeuristicPolicyDescriptorDestroy(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== SetEngineIds ==========

TEST_F(TestStaticOrderingPlugin, SetEngineIdsAcceptsValidIds)
{
    const std::array<int64_t, 2> ids{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID};
    EXPECT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, ids.data(), ids.size()),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST_F(TestStaticOrderingPlugin, SetEngineIdsAcceptsZeroCountWithNullPointer)
{
    EXPECT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, nullptr, 0),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestStaticOrderingPluginInputs, SetEngineIdsRejectsNullDescriptor)
{
    const std::array<int64_t, 1> ids{MIOPEN_ENGINE_ID};
    EXPECT_EQ(hipdnnHeuristicPolicySetEngineIds(nullptr, ids.data(), ids.size()),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingPlugin, SetEngineIdsRejectsNullPointerWithCount)
{
    EXPECT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, nullptr, 3),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== SetSerializedGraph ==========

// StaticOrdering ignores the graph contents but validates the parameter shape.

TEST_F(TestStaticOrderingPlugin, SetSerializedGraphAcceptsAnyBuffer)
{
    const std::array<uint8_t, 3> buffer{0x01, 0x02, 0x03};
    const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
    EXPECT_EQ(hipdnnHeuristicPolicySetSerializedGraph(_desc, &data),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestStaticOrderingPluginInputs, SetSerializedGraphRejectsNullDescriptor)
{
    const std::array<uint8_t, 1> buffer{0x00};
    const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
    EXPECT_EQ(hipdnnHeuristicPolicySetSerializedGraph(nullptr, &data),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingPlugin, SetSerializedGraphRejectsNullBufferStruct)
{
    EXPECT_EQ(hipdnnHeuristicPolicySetSerializedGraph(_desc, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== Finalize ==========

TEST(TestStaticOrderingPluginFinalize, RejectsNullDescriptor)
{
    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(nullptr, &applied), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingPlugin, FinalizeRejectsNullOutApplied)
{
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingPlugin, FinalizeWithNoCandidatesReportsNotApplied)
{
    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 0);
}

TEST_F(TestStaticOrderingPlugin, FinalizeSortsCandidatesAndReportsApplied)
{
    const std::array<int64_t, 3> ids{MIOPEN_DETERMINISTIC_ID, CUSTOM_ENGINE_ID, MIOPEN_ENGINE_ID};
    ASSERT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, ids.data(), ids.size()),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 1);

    size_t count = 0;
    ASSERT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(count, ids.size());

    std::vector<int64_t> sorted(count);
    ASSERT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, sorted.data(), &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(sorted.front(), MIOPEN_ENGINE_ID);
    EXPECT_EQ(sorted.back(), MIOPEN_DETERMINISTIC_ID);
}

TEST_F(TestStaticOrderingPlugin, FinalizeResetsByLastSetEngineIdsCall)
{
    // First call with one set, then re-set with empty: should report not-applied
    const std::array<int64_t, 1> ids{MIOPEN_ENGINE_ID};
    ASSERT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, ids.data(), ids.size()),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, nullptr, 0),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 0);
}

// ========== GetSortedEngineIds ==========

TEST(TestStaticOrderingPluginGet, RejectsNullDescriptor)
{
    size_t count = 0;
    EXPECT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(nullptr, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingPlugin, GetRejectsNullCountPointer)
{
    EXPECT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, nullptr, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingPlugin, GetReturnsNotInitializedBeforeFinalize)
{
    size_t count = 0;
    EXPECT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestStaticOrderingPlugin, GetClipsToCallerProvidedCapacity)
{
    const std::array<int64_t, 3> ids{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID};
    ASSERT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, ids.data(), ids.size()),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    int32_t applied = -1;
    ASSERT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(applied, 1);

    // Caller buffer is smaller than result: should clip and report the clipped count.
    std::array<int64_t, 2> outBuf{0, 0};
    size_t count = outBuf.size();
    EXPECT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, outBuf.data(), &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(count, outBuf.size());
    EXPECT_EQ(outBuf[0], MIOPEN_ENGINE_ID);
    EXPECT_EQ(outBuf[1], CUSTOM_ENGINE_ID);
}
