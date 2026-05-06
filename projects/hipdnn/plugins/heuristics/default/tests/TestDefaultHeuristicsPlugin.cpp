// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestDefaultHeuristicsPlugin.cpp
 * @brief Plugin-level tests for the merged default heuristics plugin.
 *
 * Covers metadata, logging, handle lifecycle, and the policy-enumeration entry
 * points (GetAllPolicyIds, GetPolicyName) that are shared across the bundled
 * Config and StaticOrdering policies.
 *
 * Per-policy descriptor behaviors live in TestConfigPolicy.cpp and
 * TestStaticOrderingPolicy.cpp.
 */

// HIPDNN_PLUGIN_STATIC_DEFINE / HIPDNN_HEURISTIC_PLUGIN_STATIC_DEFINE come
// from target_compile_definitions in this test's CMakeLists.txt.

#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <string_view>
#include <vector>

namespace
{

const int64_t CONFIG_POLICY_ID
    = hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::Config");
const int64_t STATIC_ORDERING_POLICY_ID
    = hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering");

} // namespace

// ========== Plugin Metadata ==========

TEST(TestDefaultHeuristicsPluginMetadata, GetNameReturnsExpectedPluginName)
{
    const char* name = nullptr;
    EXPECT_EQ(hipdnnPluginGetName(&name), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(name, "DefaultHeuristicsPlugin");
}

TEST(TestDefaultHeuristicsPluginMetadata, GetNameRejectsNull)
{
    EXPECT_EQ(hipdnnPluginGetName(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestDefaultHeuristicsPluginMetadata, GetVersionReturnsNonEmpty)
{
    const char* version = nullptr;
    EXPECT_EQ(hipdnnPluginGetVersion(&version), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(version, nullptr);
    EXPECT_GT(std::string_view(version).size(), 0u);
}

TEST(TestDefaultHeuristicsPluginMetadata, GetVersionRejectsNull)
{
    EXPECT_EQ(hipdnnPluginGetVersion(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestDefaultHeuristicsPluginMetadata, GetApiVersionReturnsNonEmpty)
{
    const char* version = nullptr;
    EXPECT_EQ(hipdnnPluginGetApiVersion(&version), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(version, nullptr);
    EXPECT_GT(std::string_view(version).size(), 0u);
}

TEST(TestDefaultHeuristicsPluginMetadata, GetApiVersionRejectsNull)
{
    EXPECT_EQ(hipdnnPluginGetApiVersion(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestDefaultHeuristicsPluginMetadata, GetTypeReturnsHeuristic)
{
    hipdnnPluginType_t type = HIPDNN_PLUGIN_TYPE_UNSPECIFIED;
    EXPECT_EQ(hipdnnPluginGetType(&type), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(type, HIPDNN_PLUGIN_TYPE_HEURISTIC);
}

TEST(TestDefaultHeuristicsPluginMetadata, GetTypeRejectsNull)
{
    EXPECT_EQ(hipdnnPluginGetType(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== Logging ==========

TEST(TestDefaultHeuristicsPluginLogging, SetLoggingCallbackSucceedsWithNullCallback)
{
    EXPECT_EQ(hipdnnPluginSetLoggingCallback(nullptr), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestDefaultHeuristicsPluginLogging, SetLogLevelSucceeds)
{
    EXPECT_EQ(hipdnnPluginSetLogLevel(HIPDNN_SEV_WARN), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestDefaultHeuristicsPluginLogging, GetLastErrorStringWithNullDoesNotCrash)
{
    EXPECT_NO_FATAL_FAILURE(hipdnnPluginGetLastErrorString(nullptr));
}

TEST(TestDefaultHeuristicsPluginLogging, GetLastErrorStringReturnsNonNullMessage)
{
    const char* msg = nullptr;
    hipdnnPluginGetLastErrorString(&msg);
    EXPECT_NE(msg, nullptr);
}

// ========== Handle Lifecycle ==========

TEST(TestDefaultHeuristicsPluginHandle, CreateAndDestroySucceed)
{
    hipdnnHeuristicHandle_t handle = nullptr;
    EXPECT_EQ(hipdnnHeuristicHandleCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_NE(handle, nullptr);
    EXPECT_EQ(hipdnnHeuristicHandleDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestDefaultHeuristicsPluginHandle, CreateRejectsNullOutPointer)
{
    EXPECT_EQ(hipdnnHeuristicHandleCreate(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestDefaultHeuristicsPluginHandle, DestroyRejectsNullHandle)
{
    EXPECT_EQ(hipdnnHeuristicHandleDestroy(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestDefaultHeuristicsPluginHandle, SetDevicePropertiesAcceptsNonEmptyBuffer)
{
    hipdnnHeuristicHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnHeuristicHandleCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
    const std::array<uint8_t, 4> buffer{0xDE, 0xAD, 0xBE, 0xEF};
    const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
    EXPECT_EQ(hipdnnHeuristicHandleSetDeviceProperties(handle, &data),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnHeuristicHandleDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestDefaultHeuristicsPluginHandle, SetDevicePropertiesRejectsNullHandle)
{
    const std::array<uint8_t, 1> buffer{0x01};
    const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
    EXPECT_EQ(hipdnnHeuristicHandleSetDeviceProperties(nullptr, &data),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestDefaultHeuristicsPluginHandle, SetDevicePropertiesRejectsNullBufferStruct)
{
    hipdnnHeuristicHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnHeuristicHandleCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnHeuristicHandleSetDeviceProperties(handle, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(hipdnnHeuristicHandleDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestDefaultHeuristicsPluginHandle, SetDevicePropertiesRejectsNullBufferPointer)
{
    hipdnnHeuristicHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnHeuristicHandleCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
    const hipdnnPluginConstData_t data{nullptr, 4};
    EXPECT_EQ(hipdnnHeuristicHandleSetDeviceProperties(handle, &data),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(hipdnnHeuristicHandleDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestDefaultHeuristicsPluginHandle, SetDevicePropertiesRejectsZeroSize)
{
    hipdnnHeuristicHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnHeuristicHandleCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
    const std::array<uint8_t, 1> buffer{0x01};
    const hipdnnPluginConstData_t data{buffer.data(), 0};
    EXPECT_EQ(hipdnnHeuristicHandleSetDeviceProperties(handle, &data),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(hipdnnHeuristicHandleDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

// ========== Policy Enumeration ==========

TEST(TestDefaultHeuristicsPluginPolicyIds, ReturnsTwoPoliciesInExpectedOrder)
{
    uint32_t numPolicies = 0;
    std::array<int64_t, 4> ids{0, 0, 0, 0};
    EXPECT_EQ(hipdnnHeuristicPluginGetAllPolicyIds(ids.data(),
                                                   static_cast<uint32_t>(ids.size()),
                                                   &numPolicies),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(numPolicies, 2u);
    // Order matters for the framework: Config runs first, StaticOrdering second.
    EXPECT_EQ(ids[0], CONFIG_POLICY_ID);
    EXPECT_EQ(ids[1], STATIC_ORDERING_POLICY_ID);
}

TEST(TestDefaultHeuristicsPluginPolicyIds, QueryCountWithNullArrayReturnsCountOnly)
{
    uint32_t numPolicies = 0;
    EXPECT_EQ(hipdnnHeuristicPluginGetAllPolicyIds(nullptr, 0, &numPolicies),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(numPolicies, 2u);
}

TEST(TestDefaultHeuristicsPluginPolicyIds, RejectsNullCountPointer)
{
    std::array<int64_t, 2> ids{0, 0};
    EXPECT_EQ(hipdnnHeuristicPluginGetAllPolicyIds(
                  ids.data(), static_cast<uint32_t>(ids.size()), nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestDefaultHeuristicsPluginPolicyIds, RejectsBufferSmallerThanPolicyCount)
{
    uint32_t numPolicies = 0;
    std::array<int64_t, 1> ids{0};
    EXPECT_EQ(hipdnnHeuristicPluginGetAllPolicyIds(
                  ids.data(), static_cast<uint32_t>(ids.size()), &numPolicies),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(numPolicies, 2u);
}

TEST(TestDefaultHeuristicsPluginPolicyName, ReturnsConfigName)
{
    const char* name = nullptr;
    EXPECT_EQ(hipdnnHeuristicPluginGetPolicyName(CONFIG_POLICY_ID, &name),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(name, "SelectionHeuristic::Config");
}

TEST(TestDefaultHeuristicsPluginPolicyName, ReturnsStaticOrderingName)
{
    const char* name = nullptr;
    EXPECT_EQ(hipdnnHeuristicPluginGetPolicyName(STATIC_ORDERING_POLICY_ID, &name),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(name, "SelectionHeuristic::StaticOrdering");
}

TEST(TestDefaultHeuristicsPluginPolicyName, RejectsUnknownPolicyId)
{
    const char* name = nullptr;
    EXPECT_EQ(hipdnnHeuristicPluginGetPolicyName(static_cast<int64_t>(0x1122334455667788ULL), &name),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestDefaultHeuristicsPluginPolicyName, RejectsNullOutPointer)
{
    EXPECT_EQ(hipdnnHeuristicPluginGetPolicyName(CONFIG_POLICY_ID, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== Cross-policy descriptor lifecycle ==========

TEST(TestDefaultHeuristicsPluginPolicyDescriptor, RejectsUnknownPolicyId)
{
    hipdnnHeuristicHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnHeuristicHandleCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    EXPECT_EQ(hipdnnHeuristicPolicyDescriptorCreate(
                  handle, static_cast<int64_t>(0x9988776655443322ULL), &desc),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(desc, nullptr);
    EXPECT_EQ(hipdnnHeuristicHandleDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestDefaultHeuristicsPluginPolicyDescriptor, OneHandleSupportsBothPolicyDescriptors)
{
    // The plugin is 1:N - one handle can construct descriptors for every
    // policy it advertises, concurrently.
    hipdnnHeuristicHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnHeuristicHandleCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    hipdnnHeuristicPolicyDescriptor_t configDesc = nullptr;
    hipdnnHeuristicPolicyDescriptor_t staticDesc = nullptr;
    EXPECT_EQ(hipdnnHeuristicPolicyDescriptorCreate(handle, CONFIG_POLICY_ID, &configDesc),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnHeuristicPolicyDescriptorCreate(handle, STATIC_ORDERING_POLICY_ID, &staticDesc),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_NE(configDesc, nullptr);
    EXPECT_NE(staticDesc, nullptr);
    EXPECT_NE(configDesc, staticDesc);

    EXPECT_EQ(hipdnnHeuristicPolicyDescriptorDestroy(configDesc), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnHeuristicPolicyDescriptorDestroy(staticDesc), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnHeuristicHandleDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}
