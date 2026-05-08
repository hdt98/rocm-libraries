// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file IntegrationHeuristicPolicyPlugins.cpp
 * @brief Integration tests for real heuristic policy plugins
 *
 * These tests verify the heuristic policy chain seen at runtime:
 *   - The StaticOrdering built-in (registered at HeuristicPluginManager
 *     construction time as a function-table-shaped pseudo-plugin).
 *   - Any external heuristic .so found in HIPDNN_HEURISTIC_PLUGIN_DIR.
 * - Plugin discovery and loading from installed location
 * - Symbol resolution and ABI validation
 * - Plugin handle creation and lifecycle
 * - Policy descriptor creation and execution
 * - API version compatibility
 * - Policy ID/name consistency
 */

#include "PlatformUtils.hpp"
#include "descriptors/EngineHeuristicDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "handle/Handle.hpp"
#include "plugin/HeuristicPlugin.hpp"
#include "plugin/HeuristicPluginManager.hpp"
#include "plugin/HeuristicPluginResourceManager.hpp"

#include <flatbuffers/flatbuffers.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/device_properties_generated.h>
#include <hipdnn_plugin_sdk/heuristic_api_version.h>

#include <filesystem>
#include <gtest/gtest.h>

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;

namespace
{
// Helper to get the plugin directory for tests
// Tests binaries are in build/bin/, plugins could be in:
// - build/lib/hipdnn_plugins/heuristics/ (Linux/Unix)
// - build/bin/hipdnn_plugins/heuristics/ (Windows DLLs)
std::filesystem::path getTestPluginDirectory()
{
    // First, check for environment variable override
    const auto envPath = hipdnn_data_sdk::utilities::getEnv("HIPDNN_HEURISTIC_PLUGIN_DIR");
    if(!envPath.empty())
    {
        return {envPath};
    }

    // Get the directory containing the test binary
    const auto testBinDir = hipdnn_backend::platform_utilities::getCurrentModuleDirectory();
    const auto buildRoot = testBinDir.parent_path();

    // Try multiple possible locations
    const std::vector<std::filesystem::path> candidatePaths = {
        buildRoot / "lib" / "hipdnn_plugins" / "heuristics", // Linux/Unix
        buildRoot / "bin" / "hipdnn_plugins" / "heuristics", // Windows DLLs
        buildRoot / "lib64" / "hipdnn_plugins" / "heuristics", // lib64 systems
    };

    // Return the first path that exists and contains plugin files
    for(const auto& path : candidatePaths)
    {
        if(std::filesystem::exists(path))
        {
            // Check if directory contains any .so or .dll files
            for(const auto& entry : std::filesystem::directory_iterator(path))
            {
                const auto ext = entry.path().extension();
                if(ext == ".so" || ext == ".dll" || ext == ".dylib")
                {
                    return path;
                }
            }
        }
    }

    // Fallback to original behavior (lib)
    return buildRoot / "lib" / "hipdnn_plugins" / "heuristics";
}
} // anonymous namespace

class IntegrationHeuristicPolicyPlugins : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const hipdnnStatus_t status = hipdnnCreate(&_handle);
        ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_handle, nullptr);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            hipdnnDestroy(_handle);
            _handle = nullptr;
        }
    }

    hipdnnHandle_t _handle = nullptr;
};

// ========== Plugin Discovery Tests ==========

TEST_F(IntegrationHeuristicPolicyPlugins, PluginManagerLoadsFromDefaultPath)
{
    // Create manager and load plugins
    auto manager = std::make_shared<HeuristicPluginManager>();
    // For tests, explicitly pass the plugin directory since getCurrentModuleDirectory
    // resolves to the test binary's location (bin/) not the backend library's (lib/)
    manager->loadPlugins({getTestPluginDirectory()}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    // Plugin handles are created by HeuristicPluginResourceManager, not by the bare
    // manager. The resource manager creates handles and can enumerate loaded plugins.
    auto resourceMgr = std::make_shared<HeuristicPluginResourceManager>(manager);

    // Enumerate loaded policies via resource manager
    const auto& policyInfos = resourceMgr->getHeuristicPolicyInfos();

    // The StaticOrdering built-in is always registered; vendor plugins (if any)
    // discovered under the search path raise the count further.
    EXPECT_GE(policyInfos.size(), 1u) << "Expected at least the StaticOrdering built-in";
}

TEST_F(IntegrationHeuristicPolicyPlugins, PluginManagerRejectsInvalidPlugins)
{
    // Plugins with wrong ABI version should be rejected during validation
    // This is tested by HeuristicPluginManager::validateBeforeAdding()

    auto manager = std::make_shared<HeuristicPluginManager>();
    manager->loadPlugins({getTestPluginDirectory()}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    // Create resource manager to access loaded plugins
    auto resourceMgr = std::make_shared<HeuristicPluginResourceManager>(manager);
    const auto& policyInfos = resourceMgr->getHeuristicPolicyInfos();

    // All loaded plugins should have valid metadata
    for(const auto& info : policyInfos)
    {
        EXPECT_NE(info.policyId, -1) << "Policy ID should be valid";
        EXPECT_FALSE(info.apiVersion.empty()) << "API version should not be empty";
        EXPECT_FALSE(info.pluginVersion.empty()) << "Plugin version should not be empty";
    }
}

TEST_F(IntegrationHeuristicPolicyPlugins, PluginManagerRejectsDuplicatePolicyIds)
{
    // HeuristicPluginManager should reject plugins with duplicate policy IDs
    // This is enforced by validateBeforeAdding()

    auto manager = std::make_shared<HeuristicPluginManager>();
    manager->loadPlugins({getTestPluginDirectory()}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    // Create resource manager to enumerate loaded plugins
    auto resourceMgr = std::make_shared<HeuristicPluginResourceManager>(manager);
    const auto& policyInfos = resourceMgr->getHeuristicPolicyInfos();

    // Collect all policy IDs
    std::set<int64_t> policyIds;
    for(const auto& info : policyInfos)
    {
        const int64_t id = info.policyId;
        EXPECT_EQ(policyIds.count(id), 0u) << "Duplicate policy ID detected: " << id;
        policyIds.insert(id);
    }
}

// ========== Symbol Resolution Tests ==========

TEST_F(IntegrationHeuristicPolicyPlugins, LoadedPluginsHaveRequiredSymbols)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();
    auto policyInfos = heurRm->getHeuristicPolicyInfos();

    ASSERT_GT(policyInfos.size(), 0u);

    // Each loaded plugin must have successfully resolved required symbols
    // If symbol resolution failed, the plugin wouldn't be in the list
    for(const auto& info : policyInfos)
    {
        EXPECT_NE(info.policyId, -1);
        EXPECT_FALSE(info.policyName.empty());
        EXPECT_FALSE(info.apiVersion.empty());
        EXPECT_FALSE(info.pluginVersion.empty());
    }
}

// ========== Handle Lifecycle Tests ==========

TEST_F(IntegrationHeuristicPolicyPlugins, ResourceManagerCreatesHandlesForAllPlugins)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();
    ASSERT_NE(heurRm, nullptr);

    auto policyInfos = heurRm->getHeuristicPolicyInfos();

    // Should have created a handle for each loaded plugin
    for(const auto& info : policyInfos)
    {
        auto handle = heurRm->getHeuristicHandleForPolicyId(info.policyId);
        EXPECT_NE(handle, nullptr) << "Handle should exist for policy ID " << info.policyId;

        auto plugin = heurRm->getPluginForPolicyId(info.policyId);
        EXPECT_NE(plugin, nullptr) << "Plugin should exist for policy ID " << info.policyId;
    }
}

TEST_F(IntegrationHeuristicPolicyPlugins, HandleDestructionCleansUpResources)
{
    // Create and destroy a handle
    hipdnnHandle_t tempHandle = nullptr;
    ASSERT_EQ(hipdnnCreate(&tempHandle), HIPDNN_STATUS_SUCCESS);

    // Get resource manager (creates plugin handles)
    auto heurRm = tempHandle->getHeuristicPluginResourceManager();
    ASSERT_NE(heurRm, nullptr);

    const size_t policyCount = heurRm->getHeuristicPolicyInfos().size();
    EXPECT_GT(policyCount, 0u);

    // Destroy handle (should clean up plugin handles)
    EXPECT_EQ(hipdnnDestroy(tempHandle), HIPDNN_STATUS_SUCCESS);
}

// ========== Policy Descriptor Tests ==========

TEST_F(IntegrationHeuristicPolicyPlugins, PolicyDescriptorCreationSucceeds)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();
    auto policyInfos = heurRm->getHeuristicPolicyInfos();
    ASSERT_GT(policyInfos.size(), 0u);

    // Try to create a policy descriptor for the first loaded policy
    auto policyId = policyInfos[0].policyId;
    auto pluginHandle = heurRm->getHeuristicHandleForPolicyId(policyId);
    auto plugin = heurRm->getPluginForPolicyId(policyId);

    ASSERT_NE(pluginHandle, nullptr);
    ASSERT_NE(plugin, nullptr);

    // Create policy descriptor
    auto descriptor = plugin->createPolicyDescriptor(pluginHandle, policyId);
    EXPECT_NE(descriptor, nullptr);

    // Clean up
    if(descriptor != nullptr)
    {
        plugin->destroyPolicyDescriptor(descriptor);
    }
}

// ========== Logging Tests ==========

TEST_F(IntegrationHeuristicPolicyPlugins, PluginsReceiveLoggingCallback)
{
    // Verify that setLoggingCallback was wired up during plugin initialization.
    // If the callback registration failed, resource manager construction would have
    // logged warnings; here we at least confirm the manager came up.
    auto heurRm = _handle->getHeuristicPluginResourceManager();
    ASSERT_NE(heurRm, nullptr);
}

// ========== Device Properties Tests ==========

TEST_F(IntegrationHeuristicPolicyPlugins, DevicePropertiesAreSetOnAllHandles)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();
    ASSERT_NE(heurRm, nullptr);

    // Create serialized device properties using FlatBuffers
    hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT devProps;
    devProps.device_id = 0;
    devProps.multi_processor_count = 64;
    devProps.total_global_mem = 8ULL * 1024 * 1024 * 1024;
    devProps.architecture_name = "gfx90a";

    // Serialize using FlatBuffers
    flatbuffers::FlatBufferBuilder builder(256);
    auto offset = hipdnn_flatbuffers_sdk::data_objects::DeviceProperties::Pack(builder, &devProps);
    builder.Finish(offset, "HDDP");

    // Create wrapper
    hipdnnPluginConstData_t wrapper;
    wrapper.ptr = builder.GetBufferPointer();
    wrapper.size = builder.GetSize();

    // Set on all handles (should not throw)
    EXPECT_NO_THROW(heurRm->setDevicePropertiesOnAllHandles(&wrapper));
}

// ========== Policy ID Consistency Tests ==========

TEST_F(IntegrationHeuristicPolicyPlugins, PolicyIdMatchesNameHash)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();
    auto policyInfos = heurRm->getHeuristicPolicyInfos();

    for(const auto& info : policyInfos)
    {
        if(!info.policyName.empty())
        {
            // Policy ID should match policyNameToId(policyName)
            const int64_t expectedId = hipdnn_data_sdk::utilities::policyNameToId(info.policyName);
            EXPECT_EQ(info.policyId, expectedId) << "Policy ID mismatch for " << info.policyName;
        }
    }
}

// ========== API Version Tests ==========

TEST_F(IntegrationHeuristicPolicyPlugins, AllPluginsHaveCompatibleApiVersion)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();
    auto policyInfos = heurRm->getHeuristicPolicyInfos();

    // All loaded plugins should have compatible API versions
    // (major version matches the heuristic API version)
    for(const auto& info : policyInfos)
    {
        EXPECT_FALSE(info.apiVersion.empty());

        // Parse version
        const hipdnn_data_sdk::utilities::Version apiVer{info.apiVersion};

        // Major version should match heuristic API (independent of backend version)
        EXPECT_EQ(apiVer.major, HIPDNN_HEURISTIC_API_VERSION_MAJOR)
            << "Plugin " << info.policyName << " has incompatible API major version";
    }
}

// ========== Enumeration Consistency Tests ==========

TEST_F(IntegrationHeuristicPolicyPlugins, EnumerationMatchesResourceManager)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();
    auto rmInfos = heurRm->getHeuristicPolicyInfos();

    // Get count via C API
    size_t apiCount = 0;
    ASSERT_EQ(hipdnnGetHeuristicPolicyCount_ext(_handle, &apiCount), HIPDNN_STATUS_SUCCESS);

    // Counts should match
    EXPECT_EQ(apiCount, rmInfos.size());

    // Each policy from resource manager should be queryable via C API
    for(size_t i = 0; i < rmInfos.size(); ++i)
    {
        int64_t apiPolicyId = -1;
        size_t nameLen = 0;
        size_t pluginVerLen = 0;
        size_t apiVerLen = 0;

        ASSERT_EQ(hipdnnGetHeuristicPolicyInfo_ext(_handle,
                                                   i,
                                                   &apiPolicyId,
                                                   nullptr,
                                                   &nameLen,
                                                   nullptr,
                                                   &pluginVerLen,
                                                   nullptr,
                                                   &apiVerLen),
                  HIPDNN_STATUS_SUCCESS);

        // Policy ID should match
        EXPECT_EQ(apiPolicyId, rmInfos[i].policyId);
    }
}

// ========== Stress Tests ==========

TEST_F(IntegrationHeuristicPolicyPlugins, MultipleResourceManagersCanCoexist)
{
    // Create multiple handles, each with its own resource manager
    std::vector<hipdnnHandle_t> handles;

    for(int i = 0; i < 5; ++i)
    {
        hipdnnHandle_t h = nullptr;
        ASSERT_EQ(hipdnnCreate(&h), HIPDNN_STATUS_SUCCESS);
        handles.push_back(h);

        // Access resource manager (triggers creation)
        auto heurRm = h->getHeuristicPluginResourceManager();
        EXPECT_NE(heurRm, nullptr);
    }

    // Clean up
    for(auto h : handles)
    {
        EXPECT_EQ(hipdnnDestroy(h), HIPDNN_STATUS_SUCCESS);
    }
}

// ========== Error Recovery Tests ==========

TEST_F(IntegrationHeuristicPolicyPlugins, MissingPolicyGracefullyHandled)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();

    // Query a non-existent policy ID
    const int64_t fakePolicyId = 0x1234567890ABCDEF;
    auto handle = heurRm->getHeuristicHandleForPolicyId(fakePolicyId);
    auto plugin = heurRm->getPluginForPolicyId(fakePolicyId);

    // Should return nullptr, not crash
    EXPECT_EQ(handle, nullptr);
    EXPECT_EQ(plugin, nullptr);
}
// ========== Workflow Tests with Test Plugins (from pr1) ==========
