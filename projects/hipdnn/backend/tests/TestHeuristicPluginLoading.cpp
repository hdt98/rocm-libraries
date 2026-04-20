// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestHeuristicPluginLoading.cpp
 * @brief Integration tests for loading real heuristic plugins
 *
 * These tests load actual .so/.dll plugin files to verify:
 * - Symbol resolution (resolveSymbols)
 * - Constructor with SharedLibrary
 * - Error handling for missing/incomplete symbols
 * - Optional symbol handling
 */

#include "HipdnnException.hpp"
#include "plugin/HeuristicPlugin.hpp"
#include "plugin/SharedLibrary.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;

namespace
{
// Helper to get test plugin path
std::filesystem::path getTestPluginPath(const std::string& pluginName)
{
    // The test plugins are in CMAKE_BINARY_DIR/lib/test_plugins/custom/
    // When run via CTest, working directory is build/backend/tests, so go up two levels
    // When run directly from build/, use current directory
    auto currentPath = std::filesystem::current_path();
    auto pluginDir = currentPath / "lib" / "test_plugins" / "custom";

    // If lib doesn't exist here, try going up directories (for CTest execution)
    if(!std::filesystem::exists(pluginDir))
    {
        pluginDir = currentPath / ".." / ".." / "lib" / "test_plugins" / "custom";
    }

#ifdef _WIN32
    const auto pluginPath = pluginDir / (pluginName + ".dll");
#else
    const auto pluginPath = pluginDir / ("lib" + pluginName + ".so");
#endif

    return std::filesystem::canonical(pluginPath.lexically_normal());
}

// Wrapper class to access protected constructor
class TestableHeuristicPlugin : public HeuristicPlugin
{
public:
    explicit TestableHeuristicPlugin(SharedLibrary&& lib)
        : HeuristicPlugin(std::move(lib))
    {
    }
};

} // anonymous namespace

class TestHeuristicPluginLoading : public ::testing::Test
{
protected:
    void SetUp() override {}

    void TearDown() override {}
};

// Fixture that loads the good plugin for tests that need it
class TestHeuristicPluginLoadedGood : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto pluginPath = getTestPluginPath(TEST_GOOD_HEURISTIC_PLUGIN_NAME);
        ASSERT_TRUE(std::filesystem::exists(pluginPath))
            << "Test plugin not found: " << pluginPath
            << "\nMake sure test_plugins are built before running tests";

        SharedLibrary lib(pluginPath);
        _pluginPtr = std::make_unique<TestableHeuristicPlugin>(std::move(lib));
    }

    void TearDown() override
    {
        _pluginPtr.reset();
    }

    TestableHeuristicPlugin& plugin()
    {
        return *_pluginPtr;
    }

private:
    std::unique_ptr<TestableHeuristicPlugin> _pluginPtr;
};

// ========== Good Plugin Loading Tests ==========

TEST_F(TestHeuristicPluginLoading, LoadGoodPluginSucceeds)
{
    const auto pluginPath = getTestPluginPath(TEST_GOOD_HEURISTIC_PLUGIN_NAME);

    ASSERT_TRUE(std::filesystem::exists(pluginPath))
        << "Test plugin not found: " << pluginPath
        << "\nMake sure test_plugins are built before running tests";

    // Load the plugin
    SharedLibrary lib(pluginPath);
    // NOLINTNEXTLINE(misc-const-correctness)
    ASSERT_NO_THROW({ TestableHeuristicPlugin plugin(std::move(lib)); });
}

TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanQueryApiVersion)
{
    const auto version = plugin().apiVersion();
    EXPECT_FALSE(version.empty());
    EXPECT_EQ(version, "1.0.0"); // HIPDNN_HEURISTIC_API_VERSION
}

TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanQueryPolicyId)
{
    const auto policyId = plugin().policyId();
    EXPECT_EQ(policyId, 0x1234567890ABCDEF); // TEST_POLICY_ID from plugin
}

TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanQueryPolicyName)
{
    const auto name = plugin().policyName();
    EXPECT_EQ(name, "TestGoodHeuristicPolicy");
}

TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanQueryPluginVersion)
{
    const auto version = plugin().pluginVersion();
    EXPECT_EQ(version, "1.0.0");
}

TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanCreateAndDestroyHandle)
{
    hipdnnHeuristicHandle_t handle = nullptr;
    ASSERT_NO_THROW({ handle = plugin().createHandle(); });
    EXPECT_NE(handle, nullptr);

    ASSERT_NO_THROW({ plugin().destroyHandle(handle); });
}

TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanSetDeviceProperties)
{
    const auto handle = plugin().createHandle();
    ASSERT_NE(handle, nullptr);

    hipdnnPluginConstData_t deviceProps{};
    deviceProps.ptr = nullptr;
    deviceProps.size = 0;

    ASSERT_NO_THROW({ plugin().setDeviceProperties(handle, &deviceProps); });

    plugin().destroyHandle(handle);
}

TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanManagePolicyDescriptor)
{
    const auto handle = plugin().createHandle();
    ASSERT_NE(handle, nullptr);

    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    ASSERT_NO_THROW({ desc = plugin().createPolicyDescriptor(handle); });
    EXPECT_NE(desc, nullptr);

    ASSERT_NO_THROW({ plugin().destroyPolicyDescriptor(desc); });

    plugin().destroyHandle(handle);
}

TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanSetEngineIds)
{
    const auto handle = plugin().createHandle();
    const auto desc = plugin().createPolicyDescriptor(handle);

    const std::vector<int64_t> engineIds = {1, 2, 3, 4, 5};
    ASSERT_NO_THROW({ plugin().setEngineIds(desc, engineIds.data(), engineIds.size()); });

    plugin().destroyPolicyDescriptor(desc);
    plugin().destroyHandle(handle);
}

TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanSetSerializedGraph)
{
    const auto handle = plugin().createHandle();
    const auto desc = plugin().createPolicyDescriptor(handle);

    hipdnnPluginConstData_t graphData{};
    graphData.ptr = nullptr;
    graphData.size = 0;

    ASSERT_NO_THROW({ plugin().setSerializedGraph(desc, &graphData); });

    plugin().destroyPolicyDescriptor(desc);
    plugin().destroyHandle(handle);
}

TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanFinalizePolicy)
{
    const auto handle = plugin().createHandle();
    const auto desc = plugin().createPolicyDescriptor(handle);

    const std::vector<int64_t> engineIds = {1, 2, 3};
    plugin().setEngineIds(desc, engineIds.data(), engineIds.size());

    bool applied = false;
    ASSERT_NO_THROW({ applied = plugin().finalize(desc); });
    EXPECT_TRUE(applied); // Good plugin always applies

    plugin().destroyPolicyDescriptor(desc);
    plugin().destroyHandle(handle);
}

TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanGetSortedEngineIds)
{
    const auto handle = plugin().createHandle();
    const auto desc = plugin().createPolicyDescriptor(handle);

    const std::vector<int64_t> inputIds = {1, 2, 3, 4, 5};
    plugin().setEngineIds(desc, inputIds.data(), inputIds.size());
    plugin().finalize(desc);

    std::vector<int64_t> sortedIds;
    ASSERT_NO_THROW({ sortedIds = plugin().getSortedEngineIds(desc); });

    // Good plugin reverses the order
    EXPECT_EQ(sortedIds.size(), inputIds.size());
    EXPECT_EQ(sortedIds, std::vector<int64_t>({5, 4, 3, 2, 1}));

    plugin().destroyPolicyDescriptor(desc);
    plugin().destroyHandle(handle);
}

TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCompleteWorkflow)
{
    // Create handle and descriptor
    const auto handle = plugin().createHandle();
    ASSERT_NE(handle, nullptr);

    const auto desc = plugin().createPolicyDescriptor(handle);
    ASSERT_NE(desc, nullptr);

    // Set inputs
    const std::vector<int64_t> inputIds = {10, 20, 30};
    plugin().setEngineIds(desc, inputIds.data(), inputIds.size());

    hipdnnPluginConstData_t graphData{};
    graphData.ptr = nullptr;
    graphData.size = 0;
    plugin().setSerializedGraph(desc, &graphData);

    // Finalize and retrieve results
    const bool applied = plugin().finalize(desc);
    EXPECT_TRUE(applied);

    const auto sortedIds = plugin().getSortedEngineIds(desc);
    EXPECT_EQ(sortedIds, std::vector<int64_t>({30, 20, 10}));

    // Cleanup
    plugin().destroyPolicyDescriptor(desc);
    plugin().destroyHandle(handle);
}

// ========== Incomplete Plugin Tests ==========

TEST_F(TestHeuristicPluginLoading, LoadIncompletePluginThrowsException)
{
    const auto pluginPath = getTestPluginPath(TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME);

    ASSERT_TRUE(std::filesystem::exists(pluginPath)) << "Test plugin not found: " << pluginPath;

    SharedLibrary lib(pluginPath);

    // Loading should fail during symbol resolution
    EXPECT_THROW(
        {
            try
            {
                const TestableHeuristicPlugin plugin(std::move(lib));
            }
            catch(const HipdnnException& e)
            {
                // Verify the exception contains expected error details
                const std::string errorMsg(e.what());
                EXPECT_NE(errorMsg.find("HEURISTIC PLUGIN ABI INCOMPLETE"), std::string::npos);
                EXPECT_NE(errorMsg.find("Missing required symbol"), std::string::npos);
                EXPECT_NE(errorMsg.find(pluginPath.string()), std::string::npos);
                throw;
            }
        },
        HipdnnException);
}

TEST_F(TestHeuristicPluginLoading, IncompletePluginExceptionContainsSymbolName)
{
    const auto pluginPath = getTestPluginPath(TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME);
    SharedLibrary lib(pluginPath);

    EXPECT_THROW(
        {
            try
            {
                const TestableHeuristicPlugin plugin(std::move(lib));
            }
            catch(const HipdnnException& e)
            {
                const std::string errorMsg(e.what());
                // Should mention one of the missing symbols
                const bool hasFinalizeError
                    = errorMsg.find("hipdnnHeuristicPolicyFinalize") != std::string::npos;
                const bool hasGetSortedError
                    = errorMsg.find("hipdnnHeuristicPolicyGetSortedEngineIds") != std::string::npos;
                EXPECT_TRUE(hasFinalizeError || hasGetSortedError);
                throw;
            }
        },
        HipdnnException);
}

TEST_F(TestHeuristicPluginLoading, IncompletePluginExceptionHasPluginErrorStatus)
{
    const auto pluginPath = getTestPluginPath(TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME);
    SharedLibrary lib(pluginPath);

    EXPECT_THROW(
        {
            try
            {
                const TestableHeuristicPlugin plugin(std::move(lib));
            }
            catch(const HipdnnException& e)
            {
                EXPECT_EQ(e.getStatus(), HIPDNN_STATUS_PLUGIN_ERROR);
                throw;
            }
        },
        HipdnnException);
}

// ========== Optional Symbol Tests ==========

TEST_F(TestHeuristicPluginLoading, LoadPluginWithoutOptionalSymbolsSucceeds)
{
    const auto pluginPath = getTestPluginPath(TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME);

    ASSERT_TRUE(std::filesystem::exists(pluginPath)) << "Test plugin not found: " << pluginPath;

    SharedLibrary lib(pluginPath);

    // Should load successfully despite missing optional symbols
    // NOLINTNEXTLINE(misc-const-correctness)
    ASSERT_NO_THROW({ TestableHeuristicPlugin plugin(std::move(lib)); });
}

TEST_F(TestHeuristicPluginLoading, PluginWithoutOptionalPolicyNameReturnsEmpty)
{
    const auto pluginPath = getTestPluginPath(TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME);
    SharedLibrary lib(pluginPath);
    const TestableHeuristicPlugin plugin(std::move(lib));

    // Plugin doesn't implement hipdnnHeuristicGetPolicyName
    const auto name = plugin.policyName();
    EXPECT_TRUE(name.empty());
}

TEST_F(TestHeuristicPluginLoading, PluginWithoutOptionalSetLogLevelSucceeds)
{
    const auto pluginPath = getTestPluginPath(TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME);
    SharedLibrary lib(pluginPath);
    const TestableHeuristicPlugin plugin(std::move(lib));

    // Plugin doesn't implement hipdnnHeuristicSetLogLevel
    // Should return SUCCESS without calling the function
    const auto status = plugin.setLogLevel(HIPDNN_SEV_INFO);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST_F(TestHeuristicPluginLoading, PluginWithoutOptionalCanStillExecuteWorkflow)
{
    const auto pluginPath = getTestPluginPath(TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME);
    SharedLibrary lib(pluginPath);
    const TestableHeuristicPlugin plugin(std::move(lib));

    // Full workflow should work despite missing optional functions
    const auto handle = plugin.createHandle();
    ASSERT_NE(handle, nullptr);

    const auto desc = plugin.createPolicyDescriptor(handle);
    ASSERT_NE(desc, nullptr);

    const std::vector<int64_t> inputIds = {1, 2, 3};
    plugin.setEngineIds(desc, inputIds.data(), inputIds.size());

    const bool applied = plugin.finalize(desc);
    EXPECT_FALSE(applied); // This plugin declines to apply

    const auto sortedIds = plugin.getSortedEngineIds(desc);
    EXPECT_TRUE(sortedIds.empty()); // Returns empty list

    plugin.destroyPolicyDescriptor(desc);
    plugin.destroyHandle(handle);
}

// ========== Policy ID Caching Tests (Real Plugin) ==========

TEST_F(TestHeuristicPluginLoadedGood, RealPluginCachesPolicyId)
{
    // First call
    const auto id1 = plugin().policyId();
    EXPECT_EQ(id1, 0x1234567890ABCDEF);

    // Second call should return cached value
    const auto id2 = plugin().policyId();
    EXPECT_EQ(id2, id1);
}

// ========== Empty Engine List Tests ==========

TEST_F(TestHeuristicPluginLoading, GetSortedEngineIdsReturnsEmptyWhenNoEngines)
{
    const auto pluginPath = getTestPluginPath(TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME);
    SharedLibrary lib(pluginPath);
    const TestableHeuristicPlugin plugin(std::move(lib));

    const auto handle = plugin.createHandle();
    const auto desc = plugin.createPolicyDescriptor(handle);

    // Don't set any engine IDs
    plugin.finalize(desc);

    const auto sortedIds = plugin.getSortedEngineIds(desc);
    EXPECT_TRUE(sortedIds.empty());

    plugin.destroyPolicyDescriptor(desc);
    plugin.destroyHandle(handle);
}
