// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestHeuristicPluginResourceManagerAdditional.cpp
 * @brief Additional unit tests for HeuristicPluginResourceManager to improve code coverage
 *
 * These tests cover edge cases and error paths not covered by the main test file.
 */

#include "HipdnnException.hpp"
#include "plugin/HeuristicPluginManager.hpp"
#include "plugin/HeuristicPluginResourceManager.hpp"

#include <array>
#include <gtest/gtest.h>
#include <memory>

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;

class TestHeuristicPluginResourceManagerAdditional : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ========== Constructor Null Pointer Tests ==========

TEST_F(TestHeuristicPluginResourceManagerAdditional, ConstructorWithNullPluginManagerThrows)
{
    EXPECT_THROW(
        { auto rm = std::make_shared<HeuristicPluginResourceManager>(nullptr); }, HipdnnException);
}

// ========== Policy Info Caching Tests ==========

TEST_F(TestHeuristicPluginResourceManagerAdditional, PolicyInfosAreCachedAcrossMultipleCalls)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    // Multiple calls should use cache
    const auto infos1 = rm->getHeuristicPolicyInfos();
    const auto infos2 = rm->getHeuristicPolicyInfos();
    const auto infos3 = rm->getHeuristicPolicyInfos();

    // All should return same result
    EXPECT_EQ(infos1.size(), infos2.size());
    EXPECT_EQ(infos2.size(), infos3.size());
}

// ========== ToString with Plugin Data Tests ==========

TEST_F(TestHeuristicPluginResourceManagerAdditional, ToStringFormatContainsKeyElements)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    const std::string str = rm->toString();

    // Should contain the class name
    EXPECT_NE(str.find("HeuristicPluginResourceManager"), std::string::npos);

    // Should contain braces for structure
    EXPECT_NE(str.find('{'), std::string::npos);
    EXPECT_NE(str.find('}'), std::string::npos);
}

// ========== Additional Configuration Tests ==========

TEST_F(TestHeuristicPluginResourceManagerAdditional, SetPluginPathsWithEmptyVectorSucceeds)
{
    const std::vector<std::filesystem::path> emptyPaths;

    EXPECT_NO_THROW(HeuristicPluginResourceManager::setHeuristicPluginPaths(
        emptyPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
}

TEST_F(TestHeuristicPluginResourceManagerAdditional, SetPluginPathsWithAdditiveMode)
{
    const std::vector<std::filesystem::path> paths = {"/test/path1", "/test/path2"};

    EXPECT_NO_THROW(HeuristicPluginResourceManager::setHeuristicPluginPaths(
        paths, HIPDNN_PLUGIN_LOADING_ADDITIVE));
}

TEST_F(TestHeuristicPluginResourceManagerAdditional, SetPluginLogLevelWithDebugLevel)
{
    EXPECT_NO_THROW(HeuristicPluginResourceManager::setPluginLogLevel(HIPDNN_SEV_INFO));
}

TEST_F(TestHeuristicPluginResourceManagerAdditional, SetPluginLogLevelWithErrorLevel)
{
    EXPECT_NO_THROW(HeuristicPluginResourceManager::setPluginLogLevel(HIPDNN_SEV_ERROR));
}

// ========== GetLoadedPluginFiles Edge Cases ==========

TEST_F(TestHeuristicPluginResourceManagerAdditional, GetLoadedPluginFilesWithNonNullPathsSucceeds)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    size_t numPlugins = 0;
    std::array<char*, 10> paths{};

    // Query with paths array (not yet implemented, should not crash)
    EXPECT_NO_THROW(rm->getLoadedHeuristicPluginFiles(&numPlugins, paths.data(), nullptr));
}

TEST_F(TestHeuristicPluginResourceManagerAdditional, GetLoadedPluginFilesQueriesCountOnly)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    size_t numPlugins = 999; // Should be overwritten

    rm->getLoadedHeuristicPluginFiles(&numPlugins, nullptr, nullptr);

    EXPECT_EQ(numPlugins, 0u); // No plugins loaded
}

// ========== Device Properties Error Handling ==========

TEST_F(TestHeuristicPluginResourceManagerAdditional, SetDevicePropertiesWithValidDataSucceeds)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    const std::array<uint8_t, 4> fakeData = {1, 2, 3, 4};
    hipdnnPluginConstData_t deviceProps;
    deviceProps.ptr = fakeData.data();
    deviceProps.size = fakeData.size();

    // Should not throw when plugins loaded or not
    EXPECT_NO_THROW(rm->setDevicePropertiesOnAllHandles(&deviceProps));
}

// ========== Multiple Static Configuration Changes ==========

TEST_F(TestHeuristicPluginResourceManagerAdditional, MultipleUnloadingModeChangesSucceed)
{
    EXPECT_NO_THROW(
        HeuristicPluginResourceManager::setPluginUnloadingMode(HIPDNN_PLUGIN_UNLOAD_LAZY));

    EXPECT_NO_THROW(
        HeuristicPluginResourceManager::setPluginUnloadingMode(HIPDNN_PLUGIN_UNLOAD_EAGER));

    EXPECT_NO_THROW(
        HeuristicPluginResourceManager::setPluginUnloadingMode(HIPDNN_PLUGIN_UNLOAD_LAZY));
}

TEST_F(TestHeuristicPluginResourceManagerAdditional, MultiplePathConfigurationsSucceed)
{
    const std::vector<std::filesystem::path> paths1 = {"/test/path1"};
    const std::vector<std::filesystem::path> paths2 = {"/test/path2", "/test/path3"};

    EXPECT_NO_THROW(HeuristicPluginResourceManager::setHeuristicPluginPaths(
        paths1, HIPDNN_PLUGIN_LOADING_ABSOLUTE));

    EXPECT_NO_THROW(HeuristicPluginResourceManager::setHeuristicPluginPaths(
        paths2, HIPDNN_PLUGIN_LOADING_ADDITIVE));
}

// ========== Policy Lookup with Multiple Plugins ==========

TEST_F(TestHeuristicPluginResourceManagerAdditional, GetHandleForMultiplePolicyIdsAllReturnNull)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    // Try multiple non-existent policy IDs
    EXPECT_EQ(rm->getHeuristicHandleForPolicyId(1), nullptr);
    EXPECT_EQ(rm->getHeuristicHandleForPolicyId(100), nullptr);
    EXPECT_EQ(rm->getHeuristicHandleForPolicyId(0x123456), nullptr);
    EXPECT_EQ(rm->getHeuristicHandleForPolicyId(-1), nullptr);
}

TEST_F(TestHeuristicPluginResourceManagerAdditional, GetPluginForMultiplePolicyIdsAllReturnNull)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    // Try multiple non-existent policy IDs
    EXPECT_EQ(rm->getPluginForPolicyId(1), nullptr);
    EXPECT_EQ(rm->getPluginForPolicyId(100), nullptr);
    EXPECT_EQ(rm->getPluginForPolicyId(0x123456), nullptr);
    EXPECT_EQ(rm->getPluginForPolicyId(-1), nullptr);
}
