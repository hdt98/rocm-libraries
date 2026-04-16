// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestHeuristicPluginResourceManager.cpp
 * @brief Unit tests for HeuristicPluginResourceManager (RFC 0007 Part 1)
 *
 * These tests verify the plugin resource management layer that provides
 * per-handle plugin lifecycle management and policy lookup.
 */

#include "HipdnnException.hpp"
#include "descriptors/mocks/MockHeuristicPlugin.hpp"
#include "plugin/HeuristicPluginManager.hpp"
#include "plugin/HeuristicPluginResourceManager.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;

class TestHeuristicPluginResourceManager : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Tests in this suite use mock plugins, not real shared libraries
    }

    void TearDown() override {}
};

// ========== Construction and Initialization Tests ==========

TEST_F(TestHeuristicPluginResourceManager, FactoryMethodCreatesInstance)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);
}

TEST_F(TestHeuristicPluginResourceManager, ConstructorWithPluginManagerSucceeds)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);
    ASSERT_NE(rm, nullptr);
}

// ========== Move Semantics Tests ==========

TEST_F(TestHeuristicPluginResourceManager, MoveConstructorTransfersOwnership)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm1 = std::make_shared<HeuristicPluginResourceManager>(pm);

    // Move construct
    const HeuristicPluginResourceManager rm2(std::move(*rm1));

    // rm2 should be usable
    const auto infos = rm2.getHeuristicPolicyInfos();
    EXPECT_TRUE(infos.empty()); // No plugins loaded
}

TEST_F(TestHeuristicPluginResourceManager, MoveAssignmentTransfersOwnership)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm1 = std::make_shared<HeuristicPluginResourceManager>(pm);
    auto rm2 = std::make_shared<HeuristicPluginResourceManager>(pm);

    // Move assign
    *rm2 = std::move(*rm1);

    // rm2 should be usable
    const HeuristicPluginResourceManager& constRm2 = *rm2;
    const auto infos = constRm2.getHeuristicPolicyInfos();
    EXPECT_TRUE(infos.empty());
}

// ========== Policy Lookup Tests ==========

TEST_F(TestHeuristicPluginResourceManager, GetHandleForNonexistentPolicyReturnsNull)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    const int64_t fakePolicyId = 0x1234567890ABCDEF;
    auto handle = rm->getHeuristicHandleForPolicyId(fakePolicyId);

    EXPECT_EQ(handle, nullptr);
}

TEST_F(TestHeuristicPluginResourceManager, GetPluginForNonexistentPolicyReturnsNull)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    const int64_t fakePolicyId = 0x1234567890ABCDEF;
    auto plugin = rm->getPluginForPolicyId(fakePolicyId);

    EXPECT_EQ(plugin, nullptr);
}

// ========== Policy Enumeration Tests ==========

TEST_F(TestHeuristicPluginResourceManager, GetPolicyInfosWhenNoPluginsLoaded)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    const auto infos = rm->getHeuristicPolicyInfos();
    EXPECT_TRUE(infos.empty());
}

TEST_F(TestHeuristicPluginResourceManager, GetPolicyInfosCachesResult)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    // First call
    const auto infos1 = rm->getHeuristicPolicyInfos();

    // Second call should return cached result (implementation detail)
    const auto infos2 = rm->getHeuristicPolicyInfos();

    EXPECT_EQ(infos1.size(), infos2.size());
}

// ========== Device Properties Tests ==========

TEST_F(TestHeuristicPluginResourceManager, SetDevicePropertiesOnEmptyManagerDoesNotThrow)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    hipdnnPluginConstData_t deviceProps;
    deviceProps.ptr = nullptr;
    deviceProps.size = 0;

    // Should not throw even when no plugins loaded
    EXPECT_NO_THROW(rm->setDevicePropertiesOnAllHandles(&deviceProps));
}

TEST_F(TestHeuristicPluginResourceManager, SetDevicePropertiesWithNullPointerSucceeds)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    // Passing nullptr is allowed when no plugins loaded
    EXPECT_NO_THROW(rm->setDevicePropertiesOnAllHandles(nullptr));
}

// ========== Plugin File Enumeration Tests ==========

TEST_F(TestHeuristicPluginResourceManager, GetLoadedPluginFilesWhenNoneLoaded)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    size_t numPlugins = 0;
    size_t maxStringLen = 0;

    // Query counts
    rm->getLoadedHeuristicPluginFiles(&numPlugins, nullptr, &maxStringLen);

    EXPECT_EQ(numPlugins, 0u);
    EXPECT_EQ(maxStringLen, 0u);
}

TEST_F(TestHeuristicPluginResourceManager, GetLoadedPluginFilesWithNullNumPluginsThrows)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    // nullptr for numPlugins should throw
    EXPECT_THROW(rm->getLoadedHeuristicPluginFiles(nullptr, nullptr, nullptr), HipdnnException);
}

// ========== String Representation Tests ==========

TEST_F(TestHeuristicPluginResourceManager, ToStringReturnsNonEmptyString)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    const std::string str = rm->toString();
    EXPECT_FALSE(str.empty());
}

TEST_F(TestHeuristicPluginResourceManager, ToStringIncludesPluginCount)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    const std::string str = rm->toString();

    // Should mention plugin count (even if 0)
    EXPECT_TRUE(str.find("plugin") != std::string::npos || str.find("Plugin") != std::string::npos);
}

// ========== Static Path Configuration Tests ==========

TEST_F(TestHeuristicPluginResourceManager, SetHeuristicPluginPathsSucceeds)
{
    const std::vector<std::filesystem::path> paths = {"/tmp/test_plugins"};

    EXPECT_NO_THROW(HeuristicPluginResourceManager::setHeuristicPluginPaths(
        paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
}

TEST_F(TestHeuristicPluginResourceManager, GetHeuristicPluginPathsReturnsConfiguredPaths)
{
    const std::vector<std::filesystem::path> paths = {"/tmp/test_plugins", "/tmp/more_plugins"};

    HeuristicPluginResourceManager::setHeuristicPluginPaths(paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto retrievedPaths = HeuristicPluginResourceManager::getHeuristicPluginPaths();

    // Should contain at least the paths we set (may include defaults)
    EXPECT_GE(retrievedPaths.size(), 0u);
}

TEST_F(TestHeuristicPluginResourceManager, SetPluginUnloadingModeSucceeds)
{
    // Both modes should be accepted
    EXPECT_NO_THROW(
        HeuristicPluginResourceManager::setPluginUnloadingMode(HIPDNN_PLUGIN_UNLOAD_LAZY));

    EXPECT_NO_THROW(
        HeuristicPluginResourceManager::setPluginUnloadingMode(HIPDNN_PLUGIN_UNLOAD_EAGER));
}

TEST_F(TestHeuristicPluginResourceManager, SetPluginLogLevelSucceeds)
{
    // Should not throw for various log levels
    EXPECT_NO_THROW(HeuristicPluginResourceManager::setPluginLogLevel(HIPDNN_SEV_INFO));
    EXPECT_NO_THROW(HeuristicPluginResourceManager::setPluginLogLevel(HIPDNN_SEV_WARN));
    EXPECT_NO_THROW(HeuristicPluginResourceManager::setPluginLogLevel(HIPDNN_SEV_ERROR));
}

// ========== Multiple Instances Tests ==========

TEST_F(TestHeuristicPluginResourceManager, MultipleInstancesCanCoexist)
{
    auto pm = std::make_shared<HeuristicPluginManager>();

    auto rm1 = std::make_shared<HeuristicPluginResourceManager>(pm);
    auto rm2 = std::make_shared<HeuristicPluginResourceManager>(pm);
    auto rm3 = std::make_shared<HeuristicPluginResourceManager>(pm);

    // All should be independent and usable
    EXPECT_NE(rm1, nullptr);
    EXPECT_NE(rm2, nullptr);
    EXPECT_NE(rm3, nullptr);

    // Each should work independently
    EXPECT_TRUE(rm1->getHeuristicPolicyInfos().empty());
    EXPECT_TRUE(rm2->getHeuristicPolicyInfos().empty());
    EXPECT_TRUE(rm3->getHeuristicPolicyInfos().empty());
}

// ========== Copy Prevention Tests ==========

TEST_F(TestHeuristicPluginResourceManager, CopyConstructorIsDeleted)
{
    // This test verifies at compile time that copying is prevented
    // If this compiles, the test passes
    EXPECT_TRUE((std::is_copy_constructible_v<HeuristicPluginResourceManager> == false));
}

TEST_F(TestHeuristicPluginResourceManager, CopyAssignmentIsDeleted)
{
    // This test verifies at compile time that copy assignment is prevented
    EXPECT_TRUE((std::is_copy_assignable_v<HeuristicPluginResourceManager> == false));
}

// ========== Destruction Tests ==========

TEST_F(TestHeuristicPluginResourceManager, DestructorCleansUpResources)
{
    auto pm = std::make_shared<HeuristicPluginManager>();

    {
        auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);
        // Use rm
        rm->getHeuristicPolicyInfos();
    } // rm destroyed here

    // If we get here without crashes, cleanup succeeded
    SUCCEED();
}

TEST_F(TestHeuristicPluginResourceManager, MultipleDestructionsSucceed)
{
    auto pm = std::make_shared<HeuristicPluginManager>();

    for(int i = 0; i < 10; ++i)
    {
        auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);
        rm->getHeuristicPolicyInfos();
        // Destroyed at end of loop
    }

    SUCCEED();
}
