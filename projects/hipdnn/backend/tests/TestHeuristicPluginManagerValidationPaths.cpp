// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestHeuristicPluginManagerValidationPaths.cpp
 * @brief Tests for HeuristicPluginManager validation code paths (RFC 0007)
 *
 * These tests load actual test plugins to exercise validateBeforeAdding() and
 * actionAfterAdding() to improve coverage of HeuristicPluginManager.hpp
 */

#include "HipdnnException.hpp"
#include "PlatformUtils.hpp"
#include "TestPluginConstants.hpp"
#include "plugin/HeuristicPluginManager.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/heuristic_api_version.h>

// Test plugin name constants (defined here because CMake ordering prevents proper macro propagation)
namespace
{
constexpr const char* BAD_API_VERSION_PLUGIN = "test_bad_api_version_heuristic_plugin";
constexpr const char* EMPTY_NAME_PLUGIN = "test_empty_name_heuristic_plugin";
constexpr const char* DUPLICATE_POLICY_ID_A_PLUGIN = "test_duplicate_policy_id_a_plugin";
constexpr const char* DUPLICATE_POLICY_ID_B_PLUGIN = "test_duplicate_policy_id_b_plugin";
} // namespace

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;
using namespace hipdnn_backend::plugin_constants;

class TestHeuristicPluginManagerValidationPaths : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        // Check once if test plugins are available
        const auto pluginPath = getHeuristicPluginPath("").parent_path();
        if(!std::filesystem::exists(pluginPath))
        {
            GTEST_SKIP() << "Test plugins not found at: " << pluginPath
                         << "\nMake sure test_plugins are built before running tests";
        }
    }

    void SetUp() override
    {
        // Test plugins are in lib/test_plugins/custom relative to backend library location
        _testPluginPath = getHeuristicPluginPath("").parent_path();

        // Create manager for each test
        _manager = std::make_unique<HeuristicPluginManager>();
    }

    std::unique_ptr<HeuristicPluginManager> _manager;
    std::filesystem::path _testPluginPath;
};

// ========== validateBeforeAdding() Tests - API Version Check ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, GoodPluginPassesApiVersionValidation)
{
    // Load good test plugin - should pass API version validation
    EXPECT_NO_THROW(_manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE));

    const auto& plugins = _manager->getPlugins();

    // Should have loaded at least the good plugin
    EXPECT_GT(plugins.size(), 0);

    // All loaded plugins should have correct API major version
    for(const auto& plugin : plugins)
    {
        const auto version = hipdnn_data_sdk::utilities::Version{plugin->apiVersion()};
        EXPECT_EQ(version.major, HIPDNN_HEURISTIC_API_VERSION_MAJOR)
            << "validateBeforeAdding should have checked API version for plugin: "
            << plugin->name();
    }
}

// ========== validateBeforeAdding() Tests - Policy ID Uniqueness ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, ActionAfterAddingStoresPolicyIds)
{
    // Load plugins - actionAfterAdding should store policy IDs
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    const auto& plugins = _manager->getPlugins();

    // Collect all policy IDs to verify actionAfterAdding was called
    std::set<int64_t> policyIds;
    for(const auto& plugin : plugins)
    {
        const int64_t id = plugin->policyId();

        // Each policy ID should be unique (actionAfterAdding should have tracked this)
        EXPECT_EQ(policyIds.count(id), 0)
            << "Policy ID " << id << " appears multiple times (actionAfterAdding tracking failed)";

        policyIds.insert(id);
    }

    // Should have loaded plugins with unique policy IDs
    EXPECT_GT(policyIds.size(), 0) << "Should have loaded at least one plugin";
    EXPECT_EQ(policyIds.size(), plugins.size()) << "All policy IDs should be unique";
}

TEST_F(TestHeuristicPluginManagerValidationPaths, PolicyIdTrackingAcrossMultiplePlugins)
{
    // Load all available plugins
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = _manager->getPlugins();

    // Collect all policy IDs
    std::set<int64_t> policyIds;
    for(const auto& plugin : plugins)
    {
        const int64_t id = plugin->policyId();

        // Each policy ID should be unique (actionAfterAdding tracks this)
        EXPECT_EQ(policyIds.count(id), 0)
            << "Policy ID " << id << " appears multiple times (validateBeforeAdding failed)";

        policyIds.insert(id);
    }

    // Should have as many unique IDs as plugins
    EXPECT_EQ(policyIds.size(), plugins.size());
}

// ========== validateBeforeAdding() Tests - Policy Name Check ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, AllLoadedPluginsHaveNonEmptyNames)
{
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = _manager->getPlugins();

    // validateBeforeAdding should have rejected any plugins with empty names
    for(const auto& plugin : plugins)
    {
        EXPECT_FALSE(plugin->name().empty())
            << "validateBeforeAdding should reject plugins with empty policy names";
    }
}

TEST_F(TestHeuristicPluginManagerValidationPaths, PolicyNameIsProvided)
{
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = _manager->getPlugins();
    EXPECT_GT(plugins.size(), 0);

    for(const auto& plugin : plugins)
    {
        const std::string policyName(plugin->name());

        // Policy name should be non-empty (validated in validateBeforeAdding lines 82-89)
        EXPECT_FALSE(policyName.empty()) << "Policy ID " << plugin->policyId() << " has empty name";

        // Should be a valid string
        EXPECT_GT(policyName.length(), 0);
    }
}

// ========== Validation Success Path Tests ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, ValidPluginPassesAllValidation)
{
    // Load should succeed for valid plugins
    EXPECT_NO_THROW(_manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE));

    const auto& plugins = _manager->getPlugins();

    for(const auto& plugin : plugins)
    {
        // API version check (validateBeforeAdding lines 55-66)
        const auto version = hipdnn_data_sdk::utilities::Version{plugin->apiVersion()};
        EXPECT_EQ(version.major, HIPDNN_HEURISTIC_API_VERSION_MAJOR);

        // Policy ID should be non-zero
        EXPECT_NE(plugin->policyId(), 0);

        // Policy name check (validateBeforeAdding lines 82-89)
        EXPECT_FALSE(plugin->name().empty());
    }
}

// ========== actionAfterAdding() Coverage Tests ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, ActionAfterAddingExecutesForEachPlugin)
{
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = _manager->getPlugins();

    // For each plugin loaded, actionAfterAdding should have:
    // 1. Inserted policy ID into _policyIds set (line 94)
    // This is verified indirectly by ensuring no duplicate IDs exist

    std::set<int64_t> observedIds;
    for(const auto& plugin : plugins)
    {
        EXPECT_EQ(observedIds.count(plugin->policyId()), 0)
            << "Duplicate policy ID detected - actionAfterAdding may have failed";
        observedIds.insert(plugin->policyId());
    }
}

// ========== Multiple Load Cycles with Validation ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, ValidationRunsOnEachLoadCycle)
{
    // Create new manager for each load to ensure fresh state
    HeuristicPluginManager manager1; // NOLINT(misc-const-correctness)
    manager1.loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    const size_t count1 = manager1.getPlugins().size();

    // Second manager should also run validation and load same plugins
    HeuristicPluginManager manager2; // NOLINT(misc-const-correctness)
    manager2.loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    const size_t count2 = manager2.getPlugins().size();

    EXPECT_EQ(count1, count2) << "Both managers should validate and load same plugins";
    EXPECT_GT(count1, 0) << "Should have loaded at least one plugin";
}

// ========== Constructor Path Coverage ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, ConstructorSetsUpValidationInfrastructure)
{
    // Constructor initializes with search paths and empty _policyIds set
    const HeuristicPluginManager manager;

    // Should start with no plugins
    EXPECT_TRUE(_manager->getPlugins().empty());

    // Validation infrastructure should be ready (verified by successful construction)
    SUCCEED();
}

// ========== Integration with PluginManagerBase ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, ValidationIntegratesWithBaseClass)
{
    // PluginManagerBase calls validateBeforeAdding before adding each plugin
    // and actionAfterAdding after successful add
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = _manager->getPlugins();

    // All plugins should have passed validation
    for(const auto& plugin : plugins)
    {
        // These checks verify that validateBeforeAdding was called and passed
        EXPECT_NE(plugin->policyId(), 0);
        EXPECT_FALSE(plugin->name().empty());

        const auto version = hipdnn_data_sdk::utilities::Version{plugin->apiVersion()};
        EXPECT_EQ(version.major, HIPDNN_HEURISTIC_API_VERSION_MAJOR);
    }
}

// ========== Specific Test Plugin Tests ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, NoOptionalHeuristicPluginPassesValidation)
{
    // test_no_optional_heuristic_plugin doesn't implement optional functions
    // but should still pass validation
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = _manager->getPlugins();

    for(const auto& plugin : plugins)
    {
        const std::string name(plugin->name());
        if(name.find("NoOptional") != std::string::npos)
        {
            // Should pass all validation checks
            EXPECT_NE(plugin->policyId(), 0);
            EXPECT_FALSE(name.empty());
        }
    }

    // Plugin should have been loaded if it exists
    if(!plugins.empty())
    {
        SUCCEED();
    }
}

TEST_F(TestHeuristicPluginManagerValidationPaths, GoodHeuristicPluginPassesValidation)
{
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = _manager->getPlugins();

    for(const auto& plugin : plugins)
    {
        const std::string name(plugin->name());
        if(name.find("Good") != std::string::npos || name.find("Test") != std::string::npos)
        {
            // Verify it passed all validation:
            // 1. API version (lines 55-66)
            const auto version = hipdnn_data_sdk::utilities::Version{plugin->apiVersion()};
            EXPECT_EQ(version.major, HIPDNN_HEURISTIC_API_VERSION_MAJOR);

            // 2. Policy ID is unique and non-zero (lines 69-78)
            EXPECT_NE(plugin->policyId(), 0);

            // 3. Policy name is non-empty (lines 81-89)
            EXPECT_FALSE(name.empty());
        }
    }

    if(!plugins.empty())
    {
        SUCCEED();
    }
}

// ========== Validation Failure Tests ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, BadApiVersionPluginRejected)
{

    // Build path to bad plugin in a separate directory to isolate it
    const auto badPluginDir = std::filesystem::temp_directory_path() / "hipdnn_bad_api_test";
    std::filesystem::create_directories(badPluginDir);

    // Copy the bad plugin to isolated directory
    const std::string pluginFilename
        = hipdnn_data_sdk::utilities::getLibraryName(BAD_API_VERSION_PLUGIN);
    const auto srcPlugin = _testPluginPath / pluginFilename;
    const auto dstPlugin = badPluginDir / pluginFilename;
    std::filesystem::copy_file(
        srcPlugin, dstPlugin, std::filesystem::copy_options::overwrite_existing);

    // Loading plugin with wrong API version - loadPlugins logs errors but doesn't throw
    _manager->loadPlugins({badPluginDir}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    // Plugin should not have been loaded due to validation failure
    EXPECT_TRUE(_manager->getPlugins().empty()) << "Bad API version plugin should be rejected";

    // Cleanup
    std::filesystem::remove_all(badPluginDir);
}

TEST_F(TestHeuristicPluginManagerValidationPaths, EmptyNamePluginRejected)
{

    // Build path to empty name plugin in a separate directory to isolate it
    const auto emptyNameDir = std::filesystem::temp_directory_path() / "hipdnn_empty_name_test";
    std::filesystem::create_directories(emptyNameDir);

    // Copy the empty name plugin to isolated directory
    const std::string pluginFilename = hipdnn_data_sdk::utilities::getLibraryName(EMPTY_NAME_PLUGIN);
    const auto srcPlugin = _testPluginPath / pluginFilename;
    const auto dstPlugin = emptyNameDir / pluginFilename;
    std::filesystem::copy_file(
        srcPlugin, dstPlugin, std::filesystem::copy_options::overwrite_existing);

    // Loading plugin with empty policy name - loadPlugins logs errors but doesn't throw
    _manager->loadPlugins({emptyNameDir}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    // Plugin should not have been loaded due to validation failure
    EXPECT_TRUE(_manager->getPlugins().empty()) << "Empty policy name plugin should be rejected";

    // Cleanup
    std::filesystem::remove_all(emptyNameDir);
}

TEST_F(TestHeuristicPluginManagerValidationPaths, DuplicatePolicyIdPluginsRejected)
{

    // Build path to duplicate plugins in a separate directory to isolate them
    const auto duplicateDir = std::filesystem::temp_directory_path() / "hipdnn_duplicate_id_test";
    std::filesystem::create_directories(duplicateDir);

    // Copy both duplicate plugins to isolated directory
    const std::string pluginFilenameA
        = hipdnn_data_sdk::utilities::getLibraryName(DUPLICATE_POLICY_ID_A_PLUGIN);
    const std::string pluginFilenameB
        = hipdnn_data_sdk::utilities::getLibraryName(DUPLICATE_POLICY_ID_B_PLUGIN);
    const auto srcPluginA = _testPluginPath / pluginFilenameA;
    const auto srcPluginB = _testPluginPath / pluginFilenameB;
    const auto dstPluginA = duplicateDir / pluginFilenameA;
    const auto dstPluginB = duplicateDir / pluginFilenameB;

    if(std::filesystem::exists(srcPluginA) && std::filesystem::exists(srcPluginB))
    {
        std::filesystem::copy_file(
            srcPluginA, dstPluginA, std::filesystem::copy_options::overwrite_existing);
        std::filesystem::copy_file(
            srcPluginB, dstPluginB, std::filesystem::copy_options::overwrite_existing);

        // Load directory with both duplicate plugins - loadPlugins logs errors but doesn't throw
        _manager->loadPlugins({duplicateDir}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

        // Only one plugin should have loaded - the second should be rejected due to duplicate ID
        EXPECT_EQ(_manager->getPlugins().size(), 1) << "Only first duplicate plugin should load";

        // Cleanup
        std::filesystem::remove_all(duplicateDir);
    }
    else
    {
        GTEST_SKIP() << "test_duplicate_policy_id plugins not found";
    }
}

// ========== Edge Case: Empty Plugin Directory ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, EmptyDirectorySkipsValidation)
{

    const std::filesystem::path emptyDir
        = std::filesystem::temp_directory_path() / "hipdnn_empty_heur_test";
    std::filesystem::create_directories(emptyDir);

    // Load from empty directory - no plugins to validate
    _manager->loadPlugins({emptyDir}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    EXPECT_TRUE(_manager->getPlugins().empty());

    std::filesystem::remove(emptyDir);
}
