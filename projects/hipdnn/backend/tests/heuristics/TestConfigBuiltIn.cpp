// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestConfigBuiltIn.cpp
 * @brief Tests for the SelectionHeuristic::Config built-in.
 *
 * The built-in lives inside hipdnn_backend_private as a function-pointer table
 * (ConfigBuiltIn::populateFunctionTable) wrapped by HeuristicPlugin via
 * createBuiltIn. There is no .so to dlopen; the wrapper reaches the same code
 * paths used in production registration through HeuristicPluginManager.
 *
 * Wraps the table once via HeuristicPlugin::createBuiltIn and exercises both
 * the C-ABI rejection paths (null pointers, unknown policy IDs) and the
 * policy's end-to-end behavior driven by HIPDNN_POLICY_CONFIG_FILE_PATH:
 * matching rule reorders the candidate list, miss paths decline so the
 * outer policy loop falls through.
 */

#include "heuristics/config/ConfigBuiltIn.hpp"
#include "plugin/HeuristicPlugin.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fb = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_backend::heuristics::config::populateFunctionTable;
using hipdnn_backend::plugin::HeuristicPlugin;
using hipdnn_backend::plugin::HeuristicPluginFunctionTable;
using hipdnn_data_sdk::utilities::engineNameToId;

namespace
{

const int64_t MIOPEN_ENGINE_ID = engineNameToId("MIOPEN_ENGINE");
const int64_t MIOPEN_DETERMINISTIC_ID = engineNameToId("MIOPEN_ENGINE_DETERMINISTIC");
const int64_t CUSTOM_ENGINE_ID = engineNameToId("Plugin1::CustomEngine");

const int64_t CONFIG_POLICY_ID
    = hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::Config");

constexpr const char* OVERRIDE_ENV = "HIPDNN_POLICY_CONFIG_FILE_PATH";

/// Build a minimal serialized Graph FlatBuffer with no nodes.
std::vector<uint8_t> buildEmptyGraphBuffer()
{
    flatbuffers::FlatBufferBuilder builder;
    auto graphOffset = fb::CreateGraphDirect(builder,
                                             nullptr,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             nullptr,
                                             nullptr,
                                             ::flatbuffers::nullopt);
    fb::FinishGraphBuffer(builder, graphOffset);
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}

/// Build a serialized Graph with a single ConvolutionFwd node referencing
/// (x, w) tensors of the requested shapes.
std::vector<uint8_t> buildConvFwdGraphBuffer(const std::vector<int64_t>& xDims,
                                             const std::vector<int64_t>& xStrides,
                                             const std::vector<int64_t>& wDims,
                                             const std::vector<int64_t>& wStrides)
{
    flatbuffers::FlatBufferBuilder builder;

    constexpr int64_t X_UID = 1;
    constexpr int64_t W_UID = 2;
    constexpr int64_t Y_UID = 3;

    const std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        fb::CreateTensorAttributesDirect(
            builder, X_UID, "x", fb::DataType::FLOAT, &xStrides, &xDims),
        fb::CreateTensorAttributesDirect(
            builder, W_UID, "w", fb::DataType::FLOAT, &wStrides, &wDims),
        fb::CreateTensorAttributesDirect(
            builder, Y_UID, "y", fb::DataType::FLOAT, nullptr, nullptr),
    };

    auto convAttrs = fb::CreateConvolutionFwdAttributesDirect(builder, X_UID, W_UID, Y_UID);

    const std::vector<flatbuffers::Offset<fb::Node>> nodes{
        fb::CreateNodeDirect(builder,
                             "conv",
                             fb::DataType::FLOAT,
                             fb::NodeAttributes::ConvolutionFwdAttributes,
                             convAttrs.Union())};

    auto graphOffset = fb::CreateGraphDirect(builder,
                                             nullptr,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             &tensors,
                                             &nodes,
                                             ::flatbuffers::nullopt);
    fb::FinishGraphBuffer(builder, graphOffset);
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}

/// RAII temp directory + JSON file. Returns a path that can be assigned to
/// HIPDNN_POLICY_CONFIG_FILE_PATH; the directory is removed on destruction.
class TempJsonOverrideFile
{
public:
    explicit TempJsonOverrideFile(const std::string& contents)
        : _dir(makeUniqueDir())
        , _path(_dir.path() / "override.json")
    {
        std::ofstream(_path) << contents;
    }

    std::string path() const
    {
        return _path.string();
    }

private:
    static std::filesystem::path makeUniqueDir()
    {
        static std::atomic<uint64_t> s_counter{0};
        const auto path = std::filesystem::temp_directory_path()
                          / ("hipdnn_test_config_" + std::to_string(s_counter.fetch_add(1)));
        std::filesystem::remove_all(path);
        return path;
    }

    hipdnn_test_sdk::utilities::ScopedDirectory _dir;
    std::filesystem::path _path;
};

constexpr const char* DETERMINISTIC_RULE_JSON = R"({
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE_DETERMINISTIC",
      "tensors": [
        { "dim": [1, 3, 4, 4] },
        { "dim": [2, 3, 1, 1] }
      ]
    }
  ]
})";

const std::vector<int64_t> X_DIMS{1, 3, 4, 4};
const std::vector<int64_t> X_STRIDES{48, 16, 4, 1};
const std::vector<int64_t> W_DIMS{2, 3, 1, 1};
const std::vector<int64_t> W_STRIDES{3, 1, 1, 1};

class TestConfigBuiltIn : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _plugin = HeuristicPlugin::createBuiltIn(populateFunctionTable(), "built-in:Config-test");
        _handle = _plugin->createHandle();
        ASSERT_NE(_handle, nullptr);
        _desc = _plugin->createPolicyDescriptor(_handle, CONFIG_POLICY_ID);
        ASSERT_NE(_desc, nullptr);
    }

    void TearDown() override
    {
        if(_desc != nullptr)
        {
            _plugin->destroyPolicyDescriptor(_desc);
        }
        if(_handle != nullptr)
        {
            _plugin->destroyHandle(_handle);
        }
    }

    void setEngineIds(const std::vector<int64_t>& ids)
    {
        _plugin->setEngineIds(_desc, ids.data(), ids.size());
    }

    void setSerializedGraph(const std::vector<uint8_t>& buffer)
    {
        const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
        _plugin->setSerializedGraph(_desc, &data);
    }

    std::shared_ptr<HeuristicPlugin> _plugin;
    hipdnnHeuristicHandle_t _handle = nullptr;
    hipdnnHeuristicPolicyDescriptor_t _desc = nullptr;
};

// Convenience: grab the raw function table once for direct C-ABI rejection tests.
const HeuristicPluginFunctionTable& configAbi()
{
    static const HeuristicPluginFunctionTable s_funcs = populateFunctionTable();
    return s_funcs;
}

} // namespace

// ========== Built-in metadata exposed via the wrapper ==========

TEST_F(TestConfigBuiltIn, ReportsHeuristicPluginType)
{
    EXPECT_EQ(_plugin->type(), HIPDNN_PLUGIN_TYPE_HEURISTIC);
}

TEST_F(TestConfigBuiltIn, EnumeratesSingleConfigPolicy)
{
    const auto ids = _plugin->getAllPolicyIds();
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], CONFIG_POLICY_ID);
    EXPECT_EQ(_plugin->getPolicyName(CONFIG_POLICY_ID), "SelectionHeuristic::Config");
}

// ========== Policy Descriptor Lifecycle (BAD_PARAM via raw ABI) ==========

TEST(TestConfigBuiltInRejection, DescriptorCreateRejectsNullHandle)
{
    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    EXPECT_EQ(configAbi().policyDescriptorCreate(nullptr, CONFIG_POLICY_ID, &desc),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(desc, nullptr);
}

TEST_F(TestConfigBuiltIn, DescriptorCreateRejectsNullOutPointer)
{
    EXPECT_EQ(configAbi().policyDescriptorCreate(_handle, CONFIG_POLICY_ID, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestConfigBuiltInRejection, DescriptorDestroyRejectsNullDescriptor)
{
    EXPECT_EQ(configAbi().policyDescriptorDestroy(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigBuiltIn, DescriptorCreateRejectsUnknownPolicyId)
{
    const int64_t unknownId = hipdnn_data_sdk::utilities::policyNameToId("Vendor::NotARealPolicy");
    ASSERT_NE(unknownId, CONFIG_POLICY_ID);

    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    EXPECT_EQ(configAbi().policyDescriptorCreate(_handle, unknownId, &desc),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(desc, nullptr);
}

TEST_F(TestConfigBuiltIn, GetPolicyNameRejectsUnknownPolicyId)
{
    const int64_t unknownId = hipdnn_data_sdk::utilities::policyNameToId("Vendor::NotARealPolicy");
    ASSERT_NE(unknownId, CONFIG_POLICY_ID);

    const char* name = nullptr;
    EXPECT_EQ(configAbi().getPolicyName(unknownId, &name), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(name, nullptr);
}

// ========== SetEngineIds / SetSerializedGraph BAD_PARAM ==========

TEST(TestConfigBuiltInRejection, SetEngineIdsRejectsNullDescriptor)
{
    const std::array<int64_t, 1> ids{MIOPEN_ENGINE_ID};
    EXPECT_EQ(configAbi().policySetEngineIds(nullptr, ids.data(), ids.size()),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigBuiltIn, SetEngineIdsRejectsNullPointerWithCount)
{
    EXPECT_EQ(configAbi().policySetEngineIds(_desc, nullptr, 3), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestConfigBuiltInRejection, SetSerializedGraphRejectsNullDescriptor)
{
    const std::array<uint8_t, 1> buffer{0x00};
    const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
    EXPECT_EQ(configAbi().policySetSerializedGraph(nullptr, &data), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigBuiltIn, SetSerializedGraphRejectsNullBufferStruct)
{
    EXPECT_EQ(configAbi().policySetSerializedGraph(_desc, nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== Finalize BAD_PARAM / NOT_INITIALIZED ==========

TEST(TestConfigBuiltInRejection, FinalizeRejectsNullDescriptor)
{
    int32_t applied = -1; // NOLINT(misc-const-correctness)
    EXPECT_EQ(configAbi().policyFinalize(nullptr, &applied), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigBuiltIn, FinalizeRejectsNullOutApplied)
{
    EXPECT_EQ(configAbi().policyFinalize(_desc, nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestConfigBuiltInRejection, GetSortedRejectsNullDescriptor)
{
    size_t count = 0; // NOLINT(misc-const-correctness)
    EXPECT_EQ(configAbi().policyGetSortedEngineIds(nullptr, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigBuiltIn, GetSortedRejectsNullCountPointer)
{
    EXPECT_EQ(configAbi().policyGetSortedEngineIds(_desc, nullptr, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigBuiltIn, GetSortedReturnsNotInitializedBeforeFinalize)
{
    size_t count = 0; // NOLINT(misc-const-correctness)
    EXPECT_EQ(configAbi().policyGetSortedEngineIds(_desc, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED);
}

// ========== End-to-end: miss paths decline so the policy loop continues ==========

TEST_F(TestConfigBuiltIn, FinalizeWithEmptyCandidatesDeclines)
{
    // Even with a valid env file, no candidates means the policy can't pick
    // anything — decline rather than producing an empty list.
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));
    EXPECT_FALSE(_plugin->finalize(_desc));
}

TEST_F(TestConfigBuiltIn, FinalizeWithNoEnvDeclines)
{
    // Make sure no override file leaks in from the environment.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter overrideEnv(OVERRIDE_ENV, "");

    setEngineIds({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID});
    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));
    EXPECT_FALSE(_plugin->finalize(_desc));
}

TEST_F(TestConfigBuiltIn, FinalizeWithMissingFileDeclines)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(
        OVERRIDE_ENV, "/nonexistent/path/hipdnn_no_such_file.json");

    setEngineIds({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));
    EXPECT_FALSE(_plugin->finalize(_desc));
}

TEST_F(TestConfigBuiltIn, FinalizeWithInvalidGraphBufferDeclines)
{
    // Garbage bytes large enough to clear the null check but fail FlatBuffers
    // verification — must be tolerated quietly so the policy loop still runs.
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    const std::vector<uint8_t> garbage(64, 0xFF);
    setEngineIds({MIOPEN_ENGINE_ID});
    setSerializedGraph(garbage);
    EXPECT_FALSE(_plugin->finalize(_desc));
}

TEST_F(TestConfigBuiltIn, FinalizeWithNoMatchingRuleDeclines)
{
    // Rule targets dim [99, 99, 99, 99] — no conv in the test graph matches.
    constexpr const char* JSON = R"({
      "engine_overrides": [
        {
          "op": "conv_fprop",
          "engine_name": "MIOPEN_ENGINE_DETERMINISTIC",
          "tensors": [
            { "dim": [99, 99, 99, 99] },
            { "dim": [99, 99, 99, 99] }
          ]
        }
      ]
    })";
    const TempJsonOverrideFile json(JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setEngineIds({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));
    EXPECT_FALSE(_plugin->finalize(_desc));
}

TEST_F(TestConfigBuiltIn, FinalizeWithMatchedEngineNotInCandidatesDeclines)
{
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    // Rule selects DETERMINISTIC; candidate list omits it.
    setEngineIds({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID});
    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));
    EXPECT_FALSE(_plugin->finalize(_desc));
}

TEST_F(TestConfigBuiltIn, FinalizeWithGraphMissingNodesDeclines)
{
    // Empty graph: nothing to walk; nothing matches. Decline.
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setEngineIds({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildEmptyGraphBuffer());
    EXPECT_FALSE(_plugin->finalize(_desc));
}

// ========== End-to-end: matching rule reorders candidates ==========

TEST_F(TestConfigBuiltIn, FinalizeMatchedRuleMovesEngineToFront)
{
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setEngineIds({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));

    ASSERT_TRUE(_plugin->finalize(_desc));
    const auto sorted = _plugin->getSortedEngineIds(_desc);
    ASSERT_EQ(sorted.size(), 3u);
    EXPECT_EQ(sorted[0], MIOPEN_DETERMINISTIC_ID);
    EXPECT_EQ(sorted[1], MIOPEN_ENGINE_ID);
    EXPECT_EQ(sorted[2], CUSTOM_ENGINE_ID);
}

TEST_F(TestConfigBuiltIn, FinalizeRereadsEnvOnEachInvocation)
{
    // loadFromEnv must not be process-cached: pointing the env at a different
    // file between invocations picks up the new rule.
    const TempJsonOverrideFile firstFile(DETERMINISTIC_RULE_JSON);
    constexpr const char* SECOND_RULE = R"({
      "engine_overrides": [
        {
          "op": "conv_fprop",
          "engine_name": "MIOPEN_ENGINE",
          "tensors": [
            { "dim": [1, 3, 4, 4] },
            { "dim": [2, 3, 1, 1] }
          ]
        }
      ]
    })";
    const TempJsonOverrideFile secondFile(SECOND_RULE);

    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV, firstFile.path());

    setEngineIds({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));

    ASSERT_TRUE(_plugin->finalize(_desc));
    {
        const auto sorted = _plugin->getSortedEngineIds(_desc);
        ASSERT_FALSE(sorted.empty());
        EXPECT_EQ(sorted.front(), MIOPEN_DETERMINISTIC_ID);
    }

    env.setValue(secondFile.path());

    // Rerun finalize — the rule from secondFile should win this time.
    setEngineIds({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));
    ASSERT_TRUE(_plugin->finalize(_desc));
    {
        const auto sorted = _plugin->getSortedEngineIds(_desc);
        ASSERT_FALSE(sorted.empty());
        EXPECT_EQ(sorted.front(), MIOPEN_ENGINE_ID);
    }
}
