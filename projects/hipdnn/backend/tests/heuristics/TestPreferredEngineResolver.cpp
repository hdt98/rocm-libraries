// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestPreferredEngineResolver.cpp
 * @brief Tests for the backend preferred-engine precursor that runs ahead of
 *        the heuristic policy loop in EngineHeuristicDescriptor::finalize().
 *
 * Covers the explicit Graph.preferred_engine_id branch, the
 * HIPDNN_ENGINE_OVERRIDE_FILE conv-walk branch, the precedence between the
 * two, and the miss paths (no preferred engine, preferred not in candidates,
 * invalid graph buffer).
 */

#include "heuristics/preferred_engine/PreferredEngineResolver.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fb = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_backend::heuristics::preferred_engine::resolvePreferredEngineOrder;
using hipdnn_data_sdk::utilities::engineNameToId;

namespace
{

const int64_t MIOPEN_ENGINE_ID         = engineNameToId("MIOPEN_ENGINE");
const int64_t MIOPEN_DETERMINISTIC_ID  = engineNameToId("MIOPEN_ENGINE_DETERMINISTIC");
const int64_t CUSTOM_ENGINE_ID         = engineNameToId("Plugin1::CustomEngine");

constexpr const char* OVERRIDE_ENV        = "HIPDNN_ENGINE_OVERRIDE_FILE";
constexpr const char* DEFAULT_ENGINE_ENV  = "HIPDNN_DEFAULT_ENGINE";

/// Build a minimal serialized Graph FlatBuffer with no nodes. Pass a value to
/// set preferred_engine_id; pass nullopt to leave it unset.
std::vector<uint8_t> buildGraphBuffer(::flatbuffers::Optional<int64_t> preferredEngineId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto graphOffset = fb::CreateGraphDirect(builder,
                                             nullptr,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             nullptr,
                                             nullptr,
                                             preferredEngineId);
    fb::FinishGraphBuffer(builder, graphOffset);
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}

/// Build a serialized Graph with a single ConvolutionFwd node referencing
/// (x, w) tensors of the requested shapes. preferred_engine_id is left unset.
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

hipdnnPluginConstData_t toConstData(const std::vector<uint8_t>& buffer)
{
    return hipdnnPluginConstData_t{buffer.data(), buffer.size()};
}

/// RAII temp directory + JSON file. Returns a path that can be assigned to
/// HIPDNN_ENGINE_OVERRIDE_FILE; the directory is removed on destruction.
class TempJsonOverrideFile
{
public:
    explicit TempJsonOverrideFile(const std::string& contents)
        : _dir(makeUniqueDir())
        , _path(_dir.path() / "override.json")
    {
        std::ofstream(_path) << contents;
    }

    std::string path() const { return _path.string(); }

private:
    static std::filesystem::path makeUniqueDir()
    {
        static std::atomic<uint64_t> s_counter{0};
        const auto path = std::filesystem::temp_directory_path()
                          / ("hipdnn_test_resolver_" + std::to_string(s_counter.fetch_add(1)));
        std::filesystem::remove_all(path);
        return path;
    }

    hipdnn_test_sdk::utilities::ScopedDirectory _dir;
    std::filesystem::path                       _path;
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

} // namespace

// ========== Miss-path / null inputs ==========

TEST(TestPreferredEngineResolver, EmptyCandidatesReturnsNullopt)
{
    const auto buffer = buildGraphBuffer(CUSTOM_ENGINE_ID);
    EXPECT_FALSE(resolvePreferredEngineOrder(toConstData(buffer), {}).has_value());
}

TEST(TestPreferredEngineResolver, NullSerializedGraphReturnsNullopt)
{
    const hipdnnPluginConstData_t empty{nullptr, 0};
    const std::vector<int64_t>    candidates{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID};
    EXPECT_FALSE(resolvePreferredEngineOrder(empty, candidates).has_value());
}

TEST(TestPreferredEngineResolver, InvalidGraphBufferReturnsNullopt)
{
    // Garbage bytes large enough to clear the null check but fail FlatBuffers
    // verification — must be tolerated quietly so the policy loop still runs.
    const std::vector<uint8_t>    garbage(64, 0xFF);
    const std::vector<int64_t>    candidates{MIOPEN_ENGINE_ID};
    EXPECT_FALSE(resolvePreferredEngineOrder(toConstData(garbage), candidates).has_value());
}

TEST(TestPreferredEngineResolver, NoPreferredAndNoEnvOverrideReturnsNullopt)
{
    // Make sure no override file or default engine leaks in from the environment.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter overrideEnv(OVERRIDE_ENV, "");
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter defaultEnv(
        DEFAULT_ENGINE_ENV, "");

    const auto buffer = buildGraphBuffer(::flatbuffers::nullopt);
    const std::vector<int64_t> candidates{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID};
    EXPECT_FALSE(resolvePreferredEngineOrder(toConstData(buffer), candidates).has_value());
}

// ========== Explicit Graph.preferred_engine_id ==========

TEST(TestPreferredEngineResolver, ExplicitPreferredEngineMovesToFrontPreservingOrder)
{
    const auto buffer = buildGraphBuffer(CUSTOM_ENGINE_ID);
    const std::vector<int64_t> candidates{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID};

    const auto result = resolvePreferredEngineOrder(toConstData(buffer), candidates);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 3u);
    EXPECT_EQ((*result)[0], CUSTOM_ENGINE_ID);
    EXPECT_EQ((*result)[1], MIOPEN_ENGINE_ID);
    EXPECT_EQ((*result)[2], MIOPEN_DETERMINISTIC_ID);
}

TEST(TestPreferredEngineResolver, ExplicitPreferredNotInCandidatesReturnsNullopt)
{
    const auto buffer = buildGraphBuffer(CUSTOM_ENGINE_ID);
    const std::vector<int64_t> candidates{MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID};
    EXPECT_FALSE(resolvePreferredEngineOrder(toConstData(buffer), candidates).has_value());
}

TEST(TestPreferredEngineResolver, ExplicitPreferredWinsOverOverrideFile)
{
    // The override file would select MIOPEN_DETERMINISTIC but the explicit
    // Graph.preferred_engine_id hint must take precedence and short-circuit
    // before the file is consulted. The explicit graph carries no conv nodes,
    // so a regression that fell through to the file path would silently miss
    // and return nullopt — which would fail the assertions below.
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    const auto buffer = buildGraphBuffer(CUSTOM_ENGINE_ID);
    const std::vector<int64_t> candidates{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID};

    const auto result = resolvePreferredEngineOrder(toConstData(buffer), candidates);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->front(), CUSTOM_ENGINE_ID);
}

// ========== HIPDNN_ENGINE_OVERRIDE_FILE branch ==========

TEST(TestPreferredEngineResolver, OverrideFileMatchedRuleMovesEngineToFront)
{
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    const auto buffer = buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES);
    const std::vector<int64_t> candidates{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID};

    const auto result = resolvePreferredEngineOrder(toConstData(buffer), candidates);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 3u);
    EXPECT_EQ((*result)[0], MIOPEN_DETERMINISTIC_ID);
    EXPECT_EQ((*result)[1], MIOPEN_ENGINE_ID);
    EXPECT_EQ((*result)[2], CUSTOM_ENGINE_ID);
}

TEST(TestPreferredEngineResolver, OverrideFileNoMatchingRuleReturnsNullopt)
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

    const auto buffer = buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES);
    const std::vector<int64_t> candidates{MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID};
    EXPECT_FALSE(resolvePreferredEngineOrder(toConstData(buffer), candidates).has_value());
}

TEST(TestPreferredEngineResolver, OverrideFileMissingPathReturnsNullopt)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(
        OVERRIDE_ENV, "/nonexistent/path/hipdnn_no_such_file.json");

    const auto buffer = buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES);
    const std::vector<int64_t> candidates{MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID};
    EXPECT_FALSE(resolvePreferredEngineOrder(toConstData(buffer), candidates).has_value());
}

TEST(TestPreferredEngineResolver, OverrideFileMatchedEngineNotInCandidatesReturnsNullopt)
{
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    const auto buffer = buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES);
    const std::vector<int64_t> candidates{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID};
    EXPECT_FALSE(resolvePreferredEngineOrder(toConstData(buffer), candidates).has_value());
}

TEST(TestPreferredEngineResolver, OverrideFileRereadOnEachInvocation)
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

    const auto buffer = buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES);
    const std::vector<int64_t> candidates{MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID};

    {
        const auto first = resolvePreferredEngineOrder(toConstData(buffer), candidates);
        ASSERT_TRUE(first.has_value());
        EXPECT_EQ(first->front(), MIOPEN_DETERMINISTIC_ID);
    }

    env.setValue(secondFile.path());

    {
        const auto second = resolvePreferredEngineOrder(toConstData(buffer), candidates);
        ASSERT_TRUE(second.has_value());
        EXPECT_EQ(second->front(), MIOPEN_ENGINE_ID);
    }
}

// ========== HIPDNN_DEFAULT_ENGINE branch ==========

TEST(TestPreferredEngineResolver, DefaultEngineEnvMovesEngineToFront)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter overrideEnv(OVERRIDE_ENV, "");
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter defaultEnv(
        DEFAULT_ENGINE_ENV, "MIOPEN_ENGINE_DETERMINISTIC");

    const auto buffer = buildGraphBuffer(::flatbuffers::nullopt);
    const std::vector<int64_t> candidates{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID};

    const auto result = resolvePreferredEngineOrder(toConstData(buffer), candidates);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 3u);
    EXPECT_EQ((*result)[0], MIOPEN_DETERMINISTIC_ID);
    EXPECT_EQ((*result)[1], MIOPEN_ENGINE_ID);
    EXPECT_EQ((*result)[2], CUSTOM_ENGINE_ID);
}

TEST(TestPreferredEngineResolver, DefaultEngineEnvBlankIsTreatedAsUnset)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter overrideEnv(OVERRIDE_ENV, "");
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter defaultEnv(
        DEFAULT_ENGINE_ENV, "   ");

    const auto buffer = buildGraphBuffer(::flatbuffers::nullopt);
    const std::vector<int64_t> candidates{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID};
    EXPECT_FALSE(resolvePreferredEngineOrder(toConstData(buffer), candidates).has_value());
}

TEST(TestPreferredEngineResolver, DefaultEngineEnvNotInCandidatesReturnsNullopt)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter overrideEnv(OVERRIDE_ENV, "");
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter defaultEnv(
        DEFAULT_ENGINE_ENV, "MIOPEN_ENGINE_DETERMINISTIC");

    const auto buffer = buildGraphBuffer(::flatbuffers::nullopt);
    const std::vector<int64_t> candidates{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID};
    EXPECT_FALSE(resolvePreferredEngineOrder(toConstData(buffer), candidates).has_value());
}

TEST(TestPreferredEngineResolver, ExplicitPreferredWinsOverDefaultEngineEnv)
{
    // The default engine env would select MIOPEN_DETERMINISTIC, but the explicit
    // Graph.preferred_engine_id hint must take precedence.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter overrideEnv(OVERRIDE_ENV, "");
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter defaultEnv(
        DEFAULT_ENGINE_ENV, "MIOPEN_ENGINE_DETERMINISTIC");

    const auto buffer = buildGraphBuffer(CUSTOM_ENGINE_ID);
    const std::vector<int64_t> candidates{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID};

    const auto result = resolvePreferredEngineOrder(toConstData(buffer), candidates);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->front(), CUSTOM_ENGINE_ID);
}

TEST(TestPreferredEngineResolver, OverrideFileWinsOverDefaultEngineEnv)
{
    // The override file selects MIOPEN_DETERMINISTIC for the conv. The default
    // engine env points at MIOPEN_ENGINE; the file rule must take precedence.
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter overrideEnv(OVERRIDE_ENV,
                                                                                  json.path());
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter defaultEnv(
        DEFAULT_ENGINE_ENV, "MIOPEN_ENGINE");

    const auto buffer = buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES);
    const std::vector<int64_t> candidates{MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID};

    const auto result = resolvePreferredEngineOrder(toConstData(buffer), candidates);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->front(), MIOPEN_DETERMINISTIC_ID);
}

TEST(TestPreferredEngineResolver, DefaultEngineEnvRereadOnEachInvocation)
{
    // Mirrors OverrideFileRereadOnEachInvocation: per-call env reads let tests
    // (and operators) flip HIPDNN_DEFAULT_ENGINE between invocations.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter overrideEnv(OVERRIDE_ENV, "");
    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter defaultEnv(DEFAULT_ENGINE_ENV,
                                                                           "MIOPEN_ENGINE");

    const auto buffer = buildGraphBuffer(::flatbuffers::nullopt);
    const std::vector<int64_t> candidates{MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID};

    {
        const auto first = resolvePreferredEngineOrder(toConstData(buffer), candidates);
        ASSERT_TRUE(first.has_value());
        EXPECT_EQ(first->front(), MIOPEN_ENGINE_ID);
    }

    defaultEnv.setValue("MIOPEN_ENGINE_DETERMINISTIC");

    {
        const auto second = resolvePreferredEngineOrder(toConstData(buffer), candidates);
        ASSERT_TRUE(second.has_value());
        EXPECT_EQ(second->front(), MIOPEN_DETERMINISTIC_ID);
    }
}
