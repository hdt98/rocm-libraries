// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestConfigPlugin.cpp
 * @brief Unit tests for the Config heuristic plugin C ABI (RFC 0007)
 *
 * Links the plugin's _private STATIC archive directly so the C ABI entry points
 * can be exercised without dlopen'ing the SHARED plugin. Covers handle and
 * descriptor lifecycle, BAD_PARAM branches on every entry point, the
 * applied=0 paths (empty candidates, missing graph, preferred not in
 * candidates, no preferred resolved), the applied=1 reorder path driven by
 * Graph.preferred_engine_id, and the HIPDNN_ENGINE_OVERRIDE_FILE path that
 * resolves a preferred engine by matching a JSON rule against a conv node.
 *
 * Rule-matching internals (op/dim/stride wildcards, first-match wins, JSON
 * parsing) are exercised in TestEngineOverrideConfig.cpp; this file covers the
 * end-to-end path from env var through Finalize to the reordered output.
 */

// HIPDNN_PLUGIN_STATIC_DEFINE / HIPDNN_HEURISTIC_PLUGIN_STATIC_DEFINE come
// from target_compile_definitions in this test's CMakeLists.txt.

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
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
#include <string_view>
#include <vector>

namespace
{

const int64_t MIOPEN_ENGINE_ID = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE");
const int64_t MIOPEN_DETERMINISTIC_ID
    = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE_DETERMINISTIC");
const int64_t CUSTOM_ENGINE_ID
    = hipdnn_data_sdk::utilities::engineNameToId("Plugin1::CustomEngine");

/// Build a minimal serialized Graph FlatBuffer. Pass a value to set
/// preferred_engine_id; pass std::nullopt to leave it unset. The Config plugin
/// only touches preferred_engine_id and (for the override-file path) the conv
/// nodes - an empty node list is enough to exercise the preferred-engine
/// branch.
std::vector<uint8_t> buildGraphBuffer(::flatbuffers::Optional<int64_t> preferredEngineId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        nullptr,
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        nullptr,
        nullptr,
        preferredEngineId);
    hipdnn_flatbuffers_sdk::data_objects::FinishGraphBuffer(builder, graphOffset);
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}

/// Build a serialized Graph with a single ConvolutionFwd node and the two
/// referenced tensors so the override-file matcher has something to walk.
/// preferred_engine_id is left unset (the override-file path is only consulted
/// when the explicit hint is absent).
std::vector<uint8_t> buildConvFwdGraphBuffer(const std::vector<int64_t>& xDims,
                                             const std::vector<int64_t>& xStrides,
                                             const std::vector<int64_t>& wDims,
                                             const std::vector<int64_t>& wStrides)
{
    namespace fb = hipdnn_flatbuffers_sdk::data_objects;
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

    const std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "conv", fb::DataType::FLOAT, fb::NodeAttributes::ConvolutionFwdAttributes,
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

/// RAII temp directory + JSON file. Returns a path the test can hand to
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
        // Per-process counter for uniqueness within a run; remove any stale
        // directory left over from a prior killed run so ScopedDirectory's
        // create_directory call doesn't throw on collision.
        static std::atomic<uint64_t> s_counter{0};
        const auto path = std::filesystem::temp_directory_path()
                          / ("hipdnn_test_override_" + std::to_string(s_counter.fetch_add(1)));
        std::filesystem::remove_all(path);
        return path;
    }

    hipdnn_test_sdk::utilities::ScopedDirectory _dir;
    std::filesystem::path _path;
};

class TestConfigPlugin : public ::testing::Test
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

    void setCandidates(const std::vector<int64_t>& ids)
    {
        ASSERT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, ids.data(), ids.size()),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
    }

    void setGraph(const std::vector<uint8_t>& buffer)
    {
        const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
        ASSERT_EQ(hipdnnHeuristicPolicySetSerializedGraph(_desc, &data),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
    }

    hipdnnHeuristicHandle_t _handle = nullptr;
    hipdnnHeuristicPolicyDescriptor_t _desc = nullptr;
};

} // namespace

// ========== Plugin Metadata ==========

TEST(TestConfigPluginMetadata, GetNameReturnsExpectedPolicyName)
{
    const char* name = nullptr;
    EXPECT_EQ(hipdnnPluginGetName(&name), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(name, "SelectionHeuristic::Config");
}

TEST(TestConfigPluginMetadata, GetNameRejectsNull)
{
    EXPECT_EQ(hipdnnPluginGetName(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestConfigPluginMetadata, GetVersionReturnsNonEmpty)
{
    const char* version = nullptr;
    EXPECT_EQ(hipdnnPluginGetVersion(&version), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(version, nullptr);
    EXPECT_GT(std::string_view(version).size(), 0u);
}

TEST(TestConfigPluginMetadata, GetVersionRejectsNull)
{
    EXPECT_EQ(hipdnnPluginGetVersion(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestConfigPluginMetadata, GetApiVersionReturnsNonEmpty)
{
    const char* version = nullptr;
    EXPECT_EQ(hipdnnPluginGetApiVersion(&version), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(version, nullptr);
    EXPECT_GT(std::string_view(version).size(), 0u);
}

TEST(TestConfigPluginMetadata, GetApiVersionRejectsNull)
{
    EXPECT_EQ(hipdnnPluginGetApiVersion(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestConfigPluginMetadata, GetTypeReturnsHeuristic)
{
    hipdnnPluginType_t type = HIPDNN_PLUGIN_TYPE_UNSPECIFIED;
    EXPECT_EQ(hipdnnPluginGetType(&type), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(type, HIPDNN_PLUGIN_TYPE_HEURISTIC);
}

TEST(TestConfigPluginMetadata, GetTypeRejectsNull)
{
    EXPECT_EQ(hipdnnPluginGetType(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== Logging ==========

TEST(TestConfigPluginLogging, SetLoggingCallbackSucceedsWithNullCallback)
{
    EXPECT_EQ(hipdnnPluginSetLoggingCallback(nullptr), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestConfigPluginLogging, SetLogLevelSucceeds)
{
    EXPECT_EQ(hipdnnPluginSetLogLevel(HIPDNN_SEV_WARN), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestConfigPluginLogging, GetLastErrorStringWithNullDoesNotCrash)
{
    EXPECT_NO_FATAL_FAILURE(hipdnnPluginGetLastErrorString(nullptr));
}

TEST(TestConfigPluginLogging, GetLastErrorStringReturnsNonNullMessage)
{
    const char* msg = nullptr;
    hipdnnPluginGetLastErrorString(&msg);
    EXPECT_NE(msg, nullptr);
}

// ========== Handle Lifecycle ==========

TEST(TestConfigPluginHandle, CreateAndDestroySucceed)
{
    hipdnnHeuristicHandle_t handle = nullptr;
    EXPECT_EQ(hipdnnHeuristicHandleCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_NE(handle, nullptr);
    EXPECT_EQ(hipdnnHeuristicHandleDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestConfigPluginHandle, CreateRejectsNullOutPointer)
{
    EXPECT_EQ(hipdnnHeuristicHandleCreate(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestConfigPluginHandle, DestroyRejectsNullHandle)
{
    EXPECT_EQ(hipdnnHeuristicHandleDestroy(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigPlugin, SetDevicePropertiesAcceptsNonEmptyBuffer)
{
    const std::array<uint8_t, 4> buffer{0xDE, 0xAD, 0xBE, 0xEF};
    const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
    EXPECT_EQ(hipdnnHeuristicHandleSetDeviceProperties(_handle, &data),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST_F(TestConfigPlugin, SetDevicePropertiesRejectsNullHandle)
{
    const std::array<uint8_t, 1> buffer{0x01};
    const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
    EXPECT_EQ(hipdnnHeuristicHandleSetDeviceProperties(nullptr, &data),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigPlugin, SetDevicePropertiesRejectsNullBufferStruct)
{
    EXPECT_EQ(hipdnnHeuristicHandleSetDeviceProperties(_handle, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigPlugin, SetDevicePropertiesRejectsNullBufferPointer)
{
    const hipdnnPluginConstData_t data{nullptr, 4};
    EXPECT_EQ(hipdnnHeuristicHandleSetDeviceProperties(_handle, &data),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigPlugin, SetDevicePropertiesRejectsZeroSize)
{
    const std::array<uint8_t, 1> buffer{0x01};
    const hipdnnPluginConstData_t data{buffer.data(), 0};
    EXPECT_EQ(hipdnnHeuristicHandleSetDeviceProperties(_handle, &data),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== Policy Descriptor Lifecycle ==========

TEST_F(TestConfigPlugin, DescriptorCreateRejectsNullHandle)
{
    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    EXPECT_EQ(hipdnnHeuristicPolicyDescriptorCreate(nullptr, &desc),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(desc, nullptr);
}

TEST_F(TestConfigPlugin, DescriptorCreateRejectsNullOutPointer)
{
    EXPECT_EQ(hipdnnHeuristicPolicyDescriptorCreate(_handle, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestConfigPluginDescriptor, DestroyRejectsNullDescriptor)
{
    EXPECT_EQ(hipdnnHeuristicPolicyDescriptorDestroy(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== SetEngineIds ==========

TEST_F(TestConfigPlugin, SetEngineIdsAcceptsValidIds)
{
    const std::array<int64_t, 2> ids{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID};
    EXPECT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, ids.data(), ids.size()),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST_F(TestConfigPlugin, SetEngineIdsAcceptsZeroCountWithNullPointer)
{
    EXPECT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, nullptr, 0),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestConfigPluginInputs, SetEngineIdsRejectsNullDescriptor)
{
    const std::array<int64_t, 1> ids{MIOPEN_ENGINE_ID};
    EXPECT_EQ(hipdnnHeuristicPolicySetEngineIds(nullptr, ids.data(), ids.size()),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigPlugin, SetEngineIdsRejectsNullPointerWithCount)
{
    EXPECT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, nullptr, 3),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== SetSerializedGraph ==========

TEST_F(TestConfigPlugin, SetSerializedGraphAcceptsValidGraphBuffer)
{
    const auto buffer = buildGraphBuffer(::flatbuffers::nullopt);
    const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
    EXPECT_EQ(hipdnnHeuristicPolicySetSerializedGraph(_desc, &data),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestConfigPluginInputs, SetSerializedGraphRejectsNullDescriptor)
{
    const std::array<uint8_t, 1> buffer{0x00};
    const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
    EXPECT_EQ(hipdnnHeuristicPolicySetSerializedGraph(nullptr, &data),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigPlugin, SetSerializedGraphRejectsNullBufferStruct)
{
    EXPECT_EQ(hipdnnHeuristicPolicySetSerializedGraph(_desc, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigPlugin, SetSerializedGraphRejectsNullBufferPointer)
{
    const hipdnnPluginConstData_t data{nullptr, 16};
    EXPECT_EQ(hipdnnHeuristicPolicySetSerializedGraph(_desc, &data),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== Finalize ==========

TEST(TestConfigPluginFinalize, RejectsNullDescriptor)
{
    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(nullptr, &applied), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigPlugin, FinalizeRejectsNullOutApplied)
{
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigPlugin, FinalizeWithNoCandidatesReportsNotApplied)
{
    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 0);
}

TEST_F(TestConfigPlugin, FinalizeWithoutGraphReportsNotApplied)
{
    setCandidates({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID});
    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 0);
}

TEST_F(TestConfigPlugin, FinalizeRejectsInvalidGraphBuffer)
{
    setCandidates({MIOPEN_ENGINE_ID});
    // Junk that is large enough to pass SetSerializedGraph's null check but
    // fails FlatBuffers verification.
    const std::vector<uint8_t> garbage(64, 0xFF);
    setGraph(garbage);

    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigPlugin, FinalizeWithoutPreferredEngineReportsNotApplied)
{
    setCandidates({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID});
    setGraph(buildGraphBuffer(::flatbuffers::nullopt));

    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 0);
}

TEST_F(TestConfigPlugin, FinalizeWithPreferredNotInCandidatesReportsNotApplied)
{
    setCandidates({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setGraph(buildGraphBuffer(CUSTOM_ENGINE_ID));

    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 0);
}

TEST_F(TestConfigPlugin, FinalizeMovesPreferredEngineToFrontPreservingOrder)
{
    // Candidate order chosen so the preferred (CUSTOM) is in the middle; the
    // others should keep their relative position behind it.
    setCandidates({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setGraph(buildGraphBuffer(CUSTOM_ENGINE_ID));

    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 1);

    size_t count = 0;
    ASSERT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(count, 3u);

    std::vector<int64_t> sorted(count);
    ASSERT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, sorted.data(), &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(sorted[0], CUSTOM_ENGINE_ID);
    EXPECT_EQ(sorted[1], MIOPEN_ENGINE_ID);
    EXPECT_EQ(sorted[2], MIOPEN_DETERMINISTIC_ID);
}

TEST_F(TestConfigPlugin, FinalizeResetsByLastSetEngineIdsCall)
{
    // Initially eligible: preferred is in the candidate set.
    setCandidates({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID});
    setGraph(buildGraphBuffer(CUSTOM_ENGINE_ID));

    // SetEngineIds resets the finalized flag; the new candidate set excludes
    // the preferred engine, so re-finalize must report not-applied.
    setCandidates({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});

    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 0);
}

// ========== GetSortedEngineIds ==========

TEST(TestConfigPluginGet, RejectsNullDescriptor)
{
    size_t count = 0;
    EXPECT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(nullptr, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigPlugin, GetRejectsNullCountPointer)
{
    EXPECT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, nullptr, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigPlugin, GetReturnsNotInitializedBeforeFinalize)
{
    size_t count = 0;
    EXPECT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestConfigPlugin, GetReturnsNotInitializedWhenFinalizeReportsNotApplied)
{
    // Finalize succeeds with applied=0 (no graph); descriptor stays
    // not-finalized so Get must refuse to return data.
    setCandidates({MIOPEN_ENGINE_ID});
    int32_t applied = -1;
    ASSERT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(applied, 0);

    size_t count = 0;
    EXPECT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestConfigPlugin, GetClipsToCallerProvidedCapacity)
{
    setCandidates({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setGraph(buildGraphBuffer(CUSTOM_ENGINE_ID));

    int32_t applied = -1;
    ASSERT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(applied, 1);

    std::array<int64_t, 2> outBuf{0, 0};
    size_t count = outBuf.size();
    EXPECT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, outBuf.data(), &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(count, outBuf.size());
    EXPECT_EQ(outBuf[0], CUSTOM_ENGINE_ID);
    EXPECT_EQ(outBuf[1], MIOPEN_ENGINE_ID);
}

// ========== HIPDNN_ENGINE_OVERRIDE_FILE end-to-end ==========
//
// These tests drive the override-file branch of Finalize: when
// graph.preferred_engine_id is unset, the plugin reads
// HIPDNN_ENGINE_OVERRIDE_FILE, parses the JSON, and matches each conv-like
// node against the rules. ScopedEnvironmentVariableSetter restores the
// original env var on test exit; TempJsonOverrideFile cleans up the file.

namespace
{
constexpr const char* OVERRIDE_ENV = "HIPDNN_ENGINE_OVERRIDE_FILE";

// A rule that matches an x:[1,3,4,4] / w:[2,3,1,1] conv_fprop and selects
// MIOPEN_ENGINE_DETERMINISTIC.
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

TEST_F(TestConfigPlugin, OverrideFileMovesMatchedEngineToFront)
{
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setCandidates({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));

    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 1);

    size_t count = 0;
    ASSERT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    std::vector<int64_t> sorted(count);
    ASSERT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, sorted.data(), &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(sorted.size(), 3u);
    EXPECT_EQ(sorted[0], MIOPEN_DETERMINISTIC_ID);
    EXPECT_EQ(sorted[1], MIOPEN_ENGINE_ID);
    EXPECT_EQ(sorted[2], CUSTOM_ENGINE_ID);
}

TEST_F(TestConfigPlugin, ExplicitPreferredEngineWinsOverOverrideFile)
{
    // The override file would select MIOPEN_DETERMINISTIC, but the explicit
    // graph.preferred_engine_id hint must take precedence and short-circuit
    // before the override file is consulted.
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setCandidates({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    // Graph carries the explicit hint AND has no conv nodes - the override
    // path would silently no-op even if reached, which would let a regression
    // (consulting the file anyway) fall back to applied=1 on the wrong engine.
    setGraph(buildGraphBuffer(CUSTOM_ENGINE_ID));

    int32_t applied = -1;
    ASSERT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(applied, 1);

    size_t count = 0;
    ASSERT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    std::vector<int64_t> sorted(count);
    ASSERT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, sorted.data(), &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(sorted.front(), CUSTOM_ENGINE_ID);
}

TEST_F(TestConfigPlugin, OverrideFileWithNoMatchingRuleReportsNotApplied)
{
    // Rule targets dim [99, 99, 99, 99] - no conv in the test graph matches.
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

    setCandidates({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));

    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 0);
}

TEST_F(TestConfigPlugin, OverrideFilePathPointsAtMissingFileReportsNotApplied)
{
    // Env var set but file doesn't exist - loadFromEnv returns nullopt,
    // matchOverrideConfig returns nullopt, Finalize reports applied=0.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(
        OVERRIDE_ENV, "/nonexistent/path/hipdnn_no_such_file.json");

    setCandidates({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));

    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 0);
}

TEST_F(TestConfigPlugin, OverrideFileMatchedEngineNotInCandidatesReportsNotApplied)
{
    // Rule selects MIOPEN_DETERMINISTIC but candidates exclude it.
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setCandidates({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID});
    setGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));

    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 0);
}

TEST_F(TestConfigPlugin, OverrideFileRereadOnEachFinalize)
{
    // Confirms loadFromEnv is no longer process-cached: pointing the env at a
    // different file between Finalize calls picks up the new rule. Same desc
    // is reused (SetEngineIds re-finalizes); the second Finalize must use the
    // newly-pointed file, not the first one.
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

    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                    firstFile.path());

    setCandidates({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));

    int32_t applied = -1;
    ASSERT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(applied, 1);
    {
        size_t count = 0;
        ASSERT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, nullptr, &count),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
        std::vector<int64_t> sorted(count);
        ASSERT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, sorted.data(), &count),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
        EXPECT_EQ(sorted.front(), MIOPEN_DETERMINISTIC_ID);
    }

    // Repoint env, re-finalize via SetEngineIds, expect the new selection.
    env.setValue(secondFile.path());
    setCandidates({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});

    ASSERT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(applied, 1);
    {
        size_t count = 0;
        ASSERT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, nullptr, &count),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
        std::vector<int64_t> sorted(count);
        ASSERT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, sorted.data(), &count),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
        EXPECT_EQ(sorted.front(), MIOPEN_ENGINE_ID);
    }
}
