// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestStaticOrderingBuiltIn.cpp
 * @brief Tests for the SelectionHeuristic::StaticOrdering built-in.
 *
 * The built-in lives inside hipdnn_backend_private as a function-pointer table
 * (StaticOrderingBuiltIn::populateFunctionTable) wrapped by HeuristicPlugin via
 * createBuiltIn. There is no .so to dlopen; the wrapper reaches the same code
 * paths used in production registration through HeuristicPluginManager.
 *
 * Wraps the table once via HeuristicPlugin::createBuiltIn and exercises the
 * wrapper API (createHandle, createPolicyDescriptor, setEngineIds, finalize,
 * getSortedEngineIds). BAD_PARAM rejection paths in the C-ABI shape are driven
 * by calling the populated function-table entries directly so test code can
 * pass nullptr arguments without the wrapper translating them into exceptions.
 */

#include "heuristics/static_ordering/StaticOrderingBuiltIn.hpp"
#include "plugin/HeuristicPlugin.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

using hipdnn_backend::plugin::HeuristicPlugin;
using hipdnn_backend::plugin::HeuristicPluginFunctionTable;
using hipdnn_backend::heuristics::static_ordering::populateFunctionTable;

namespace
{

const int64_t MIOPEN_ENGINE_ID = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE");
const int64_t MIOPEN_DETERMINISTIC_ID
    = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE_DETERMINISTIC");
const int64_t CUSTOM_ENGINE_ID
    = hipdnn_data_sdk::utilities::engineNameToId("Plugin1::CustomEngine");

const int64_t STATIC_ORDERING_POLICY_ID
    = hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering");

class TestStaticOrderingBuiltIn : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _plugin = HeuristicPlugin::createBuiltIn(populateFunctionTable(),
                                                 "built-in:StaticOrdering-test");
        _handle = _plugin->createHandle();
        ASSERT_NE(_handle, nullptr);
        _desc = _plugin->createPolicyDescriptor(_handle, STATIC_ORDERING_POLICY_ID);
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

    std::shared_ptr<HeuristicPlugin>  _plugin;
    hipdnnHeuristicHandle_t           _handle = nullptr;
    hipdnnHeuristicPolicyDescriptor_t _desc   = nullptr;
};

// Convenience: grab the raw function table once for direct C-ABI rejection tests.
// Named to avoid clash with the global `abi` namespace alias from <cxxabi.h>.
const HeuristicPluginFunctionTable& staticOrderingAbi()
{
    static const HeuristicPluginFunctionTable s_funcs = populateFunctionTable();
    return s_funcs;
}

} // namespace

// ========== Built-in metadata exposed via the wrapper ==========

TEST_F(TestStaticOrderingBuiltIn, ReportsHeuristicPluginType)
{
    EXPECT_EQ(_plugin->type(), HIPDNN_PLUGIN_TYPE_HEURISTIC);
}

TEST_F(TestStaticOrderingBuiltIn, EnumeratesSingleStaticOrderingPolicy)
{
    const auto ids = _plugin->getAllPolicyIds();
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], STATIC_ORDERING_POLICY_ID);
    EXPECT_EQ(_plugin->getPolicyName(STATIC_ORDERING_POLICY_ID),
              "SelectionHeuristic::StaticOrdering");
}

// ========== Policy Descriptor Lifecycle (BAD_PARAM via raw ABI) ==========

TEST(TestStaticOrderingBuiltInRejection, DescriptorCreateRejectsNullHandle)
{
    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    EXPECT_EQ(staticOrderingAbi().policyDescriptorCreate(nullptr, STATIC_ORDERING_POLICY_ID, &desc),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(desc, nullptr);
}

TEST_F(TestStaticOrderingBuiltIn, DescriptorCreateRejectsNullOutPointer)
{
    EXPECT_EQ(staticOrderingAbi().policyDescriptorCreate(_handle, STATIC_ORDERING_POLICY_ID, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestStaticOrderingBuiltInRejection, DescriptorDestroyRejectsNullDescriptor)
{
    EXPECT_EQ(staticOrderingAbi().policyDescriptorDestroy(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingBuiltIn, DescriptorCreateRejectsUnknownPolicyId)
{
    // The built-in only owns SelectionHeuristic::StaticOrdering; any other ID
    // (here: a different policy name's hash) must be rejected with BAD_PARAM
    // and must not allocate an output descriptor.
    const int64_t unknownId = hipdnn_data_sdk::utilities::policyNameToId("Vendor::NotARealPolicy");
    ASSERT_NE(unknownId, STATIC_ORDERING_POLICY_ID);

    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    EXPECT_EQ(staticOrderingAbi().policyDescriptorCreate(_handle, unknownId, &desc),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(desc, nullptr);
}

TEST_F(TestStaticOrderingBuiltIn, GetPolicyNameRejectsUnknownPolicyId)
{
    const int64_t unknownId = hipdnn_data_sdk::utilities::policyNameToId("Vendor::NotARealPolicy");
    ASSERT_NE(unknownId, STATIC_ORDERING_POLICY_ID);

    const char* name = nullptr;
    EXPECT_EQ(staticOrderingAbi().getPolicyName(unknownId, &name), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(name, nullptr);
}

// ========== SetEngineIds ==========

TEST_F(TestStaticOrderingBuiltIn, SetEngineIdsAcceptsValidIds)
{
    const std::array<int64_t, 2> ids{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID};
    EXPECT_NO_THROW(_plugin->setEngineIds(_desc, ids.data(), ids.size()));
}

TEST_F(TestStaticOrderingBuiltIn, SetEngineIdsAcceptsZeroCountWithNullPointer)
{
    EXPECT_NO_THROW(_plugin->setEngineIds(_desc, nullptr, 0));
}

TEST(TestStaticOrderingBuiltInRejection, SetEngineIdsRejectsNullDescriptor)
{
    const std::array<int64_t, 1> ids{MIOPEN_ENGINE_ID};
    EXPECT_EQ(staticOrderingAbi().policySetEngineIds(nullptr, ids.data(), ids.size()),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingBuiltIn, SetEngineIdsRejectsNullPointerWithCount)
{
    EXPECT_EQ(staticOrderingAbi().policySetEngineIds(_desc, nullptr, 3), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== SetSerializedGraph ==========
// StaticOrdering ignores the graph contents but validates the parameter shape.

TEST_F(TestStaticOrderingBuiltIn, SetSerializedGraphAcceptsAnyBuffer)
{
    const std::array<uint8_t, 3> buffer{0x01, 0x02, 0x03};
    const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
    EXPECT_NO_THROW(_plugin->setSerializedGraph(_desc, &data));
}

TEST(TestStaticOrderingBuiltInRejection, SetSerializedGraphRejectsNullDescriptor)
{
    const std::array<uint8_t, 1> buffer{0x00};
    const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
    EXPECT_EQ(staticOrderingAbi().policySetSerializedGraph(nullptr, &data), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingBuiltIn, SetSerializedGraphRejectsNullBufferStruct)
{
    EXPECT_EQ(staticOrderingAbi().policySetSerializedGraph(_desc, nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== Finalize ==========

TEST(TestStaticOrderingBuiltInRejection, FinalizeRejectsNullDescriptor)
{
    // applied is non-const because policyFinalize takes int32_t* — clang-tidy
    // would still flag a never-modified local, so suppress with NOLINT here.
    int32_t applied = -1; // NOLINT(misc-const-correctness)
    EXPECT_EQ(staticOrderingAbi().policyFinalize(nullptr, &applied), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingBuiltIn, FinalizeRejectsNullOutApplied)
{
    EXPECT_EQ(staticOrderingAbi().policyFinalize(_desc, nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingBuiltIn, FinalizeWithNoCandidatesReportsNotApplied)
{
    EXPECT_FALSE(_plugin->finalize(_desc));
}

TEST_F(TestStaticOrderingBuiltIn, FinalizeSortsCandidatesAndReportsApplied)
{
    const std::array<int64_t, 3> ids{MIOPEN_DETERMINISTIC_ID, CUSTOM_ENGINE_ID, MIOPEN_ENGINE_ID};
    _plugin->setEngineIds(_desc, ids.data(), ids.size());

    EXPECT_TRUE(_plugin->finalize(_desc));

    const auto sorted = _plugin->getSortedEngineIds(_desc);
    ASSERT_EQ(sorted.size(), ids.size());
    EXPECT_EQ(sorted.front(), MIOPEN_ENGINE_ID);
    EXPECT_EQ(sorted.back(), MIOPEN_DETERMINISTIC_ID);
}

TEST_F(TestStaticOrderingBuiltIn, FinalizeResetsByLastSetEngineIdsCall)
{
    const std::array<int64_t, 1> ids{MIOPEN_ENGINE_ID};
    _plugin->setEngineIds(_desc, ids.data(), ids.size());
    _plugin->setEngineIds(_desc, nullptr, 0);

    EXPECT_FALSE(_plugin->finalize(_desc));
}

// ========== GetSortedEngineIds ==========

TEST(TestStaticOrderingBuiltInRejection, GetSortedRejectsNullDescriptor)
{
    size_t count = 0; // NOLINT(misc-const-correctness) — passed by-pointer, mutable required
    EXPECT_EQ(staticOrderingAbi().policyGetSortedEngineIds(nullptr, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingBuiltIn, GetSortedRejectsNullCountPointer)
{
    EXPECT_EQ(staticOrderingAbi().policyGetSortedEngineIds(_desc, nullptr, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingBuiltIn, GetSortedReturnsNotInitializedBeforeFinalize)
{
    size_t count = 0; // NOLINT(misc-const-correctness) — passed by-pointer, mutable required
    EXPECT_EQ(staticOrderingAbi().policyGetSortedEngineIds(_desc, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestStaticOrderingBuiltIn, GetSortedClipsToCallerProvidedCapacity)
{
    const std::array<int64_t, 3> ids{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID};
    _plugin->setEngineIds(_desc, ids.data(), ids.size());
    ASSERT_TRUE(_plugin->finalize(_desc));

    std::array<int64_t, 2> outBuf{0, 0};
    size_t                 count = outBuf.size();
    EXPECT_EQ(staticOrderingAbi().policyGetSortedEngineIds(_desc, outBuf.data(), &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(count, outBuf.size());
    EXPECT_EQ(outBuf[0], MIOPEN_ENGINE_ID);
    EXPECT_EQ(outBuf[1], CUSTOM_ENGINE_ID);
}
