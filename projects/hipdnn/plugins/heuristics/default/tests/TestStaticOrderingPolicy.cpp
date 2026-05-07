// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestStaticOrderingPolicy.cpp
 * @brief Per-policy tests for the SelectionHeuristic::StaticOrdering policy
 *        hosted inside the merged DefaultHeuristicsPlugin.
 *
 * Links the plugin's _private STATIC archive directly so the C ABI entry points
 * can be exercised without dlopen'ing the SHARED plugin. Covers descriptor
 * lifecycle, BAD_PARAM branches, the applied=0 empty-list path, and the
 * GetSortedEngineIds two-call query and clipping behaviors.
 *
 * Plugin-level metadata, logging, handle lifecycle, and policy enumeration live
 * in TestDefaultHeuristicsPlugin.cpp. The ordering algorithm itself
 * (utilities::sortEngineIds) is exercised by
 * backend/tests/utilities/TestEngineOrdering.cpp; this file focuses on the
 * policy's wrapper logic.
 */

// HIPDNN_PLUGIN_STATIC_DEFINE / HIPDNN_HEURISTIC_PLUGIN_STATIC_DEFINE come
// from target_compile_definitions in this test's CMakeLists.txt.

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

namespace
{

const int64_t MIOPEN_ENGINE_ID = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE");
const int64_t MIOPEN_DETERMINISTIC_ID
    = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE_DETERMINISTIC");
const int64_t CUSTOM_ENGINE_ID
    = hipdnn_data_sdk::utilities::engineNameToId("Plugin1::CustomEngine");

const int64_t STATIC_ORDERING_POLICY_ID
    = hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering");

class TestStaticOrderingPolicy : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hipdnnHeuristicHandleCreate(&_handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
        ASSERT_NE(_handle, nullptr);
        ASSERT_EQ(hipdnnHeuristicPolicyDescriptorCreate(_handle, STATIC_ORDERING_POLICY_ID, &_desc),
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

    hipdnnHeuristicHandle_t _handle = nullptr;
    hipdnnHeuristicPolicyDescriptor_t _desc = nullptr;
};

} // namespace

// ========== Policy Descriptor Lifecycle ==========

TEST_F(TestStaticOrderingPolicy, DescriptorCreateRejectsNullHandle)
{
    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    EXPECT_EQ(hipdnnHeuristicPolicyDescriptorCreate(nullptr, STATIC_ORDERING_POLICY_ID, &desc),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(desc, nullptr);
}

TEST_F(TestStaticOrderingPolicy, DescriptorCreateRejectsNullOutPointer)
{
    EXPECT_EQ(hipdnnHeuristicPolicyDescriptorCreate(_handle, STATIC_ORDERING_POLICY_ID, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestStaticOrderingPolicyDescriptor, DestroyRejectsNullDescriptor)
{
    EXPECT_EQ(hipdnnHeuristicPolicyDescriptorDestroy(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== SetEngineIds ==========

TEST_F(TestStaticOrderingPolicy, SetEngineIdsAcceptsValidIds)
{
    const std::array<int64_t, 2> ids{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID};
    EXPECT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, ids.data(), ids.size()),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST_F(TestStaticOrderingPolicy, SetEngineIdsAcceptsZeroCountWithNullPointer)
{
    EXPECT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, nullptr, 0), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestStaticOrderingPolicyInputs, SetEngineIdsRejectsNullDescriptor)
{
    const std::array<int64_t, 1> ids{MIOPEN_ENGINE_ID};
    EXPECT_EQ(hipdnnHeuristicPolicySetEngineIds(nullptr, ids.data(), ids.size()),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingPolicy, SetEngineIdsRejectsNullPointerWithCount)
{
    EXPECT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, nullptr, 3), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== SetSerializedGraph ==========

// StaticOrdering ignores the graph contents but validates the parameter shape.

TEST_F(TestStaticOrderingPolicy, SetSerializedGraphAcceptsAnyBuffer)
{
    const std::array<uint8_t, 3> buffer{0x01, 0x02, 0x03};
    const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
    EXPECT_EQ(hipdnnHeuristicPolicySetSerializedGraph(_desc, &data), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestStaticOrderingPolicyInputs, SetSerializedGraphRejectsNullDescriptor)
{
    const std::array<uint8_t, 1> buffer{0x00};
    const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
    EXPECT_EQ(hipdnnHeuristicPolicySetSerializedGraph(nullptr, &data),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingPolicy, SetSerializedGraphRejectsNullBufferStruct)
{
    EXPECT_EQ(hipdnnHeuristicPolicySetSerializedGraph(_desc, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== Finalize ==========

TEST(TestStaticOrderingPolicyFinalize, RejectsNullDescriptor)
{
    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(nullptr, &applied), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingPolicy, FinalizeRejectsNullOutApplied)
{
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingPolicy, FinalizeWithNoCandidatesReportsNotApplied)
{
    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 0);
}

TEST_F(TestStaticOrderingPolicy, FinalizeSortsCandidatesAndReportsApplied)
{
    const std::array<int64_t, 3> ids{MIOPEN_DETERMINISTIC_ID, CUSTOM_ENGINE_ID, MIOPEN_ENGINE_ID};
    ASSERT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, ids.data(), ids.size()),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 1);

    size_t count = 0;
    ASSERT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(count, ids.size());

    std::vector<int64_t> sorted(count);
    ASSERT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, sorted.data(), &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(sorted.front(), MIOPEN_ENGINE_ID);
    EXPECT_EQ(sorted.back(), MIOPEN_DETERMINISTIC_ID);
}

TEST_F(TestStaticOrderingPolicy, FinalizeResetsByLastSetEngineIdsCall)
{
    // First call with one set, then re-set with empty: should report not-applied
    const std::array<int64_t, 1> ids{MIOPEN_ENGINE_ID};
    ASSERT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, ids.data(), ids.size()),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, nullptr, 0), HIPDNN_PLUGIN_STATUS_SUCCESS);

    int32_t applied = -1;
    EXPECT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 0);
}

// ========== GetSortedEngineIds ==========

TEST(TestStaticOrderingPolicyGet, RejectsNullDescriptor)
{
    size_t count = 0;
    EXPECT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(nullptr, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingPolicy, GetRejectsNullCountPointer)
{
    EXPECT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, nullptr, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestStaticOrderingPolicy, GetReturnsNotInitializedBeforeFinalize)
{
    size_t count = 0;
    EXPECT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestStaticOrderingPolicy, GetClipsToCallerProvidedCapacity)
{
    const std::array<int64_t, 3> ids{MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID};
    ASSERT_EQ(hipdnnHeuristicPolicySetEngineIds(_desc, ids.data(), ids.size()),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    int32_t applied = -1;
    ASSERT_EQ(hipdnnHeuristicPolicyFinalize(_desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(applied, 1);

    // Caller buffer is smaller than result: should clip and report the clipped count.
    std::array<int64_t, 2> outBuf{0, 0};
    size_t count = outBuf.size();
    EXPECT_EQ(hipdnnHeuristicPolicyGetSortedEngineIds(_desc, outBuf.data(), &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(count, outBuf.size());
    EXPECT_EQ(outBuf[0], MIOPEN_ENGINE_ID);
    EXPECT_EQ(outBuf[1], CUSTOM_ENGINE_ID);
}
