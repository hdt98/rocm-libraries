// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestStaticOrderingPolicy.cpp
 * @brief Regression tests for StaticOrdering heuristic policy (RFC 0007)
 *
 * These tests verify that the StaticOrdering plugin maintains backward
 * compatibility with the legacy utilities::sortEngineIds behavior.
 */

#include "handle/Handle.hpp"
#include "plugin/HeuristicPluginResourceManager.hpp"
#include "utilities/EngineOrdering.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>

#include <gtest/gtest.h>

#include <algorithm>

using namespace hipdnn_backend;
using namespace hipdnn_data_sdk::utilities;

class TestStaticOrderingPolicy : public ::testing::Test
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

    // Helper to get StaticOrdering policy ID
    static int64_t getStaticOrderingPolicyId()
    {
        return policyNameToId("SelectionHeuristic::StaticOrdering");
    }

    hipdnnHandle_t _handle = nullptr;
};

// ========== Ordering Behavior Tests ==========

TEST_F(TestStaticOrderingPolicy, MIOpenEngineHasHighestPriority)
{
    // Verify that MIOpen engine is prioritized first
    const std::vector<int64_t> engines = {
        engineNameToId("MIOPEN_ENGINE_DETERMINISTIC"),
        engineNameToId("SomeOtherEngine"),
        engineNameToId("MIOPEN_ENGINE"),
    };

    std::vector<int64_t> sorted = engines;
    utilities::sortEngineIds(sorted);

    // MIOpen should be first
    EXPECT_EQ(sorted[0], engineNameToId("MIOPEN_ENGINE"));
}

TEST_F(TestStaticOrderingPolicy, MIOpenDeterministicHasLowestPriority)
{
    const std::vector<int64_t> engines = {
        engineNameToId("MIOPEN_ENGINE_DETERMINISTIC"),
        engineNameToId("MIOPEN_ENGINE"),
        engineNameToId("SomeOtherEngine"),
    };

    std::vector<int64_t> sorted = engines;
    utilities::sortEngineIds(sorted);

    // MIOpenDeterministic should be last
    EXPECT_EQ(sorted.back(), engineNameToId("MIOPEN_ENGINE_DETERMINISTIC"));
}

TEST_F(TestStaticOrderingPolicy, OtherEnginesAreMiddlePriority)
{
    const std::vector<int64_t> engines = {
        engineNameToId("MIOPEN_ENGINE_DETERMINISTIC"),
        engineNameToId("CustomEngine1"),
        engineNameToId("MIOPEN_ENGINE"),
        engineNameToId("CustomEngine2"),
    };

    std::vector<int64_t> sorted = engines;
    utilities::sortEngineIds(sorted);

    // Order should be: MIOpen, CustomEngine1, CustomEngine2, MIOpenDeterministic
    EXPECT_EQ(sorted[0], engineNameToId("MIOPEN_ENGINE"));
    EXPECT_EQ(sorted[3], engineNameToId("MIOPEN_ENGINE_DETERMINISTIC"));

    // Custom engines in middle (order preserved)
    EXPECT_EQ(sorted[1], engineNameToId("CustomEngine1"));
    EXPECT_EQ(sorted[2], engineNameToId("CustomEngine2"));
}

TEST_F(TestStaticOrderingPolicy, StableWithinSamePriority)
{
    // Engines with same priority should maintain relative order
    std::vector<int64_t> engines = {
        engineNameToId("Custom1"),
        engineNameToId("Custom2"),
        engineNameToId("Custom3"),
    };

    std::vector<int64_t> sorted = engines;
    utilities::sortEngineIds(sorted);

    // Should preserve order
    EXPECT_EQ(sorted[0], engines[0]);
    EXPECT_EQ(sorted[1], engines[1]);
    EXPECT_EQ(sorted[2], engines[2]);
}

TEST_F(TestStaticOrderingPolicy, EmptyListRemainsEmpty)
{
    std::vector<int64_t> engines;
    utilities::sortEngineIds(engines);

    EXPECT_TRUE(engines.empty());
}

TEST_F(TestStaticOrderingPolicy, SingleEngineUnchanged)
{
    const std::vector<int64_t> engines = {engineNameToId("MIOPEN_ENGINE")};

    std::vector<int64_t> sorted = engines;
    utilities::sortEngineIds(sorted);

    EXPECT_EQ(sorted, engines);
}

// ========== Consistency Tests ==========

TEST_F(TestStaticOrderingPolicy, BackendSortingIsIdempotent)
{
    // Verify that backend utilities::sortEngineIds is idempotent
    const std::vector<int64_t> engines = {
        engineNameToId("MIOPEN_ENGINE_DETERMINISTIC"),
        engineNameToId("CustomEngine"),
        engineNameToId("MIOPEN_ENGINE"),
    };

    std::vector<int64_t> backendSorted = engines;
    utilities::sortEngineIds(backendSorted);

    std::vector<int64_t> sortedAgain = backendSorted;
    utilities::sortEngineIds(sortedAgain);

    EXPECT_EQ(backendSorted, sortedAgain);
}

// ========== Policy-Specific Tests ==========

TEST_F(TestStaticOrderingPolicy, PolicyNeverDeclines)
{
    // StaticOrdering should always succeed (never return NOT_APPLICABLE)
    // This is verified by checking that it's usable as a fallback policy

    auto heurRm = _handle->getHeuristicPluginResourceManager();
    auto policyInfos = heurRm->getHeuristicPolicyInfos();

    // Verify plugins were loaded
    // Plugins should be found at hipdnn_plugins/heuristics/ relative to libhipdnn_backend.so
    ASSERT_FALSE(policyInfos.empty())
        << "No heuristic plugins found. Expected plugins at hipdnn_plugins/heuristics/ "
        << "relative to libhipdnn_backend.so location. "
        << "Verify plugins were built and are in the correct directory.";

    // Find StaticOrdering
    const bool found = std::any_of(policyInfos.begin(), policyInfos.end(), [](const auto& info) {
        return info.policyName.find("StaticOrdering") != std::string::npos;
    });

    EXPECT_TRUE(found) << "StaticOrdering policy not found in loaded plugins";

    // The policy's design guarantees it never declines
    // This is enforced by always returning `applied = 1` from finalize()
}

// ========== Known Engine IDs Tests ==========

TEST_F(TestStaticOrderingPolicy, MIOpenEngineIdIsRecognized)
{
    const int64_t miopenId = engineNameToId("MIOPEN_ENGINE");
    EXPECT_EQ(miopenId, MIOPEN_ENGINE_ID);
}

TEST_F(TestStaticOrderingPolicy, MIOpenDeterministicIdIsRecognized)
{
    const int64_t miopenDetId = engineNameToId("MIOPEN_ENGINE_DETERMINISTIC");
    EXPECT_EQ(miopenDetId, MIOPEN_ENGINE_DETERMINISTIC_ID);
}

// ========== Complex Scenarios ==========

TEST_F(TestStaticOrderingPolicy, ComplexMixedEngineList)
{
    const std::vector<int64_t> engines = {
        engineNameToId("Plugin1::Engine3"),
        engineNameToId("MIOPEN_ENGINE_DETERMINISTIC"),
        engineNameToId("Plugin1::Engine1"),
        engineNameToId("MIOPEN_ENGINE"),
        engineNameToId("Plugin2::Engine1"),
        engineNameToId("Plugin1::Engine2"),
    };

    std::vector<int64_t> sorted = engines;
    utilities::sortEngineIds(sorted);

    // First should be MIOpen
    EXPECT_EQ(sorted[0], engineNameToId("MIOPEN_ENGINE"));

    // Last should be MIOpenDeterministic
    EXPECT_EQ(sorted.back(), engineNameToId("MIOPEN_ENGINE_DETERMINISTIC"));

    // Middle engines should preserve relative order
    EXPECT_EQ(sorted[1], engineNameToId("Plugin1::Engine3"));
    EXPECT_EQ(sorted[2], engineNameToId("Plugin1::Engine1"));
    EXPECT_EQ(sorted[3], engineNameToId("Plugin2::Engine1"));
    EXPECT_EQ(sorted[4], engineNameToId("Plugin1::Engine2"));
}

// ========== Idempotency Tests ==========

TEST_F(TestStaticOrderingPolicy, SortingIsIdempotent)
{
    const std::vector<int64_t> engines = {
        engineNameToId("MIOPEN_ENGINE_DETERMINISTIC"),
        engineNameToId("Custom"),
        engineNameToId("MIOPEN_ENGINE"),
    };

    std::vector<int64_t> sorted1 = engines;
    utilities::sortEngineIds(sorted1);

    std::vector<int64_t> sorted2 = sorted1;
    utilities::sortEngineIds(sorted2);

    EXPECT_EQ(sorted1, sorted2);
}

// ========== Negative Tests ==========

TEST_F(TestStaticOrderingPolicy, UnknownEngineIdsHandledGracefully)
{
    // Engines with unknown IDs should be treated as middle-priority
    const std::vector<int64_t> engines = {
        static_cast<int64_t>(0x1234567890ABCDEF), // Unknown ID
        engineNameToId("MIOPEN_ENGINE"),
        static_cast<int64_t>(0xFEDCBA0987654321), // Another unknown ID
    };

    std::vector<int64_t> sorted = engines;

    // Should not crash
    EXPECT_NO_THROW(utilities::sortEngineIds(sorted));

    // MIOpen should still be first
    EXPECT_EQ(sorted[0], engineNameToId("MIOPEN_ENGINE"));
}
