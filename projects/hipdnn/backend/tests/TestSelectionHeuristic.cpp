// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestSelectionHeuristic.cpp
 * @brief Unit tests for SelectionHeuristic RAII wrapper (RFC 0007 Section 7)
 *
 * Tests the C++ facade that wraps hipdnnHeuristicPolicyDescriptor_t lifecycle
 * and provides clean API over the heuristic plugin C ABI.
 */

#include "heuristics/SelectionHeuristic.hpp"

#include "descriptors/mocks/MockHeuristicPlugin.hpp"
#include "plugin/HeuristicPlugin.hpp"

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>

using namespace hipdnn_backend::heuristics;
using namespace hipdnn_backend::plugin;

class TestSelectionHeuristic : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a mock plugin handle (just a non-null pointer for testing)
        _mockHandle = reinterpret_cast<hipdnnHeuristicHandle_t>(this);
    }

    hipdnnHeuristicHandle_t _mockHandle = nullptr;
};

// ========== Constructor Tests ==========

TEST_F(TestSelectionHeuristic, ConstructorWithValidInputs)
{
    auto mockPlugin = std::make_unique<MockHeuristicPlugin>();

    // Expect createPolicyDescriptor to be called
    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x1234);

    EXPECT_CALL(*mockPlugin, createPolicyDescriptor(_mockHandle))
        .WillOnce(::testing::Return(mockDescriptor));

    EXPECT_CALL(*mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    // Should not throw
    SelectionHeuristic heuristic(mockPlugin.get(), _mockHandle); // NOLINT(misc-const-correctness)
}

TEST_F(TestSelectionHeuristic, ConstructorThrowsOnNullPlugin)
{
    // Null plugin should throw
    EXPECT_THROW(
        { SelectionHeuristic heuristic(nullptr, _mockHandle); }, // NOLINT(misc-const-correctness)
        hipdnn_backend::HipdnnException);
}

TEST_F(TestSelectionHeuristic, ConstructorThrowsOnNullHandle)
{
    auto mockPlugin = std::make_unique<MockHeuristicPlugin>();

    // Null handle should throw
    EXPECT_THROW(
        {
            SelectionHeuristic heuristic(mockPlugin.get(), nullptr);
        }, // NOLINT(misc-const-correctness)
        hipdnn_backend::HipdnnException);
}

// ========== Move Semantics Tests ==========

TEST_F(TestSelectionHeuristic, MoveConstructor)
{
    auto mockPlugin = std::make_unique<MockHeuristicPlugin>();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x5678);

    EXPECT_CALL(*mockPlugin, createPolicyDescriptor(_mockHandle))
        .WillOnce(::testing::Return(mockDescriptor));

    // Destroy should be called exactly once (when moved-to object is destroyed)
    EXPECT_CALL(*mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    {
        SelectionHeuristic heuristic1(mockPlugin.get(), _mockHandle);

        // Move construct
        SelectionHeuristic heuristic2(std::move(heuristic1)); // NOLINT(misc-const-correctness)

        // heuristic2 should now own the descriptor
        // heuristic1 should be empty (moved-from state)
    } // Both destructors called, but only heuristic2 has valid descriptor
}

TEST_F(TestSelectionHeuristic, MoveAssignment)
{
    auto mockPlugin = std::make_unique<MockHeuristicPlugin>();

    auto mockDescriptor1 = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x1111);
    auto mockDescriptor2 = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x2222);

    EXPECT_CALL(*mockPlugin, createPolicyDescriptor(_mockHandle))
        .WillOnce(::testing::Return(mockDescriptor1))
        .WillOnce(::testing::Return(mockDescriptor2));

    // First descriptor destroyed during move assignment
    // Second descriptor destroyed when moved-to object is destroyed
    EXPECT_CALL(*mockPlugin, destroyPolicyDescriptor(mockDescriptor1)).Times(1);
    EXPECT_CALL(*mockPlugin, destroyPolicyDescriptor(mockDescriptor2)).Times(1);

    {
        SelectionHeuristic heuristic1(mockPlugin.get(), _mockHandle);
        SelectionHeuristic heuristic2(mockPlugin.get(), _mockHandle);

        // Move assign
        heuristic1 = std::move(heuristic2);

        // heuristic1 should now own mockDescriptor2
        // heuristic2 should be empty
        // mockDescriptor1 should have been destroyed
    }
}

TEST_F(TestSelectionHeuristic, MoveAssignmentSelfAssignment)
{
    auto mockPlugin = std::make_unique<MockHeuristicPlugin>();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x9999);

    EXPECT_CALL(*mockPlugin, createPolicyDescriptor(_mockHandle))
        .WillOnce(::testing::Return(mockDescriptor));

    EXPECT_CALL(*mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    {
        SelectionHeuristic heuristic(mockPlugin.get(), _mockHandle);

        // Self-assignment should be safe (use reference to avoid warning)
        SelectionHeuristic& heuristicRef = heuristic;
        heuristic = std::move(heuristicRef);

        // Descriptor should still be valid
    }
}

// ========== API Tests ==========

TEST_F(TestSelectionHeuristic, SetEngineIds)
{
    auto mockPlugin = std::make_unique<MockHeuristicPlugin>();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xAAAA);

    EXPECT_CALL(*mockPlugin, createPolicyDescriptor(_mockHandle))
        .WillOnce(::testing::Return(mockDescriptor));

    std::vector<int64_t> testEngineIds = {1, 2, 3, 4, 5};

    EXPECT_CALL(
        *mockPlugin,
        setEngineIds(mockDescriptor, ::testing::Pointee(testEngineIds[0]), testEngineIds.size()))
        .Times(1);

    EXPECT_CALL(*mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    SelectionHeuristic heuristic(mockPlugin.get(), _mockHandle);
    heuristic.setEngineIds(testEngineIds);
}

TEST_F(TestSelectionHeuristic, SetSerializedGraph)
{
    auto mockPlugin = std::make_unique<MockHeuristicPlugin>();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xBBBB);

    EXPECT_CALL(*mockPlugin, createPolicyDescriptor(_mockHandle))
        .WillOnce(::testing::Return(mockDescriptor));

    std::vector<uint8_t> graphData = {0x01, 0x02, 0x03};
    const hipdnnPluginConstData_t serializedGraph{graphData.data(), graphData.size()};

    EXPECT_CALL(*mockPlugin, setSerializedGraph(mockDescriptor, &serializedGraph)).Times(1);

    EXPECT_CALL(*mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    SelectionHeuristic heuristic(mockPlugin.get(), _mockHandle);
    heuristic.setSerializedGraph(&serializedGraph);
}

TEST_F(TestSelectionHeuristic, FinalizeReturnsTrue)
{
    auto mockPlugin = std::make_unique<MockHeuristicPlugin>();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xCCCC);

    EXPECT_CALL(*mockPlugin, createPolicyDescriptor(_mockHandle))
        .WillOnce(::testing::Return(mockDescriptor));

    EXPECT_CALL(*mockPlugin, finalize(mockDescriptor)).WillOnce(::testing::Return(true));

    EXPECT_CALL(*mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    SelectionHeuristic heuristic(mockPlugin.get(), _mockHandle);
    EXPECT_TRUE(heuristic.finalize());
}

TEST_F(TestSelectionHeuristic, FinalizeReturnsFalse)
{
    auto mockPlugin = std::make_unique<MockHeuristicPlugin>();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xDDDD);

    EXPECT_CALL(*mockPlugin, createPolicyDescriptor(_mockHandle))
        .WillOnce(::testing::Return(mockDescriptor));

    EXPECT_CALL(*mockPlugin, finalize(mockDescriptor)).WillOnce(::testing::Return(false));

    EXPECT_CALL(*mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    SelectionHeuristic heuristic(mockPlugin.get(), _mockHandle);
    EXPECT_FALSE(heuristic.finalize());
}

TEST_F(TestSelectionHeuristic, GetSortedEngineIds)
{
    auto mockPlugin = std::make_unique<MockHeuristicPlugin>();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xEEEE);

    EXPECT_CALL(*mockPlugin, createPolicyDescriptor(_mockHandle))
        .WillOnce(::testing::Return(mockDescriptor));

    const std::vector<int64_t> expectedIds = {5, 4, 3, 2, 1};

    EXPECT_CALL(*mockPlugin, getSortedEngineIds(mockDescriptor))
        .WillOnce(::testing::Return(expectedIds));

    EXPECT_CALL(*mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    SelectionHeuristic heuristic(mockPlugin.get(), _mockHandle);
    auto result = heuristic.getSortedEngineIds();

    EXPECT_EQ(result, expectedIds);
}

// ========== Exception Safety Tests ==========

TEST_F(TestSelectionHeuristic, DestructorHandlesExceptionInCleanup)
{
    auto mockPlugin = std::make_unique<MockHeuristicPlugin>();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xFFFF);

    EXPECT_CALL(*mockPlugin, createPolicyDescriptor(_mockHandle))
        .WillOnce(::testing::Return(mockDescriptor));

    // Destructor should catch and suppress exceptions
    EXPECT_CALL(*mockPlugin, destroyPolicyDescriptor(mockDescriptor))
        .WillOnce(::testing::Throw(
            hipdnn_backend::HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR, "Cleanup failed")));

    // Should not throw from destructor
    {
        SelectionHeuristic heuristic(mockPlugin.get(), _mockHandle);
    } // NOLINT(misc-const-correctness)
}

TEST_F(TestSelectionHeuristic, MoveAssignmentHandlesExceptionInCleanup)
{
    auto mockPlugin = std::make_unique<MockHeuristicPlugin>();

    auto mockDescriptor1 = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x1001);
    auto mockDescriptor2 = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x2002);

    EXPECT_CALL(*mockPlugin, createPolicyDescriptor(_mockHandle))
        .WillOnce(::testing::Return(mockDescriptor1))
        .WillOnce(::testing::Return(mockDescriptor2));

    // First descriptor cleanup throws during move assignment
    EXPECT_CALL(*mockPlugin, destroyPolicyDescriptor(mockDescriptor1))
        .WillOnce(::testing::Throw(std::runtime_error("Cleanup error")));

    // Second descriptor cleanup succeeds
    EXPECT_CALL(*mockPlugin, destroyPolicyDescriptor(mockDescriptor2)).Times(1);

    {
        SelectionHeuristic heuristic1(mockPlugin.get(), _mockHandle);
        SelectionHeuristic heuristic2(mockPlugin.get(), _mockHandle);

        // Move assignment should not throw even though cleanup of old descriptor throws
        heuristic1 = std::move(heuristic2);
    }
}
