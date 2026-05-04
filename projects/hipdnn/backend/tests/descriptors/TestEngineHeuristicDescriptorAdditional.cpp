// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestEngineHeuristicDescriptorAdditional.cpp
 * @brief Additional coverage tests for EngineHeuristicDescriptor (RFC 0007)
 *
 * These tests target uncovered code paths to improve coverage from 67.84% to 80%+
 */

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TestMacros.hpp"
#include "descriptors/EngineConfigDescriptor.hpp"
#include "descriptors/EngineDescriptor.hpp"
#include "descriptors/EngineHeuristicDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "heuristics/SelectionHeuristic.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockDescriptor.hpp"
#include "mocks/MockEnginePluginResourceManager.hpp"
#include "mocks/MockHandle.hpp"
#include "mocks/MockHeuristicPlugin.hpp"
#include "mocks/MockHeuristicPluginResourceManager.hpp"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>

#include <cstdlib>
#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace plugin;
using namespace hipdnn_backend::test_utilities;
using namespace ::testing;

class TestEngineHeuristicDescriptorAdditional : public ::testing::Test
{
public:
    std::shared_ptr<EngineHeuristicDescriptor> getEngineHeuristicDescriptor() const
    {
        return _engineHeuristicWrapper->asDescriptor<EngineHeuristicDescriptor>();
    }

    std::shared_ptr<MockGraphDescriptor> getMockGraph() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockGraphDescriptor>(
            _mockGraphWrapper.get());
    }

    void setGraph() const
    {
        EXPECT_CALL(*getMockGraph(), isFinalized()).WillRepeatedly(Return(true));
        EXPECT_CALL(*getMockGraph(), getHandle()).WillRepeatedly(Return(_mockHandle.get()));
        EXPECT_CALL(*_mockHandle, getPluginResourceManager())
            .WillRepeatedly(Return(_mockEnginePluginResourceManager));
        EXPECT_CALL(*_mockHandle, getHeuristicPluginResourceManager())
            .WillRepeatedly(Return(_mockHeuristicPluginResourceManager));

        setupMockHeuristicPlugin();

        ASSERT_NO_THROW(
            getEngineHeuristicDescriptor()->setAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                         1,
                                                         &_mockGraphWrapper));
    }

    void setHeuristicMode() const
    {
        hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;
        ASSERT_NO_THROW(getEngineHeuristicDescriptor()->setAttribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &mode));
    }

    void setupMockHeuristicPlugin() const
    {
        const int64_t configPolicyId
            = hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::Config");
        const int64_t staticOrderingPolicyId
            = hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering");

        auto mockHandle = reinterpret_cast<hipdnnHeuristicHandle_t>(0x1234);
        auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x5678);

        // For custom/unknown policy IDs, return nullptr (set up general expectations first)
        EXPECT_CALL(*_mockHeuristicPluginResourceManager, getPluginForPolicyId(_))
            .WillRepeatedly(Return(nullptr));
        EXPECT_CALL(*_mockHeuristicPluginResourceManager, getHeuristicHandleForPolicyId(_))
            .WillRepeatedly(Return(nullptr));

        // Override for default policies: Config returns nullptr (skips), StaticOrdering succeeds
        EXPECT_CALL(*_mockHeuristicPluginResourceManager, getPluginForPolicyId(configPolicyId))
            .WillRepeatedly(Return(nullptr));
        EXPECT_CALL(*_mockHeuristicPluginResourceManager,
                    getHeuristicHandleForPolicyId(configPolicyId))
            .WillRepeatedly(Return(nullptr));

        EXPECT_CALL(*_mockHeuristicPluginResourceManager,
                    getPluginForPolicyId(staticOrderingPolicyId))
            .WillRepeatedly(Return(_mockHeuristicPlugin.get()));
        EXPECT_CALL(*_mockHeuristicPluginResourceManager,
                    getHeuristicHandleForPolicyId(staticOrderingPolicyId))
            .WillRepeatedly(Return(mockHandle));

        EXPECT_CALL(*_mockHeuristicPlugin, setDeviceProperties(mockHandle, _))
            .WillRepeatedly(Return());
        EXPECT_CALL(*_mockHeuristicPlugin, createPolicyDescriptor(mockHandle))
            .WillRepeatedly(Return(mockDescriptor));
        EXPECT_CALL(*_mockHeuristicPlugin, destroyPolicyDescriptor(mockDescriptor))
            .WillRepeatedly(Return());

        EXPECT_CALL(*_mockHeuristicPlugin, setEngineIds(mockDescriptor, _, _))
            .WillRepeatedly([this](hipdnnHeuristicPolicyDescriptor_t,
                                   const int64_t* engineIds,
                                   size_t engineIdCount) {
                _mockStoredEngineIds.assign(engineIds, engineIds + engineIdCount);
            });

        EXPECT_CALL(*_mockHeuristicPlugin, setSerializedGraph(mockDescriptor, _))
            .WillRepeatedly(Return());
        EXPECT_CALL(*_mockHeuristicPlugin, finalize(mockDescriptor)).WillRepeatedly(Return(true));

        EXPECT_CALL(*_mockHeuristicPlugin, getSortedEngineIds(mockDescriptor))
            .WillRepeatedly([this]() { return _mockStoredEngineIds; });

        EXPECT_CALL(*getMockGraph(), getSerializedGraph()).WillRepeatedly([]() {
            static const std::vector<uint8_t> s_dummyData = {0x01, 0x02, 0x03};
            return hipdnnPluginConstData_t{s_dummyData.data(), s_dummyData.size()};
        });
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _engineHeuristicWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockGraphWrapper = nullptr;
    std::unique_ptr<NiceMock<MockHandle>> _mockHandle = nullptr;
    std::shared_ptr<NiceMock<MockEnginePluginResourceManager>> _mockEnginePluginResourceManager
        = nullptr;
    std::shared_ptr<NiceMock<MockHeuristicPluginResourceManager>>
        _mockHeuristicPluginResourceManager = nullptr;
    std::shared_ptr<NiceMock<MockHeuristicPlugin>> _mockHeuristicPlugin = nullptr;
    mutable std::vector<int64_t> _mockStoredEngineIds;

    void SetUp() override
    {
        _engineHeuristicWrapper = createDescriptor<EngineHeuristicDescriptor>();
        _mockGraphWrapper = createDescriptor<MockGraphDescriptor>();
        _mockHandle = std::make_unique<NiceMock<MockHandle>>();
        _mockEnginePluginResourceManager
            = std::make_shared<NiceMock<MockEnginePluginResourceManager>>();
        _mockHeuristicPluginResourceManager
            = std::make_shared<NiceMock<MockHeuristicPluginResourceManager>>();
        _mockHeuristicPlugin = std::make_shared<NiceMock<MockHeuristicPlugin>>();
    }

    void TearDown() override
    {
        _engineHeuristicWrapper.reset();
        _mockGraphWrapper.reset();
    }
};

// ========== Policy Order API Tests ==========

TEST_F(TestEngineHeuristicDescriptorAdditional, SetPolicyOrderValid)
{
    auto heur = getEngineHeuristicDescriptor();

    const std::vector<int64_t> policyIds = {
        hipdnn_data_sdk::utilities::policyNameToId("Policy1"),
        hipdnn_data_sdk::utilities::policyNameToId("Policy2"),
        hipdnn_data_sdk::utilities::policyNameToId("Policy3"),
    };

    ASSERT_NO_THROW(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                       HIPDNN_TYPE_INT64,
                                       static_cast<int64_t>(policyIds.size()),
                                       policyIds.data()));
}

TEST_F(TestEngineHeuristicDescriptorAdditional, SetPolicyOrderInvalidType)
{
    auto heur = getEngineHeuristicDescriptor();
    const char dummy = '\0';

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT, HIPDNN_TYPE_CHAR, 1, &dummy),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineHeuristicDescriptorAdditional, SetPolicyOrderNullPointer)
{
    auto heur = getEngineHeuristicDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT, HIPDNN_TYPE_INT64, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestEngineHeuristicDescriptorAdditional, GetPolicyOrderWhenNotSet)
{
    auto heur = getEngineHeuristicDescriptor();
    setGraph();
    setHeuristicMode();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{1}));
    ASSERT_NO_THROW(heur->finalize());

    int64_t count = 999;
    ASSERT_NO_THROW(heur->getAttribute(
        HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT, HIPDNN_TYPE_INT64, 0, &count, nullptr));
    ASSERT_EQ(count, 0);
}

TEST_F(TestEngineHeuristicDescriptorAdditional, GetPolicyOrderCountOnly)
{
    auto heur = getEngineHeuristicDescriptor();

    const std::vector<int64_t> policyIds = {
        hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering"),
    };

    ASSERT_NO_THROW(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                       HIPDNN_TYPE_INT64,
                                       static_cast<int64_t>(policyIds.size()),
                                       policyIds.data()));

    setGraph();
    setHeuristicMode();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{1}));
    ASSERT_NO_THROW(heur->finalize());

    int64_t count = 0;
    ASSERT_NO_THROW(heur->getAttribute(
        HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT, HIPDNN_TYPE_INT64, 0, &count, nullptr));
    ASSERT_EQ(count, static_cast<int64_t>(policyIds.size()));
}

TEST_F(TestEngineHeuristicDescriptorAdditional, GetPolicyOrderInvalidType)
{
    auto heur = getEngineHeuristicDescriptor();
    setGraph();
    setHeuristicMode();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{1}));
    ASSERT_NO_THROW(heur->finalize());

    std::vector<char> buffer(256);
    int64_t count = 0;
    ASSERT_THROW_HIPDNN_STATUS(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                                  HIPDNN_TYPE_CHAR,
                                                  256,
                                                  &count,
                                                  buffer.data()),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineHeuristicDescriptorAdditional, GetPolicyOrderNullPointer)
{
    auto heur = getEngineHeuristicDescriptor();

    const std::vector<int64_t> policyIds = {
        hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering"),
    };

    ASSERT_NO_THROW(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                       HIPDNN_TYPE_INT64,
                                       static_cast<int64_t>(policyIds.size()),
                                       policyIds.data()));

    setGraph();
    setHeuristicMode();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{1}));
    ASSERT_NO_THROW(heur->finalize());

    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(
            HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT, HIPDNN_TYPE_INT64, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestEngineHeuristicDescriptorAdditional, GetPolicyOrderBufferTooSmall)
{
    auto heur = getEngineHeuristicDescriptor();

    const std::vector<int64_t> policyIds = {
        hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering"),
        hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::Config"),
    };

    ASSERT_NO_THROW(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                       HIPDNN_TYPE_INT64,
                                       static_cast<int64_t>(policyIds.size()),
                                       policyIds.data()));

    setGraph();
    setHeuristicMode();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{1}));
    ASSERT_NO_THROW(heur->finalize());

    // Request fewer elements than the descriptor holds; should truncate.
    std::vector<int64_t> buffer(1);
    int64_t count = 0;
    ASSERT_NO_THROW(heur->getAttribute(
        HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT, HIPDNN_TYPE_INT64, 1, &count, buffer.data()));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(buffer[0], policyIds[0]);
}

TEST_F(TestEngineHeuristicDescriptorAdditional, GetPolicyOrderRoundTrip)
{
    auto heur = getEngineHeuristicDescriptor();

    const std::vector<int64_t> policyIds = {
        hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering"),
        hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::Config"),
        hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering"),
    };

    ASSERT_NO_THROW(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                       HIPDNN_TYPE_INT64,
                                       static_cast<int64_t>(policyIds.size()),
                                       policyIds.data()));

    setGraph();
    setHeuristicMode();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{1}));
    ASSERT_NO_THROW(heur->finalize());

    std::vector<int64_t> getBuffer(policyIds.size());
    int64_t count = 0;
    ASSERT_NO_THROW(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                       HIPDNN_TYPE_INT64,
                                       static_cast<int64_t>(getBuffer.size()),
                                       &count,
                                       getBuffer.data()));

    ASSERT_EQ(count, static_cast<int64_t>(policyIds.size()));
    for(size_t i = 0; i < policyIds.size(); ++i)
    {
        ASSERT_EQ(getBuffer[i], policyIds[i]);
    }
}

// ========== Exception Handling Tests ==========

TEST_F(TestEngineHeuristicDescriptorAdditional, FinalizeWithAllPoliciesFailing)
{
    auto heur = getEngineHeuristicDescriptor();
    setGraph();
    setHeuristicMode();

    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{1, 2}));

    // Make both policies fail
    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x5678);

    EXPECT_CALL(*_mockHeuristicPlugin, finalize(mockDescriptor)).WillRepeatedly(Return(false));

    // finalize() should throw when all policies fail
    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestEngineHeuristicDescriptorAdditional, FinalizeWithPolicyThrowingException)
{
    auto heur = getEngineHeuristicDescriptor();
    setGraph();
    setHeuristicMode();

    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{1}));

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x5678);

    // First call to setEngineIds throws
    EXPECT_CALL(*_mockHeuristicPlugin, setEngineIds(mockDescriptor, _, _))
        .WillOnce(
            Throw(HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR, "Mock setEngineIds failure")));

    // finalize() should throw when all policies fail (including exception paths)
    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_INTERNAL_ERROR);
}

// ========== toString Tests ==========

TEST_F(TestEngineHeuristicDescriptorAdditional, ToStringBeforeFinalize)
{
    auto heur = getEngineHeuristicDescriptor();
    const std::string str = heur->toString();
    ASSERT_NE(str.find("EngineHeuristicDescriptor"), std::string::npos);
    ASSERT_NE(str.find("unset"), std::string::npos);
}

TEST_F(TestEngineHeuristicDescriptorAdditional, ToStringAfterSetHeurMode)
{
    auto heur = getEngineHeuristicDescriptor();
    setHeuristicMode();
    const std::string str = heur->toString();
    ASSERT_NE(str.find("EngineHeuristicDescriptor"), std::string::npos);
    ASSERT_NE(str.find("heuristicMode"), std::string::npos);
}

TEST_F(TestEngineHeuristicDescriptorAdditional, ToStringAfterSetGraph)
{
    auto heur = getEngineHeuristicDescriptor();
    setGraph();
    const std::string str = heur->toString();
    ASSERT_NE(str.find("graph="), std::string::npos);
    ASSERT_EQ(str.find("graph=null"), std::string::npos); // Should not be null
}

TEST_F(TestEngineHeuristicDescriptorAdditional, ToStringWithPolicyOrder)
{
    auto heur = getEngineHeuristicDescriptor();

    const std::vector<int64_t> policyIds = {
        hipdnn_data_sdk::utilities::policyNameToId("Policy1"),
        hipdnn_data_sdk::utilities::policyNameToId("Policy2"),
    };

    ASSERT_NO_THROW(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                       HIPDNN_TYPE_INT64,
                                       static_cast<int64_t>(policyIds.size()),
                                       policyIds.data()));

    const std::string str = heur->toString();
    ASSERT_NE(str.find("policyOrder"), std::string::npos);
    ASSERT_NE(str.find(hipdnn_data_sdk::utilities::formatEngineIdHex(policyIds[0])),
              std::string::npos);
    ASSERT_NE(str.find(hipdnn_data_sdk::utilities::formatEngineIdHex(policyIds[1])),
              std::string::npos);
}

// ========== Edge Case Tests ==========

TEST_F(TestEngineHeuristicDescriptorAdditional, SetEmptyPolicyOrder)
{
    auto heur = getEngineHeuristicDescriptor();

    // Setting an empty policy order is allowed at the attribute level; finalize()
    // would later fail because no policy can be selected, but that is exercised by
    // FinalizeWithAllPoliciesFailing. Here we only verify the attribute path.
    ASSERT_NO_THROW(heur->setAttribute(
        HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT, HIPDNN_TYPE_INT64, 0, nullptr));
}

TEST_F(TestEngineHeuristicDescriptorAdditional, GetPolicyOrderNullElementCount)
{
    auto heur = getEngineHeuristicDescriptor();

    const std::vector<int64_t> policyIds = {
        hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering"),
    };

    ASSERT_NO_THROW(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                       HIPDNN_TYPE_INT64,
                                       static_cast<int64_t>(policyIds.size()),
                                       policyIds.data()));

    setGraph();
    setHeuristicMode();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{1}));
    ASSERT_NO_THROW(heur->finalize());

    std::vector<int64_t> getBuffer(16);
    ASSERT_THROW_HIPDNN_STATUS(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                                  HIPDNN_TYPE_INT64,
                                                  static_cast<int64_t>(getBuffer.size()),
                                                  nullptr,
                                                  getBuffer.data()),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestEngineHeuristicDescriptorAdditional, MultipleSetPolicyOrderCalls)
{
    auto heur = getEngineHeuristicDescriptor();

    // First set
    {
        const std::vector<int64_t> policyIds = {
            hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering"),
        };
        ASSERT_NO_THROW(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                           HIPDNN_TYPE_INT64,
                                           static_cast<int64_t>(policyIds.size()),
                                           policyIds.data()));
    }

    // Second set should override
    const std::vector<int64_t> secondPolicyIds = {
        hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::Config"),
        hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering"),
    };
    ASSERT_NO_THROW(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                       HIPDNN_TYPE_INT64,
                                       static_cast<int64_t>(secondPolicyIds.size()),
                                       secondPolicyIds.data()));

    setGraph();
    setHeuristicMode();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{1}));
    ASSERT_NO_THROW(heur->finalize());

    std::vector<int64_t> getBuffer(secondPolicyIds.size());
    int64_t count = 0;
    ASSERT_NO_THROW(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                       HIPDNN_TYPE_INT64,
                                       static_cast<int64_t>(getBuffer.size()),
                                       &count,
                                       getBuffer.data()));

    ASSERT_EQ(count, static_cast<int64_t>(secondPolicyIds.size()));
    for(size_t i = 0; i < secondPolicyIds.size(); ++i)
    {
        ASSERT_EQ(getBuffer[i], secondPolicyIds[i]);
    }
}
