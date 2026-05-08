// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace plugin;
using namespace hipdnn_backend::test_utilities;
using namespace ::testing;

using ::testing::Return;

class TestEngineHeuristicDescriptor : public ::testing::Test
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

    std::shared_ptr<MockGraphDescriptor> getMockGraphBadType() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockGraphDescriptor>(
            _mockGraphBadTypeWrapper.get());
    }

    std::shared_ptr<MockDescriptor<EngineHeuristicDescriptor>> getMockWrongType() const
    {
        return _mockWrongTypeWrapper->asDescriptor<MockDescriptor<EngineHeuristicDescriptor>>();
    }

    void setGraph() const
    {
        EXPECT_CALL(*getMockGraph(), isFinalized()).WillRepeatedly(Return(true));
        EXPECT_CALL(*getMockGraph(), getHandle()).WillRepeatedly(Return(_mockHandle.get()));
        EXPECT_CALL(*_mockHandle, getPluginResourceManager())
            .WillRepeatedly(Return(_mockEnginePluginResourceManager));
        EXPECT_CALL(*_mockHandle, getHeuristicPluginResourceManager())
            .WillRepeatedly(Return(_mockHeuristicPluginResourceManager));

        // Set up mock heuristic plugin automatically when graph is set
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

    void makeEngineHeuristicFinalized() const
    {
        setGraph(); // This now automatically sets up the heuristic mock
        setHeuristicMode();
        EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
            .WillRepeatedly(Return(std::vector<int64_t>{0, 1, 2}));
        ASSERT_NO_THROW(getEngineHeuristicDescriptor()->finalize());
    }

    void setupMockHeuristicPlugin() const
    {
        const int64_t staticOrderingPolicyId
            = hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering");

        // Mock handles and descriptors
        auto mockHandle = reinterpret_cast<hipdnnHeuristicHandle_t>(0x1234);
        auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x5678);

        // Catch-all for unknown policy IDs (must be set first; gmock matches LIFO).
        EXPECT_CALL(*_mockHeuristicPluginResourceManager, getPluginForPolicyId(_))
            .WillRepeatedly(Return(nullptr));
        EXPECT_CALL(*_mockHeuristicPluginResourceManager, getHeuristicHandleForPolicyId(_))
            .WillRepeatedly(Return(nullptr));

        // StaticOrdering policy succeeds
        EXPECT_CALL(*_mockHeuristicPluginResourceManager,
                    getPluginForPolicyId(staticOrderingPolicyId))
            .WillRepeatedly(Return(_mockHeuristicPlugin.get()));
        EXPECT_CALL(*_mockHeuristicPluginResourceManager,
                    getHeuristicHandleForPolicyId(staticOrderingPolicyId))
            .WillRepeatedly(Return(mockHandle));
        EXPECT_CALL(*_mockHeuristicPluginResourceManager, setDevicePropertiesOnAllHandles(_))
            .WillRepeatedly(Return());

        // Set up expectations for the mock plugin
        EXPECT_CALL(*_mockHeuristicPlugin, setDeviceProperties(mockHandle, _))
            .WillRepeatedly(Return());
        EXPECT_CALL(*_mockHeuristicPlugin,
                    createPolicyDescriptor(mockHandle, staticOrderingPolicyId))
            .WillRepeatedly(Return(mockDescriptor));
        EXPECT_CALL(*_mockHeuristicPlugin, destroyPolicyDescriptor(mockDescriptor))
            .WillRepeatedly(Return());

        // Store engine IDs when setEngineIds is called
        EXPECT_CALL(*_mockHeuristicPlugin, setEngineIds(mockDescriptor, _, _))
            .WillRepeatedly([this](hipdnnHeuristicPolicyDescriptor_t,
                                   const int64_t* engineIds,
                                   size_t engineIdCount) {
                _mockStoredEngineIds.assign(engineIds, engineIds + engineIdCount);
            });

        EXPECT_CALL(*_mockHeuristicPlugin, setSerializedGraph(mockDescriptor, _))
            .WillRepeatedly(Return());
        EXPECT_CALL(*_mockHeuristicPlugin, finalize(mockDescriptor))
            .WillRepeatedly(Return(true)); // Always succeed

        // Return the same engine IDs that were set
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
    std::unique_ptr<HipdnnBackendDescriptor> _mockGraphBadTypeWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockWrongTypeWrapper = nullptr;
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
        _mockGraphBadTypeWrapper = createDescriptor<MockGraphDescriptor>();
        _mockWrongTypeWrapper = createDescriptor<MockDescriptor<EngineHeuristicDescriptor>>();
        _mockHandle = std::make_unique<NiceMock<MockHandle>>();
        _mockEnginePluginResourceManager
            = std::make_shared<NiceMock<MockEnginePluginResourceManager>>();
        _mockHeuristicPluginResourceManager
            = std::make_shared<NiceMock<MockHeuristicPluginResourceManager>>();
        _mockHeuristicPlugin = std::make_shared<NiceMock<MockHeuristicPlugin>>();
    }

    void TearDown() override
    {
        // Destroy descriptor before mocks to ensure proper cleanup order
        _engineHeuristicWrapper.reset();
        _mockGraphWrapper.reset();
        _mockGraphBadTypeWrapper.reset();
        _mockWrongTypeWrapper.reset();
        _engineDetailBuffers.clear();
    }

    hipdnnPluginConstData_t serializeEngineDetails(int64_t gidx)
    {
        flatbuffers::FlatBufferBuilder builder;
        hipdnn_flatbuffers_sdk::data_objects::EngineDetailsBuilder engineDetailsBuilder(builder);
        engineDetailsBuilder.add_engine_id(gidx);
        builder.Finish(engineDetailsBuilder.Finish());
        auto engineDetailsBuffer = builder.Release();
        hipdnnPluginConstData_t serializedEngineDetails
            = {engineDetailsBuffer.data(), engineDetailsBuffer.size()};
        _engineDetailBuffers.push_back(std::move(engineDetailsBuffer));
        return serializedEngineDetails;
    }

    std::vector<flatbuffers::DetachedBuffer> _engineDetailBuffers;
};

TEST_F(TestEngineHeuristicDescriptor, CreateEngineHeuristicDescriptor)
{
    auto heur = getEngineHeuristicDescriptor();
    ASSERT_NE(heur, nullptr);
    ASSERT_FALSE(heur->isFinalized());
    ASSERT_EQ(heur->getType(), HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
}

TEST_F(TestEngineHeuristicDescriptor, SetEngineHeuristicDescriptorGraph)
{
    auto heur = getEngineHeuristicDescriptor();

    EXPECT_CALL(*getMockGraphBadType(), isFinalized()).Times(1);
    EXPECT_CALL(*getMockGraph(), isFinalized()).WillOnce(Return(false)).WillOnce(Return(true));

    // isFinalized->false
    ASSERT_THROW_HIPDNN_STATUS(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_mockGraphWrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    // isFinalized()->true
    ASSERT_NO_THROW(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_mockGraphWrapper));

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(
            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_INT64, 1, &_mockGraphWrapper),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  2,
                                                  &_mockGraphWrapper),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(
            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t graph = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(
            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_mockGraphBadTypeWrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_THROW_HIPDNN_STATUS(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_mockWrongTypeWrapper),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineHeuristicDescriptor, SetEngineHeuristicDescriptorHeurMode)
{
    auto heur = getEngineHeuristicDescriptor();
    hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;
    auto unsupportedMode = static_cast<hipdnnBackendHeurMode_t>(999);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &mode),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 2, &mode),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &unsupportedMode),
        HIPDNN_STATUS_NOT_SUPPORTED);

    ASSERT_NO_THROW(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &mode));
}

TEST_F(TestEngineHeuristicDescriptor, SetEngineHeuristicDescriptorUnsupportedAttr)
{
    auto heur = getEngineHeuristicDescriptor();
    int32_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_INT32, 1, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestEngineHeuristicDescriptor, SetAttrOnFinalizedEngineHeuristicDescriptor)
{
    auto heur = getEngineHeuristicDescriptor();
    makeEngineHeuristicFinalized();

    ASSERT_THROW_HIPDNN_STATUS(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_mockGraphWrapper),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestEngineHeuristicDescriptor, FinalizeEngineHeuristicDescriptor)
{
    auto heur = getEngineHeuristicDescriptor();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds)
        .Times(AnyNumber()); //Uninteresting call

    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);

    setGraph();
    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);

    setHeuristicMode();
    ASSERT_NO_THROW(heur->finalize());

    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineHeuristicDescriptor, FinalizeEngineHeuristicDescriptorReverseOrder)
{
    auto heur = getEngineHeuristicDescriptor();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds)
        .Times(AnyNumber()); //Uninteresting call

    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);

    setHeuristicMode();
    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);

    setGraph();
    ASSERT_NO_THROW(heur->finalize());

    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineHeuristicDescriptor, GetAttrOnUnfinalizedEngineHeuristicDescriptor)
{
    auto heur = getEngineHeuristicDescriptor();
    hipdnnBackendDescriptor_t dummyGraph = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummyGraph),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestEngineHeuristicDescriptor, GetEngineHeuristicDescriptorUnsupportedAttr)
{
    auto heur = getEngineHeuristicDescriptor();
    hipdnnBackendHeurMode_t dummy;

    makeEngineHeuristicFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestEngineHeuristicDescriptor, GetEngineHeuristicDescriptorGraph)
{
    auto heur = getEngineHeuristicDescriptor();
    ScopedDescriptor graph;
    ScopedDescriptor graph2;

    makeEngineHeuristicFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(
            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_INT64, 1, nullptr, graph.getPtr()),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  2,
                                                  nullptr,
                                                  graph.getPtr()),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       nullptr,
                                       static_cast<void*>(graph.getPtr())));
    ASSERT_EQ(*graph.get(), *(_mockGraphWrapper.get()));

    int64_t count;
    ASSERT_NO_THROW(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(graph2.getPtr())));
    ASSERT_EQ(count, 1);
}

TEST_F(TestEngineHeuristicDescriptor, GetEngineHeuristicDescriptorEngineConfigs)
{
    auto heur = getEngineHeuristicDescriptor();
    makeEngineHeuristicFinalized();

    EXPECT_CALL(*getMockGraph(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*_mockEnginePluginResourceManager, getEngineDetails(_, _, _))
        .WillRepeatedly(
            Invoke([this](int64_t engineId, const GraphDescriptor*, hipdnnPluginConstData_t* d) {
                *d = this->serializeEngineDetails(engineId);
            }));
    EXPECT_CALL(*_mockEnginePluginResourceManager, destroyEngineDetails(_, _))
        .WillRepeatedly(Return());

    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_INT64, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM);

    int64_t count = 0;
    ASSERT_NO_THROW(heur->getAttribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr));
    ASSERT_EQ(count, 3);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(
            HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    std::vector<ScopedDescriptor> ownedConfigs(3);
    for(auto& owned : ownedConfigs)
    {
        owned = ScopedDescriptor(createDescriptorPtr<EngineConfigDescriptor>());
    }
    std::vector<hipdnnBackendDescriptor_t> configs;
    configs.reserve(ownedConfigs.size());
    for(auto& owned : ownedConfigs)
    {
        configs.push_back(owned.get());
    }

    ASSERT_THROW_HIPDNN_STATUS(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  3,
                                                  nullptr,
                                                  static_cast<void*>(configs.data())),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    count = 0;
    ASSERT_NO_THROW(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       3,
                                       &count,
                                       static_cast<void*>(configs.data())));
    ASSERT_EQ(count, 3);

    ScopedDescriptor singleConfig(createDescriptorPtr<EngineConfigDescriptor>());

    count = 0;
    ASSERT_NO_THROW(heur->getAttribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, &singleConfig));
    ASSERT_EQ(count, 1);
}

TEST_F(TestEngineHeuristicDescriptor, GetEngineConfigsWithNullConfig)
{
    auto heur = getEngineHeuristicDescriptor();
    makeEngineHeuristicFinalized();

    EXPECT_CALL(*getMockGraph(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*_mockEnginePluginResourceManager, getEngineDetails(_, _, _))
        .WillRepeatedly(
            Invoke([this](int64_t engineId, const GraphDescriptor*, hipdnnPluginConstData_t* d) {
                *d = this->serializeEngineDetails(engineId);
            }));
    EXPECT_CALL(*_mockEnginePluginResourceManager, destroyEngineDetails(_, _))
        .WillRepeatedly(Return());

    std::vector<ScopedDescriptor> ownedConfigs(3);
    ownedConfigs[0] = ScopedDescriptor(createDescriptorPtr<EngineConfigDescriptor>());
    // ownedConfigs[1] left as default (nullptr) to trigger the null-element path
    ownedConfigs[2] = ScopedDescriptor(createDescriptorPtr<EngineConfigDescriptor>());

    std::vector<hipdnnBackendDescriptor_t> configs;
    configs.reserve(ownedConfigs.size());
    for(auto& owned : ownedConfigs)
    {
        configs.push_back(owned.get());
    }

    EXPECT_CALL(*getMockGraph(), isFinalized()).WillRepeatedly(Return(true));
    int64_t count = 0;
    ASSERT_THROW_HIPDNN_STATUS(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  3,
                                                  &count,
                                                  configs.data()),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestEngineHeuristicDescriptor, GetEngineConfigsWithNoEngineIds)
{
    auto heur = getEngineHeuristicDescriptor();
    setGraph();
    setHeuristicMode();

    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{}));

    ASSERT_NO_THROW(heur->finalize());

    std::vector<ScopedDescriptor> ownedConfigs(3);
    for(auto& owned : ownedConfigs)
    {
        owned = ScopedDescriptor(createDescriptorPtr<EngineConfigDescriptor>());
    }
    std::vector<hipdnnBackendDescriptor_t> configs;
    configs.reserve(ownedConfigs.size());
    for(auto& owned : ownedConfigs)
    {
        configs.push_back(owned.get());
    }

    int64_t count = 0;
    ASSERT_NO_THROW(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       3,
                                       &count,
                                       static_cast<void*>(configs.data())));
    ASSERT_EQ(count, 0);
}

TEST_F(TestEngineHeuristicDescriptor, GetEngineConfigsRequestMoreThanAvailable)
{
    auto heur = getEngineHeuristicDescriptor();
    makeEngineHeuristicFinalized();

    EXPECT_CALL(*getMockGraph(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*_mockEnginePluginResourceManager, getEngineDetails(_, _, _))
        .WillRepeatedly(
            Invoke([this](int64_t engineId, const GraphDescriptor*, hipdnnPluginConstData_t* d) {
                *d = this->serializeEngineDetails(engineId);
            }));
    EXPECT_CALL(*_mockEnginePluginResourceManager, destroyEngineDetails(_, _))
        .WillRepeatedly(Return());

    std::vector<ScopedDescriptor> ownedConfigs(5);
    for(size_t i = 0; i < 3; ++i)
    {
        ownedConfigs[i] = ScopedDescriptor(createDescriptorPtr<EngineConfigDescriptor>());
    }
    std::vector<hipdnnBackendDescriptor_t> configs;
    configs.reserve(ownedConfigs.size());
    for(auto& owned : ownedConfigs)
    {
        configs.push_back(owned.get());
    }

    int64_t count = 0;
    ASSERT_NO_THROW(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       5,
                                       &count,
                                       static_cast<void*>(configs.data())));
    ASSERT_EQ(count, 3);
}

TEST_F(TestEngineHeuristicDescriptor, GetEngineConfigsCountOnly)
{
    auto heur = getEngineHeuristicDescriptor();
    makeEngineHeuristicFinalized();

    EXPECT_CALL(*getMockGraph(), isFinalized()).WillRepeatedly(Return(true));

    int64_t count = 0;
    ASSERT_NO_THROW(heur->getAttribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr));
    ASSERT_EQ(count, 3);
}

TEST_F(TestEngineHeuristicDescriptor, GetEngineHeuristicDescriptorHeurMode)
{
    auto heur = getEngineHeuristicDescriptor();
    hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;

    makeEngineHeuristicFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &mode),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 2, nullptr, &mode),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(
        heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr, &mode));
    ASSERT_EQ(mode, HIPDNN_HEUR_MODE_FALLBACK);

    int64_t count = 0;
    ASSERT_NO_THROW(
        heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &count, &mode));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(mode, HIPDNN_HEUR_MODE_FALLBACK);
}

TEST_F(TestEngineHeuristicDescriptor, GetGraphThrowsIfNotFinalized)
{
    auto heur = getEngineHeuristicDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(heur->getGraph(), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestEngineHeuristicDescriptor, GetGraphReturnsPointerIfFinalized)
{
    auto heur = getEngineHeuristicDescriptor();
    makeEngineHeuristicFinalized();
    auto graphPtr = heur->getGraph();
    ASSERT_NE(graphPtr, nullptr);
    ASSERT_EQ(static_cast<const IBackendDescriptor*>(graphPtr.get()),
              static_cast<const IBackendDescriptor*>(getMockGraph().get()));
}

TEST_F(TestEngineHeuristicDescriptor, SetFindFirst)
{
    auto heur = getEngineHeuristicDescriptor();
    bool findFirst = true;
    ASSERT_NO_THROW(heur->setAttribute(
        HIPDNN_ATTR_ENGINEHEUR_FIND_FIRST_EXT, HIPDNN_TYPE_BOOLEAN, 1, &findFirst));
}

TEST_F(TestEngineHeuristicDescriptor, SetFindFirstInvalidType)
{
    auto heur = getEngineHeuristicDescriptor();
    bool findFirst = true;
    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_FIND_FIRST_EXT, HIPDNN_TYPE_INT64, 1, &findFirst),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineHeuristicDescriptor, GetFindFirstAfterFinalize)
{
    auto heur = getEngineHeuristicDescriptor();
    bool findFirst = true;
    ASSERT_NO_THROW(heur->setAttribute(
        HIPDNN_ATTR_ENGINEHEUR_FIND_FIRST_EXT, HIPDNN_TYPE_BOOLEAN, 1, &findFirst));

    setGraph();
    setHeuristicMode();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, true))
        .WillOnce(Return(std::vector<int64_t>{42}));
    ASSERT_NO_THROW(heur->finalize());

    bool result = false;
    int64_t count = 0;
    ASSERT_NO_THROW(heur->getAttribute(
        HIPDNN_ATTR_ENGINEHEUR_FIND_FIRST_EXT, HIPDNN_TYPE_BOOLEAN, 1, &count, &result));
    ASSERT_TRUE(result);
    ASSERT_EQ(count, 1);
}

TEST_F(TestEngineHeuristicDescriptor, FinalizeWithFindFirstPassesToPluginManager)
{
    auto heur = getEngineHeuristicDescriptor();
    bool findFirst = true;
    ASSERT_NO_THROW(heur->setAttribute(
        HIPDNN_ATTR_ENGINEHEUR_FIND_FIRST_EXT, HIPDNN_TYPE_BOOLEAN, 1, &findFirst));

    setGraph();
    setHeuristicMode();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, true))
        .WillOnce(Return(std::vector<int64_t>{1}));
    ASSERT_NO_THROW(heur->finalize());
}

// ========== Policy Order API Tests ==========

TEST_F(TestEngineHeuristicDescriptor, SetPolicyOrderValid)
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

TEST_F(TestEngineHeuristicDescriptor, SetPolicyOrderInvalidType)
{
    auto heur = getEngineHeuristicDescriptor();
    const char dummy = '\0';

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT, HIPDNN_TYPE_CHAR, 1, &dummy),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineHeuristicDescriptor, SetPolicyOrderNullPointer)
{
    auto heur = getEngineHeuristicDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT, HIPDNN_TYPE_INT64, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestEngineHeuristicDescriptor, GetPolicyOrderWhenNotSet)
{
    // Make sure no env-var override leaks in from the surrounding shell.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter envGuard(
        "HIPDNN_HEURISTIC_POLICY_ORDER", "");

    auto heur = getEngineHeuristicDescriptor();
    setGraph();
    setHeuristicMode();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{1}));
    ASSERT_NO_THROW(heur->finalize());

    // With no descriptor-level override and no env var, resolveHeuristicPolicyOrder
    // returns the built-in default: just StaticOrdering.
    int64_t count = 999;
    ASSERT_NO_THROW(heur->getAttribute(
        HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT, HIPDNN_TYPE_INT64, 0, &count, nullptr));
    ASSERT_EQ(count, 1);

    std::vector<int64_t> buffer(1);
    ASSERT_NO_THROW(heur->getAttribute(
        HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT, HIPDNN_TYPE_INT64, 1, &count, buffer.data()));
    ASSERT_EQ(count, 1);
    EXPECT_EQ(buffer[0],
              hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering"));
}

TEST_F(TestEngineHeuristicDescriptor, GetPolicyOrderCountOnly)
{
    auto heur = getEngineHeuristicDescriptor();

    // Caller-provided policy list is preserved verbatim; nothing is prepended.
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

TEST_F(TestEngineHeuristicDescriptor, GetPolicyOrderInvalidType)
{
    auto heur = getEngineHeuristicDescriptor();
    setGraph();
    setHeuristicMode();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{1}));
    ASSERT_NO_THROW(heur->finalize());

    std::vector<char> buffer(256);
    int64_t count = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(
            HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT, HIPDNN_TYPE_CHAR, 256, &count, buffer.data()),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineHeuristicDescriptor, GetPolicyOrderNullPointer)
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

TEST_F(TestEngineHeuristicDescriptor, GetPolicyOrderNegativeRequestedCount)
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

    std::vector<int64_t> buffer(4);
    int64_t count = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        heur->getAttribute(
            HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT, HIPDNN_TYPE_INT64, -1, &count, buffer.data()),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineHeuristicDescriptor, GetPolicyOrderBufferTooSmall)
{
    auto heur = getEngineHeuristicDescriptor();

    // The caller-supplied list is stored verbatim; no dedup, no prepend.
    const int64_t firstId
        = hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering");
    const int64_t secondId = hipdnn_data_sdk::utilities::policyNameToId("Vendor::Other");
    const std::vector<int64_t> policyIds = {firstId, secondId};

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
    ASSERT_EQ(buffer[0], firstId);
}

TEST_F(TestEngineHeuristicDescriptor, GetPolicyOrderRoundTrip)
{
    auto heur = getEngineHeuristicDescriptor();

    // The descriptor stores the caller-supplied list verbatim, including
    // duplicates and unknown policies — nothing is prepended or dedup'd.
    const int64_t otherId
        = hipdnn_data_sdk::utilities::policyNameToId("Vendor::Other");
    const int64_t staticOrderingId
        = hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering");
    const std::vector<int64_t> policyIds = {staticOrderingId, otherId, staticOrderingId};
    const std::vector<int64_t>& expected = policyIds;

    ASSERT_NO_THROW(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                       HIPDNN_TYPE_INT64,
                                       static_cast<int64_t>(policyIds.size()),
                                       policyIds.data()));

    setGraph();
    setHeuristicMode();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{1}));
    ASSERT_NO_THROW(heur->finalize());

    std::vector<int64_t> getBuffer(expected.size());
    int64_t count = 0;
    ASSERT_NO_THROW(heur->getAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                       HIPDNN_TYPE_INT64,
                                       static_cast<int64_t>(getBuffer.size()),
                                       &count,
                                       getBuffer.data()));

    ASSERT_EQ(count, static_cast<int64_t>(expected.size()));
    for(size_t i = 0; i < expected.size(); ++i)
    {
        ASSERT_EQ(getBuffer[i], expected[i]);
    }
}

// ========== Exception Handling Tests ==========

TEST_F(TestEngineHeuristicDescriptor, FinalizeWithAllPoliciesFailing)
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

TEST_F(TestEngineHeuristicDescriptor, FinalizeWithPolicyThrowingException)
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

TEST_F(TestEngineHeuristicDescriptor, ToStringBeforeFinalize)
{
    auto heur = getEngineHeuristicDescriptor();
    const std::string str = heur->toString();
    ASSERT_NE(str.find("EngineHeuristicDescriptor"), std::string::npos);
    ASSERT_NE(str.find("unset"), std::string::npos);
}

TEST_F(TestEngineHeuristicDescriptor, ToStringAfterSetHeurMode)
{
    auto heur = getEngineHeuristicDescriptor();
    setHeuristicMode();
    const std::string str = heur->toString();
    ASSERT_NE(str.find("EngineHeuristicDescriptor"), std::string::npos);
    ASSERT_NE(str.find("heuristicMode"), std::string::npos);
}

TEST_F(TestEngineHeuristicDescriptor, ToStringAfterSetGraph)
{
    auto heur = getEngineHeuristicDescriptor();
    setGraph();
    const std::string str = heur->toString();
    ASSERT_NE(str.find("graph="), std::string::npos);
    ASSERT_EQ(str.find("graph=null"), std::string::npos); // Should not be null
}

TEST_F(TestEngineHeuristicDescriptor, ToStringWithPolicyOrder)
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

TEST_F(TestEngineHeuristicDescriptor, SetEmptyPolicyOrder)
{
    auto heur = getEngineHeuristicDescriptor();

    // Setting an empty policy order is allowed at the attribute level; finalize()
    // would later fail because no policy can be selected, but that is exercised by
    // FinalizeWithAllPoliciesFailing. Here we only verify the attribute path.
    ASSERT_NO_THROW(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT, HIPDNN_TYPE_INT64, 0, nullptr));
}

TEST_F(TestEngineHeuristicDescriptor, GetPolicyOrderNullElementCount)
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

TEST_F(TestEngineHeuristicDescriptor, MultipleSetPolicyOrderCalls)
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
        hipdnn_data_sdk::utilities::policyNameToId("Vendor::Other"),
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

// ========== Policy Order Resolution: Environment Variable ==========

TEST_F(TestEngineHeuristicDescriptor, EnvironmentVariablePolicyOrderIsRespected)
{
    // The mock setup in setupMockHeuristicPlugin() makes the catch-all return a
    // null handle for any unknown policy and StaticOrdering succeed. With no
    // descriptor-level override, the default order [StaticOrdering] therefore
    // succeeds. Restricting the env-var order to a policy nothing maps to should
    // make finalize() throw, proving the env var supersedes the default.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter guard(
        "HIPDNN_HEURISTIC_POLICY_ORDER", "Vendor::Unregistered");

    auto heur = getEngineHeuristicDescriptor();
    setGraph();
    setHeuristicMode();

    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{1, 2}));

    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestEngineHeuristicDescriptor, DescriptorPolicyOrderTakesPrecedenceOverEnvironment)
{
    // Same mock setup. Env var lists an unregistered policy (which would throw
    // on its own), but the descriptor-level attribute lists only StaticOrdering.
    // The descriptor attribute is highest priority, so finalize() must succeed.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter guard(
        "HIPDNN_HEURISTIC_POLICY_ORDER", "Vendor::Unregistered");

    auto heur = getEngineHeuristicDescriptor();

    const std::vector<int64_t> descriptorOrder = {
        hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering"),
    };
    ASSERT_NO_THROW(heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                       HIPDNN_TYPE_INT64,
                                       static_cast<int64_t>(descriptorOrder.size()),
                                       descriptorOrder.data()));

    setGraph();
    setHeuristicMode();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{1, 2}));

    ASSERT_NO_THROW(heur->finalize());
}

// ========== Failure Handling: Empty Policy List ==========

TEST_F(TestEngineHeuristicDescriptor, FinalizeWithEmptyPolicyListThrows)
{
    // Empty policy list reaches the "no policy succeeded" path via a different
    // route from FinalizeWithAllPoliciesFailing: the outer loop never executes
    // because there are no slots to try. Both paths must produce the same throw.
    auto heur = getEngineHeuristicDescriptor();

    ASSERT_NO_THROW(
        heur->setAttribute(HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT, HIPDNN_TYPE_INT64, 0, nullptr));

    setGraph();
    setHeuristicMode();
    EXPECT_CALL(*_mockEnginePluginResourceManager, getApplicableEngineIds(_, _))
        .WillRepeatedly(Return(std::vector<int64_t>{1, 2}));

    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_INTERNAL_ERROR);
}
