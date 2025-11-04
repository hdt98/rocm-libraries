/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier:  MIT */

#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <hipdnn_sdk/plugin/PluginFlatbufferTypeHelpers.hpp>
#include <miopen/miopen.h>
#include <string>

#include "MiopenBatchnormPlanBuilder.hpp"
#include "engines/plans/MiopenBatchnormBwdPlan.hpp"
#include "engines/plans/MiopenBatchnormFwdInferencePlan.hpp"
#include "engines/plans/MiopenBatchnormFwdTrainingPlan.hpp"

namespace miopen_legacy_plugin
{

namespace
{

std::string getNodeName(const hipdnn_sdk::data_objects::Node& node)
{
    return node.name() != nullptr ? node.name()->str() : "";
}

} // namespace

bool MiopenBatchnormPlanBuilder::isApplicable(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::IGraph& opGraph) const
{
    if(opGraph.nodeCount() != 1)
    {
        HIPDNN_LOG_INFO(
            "Batchnorm plan builder is applicable only for single node graphs. Graph has {} nodes",
            opGraph.nodeCount());
        return false;
    }

    if(!opGraph.hasOnlySupportedAttributes(std::set<hipdnn_sdk::data_objects::NodeAttributes>{
           hipdnn_sdk::data_objects::NodeAttributes::BatchnormAttributes,
           hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
           hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes}))
    {
        HIPDNN_LOG_INFO("Batchnorm plan builder is not applicable for this graph");
        return false;
    }

    // Check if batchnorm training node has running statistics
    // API mismatch: hipDNN graph API uses separate prev/next buffers for running statistics,
    // but MIOpen requires single IN/OUT buffers. This cannot be correctly bridged without
    // either updating MIOpen API or implementing buffer copy operations.
    const auto& node = opGraph.getNode(0);

    // Only batchnorm training (BatchnormAttributes) has running statistics
    if(node.attributes_type() == hipdnn_sdk::data_objects::NodeAttributes::BatchnormAttributes)
    {
        const auto* attr = node.attributes_as_BatchnormAttributes();
        if(attr != nullptr && attr->prev_running_mean_tensor_uid().has_value()
           && attr->prev_running_variance_tensor_uid().has_value()
           && attr->momentum_tensor_uid().has_value()
           && attr->next_running_mean_tensor_uid().has_value()
           && attr->next_running_variance_tensor_uid().has_value())
        {
            HIPDNN_LOG_INFO("Batchnorm plan builder does not support running statistics");
            return false;
        }
    }

    return true;
}

size_t MiopenBatchnormPlanBuilder::getWorkspaceSize(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    [[maybe_unused]] const hipdnn_plugin::IGraph& opGraph) const
{
    //batchnorm plan builder does not require workspace size
    return 0u;
}

namespace
{

void buildPlanInferenceSingleNode([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                                  const hipdnn_plugin::IGraph& opGraph,
                                  const hipdnn_sdk::data_objects::Node& node,
                                  HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto* attr = node.attributes_as_BatchnormInferenceAttributes();
    if(attr == nullptr)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Failed to convert node attributes to BatchnormInferenceAttributes for node: "
                + getNodeName(node));
    }

    BatchnormFwdInferenceParams params(*attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormFwdInferencePlan>(std::move(params));
    executionContext.setPlan(std::move(plan));
}

void buildPlanFwdTrainingSingleNode([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                                    const hipdnn_plugin::IGraph& opGraph,
                                    const hipdnn_sdk::data_objects::Node& node,
                                    HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto* attr = node.attributes_as_BatchnormAttributes();
    if(attr == nullptr)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Failed to convert node attributes to BatchnormAttributes for node: "
                + getNodeName(node));
    }

    BatchnormFwdTrainingParams params(*attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormFwdTrainingPlan>(std::move(params));
    executionContext.setPlan(std::move(plan));
}

void buildPlanBwdSingleNode([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                            const hipdnn_plugin::IGraph& opGraph,
                            const hipdnn_sdk::data_objects::Node& node,
                            HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto* attr = node.attributes_as_BatchnormBackwardAttributes();
    if(attr == nullptr)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Failed to convert node attributes to BatchnormBackwardAttributes for node: "
                + getNodeName(node));
    }

    BatchnormBwdParams params(*attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormBwdPlan>(std::move(params));
    executionContext.setPlan(std::move(plan));
}

} // namespace

void MiopenBatchnormPlanBuilder::buildPlan(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    const auto& node = opGraph.getNode(0);

    std::string nodeName = getNodeName(node);
    switch(node.attributes_type())
    {
    case hipdnn_sdk::data_objects::NodeAttributes::BatchnormAttributes:
        HIPDNN_LOG_INFO("Building batchnorm fwd training plan for node: {}", nodeName);
        buildPlanFwdTrainingSingleNode(handle, opGraph, node, executionContext);
        break;
    case hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
        HIPDNN_LOG_INFO("Building batchnorm fwd inference plan for node: {}", nodeName);
        buildPlanInferenceSingleNode(handle, opGraph, node, executionContext);
        break;
    case hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
        HIPDNN_LOG_INFO("Building batchnorm backward plan for node: {}", nodeName);
        buildPlanBwdSingleNode(handle, opGraph, node, executionContext);
        break;
    default:
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for batchnorm plan builder: "
                + std::string(hipdnn_sdk::data_objects::toString(node.attributes_type())));
    }
}

} // namespace miopen_legacy_plugin
