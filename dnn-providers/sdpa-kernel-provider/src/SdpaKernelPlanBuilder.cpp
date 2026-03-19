// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaKernelPlanBuilder.hpp"
#include "SdpaKernelPlan.hpp"
#include <iostream>

namespace sdpa_kernel_provider
{

bool SdpaKernelPlanBuilder::isApplicable(
    const SdpaKernelHandle& /*handle*/,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    auto& nodeWrappers = opGraph.nodeWrappers();

    if(nodeWrappers.size() != 1
       || nodeWrappers.front()->attributesType()
              != hipdnn_data_sdk::data_objects::NodeAttributes::SdpaAttributes)
    {
        return false;
    }

    // TODO: Add more expansive checks
    HIPDNN_PLUGIN_LOG_WARN("SdpaKernelPlanBuilder::isApplicable not fully implemented");

    return true;
}

size_t SdpaKernelPlanBuilder::getMaxWorkspaceSize(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    const SdpaKernelSettings& /* executionSettings */) const
{
    // Get SDPA attributes from the first (and only) node
    auto& sdpaNode = opGraph.getNodeWrapper(0);
    auto& sdpaAttrs = sdpaNode.attributesAs<hipdnn_data_sdk::data_objects::SdpaAttributes>();

    // Get Q tensor to extract batch, head, sequence dimensions
    auto& tensorMap = opGraph.getTensorMap();
    const hipdnn_data_sdk::data_objects::TensorAttributes* qTensor = tensorMap.at(sdpaAttrs.q_tensor_uid());
    auto* qDims = qTensor->dims();

    // Q tensor layout: [B, H_q, S_q, D]
    // Workspace = LSE buffer: [B, H_q, S_q] in float32
    size_t B   = static_cast<size_t>(qDims->Get(0));  // batch
    size_t H_q = static_cast<size_t>(qDims->Get(1));  // num heads
    size_t S_q = static_cast<size_t>(qDims->Get(2));  // sequence length

    return B * H_q * S_q * sizeof(float);
}

void SdpaKernelPlanBuilder::initializeExecutionSettings(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    SdpaKernelSettings& /* executionSettings */) const
{
    HIPDNN_PLUGIN_LOG_ERROR("SdpaKernelPlanBuilder::initializeExecutionContext not implemented");
}

void SdpaKernelPlanBuilder::buildPlan(
    const SdpaKernelHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    SdpaKernelContext& executionContext) const
{
    // Compute workspace size
    size_t workspaceSize = getMaxWorkspaceSize(handle, opGraph, SdpaKernelSettings());

    // Create plan with precomputed workspace
    executionContext.setPlan(std::make_unique<SdpaKernelPlan>(workspaceSize));
}

std::vector<hipdnn_data_sdk::data_objects::KnobT> SdpaKernelPlanBuilder::getCustomKnobs(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */) const
{
    return {};
}

} // namespace sdpa_kernel_provider
