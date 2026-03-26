// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkFmhaBwdPlanBuilder.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "CkFmhaBwdPlan.hpp"
#include "CkFmhaContext.hpp"
#include "engines/CkFmhaParamParser.hpp"

namespace ck_fmha_plugin {

bool CkFmhaBwdPlanBuilder::isApplicable(
    const CkFmhaHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const {
    if (!CkFmhaParamParser::isBwdSdpaGraph(opGraph)) return false;

    try {
        auto params = CkFmhaParamParser::parseBwdGraph(opGraph);
        auto problem = CkFmhaParamParser::buildBwdProblem(params, handle.gfxArch());
        auto plan = handle.dispatcher()->plan(problem);
        return plan.is_valid();
    } catch (...) {
        return false;
    }
}

size_t CkFmhaBwdPlanBuilder::getMaxWorkspaceSize(
    const CkFmhaHandle& handle, const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    const CkFmhaSettings&) const {
    auto params = CkFmhaParamParser::parseBwdGraph(opGraph);
    auto problem = CkFmhaParamParser::buildBwdProblem(params, handle.gfxArch());
    auto ws = ck_tile::dispatcher::bwd_workspace_info(problem);
    return ws.total_bytes;
}

void CkFmhaBwdPlanBuilder::initializeExecutionSettings(
    const CkFmhaHandle&, const hipdnn_data_sdk::flatbuffer_utilities::IGraph&,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig&, CkFmhaSettings&) const {}

void CkFmhaBwdPlanBuilder::buildPlan(const CkFmhaHandle& handle,
                                     const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                                     const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig&,
                                     CkFmhaContext& ctx) const {
    auto params = CkFmhaParamParser::parseBwdGraph(opGraph);
    auto problem = CkFmhaParamParser::buildBwdProblem(params, handle.gfxArch());

    auto cache_key = problem.canonical_key();
    auto* cached = handle.getCachedPlan(cache_key);
    if (cached == nullptr) {
        auto plan = handle.dispatcher()->plan(problem);
        if (!plan.is_valid()) {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "No CK FMHA backward kernel for: " + problem.to_string());
        }
        const_cast<CkFmhaHandle&>(handle).cachePlan(cache_key, std::move(plan));
        cached = handle.getCachedPlan(cache_key);
    }

    auto ws = ck_tile::dispatcher::bwd_workspace_info(problem);
    ctx.setPlan(std::make_unique<CkFmhaBwdPlan>(params, *cached, ws));
}

std::vector<hipdnn_data_sdk::data_objects::KnobT> CkFmhaBwdPlanBuilder::getCustomKnobs(
    const CkFmhaHandle&, const hipdnn_data_sdk::flatbuffer_utilities::IGraph&) const {
    return {};
}

}  // namespace ck_fmha_plugin
