// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkFmhaFwdPlanBuilder.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "CkFmhaContext.hpp"
#include "CkFmhaFwdPlan.hpp"
#include "engines/CkFmhaParamParser.hpp"

namespace ck_fmha_plugin {

bool CkFmhaFwdPlanBuilder::isApplicable(
    const CkFmhaHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const {
    if (!CkFmhaParamParser::isFwdSdpaGraph(opGraph)) return false;

    try {
        auto params = CkFmhaParamParser::parseFwdGraph(opGraph);
        auto problem = CkFmhaParamParser::buildFwdProblem(params, handle.gfxArch());

        if (handle.dispatcher()->select_kernel(problem) != nullptr) return true;

        // No precompiled kernel -- try JIT if enabled
        return handle.jitAndLoad(problem);
    } catch (...) {
        return false;
    }
}

size_t CkFmhaFwdPlanBuilder::getMaxWorkspaceSize(
    const CkFmhaHandle&, const hipdnn_data_sdk::flatbuffer_utilities::IGraph&,
    const CkFmhaSettings&) const {
    return 0;
}

void CkFmhaFwdPlanBuilder::initializeExecutionSettings(
    const CkFmhaHandle&, const hipdnn_data_sdk::flatbuffer_utilities::IGraph&,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig&, CkFmhaSettings&) const {}

void CkFmhaFwdPlanBuilder::buildPlan(const CkFmhaHandle& handle,
                                     const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                                     const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig&,
                                     CkFmhaContext& ctx) const {
    auto params = CkFmhaParamParser::parseFwdGraph(opGraph);
    auto problem = CkFmhaParamParser::buildFwdProblem(params, handle.gfxArch());

    auto cache_key = problem.canonical_key();
    auto* cached = handle.getCachedPlan(cache_key);
    if (cached == nullptr) {
        auto plan = handle.dispatcher()->plan(problem);

        // JIT fallback: if no plan from precompiled kernels, try JIT
        if (!plan.is_valid()) {
            if (handle.jitAndLoad(problem)) plan = handle.dispatcher()->plan(problem);
        }

        if (!plan.is_valid()) {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "No CK FMHA forward kernel for: " + problem.to_string());
        }
        handle.cachePlan(cache_key, std::move(plan));
        cached = handle.getCachedPlan(cache_key);
    }

    ctx.setPlan(std::make_unique<CkFmhaFwdPlan>(params, *cached));
}

std::vector<hipdnn_data_sdk::data_objects::KnobT> CkFmhaFwdPlanBuilder::getCustomKnobs(
    const CkFmhaHandle&, const hipdnn_data_sdk::flatbuffer_utilities::IGraph&) const {
    return {};
}

}  // namespace ck_fmha_plugin
