// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkFmhaEngine.hpp"

#include <hipdnn_data_sdk/data_objects/engine_details_generated.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "CkFmhaContext.hpp"

namespace ck_fmha_plugin {

CkFmhaEngine::CkFmhaEngine(int64_t id) : id_(id) {}

int64_t CkFmhaEngine::id() const {
    return id_;
}

bool CkFmhaEngine::isApplicable(
    CkFmhaHandle& handle, const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const {
    for (const auto& builder : plan_builders_) {
        if (builder->isApplicable(handle, opGraph)) return true;
    }
    return false;
}

void CkFmhaEngine::getDetails(CkFmhaHandle& handle,
                              const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                              hipdnnPluginConstData_t& detailsOut) const {
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Knob>> knobs_vec;

    for (const auto& pb : plan_builders_) {
        auto custom = pb->getCustomKnobs(handle, opGraph);
        for (const auto& knob_t : custom) {
            knobs_vec.push_back(hipdnn_data_sdk::data_objects::Knob::Pack(builder, &knob_t));
        }
        if (!custom.empty()) break;
    }

    auto knobs = builder.CreateVector(knobs_vec);
    auto details = hipdnn_data_sdk::data_objects::CreateEngineDetails(builder, id_, knobs);
    builder.Finish(details);

    auto detached = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detached->data();
    detailsOut.size = detached->size();
    handle.storeEngineDetailsDetachedBuffer(detailsOut.ptr, std::move(detached));
}

size_t CkFmhaEngine::getMaxWorkspaceSize(
    const CkFmhaHandle& handle, const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const {
    size_t max_ws = 0;
    for (const auto& pb : plan_builders_) {
        if (pb->isApplicable(handle, opGraph)) {
            CkFmhaSettings settings;
            pb->initializeExecutionSettings(handle, opGraph, engineConfig, settings);
            max_ws = std::max(max_ws, pb->getMaxWorkspaceSize(handle, opGraph, settings));
        }
    }
    return max_ws;
}

void CkFmhaEngine::initializeExecutionContext(
    const CkFmhaHandle& handle, const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    CkFmhaContext& ctx) const {
    CkFmhaSettings settings;
    for (const auto& pb : plan_builders_) {
        if (pb->isApplicable(handle, opGraph)) {
            pb->initializeExecutionSettings(handle, opGraph, engineConfig, settings);
            break;
        }
    }
    ctx.setExecutionSettings(settings);

    for (const auto& pb : plan_builders_) {
        if (pb->isApplicable(handle, opGraph)) {
            pb->buildPlan(handle, opGraph, engineConfig, ctx);
            break;
        }
    }
}

void CkFmhaEngine::addPlanBuilder(
    std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<CkFmhaHandle, CkFmhaSettings, CkFmhaContext>>
        builder) {
    plan_builders_.push_back(std::move(builder));
}

}  // namespace ck_fmha_plugin
