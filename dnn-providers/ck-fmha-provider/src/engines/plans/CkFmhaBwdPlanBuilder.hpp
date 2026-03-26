// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "CkFmhaHandle.hpp"

namespace ck_fmha_plugin {

class CkFmhaBwdPlanBuilder
    : public hipdnn_plugin_sdk::IPlanBuilder<CkFmhaHandle, CkFmhaSettings, CkFmhaContext> {
   public:
    bool isApplicable(const CkFmhaHandle& handle,
                      const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const CkFmhaHandle& handle,
                               const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const CkFmhaSettings& settings) const override;

    void initializeExecutionSettings(
        const CkFmhaHandle& handle, const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        CkFmhaSettings& settings) const override;

    void buildPlan(const CkFmhaHandle& handle,
                   const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                   const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                   CkFmhaContext& ctx) const override;

    std::vector<hipdnn_data_sdk::data_objects::KnobT> getCustomKnobs(
        const CkFmhaHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
};

}  // namespace ck_fmha_plugin
