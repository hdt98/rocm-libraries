// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>
#include <memory>
#include <vector>

#include "CkFmhaHandle.hpp"

namespace ck_fmha_plugin {

class CkFmhaEngine
    : public hipdnn_plugin_sdk::IEngine<CkFmhaHandle, CkFmhaSettings, CkFmhaContext> {
   public:
    explicit CkFmhaEngine(int64_t id);

    int64_t id() const override;

    bool isApplicable(CkFmhaHandle& handle,
                      const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(CkFmhaHandle& handle,
                    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    size_t getMaxWorkspaceSize(
        const CkFmhaHandle& handle, const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const override;

    void initializeExecutionContext(
        const CkFmhaHandle& handle, const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        CkFmhaContext& executionContext) const override;

    void addPlanBuilder(
        std::unique_ptr<
            hipdnn_plugin_sdk::IPlanBuilder<CkFmhaHandle, CkFmhaSettings, CkFmhaContext>>
            planBuilder);

   private:
    int64_t id_;
    std::vector<std::unique_ptr<
        hipdnn_plugin_sdk::IPlanBuilder<CkFmhaHandle, CkFmhaSettings, CkFmhaContext>>>
        plan_builders_;
};

}  // namespace ck_fmha_plugin
