// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkFmhaContainer.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/CkFmhaEngine.hpp"
#include "engines/plans/CkFmhaBwdPlanBuilder.hpp"
#include "engines/plans/CkFmhaFwdPlanBuilder.hpp"

namespace ck_fmha_plugin {

HIPDNN_REGISTER_ENGINE(CK_FMHA_ENGINE, "CK_FMHA_ENGINE")

const std::vector<CkFmhaContainer::EngineDefinition>& CkFmhaContainer::getEngineDefinitions() {
    static const std::vector<EngineDefinition> s_engineDefinitions = {
        {CK_FMHA_ENGINE_ID,
         []() -> std::unique_ptr<
                  hipdnn_plugin_sdk::IEngine<CkFmhaHandle, CkFmhaSettings, CkFmhaContext>> {
             auto engine = std::make_unique<CkFmhaEngine>(CK_FMHA_ENGINE_ID);
             engine->addPlanBuilder(std::make_unique<CkFmhaFwdPlanBuilder>());
             engine->addPlanBuilder(std::make_unique<CkFmhaBwdPlanBuilder>());
             return engine;
         }}};
    return s_engineDefinitions;
}

uint32_t CkFmhaContainer::copyEngineIds(int64_t* engineIds, uint32_t maxEngines,
                                        uint32_t& numEngines) {
    const auto& defs = getEngineDefinitions();
    auto total = static_cast<uint32_t>(defs.size());

    if (maxEngines == 0) {
        numEngines = total;
        return total;
    }

    auto count = std::min(maxEngines, total);
    for (uint32_t i = 0; i < count; ++i) engineIds[i] = defs[i].id;
    numEngines = count;
    return total;
}

CkFmhaContainer::CkFmhaContainer() {
    HIPDNN_PLUGIN_LOG_INFO("Creating CkFmhaContainer");
    engine_manager_ = std::make_unique<
        hipdnn_plugin_sdk::EngineManager<CkFmhaHandle, CkFmhaSettings, CkFmhaContext>>();

    for (const auto& def : getEngineDefinitions()) engine_manager_->addEngine(def.createEngine());
}

CkFmhaContainer::~CkFmhaContainer() {
    HIPDNN_PLUGIN_LOG_INFO("Destroying CkFmhaContainer");
}

hipdnn_plugin_sdk::EngineManager<CkFmhaHandle, CkFmhaSettings, CkFmhaContext>&
CkFmhaContainer::getEngineManager() {
    return *engine_manager_;
}

}  // namespace ck_fmha_plugin
