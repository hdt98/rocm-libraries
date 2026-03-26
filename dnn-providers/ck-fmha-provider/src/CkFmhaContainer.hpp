// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "CkFmhaHandle.hpp"

namespace ck_fmha_plugin {

class CkFmhaContainer {
   public:
    CkFmhaContainer();
    ~CkFmhaContainer();

    static uint32_t copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines);

    hipdnn_plugin_sdk::EngineManager<CkFmhaHandle, CkFmhaSettings, CkFmhaContext>&
    getEngineManager();

   private:
    struct EngineDefinition {
        int64_t id;
        std::function<std::unique_ptr<
            hipdnn_plugin_sdk::IEngine<CkFmhaHandle, CkFmhaSettings, CkFmhaContext>>()>
            createEngine;
    };

    static const std::vector<EngineDefinition>& getEngineDefinitions();

    std::unique_ptr<hipdnn_plugin_sdk::EngineManager<CkFmhaHandle, CkFmhaSettings, CkFmhaContext>>
        engine_manager_;
};

}  // namespace ck_fmha_plugin
