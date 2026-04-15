// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <set>
#include <string>

#include "HeuristicPlugin.hpp"
#include "HipdnnException.hpp"
#include "PluginCore.hpp"
#include <hipdnn_backend/version.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>

namespace hipdnn_backend::plugin
{

/**
 * @brief Manager for loading and validating heuristic plugin shared libraries.
 *
 * This class extends PluginManagerBase to provide heuristic-specific plugin
 * discovery, loading, and validation. It uses a separate search path from
 * engine plugins and validates heuristic-specific constraints.
 *
 * Validation includes:
 * - Heuristic C ABI major version compatibility
 * - Unique policy IDs across all loaded heuristic plugins
 * - Policy ID ↔ policy name consistency (when GetPolicyName is provided)
 */
class HeuristicPluginManager : public PluginManagerBase<HeuristicPlugin>
{
public:
    HeuristicPluginManager()
        : PluginManagerBase<HeuristicPlugin>(getPluginSearchPaths(
              "HIPDNN_HEURISTIC_PLUGIN_DIR", {std::filesystem::path("hipdnn_plugins/heuristics/")}))
    {
    }

protected:
    void validateBeforeAdding(const HeuristicPlugin& plugin) override
    {
        using hipdnn_data_sdk::utilities::Version;

        // Validate heuristic C ABI major version
        // Note: We're using HIPDNN_BACKEND_VERSION_MAJOR as a proxy for now.
        // TODO: Define separate HIPDNN_HEURISTIC_API_VERSION when schema is finalized.
        if(Version{plugin.apiVersion()}.major != HIPDNN_BACKEND_VERSION_MAJOR)
        {
            throw HipdnnException(
                HIPDNN_STATUS_PLUGIN_ERROR,
                "Heuristic plugin API major version (" + std::string(plugin.apiVersion())
                    + ") does not match backend major version ("
                    + std::to_string(HIPDNN_BACKEND_VERSION_MAJOR) + ")");
        }

        // Validate unique policy ID
        int64_t policyId = plugin.policyId();
        if(_policyIds.find(policyId) != _policyIds.end())
        {
            throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                  "Policy ID " + std::to_string(policyId)
                                      + " already exists in the list of loaded heuristic plugins");
        }

        // Validate policy ID ↔ policy name consistency (RFC §11, §5.3.1)
        // If GetPolicyName is provided, engineNameToId(name) MUST equal GetPolicyId()
        auto policyNameView = plugin.policyName();
        if(!policyNameView.empty())
        {
            std::string policyName(policyNameView);
            int64_t expectedId = hipdnn_data_sdk::utilities::engineNameToId(policyName);
            if(expectedId != policyId)
            {
                throw HipdnnException(
                    HIPDNN_STATUS_PLUGIN_ERROR,
                    "Policy ID mismatch: GetPolicyId() returned " + std::to_string(policyId)
                        + " but engineNameToId(\"" + policyName + "\") = "
                        + std::to_string(expectedId)
                        + ". Plugin must return consistent ID and name (RFC §11, §5.3.1)");
            }
        }
    }

    void actionAfterAdding(const HeuristicPlugin& plugin) override
    {
        _policyIds.insert(plugin.policyId());
    }

private:
    std::set<int64_t> _policyIds;
};

} // namespace hipdnn_backend::plugin
