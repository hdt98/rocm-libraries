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
#include <hipdnn_plugin_sdk/heuristic_api_version.h>

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
 * - Unique policy IDs across all loaded heuristic plugins (one plugin may expose many)
 * - Plugin (library) name is provided via hipdnnPluginGetName()
 *
 * A single heuristic plugin shared library may expose one or more selection
 * policies. Each policy is identified by a stable int64 policy ID derived from
 * the canonical policy name (FNV-1a hash via policyNameToId). The plugin layer
 * validates the policy ID/name pairing eagerly at load; this manager enforces
 * uniqueness across all loaded plugins.
 */
class HeuristicPluginManager : public PluginManagerBase<HeuristicPlugin>
{
public:
    HeuristicPluginManager()
        : PluginManagerBase<HeuristicPlugin>(getPluginSearchPaths(
              "HIPDNN_HEURISTIC_PLUGIN_DIR",
              {std::filesystem::path("hipdnn_plugins/heuristics/")}))
    {
    }

protected:
    void validateBeforeAdding(const HeuristicPlugin& plugin) override
    {
        using hipdnn_data_sdk::utilities::Version;

        // Validate heuristic C ABI major version against the heuristic API version
        // (the heuristic plugin API has independent versioning from the backend)
        if(Version{plugin.apiVersion()}.major != HIPDNN_HEURISTIC_API_VERSION_MAJOR)
        {
            throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                  "ERROR: HEURISTIC PLUGIN ABI VALIDATION FAILED\n"
                                  "Plugin API major version ("
                                      + std::string(plugin.apiVersion())
                                      + ") does not match expected heuristic API major version ("
                                      + std::to_string(HIPDNN_HEURISTIC_API_VERSION_MAJOR) + ")\n"
                                      + "The plugin was compiled against an incompatible version "
                                        "of HeuristicsPluginApi.h\n"
                                      + "Expected API version: " HIPDNN_HEURISTIC_API_VERSION);
        }

        // Validate plugin (library) name is provided
        if(plugin.name().empty())
        {
            throw HipdnnException(
                HIPDNN_STATUS_PLUGIN_ERROR,
                "ERROR: HEURISTIC PLUGIN VALIDATION FAILED\n"
                "Plugin name is required but was not provided.\n"
                "Plugin must implement hipdnnPluginGetName() and return a non-empty name.");
        }

        // Validate every policy ID is globally unique across loaded plugins.
        // The plugin layer (HeuristicPlugin::resolveSymbols) already checks intra-plugin
        // uniqueness and policyNameToId(name) == policyId; here we extend the check across
        // the full set of loaded plugins.
        const auto policyIds = plugin.getAllPolicyIds();
        for(const int64_t policyId : policyIds)
        {
            if(_policyIds.find(policyId) != _policyIds.end())
            {
                throw HipdnnException(
                    HIPDNN_STATUS_PLUGIN_ERROR,
                    "ERROR: HEURISTIC PLUGIN VALIDATION FAILED\n"
                    "Policy ID "
                        + std::to_string(policyId)
                        + " already exists in the list of loaded heuristic plugins.\n"
                        + "Each policy must have a unique ID.");
            }
        }
    }

    void actionAfterAdding(const HeuristicPlugin& plugin) override
    {
        const auto policyIds = plugin.getAllPolicyIds();
        _policyIds.insert(policyIds.begin(), policyIds.end());
    }

    void actionAfterClearing() override
    {
        _policyIds.clear();
    }

private:
    std::set<int64_t> _policyIds;
};

} // namespace hipdnn_backend::plugin
