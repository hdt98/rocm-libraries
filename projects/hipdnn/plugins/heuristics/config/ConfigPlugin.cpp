// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file ConfigPlugin.cpp
 * @brief Config heuristic plugin - honors graph-level and config-file preferences
 *
 * This plugin implements configuration-based engine selection. During Finalize
 * it resolves a preferred engine ID from two sources, in priority order:
 *   1. Graph.preferred_engine_id, set via the frontend
 *      Graph::set_preferred_engine_id_ext() setter and packed into the
 *      serialized FlatBuffer graph.
 *   2. A rule from the JSON file pointed to by HIPDNN_ENGINE_OVERRIDE_FILE that
 *      matches the first conv-like node in the graph (parsed and cached for the
 *      process by EngineOverrideConfig::loadFromEnv).
 *
 * If a preferred ID is resolved and it appears in the candidate list, the
 * plugin reorders the candidates so the preferred ID comes first while
 * preserving the relative order of the remaining IDs. Otherwise it returns
 * NOT_APPLICABLE so the next policy in the chain (StaticOrdering by default)
 * runs.
 *
 * Graph-level preferences are interpreted by this policy, not by hard-coded
 * backend logic.
 */

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/HeuristicValidation.hpp>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

#include "EngineOverrideConfig.hpp"

// Plugin version and identification
static constexpr const char* PLUGIN_VERSION = "1.0.0";
static constexpr const char* API_VERSION = "0.0.1"; // Match HIPDNN_HEURISTIC_API_VERSION
static constexpr const char* POLICY_NAME = "SelectionHeuristic::Config";

// Logging callback and log level
static hipdnnCallback_t g_loggingCallback = nullptr;
static hipdnnSeverity_t g_logLevel = HIPDNN_SEV_INFO;

// Log buffer size
constexpr size_t LOG_BUFFER_SIZE = 1024;

// Helper macro for logging
#define CONFIG_LOG(severity, ...)                                            \
    do                                                                       \
    {                                                                        \
        if(g_loggingCallback != nullptr && (severity) >= g_logLevel)         \
        {                                                                    \
            std::array<char, LOG_BUFFER_SIZE> buffer;                        \
            snprintf(buffer.data(), buffer.size(), "[Config] " __VA_ARGS__); \
            g_loggingCallback(severity, buffer.data());                      \
        }                                                                    \
    } while(0)

//=============================================================================
// Plugin Handle Implementation
//=============================================================================

struct ConfigHandle
{
    // Device properties (stored when SetDeviceProperties is called)
    std::vector<uint8_t> devicePropertiesBuffer;
    bool devicePropertiesSet = false;

    ConfigHandle()
    {
        CONFIG_LOG(HIPDNN_SEV_INFO, "Handle created");
    }

    ~ConfigHandle()
    {
        CONFIG_LOG(HIPDNN_SEV_INFO, "Handle destroyed");
    }
};

//=============================================================================
// Policy Descriptor Implementation
//=============================================================================

struct ConfigPolicyDescriptor
{
    ConfigHandle* handle = nullptr;
    std::vector<int64_t> candidateEngineIds;
    std::vector<uint8_t> serializedGraphBuffer;
    std::vector<int64_t> sortedEngineIds;
    bool finalized = false;

    ConfigPolicyDescriptor(ConfigHandle* h)
        : handle(h)
    {
        CONFIG_LOG(HIPDNN_SEV_INFO, "Policy descriptor created");
    }

    ~ConfigPolicyDescriptor()
    {
        CONFIG_LOG(HIPDNN_SEV_INFO, "Policy descriptor destroyed");
    }
};

//=============================================================================
// C ABI Implementation - Base Plugin API (from PluginApi.h)
//=============================================================================

extern "C" {

// Base plugin metadata
HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    if(name == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *name = POLICY_NAME;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    if(version == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *version = PLUGIN_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetApiVersion(const char** version)
{
    if(version == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *version = API_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    if(type == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *type = HIPDNN_PLUGIN_TYPE_HEURISTIC;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// Base plugin logging
HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)
{
    g_loggingCallback = callback;
    CONFIG_LOG(HIPDNN_SEV_INFO, "Logging callback set");
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginSetLogLevel(hipdnnSeverity_t level)
{
    g_logLevel = level;
    CONFIG_LOG(HIPDNN_SEV_INFO, "Log level set to %d", level);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT void hipdnnPluginGetLastErrorString(const char** errorStr)
{
    if(errorStr == nullptr)
    {
        return;
    }
    *errorStr = "No error information available";
}

//=============================================================================
// C ABI Implementation - Heuristic Plugin Extensions
//=============================================================================

//=============================================================================
// C ABI Implementation - Handle Lifecycle
//=============================================================================

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicHandleCreate(hipdnnHeuristicHandle_t* outHandle)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(outHandle, CONFIG_LOG, "HandleCreate: null output pointer");

    try
    {
        auto handle = new ConfigHandle();
        *outHandle = reinterpret_cast<hipdnnHeuristicHandle_t>(handle);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "HandleCreate failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicHandleDestroy(hipdnnHeuristicHandle_t handle)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(handle, CONFIG_LOG, "HandleDestroy: null handle");

    try
    {
        auto h = reinterpret_cast<ConfigHandle*>(handle);
        delete h;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "HandleDestroy failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicHandleSetDeviceProperties(
    hipdnnHeuristicHandle_t handle, const hipdnnPluginConstData_t* devicePropsSerialized)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(handle, CONFIG_LOG, "SetDeviceProperties: null handle");
    HIPDNN_PLUGIN_REQUIRE_CONST_DATA(
        devicePropsSerialized, true, CONFIG_LOG, "SetDeviceProperties: invalid buffer");

    try
    {
        auto h = reinterpret_cast<ConfigHandle*>(handle);

        // Store device properties (Config doesn't use them currently, but store for completeness)
        const auto* data = reinterpret_cast<const uint8_t*>(devicePropsSerialized->ptr);
        h->devicePropertiesBuffer.assign(data, data + devicePropsSerialized->size);
        h->devicePropertiesSet = true;

        CONFIG_LOG(
            HIPDNN_SEV_INFO, "Device properties set (%zu bytes)", devicePropsSerialized->size);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetDeviceProperties failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

//=============================================================================
// C ABI Implementation - Policy Descriptor Lifecycle
//=============================================================================

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPolicyDescriptorCreate(
    hipdnnHeuristicHandle_t pluginHandle, hipdnnHeuristicPolicyDescriptor_t* outDesc)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(pluginHandle, CONFIG_LOG, "PolicyDescriptorCreate: null handle");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        outDesc, CONFIG_LOG, "PolicyDescriptorCreate: null output pointer");

    try
    {
        auto h = reinterpret_cast<ConfigHandle*>(pluginHandle);
        auto desc = new ConfigPolicyDescriptor(h);
        *outDesc = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(desc);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyDescriptorCreate failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t desc)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, CONFIG_LOG, "PolicyDescriptorDestroy: null descriptor");

    try
    {
        auto d = reinterpret_cast<ConfigPolicyDescriptor*>(desc);
        delete d;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyDescriptorDestroy failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

//=============================================================================
// C ABI Implementation - Policy Inputs
//=============================================================================

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPolicySetEngineIds(
    hipdnnHeuristicPolicyDescriptor_t desc, const int64_t* engineIds, size_t engineIdCount)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, CONFIG_LOG, "SetEngineIds: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_ARRAY(
        engineIds, engineIdCount, CONFIG_LOG, "SetEngineIds: null engine_ids with count > 0");

    try
    {
        auto d = reinterpret_cast<ConfigPolicyDescriptor*>(desc);
        d->candidateEngineIds.assign(engineIds, engineIds + engineIdCount);
        d->finalized = false; // Reset finalized state when inputs change

        CONFIG_LOG(HIPDNN_SEV_INFO, "SetEngineIds: %zu candidate engines", engineIdCount);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetEngineIds failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPolicySetSerializedGraph(
    hipdnnHeuristicPolicyDescriptor_t desc, const hipdnnPluginConstData_t* serializedGraph)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, CONFIG_LOG, "SetSerializedGraph: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_CONST_DATA(
        serializedGraph, false, CONFIG_LOG, "SetSerializedGraph: null graph pointer");

    try
    {
        auto d = reinterpret_cast<ConfigPolicyDescriptor*>(desc);

        // Store serialized graph for parsing during finalize
        const auto* data = reinterpret_cast<const uint8_t*>(serializedGraph->ptr);
        d->serializedGraphBuffer.assign(data, data + serializedGraph->size);

        CONFIG_LOG(
            HIPDNN_SEV_INFO, "SetSerializedGraph: graph stored (%zu bytes)", serializedGraph->size);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetSerializedGraph failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

} // extern "C"

//=============================================================================
// C ABI Implementation - Selection
//=============================================================================

namespace
{

using hipdnn_flatbuffers_sdk::data_objects::Graph;

std::vector<int64_t> toVector(const flatbuffers::Vector<int64_t>* fb)
{
    if(fb == nullptr)
    {
        return {};
    }
    return {fb->begin(), fb->end()};
}

struct TensorDimsStrides
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

/// Extract every (uid, dims, strides) triple from the FlatBuffer graph so that
/// node-level UID lookups during the conv scan are O(1). Owns the dim/stride
/// vectors so callers can hand TensorView pointers into them safely.
std::unordered_map<int64_t, TensorDimsStrides> indexTensorsByUid(const Graph* graph)
{
    std::unordered_map<int64_t, TensorDimsStrides> out;
    const auto* tensors = graph->tensors();
    if(tensors == nullptr)
    {
        return out;
    }
    out.reserve(tensors->size());
    for(const auto* t : *tensors)
    {
        if(t == nullptr)
        {
            continue;
        }
        out.emplace(t->uid(), TensorDimsStrides{toVector(t->dims()), toVector(t->strides())});
    }
    return out;
}

/// Walk conv-like nodes; for each, look the override config up and return the
/// first match (first conv node, first matching rule).
std::optional<int64_t>
    matchOverrideConfig(const Graph* graph,
                        const std::unordered_map<int64_t, TensorDimsStrides>& tensorIndex)
{
    const auto config = hipdnn_heuristic_config::EngineOverrideConfig::loadFromEnv();
    if(!config.has_value())
    {
        return std::nullopt;
    }
    const auto* nodes = graph->nodes();
    if(nodes == nullptr)
    {
        return std::nullopt;
    }

    auto viewFor = [&](int64_t uid) -> const TensorDimsStrides* {
        auto it = tensorIndex.find(uid);
        return it == tensorIndex.end() ? nullptr : &it->second;
    };

    auto buildView = [&](const TensorDimsStrides* t) {
        return hipdnn_heuristic_config::TensorView{&t->dims, &t->strides};
    };

    for(const auto* node : *nodes)
    {
        if(node == nullptr)
        {
            continue;
        }

        const char* op = nullptr;
        const TensorDimsStrides* a = nullptr;
        const TensorDimsStrides* b = nullptr;

        if(const auto* fwd = node->attributes_as_ConvolutionFwdAttributes())
        {
            op = "conv_fprop";
            a = viewFor(fwd->x_tensor_uid());
            b = viewFor(fwd->w_tensor_uid());
        }
        else if(const auto* bwd = node->attributes_as_ConvolutionBwdAttributes())
        {
            op = "conv_dgrad";
            a = viewFor(bwd->dy_tensor_uid());
            b = viewFor(bwd->w_tensor_uid());
        }
        else if(const auto* wrw = node->attributes_as_ConvolutionWrwAttributes())
        {
            op = "conv_wgrad";
            a = viewFor(wrw->x_tensor_uid());
            b = viewFor(wrw->dy_tensor_uid());
        }

        if(op == nullptr || a == nullptr || b == nullptr)
        {
            continue;
        }

        const std::vector<hipdnn_heuristic_config::TensorView> views{buildView(a), buildView(b)};
        auto match = config->matchOperation(op, views);
        if(match.has_value())
        {
            return match;
        }
    }
    return std::nullopt;
}

} // namespace

extern "C" {

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyFinalize(hipdnnHeuristicPolicyDescriptor_t desc, int32_t* outApplied)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, CONFIG_LOG, "PolicyFinalize: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(outApplied, CONFIG_LOG, "PolicyFinalize: null output pointer");

    try
    {
        auto d = reinterpret_cast<ConfigPolicyDescriptor*>(desc);

        if(d->candidateEngineIds.empty())
        {
            CONFIG_LOG(HIPDNN_SEV_WARN, "PolicyFinalize: no candidate engines - not applicable");
            *outApplied = 0;
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        if(d->serializedGraphBuffer.empty())
        {
            CONFIG_LOG(HIPDNN_SEV_WARN, "PolicyFinalize: no serialized graph - not applicable");
            *outApplied = 0;
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        // Parse the serialized graph using FlatBuffers
        flatbuffers::Verifier verifier(d->serializedGraphBuffer.data(),
                                       d->serializedGraphBuffer.size());
        if(!verifier.VerifyBuffer<hipdnn_flatbuffers_sdk::data_objects::Graph>())
        {
            CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyFinalize: invalid graph buffer");
            return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
        }

        const auto* graph
            = hipdnn_flatbuffers_sdk::data_objects::GetGraph(d->serializedGraphBuffer.data());
        if(graph == nullptr)
        {
            CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyFinalize: failed to get graph root");
            return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
        }

        // Resolve a preferred engine ID, in priority order:
        //   1. Explicit graph.preferred_engine_id (set via the frontend setter).
        //   2. HIPDNN_ENGINE_OVERRIDE_FILE rule that matches a conv node.
        // Earlier signals win; nothing matched -> policy not applicable.
        std::optional<int64_t> preferredEngineId;
        const char* preferredSource = nullptr;

        if(auto preferredEngineIdOpt = graph->preferred_engine_id();
           preferredEngineIdOpt.has_value())
        {
            preferredEngineId = preferredEngineIdOpt;
            preferredSource = "graph.preferred_engine_id";
        }
        else
        {
            const auto tensorIndex = indexTensorsByUid(graph);
            preferredEngineId = matchOverrideConfig(graph, tensorIndex);
            if(preferredEngineId.has_value())
            {
                preferredSource = "HIPDNN_ENGINE_OVERRIDE_FILE";
            }
        }

        if(!preferredEngineId.has_value())
        {
            CONFIG_LOG(HIPDNN_SEV_INFO,
                       "PolicyFinalize: no preferred engine resolved - not applicable");
            *outApplied = 0;
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        // Check if the preferred engine ID is in the candidates
        auto it = std::find(
            d->candidateEngineIds.begin(), d->candidateEngineIds.end(), *preferredEngineId);
        if(it == d->candidateEngineIds.end())
        {
            CONFIG_LOG(HIPDNN_SEV_INFO,
                       "PolicyFinalize: preferred engine 0x%llx (from %s) not in candidates - "
                       "not applicable",
                       static_cast<long long>(*preferredEngineId),
                       preferredSource);
            *outApplied = 0;
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        d->sortedEngineIds.clear();
        d->sortedEngineIds.reserve(d->candidateEngineIds.size());
        d->sortedEngineIds.push_back(*preferredEngineId);
        for(const int64_t engineId : d->candidateEngineIds)
        {
            if(engineId != *preferredEngineId)
            {
                d->sortedEngineIds.push_back(engineId);
            }
        }

        d->finalized = true;
        *outApplied = 1;

        CONFIG_LOG(HIPDNN_SEV_INFO,
                   "PolicyFinalize: reordered %zu engines with preferred engine 0x%llx (from %s) "
                   "first",
                   d->sortedEngineIds.size(),
                   static_cast<long long>(*preferredEngineId),
                   preferredSource);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyFinalize failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPolicyGetSortedEngineIds(
    hipdnnHeuristicPolicyDescriptor_t desc, int64_t* engineIds, size_t* numEngines)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, CONFIG_LOG, "GetSortedEngineIds: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        numEngines, CONFIG_LOG, "GetSortedEngineIds: null num_engines pointer");

    try
    {
        auto d = reinterpret_cast<ConfigPolicyDescriptor*>(desc);

        if(!d->finalized)
        {
            CONFIG_LOG(HIPDNN_SEV_ERROR, "GetSortedEngineIds: descriptor not finalized");
            return HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED;
        }

        // Return count only if engineIds is null
        if(engineIds == nullptr)
        {
            *numEngines = d->sortedEngineIds.size();
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        // Copy sorted engine IDs (clip to actual available count)
        *numEngines = std::min(*numEngines, d->sortedEngineIds.size());
        std::copy_n(d->sortedEngineIds.begin(), *numEngines, engineIds);

        CONFIG_LOG(HIPDNN_SEV_INFO, "GetSortedEngineIds: returned %zu engines", *numEngines);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "GetSortedEngineIds failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

} // extern "C"
