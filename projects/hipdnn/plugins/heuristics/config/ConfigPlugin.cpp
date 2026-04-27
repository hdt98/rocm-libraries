// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file ConfigPlugin.cpp
 * @brief Config heuristic plugin - honors graph-level preferences
 *
 * This plugin implements configuration-based engine selection by interpreting
 * graph-level preferences such as preferred_engine_id. It allows users to
 * directly specify engine ordering preferences in their operation graphs.
 *
 * RFC 0007 Section 13.3: Graph-level preferences are interpreted by this policy,
 * not by hard-coded backend logic.
 */

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <vector>

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
    if(outHandle == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "HandleCreate: null output pointer");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

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
    if(handle == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "HandleDestroy: null handle");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

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
    if(handle == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetDeviceProperties: null handle");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    if(devicePropsSerialized == nullptr || devicePropsSerialized->ptr == nullptr
       || devicePropsSerialized->size == 0)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetDeviceProperties: invalid buffer");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

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
    if(pluginHandle == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyDescriptorCreate: null handle");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    if(outDesc == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyDescriptorCreate: null output pointer");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

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
    if(desc == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyDescriptorDestroy: null descriptor");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

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
    if(desc == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetEngineIds: null descriptor");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    if(engineIds == nullptr && engineIdCount > 0)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetEngineIds: null engine_ids with count > 0");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

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
    if(desc == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetSerializedGraph: null descriptor");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    if(serializedGraph == nullptr || serializedGraph->ptr == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetSerializedGraph: null graph pointer");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

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

//=============================================================================
// C ABI Implementation - Selection
//=============================================================================

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyFinalize(hipdnnHeuristicPolicyDescriptor_t desc, int32_t* outApplied)
{
    if(desc == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyFinalize: null descriptor");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    if(outApplied == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyFinalize: null output pointer");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

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

        auto graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(d->serializedGraphBuffer.data());
        if(graph == nullptr)
        {
            CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyFinalize: failed to get graph root");
            return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
        }

        // Check if graph has a preferred_engine_id
        // In FlatBuffers, optional int64 fields use flatbuffers::Optional
        // If null, the field was not set
        auto preferredEngineIdOpt = graph->preferred_engine_id();

        if(!preferredEngineIdOpt.has_value())
        {
            // No preference set - policy not applicable
            CONFIG_LOG(HIPDNN_SEV_INFO,
                       "PolicyFinalize: no preferred_engine_id set - not applicable");
            *outApplied = 0;
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        const int64_t preferredEngineId = preferredEngineIdOpt.value();

        // Check if the preferred engine ID is in the candidates
        auto it = std::find(
            d->candidateEngineIds.begin(), d->candidateEngineIds.end(), preferredEngineId);

        if(it == d->candidateEngineIds.end())
        {
            // Preferred engine not in candidates - policy not applicable
            CONFIG_LOG(
                HIPDNN_SEV_INFO,
                "PolicyFinalize: preferred_engine_id 0x%llx not in candidates - not applicable",
                static_cast<long long>(preferredEngineId));
            *outApplied = 0;
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        // Reorder candidates: preferred engine first, others maintain relative order
        d->sortedEngineIds.clear();
        d->sortedEngineIds.reserve(d->candidateEngineIds.size());

        // Add preferred engine first
        d->sortedEngineIds.push_back(preferredEngineId);

        // Add remaining engines in original order
        for(const int64_t engineId : d->candidateEngineIds)
        {
            if(engineId != preferredEngineId)
            {
                d->sortedEngineIds.push_back(engineId);
            }
        }

        d->finalized = true;
        *outApplied = 1; // Policy succeeded

        CONFIG_LOG(HIPDNN_SEV_INFO,
                   "PolicyFinalize: reordered %zu engines with preferred engine 0x%llx first",
                   d->sortedEngineIds.size(),
                   static_cast<long long>(preferredEngineId));
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
    if(desc == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "GetSortedEngineIds: null descriptor");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    if(numEngines == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "GetSortedEngineIds: null num_engines pointer");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

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
