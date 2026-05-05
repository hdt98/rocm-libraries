// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file StaticOrderingPlugin.cpp
 * @brief StaticOrdering heuristic plugin - wraps utilities::sortEngineIds
 *
 * This plugin implements the legacy static ordering heuristic that prioritizes
 * MIOPEN_ENGINE and deprioritizes MIOPEN_ENGINE_DETERMINISTIC.
 *
 * RFC 0007 Section 9: This is a well-known policy plugin that provides
 * backward-compatible ordering semantics.
 */

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/EngineOrdering.hpp>
#include <hipdnn_plugin_sdk/HeuristicValidation.hpp>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <vector>

// Plugin version and identification
static constexpr const char* PLUGIN_VERSION = "1.0.0";
static constexpr const char* API_VERSION = "0.0.1"; // Match HIPDNN_HEURISTIC_API_VERSION
static constexpr const char* POLICY_NAME = "SelectionHeuristic::StaticOrdering";

// Logging callback and log level
static hipdnnCallback_t g_loggingCallback = nullptr;
static hipdnnSeverity_t g_logLevel = HIPDNN_SEV_INFO;

// Log buffer size
constexpr size_t LOG_BUFFER_SIZE = 1024;

// Helper macro for logging
#define STATIC_ORDERING_LOG(severity, ...)                                           \
    do                                                                               \
    {                                                                                \
        if(g_loggingCallback != nullptr && (severity) >= g_logLevel)                 \
        {                                                                            \
            std::array<char, LOG_BUFFER_SIZE> buffer;                                \
            snprintf(buffer.data(), buffer.size(), "[StaticOrdering] " __VA_ARGS__); \
            g_loggingCallback(severity, buffer.data());                              \
        }                                                                            \
    } while(0)

//=============================================================================
// Plugin Handle Implementation
//=============================================================================

struct StaticOrderingHandle
{
    // Device properties (stored when SetDeviceProperties is called)
    std::vector<uint8_t> devicePropertiesBuffer;
    bool devicePropertiesSet = false;

    StaticOrderingHandle()
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_INFO, "Handle created");
    }

    ~StaticOrderingHandle()
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_INFO, "Handle destroyed");
    }
};

//=============================================================================
// Policy Descriptor Implementation
//=============================================================================

struct StaticOrderingPolicyDescriptor
{
    StaticOrderingHandle* handle = nullptr;
    std::vector<int64_t> candidateEngineIds;
    std::vector<int64_t> sortedEngineIds;
    bool finalized = false;

    StaticOrderingPolicyDescriptor(StaticOrderingHandle* h)
        : handle(h)
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_INFO, "Policy descriptor created");
    }

    ~StaticOrderingPolicyDescriptor()
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_INFO, "Policy descriptor destroyed");
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
    STATIC_ORDERING_LOG(HIPDNN_SEV_INFO, "Logging callback set");
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginSetLogLevel(hipdnnSeverity_t level)
{
    g_logLevel = level;
    STATIC_ORDERING_LOG(HIPDNN_SEV_INFO, "Log level set to %d", level);
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
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        outHandle, STATIC_ORDERING_LOG, "HandleCreate: null output pointer");

    try
    {
        auto handle = new StaticOrderingHandle();
        *outHandle = reinterpret_cast<hipdnnHeuristicHandle_t>(handle);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "HandleCreate failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicHandleDestroy(hipdnnHeuristicHandle_t handle)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(handle, STATIC_ORDERING_LOG, "HandleDestroy: null handle");

    try
    {
        auto h = reinterpret_cast<StaticOrderingHandle*>(handle);
        delete h;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "HandleDestroy failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicHandleSetDeviceProperties(
    hipdnnHeuristicHandle_t handle, const hipdnnPluginConstData_t* devicePropsSerialized)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        handle, STATIC_ORDERING_LOG, "SetDeviceProperties: null handle");
    HIPDNN_PLUGIN_REQUIRE_CONST_DATA(devicePropsSerialized,
                                     true,
                                     STATIC_ORDERING_LOG,
                                     "SetDeviceProperties: invalid buffer");

    try
    {
        auto h = reinterpret_cast<StaticOrderingHandle*>(handle);

        // Store device properties (StaticOrdering doesn't use them, but we store for completeness)
        const auto* data = reinterpret_cast<const uint8_t*>(devicePropsSerialized->ptr);
        h->devicePropertiesBuffer.assign(data, data + devicePropsSerialized->size);
        h->devicePropertiesSet = true;

        STATIC_ORDERING_LOG(
            HIPDNN_SEV_INFO, "Device properties set (%zu bytes)", devicePropsSerialized->size);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "SetDeviceProperties failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

//=============================================================================
// C ABI Implementation - Policy Descriptor Lifecycle
//=============================================================================

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPolicyDescriptorCreate(
    hipdnnHeuristicHandle_t pluginHandle, hipdnnHeuristicPolicyDescriptor_t* outDesc)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        pluginHandle, STATIC_ORDERING_LOG, "PolicyDescriptorCreate: null handle");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        outDesc, STATIC_ORDERING_LOG, "PolicyDescriptorCreate: null output pointer");

    try
    {
        auto h = reinterpret_cast<StaticOrderingHandle*>(pluginHandle);
        auto desc = new StaticOrderingPolicyDescriptor(h);
        *outDesc = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(desc);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "PolicyDescriptorCreate failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t desc)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        desc, STATIC_ORDERING_LOG, "PolicyDescriptorDestroy: null descriptor");

    try
    {
        auto d = reinterpret_cast<StaticOrderingPolicyDescriptor*>(desc);
        delete d;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "PolicyDescriptorDestroy failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

//=============================================================================
// C ABI Implementation - Policy Inputs
//=============================================================================

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPolicySetEngineIds(
    hipdnnHeuristicPolicyDescriptor_t desc, const int64_t* engineIds, size_t engineIdCount)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, STATIC_ORDERING_LOG, "SetEngineIds: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_ARRAY(engineIds,
                                engineIdCount,
                                STATIC_ORDERING_LOG,
                                "SetEngineIds: null engine_ids with count > 0");

    try
    {
        auto d = reinterpret_cast<StaticOrderingPolicyDescriptor*>(desc);
        d->candidateEngineIds.assign(engineIds, engineIds + engineIdCount);
        d->finalized = false; // Reset finalized state when inputs change

        STATIC_ORDERING_LOG(HIPDNN_SEV_INFO, "SetEngineIds: %zu candidate engines", engineIdCount);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "SetEngineIds failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPolicySetSerializedGraph(
    hipdnnHeuristicPolicyDescriptor_t desc, const hipdnnPluginConstData_t* serializedGraph)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        desc, STATIC_ORDERING_LOG, "SetSerializedGraph: null descriptor");
    // StaticOrdering doesn't use the graph, but validate the parameter
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        serializedGraph, STATIC_ORDERING_LOG, "SetSerializedGraph: null graph pointer");

    STATIC_ORDERING_LOG(HIPDNN_SEV_INFO,
                        "SetSerializedGraph: graph received (%zu bytes) - not used by "
                        "StaticOrdering",
                        serializedGraph->size);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

//=============================================================================
// C ABI Implementation - Selection
//=============================================================================

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyFinalize(hipdnnHeuristicPolicyDescriptor_t desc, int32_t* outApplied)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, STATIC_ORDERING_LOG, "PolicyFinalize: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        outApplied, STATIC_ORDERING_LOG, "PolicyFinalize: null output pointer");

    try
    {
        auto d = reinterpret_cast<StaticOrderingPolicyDescriptor*>(desc);

        if(d->candidateEngineIds.empty())
        {
            STATIC_ORDERING_LOG(HIPDNN_SEV_WARN,
                                "PolicyFinalize: no candidate engines - not applicable");
            *outApplied = 0;
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        // Copy candidates and apply static sorting
        d->sortedEngineIds = d->candidateEngineIds;
        hipdnn_data_sdk::utilities::sortEngineIds(d->sortedEngineIds);

        d->finalized = true;
        *outApplied = 1; // Policy succeeded

        STATIC_ORDERING_LOG(HIPDNN_SEV_INFO,
                            "PolicyFinalize: sorted %zu engines using static ordering",
                            d->sortedEngineIds.size());
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "PolicyFinalize failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPolicyGetSortedEngineIds(
    hipdnnHeuristicPolicyDescriptor_t desc, int64_t* engineIds, size_t* numEngines)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        desc, STATIC_ORDERING_LOG, "GetSortedEngineIds: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        numEngines, STATIC_ORDERING_LOG, "GetSortedEngineIds: null num_engines pointer");

    try
    {
        auto d = reinterpret_cast<StaticOrderingPolicyDescriptor*>(desc);

        if(!d->finalized)
        {
            STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "GetSortedEngineIds: descriptor not finalized");
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

        STATIC_ORDERING_LOG(
            HIPDNN_SEV_INFO, "GetSortedEngineIds: returned %zu engines", *numEngines);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "GetSortedEngineIds failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

} // extern "C"
