// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file DefaultHeuristicsPlugin.cpp
 * @brief Default heuristic plugin bundling the well-known Config and
 *        StaticOrdering policies into a single shared library.
 *
 * Two policies live behind one plugin handle, exercising the 1:N model of the
 * heuristic plugin C ABI:
 *
 *  - SelectionHeuristic::Config — honors graph-level preferences
 *      (Graph.preferred_engine_id) and the HIPDNN_ENGINE_OVERRIDE_FILE JSON.
 *      If a preferred engine is resolved and present in the candidate list,
 *      reorders so it comes first; otherwise declines.
 *
 *  - SelectionHeuristic::StaticOrdering — wraps utilities::sortEngineIds, the
 *      legacy MIOPEN_ENGINE-first / MIOPEN_ENGINE_DETERMINISTIC-last fallback.
 *
 * The C ABI dispatches on policyId at descriptor creation; from then on the
 * descriptor's polymorphic methods carry the per-policy logic.
 */

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/EngineOrdering.hpp>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/HeuristicValidation.hpp>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "EngineOverrideConfig.hpp"

// Plugin identification
static constexpr const char* PLUGIN_VERSION = "1.0.0";
static constexpr const char* API_VERSION = "0.0.1"; // Match HIPDNN_HEURISTIC_API_VERSION
static constexpr const char* PLUGIN_NAME = "DefaultHeuristicsPlugin";
static constexpr const char* CONFIG_POLICY_NAME = "SelectionHeuristic::Config";
static constexpr const char* STATIC_ORDERING_POLICY_NAME = "SelectionHeuristic::StaticOrdering";

// Logging callback and log level
static hipdnnCallback_t g_loggingCallback = nullptr;
static hipdnnSeverity_t g_logLevel = HIPDNN_SEV_INFO;

// Log buffer size
constexpr size_t LOG_BUFFER_SIZE = 1024;

// Helper macro for logging
#define DEFAULT_HEURISTICS_LOG(severity, ...)                                           \
    do                                                                                  \
    {                                                                                   \
        if(g_loggingCallback != nullptr && (severity) >= g_logLevel)                    \
        {                                                                               \
            std::array<char, LOG_BUFFER_SIZE> buffer;                                   \
            snprintf(buffer.data(), buffer.size(), "[DefaultHeuristics] " __VA_ARGS__); \
            g_loggingCallback(severity, buffer.data());                                 \
        }                                                                               \
    } while(0)

namespace
{

int64_t configPolicyId()
{
    static const int64_t s_id = hipdnn_data_sdk::utilities::policyNameToId(CONFIG_POLICY_NAME);
    return s_id;
}

int64_t staticOrderingPolicyId()
{
    static const int64_t s_id
        = hipdnn_data_sdk::utilities::policyNameToId(STATIC_ORDERING_POLICY_NAME);
    return s_id;
}

} // namespace

//=============================================================================
// Plugin Handle Implementation
//=============================================================================

struct DefaultHeuristicsHandle
{
    std::vector<uint8_t> devicePropertiesBuffer;
    bool devicePropertiesSet = false;

    DefaultHeuristicsHandle()
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_INFO, "Handle created");
    }

    ~DefaultHeuristicsHandle()
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_INFO, "Handle destroyed");
    }
};

//=============================================================================
// Policy Descriptor Polymorphism
//=============================================================================

namespace
{

/// Thrown by a policy's finalize() when caller-supplied data fails validation
/// (e.g. malformed serialized graph). The C ABI catches this and maps it to
/// HIPDNN_PLUGIN_STATUS_BAD_PARAM rather than INTERNAL_ERROR.
class BadParamError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/// Common interface that the C ABI dispatches through.
struct PolicyDescriptor
{
    DefaultHeuristicsHandle* handle = nullptr;
    std::vector<int64_t> candidateEngineIds;
    std::vector<int64_t> sortedEngineIds;
    bool finalized = false;

    explicit PolicyDescriptor(DefaultHeuristicsHandle* h)
        : handle(h)
    {
    }

    virtual ~PolicyDescriptor() = default;
    PolicyDescriptor(const PolicyDescriptor&) = delete;
    PolicyDescriptor& operator=(const PolicyDescriptor&) = delete;
    PolicyDescriptor(PolicyDescriptor&&) = delete;
    PolicyDescriptor& operator=(PolicyDescriptor&&) = delete;

    /// Override default behavior of ignoring serialized graph (StaticOrdering).
    virtual void onSetSerializedGraph(const uint8_t* /*data*/, size_t /*size*/) {}

    /// Run the policy. Sets sortedEngineIds + finalized when applied. Returns
    /// 1 if applied, 0 if not applicable. Throws on hard failure.
    virtual int finalize() = 0;
};

struct StaticOrderingDescriptor : PolicyDescriptor
{
    using PolicyDescriptor::PolicyDescriptor;

    int finalize() override
    {
        if(candidateEngineIds.empty())
        {
            DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_WARN,
                                   "StaticOrdering: no candidate engines - not applicable");
            return 0;
        }

        sortedEngineIds = candidateEngineIds;
        hipdnn_data_sdk::utilities::sortEngineIds(sortedEngineIds);
        finalized = true;
        DEFAULT_HEURISTICS_LOG(
            HIPDNN_SEV_INFO, "StaticOrdering: sorted %zu engines", sortedEngineIds.size());
        return 1;
    }
};

struct ConfigDescriptor : PolicyDescriptor
{
    std::vector<uint8_t> serializedGraphBuffer;

    using PolicyDescriptor::PolicyDescriptor;

    void onSetSerializedGraph(const uint8_t* data, size_t size) override
    {
        serializedGraphBuffer.assign(data, data + size);
    }

    int finalize() override;
};

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

/// Index every (uid, dims, strides) triple from the FlatBuffer graph for O(1)
/// lookup during the conv scan. Owns the dim/stride vectors.
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

int ConfigDescriptor::finalize()
{
    if(candidateEngineIds.empty())
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_WARN, "Config: no candidate engines - not applicable");
        return 0;
    }
    if(serializedGraphBuffer.empty())
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_WARN, "Config: no serialized graph - not applicable");
        return 0;
    }

    flatbuffers::Verifier verifier(serializedGraphBuffer.data(), serializedGraphBuffer.size());
    if(!verifier.VerifyBuffer<Graph>())
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_ERROR, "Config: invalid graph buffer");
        throw BadParamError("invalid serialized graph");
    }

    const auto* graph
        = hipdnn_flatbuffers_sdk::data_objects::GetGraph(serializedGraphBuffer.data());
    if(graph == nullptr)
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_ERROR, "Config: failed to get graph root");
        throw BadParamError("graph root null");
    }

    // Resolve a preferred engine ID, in priority order:
    //   1. Explicit graph.preferred_engine_id (set via the frontend setter).
    //   2. HIPDNN_ENGINE_OVERRIDE_FILE rule that matches a conv node.
    std::optional<int64_t> preferredEngineId;
    const char* preferredSource = nullptr;
    if(auto preferredEngineIdOpt = graph->preferred_engine_id(); preferredEngineIdOpt.has_value())
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
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_INFO,
                               "Config: no preferred engine resolved - not applicable");
        return 0;
    }

    auto it = std::find(candidateEngineIds.begin(), candidateEngineIds.end(), *preferredEngineId);
    if(it == candidateEngineIds.end())
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_INFO,
                               "Config: preferred engine 0x%llx (from %s) not in candidates - "
                               "not applicable",
                               static_cast<long long>(*preferredEngineId),
                               preferredSource);
        return 0;
    }

    sortedEngineIds.clear();
    sortedEngineIds.reserve(candidateEngineIds.size());
    sortedEngineIds.push_back(*preferredEngineId);
    for(const int64_t engineId : candidateEngineIds)
    {
        if(engineId != *preferredEngineId)
        {
            sortedEngineIds.push_back(engineId);
        }
    }
    finalized = true;

    DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_INFO,
                           "Config: reordered %zu engines with preferred 0x%llx (from %s) first",
                           sortedEngineIds.size(),
                           static_cast<long long>(*preferredEngineId),
                           preferredSource);
    return 1;
}

} // namespace

//=============================================================================
// C ABI Implementation
//=============================================================================

extern "C" {

// ---- Base plugin metadata --------------------------------------------------

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    if(name == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *name = PLUGIN_NAME;
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

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)
{
    g_loggingCallback = callback;
    DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_INFO, "Logging callback set");
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginSetLogLevel(hipdnnSeverity_t level)
{
    g_logLevel = level;
    DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_INFO, "Log level set to %d", level);
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

// ---- Policy enumeration ----------------------------------------------------

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPluginGetAllPolicyIds(
    int64_t* policyIds, uint32_t maxPolicies, uint32_t* numPolicies)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        numPolicies, DEFAULT_HEURISTICS_LOG, "GetAllPolicyIds: null num_policies");

    constexpr uint32_t TOTAL_POLICIES = 2;
    *numPolicies = TOTAL_POLICIES;
    if(policyIds == nullptr || maxPolicies == 0)
    {
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    if(maxPolicies < TOTAL_POLICIES)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    // Order matters: Config runs first, then StaticOrdering as fallback.
    policyIds[0] = configPolicyId();
    policyIds[1] = staticOrderingPolicyId();
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPluginGetPolicyName(int64_t policyId, const char** name)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        name, DEFAULT_HEURISTICS_LOG, "GetPolicyName: null output pointer");

    if(policyId == configPolicyId())
    {
        *name = CONFIG_POLICY_NAME;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    if(policyId == staticOrderingPolicyId())
    {
        *name = STATIC_ORDERING_POLICY_NAME;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_ERROR, "GetPolicyName: unknown policy ID");
    return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
}

// ---- Handle lifecycle ------------------------------------------------------

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicHandleCreate(hipdnnHeuristicHandle_t* outHandle)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        outHandle, DEFAULT_HEURISTICS_LOG, "HandleCreate: null output pointer");

    try
    {
        auto handle = new DefaultHeuristicsHandle();
        *outHandle = reinterpret_cast<hipdnnHeuristicHandle_t>(handle);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_ERROR, "HandleCreate failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicHandleDestroy(hipdnnHeuristicHandle_t handle)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(handle, DEFAULT_HEURISTICS_LOG, "HandleDestroy: null handle");

    try
    {
        auto h = reinterpret_cast<DefaultHeuristicsHandle*>(handle);
        delete h;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_ERROR, "HandleDestroy failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicHandleSetDeviceProperties(
    hipdnnHeuristicHandle_t handle, const hipdnnPluginConstData_t* devicePropsSerialized)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        handle, DEFAULT_HEURISTICS_LOG, "SetDeviceProperties: null handle");
    HIPDNN_PLUGIN_REQUIRE_CONST_DATA(
        devicePropsSerialized, true, DEFAULT_HEURISTICS_LOG, "SetDeviceProperties: invalid buffer");

    try
    {
        auto h = reinterpret_cast<DefaultHeuristicsHandle*>(handle);
        const auto* data = reinterpret_cast<const uint8_t*>(devicePropsSerialized->ptr);
        h->devicePropertiesBuffer.assign(data, data + devicePropsSerialized->size);
        h->devicePropertiesSet = true;

        DEFAULT_HEURISTICS_LOG(
            HIPDNN_SEV_INFO, "Device properties set (%zu bytes)", devicePropsSerialized->size);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_ERROR, "SetDeviceProperties failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

// ---- Policy descriptor lifecycle -------------------------------------------

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyDescriptorCreate(hipdnnHeuristicHandle_t pluginHandle,
                                          int64_t policyId,
                                          hipdnnHeuristicPolicyDescriptor_t* outDesc)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        pluginHandle, DEFAULT_HEURISTICS_LOG, "PolicyDescriptorCreate: null handle");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        outDesc, DEFAULT_HEURISTICS_LOG, "PolicyDescriptorCreate: null output pointer");

    try
    {
        auto h = reinterpret_cast<DefaultHeuristicsHandle*>(pluginHandle);
        std::unique_ptr<PolicyDescriptor> desc;
        if(policyId == configPolicyId())
        {
            desc = std::make_unique<ConfigDescriptor>(h);
        }
        else if(policyId == staticOrderingPolicyId())
        {
            desc = std::make_unique<StaticOrderingDescriptor>(h);
        }
        else
        {
            DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_ERROR, "PolicyDescriptorCreate: unknown policy ID");
            return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
        }
        *outDesc = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(desc.release());
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_ERROR, "PolicyDescriptorCreate failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t desc)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        desc, DEFAULT_HEURISTICS_LOG, "PolicyDescriptorDestroy: null descriptor");

    try
    {
        auto d = reinterpret_cast<PolicyDescriptor*>(desc);
        delete d;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_ERROR, "PolicyDescriptorDestroy failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

// ---- Policy inputs ---------------------------------------------------------

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPolicySetEngineIds(
    hipdnnHeuristicPolicyDescriptor_t desc, const int64_t* engineIds, size_t engineIdCount)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, DEFAULT_HEURISTICS_LOG, "SetEngineIds: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_ARRAY(engineIds,
                                engineIdCount,
                                DEFAULT_HEURISTICS_LOG,
                                "SetEngineIds: null engine_ids with count > 0");

    try
    {
        auto d = reinterpret_cast<PolicyDescriptor*>(desc);
        d->candidateEngineIds.assign(engineIds, engineIds + engineIdCount);
        d->finalized = false;
        DEFAULT_HEURISTICS_LOG(
            HIPDNN_SEV_INFO, "SetEngineIds: %zu candidate engines", engineIdCount);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_ERROR, "SetEngineIds failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPolicySetSerializedGraph(
    hipdnnHeuristicPolicyDescriptor_t desc, const hipdnnPluginConstData_t* serializedGraph)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        desc, DEFAULT_HEURISTICS_LOG, "SetSerializedGraph: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_CONST_DATA(
        serializedGraph, false, DEFAULT_HEURISTICS_LOG, "SetSerializedGraph: invalid graph buffer");

    try
    {
        auto d = reinterpret_cast<PolicyDescriptor*>(desc);
        const auto* data = reinterpret_cast<const uint8_t*>(serializedGraph->ptr);
        d->onSetSerializedGraph(data, serializedGraph->size);
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_INFO,
                               "SetSerializedGraph: graph received (%zu bytes)",
                               serializedGraph->size);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_ERROR, "SetSerializedGraph failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

// ---- Selection -------------------------------------------------------------

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyFinalize(hipdnnHeuristicPolicyDescriptor_t desc, int32_t* outApplied)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, DEFAULT_HEURISTICS_LOG, "PolicyFinalize: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        outApplied, DEFAULT_HEURISTICS_LOG, "PolicyFinalize: null output pointer");

    try
    {
        auto d = reinterpret_cast<PolicyDescriptor*>(desc);
        *outApplied = d->finalize();
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const BadParamError& e)
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_ERROR, "PolicyFinalize bad param: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    catch(const std::exception& e)
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_ERROR, "PolicyFinalize failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPolicyGetSortedEngineIds(
    hipdnnHeuristicPolicyDescriptor_t desc, int64_t* engineIds, size_t* numEngines)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        desc, DEFAULT_HEURISTICS_LOG, "GetSortedEngineIds: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        numEngines, DEFAULT_HEURISTICS_LOG, "GetSortedEngineIds: null num_engines pointer");

    try
    {
        auto d = reinterpret_cast<PolicyDescriptor*>(desc);
        if(!d->finalized)
        {
            DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_ERROR,
                                   "GetSortedEngineIds: descriptor not finalized");
            return HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED;
        }
        if(engineIds == nullptr)
        {
            *numEngines = d->sortedEngineIds.size();
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }
        *numEngines = std::min(*numEngines, d->sortedEngineIds.size());
        std::copy_n(d->sortedEngineIds.begin(), *numEngines, engineIds);
        DEFAULT_HEURISTICS_LOG(
            HIPDNN_SEV_INFO, "GetSortedEngineIds: returned %zu engines", *numEngines);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        DEFAULT_HEURISTICS_LOG(HIPDNN_SEV_ERROR, "GetSortedEngineIds failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

} // extern "C"
