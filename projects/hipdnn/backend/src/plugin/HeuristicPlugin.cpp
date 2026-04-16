// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HeuristicPlugin.hpp"

#include "HipdnnException.hpp"
#include "logging/Logging.hpp"

namespace hipdnn_backend::plugin
{

HeuristicPlugin::HeuristicPlugin(SharedLibrary&& lib)
    : _lib(std::move(lib))
{
    resolveSymbols();
#ifndef NDEBUG
    _initialized = true;
#endif
}

HeuristicPlugin::HeuristicPlugin() = default;

void HeuristicPlugin::resolveSymbols()
{
    // Required module metadata symbols
    _funcGetApiVersion = _lib.getSymbol<decltype(_funcGetApiVersion)>("hipdnnHeuristicGetApiVersion");
    _funcGetPolicyId   = _lib.getSymbol<decltype(_funcGetPolicyId)>("hipdnnHeuristicGetPolicyId");
    _funcGetPluginVersion
        = _lib.getSymbol<decltype(_funcGetPluginVersion)>("hipdnnHeuristicGetPluginVersion");
    _funcSetLoggingCallback
        = _lib.getSymbol<decltype(_funcSetLoggingCallback)>("hipdnnHeuristicSetLoggingCallback");
    _funcGetLastErrorString
        = _lib.getSymbol<decltype(_funcGetLastErrorString)>("hipdnnHeuristicGetLastErrorString");

    // Optional module metadata symbols
    tryAssignSymbol(_funcGetPolicyName, "hipdnnHeuristicGetPolicyName");
    tryAssignSymbol(_funcSetLogLevel, "hipdnnHeuristicSetLogLevel");

    // Required handle lifecycle symbols
    _funcHandleCreate = _lib.getSymbol<decltype(_funcHandleCreate)>("hipdnnHeuristicHandleCreate");
    _funcHandleDestroy
        = _lib.getSymbol<decltype(_funcHandleDestroy)>("hipdnnHeuristicHandleDestroy");
    _funcHandleSetDeviceProperties = _lib.getSymbol<decltype(_funcHandleSetDeviceProperties)>(
        "hipdnnHeuristicHandleSetDeviceProperties");

    // Required policy descriptor lifecycle symbols
    _funcPolicyDescriptorCreate = _lib.getSymbol<decltype(_funcPolicyDescriptorCreate)>(
        "hipdnnHeuristicPolicyDescriptorCreate");
    _funcPolicyDescriptorDestroy = _lib.getSymbol<decltype(_funcPolicyDescriptorDestroy)>(
        "hipdnnHeuristicPolicyDescriptorDestroy");

    // Required policy input symbols
    _funcPolicySetEngineIds
        = _lib.getSymbol<decltype(_funcPolicySetEngineIds)>("hipdnnHeuristicPolicySetEngineIds");
    _funcPolicySetSerializedGraph = _lib.getSymbol<decltype(_funcPolicySetSerializedGraph)>(
        "hipdnnHeuristicPolicySetSerializedGraph");

    // Required selection symbols
    _funcPolicyFinalize
        = _lib.getSymbol<decltype(_funcPolicyFinalize)>("hipdnnHeuristicPolicyFinalize");
    _funcPolicyGetSortedEngineIds = _lib.getSymbol<decltype(_funcPolicyGetSortedEngineIds)>(
        "hipdnnHeuristicPolicyGetSortedEngineIds");
}

std::string_view HeuristicPlugin::apiVersion() const
{
    const char* version = nullptr;
    invokeHeuristicFunction("get API version", _funcGetApiVersion, &version);
    return version;
}

int64_t HeuristicPlugin::policyId() const
{
    if(_policyId != -1)
    {
        return _policyId;
    }

    int64_t id = -1;
    invokeHeuristicFunction("get policy ID", _funcGetPolicyId, &id);
    _policyId = id;
    return _policyId;
}

std::string_view HeuristicPlugin::policyName() const
{
    if(_funcGetPolicyName == nullptr)
    {
        return ""; // Optional function not implemented
    }

    const char* name = nullptr;
    auto status      = _funcGetPolicyName(&name);
    if(status == HIPDNN_PLUGIN_STATUS_INVALID_VALUE)
    {
        return ""; // Plugin chose not to provide a name
    }

    if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
    {
        throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                              std::string("Heuristic plugin failed to get policy name. Status: ")
                                  + std::to_string(status)
                                  + ", Error: " + std::string(getLastErrorString()));
    }

    return (name != nullptr) ? name : "";
}

std::string_view HeuristicPlugin::pluginVersion() const
{
    const char* version = nullptr;
    invokeHeuristicFunction("get plugin version", _funcGetPluginVersion, &version);
    return version;
}

hipdnnPluginStatus_t HeuristicPlugin::setLoggingCallback(hipdnnHeuristicLoggingCallback_t callback) const
{
    return _funcSetLoggingCallback(callback);
}

// 2-parameter callback overload for PluginManagerBase compatibility
// Uses the existing heuristicLoggingCallback which wraps 2-param to 3-param
hipdnnPluginStatus_t HeuristicPlugin::setLoggingCallback(hipdnnCallback_t /*callback*/) const
{
    // The heuristicLoggingCallback function in logging/Logging.cpp already
    // wraps the backend's 2-param callback to the plugin's 3-param interface
    // by combining componentPrefix and message before calling backendLoggingCallback
    // We ignore the passed callback and use the global heuristicLoggingCallback
    return setLoggingCallback(hipdnn_backend::logging::heuristicLoggingCallback);
}

hipdnnPluginStatus_t HeuristicPlugin::setLogLevel(hipdnnSeverity_t level) const
{
    if(_funcSetLogLevel == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_SUCCESS; // Optional function not implemented
    }
    return _funcSetLogLevel(level);
}

hipdnnHeuristicHandle_t HeuristicPlugin::createHandle() const
{
    hipdnnHeuristicHandle_t handle = nullptr;
    invokeHeuristicFunction("create handle", _funcHandleCreate, &handle);
    return handle;
}

void HeuristicPlugin::destroyHandle(hipdnnHeuristicHandle_t handle) const
{
    invokeHeuristicFunction("destroy handle", _funcHandleDestroy, handle);
}

void HeuristicPlugin::setDeviceProperties(hipdnnHeuristicHandle_t handle,
                                          const hipdnnPluginConstData_t* devicePropsSerialized) const
{
    invokeHeuristicFunction(
        "set device properties", _funcHandleSetDeviceProperties, handle, devicePropsSerialized);
}

hipdnnHeuristicPolicyDescriptor_t
    HeuristicPlugin::createPolicyDescriptor(hipdnnHeuristicHandle_t pluginHandle) const
{
    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    invokeHeuristicFunction(
        "create policy descriptor", _funcPolicyDescriptorCreate, pluginHandle, &desc);
    return desc;
}

void HeuristicPlugin::destroyPolicyDescriptor(hipdnnHeuristicPolicyDescriptor_t desc) const
{
    invokeHeuristicFunction("destroy policy descriptor", _funcPolicyDescriptorDestroy, desc);
}

void HeuristicPlugin::setEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                   const int64_t* engineIds,
                                   size_t engineIdCount) const
{
    invokeHeuristicFunction(
        "set engine IDs", _funcPolicySetEngineIds, desc, engineIds, engineIdCount);
}

void HeuristicPlugin::setSerializedGraph(hipdnnHeuristicPolicyDescriptor_t desc,
                                         const hipdnnPluginConstData_t* serializedGraph) const
{
    invokeHeuristicFunction(
        "set serialized graph", _funcPolicySetSerializedGraph, desc, serializedGraph);
}

bool HeuristicPlugin::finalize(hipdnnHeuristicPolicyDescriptor_t desc) const
{
    int32_t applied = 0;
    invokeHeuristicFunction("finalize policy", _funcPolicyFinalize, desc, &applied);
    return applied != 0;
}

std::vector<int64_t>
    HeuristicPlugin::getSortedEngineIds(hipdnnHeuristicPolicyDescriptor_t desc) const
{
    // Query the count first (pass nullptr for engine_ids)
    uint32_t count = 0;
    invokeHeuristicFunction(
        "get sorted engine IDs count", _funcPolicyGetSortedEngineIds, desc, nullptr, &count);

    if(count == 0)
    {
        return {};
    }

    // Retrieve the actual IDs
    std::vector<int64_t> ids(count);
    uint32_t actualCount = count;
    invokeHeuristicFunction("get sorted engine IDs",
                            _funcPolicyGetSortedEngineIds,
                            desc,
                            ids.data(),
                            &actualCount);

    ids.resize(actualCount);
    return ids;
}

std::string_view HeuristicPlugin::getLastErrorString() const noexcept
{
    const char* error = nullptr;
    _funcGetLastErrorString(&error);
    return (error != nullptr) ? error : "";
}

} // namespace hipdnn_backend::plugin
