// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HeuristicPlugin.hpp"

#include "HipdnnException.hpp"

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

HeuristicPlugin::HeuristicPlugin()
    : _lib()
{
}

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
    if(status == HIPDNN_HEURISTIC_STATUS_NOT_SUPPORTED)
    {
        return ""; // Plugin chose not to provide a name
    }

    if(status != HIPDNN_HEURISTIC_STATUS_SUCCESS)
    {
        throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                              std::string("Heuristic plugin failed to get policy name. Status: ")
                                  + std::to_string(status)
                                  + ", Error: " + std::string(getLastErrorString()));
    }

    return name ? name : "";
}

std::string_view HeuristicPlugin::pluginVersion() const
{
    const char* version = nullptr;
    invokeHeuristicFunction("get plugin version", _funcGetPluginVersion, &version);
    return version;
}

hipdnnHeuristicStatus_t HeuristicPlugin::setLoggingCallback(hipdnnCallback_t callback) const
{
    return _funcSetLoggingCallback(callback);
}

hipdnnHeuristicStatus_t HeuristicPlugin::setLogLevel(hipdnnSeverity_t level) const
{
    if(_funcSetLogLevel == nullptr)
    {
        return HIPDNN_HEURISTIC_STATUS_SUCCESS; // Optional function not implemented
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
                                          const uint8_t* devicePropsSerializedPtr,
                                          size_t devicePropsSerializedSize) const
{
    invokeHeuristicFunction("set device properties",
                            _funcHandleSetDeviceProperties,
                            handle,
                            devicePropsSerializedPtr,
                            devicePropsSerializedSize);
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
                                         const uint8_t* serializedGraphPtr,
                                         size_t serializedGraphSize) const
{
    invokeHeuristicFunction("set serialized graph",
                            _funcPolicySetSerializedGraph,
                            desc,
                            serializedGraphPtr,
                            serializedGraphSize);
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
    // Query the count first
    size_t count = 0;
    invokeHeuristicFunction(
        "get sorted engine IDs count", _funcPolicyGetSortedEngineIds, desc, nullptr, 0, &count);

    if(count == 0)
    {
        return {};
    }

    // Retrieve the actual IDs
    std::vector<int64_t> ids(count);
    size_t actualCount = 0;
    invokeHeuristicFunction("get sorted engine IDs",
                            _funcPolicyGetSortedEngineIds,
                            desc,
                            ids.data(),
                            count,
                            &actualCount);

    ids.resize(actualCount);
    return ids;
}

std::string_view HeuristicPlugin::getLastErrorString() const noexcept
{
    const char* error = nullptr;
    _funcGetLastErrorString(&error);
    return error ? error : "";
}

} // namespace hipdnn_backend::plugin
