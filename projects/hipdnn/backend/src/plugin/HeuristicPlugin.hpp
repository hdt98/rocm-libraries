// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <vector>

#include "PluginCore.hpp"
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>

namespace hipdnn_backend::plugin
{

/**
 * @brief Wrapper for a heuristic plugin shared library.
 *
 * This class provides a C++ interface to the heuristic plugin C ABI defined in
 * HeuristicsPluginApi.h. It manages symbol resolution and provides type-safe
 * wrappers around the C function pointers.
 *
 * Heuristic plugins are separate from engine plugins and do NOT implement PluginApi.h.
 */
class HeuristicPlugin
{
protected:
    // Protected constructor to prevent direct instantiation
    explicit HeuristicPlugin(SharedLibrary&& lib);

    // For mocking in tests
    HeuristicPlugin();

public:
    // Module metadata (called before handle creation)
    virtual std::string_view apiVersion() const;
    virtual int64_t policyId() const;
    virtual std::string_view policyName() const;
    virtual std::string_view pluginVersion() const;

    // Logging setup (called at module load time)
    hipdnnHeuristicStatus_t setLoggingCallback(hipdnnCallback_t callback) const;
    hipdnnHeuristicStatus_t setLogLevel(hipdnnSeverity_t level) const;

    // Plugin handle lifecycle
    virtual hipdnnHeuristicHandle_t createHandle() const;
    virtual void destroyHandle(hipdnnHeuristicHandle_t handle) const;
    virtual void setDeviceProperties(hipdnnHeuristicHandle_t handle,
                                     const uint8_t* devicePropsSerializedPtr,
                                     size_t devicePropsSerializedSize) const;

    // Policy descriptor lifecycle
    virtual hipdnnHeuristicPolicyDescriptor_t
        createPolicyDescriptor(hipdnnHeuristicHandle_t pluginHandle) const;
    virtual void destroyPolicyDescriptor(hipdnnHeuristicPolicyDescriptor_t desc) const;

    // Policy inputs
    virtual void setEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                              const int64_t* engineIds,
                              size_t engineIdCount) const;
    virtual void setSerializedGraph(hipdnnHeuristicPolicyDescriptor_t desc,
                                    const uint8_t* serializedGraphPtr,
                                    size_t serializedGraphSize) const;

    // Selection execution
    virtual bool finalize(hipdnnHeuristicPolicyDescriptor_t desc) const;
    virtual std::vector<int64_t> getSortedEngineIds(hipdnnHeuristicPolicyDescriptor_t desc) const;

protected:
    // Error handling helper (must not throw, used during error handling)
    std::string_view getLastErrorString() const noexcept;

    template <typename Callable, typename... Args>
    void invokeHeuristicFunction(const char* description, Callable&& func, Args&&... args) const
    {
        auto status = func(std::forward<Args>(args)...);
        if(status != HIPDNN_HEURISTIC_STATUS_SUCCESS)
        {
            throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                  std::string("Heuristic plugin failed to ") + description
                                      + ". Status: " + std::to_string(status)
                                      + ", Error: " + std::string(getLastErrorString()));
        }
    }

    template <class F>
    bool tryAssignSymbol(F& functionPtr, const char* symbolName)
    {
        try
        {
            functionPtr = _lib.getSymbol<F>(symbolName);
            return true;
        }
        catch(const HipdnnException&)
        {
            functionPtr = nullptr;
            return false;
        }
    }

    SharedLibrary _lib;

private:
    void resolveSymbols();

#ifndef NDEBUG
    bool _initialized = false;
#endif

    // Cached metadata
    mutable int64_t _policyId = -1;

    // Module metadata function pointers
    hipdnnHeuristicStatus_t (*_funcGetApiVersion)(const char**);
    hipdnnHeuristicStatus_t (*_funcGetPolicyId)(int64_t*);
    hipdnnHeuristicStatus_t (*_funcGetPolicyName)(const char**);
    hipdnnHeuristicStatus_t (*_funcGetPluginVersion)(const char**);
    hipdnnHeuristicStatus_t (*_funcSetLoggingCallback)(hipdnnCallback_t);
    hipdnnHeuristicStatus_t (*_funcSetLogLevel)(hipdnnSeverity_t);
    void (*_funcGetLastErrorString)(const char**);

    // Handle lifecycle function pointers
    hipdnnHeuristicStatus_t (*_funcHandleCreate)(hipdnnHeuristicHandle_t*);
    hipdnnHeuristicStatus_t (*_funcHandleDestroy)(hipdnnHeuristicHandle_t);
    hipdnnHeuristicStatus_t (*_funcHandleSetDeviceProperties)(hipdnnHeuristicHandle_t,
                                                              const uint8_t*,
                                                              size_t);

    // Policy descriptor lifecycle function pointers
    hipdnnHeuristicStatus_t (*_funcPolicyDescriptorCreate)(hipdnnHeuristicHandle_t,
                                                           hipdnnHeuristicPolicyDescriptor_t*);
    hipdnnHeuristicStatus_t (*_funcPolicyDescriptorDestroy)(hipdnnHeuristicPolicyDescriptor_t);

    // Policy input function pointers
    hipdnnHeuristicStatus_t (*_funcPolicySetEngineIds)(hipdnnHeuristicPolicyDescriptor_t,
                                                       const int64_t*,
                                                       size_t);
    hipdnnHeuristicStatus_t (*_funcPolicySetSerializedGraph)(hipdnnHeuristicPolicyDescriptor_t,
                                                             const uint8_t*,
                                                             size_t);

    // Selection function pointers
    hipdnnHeuristicStatus_t (*_funcPolicyFinalize)(hipdnnHeuristicPolicyDescriptor_t, int32_t*);
    hipdnnHeuristicStatus_t (*_funcPolicyGetSortedEngineIds)(hipdnnHeuristicPolicyDescriptor_t,
                                                             int64_t*,
                                                             size_t,
                                                             size_t*);

    friend class PluginManagerBase<HeuristicPlugin>;
};

} // namespace hipdnn_backend::plugin
