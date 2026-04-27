// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>
#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>
#include <hipdnn_plugin_sdk/heuristic_api_version.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace
{
// Simple handle implementation for test plugins
struct HeuristicHandleImpl
{
    int handleId;
    bool devicePropsSet{false};
};

// Simple policy descriptor implementation for test plugins
struct PolicyDescriptorImpl
{
    std::vector<int64_t> inputEngineIds;
    std::vector<uint8_t> serializedGraph;
    std::vector<int64_t> sortedEngineIds;
    bool finalized{false};
};

// Callback state
// NOLINTBEGIN(readability-identifier-naming)
hipdnnCallback_t g_loggingCallback = nullptr;
hipdnnSeverity_t g_logLevel = HIPDNN_SEV_INFO;
// NOLINTEND(readability-identifier-naming)

} // anonymous namespace

/**
 * @brief Base class for test heuristic plugins with common implementations
 *
 * Derived classes only need to override:
 * - getPolicyName() - return the policy name
 * - getApiVersion() - (optional) override to return wrong version for testing
 * - getPluginVersion() - (optional) override to customize version
 */
class TestHeuristicPluginBase
{
public:
    virtual ~TestHeuristicPluginBase() = default;

    // Virtual methods to be overridden by derived classes
    virtual const char* getPolicyName() const = 0;

    virtual const char* getPluginVersion() const
    {
        return "1.0.0";
    }

    virtual const char* getApiVersion() const
    {
        return HIPDNN_HEURISTIC_API_VERSION;
    }

    // Static instance management using Meyer's singleton pattern to avoid ODR issues
    static std::unique_ptr<TestHeuristicPluginBase>& getInstanceStorage()
    {
        static std::unique_ptr<TestHeuristicPluginBase> sInstance = nullptr; // NOLINT(readability-identifier-naming)
        return sInstance;
    }

    static void setInstance(std::unique_ptr<TestHeuristicPluginBase> instance)
    {
        getInstanceStorage() = std::move(instance);
    }

    static TestHeuristicPluginBase* getInstance()
    {
        return getInstanceStorage().get();
    }

    // ========== Common API Implementations ==========

    static hipdnnPluginStatus_t getName(const char** name)
    {
        if(name == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "name pointer is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }
        if(getInstance() == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "plugin instance is null");
            return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
        }
        *name = getInstance()->getPolicyName();
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    static hipdnnPluginStatus_t getVersion(const char** version)
    {
        if(version == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "version pointer is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }
        if(getInstance() == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "plugin instance is null");
            return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
        }
        *version = getInstance()->getPluginVersion();
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    static hipdnnPluginStatus_t getApiVersion(const char** version)
    {
        if(version == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "version pointer is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }
        if(getInstance() == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "plugin instance is null");
            return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
        }
        *version = getInstance()->getApiVersion();
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    static hipdnnPluginStatus_t getType(hipdnnPluginType_t* type)
    {
        if(type == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "type pointer is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }
        *type = HIPDNN_PLUGIN_TYPE_HEURISTIC;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    static hipdnnPluginStatus_t setLoggingCallback(hipdnnCallback_t callback)
    {
        g_loggingCallback = callback;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    static hipdnnPluginStatus_t setLogLevel(hipdnnSeverity_t level)
    {
        g_logLevel = level;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    static void getLastErrorString(const char** errorStr)
    {
        if(errorStr != nullptr)
        {
            *errorStr = hipdnn_plugin_sdk::PluginLastErrorManager::getLastError();
        }
    }

    static hipdnnPluginStatus_t handleCreate(hipdnnHeuristicHandle_t* outHandle)
    {
        if(outHandle == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "out_handle pointer is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }

        auto* handle = new HeuristicHandleImpl{42, false};
        *outHandle = reinterpret_cast<hipdnnHeuristicHandle_t>(handle);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    static hipdnnPluginStatus_t handleDestroy(hipdnnHeuristicHandle_t handle)
    {
        if(handle == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "handle is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }

        auto* impl = reinterpret_cast<HeuristicHandleImpl*>(handle);
        delete impl;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    static hipdnnPluginStatus_t
        handleSetDeviceProperties(hipdnnHeuristicHandle_t handle,
                                   const hipdnnPluginConstData_t* devicePropsSerialized)
    {
        if(handle == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "handle is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }
        if(devicePropsSerialized == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "devicePropsSerialized pointer is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }

        auto* impl = reinterpret_cast<HeuristicHandleImpl*>(handle);
        impl->devicePropsSet = true;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    static hipdnnPluginStatus_t
        policyDescriptorCreate(hipdnnHeuristicHandle_t handle,
                               hipdnnHeuristicPolicyDescriptor_t* outDescriptor)
    {
        if(handle == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "handle is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }
        if(outDescriptor == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "out_descriptor pointer is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }

        auto* desc = new PolicyDescriptorImpl{};
        *outDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(desc);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    static hipdnnPluginStatus_t
        policyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t descriptor)
    {
        if(descriptor == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "descriptor is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }

        auto* impl = reinterpret_cast<PolicyDescriptorImpl*>(descriptor);
        delete impl;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    static hipdnnPluginStatus_t policySetEngineIds(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                                    const int64_t* engineIds,
                                                    size_t engineIdCount)
    {
        if(descriptor == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "descriptor is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }
        if(engineIds == nullptr && engineIdCount > 0)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "engine_ids is null but count > 0");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }

        auto* impl = reinterpret_cast<PolicyDescriptorImpl*>(descriptor);
        impl->inputEngineIds.assign(engineIds, engineIds + engineIdCount);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    static hipdnnPluginStatus_t
        policySetSerializedGraph(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                 const hipdnnPluginConstData_t* serializedGraph)
    {
        if(descriptor == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "descriptor is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }
        if(serializedGraph == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "serialized_graph pointer is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }

        auto* impl = reinterpret_cast<PolicyDescriptorImpl*>(descriptor);
        const auto* bytes = static_cast<const uint8_t*>(serializedGraph->ptr);
        impl->serializedGraph.assign(bytes, bytes + serializedGraph->size);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    static hipdnnPluginStatus_t policyFinalize(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                                int32_t* applied)
    {
        if(descriptor == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "descriptor is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }
        if(applied == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "applied pointer is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }

        auto* impl = reinterpret_cast<PolicyDescriptorImpl*>(descriptor);

        // Simple policy: reverse the input order
        impl->sortedEngineIds = impl->inputEngineIds;
        std::reverse(impl->sortedEngineIds.begin(), impl->sortedEngineIds.end());

        impl->finalized = true;
        *applied = 1; // Policy applied
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    static hipdnnPluginStatus_t policyGetSortedEngineIds(
        hipdnnHeuristicPolicyDescriptor_t descriptor, int64_t* engineIds, size_t* count)
    {
        if(descriptor == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "descriptor is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }
        if(count == nullptr)
        {
            hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "count pointer is null");
            return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        }

        auto* impl = reinterpret_cast<PolicyDescriptorImpl*>(descriptor);

        if(engineIds == nullptr)
        {
            // Query mode: return count only
            *count = impl->sortedEngineIds.size();
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        // Retrieve mode: copy IDs
        const size_t numToCopy = std::min(*count, impl->sortedEngineIds.size());
        std::memcpy(engineIds, impl->sortedEngineIds.data(), numToCopy * sizeof(int64_t));
        *count = numToCopy;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

};

// Macro to register all heuristic plugin API functions
// NOLINTBEGIN(readability-identifier-naming,cppcoreguidelines-macro-usage)
#define REGISTER_HEURISTIC_PLUGIN_API()                                                           \
    extern "C" {                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)                                   \
    {                                                                                             \
        return TestHeuristicPluginBase::getName(name);                                            \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)                             \
    {                                                                                             \
        return TestHeuristicPluginBase::getVersion(version);                                      \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginGetApiVersion(const char** version)                          \
    {                                                                                             \
        return TestHeuristicPluginBase::getApiVersion(version);                                   \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)                            \
    {                                                                                             \
        return TestHeuristicPluginBase::getType(type);                                            \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)                \
    {                                                                                             \
        return TestHeuristicPluginBase::setLoggingCallback(callback);                             \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginSetLogLevel(hipdnnSeverity_t level)                          \
    {                                                                                             \
        return TestHeuristicPluginBase::setLogLevel(level);                                       \
    }                                                                                             \
                                                                                                  \
    void hipdnnPluginGetLastErrorString(const char** errorStr)                                    \
    {                                                                                             \
        TestHeuristicPluginBase::getLastErrorString(errorStr);                                    \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnHeuristicHandleCreate(hipdnnHeuristicHandle_t* outHandle)          \
    {                                                                                             \
        return TestHeuristicPluginBase::handleCreate(outHandle);                                  \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnHeuristicHandleDestroy(hipdnnHeuristicHandle_t handle)             \
    {                                                                                             \
        return TestHeuristicPluginBase::handleDestroy(handle);                                    \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnHeuristicHandleSetDeviceProperties(hipdnnHeuristicHandle_t handle,                  \
                                                  const hipdnnPluginConstData_t* deviceProps)     \
    {                                                                                             \
        return TestHeuristicPluginBase::handleSetDeviceProperties(handle, deviceProps);           \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnHeuristicPolicyDescriptorCreate(hipdnnHeuristicHandle_t handle,                     \
                                               hipdnnHeuristicPolicyDescriptor_t* outDescriptor)  \
    {                                                                                             \
        return TestHeuristicPluginBase::policyDescriptorCreate(handle, outDescriptor);            \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnHeuristicPolicyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t descriptor)      \
    {                                                                                             \
        return TestHeuristicPluginBase::policyDescriptorDestroy(descriptor);                      \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnHeuristicPolicySetEngineIds(hipdnnHeuristicPolicyDescriptor_t descriptor,           \
                                           const int64_t* engineIds,                              \
                                           size_t engineIdCount)                                  \
    {                                                                                             \
        return TestHeuristicPluginBase::policySetEngineIds(descriptor, engineIds, engineIdCount); \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnHeuristicPolicySetSerializedGraph(hipdnnHeuristicPolicyDescriptor_t descriptor,     \
                                                 const hipdnnPluginConstData_t* serializedGraph)  \
    {                                                                                             \
        return TestHeuristicPluginBase::policySetSerializedGraph(descriptor, serializedGraph);    \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnHeuristicPolicyFinalize(hipdnnHeuristicPolicyDescriptor_t desc,    \
                                                        int32_t* applied)                         \
    {                                                                                             \
        return TestHeuristicPluginBase::policyFinalize(desc, applied);                            \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnHeuristicPolicyGetSortedEngineIds(hipdnnHeuristicPolicyDescriptor_t descriptor,     \
                                                 int64_t* engineIds,                              \
                                                 size_t* count)                                   \
    {                                                                                             \
        return TestHeuristicPluginBase::policyGetSortedEngineIds(descriptor, engineIds, count);   \
    }                                                                                             \
    }
// NOLINTEND(readability-identifier-naming,cppcoreguidelines-macro-usage)
