// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HeuristicPluginResourceManager.hpp"

#include "HeuristicPlugin.hpp"
#include "HeuristicPluginManager.hpp"
#include "HipdnnBackendPluginUnloadingMode.h"
#include "HipdnnException.hpp"
#include "logging/Logging.hpp"
#include <sstream>

namespace hipdnn_backend::plugin
{

namespace
{
// Static storage for plugin paths and manager (matching EnginePluginResourceManager pattern)
struct StaticHeuristicPluginState
{
    std::shared_ptr<HeuristicPluginManager> manager;
    std::set<std::filesystem::path> customPaths;
    hipdnnPluginLoadingMode_ext_t loadingMode = HIPDNN_PLUGIN_LOADING_ADDITIVE;
    hipdnnPluginUnloadingMode_ext_t unloadingMode = HIPDNN_PLUGIN_UNLOAD_LAZY;
    bool handleActive = false;
};

StaticHeuristicPluginState& getStaticState()
{
    static StaticHeuristicPluginState s_gState;
    return s_gState;
}

std::shared_ptr<HeuristicPluginManager> getOrCreateManager()
{
    auto& state = getStaticState();

    if(!state.manager)
    {
        state.manager = std::make_shared<HeuristicPluginManager>();
        state.manager->loadPlugins(state.customPaths, state.loadingMode);
    }

    return state.manager;
}

void resetManagerIfNeeded()
{
    auto& state = getStaticState();

    if(state.handleActive)
    {
        throw HipdnnException(
            HIPDNN_STATUS_BAD_PARAM,
            "Cannot change heuristic plugin paths while handles are active. "
            "Destroy all handles first or use HIPDNN_PLUGIN_LOADING_ADDITIVE mode.");
    }

    if(state.unloadingMode == HIPDNN_PLUGIN_UNLOAD_EAGER)
    {
        state.manager.reset();
    }
}

} // anonymous namespace

HeuristicPluginResourceManager::HeuristicPluginResourceManager() = default;

HeuristicPluginResourceManager::HeuristicPluginResourceManager(
    std::shared_ptr<HeuristicPluginManager> pm)
    : _pm(std::move(pm))
{
    THROW_IF_FALSE(_pm != nullptr, HIPDNN_STATUS_BAD_PARAM, "Plugin manager cannot be null");

    // Create plugin handles for all loaded heuristic plugins
    const auto& plugins = _pm->getPlugins();
    for(const auto& plugin : plugins)
    {
        try
        {
            // Set logging callback before creating handle
            // Use heuristicLoggingCallback which handles the 3-parameter signature
            auto logStatus = plugin->setLoggingCallback(hipdnn_backend::logging::heuristicLoggingCallback);
            if(logStatus != HIPDNN_PLUGIN_STATUS_SUCCESS)
            {
                HIPDNN_BACKEND_LOG_WARN(
                    "Failed to set logging callback on heuristic plugin with policy ID {}",
                    plugin->policyId());
            }

            // Set log level (optional)
            hipdnnSeverity_t level = HIPDNN_SEV_INFO;
            hipdnn_backend::logging::getGlobalLogLevel(level);
            plugin->setLogLevel(level);

            // Create plugin handle
            auto handle = plugin->createHandle();
            _handleToPlugin[handle] = plugin.get();
            _policyIdToHandle[plugin->policyId()] = handle;

            HIPDNN_BACKEND_LOG_INFO("Created heuristic plugin handle for policy ID {} ({})",
                                   plugin->policyId(),
                                   plugin->policyName());
        }
        catch(const HipdnnException& e)
        {
            HIPDNN_BACKEND_LOG_ERROR("Failed to create handle for heuristic plugin with policy ID {}: {}",
                                    plugin->policyId(),
                                    e.what());
            // Continue loading other plugins
        }
    }

    getStaticState().handleActive = true;
}

HeuristicPluginResourceManager::~HeuristicPluginResourceManager()
{
    // Destroy all plugin handles
    for(auto& [handle, plugin] : _handleToPlugin)
    {
        try
        {
            plugin->destroyHandle(handle);
        }
        catch(const HipdnnException& e)
        {
            HIPDNN_BACKEND_LOG_ERROR("Failed to destroy heuristic plugin handle: {}", e.what());
        }
    }

    _handleToPlugin.clear();
    _policyIdToHandle.clear();

    getStaticState().handleActive = false;
}

HeuristicPluginResourceManager::HeuristicPluginResourceManager(
    HeuristicPluginResourceManager&& other) noexcept
    : _pm(std::move(other._pm)),
      _handleToPlugin(std::move(other._handleToPlugin)),
      _policyIdToHandle(std::move(other._policyIdToHandle)),
      _cachedPolicyInfos(std::move(other._cachedPolicyInfos))
{
}

HeuristicPluginResourceManager&
    HeuristicPluginResourceManager::operator=(HeuristicPluginResourceManager&& other) noexcept
{
    if(this != &other)
    {
        _pm = std::move(other._pm);
        _handleToPlugin = std::move(other._handleToPlugin);
        _policyIdToHandle = std::move(other._policyIdToHandle);
        _cachedPolicyInfos = std::move(other._cachedPolicyInfos);
    }
    return *this;
}

void HeuristicPluginResourceManager::setHeuristicPluginPaths(
    const std::vector<std::filesystem::path>& pluginPaths, hipdnnPluginLoadingMode_ext_t loadingMode)
{
    resetManagerIfNeeded();

    auto& state = getStaticState();
    state.customPaths.clear();
    state.customPaths.insert(pluginPaths.begin(), pluginPaths.end());
    state.loadingMode = loadingMode;

    // Force reload on next create()
    state.manager.reset();
}

std::set<std::filesystem::path> HeuristicPluginResourceManager::getHeuristicPluginPaths()
{
    return getStaticState().customPaths;
}

void HeuristicPluginResourceManager::setPluginUnloadingMode(hipdnnPluginUnloadingMode_ext_t mode)
{
    getStaticState().unloadingMode = mode;
}

void HeuristicPluginResourceManager::setPluginLogLevel(hipdnnSeverity_t level)
{
    auto& state = getStaticState();
    if(!state.manager)
    {
        return; // No plugins loaded yet
    }

    const auto& plugins = state.manager->getPlugins();
    for(const auto& plugin : plugins)
    {
        auto status = plugin->setLogLevel(level);
        if(status != HIPDNN_PLUGIN_STATUS_SUCCESS
           && status != HIPDNN_PLUGIN_STATUS_INVALID_VALUE)
        {
            HIPDNN_BACKEND_LOG_WARN("Failed to set log level on heuristic plugin with policy ID {}",
                                   plugin->policyId());
        }
    }
}

std::shared_ptr<HeuristicPluginResourceManager> HeuristicPluginResourceManager::create()
{
    auto manager = getOrCreateManager();
    return std::make_shared<HeuristicPluginResourceManager>(manager);
}

hipdnnHeuristicHandle_t
    HeuristicPluginResourceManager::getHeuristicHandleForPolicyId(int64_t policyId) const
{
    auto it = _policyIdToHandle.find(policyId);
    if(it == _policyIdToHandle.end())
    {
        return nullptr;
    }
    return it->second;
}

const HeuristicPlugin*
    HeuristicPluginResourceManager::getPluginForPolicyId(int64_t policyId) const
{
    auto handle = getHeuristicHandleForPolicyId(policyId);
    if(handle == nullptr)
    {
        return nullptr;
    }

    auto it = _handleToPlugin.find(handle);
    if(it == _handleToPlugin.end())
    {
        return nullptr;
    }
    return it->second;
}

void HeuristicPluginResourceManager::setDevicePropertiesOnAllHandles(
    const hipdnnPluginConstData_t* devicePropsSerialized) const
{
    for(const auto& [handle, plugin] : _handleToPlugin)
    {
        try
        {
            plugin->setDeviceProperties(handle, devicePropsSerialized);
        }
        catch(const HipdnnException& e)
        {
            HIPDNN_BACKEND_LOG_WARN(
                "Failed to set device properties on heuristic plugin with policy ID {}: {}",
                plugin->policyId(),
                e.what());
            // Continue with other plugins
        }
    }
}

std::vector<HeuristicPolicyInfo> HeuristicPluginResourceManager::getHeuristicPolicyInfos() const
{
    if(_cachedPolicyInfos)
    {
        return *_cachedPolicyInfos;
    }

    std::vector<HeuristicPolicyInfo> infos;
    infos.reserve(_handleToPlugin.size());

    for(const auto& [handle, plugin] : _handleToPlugin)
    {
        HeuristicPolicyInfo info;
        info.policyId = plugin->policyId();
        info.policyName = std::string(plugin->policyName());
        info.pluginVersion = std::string(plugin->pluginVersion());
        info.apiVersion = std::string(plugin->apiVersion());
        infos.push_back(info);
    }

    _cachedPolicyInfos = infos;
    return infos;
}

void HeuristicPluginResourceManager::getLoadedHeuristicPluginFiles(size_t* numPlugins,
                                                                   char** pluginPaths,
                                                                   size_t* maxStringLen) const
{
    THROW_IF_FALSE(numPlugins != nullptr, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "numPlugins is null");

    if(!_pm)
    {
        *numPlugins = 0;
        if(maxStringLen != nullptr)
        {
            *maxStringLen = 0;
        }
        return;
    }

    const auto& plugins = _pm->getPlugins();
    *numPlugins = plugins.size();

    if(maxStringLen != nullptr)
    {
        size_t maxLen = 0;
        for(const auto& plugin : plugins)
        {
            // TODO: Add getFilePath() method to HeuristicPlugin when needed
            // For now, approximate with plugin version length
            maxLen = std::max(maxLen, plugin->pluginVersion().size());
        }
        *maxStringLen = maxLen;
    }

    // TODO: Implement actual path retrieval when SharedLibrary exposes getPath()
    if(pluginPaths != nullptr)
    {
        HIPDNN_BACKEND_LOG_WARN("getLoadedHeuristicPluginFiles: path retrieval not yet implemented");
    }
}

std::string HeuristicPluginResourceManager::toString() const
{
    std::ostringstream oss;
    oss << "HeuristicPluginResourceManager {\n";
    oss << "  Loaded plugins: " << _handleToPlugin.size() << "\n";

    auto infos = getHeuristicPolicyInfos();
    for(const auto& info : infos)
    {
        oss << "    Policy ID: " << info.policyId;
        if(!info.policyName.empty())
        {
            oss << " (" << info.policyName << ")";
        }
        oss << ", Plugin Version: " << info.pluginVersion;
        oss << ", API Version: " << info.apiVersion << "\n";
    }

    oss << "}";
    return oss.str();
}

} // namespace hipdnn_backend::plugin
