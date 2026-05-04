// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SelectionHeuristic.hpp"

#include <string>

#include "HipdnnException.hpp"
#include "plugin/HeuristicPlugin.hpp"
#include "plugin/HeuristicPluginResourceManager.hpp"

namespace hipdnn_backend::heuristics
{

SelectionHeuristic::SelectionHeuristic(
    std::shared_ptr<plugin::HeuristicPluginResourceManager> resourceManager, int64_t policyId)
    : _resourceManager(std::move(resourceManager))
    , _policyId(policyId)
{
    THROW_IF_FALSE(_resourceManager != nullptr,
                   HIPDNN_STATUS_BAD_PARAM,
                   "HeuristicPluginResourceManager pointer cannot be null");

    auto pluginHandle = _resourceManager->getHeuristicHandleForPolicyId(_policyId);
    THROW_IF_FALSE(pluginHandle != nullptr,
                   HIPDNN_STATUS_BAD_PARAM,
                   "No heuristic plugin handle loaded for policy ID "
                       + std::to_string(_policyId));

    auto plugin = _resourceManager->getPluginForPolicyId(_policyId);
    THROW_IF_FALSE(plugin != nullptr,
                   HIPDNN_STATUS_BAD_PARAM,
                   "No heuristic plugin loaded for policy ID " + std::to_string(_policyId));

    _descriptor = plugin->createPolicyDescriptor(pluginHandle);
}

SelectionHeuristic::~SelectionHeuristic()
{
    if(_descriptor != nullptr && _resourceManager != nullptr)
    {
        try
        {
            auto plugin = lookupPlugin();
            if(plugin != nullptr)
            {
                plugin->destroyPolicyDescriptor(_descriptor);
            }
        }
        catch(const HipdnnException&) // NOLINT(bugprone-empty-catch)
        {
            // Log but don't throw from destructor
            // The plugin's logging callback should have already reported this
        }
        _descriptor = nullptr;
    }
}

SelectionHeuristic::SelectionHeuristic(SelectionHeuristic&& other) noexcept
    : _resourceManager(std::move(other._resourceManager))
    , _policyId(other._policyId)
    , _descriptor(other._descriptor)
{
    other._policyId = 0;
    other._descriptor = nullptr;
}

SelectionHeuristic& SelectionHeuristic::operator=(SelectionHeuristic&& other) noexcept
{
    if(this != &other)
    {
        // Clean up current descriptor
        if(_descriptor != nullptr && _resourceManager != nullptr)
        {
            try
            {
                auto plugin = lookupPlugin();
                if(plugin != nullptr)
                {
                    plugin->destroyPolicyDescriptor(_descriptor);
                }
            }
            catch(...) // NOLINT(bugprone-empty-catch)
            {
                // Ignore exceptions in move assignment cleanup
            }
        }

        // Move from other
        _resourceManager = std::move(other._resourceManager);
        _policyId = other._policyId;
        _descriptor = other._descriptor;
        other._policyId = 0;
        other._descriptor = nullptr;
    }
    return *this;
}

void SelectionHeuristic::setEngineIds(const std::vector<int64_t>& engineIds)
{
    THROW_IF_FALSE(
        _descriptor != nullptr, HIPDNN_STATUS_NOT_INITIALIZED, "Policy descriptor not initialized");

    auto plugin = lookupPlugin();
    THROW_IF_FALSE(plugin != nullptr,
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "Heuristic plugin no longer registered for policy ID "
                       + std::to_string(_policyId));

    plugin->setEngineIds(_descriptor, engineIds.data(), engineIds.size());
}

void SelectionHeuristic::setSerializedGraph(const hipdnnPluginConstData_t* serializedGraph)
{
    THROW_IF_FALSE(
        _descriptor != nullptr, HIPDNN_STATUS_NOT_INITIALIZED, "Policy descriptor not initialized");
    THROW_IF_FALSE(serializedGraph != nullptr,
                   HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                   "Serialized graph pointer cannot be null");

    auto plugin = lookupPlugin();
    THROW_IF_FALSE(plugin != nullptr,
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "Heuristic plugin no longer registered for policy ID "
                       + std::to_string(_policyId));

    plugin->setSerializedGraph(_descriptor, serializedGraph);
}

bool SelectionHeuristic::finalize()
{
    THROW_IF_FALSE(
        _descriptor != nullptr, HIPDNN_STATUS_NOT_INITIALIZED, "Policy descriptor not initialized");

    auto plugin = lookupPlugin();
    THROW_IF_FALSE(plugin != nullptr,
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "Heuristic plugin no longer registered for policy ID "
                       + std::to_string(_policyId));

    // Call the plugin's finalize method
    // Returns true if policy succeeded (won the outer loop)
    // Returns false if not applicable or declined
    return plugin->finalize(_descriptor);
}

std::vector<int64_t> SelectionHeuristic::getSortedEngineIds()
{
    THROW_IF_FALSE(
        _descriptor != nullptr, HIPDNN_STATUS_NOT_INITIALIZED, "Policy descriptor not initialized");

    auto plugin = lookupPlugin();
    THROW_IF_FALSE(plugin != nullptr,
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "Heuristic plugin no longer registered for policy ID "
                       + std::to_string(_policyId));

    // Call the plugin's getSortedEngineIds method
    // This is valid only after finalize() returned true
    return plugin->getSortedEngineIds(_descriptor);
}

const plugin::HeuristicPlugin* SelectionHeuristic::lookupPlugin() const
{
    if(_resourceManager == nullptr)
    {
        return nullptr;
    }
    return _resourceManager->getPluginForPolicyId(_policyId);
}

} // namespace hipdnn_backend::heuristics
