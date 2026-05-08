// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "PreferredEngineResolver.hpp"

#include "EngineOverrideConfig.hpp"
#include "logging/Logging.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>

#include <algorithm>
#include <unordered_map>

namespace hipdnn_backend::heuristics::preferred_engine
{
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

/// Index every (uid, dims, strides) triple from the FlatBuffer graph for O(1)
/// lookup during the conv scan. Owns the dim/stride vectors.
std::unordered_map<int64_t, TensorDimsStrides> indexTensorsByUid(const Graph* graph)
{
    std::unordered_map<int64_t, TensorDimsStrides> out;
    const auto*                                    tensors = graph->tensors();
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
    matchOverrideConfig(const Graph*                                          graph,
                        const std::unordered_map<int64_t, TensorDimsStrides>& tensorIndex)
{
    const auto config = EngineOverrideConfig::loadFromEnv();
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
        return TensorView{&t->dims, &t->strides};
    };

    for(const auto* node : *nodes)
    {
        if(node == nullptr)
        {
            continue;
        }

        const char*              op = nullptr;
        const TensorDimsStrides* a  = nullptr;
        const TensorDimsStrides* b  = nullptr;

        if(const auto* fwd = node->attributes_as_ConvolutionFwdAttributes())
        {
            op = "conv_fprop";
            a  = viewFor(fwd->x_tensor_uid());
            b  = viewFor(fwd->w_tensor_uid());
        }
        else if(const auto* bwd = node->attributes_as_ConvolutionBwdAttributes())
        {
            op = "conv_dgrad";
            a  = viewFor(bwd->dy_tensor_uid());
            b  = viewFor(bwd->w_tensor_uid());
        }
        else if(const auto* wrw = node->attributes_as_ConvolutionWrwAttributes())
        {
            op = "conv_wgrad";
            a  = viewFor(wrw->x_tensor_uid());
            b  = viewFor(wrw->dy_tensor_uid());
        }

        if(op == nullptr || a == nullptr || b == nullptr)
        {
            continue;
        }

        const std::vector<TensorView> views{buildView(a), buildView(b)};
        auto                          match = config->matchOperation(op, views);
        if(match.has_value())
        {
            return match;
        }
    }
    return std::nullopt;
}

} // namespace

std::optional<std::vector<int64_t>>
    resolvePreferredEngineOrder(const hipdnnPluginConstData_t& serializedGraph,
                                const std::vector<int64_t>&    candidateEngineIds)
{
    if(candidateEngineIds.empty() || serializedGraph.ptr == nullptr || serializedGraph.size == 0)
    {
        return std::nullopt;
    }

    const auto* bytes = static_cast<const uint8_t*>(serializedGraph.ptr);
    flatbuffers::Verifier verifier(bytes, serializedGraph.size);
    if(!verifier.VerifyBuffer<Graph>())
    {
        HIPDNN_BACKEND_LOG_WARN(
            "Preferred-engine precursor: invalid serialized graph, skipping override.");
        return std::nullopt;
    }

    const auto* graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(bytes);
    if(graph == nullptr)
    {
        return std::nullopt;
    }

    // 1. Explicit graph.preferred_engine_id (set via the frontend setter).
    // 2. HIPDNN_ENGINE_OVERRIDE_FILE rule matching a conv node.
    std::optional<int64_t> preferredEngineId;
    const char*            preferredSource = nullptr;
    if(auto preferredEngineIdOpt = graph->preferred_engine_id(); preferredEngineIdOpt.has_value())
    {
        preferredEngineId = preferredEngineIdOpt;
        preferredSource   = "graph.preferred_engine_id";
    }
    else
    {
        const auto tensorIndex = indexTensorsByUid(graph);
        preferredEngineId      = matchOverrideConfig(graph, tensorIndex);
        if(preferredEngineId.has_value())
        {
            preferredSource = "HIPDNN_ENGINE_OVERRIDE_FILE";
        }
    }

    if(!preferredEngineId.has_value())
    {
        return std::nullopt;
    }

    auto it = std::find(
        candidateEngineIds.begin(), candidateEngineIds.end(), *preferredEngineId);
    if(it == candidateEngineIds.end())
    {
        HIPDNN_BACKEND_LOG_INFO(
            "Preferred-engine precursor: preferred engine 0x{:x} (from {}) not in candidates.",
            static_cast<uint64_t>(*preferredEngineId),
            preferredSource);
        return std::nullopt;
    }

    std::vector<int64_t> reordered;
    reordered.reserve(candidateEngineIds.size());
    reordered.push_back(*preferredEngineId);
    for(const int64_t engineId : candidateEngineIds)
    {
        if(engineId != *preferredEngineId)
        {
            reordered.push_back(engineId);
        }
    }

    HIPDNN_BACKEND_LOG_INFO(
        "Preferred-engine precursor: reordered {} engines with preferred 0x{:x} (from {}) first.",
        reordered.size(),
        static_cast<uint64_t>(*preferredEngineId),
        preferredSource);
    return reordered;
}

} // namespace hipdnn_backend::heuristics::preferred_engine
