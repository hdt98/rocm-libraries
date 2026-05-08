// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace hipdnn_backend::heuristics::preferred_engine
{

/**
 * @brief Backend-side preferred-engine resolver (runs as a precursor to
 *        the heuristic policy loop).
 *
 * Resolves an explicitly preferred engine ID for a graph from the
 * HIPDNN_ENGINE_OVERRIDE_FILE rule that matches a conv node in the graph.
 *
 * If a preferred engine is resolved AND it appears in @p candidateEngineIds,
 * returns the candidate list reordered with the preferred engine first.
 * Otherwise (no preferred engine, or the preferred engine is not a candidate)
 * returns std::nullopt — the caller proceeds with the regular policy loop.
 *
 * The Graph.preferred_engine_id setter is honored by the frontend as a
 * post-hoc reorder of the heuristic-ranked engine configs; it is *not*
 * consumed here. This module covers only the env/config-file knob, which
 * needs visibility into the serialized graph to match conv shapes.
 *
 * @param serializedGraph         FlatBuffer-serialized graph (must reference
 *                                a valid Graph root). nullptr is treated as
 *                                "no graph" and yields std::nullopt.
 * @param candidateEngineIds      Engine IDs the policy loop would otherwise
 *                                see, in their pre-policy order.
 * @returns Reordered engine IDs (preferred first) on hit; nullopt on miss.
 */
std::optional<std::vector<int64_t>>
    resolvePreferredEngineOrder(const hipdnnPluginConstData_t& serializedGraph,
                                const std::vector<int64_t>& candidateEngineIds);

} // namespace hipdnn_backend::heuristics::preferred_engine
