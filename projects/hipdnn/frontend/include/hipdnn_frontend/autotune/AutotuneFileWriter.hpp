// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file AutotuneFileWriter.hpp
 * @brief JSON file writer for persisting autotuning results
 *
 * Writes autotune results in EngineOverrideConfig-compatible JSON format,
 * allowing the results to be loaded on subsequent runs via
 * HIPDNN_ENGINE_OVERRIDE_FILE. Supports append/replace semantics and
 * atomic file writes via temp file + rename.
 */

#pragma once

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>

#include <nlohmann/json.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace hipdnn_frontend
{
namespace autotune
{

/// Serialize a KnobSetting to a JSON object.
inline nlohmann::json knobSettingToJson(const KnobSetting& setting)
{
    nlohmann::json knob;
    knob["name"] = setting.knobId();

    std::visit(
        [&knob](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr(std::is_same_v<T, int64_t>)
            {
                knob["type"] = "int";
                knob["value"] = value;
            }
            else if constexpr(std::is_same_v<T, double>)
            {
                knob["type"] = "double";
                knob["value"] = value;
            }
            else if constexpr(std::is_same_v<T, std::string>)
            {
                knob["type"] = "string";
                knob["value"] = value;
            }
        },
        setting.value());

    return knob;
}

/// Get the string representation of an AutotuneStrategy
inline std::string strategyToString(AutotuneStrategy strategy)
{
    switch(strategy)
    {
    case AutotuneStrategy::SINGLE_SHOT:
        return "SINGLE_SHOT";
    case AutotuneStrategy::FIXED_AVERAGE:
        return "FIXED_AVERAGE";
    case AutotuneStrategy::RUN_UNTIL_STABLE:
        return "RUN_UNTIL_STABLE";
    default:
        return "UNKNOWN";
    }
}

/// Get the string representation of a TuneMode
inline std::string tuneModeToString(TuneMode mode)
{
    switch(mode)
    {
    case TuneMode::AUTO:
        return "AUTO";
    case TuneMode::EXHAUSTIVE:
        return "EXHAUSTIVE";
    default:
        return "UNKNOWN";
    }
}

/// Build a single JSON engine_overrides entry from an AutotuneResult.
///
/// @param result The autotune result to serialize
/// @param opName The operation name for the entry (e.g. "conv_fprop")
/// @param tensorDims Tensor dimensions for the entry (one vector<int64_t> per tensor)
/// @return A nlohmann::json object representing the entry
inline nlohmann::json buildOverrideEntry(const AutotuneResult& result,
                                         const std::string& opName,
                                         const std::vector<std::vector<int64_t>>& tensorDims)
{
    nlohmann::json entry;
    entry["op"] = opName;
    entry["engine_name"] = result.engineName;

    // Tensor patterns
    nlohmann::json tensors = nlohmann::json::array();
    for(const auto& dims : tensorDims)
    {
        nlohmann::json t;
        t["dim"] = dims;
        tensors.push_back(std::move(t));
    }
    entry["tensors"] = std::move(tensors);

    // Knob settings
    if(!result.knobSettings.empty())
    {
        nlohmann::json knobs = nlohmann::json::array();
        for(const auto& setting : result.knobSettings)
        {
            knobs.push_back(knobSettingToJson(setting));
        }
        entry["knobs"] = std::move(knobs);
    }

    // Autotune metadata
    nlohmann::json metadata;
    metadata["min_time_ms"] = result.minTimeMs;
    metadata["avg_time_ms"] = result.avgTimeMs;
    metadata["stddev_ms"] = result.stddevMs;
    metadata["iterations_run"] = result.iterationsRun;
    metadata["mode"] = tuneModeToString(result.modeUsed);
    metadata["rank"] = result.rank;
    metadata["workspace_size"] = result.workspaceSize;
    if(result.ranExhaustive)
    {
        metadata["ran_exhaustive"] = true;
    }
    if(!result.converged)
    {
        metadata["converged"] = false;
    }
    entry["autotune_metadata"] = std::move(metadata);

    return entry;
}

/// Write autotuning results to a JSON file in EngineOverrideConfig format.
///
/// The file format matches what EngineOverrideConfig::load() expects:
/// @code{.json}
/// {
///   "engine_overrides": [
///     {
///       "op": "conv_fprop",
///       "engine_name": "MIOPEN_ENGINE",
///       "tensors": [ { "dim": [1, 3, 224, 224] }, { "dim": [64, 3, 7, 7] } ],
///       "knobs": [ { "name": "SPLIT_K", "type": "int", "value": 2 } ],
///       "autotune_metadata": { "min_time_ms": 1.23, "rank": 0 }
///     }
///   ]
/// }
/// @endcode
///
/// @param filePath Output file path
/// @param opName The operation name to use in entries
/// @param results Ranked autotune results (only succeeded entries are written)
/// @param deleteAllExisting When true, starts with an empty file; when false,
///        loads existing entries and replaces matching (op, tensors) entries
/// @param tensorDims Tensor dimensions for the entry
/// @return Error on I/O failure
inline Error writeAutotuneResults(const std::string& filePath,
                                  const std::string& opName,
                                  const std::vector<AutotuneResult>& results,
                                  bool deleteAllExisting,
                                  const std::vector<std::vector<int64_t>>& tensorDims)
{
    nlohmann::json root;

    // Load existing file content unless we're deleting it all
    if(!deleteAllExisting && std::filesystem::exists(filePath))
    {
        try
        {
            std::ifstream existingFile(filePath);
            if(existingFile.is_open())
            {
                root = nlohmann::json::parse(existingFile);
            }
        }
        catch(const nlohmann::json::exception& e)
        {
            HIPDNN_FE_LOG_WARN("AutotuneFileWriter: failed to parse existing file "
                               << filePath << ": " << e.what() << ". Starting fresh.");
            root = nlohmann::json::object();
        }
    }

    if(!root.contains("engine_overrides") || !root["engine_overrides"].is_array())
    {
        root["engine_overrides"] = nlohmann::json::array();
    }

    // Build new entries from successful results
    std::vector<nlohmann::json> newEntries;
    for(const auto& result : results)
    {
        if(!result.succeeded)
        {
            continue;
        }

        auto entry = buildOverrideEntry(result, opName, tensorDims);
        newEntries.push_back(std::move(entry));
    }

    if(newEntries.empty())
    {
        HIPDNN_FE_LOG_WARN("AutotuneFileWriter: no successful results to write");
        return {ErrorCode::OK, ""};
    }

    // Remove pre-existing entries that match the new batch's (op, tensors) signature,
    // then append all new entries. This preserves multiple results within the same
    // autotune session (ranked list) while replacing entries from previous sessions.
    auto& overrides = root["engine_overrides"];

    if(!overrides.empty() && !newEntries.empty())
    {
        const auto& refOp = newEntries[0]["op"];
        const auto& refTensors = newEntries[0]["tensors"];

        // Erase pre-existing entries matching the new batch's (op, tensors) key
        overrides.erase(std::remove_if(overrides.begin(),
                                       overrides.end(),
                                       [&](const nlohmann::json& existing) {
                                           return existing.contains("op") && existing["op"] == refOp
                                                  && existing.contains("tensors")
                                                  && existing["tensors"] == refTensors;
                                       }),
                        overrides.end());
    }

    for(auto& newEntry : newEntries)
    {
        overrides.push_back(std::move(newEntry));
    }

    // Atomic write: write to temp file, then rename
    const std::string tempPath = filePath + ".tmp";
    {
        std::ofstream outFile(tempPath);
        if(!outFile.is_open())
        {
            return {ErrorCode::INVALID_VALUE,
                    "AutotuneFileWriter: cannot open temp file for writing: " + tempPath};
        }
        outFile << root.dump(2) << '\n';
        outFile.flush();
        if(!outFile.good())
        {
            return {ErrorCode::INVALID_VALUE,
                    "AutotuneFileWriter: write to temp file failed: " + tempPath};
        }
    }

    // Rename temp file to target (atomic on POSIX, best-effort on Windows)
    std::error_code ec;
    std::filesystem::rename(tempPath, filePath, ec);
    if(ec.value() != 0)
    {
        // Fallback: try remove + rename
        std::filesystem::remove(filePath, ec);
        std::filesystem::rename(tempPath, filePath, ec);
        if(ec.value() != 0)
        {
            return {ErrorCode::INVALID_VALUE,
                    "AutotuneFileWriter: failed to rename temp file to " + filePath + ": "
                        + ec.message()};
        }
    }

    HIPDNN_FE_LOG_INFO("AutotuneFileWriter: wrote " << newEntries.size() << " entries to "
                                                    << filePath);
    return {ErrorCode::OK, ""};
}

} // namespace autotune
} // namespace hipdnn_frontend

#endif // HIPDNN_FRONTEND_SKIP_JSON_LIB
