// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

/**
 * @file HeuristicsPluginApi.h
 * @brief hipDNN Heuristics Plugin API
 *
 * This file contains the definitions and declarations for the hipDNN Heuristics Plugin API.
 * The API allows users to create custom heuristic/selection policy plugins for engine ordering.
 *
 * IMPORTANT: This is a separate C ABI from the engine plugin API. A single .so is either an
 * engine plugin OR a heuristic plugin, never both. Heuristic plugins do NOT export
 * hipdnnPluginGetType, hipdnnPluginGetName, or any engine plugin entry points.
 *
 * Status codes: Heuristic plugins use hipdnnPluginStatus_t (same as engine plugins) for
 * all return values. This includes HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE for policy decline.
 *
 * Serialized data: Device properties and operation graphs cross the ABI as
 * hipdnnPluginConstData_t* (same wrapper type as engine plugin APIs), not as raw pointers.
 */

#ifdef _WIN32
#ifdef HIPDNN_HEURISTIC_PLUGIN_STATIC_DEFINE
#define HIPDNN_HEURISTIC_PLUGIN_EXPORT
#else
#define HIPDNN_HEURISTIC_PLUGIN_EXPORT __declspec(dllexport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define HIPDNN_HEURISTIC_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#error "Unsupported platform or compiler"
#endif

#ifdef __cplusplus
#define HIPDNN_HEURISTIC_PLUGIN_NODISCARD [[nodiscard]]
#else
#define HIPDNN_HEURISTIC_PLUGIN_NODISCARD
#endif

// NOLINTBEGIN
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup HeuristicPluginDataTypes Heuristic Plugin Data Types
 * @brief Data types used in the Heuristics Plugin API.
 * @{
 */

/**
 * @brief Logging callback type for heuristic plugins.
 *
 * This callback allows plugins to emit log messages with a component prefix.
 * The host provides this callback via hipdnnHeuristicSetLoggingCallback.
 *
 * @param[in] severity Log severity level (HIPDNN_SEV_INFO, HIPDNN_SEV_WARN, etc.)
 * @param[in] component_prefix Component name prefix (e.g., "[StaticOrdering] ", "[Config] ")
 * @param[in] message The log message content
 */
typedef void (*hipdnnHeuristicLoggingCallback_t)(hipdnnSeverity_t severity,
                                                  const char* component_prefix,
                                                  const char* message);

/**
 * @brief Opaque handle for a heuristic plugin session.
 *
 * This handle represents a long-lived session object per loaded heuristic module
 * per hipdnnHandle. It stores plugin state (caches, tuning data, etc.) and receives
 * device properties via hipdnnHeuristicHandleSetDeviceProperties.
 *
 * Threading: This handle is NOT thread-safe (single-thread only). Concurrent use
 * requires separate hipdnnHandle instances.
 */
typedef struct hipdnnHeuristicHandle_opaque* hipdnnHeuristicHandle_t;

/**
 * @brief Opaque handle for a heuristic policy descriptor.
 *
 * This handle represents per-EngineHeuristicDescriptor slot state: candidate engine IDs,
 * serialized graph bytes, and finalize result. Owned by the EngineHeuristicDescriptor,
 * created when the policy list is established, destroyed with the descriptor.
 */
typedef struct hipdnnHeuristicPolicyDescriptor_opaque* hipdnnHeuristicPolicyDescriptor_t;

/** @} */ // End of HeuristicPluginDataTypes group

/**
 * @defgroup HeuristicPluginModuleMetadata Heuristic Plugin Module Metadata
 * @brief Functions that each heuristic plugin module must implement.
 * @{
 */

/**
 * @brief Retrieves the API version of the heuristic plugin interface.
 *
 * Returns the semantic version of the heuristics C ABI that this plugin implements
 * (e.g., "1.0.0"). The host rejects plugins with incompatible major versions.
 *
 * @param[out] version Pointer to receive the API version string (NUL-terminated).
 *                     The pointer remains valid for the plugin's lifetime.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success, error code otherwise.
 */
HIPDNN_HEURISTIC_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicGetApiVersion(const char** version);

/**
 * @brief Retrieves the stable policy ID for this heuristic plugin.
 *
 * Returns the int64_t policy identifier that MUST equal engineNameToId(canonical_utf8_name)
 * for the plugin's documented policy name. The host matches this value against the resolved
 * orderedPolicyIds after hashing user-supplied policy name strings.
 *
 * @param[out] policy_id Pointer to receive the policy ID.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success, error code otherwise.
 */
HIPDNN_HEURISTIC_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicGetPolicyId(int64_t* policy_id);

/**
 * @brief Retrieves the canonical policy name (optional).
 *
 * Returns the NUL-terminated UTF-8 canonical name that users reference in configuration
 * (e.g., "SelectionHeuristic::Config", "SelectionHeuristic::StaticOrdering"). Used for
 * logging and enumeration.
 *
 * IMPORTANT: If this function is exported, the host validates at load time that
 * engineNameToId(policy_name) == hipdnnHeuristicGetPolicyId() and REJECTS the module
 * on mismatch. This catches mistaken or overlapping well-known implementations early.
 *
 * @param[out] policy_name Pointer to receive the policy name string (NUL-terminated).
 *                         The pointer remains valid for the plugin's lifetime.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success,
 *         HIPDNN_PLUGIN_STATUS_INVALID_VALUE if not implemented.
 */
HIPDNN_HEURISTIC_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicGetPolicyName(const char** policy_name);

/**
 * @brief Retrieves the plugin implementation version (informational).
 *
 * Returns the version of the plugin implementation (e.g., "2.1.3").
 * This is separate from the API version and is informational only.
 *
 * @param[out] version Pointer to receive the version string (NUL-terminated).
 *                     The pointer remains valid for the plugin's lifetime.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success, error code otherwise.
 */
HIPDNN_HEURISTIC_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicGetPluginVersion(const char** version);

/**
 * @brief Sets the logging callback function for the plugin.
 *
 * The host calls this once after loading the heuristic .so, before any other operations.
 * The plugin should store the callback and use it for all internal logging.
 *
 * The callback signature allows plugins to include a component prefix (e.g., "[StaticOrdering] ")
 * which helps identify log messages from specific heuristic policies.
 *
 * @param[in] callback The logging callback function to use.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success, error code otherwise.
 */
HIPDNN_HEURISTIC_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicSetLoggingCallback(hipdnnHeuristicLoggingCallback_t callback);

/**
 * @brief Sets the log level for the plugin (optional).
 *
 * Synchronizes the plugin's log level with the backend's global log level.
 * Called after loading the plugin and when the global log level changes.
 *
 * @param[in] level The log level to set.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success,
 *         HIPDNN_PLUGIN_STATUS_INVALID_VALUE if not implemented.
 */
HIPDNN_HEURISTIC_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicSetLogLevel(hipdnnSeverity_t level);

/**
 * @brief Retrieves the last error string from the plugin.
 *
 * Returns a per-thread error message from the most recent failed call.
 * The pointer is valid only for immediate use; do not store it.
 *
 * @param[out] error_str Pointer to receive the error string (NUL-terminated).
 *                       If NULL, this function does nothing.
 *
 * @note Plugins must maintain error strings on a per-thread basis.
 *       Call this immediately after receiving an error status in the same thread.
 */
HIPDNN_HEURISTIC_PLUGIN_EXPORT void hipdnnHeuristicGetLastErrorString(const char** error_str);

/**
 * @brief The maximum length for heuristic plugin error strings.
 *
 * Plugins are recommended to adhere to this value.
 * The length includes the null-terminating character.
 */
#define HIPDNN_HEURISTIC_PLUGIN_ERROR_STRING_MAX_LENGTH 2048

/** @} */ // End of HeuristicPluginModuleMetadata group

/**
 * @defgroup HeuristicPluginHandleLifecycle Heuristic Plugin Handle Lifecycle
 * @brief Functions for managing the plugin session handle.
 * @{
 */

/**
 * @brief Creates a new heuristic plugin handle.
 *
 * The host calls this once per loaded heuristic module per hipdnnHandle.
 * The handle stores session state (caches, tuning data, etc.).
 *
 * @param[out] out_handle Pointer to receive the created handle.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success, error code otherwise.
 *
 * @note The caller must destroy the handle via hipdnnHeuristicHandleDestroy to avoid leaks.
 */
HIPDNN_HEURISTIC_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicHandleCreate(hipdnnHeuristicHandle_t* out_handle);

/**
 * @brief Destroys a heuristic plugin handle and releases associated resources.
 *
 * @param[in] handle The handle to destroy.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success, error code otherwise.
 *
 * @note The handle becomes invalid after this call.
 */
HIPDNN_HEURISTIC_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicHandleDestroy(hipdnnHeuristicHandle_t handle);

/**
 * @brief Sets device properties on the plugin handle.
 *
 * Provides serialized device properties (FlatBuffer) to the plugin. The host builds this
 * buffer from resolved DeviceProperties (via queryDeviceProperties() or descriptor override)
 * and passes it ONCE PER DISTINCT hipdnnHeuristicHandle_t BEFORE calling Finalize on any
 * policy descriptor created with that handle.
 *
 * The buffer contains a FlatBuffer-serialized device properties table (schema from data-SDK).
 * Plugins MUST verify the buffer with flatbuffers::Verifier and reject malformed/incompatible
 * data. Plugins MUST NOT call HIP APIs (hipGetDevice, hipGetDeviceProperties, etc.).
 *
 * Plugins query this handle state during Finalize (and as needed elsewhere on that session).
 * The policy descriptor does NOT carry a parallel device-properties buffer.
 *
 * @param[in] handle The plugin handle.
 * @param[in] device_props_serialized Pointer to hipdnnPluginConstData_t containing serialized
 *                                    device properties buffer. Must remain valid for the
 *                                    duration of this call.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success,
 *         HIPDNN_PLUGIN_STATUS_BAD_PARAM if buffer is malformed or incompatible.
 */
HIPDNN_HEURISTIC_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicHandleSetDeviceProperties(hipdnnHeuristicHandle_t handle,
                                             const hipdnnPluginConstData_t* device_props_serialized);

/** @} */ // End of HeuristicPluginHandleLifecycle group

/**
 * @defgroup HeuristicPolicyDescriptorLifecycle Heuristic Policy Descriptor Lifecycle
 * @brief Functions for managing policy descriptors (per-slot objects).
 * @{
 */

/**
 * @brief Creates a new policy descriptor.
 *
 * The host calls this once per policy slot in EngineHeuristicDescriptor, binding the
 * descriptor to the given plugin handle. The descriptor stores per-slot state:
 * candidate engine IDs, serialized graph, and finalize result.
 *
 * This BINDS the policy to the handle BEFORE Finalize, so selection code can treat the
 * handle as the source of device-properties session state (SetDeviceProperties).
 *
 * Lifecycle: Owned by EngineHeuristicDescriptor, created when the policy list is established,
 * destroyed with the descriptor.
 *
 * @param[in] plugin_handle The plugin handle this descriptor is bound to.
 * @param[out] out_desc Pointer to receive the created policy descriptor.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success, error code otherwise.
 */
HIPDNN_HEURISTIC_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyDescriptorCreate(hipdnnHeuristicHandle_t plugin_handle,
                                          hipdnnHeuristicPolicyDescriptor_t* out_desc);

/**
 * @brief Destroys a policy descriptor and releases associated resources.
 *
 * @param[in] desc The policy descriptor to destroy.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success, error code otherwise.
 *
 * @note The descriptor becomes invalid after this call.
 */
HIPDNN_HEURISTIC_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t desc);

/** @} */ // End of HeuristicPolicyDescriptorLifecycle group

/**
 * @defgroup HeuristicPolicyInputs Heuristic Policy Inputs
 * @brief Functions for setting inputs on policy descriptors.
 * @{
 */

/**
 * @brief Sets the candidate engine IDs on the policy descriptor.
 *
 * Provides the list of candidate engine IDs from EnginePluginResourceManager::getApplicableEngineIds.
 * The plugin must produce a reordered subset or permutation of these IDs.
 *
 * @param[in] desc The policy descriptor.
 * @param[in] engine_ids Array of candidate engine IDs.
 * @param[in] engine_id_count Number of engine IDs in the array.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success, error code otherwise.
 */
HIPDNN_HEURISTIC_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicySetEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                      const int64_t* engine_ids,
                                      size_t engine_id_count);

/**
 * @brief Sets the serialized operation graph on the policy descriptor.
 *
 * Provides the FlatBuffer-serialized operation graph from GraphDescriptor::getSerializedGraph().
 * The buffer contains the canonical graph representation used by the backend.
 *
 * Plugins that need structured access should parse the buffer using data-SDK generated types,
 * subject to schema version rules.
 *
 * @param[in] desc The policy descriptor.
 * @param[in] serialized_graph Pointer to hipdnnPluginConstData_t containing serialized graph
 *                             buffer. Must remain valid for the duration of this call.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success,
 *         HIPDNN_PLUGIN_STATUS_BAD_PARAM if buffer is malformed or incompatible.
 */
HIPDNN_HEURISTIC_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicySetSerializedGraph(hipdnnHeuristicPolicyDescriptor_t desc,
                                            const hipdnnPluginConstData_t* serialized_graph);

/** @} */ // End of HeuristicPolicyInputs group

/**
 * @defgroup HeuristicPolicySelection Heuristic Policy Selection
 * @brief Functions for executing policy selection and retrieving results.
 * @{
 */

/**
 * @brief Executes the policy selection logic.
 *
 * Performs applicability checking and engine ordering based on the inputs previously set
 * via SetEngineIds, SetSerializedGraph, and HandleSetDeviceProperties.
 *
 * IMPORTANT: This assumes current device-properties bytes were applied to this policy's
 * hipdnnHeuristicHandle_t via hipdnnHeuristicHandleSetDeviceProperties earlier in the
 * same EngineHeuristicDescriptor::finalize(). Plugins query that handle state as needed;
 * the host does NOT pass device properties again on this call.
 *
 * Two-phase design: This function performs the selection work; GetSortedEngineIds retrieves
 * the result. This allows future async implementations without changing function names.
 *
 * @param[in] desc The policy descriptor.
 * @param[out] out_applied Pointer to receive the result:
 *                         - Set to 1 if policy succeeded (host then calls GetSortedEngineIds)
 *                         - Set to 0 if not applicable or declined (host continues outer loop)
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success (check out_applied for applicability),
 *         error code on failure.
 *
 * @note Calls on descriptors bound to a given hipdnnHeuristicHandle_t must occur on a thread
 *       consistent with that handle's single-thread contract.
 */
HIPDNN_HEURISTIC_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyFinalize(hipdnnHeuristicPolicyDescriptor_t desc, int32_t* out_applied);

/**
 * @brief Retrieves the sorted engine IDs after successful finalize.
 *
 * Valid only after Finalize returned with out_applied == 1.
 *
 * The output IDs must be a permutation or subset of the input IDs from SetEngineIds.
 * The host validates this constraint.
 *
 * This function supports two usage patterns:
 * 1. Query count: Pass engine_ids = nullptr to get the count in num_engines
 * 2. Retrieve IDs: Pass non-null engine_ids and capacity in *num_engines,
 *    receive actual count in *num_engines
 *
 * @param[in] desc The policy descriptor.
 * @param[out] engine_ids Array to receive the sorted engine IDs, or nullptr to query count.
 * @param[in,out] num_engines Input: capacity of engine_ids array (ignored if engine_ids is null).
 *                            Output: number of IDs available/written.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success,
 *         HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED if descriptor not finalized,
 *         error code on other failures.
 */
HIPDNN_HEURISTIC_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyGetSortedEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                            int64_t* engine_ids,
                                            uint32_t* num_engines);

/** @} */ // End of HeuristicPolicySelection group

#ifdef __cplusplus
}
#endif
// NOLINTEND
