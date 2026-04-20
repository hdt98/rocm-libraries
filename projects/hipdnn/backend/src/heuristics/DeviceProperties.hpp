// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace hipdnn_backend::heuristics
{

/**
 * @brief Device properties for heuristic plugin selection.
 *
 * This structure carries device facts needed for engine selection heuristics.
 * It is filled by the backend (via queryDeviceProperties() or descriptor override)
 * and serialized to FlatBuffer format before being passed to heuristic plugins.
 *
 * RFC 0007 Reference: Section 6.1
 */
struct DeviceProperties
{
    int deviceId = -1; ///< Device ID from hipGetDevice
    int multiProcessorCount = 0; ///< Number of multiprocessors (compute units)
    size_t totalGlobalMem = 0; ///< Total global memory in bytes
    std::string architectureName; ///< GPU architecture name (e.g., "gfx90a", "gfx942")

    // Future optional fields can be added here and to the FlatBuffer schema
    // Example: int wavefrontSize = 0;
};

/**
 * @brief Query device properties from HIP.
 *
 * Calls hipGetDevice() and hipGetDeviceProperties() to obtain current
 * device information. This is the default acquisition method when no
 * descriptor override is provided.
 *
 * Plugins must NOT call this function. They receive serialized device
 * properties via hipdnnHeuristicHandleSetDeviceProperties.
 *
 * RFC 0007 Reference: Section 6.2
 *
 * @return DeviceProperties populated from HIP, or default values on error.
 */
DeviceProperties queryDeviceProperties();

/**
 * @brief Serialize DeviceProperties to FlatBuffer format.
 *
 * Builds a FlatBuffer-serialized representation of the device properties
 * using the schema from data_sdk/schemas/device_properties.fbs.
 *
 * The returned buffer is owned by the caller and must remain valid while
 * any hipdnnPluginConstData_t wrappers pointing to it are in use.
 *
 * RFC 0007 Reference: Section 13.2
 *
 * @param props Device properties to serialize.
 * @return Vector containing the serialized FlatBuffer bytes.
 */
std::vector<uint8_t> serializeDeviceProperties(const DeviceProperties& props);

/**
 * @brief Deserialize DeviceProperties from FlatBuffer format.
 *
 * Verifies and parses a FlatBuffer-serialized device properties buffer.
 * Throws HipdnnException if the buffer is malformed or incompatible.
 *
 * This is primarily for testing and backend internal use. Plugins should
 * use the generated FlatBuffer accessors directly.
 *
 * @param buffer Pointer to serialized FlatBuffer bytes.
 * @param size Size of the buffer in bytes.
 * @return DeviceProperties deserialized from the buffer.
 * @throws HipdnnException if buffer is invalid.
 */
DeviceProperties deserializeDeviceProperties(const uint8_t* buffer, size_t size);

/**
 * @brief Wrap serialized device properties in hipdnnPluginConstData_t.
 *
 * Creates a hipdnnPluginConstData_t wrapper pointing to the serialized buffer.
 * The buffer must remain valid while the wrapper is in use.
 *
 * @param serializedBuffer Reference to the serialized buffer (must outlive the wrapper).
 * @return hipdnnPluginConstData_t wrapper pointing to the buffer.
 */
inline hipdnnPluginConstData_t
    wrapSerializedDeviceProperties(const std::vector<uint8_t>& serializedBuffer)
{
    hipdnnPluginConstData_t wrapper;
    wrapper.ptr = serializedBuffer.data();
    wrapper.size = serializedBuffer.size();
    return wrapper;
}

} // namespace hipdnn_backend::heuristics
