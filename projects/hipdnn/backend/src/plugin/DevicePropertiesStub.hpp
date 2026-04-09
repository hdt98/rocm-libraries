// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hipdnn_backend::plugin
{

/**
 * @brief STUB: Device properties structure.
 *
 * This is a temporary stub that will be replaced by the implementation from
 * RFC Section 6 (Device Properties). Another developer is working on that section.
 *
 * TODO: Replace with actual DeviceProperties from Section 6
 */
struct DeviceProperties
{
    int deviceId            = -1;
    int multiProcessorCount = 0;
    size_t totalGlobalMem   = 0;
};

/**
 * @brief STUB: Query device properties from HIP.
 *
 * This is a temporary stub that will be replaced by the implementation from
 * RFC Section 6.2 (queryDeviceProperties).
 *
 * TODO: Replace with actual queryDeviceProperties() from Section 6
 *
 * @return DeviceProperties with default/stub values
 */
inline DeviceProperties queryDeviceProperties()
{
    // Stub implementation - returns default values
    // Real implementation will call hipGetDevice() and hipGetDeviceProperties()
    DeviceProperties props;
    props.deviceId            = 0;
    props.multiProcessorCount = 110; // Example: MI250X has 110 CUs
    props.totalGlobalMem      = 64ULL * 1024 * 1024 * 1024; // 64 GB
    return props;
}

/**
 * @brief STUB: Serialize device properties to FlatBuffer.
 *
 * This is a temporary stub that will be replaced by the implementation from
 * RFC Section 13.2 (Serialized device properties - FlatBuffer).
 *
 * TODO: Replace with actual FlatBuffer serialization from Section 13.2
 *
 * @param props Device properties to serialize
 * @return Serialized FlatBuffer bytes
 */
inline std::vector<uint8_t> serializeDeviceProperties(const DeviceProperties& props)
{
    // Stub implementation - returns a minimal buffer with a magic header
    // Real implementation will use FlatBuffer schema from data-SDK
    std::vector<uint8_t> buffer;

    // Simple serialization: [magic(4 bytes)][deviceId(4)][multiProcessorCount(4)][totalGlobalMem(8)]
    const uint32_t magic = 0xDEADBEEF; // Stub magic number
    buffer.resize(sizeof(magic) + sizeof(props.deviceId) + sizeof(props.multiProcessorCount)
                  + sizeof(props.totalGlobalMem));

    size_t offset = 0;
    std::memcpy(buffer.data() + offset, &magic, sizeof(magic));
    offset += sizeof(magic);
    std::memcpy(buffer.data() + offset, &props.deviceId, sizeof(props.deviceId));
    offset += sizeof(props.deviceId);
    std::memcpy(buffer.data() + offset, &props.multiProcessorCount, sizeof(props.multiProcessorCount));
    offset += sizeof(props.multiProcessorCount);
    std::memcpy(buffer.data() + offset, &props.totalGlobalMem, sizeof(props.totalGlobalMem));

    return buffer;
}

} // namespace hipdnn_backend::plugin
