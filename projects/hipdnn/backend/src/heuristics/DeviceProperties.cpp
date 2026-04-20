// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "DeviceProperties.hpp"

#include <hip/hip_runtime.h>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>

#include "HipdnnException.hpp"
#include "logging/Logging.hpp"

// Include generated FlatBuffer header
#include <hipdnn_data_sdk/data_objects/device_properties_generated.h>

namespace hipdnn_backend::heuristics
{

DeviceProperties queryDeviceProperties()
{
    DeviceProperties out{};

    // Get current device
    int device = 0;
    hipError_t err = hipGetDevice(&device);
    if(err != hipSuccess)
    {
        HIPDNN_BACKEND_LOG_WARN("hipGetDevice failed: {}. Using default device properties.",
                                hipGetErrorString(err));
        return out;
    }
    out.deviceId = device;

    // Get device properties
    hipDeviceProp_t hipProps{};
    err = hipGetDeviceProperties(&hipProps, device);
    if(err != hipSuccess)
    {
        HIPDNN_BACKEND_LOG_WARN(
            "hipGetDeviceProperties failed: {}. Using partial device properties.",
            hipGetErrorString(err));
        return out;
    }

    out.multiProcessorCount = hipProps.multiProcessorCount;
    out.totalGlobalMem = hipProps.totalGlobalMem;
    out.architectureName = hipProps.gcnArchName;

    HIPDNN_BACKEND_LOG_DEBUG(
        "Queried device properties: deviceId={}, multiProcessorCount={}, totalGlobalMem={} bytes, "
        "architectureName={}",
        out.deviceId,
        out.multiProcessorCount,
        out.totalGlobalMem,
        out.architectureName);

    return out;
}

std::vector<uint8_t> serializeDeviceProperties(const DeviceProperties& props)
{
    flatbuffers::FlatBufferBuilder builder(256);

    // Create architecture name string
    auto archNameOffset = builder.CreateString(props.architectureName);

    // Build the DeviceProperties table
    auto devicePropsOffset = hipdnn_data_sdk::data_objects::CreateDeviceProperties(
        builder,
        props.deviceId,
        props.multiProcessorCount,
        props.totalGlobalMem,
        archNameOffset);

    builder.Finish(devicePropsOffset, "HDDP"); // File identifier for versioning

    // Copy to vector
    uint8_t* buf = builder.GetBufferPointer();
    const size_t size = builder.GetSize();
    return {buf, buf + size};
}

DeviceProperties deserializeDeviceProperties(const uint8_t* buffer, size_t size)
{
    THROW_IF_FALSE(buffer != nullptr,
                   HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                   "Device properties buffer cannot be null");
    THROW_IF_FALSE(size > 0, HIPDNN_STATUS_BAD_PARAM, "Device properties buffer size must be > 0");

    // Verify the buffer
    flatbuffers::Verifier verifier(buffer, size);
    if(!verifier.VerifyBuffer<hipdnn_data_sdk::data_objects::DeviceProperties>("HDDP"))
    {
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "Invalid device properties buffer: failed FlatBuffer verification");
    }

    // Get the root table
    auto devicePropsFB = hipdnn_data_sdk::data_objects::GetDeviceProperties(buffer);
    if(devicePropsFB == nullptr)
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "Failed to get DeviceProperties root table");
    }

    // Extract fields
    DeviceProperties props;
    props.deviceId = devicePropsFB->device_id();
    props.multiProcessorCount = devicePropsFB->multi_processor_count();
    props.totalGlobalMem = devicePropsFB->total_global_mem();
    if(devicePropsFB->architecture_name() != nullptr)
    {
        props.architectureName = devicePropsFB->architecture_name()->str();
    }

    return props;
}

} // namespace hipdnn_backend::heuristics
