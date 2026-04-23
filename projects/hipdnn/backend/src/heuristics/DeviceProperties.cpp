// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "DeviceProperties.hpp"

#include <flatbuffers/flatbuffers.h>
#include <hip/hip_runtime.h>

#include "HipdnnException.hpp"

namespace hipdnn_backend::heuristics
{

hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT queryDeviceProperties()
{
    hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT devProps;

    // Get current device
    int currentDevice;
    auto status = hipGetDevice(&currentDevice);
    if(status != hipSuccess)
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR, "Failed to get current device");
    }

    // Get device properties
    hipDeviceProp_t hipProps;
    status = hipGetDeviceProperties(&hipProps, currentDevice);
    if(status != hipSuccess)
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR, "Failed to get device properties");
    }

    // Populate DevicePropertiesT
    devProps.device_id              = currentDevice;
    devProps.multi_processor_count  = hipProps.multiProcessorCount;
    devProps.total_global_mem       = hipProps.totalGlobalMem;
    devProps.architecture_name      = hipProps.gcnArchName;

    return devProps;
}

std::vector<uint8_t>
    serializeDeviceProperties(const hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT& props)
{
    flatbuffers::FlatBufferBuilder builder(256);
    auto offset = hipdnn_flatbuffers_sdk::data_objects::DeviceProperties::Pack(builder, &props);
    builder.Finish(offset, "HDDP");
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

} // namespace hipdnn_backend::heuristics
