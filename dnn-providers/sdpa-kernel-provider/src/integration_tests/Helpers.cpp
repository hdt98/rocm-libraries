// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Helpers.hpp"

namespace sdpa_integration_common
{

std::string getDeviceString(hipStream_t stream)
{
    int deviceId = -1;
    auto status = hipStreamGetDevice(stream, &deviceId);
    if(status != hipSuccess)
    {
        throw std::runtime_error("hipStreamGetDevice failed with error code: "
                                 + std::to_string(status));
    }

    hipDeviceProp_t props;
    status = hipGetDeviceProperties(&props, deviceId);
    if(status != hipSuccess)
    {
        throw std::runtime_error("hipGetDeviceProperties failed with error code: "
                                 + std::to_string(status));
    }
    std::string archStr(props.gcnArchName);

    return archStr.substr(0, archStr.find(':'));
}

}
