// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace sdpa_integration_common
{

std::string getDeviceString(hipStream_t stream);

}
