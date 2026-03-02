// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>

namespace ck {

enum class DeviceArch
{
    All,
    Gfx950
};

__device__
consteval bool matches_with_compilation_target(DeviceArch device_arch)
{
  switch(device_arch)
  {
    case DeviceArch::All: return true;
    case DeviceArch::Gfx950:
#ifdef __gfx950__
      return true;
#else      
      return false;
#endif
    default: return false;
  }
};

}
