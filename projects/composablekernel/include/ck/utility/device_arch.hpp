// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {

enum class DeviceArch
{
    All,
    Gfx950
};

inline std::ostream& operator<<(std::ostream& os, const DeviceArch& device_arch)
{
  switch(device_arch)
  {    
    case DeviceArch::All: os << "All"; break;
    case DeviceArch::Gfx950: os << "gfx950"; break;
    default: os << "Unknown"; break;
  }
  return os;
};

// Compile-time check if the device architecture matches with the compilation target
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

// Runtime check if the device architecture is supported by the current device
inline bool is_supported(DeviceArch device_arch)
{
  const auto& device_name = get_device_name();
  switch(device_arch)
  {    
    case DeviceArch::All: return true;
    case DeviceArch::Gfx950: return device_name == "gfx950";
    default: return false;
  }
};

}
