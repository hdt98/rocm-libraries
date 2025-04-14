// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <map>
#include <iostream>
#include "ck_tile/core/config.hpp"

namespace ck_tile {
inline std::string get_device_name()
{
    static std::string cached_name;
    static bool is_initialized = false;

    if(is_initialized)
        return cached_name;
    hipDeviceProp_t props{};
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return std::string();
    }

    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
    {
        return std::string();
    }
    const std::string raw_name(props.gcnArchName);
    const auto name = raw_name.substr(0, raw_name.find(':')); // str.substr(0, npos) returns str.

    cached_name    = name;
    is_initialized = true;
    return cached_name;
}

inline bool is_gfx13_supported()
{
    return ck_tile::get_device_name() == "gfx1300" || ck_tile::get_device_name() == "gfx1301" ||
           ck_tile::get_device_name() == "gfx1302";
}

} // namespace ck_tile
