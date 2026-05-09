// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_DRIVER_ENV_HPP
#define GUARD_MIOPEN_DRIVER_ENV_HPP

#include <cstdlib>
#include <cstdint>
#include <string>

// Driver-local env var helpers replacing the internal miopen/env.hpp macros.

namespace driver_env {

// Returns true if the env var is set to a truthy value (non-empty, not "0").
inline bool enabled(const char* name)
{
    const char* val = std::getenv(name);
    if(val == nullptr || val[0] == '\0' || (val[0] == '0' && val[1] == '\0'))
        return false;
    return true;
}

// Returns true if the env var is unset, empty, or "0".
inline bool disabled(const char* name)
{
    return !enabled(name);
}

// Returns the uint64 value of the env var, or default_val if unset/empty.
inline uint64_t value_uint64(const char* name, uint64_t default_val = 0)
{
    const char* val = std::getenv(name);
    if(val == nullptr || val[0] == '\0')
        return default_val;
    return std::strtoull(val, nullptr, 0);
}

} // namespace driver_env

#endif // GUARD_MIOPEN_DRIVER_ENV_HPP
