// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef GUARD_DRIVER_ENV_HPP
#define GUARD_DRIVER_ENV_HPP

#include <cstdlib>
#include <cstdint>

namespace driver_env {
inline bool enabled(const char* name) {
    const char* val = std::getenv(name);
    return val && val[0] && !(val[0] == '0' && val[1] == '\0');
}
inline bool disabled(const char* name) { return !enabled(name); }
inline uint64_t value_uint64(const char* name, uint64_t default_val = 0) {
    const char* val = std::getenv(name);
    return (val && val[0]) ? std::strtoull(val, nullptr, 0) : default_val;
}
} // namespace driver_env

#endif
