// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef GUARD_COMMON_UTILS_ERRORS_HPP
#define GUARD_COMMON_UTILS_ERRORS_HPP

#include <stdexcept>
#include <string>

namespace common_utils {
[[noreturn]] inline void Throw(const std::string& file, int line, const std::string& msg)
{
    throw std::runtime_error(file + ":" + std::to_string(line) + ": " + msg);
}
} // namespace common_utils

#define COMMON_THROW(msg)                                                  \
    do { common_utils::Throw(__FILE__, __LINE__, msg); } while(false)

#define COMMON_THROW_IF(condition, msg)                                    \
    do {                                                                   \
        if((condition))                                                    \
            common_utils::Throw(__FILE__, __LINE__,                        \
                std::string(msg) + ", failed: " #condition);              \
    } while(false)

#define COMMON_THROW_HIP_STATUS(status, msg)                               \
    do {                                                                   \
        common_utils::Throw(__FILE__, __LINE__,                            \
            std::string(msg) + " (HIP error "                             \
            + std::to_string(static_cast<int>(status)) + ")");            \
    } while(false)

#endif
