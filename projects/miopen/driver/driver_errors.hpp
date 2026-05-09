// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_DRIVER_ERRORS_HPP
#define GUARD_MIOPEN_DRIVER_ERRORS_HPP

#include <stdexcept>
#include <string>

// Driver-local replacements for the internal miopen/errors.hpp macros.
// These throw standard C++ exceptions instead of miopen::Exception,
// so the driver does not depend on internal MIOpen symbols.

namespace driver_errors_detail {

[[noreturn]] inline void Throw(const std::string& file, int line, const std::string& msg)
{
    throw std::runtime_error(file + ":" + std::to_string(line) + ": " + msg);
}

} // namespace driver_errors_detail

#define DRIVER_THROW(...)                                                  \
    do                                                                     \
    {                                                                      \
        driver_errors_detail::Throw(__FILE__, __LINE__, __VA_ARGS__);      \
    } while(false)

#define DRIVER_THROW_IF(condition, msg)                                               \
    do                                                                                \
    {                                                                                 \
        if((condition))                                                               \
        {                                                                             \
            driver_errors_detail::Throw(__FILE__,                                     \
                                        __LINE__,                                     \
                                        std::string(msg) + ", failed: " #condition);  \
        }                                                                             \
    } while(false)

#define DRIVER_THROW_HIP_STATUS(status, msg)                                              \
    do                                                                                    \
    {                                                                                     \
        driver_errors_detail::Throw(__FILE__, __LINE__,                                   \
            std::string(msg) + " (HIP error " + std::to_string(static_cast<int>(status)) \
            + ")");                                                                       \
    } while(false)

#endif // GUARD_MIOPEN_DRIVER_ERRORS_HPP
