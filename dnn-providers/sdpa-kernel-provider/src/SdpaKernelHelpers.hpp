// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hip/hip_runtime.h>

#include <string>
#include <vector>

namespace sdpa_kernel_provider
{

// Helper to construct vector from a list of non-copyable types when initializer lists can't work
template <typename T, typename... Args>
std::vector<T> makeVector(Args&&... args)
{
    std::vector<T> v;
    v.reserve(sizeof...(args));
    (v.push_back(std::forward<Args>(args)), ...);
    return v;
}

std::string getDeviceString(hipStream_t stream);

/**
 * @brief Macro that returns and prints info log on passing provided condition.
 * Arguments after first are passed into std::format().
 * Requires a SDPA_PROVIDER_LOG_PREFIX in scope which will prefix the log
 */
#define SDPA_PROVIDER_RETURN_FALSE_IF(condition, ...)                    \
    do                                                                   \
    {                                                                    \
        if(condition)                                                    \
        {                                                                \
            HIPDNN_PLUGIN_LOG_INFO(std::string{SDPA_PROVIDER_LOG_PREFIX} \
                                   + std::format(__VA_ARGS__));          \
            return false;                                                \
        }                                                                \
    } while(0)

}
