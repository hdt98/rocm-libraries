// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#ifndef CK_CODE_GEN_RTC

#include "ck/utility/env.hpp"

#ifndef __HIPCC_RTC__
#include <iostream>
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

namespace ck {

/// Log a message to stdout when CK_LOGGING is enabled.
/// Safe to call from __host__ __device__ contexts: the logging body is compiled away on device.
/// File, line, and function are captured automatically at the call site via __builtin_* defaults.
template <typename... Ts>
struct LogInfo
{
    __host__ __device__
    LogInfo([[maybe_unused]] Ts... args,
          [[maybe_unused]] const char* file = __builtin_FILE(),
          [[maybe_unused]] int line         = __builtin_LINE(),
          [[maybe_unused]] const char* func = __builtin_FUNCTION())
    {
#ifndef __HIP_DEVICE_COMPILE__
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            (std::cout << ... << args)
                << " In " << file << ":" << line << ", in function: " << func << std::endl;
        }
#endif
    };
};

// Deduction guide: LogInfo("msg", val) deduces LogInfo<const char*, typeof(val)>
template <typename... Ts>
LogInfo(Ts...) -> LogInfo<Ts...>;

} // namespace ck

#pragma clang diagnostic pop
#endif
