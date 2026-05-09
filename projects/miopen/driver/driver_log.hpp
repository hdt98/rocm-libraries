// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef GUARD_DRIVER_LOG_HPP
#define GUARD_DRIVER_LOG_HPP

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace driver_log {
enum class Level : int {
    Default = 0, Quiet = 1, Fatal = 2, Error = 3,
    Warning = 4, Info = 5, Info2 = 6, Trace = 7,
};

inline Level GetConfiguredLevel() {
    static const Level lvl = [] {
        const char* val = std::getenv("MIOPEN_LOG_LEVEL");
        if(!val || !val[0]) return Level::Warning;
        int v = std::atoi(val);
        return (v > 0 && v <= 7) ? static_cast<Level>(v) : Level::Warning;
    }();
    return lvl;
}

inline bool IsEnabled(Level level) { return level <= GetConfiguredLevel(); }

inline std::string_view LevelTag(Level level) {
    switch(level) {
    case Level::Fatal: return "Fatal"; case Level::Error: return "Error";
    case Level::Warning: return "Warning"; case Level::Info: return "Info";
    case Level::Info2: return "Info2"; case Level::Trace: return "Trace";
    default: return "Log";
    }
}
} // namespace driver_log

#define DRIVER_LOG(level, ...)                                               \
    do { if(driver_log::IsEnabled(level))                                    \
        std::cerr << "MIOpenDriver " << driver_log::LevelTag(level)          \
                  << ": " << __VA_ARGS__ << std::endl; } while(false)

#define DRIVER_LOG_ERROR(...)   DRIVER_LOG(driver_log::Level::Error, __VA_ARGS__)
#define DRIVER_LOG_WARNING(...) DRIVER_LOG(driver_log::Level::Warning, __VA_ARGS__)
#define DRIVER_LOG_INFO2(...)   DRIVER_LOG(driver_log::Level::Info2, __VA_ARGS__)

#endif
