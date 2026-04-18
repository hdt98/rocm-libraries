// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkFmhaRtcStats.hpp"

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace ck_fmha_plugin {

RtcStats& RtcStats::instance() {
    static RtcStats s;
    return s;
}

std::string RtcStats::summary() const {
    auto attempts = compile_attempts.load(std::memory_order_relaxed);
    auto succ     = compile_successes.load(std::memory_order_relaxed);
    auto fail     = compile_failures.load(std::memory_order_relaxed);
    auto hits     = cache_hits.load(std::memory_order_relaxed);
    auto misses   = cache_misses.load(std::memory_order_relaxed);
    auto comp_us  = compile_time_us.load(std::memory_order_relaxed);
    auto load_us  = cache_load_time_us.load(std::memory_order_relaxed);

    const double avg_cold_ms = misses > 0 ? (double(comp_us) / misses) / 1000.0 : 0.0;
    const double avg_hot_ms  = hits > 0 ? (double(load_us) / hits) / 1000.0 : 0.0;
    const double total_comp_s = comp_us / 1e6;
    const double total_load_s = load_us / 1e6;

    std::ostringstream oss;
    oss << "CK FMHA hipRTC stats: "
        << "attempts=" << attempts
        << " (success=" << succ << ", fail=" << fail << ")"
        << ", cache=" << hits << "/" << (hits + misses) << " hits"
        << ", cold_avg=" << std::fixed << std::setprecision(1) << avg_cold_ms << "ms"
        << ", hot_avg=" << std::fixed << std::setprecision(2) << avg_hot_ms << "ms"
        << ", total_compile=" << std::fixed << std::setprecision(2) << total_comp_s << "s"
        << ", total_cache_load=" << std::fixed << std::setprecision(2) << total_load_s << "s";
    return oss.str();
}

void RtcStats::reset() {
    compile_attempts.store(0, std::memory_order_relaxed);
    compile_successes.store(0, std::memory_order_relaxed);
    compile_failures.store(0, std::memory_order_relaxed);
    cache_hits.store(0, std::memory_order_relaxed);
    cache_misses.store(0, std::memory_order_relaxed);
    compile_time_us.store(0, std::memory_order_relaxed);
    cache_load_time_us.store(0, std::memory_order_relaxed);
}

void RtcStats::maybeInstallAtExitDump() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        const char* env = std::getenv("CK_FMHA_RTC_STATS");
        if (env == nullptr || std::string(env) != "1") return;
        std::atexit([] {
            std::fprintf(stderr, "[%s]\n", RtcStats::instance().summary().c_str());
            std::fflush(stderr);
        });
    });
}

}  // namespace ck_fmha_plugin
