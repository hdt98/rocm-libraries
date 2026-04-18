// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Phase 17: process-wide telemetry for the hipRTC backend. Captures
// the counts the operators care about for SLO bookkeeping:
//
//   - total compile attempts
//   - cache hits  vs cold compiles
//   - cumulative wall-time spent in hipRTC
//   - last compile's error message (for `operator<<` into a log)
//
// Stats are cheap to access (all fields atomic) and can be queried by
// the `CkFmhaRtcStatsDump` tool or surfaced via `HIPDNN_PLUGIN_LOG_INFO`
// at handle-destruction time. Callers set `CK_FMHA_RTC_STATS=1` to
// force a dump to stderr on process exit.
//
// All fields are 64-bit atomic. No locking needed on the hot path.

#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>

namespace ck_fmha_plugin {

struct RtcStats {
    std::atomic<std::uint64_t> compile_attempts{0};
    std::atomic<std::uint64_t> compile_successes{0};
    std::atomic<std::uint64_t> compile_failures{0};
    std::atomic<std::uint64_t> cache_hits{0};
    std::atomic<std::uint64_t> cache_misses{0};
    // Wall time spent compiling (microseconds). Summed across attempts
    // including failures, to make the average cost inspectable.
    std::atomic<std::uint64_t> compile_time_us{0};
    // Wall time spent loading cached HSACOs (microseconds). Separate
    // counter so `compile_time_us / cache_misses` gives a clean
    // cold-compile average.
    std::atomic<std::uint64_t> cache_load_time_us{0};

    // Last failure's backtrace / stderr capture. Written under
    // last_error_mutex_; read rarely (during log dump).
    std::mutex  last_error_mutex;
    std::string last_error;

    /// Singleton accessor -- the stats live for the whole process.
    static RtcStats& instance();

    /// One-line summary suitable for logs / --help output.
    std::string summary() const;

    /// Resets all counters (except the last_error slot). Used by the
    /// unit tests so they can isolate their measurements.
    void reset();

    /// Install a stderr-on-exit dumper when CK_FMHA_RTC_STATS=1 is set.
    /// Idempotent; safe to call multiple times. Called from
    /// `CkFmhaHandle::CkFmhaHandle()` so the user gets a summary
    /// without having to instrument their code.
    static void maybeInstallAtExitDump();
};

/// RAII helper: times the enclosed block and adds the elapsed
/// microseconds to `counter`. Use at the top of compile_rtc / the
/// cache-hit path to accumulate wall-time.
class RtcTimer {
public:
    explicit RtcTimer(std::atomic<std::uint64_t>& counter) : counter_(counter), start_(clock::now()) {}
    ~RtcTimer() {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start_);
        counter_.fetch_add(static_cast<std::uint64_t>(elapsed.count()), std::memory_order_relaxed);
    }

private:
    using clock = std::chrono::steady_clock;
    std::atomic<std::uint64_t>& counter_;
    clock::time_point          start_;
};

}  // namespace ck_fmha_plugin
