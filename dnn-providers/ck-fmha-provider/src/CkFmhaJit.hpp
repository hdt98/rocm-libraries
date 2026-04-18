// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ck_tile/dispatcher_fmha.hpp>
#include <string>

namespace ck_fmha_plugin {

struct JitResult {
    bool success = false;
    std::string so_path;  // hipcc path: .so on disk; rtc path: empty
    std::string error;
    double build_time_s = 0.0;
    bool already_registered = false;  // rtc path: set true when the RTC
                                       // backend registered directly into
                                       // the live registry (no dlopen needed)
};

/// JIT backend selector. Controlled at runtime by CK_FMHA_JIT_BACKEND:
///   "rtc"   -- Phase 9+: in-process hipRTC via ck::host::fmha_rtc
///   "hipcc" -- legacy Python subprocess + hipcc + .so + dlopen path
///   "auto"  -- try RTC first, fall back to hipcc on failure (default
///              during the migration; flips to pure RTC in Phase 10)
enum class JitBackend { Rtc, Hipcc, Auto };
JitBackend pick_jit_backend();

/// Primary entry point: selects backend from CK_FMHA_JIT_BACKEND.
/// When `registry` is non-null and the RTC backend is selected, the
/// newly compiled kernel is registered directly into that registry
/// (no dlopen step required). When `registry` is null the RTC backend
/// falls back to the singleton FmhaRegistry::instance().
JitResult jit_compile_kernel(const ck_tile::dispatcher::FmhaProblem& problem,
                             const std::string& gfx_arch, const std::string& output_dir = "",
                             ck_tile::dispatcher::FmhaRegistry* registry = nullptr);

namespace jit {

/// Phase 9 RTC backend. Uses ck::host::fmha_rtc to compile the kernel
/// in-process via hipRTC and register it into the supplied registry
/// (or FmhaRegistry::instance() when nullptr). No Python, no .so, no
/// dlopen. Returns already_registered=true on success.
JitResult compile_rtc(const ck_tile::dispatcher::FmhaProblem& problem,
                      const std::string& gfx_arch, const std::string& output_dir,
                      ck_tile::dispatcher::FmhaRegistry* registry = nullptr);

/// Legacy hipcc backend. Shells out to python3 -> hipcc -> .so.
JitResult compile_hipcc(const ck_tile::dispatcher::FmhaProblem& problem,
                        const std::string& gfx_arch, const std::string& output_dir);

/// Auto: compile_rtc first, fall back to compile_hipcc on failure.
JitResult compile_auto(const ck_tile::dispatcher::FmhaProblem& problem,
                       const std::string& gfx_arch, const std::string& output_dir,
                       ck_tile::dispatcher::FmhaRegistry* registry = nullptr);

}  // namespace jit

/// Load a JIT-compiled .so (hipcc path) and merge its kernels into the
/// given registry. Returns the number of kernels merged.
/// Not used by the RTC path -- RTC registers directly into the live
/// registry via the ck_host facade.
int load_jit_library(const std::string& so_path, ck_tile::dispatcher::FmhaRegistry& registry,
                     const std::string& gfx_arch);

}  // namespace ck_fmha_plugin
