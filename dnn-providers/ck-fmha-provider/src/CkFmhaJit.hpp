// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ck_tile/dispatcher_fmha.hpp>
#include <string>

namespace ck_fmha_plugin {

struct JitResult {
    bool success = false;
    std::string so_path;
    std::string error;
    double build_time_s = 0.0;
};

/// JIT-compile a kernel for the given problem via the Python codegen pipeline.
///
/// Flow: FmhaProblem → kernel config → generate_fmha_fallback.py → hipcc → .so
///       → dlopen → ck_fmha_register_kernels → merge into registry
///
/// The compiled .so is cached to `output_dir` (defaults to CK_FMHA_JIT_CACHE_DIR
/// or /tmp/ck_fmha_jit/). Subsequent calls for the same config load instantly.
///
/// This shells out to Python + hipcc (~20-40s first time, ~1ms cached).
JitResult jit_compile_kernel(const ck_tile::dispatcher::FmhaProblem& problem,
                             const std::string& gfx_arch, const std::string& output_dir = "");

/// Load a JIT-compiled .so and merge its kernels into the given registry.
/// Returns the number of kernels merged.
int load_jit_library(const std::string& so_path, ck_tile::dispatcher::FmhaRegistry& registry,
                     const std::string& gfx_arch);

}  // namespace ck_fmha_plugin
