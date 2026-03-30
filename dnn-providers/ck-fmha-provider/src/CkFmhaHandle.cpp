// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkFmhaHandle.hpp"

#include <dlfcn.h>

#include <cstdlib>
#include <filesystem>

#include "CkFmhaJit.hpp"

// REGISTER_GENERATED_KERNELS is defined via -include of the generated dispatch header
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "CkFmhaContainer.hpp"

namespace {

std::string detect_gfx_arch() {
    hipDeviceProp_t prop;
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess)
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "CkFmhaHandle: hipGetDevice failed");
    if (hipGetDeviceProperties(&prop, device) != hipSuccess)
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "CkFmhaHandle: hipGetDeviceProperties failed");
    return prop.gcnArchName;
}

}  // namespace

CkFmhaHandle::CkFmhaHandle() {
    gfx_arch_ = detect_gfx_arch();
    auto colon = gfx_arch_.find(':');
    if (colon != std::string::npos) gfx_arch_ = gfx_arch_.substr(0, colon);

    registry_ = ck_tile::dispatcher::make_fmha_registry("ck_fmha_" + gfx_arch_);

    REGISTER_GENERATED_KERNELS(*registry_, gfx_arch_);
    registry_->filter_by_arch(gfx_arch_);

    dispatcher_ = std::make_shared<ck_tile::dispatcher::FmhaDispatcher>(registry_.get(), gfx_arch_);
    dispatcher_->set_benchmarking(false);

    const char* lib_path_env = std::getenv("CK_FMHA_KERNEL_LIB_PATH");
    if (lib_path_env != nullptr) loadSupplementalKernels(lib_path_env);

    HIPDNN_PLUGIN_LOG_INFO("CkFmhaHandle: arch=" << gfx_arch_
                                                 << " kernels=" << registry_->get_all().size()
                                                 << " jit=" << (jitEnabled() ? "on" : "off"));
}

CkFmhaHandle::~CkFmhaHandle() {
    for (auto* h : supplemental_lib_handles_) {
        if (h != nullptr) dlclose(h);
    }
}

hipdnn_plugin_sdk::EngineManager<CkFmhaHandle, ck_fmha_plugin::CkFmhaSettings, CkFmhaContext>&
CkFmhaHandle::getEngineManager() {
    return container->getEngineManager();
}

const ck_tile::dispatcher::FmhaExecutionPlan* CkFmhaHandle::getCachedPlan(
    const std::string& key) const {
    std::lock_guard<std::mutex> lock(plan_cache_mutex_);
    auto it = plan_cache_.find(key);
    return (it != plan_cache_.end()) ? &it->second : nullptr;
}

void CkFmhaHandle::cachePlan(const std::string& key,
                             ck_tile::dispatcher::FmhaExecutionPlan plan) const {
    std::lock_guard<std::mutex> lock(plan_cache_mutex_);
    plan_cache_.emplace(std::move(key), std::move(plan));
}

bool CkFmhaHandle::jitEnabled() {
    const char* env = std::getenv("CK_FMHA_ENABLE_JIT");
    return env != nullptr && std::string(env) == "1";
}

bool CkFmhaHandle::jitAndLoad(const ck_tile::dispatcher::FmhaProblem& problem) const {
    if (!jitEnabled()) return false;

    std::lock_guard<std::mutex> lock(jit_mutex_);

    // Double-check after acquiring lock -- another thread may have JIT'd the same kernel
    if (dispatcher_->select_kernel(problem) != nullptr) return true;

    HIPDNN_PLUGIN_LOG_INFO("JIT: compiling kernel for " << problem.to_string());

    auto result = ck_fmha_plugin::jit_compile_kernel(problem, gfx_arch_);
    if (!result.success) {
        HIPDNN_PLUGIN_LOG_WARN("JIT compilation failed: " << result.error);
        return false;
    }

    HIPDNN_PLUGIN_LOG_INFO("JIT: compiled in " << result.build_time_s << "s -> " << result.so_path);

    int added = ck_fmha_plugin::load_jit_library(result.so_path, *registry_, gfx_arch_);
    if (added <= 0) {
        HIPDNN_PLUGIN_LOG_WARN("JIT: no kernels loaded from " << result.so_path);
        return false;
    }

    HIPDNN_PLUGIN_LOG_INFO("JIT: loaded " << added
                                          << " kernel(s), total=" << registry_->get_all().size());

    return dispatcher_->select_kernel(problem) != nullptr;
}

void CkFmhaHandle::loadSupplementalKernels(const std::string& dir_path) {
    using RegisterFn = int (*)(ck_tile::dispatcher::FmhaRegistry&, const char*);

    namespace fs = std::filesystem;
    std::error_code ec;
    for (auto& entry : fs::directory_iterator(dir_path, ec)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".so") continue;

        void* lib = dlopen(entry.path().c_str(), RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) {
            HIPDNN_PLUGIN_LOG_WARN("Failed to load kernel library: " << entry.path() << " ("
                                                                     << dlerror() << ")");
            continue;
        }

        auto* reg_fn = reinterpret_cast<RegisterFn>(dlsym(lib, "ck_fmha_register_kernels"));
        if (reg_fn == nullptr) {
            HIPDNN_PLUGIN_LOG_WARN("No ck_fmha_register_kernels in: " << entry.path());
            dlclose(lib);
            continue;
        }

        auto temp_reg = ck_tile::dispatcher::make_fmha_registry("supplemental");
        reg_fn(*temp_reg, gfx_arch_.c_str());
        registry_->merge_from(*temp_reg);
        supplemental_lib_handles_.push_back(lib);

        HIPDNN_PLUGIN_LOG_INFO("Loaded supplemental kernels from: " << entry.path());
    }
}
