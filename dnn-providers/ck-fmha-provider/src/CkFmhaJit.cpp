// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkFmhaJit.hpp"

#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <sstream>
#include <vector>

namespace ck_fmha_plugin {

namespace {

std::string get_jit_cache_dir(const std::string& user_dir) {
    if (!user_dir.empty()) return user_dir;
    const char* env = std::getenv("CK_FMHA_JIT_CACHE_DIR");
    if (env != nullptr) return env;
    return "/tmp/ck_fmha_jit";
}

std::string problem_to_jit_name(const ck_tile::dispatcher::FmhaProblem& p) {
    std::ostringstream ss;
    ss << p.data_type << "_h" << p.hdim_q;
    if (p.hdim_v != p.hdim_q) ss << "x" << p.hdim_v;
    ss << "_m" << p.mask_type << "_b" << p.bias_type;
    if (p.has_lse) ss << "_lse";
    if (p.has_dropout) ss << "_drop";
    if (p.has_dbias) ss << "_dbias";
    return ss.str();
}

int run_subprocess(const std::vector<std::string>& args) {
    std::vector<const char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& a : args) argv.push_back(a.c_str());
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) return -1;

    if (pid == 0) {
        execvp(argv[0], const_cast<char* const*>(argv.data()));
        _exit(127);
    }

    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

}  // namespace

JitResult jit_compile_kernel(const ck_tile::dispatcher::FmhaProblem& problem,
                             const std::string& gfx_arch, const std::string& output_dir) {
    auto start = std::chrono::steady_clock::now();

    std::string cache_dir = get_jit_cache_dir(output_dir);
    std::filesystem::create_directories(cache_dir);

    std::string kernel_name = problem_to_jit_name(problem);
    std::string so_name = "libdispatcher_fmha_" + kernel_name + ".so";
    std::string so_path = cache_dir + "/" + so_name;

    if (std::filesystem::exists(so_path)) {
        auto elapsed = std::chrono::steady_clock::now() - start;
        return {true, so_path, "", std::chrono::duration<double>(elapsed).count()};
    }

    const char* py_path_env = std::getenv("CK_DISPATCHER_PYTHON_PATH");
    if (py_path_env == nullptr) {
        return {false, "", "CK_DISPATCHER_PYTHON_PATH environment variable not set", 0.0};
    }
    std::string py_path(py_path_env);

    std::ostringstream script;
    script << "import sys; sys.path.insert(0, '" << py_path << "'); "
           << "from fmha_utils import FmhaKernelConfig, setup_fmha_dispatcher; "
           << "cfg = FmhaKernelConfig(" << "data_type='" << problem.data_type << "', "
           << "hdim_q=" << problem.hdim_q << ", " << "hdim_v=" << problem.hdim_v << ", "
           << "gfx_arch='" << gfx_arch << "'); " << "r = setup_fmha_dispatcher(cfg, "
           << "output_dir=__import__('pathlib').Path('" << cache_dir << "')); "
           << "exit(0 if r.success else 1)";

    std::vector<std::string> args = {"python3", "-c", script.str()};

    HIPDNN_PLUGIN_LOG_INFO("JIT compile: " << kernel_name << " for " << gfx_arch);

    int rc = run_subprocess(args);
    auto elapsed = std::chrono::steady_clock::now() - start;
    double secs = std::chrono::duration<double>(elapsed).count();

    if (rc != 0 || !std::filesystem::exists(so_path)) {
        return {false, "", "JIT compilation failed (rc=" + std::to_string(rc) + ")", secs};
    }

    return {true, so_path, "", secs};
}

int load_jit_library(const std::string& so_path, ck_tile::dispatcher::FmhaRegistry& registry,
                     const std::string& gfx_arch) {
    using RegisterFn = int (*)(ck_tile::dispatcher::FmhaRegistry&, const char*);

    void* lib = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        HIPDNN_PLUGIN_LOG_WARN("Failed to load JIT library: " << dlerror());
        return 0;
    }

    auto* fn = reinterpret_cast<RegisterFn>(dlsym(lib, "ck_fmha_register_kernels"));
    if (fn == nullptr) {
        HIPDNN_PLUGIN_LOG_WARN("No ck_fmha_register_kernels in: " << so_path);
        dlclose(lib);
        return 0;
    }

    auto before = registry.get_all().size();
    fn(registry, gfx_arch.c_str());
    auto after = registry.get_all().size();

    HIPDNN_PLUGIN_LOG_INFO("JIT loaded " << (after - before) << " kernels from: " << so_path);
    return static_cast<int>(after - before);
}

}  // namespace ck_fmha_plugin
