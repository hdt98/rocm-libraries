// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkFmhaJit.hpp"

#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
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

std::string run_subprocess_capture(const std::vector<std::string>& args) {
    int pipefd[2];
    if (pipe(pipefd) < 0) return "";

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return "";
    }

    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);

        std::vector<const char*> argv;
        for (const auto& a : args) argv.push_back(a.c_str());
        argv.push_back(nullptr);
        execvp(argv[0], const_cast<char* const*>(argv.data()));
        _exit(127);
    }

    close(pipefd[1]);
    std::string output;
    std::array<char, 4096> buf;
    ssize_t n;
    while ((n = read(pipefd[0], buf.data(), buf.size())) > 0)
        output.append(buf.data(), static_cast<size_t>(n));
    close(pipefd[0]);

    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return "";

    // Trim trailing whitespace
    while (!output.empty() && (output.back() == '\n' || output.back() == ' ')) output.pop_back();
    return output;
}

}  // namespace

JitResult jit_compile_kernel(const ck_tile::dispatcher::FmhaProblem& problem,
                             const std::string& gfx_arch, const std::string& output_dir) {
    auto start = std::chrono::steady_clock::now();

    std::string cache_dir = get_jit_cache_dir(output_dir);
    std::filesystem::create_directories(cache_dir);

    const char* py_path_env = std::getenv("CK_DISPATCHER_PYTHON_PATH");
    if (py_path_env == nullptr)
        return {false, "", "CK_DISPATCHER_PYTHON_PATH environment variable not set", 0.0};

    std::string py_path(py_path_env);

    // The script compiles a kernel and prints the .so path to stdout.
    // setup_fmha_dispatcher handles caching internally -- if the .so
    // already exists it loads instantly.
    std::ostringstream script;
    script << "import sys; " << "sys.path.insert(0, '" << py_path << "'); "
           << "sys.path.insert(0, '" << py_path << "/../codegen'); "
           << "from fmha_utils import FmhaKernelConfig, setup_fmha_dispatcher; "
           << "from pathlib import Path; " << "cfg = FmhaKernelConfig(" << "data_type='"
           << problem.data_type << "', " << "hdim_q=" << problem.hdim_q << ", "
           << "hdim_v=" << problem.hdim_v << ", " << "gfx_arch='" << gfx_arch << "'" << "); "
           << "r = setup_fmha_dispatcher(cfg, " << "output_dir=Path('" << cache_dir << "')); "
           << "print(r.library_path if r.success else 'FAIL:' + str(r.error))";

    std::vector<std::string> args = {"python3", "-c", script.str()};

    HIPDNN_PLUGIN_LOG_INFO("JIT compile: hdim=" << problem.hdim_q << "x" << problem.hdim_v << " "
                                                << problem.data_type << " for " << gfx_arch);

    std::string output = run_subprocess_capture(args);
    auto elapsed = std::chrono::steady_clock::now() - start;
    double secs = std::chrono::duration<double>(elapsed).count();

    if (output.empty() || output.substr(0, 5) == "FAIL:") {
        std::string err = output.empty() ? "subprocess failed" : output.substr(5);
        HIPDNN_PLUGIN_LOG_WARN("JIT compilation failed: " << err);
        return {false, "", err, secs};
    }

    if (!std::filesystem::exists(output)) {
        HIPDNN_PLUGIN_LOG_WARN("JIT .so not found: " << output);
        return {false, "", "JIT .so not found at: " + output, secs};
    }

    HIPDNN_PLUGIN_LOG_INFO("JIT compiled in " << secs << "s -> " << output);
    return {true, output, "", secs};
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
