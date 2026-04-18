// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkFmhaJit.hpp"

#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <sstream>
#include <vector>

// Phase 9: the RTC backend is an optional compile-time feature. When
// CK_FMHA_WITH_RTC is defined, the plugin links against
// composable_kernel::ck_host and invokes ck::host::fmha_rtc::compile_fwd
// in-process. When it isn't defined, compile_rtc returns a
// "not available" result and compile_auto always falls back to hipcc.
#ifdef CK_FMHA_WITH_RTC
#include <ck/host/fmha_rtc.hpp>

#include "CkFmhaRtcKernelInstance.hpp"
#include "CkFmhaRtcStats.hpp"
#endif

#include <cmath>

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

// ===========================================================================
// Phase 9: backend selection.
// ===========================================================================

JitBackend pick_jit_backend() {
    const char* env = std::getenv("CK_FMHA_JIT_BACKEND");
    // Default: prefer RTC when the build opted in (Phase 10), otherwise
    // Auto, which tries RTC and falls back to hipcc. The
    // CK_FMHA_DEFAULT_BACKEND_RTC compile-time flag is set by the build
    // system once RTC is deemed production-ready on the deployment's
    // arch / shape matrix. Until then, Auto is the safe default.
    JitBackend def =
#ifdef CK_FMHA_DEFAULT_BACKEND_RTC
        JitBackend::Rtc;
#else
        JitBackend::Auto;
#endif
    if (env == nullptr) return def;
    if (std::strcmp(env, "rtc") == 0)   return JitBackend::Rtc;
    if (std::strcmp(env, "hipcc") == 0) return JitBackend::Hipcc;
    if (std::strcmp(env, "auto") == 0)  return JitBackend::Auto;
    HIPDNN_PLUGIN_LOG_WARN("Unknown CK_FMHA_JIT_BACKEND='" << env
                            << "'; using compile-time default");
    return def;
}

namespace jit {

JitResult compile_rtc(const ck_tile::dispatcher::FmhaProblem& problem,
                      const std::string& gfx_arch, const std::string& /*output_dir*/,
                      ck_tile::dispatcher::FmhaRegistry* registry) {
#ifdef CK_FMHA_WITH_RTC
    auto start = std::chrono::steady_clock::now();
    auto& stats = RtcStats::instance();
    stats.compile_attempts.fetch_add(1, std::memory_order_relaxed);
    try {
        // Map the dispatcher's FmhaProblem onto ck_host's per-family
        // Problem. Phase 9 only binds the Fwd family -- other families
        // (pagedkv, splitkv, appendkv, batch_prefill, bwd) need their
        // own parameter mapping added here in follow-up work.
        if (problem.api_family != ck_tile::dispatcher::FmhaApiFamily::Fwd) {
            return {false, "", "RTC backend: non-fwd family not yet wired",
                    0.0, false};
        }

        ck::host::device_fmha_fwd::Problem cpp_prob;
        cpp_prob.M             = static_cast<std::size_t>(problem.seqlen_q);
        cpp_prob.N             = static_cast<std::size_t>(problem.seqlen_k);
        cpp_prob.K             = static_cast<std::size_t>(problem.hdim_q);
        cpp_prob.O             = static_cast<std::size_t>(problem.hdim_v);
        cpp_prob.batch         = static_cast<std::size_t>(problem.batch);
        cpp_prob.nhead         = static_cast<std::size_t>(problem.nhead_q);
        cpp_prob.dtype         = ck::host::DataType::Half;  // Phase 9 placeholder
        cpp_prob.is_v_rowmajor = problem.is_v_rowmajor;
        cpp_prob.is_causal     = (problem.mask_type != 0);
        cpp_prob.has_bias      = (problem.bias_type != 0);

        auto solutions = cpp_prob.GetSolutions(gfx_arch);
        if (solutions.empty()) {
            return {false, "",
                    "RTC backend: no solutions for this problem+arch", 0.0,
                    false};
        }

        ck::host::fmha_rtc::FwdLaunchParams params;
        params.batch   = cpp_prob.batch;
        params.nhead   = cpp_prob.nhead;
        // GQA / MQA: Q and O tensors have `nhead_q` heads, K/V have
        // `nhead_k`. When the two are equal this collapses back to
        // pure MHA. The wrapper resolves `nhead_ratio_qk` at launch
        // time from desc.nhead / desc.nhead_k.
        const auto nhead_k_runtime = static_cast<std::size_t>(
            problem.nhead_k > 0 ? problem.nhead_k : problem.nhead_q);
        params.nhead_k = nhead_k_runtime;
        params.M = cpp_prob.M; params.N = cpp_prob.N;
        params.K = cpp_prob.K; params.O = cpp_prob.O;
        params.scale_s = 1.0f / static_cast<float>(
            cpp_prob.K > 0 ? std::sqrt(static_cast<double>(cpp_prob.K)) : 1.0);
        // BHSD row-major strides for Q / O (using nhead_q) and K / V
        // (using nhead_k). The batch-stride accounts for the per-head
        // count of the corresponding tensor.
        params.q_stride_m     = cpp_prob.K;
        params.q_stride_nhead = cpp_prob.M * cpp_prob.K;
        params.q_stride_batch = cpp_prob.nhead * cpp_prob.M * cpp_prob.K;
        params.k_stride_n     = cpp_prob.K;
        params.k_stride_nhead = cpp_prob.N * cpp_prob.K;
        params.k_stride_batch = nhead_k_runtime * cpp_prob.N * cpp_prob.K;
        params.v_stride_n     = cpp_prob.O;
        params.v_stride_nhead = cpp_prob.N * cpp_prob.O;
        params.v_stride_batch = nhead_k_runtime * cpp_prob.N * cpp_prob.O;
        params.o_stride_m     = cpp_prob.O;
        params.o_stride_nhead = cpp_prob.M * cpp_prob.O;
        params.o_stride_batch = cpp_prob.nhead * cpp_prob.M * cpp_prob.O;

        auto& solution = solutions.front();
        auto compiled = ck::host::fmha_rtc::compile_fwd(
            cpp_prob, solution, gfx_arch, params);

        // Split the elapsed time into hot (cache-hit) vs cold (full
        // recompile) buckets. `compiled.from_cache` is set by
        // ck::host::fmha_rtc::compile_fwd() based on which branch of
        // its cache lookup it took.
        auto elapsed_us = static_cast<std::uint64_t>(compiled.build_time_s * 1e6);
        if (compiled.from_cache) {
            stats.cache_hits.fetch_add(1, std::memory_order_relaxed);
            stats.cache_load_time_us.fetch_add(elapsed_us, std::memory_order_relaxed);
        } else {
            stats.cache_misses.fetch_add(1, std::memory_order_relaxed);
            stats.compile_time_us.fetch_add(elapsed_us, std::memory_order_relaxed);
        }

        // Build an FmhaKernelKey for the dispatcher. We fill in the
        // distinguishing signature fields (family, dtype, mask, bias,
        // group mode, hdim) and the algorithm's TileShape so that
        // FmhaDispatcher::select_first_fit's scoring picks us for
        // this (problem, arch) tuple. Other fields default (they are
        // all false / zero for the MGX signature covered in Phase 9).
        //
        // Because the MGX template bakes batch/nhead/seqlen as
        // constexpr into the kernel descriptor, we also hash the
        // runtime shape into `algorithm.selection_rank` so distinct
        // shapes register as distinct entries in FmhaRegistry; without
        // this they would collide on the same (signature, tile_shape)
        // slot and the first compile would be reused for every later
        // shape (see RtcFmhaKernelInstance::supports() for the
        // matching shape-level gate).
        ck_tile::dispatcher::FmhaKernelKey key;
        key.signature.family     = ck_tile::dispatcher::FmhaKernelFamily::Fwd;
        key.signature.data_type  = problem.data_type;
        key.signature.mask_type  = problem.mask_type;
        key.signature.bias_type  = problem.bias_type;
        key.signature.is_group_mode = problem.is_group_mode;
        key.signature.is_v_rowmajor = problem.is_v_rowmajor;
        key.signature.hdim_q        = static_cast<std::uint16_t>(problem.hdim_q);
        key.signature.hdim_v        = static_cast<std::uint16_t>(problem.hdim_v);
        key.gfx_arch                = gfx_arch;
        key.algorithm.tile_shape.m0 =
            static_cast<std::uint16_t>(solution.GetTemplateParameter<std::size_t>("BM0"));
        key.algorithm.tile_shape.n0 =
            static_cast<std::uint16_t>(solution.GetTemplateParameter<std::size_t>("BN0"));
        key.algorithm.tile_shape.k0 =
            static_cast<std::uint16_t>(solution.GetTemplateParameter<std::size_t>("BK0"));
        key.algorithm.tile_shape.n1 =
            static_cast<std::uint16_t>(solution.GetTemplateParameter<std::size_t>("BN1"));
        key.algorithm.tile_shape.k1 =
            static_cast<std::uint16_t>(solution.GetTemplateParameter<std::size_t>("BK1"));
        // Disambiguate per-shape registry entries. 32-bit mix is
        // collision-safe for any practical (B,H,Sq,Sk) quad.
        auto shape_hash = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(problem.batch) * 1315423911u) ^
            (static_cast<std::uint64_t>(problem.nhead_q) * 2654435761u) ^
            (static_cast<std::uint64_t>(problem.nhead_k) * 40503u) ^
            (static_cast<std::uint64_t>(problem.seqlen_q) * 2246822507u) ^
            (static_cast<std::uint64_t>(problem.seqlen_k) * 3266489917u));
        key.algorithm.selection_rank = static_cast<int>(shape_hash & 0x7fffffff);

        std::string name = "rtc_fwd_" + problem.data_type + "_" + gfx_arch + "_B" +
                           std::to_string(problem.batch) + "H" + std::to_string(problem.nhead_q) +
                           "Sq" + std::to_string(problem.seqlen_q) + "Sk" +
                           std::to_string(problem.seqlen_k);
        auto instance = std::make_shared<RtcFmhaKernelInstance>(
            std::move(compiled), std::move(solution), key, std::move(name),
            problem.batch, problem.nhead_q, problem.nhead_k, problem.seqlen_q, problem.seqlen_k);

        auto* target_registry = registry != nullptr ? registry
                                                    : &ck_tile::dispatcher::FmhaRegistry::instance();
        if (!target_registry->register_kernel(instance)) {
            return {false, "",
                    "RTC backend: failed to register kernel into FmhaRegistry (duplicate key?)",
                    0.0, false};
        }

        auto elapsed = std::chrono::steady_clock::now() - start;
        double secs = std::chrono::duration<double>(elapsed).count();
        stats.compile_successes.fetch_add(1, std::memory_order_relaxed);
        HIPDNN_PLUGIN_LOG_INFO("RTC backend compiled + registered kernel in " << secs << "s "
                                                                              << (compiled.from_cache ? "(cache hit)" : "(cold)"));
        return {true, "", "", secs, /*already_registered=*/true};
    } catch (const std::exception& e) {
        auto elapsed = std::chrono::steady_clock::now() - start;
        double secs = std::chrono::duration<double>(elapsed).count();
        stats.compile_failures.fetch_add(1, std::memory_order_relaxed);
        {
            // Capture the failure for the at-exit summary dump. A
            // non-RTC-capable caller can still retrieve the message
            // via `RtcStats::instance().last_error`.
            std::lock_guard<std::mutex> lock(stats.last_error_mutex);
            stats.last_error = std::string("RTC compile failed: ") + e.what();
        }
        return {false, "", std::string("RTC compile failed: ") + e.what(),
                secs, false};
    }
#else
    (void)problem; (void)gfx_arch; (void)registry;
    return {false, "",
            "RTC backend: not compiled in (CK_FMHA_WITH_RTC not defined)",
            0.0, false};
#endif
}

JitResult compile_auto(const ck_tile::dispatcher::FmhaProblem& problem,
                       const std::string& gfx_arch, const std::string& output_dir,
                       ck_tile::dispatcher::FmhaRegistry* registry) {
    auto r = compile_rtc(problem, gfx_arch, output_dir, registry);
    if (r.success) return r;
    HIPDNN_PLUGIN_LOG_INFO("CK_FMHA_JIT_BACKEND=auto: RTC failed ("
                           << r.error << "), falling back to hipcc");
    return compile_hipcc(problem, gfx_arch, output_dir);
}

}  // namespace jit

JitResult jit_compile_kernel(const ck_tile::dispatcher::FmhaProblem& problem,
                             const std::string& gfx_arch, const std::string& output_dir,
                             ck_tile::dispatcher::FmhaRegistry* registry) {
    switch (pick_jit_backend()) {
        case JitBackend::Rtc:
            return jit::compile_rtc(problem, gfx_arch, output_dir, registry);
        case JitBackend::Hipcc:
            return jit::compile_hipcc(problem, gfx_arch, output_dir);
        case JitBackend::Auto:
            return jit::compile_auto(problem, gfx_arch, output_dir, registry);
    }
    return {false, "", "unreachable", 0.0, false};
}

// ===========================================================================
// Legacy hipcc backend. Renamed from jit_compile_kernel -> jit::compile_hipcc.
// ===========================================================================

namespace jit {

JitResult compile_hipcc(const ck_tile::dispatcher::FmhaProblem& problem,
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
    return {true, output, "", secs, /*already_registered=*/false};
}

}  // namespace jit

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
