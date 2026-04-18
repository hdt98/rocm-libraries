// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// End-to-end hipRTC validation demo: hipDNN Graph API → CK FMHA plugin
// → hipRTC-compiled kernel, with CPU reference for numerical validation.
//
// For each shape the demo:
//   1. Builds an SDPA forward graph via the hipDNN Graph API.
//   2. Calls `graph.build(handle)` which discovers the CK plugin,
//      pipelines through the dispatcher, and invokes
//      `jit_compile_kernel` -> `compile_rtc()` to hipRTC-compile the
//      kernel in-process (no Python, no hipcc subprocess, no .so).
//   3. Calls `graph.execute(...)` to launch the RTC-compiled kernel
//      through the dispatcher.
//   4. Copies the output back and compares against a CPU reference
//      computed as softmax(Q K^T / sqrt(d)) V.
//
// Build:
//   cmake --build . --target ck_fmha_e2e_rtc_demo
//
// Run:
//   HIPDNN_PLUGIN_PATH=<build_dir>              \
//   CK_FMHA_ENABLE_JIT=1                         \
//   CK_FMHA_JIT_BACKEND=rtc                      \
//   ./ck_fmha_e2e_rtc_demo [--warmup N] [--repeat N]

#include <fcntl.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hipdnn_frontend.hpp>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#define HIP_CHECK(expr)                                                                      \
    do {                                                                                     \
        hipError_t _e = (expr);                                                              \
        if (_e != hipSuccess) {                                                              \
            std::cerr << "HIP error: " << hipGetErrorString(_e) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                              \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

#define HIPDNN_FE_CHECK(err)                                                      \
    do {                                                                          \
        auto _e = (err);                                                          \
        if (_e.get_code() != hipdnn_frontend::ErrorCode::OK) {                    \
            throw std::runtime_error(std::string("hipDNN: ") + _e.get_message()); \
        }                                                                         \
    } while (0)

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace {

struct Shape {
    int batch, nhead_q, nhead_k, seqlen_q, seqlen_k, hdim_q, hdim_v;
    uint64_t fwd_flops() const {
        return 2ULL * batch * nhead_q * seqlen_q * seqlen_k * (hdim_q + hdim_v);
    }
    std::string label() const {
        char buf[128];
        if (nhead_q == nhead_k) {
            std::snprintf(buf, sizeof(buf), "B=%d H=%d Sq=%d Sk=%d Dq=%d Dv=%d", batch, nhead_q,
                          seqlen_q, seqlen_k, hdim_q, hdim_v);
        } else {
            std::snprintf(buf, sizeof(buf), "B=%d Hq=%d Hk=%d Sq=%d Sk=%d Dq=%d Dv=%d", batch,
                          nhead_q, nhead_k, seqlen_q, seqlen_k, hdim_q, hdim_v);
        }
        return buf;
    }
    size_t q_count() const {
        return size_t(batch) * nhead_q * seqlen_q * hdim_q;
    }
    size_t k_count() const {
        return size_t(batch) * nhead_k * seqlen_k * hdim_q;
    }
    size_t v_count() const {
        return size_t(batch) * nhead_k * seqlen_k * hdim_v;
    }
    size_t o_count() const {
        return size_t(batch) * nhead_q * seqlen_q * hdim_v;
    }
};

std::vector<__half> random_fp16(size_t n, unsigned seed, float scale = 0.02f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<__half> v(n);
    for (auto& x : v) x = __float2half(dist(rng));
    return v;
}

// CPU reference: softmax(Q K^T / sqrt(d)) V, BHSD layout, no mask, no bias.
// Handles GQA by rotating Q heads onto K/V heads: K/V head index =
// floor(h * nhead_k / nhead_q).
std::vector<float> cpu_attention_ref(const std::vector<__half>& q,
                                     const std::vector<__half>& k,
                                     const std::vector<__half>& v,
                                     const Shape& s) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(s.hdim_q));
    std::vector<float> out(s.o_count(), 0.0f);

    for (int b = 0; b < s.batch; ++b) {
        for (int h = 0; h < s.nhead_q; ++h) {
            const int hk = (h * s.nhead_k) / s.nhead_q;
            for (int i = 0; i < s.seqlen_q; ++i) {
                std::vector<float> scores(s.seqlen_k, 0.0f);
                for (int j = 0; j < s.seqlen_k; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < s.hdim_q; ++d) {
                        size_t q_idx = (((size_t)b * s.nhead_q + h) * s.seqlen_q + i) * s.hdim_q + d;
                        size_t k_idx = (((size_t)b * s.nhead_k + hk) * s.seqlen_k + j) * s.hdim_q + d;
                        dot += __half2float(q[q_idx]) * __half2float(k[k_idx]);
                    }
                    scores[j] = dot * scale;
                }

                float mx = *std::max_element(scores.begin(), scores.end());
                float sum = 0.0f;
                for (auto& x : scores) {
                    x = std::exp(x - mx);
                    sum += x;
                }
                const float inv = 1.0f / sum;
                for (auto& x : scores) x *= inv;

                for (int d = 0; d < s.hdim_v; ++d) {
                    float acc = 0.0f;
                    for (int j = 0; j < s.seqlen_k; ++j) {
                        size_t v_idx =
                            (((size_t)b * s.nhead_k + hk) * s.seqlen_k + j) * s.hdim_v + d;
                        acc += scores[j] * __half2float(v[v_idx]);
                    }
                    size_t o_idx =
                        (((size_t)b * s.nhead_q + h) * s.seqlen_q + i) * s.hdim_v + d;
                    out[o_idx] = acc;
                }
            }
        }
    }
    return out;
}

struct Metrics {
    bool ok         = false;
    double max_abs  = 0.0;
    double mean_abs = 0.0;
    double ref_rms  = 0.0;  // |ref|_rms, lets max_abs be read relative to output magnitude
};

// Scale-aware allclose: tolerances must track the reference's RMS
// magnitude, not a fixed constant. fp16 inputs at N(0, 0.02^2) give
// ref_rms ~= 1e-3, for which a constant atol of 3e-2 is ~30x the
// signal and accepts essentially anything. We gate on
// `max_abs <= 1e-3 * ref_rms + 0.02 * |ref|` which catches the
// shape-mismatch failure mode exposed by the 7.9 s vs 0.09 s
// build-time asymmetry (see inline commentary in CkFmhaJit.cpp).
Metrics compare(const std::vector<__half>& got, const std::vector<float>& ref) {
    Metrics m;
    if (got.size() != ref.size()) return m;
    double abs_sum = 0.0;
    double ref_sq  = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double g = __half2float(got[i]);
        double r = ref[i];
        double abs_err = std::fabs(g - r);
        m.max_abs = std::max(m.max_abs, abs_err);
        abs_sum += abs_err;
        ref_sq += r * r;
    }
    const double n = static_cast<double>(got.size());
    m.mean_abs     = abs_sum / n;
    m.ref_rms      = std::sqrt(ref_sq / n);
    const double atol = std::max(1e-3 * m.ref_rms, 1e-6);
    const double rtol = 2e-2;
    m.ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
        double r = ref[i];
        double abs_err = std::fabs(static_cast<double>(__half2float(got[i])) - r);
        if (abs_err > atol + rtol * std::fabs(r)) {
            m.ok = false;
            break;
        }
    }
    return m;
}

struct RunResult {
    bool success      = false;
    double build_ms   = 0.0;
    double exec_ms    = 0.0;
    std::vector<__half> o_host;
    std::string error;
};

// RAII helper that redirects stderr to /dev/null during hipRTC
// compilation. hipRTC emits ~12 harmless `-Wlifetime-safety-...`
// warnings per compile that drown out the demo's output; they are
// produced by the stripped CK Tile headers for CK Tile's own
// lifetime-analysis pragmas. Users who want to see them can bypass
// the silencer by setting CK_FMHA_RTC_SHOW_BUILD_STDERR=1.
struct StderrSilencer {
    int saved_fd   = -1;
    int devnull_fd = -1;
    bool active    = false;

    StderrSilencer() {
        if (std::getenv("CK_FMHA_RTC_SHOW_BUILD_STDERR") != nullptr) return;
        saved_fd = dup(STDERR_FILENO);
        devnull_fd = open("/dev/null", O_WRONLY);
        if (saved_fd >= 0 && devnull_fd >= 0) {
            std::fflush(stderr);
            dup2(devnull_fd, STDERR_FILENO);
            active = true;
        }
    }
    ~StderrSilencer() {
        if (active) {
            std::fflush(stderr);
            dup2(saved_fd, STDERR_FILENO);
        }
        if (saved_fd >= 0) close(saved_fd);
        if (devnull_fd >= 0) close(devnull_fd);
    }
};

RunResult run_fwd(hipdnnHandle_t handle, const Shape& s, const std::vector<__half>& q_host,
                  const std::vector<__half>& k_host, const std::vector<__half>& v_host, int warmup,
                  int repeat) {
    RunResult r;
    try {
        Graph graph;
        graph.set_io_data_type(DataType::HALF)
            .set_compute_data_type(DataType::FLOAT)
            .set_name("sdpa_fwd_rtc");

        // BHSD layout: [batch, head, seq, dim] with row-major strides.
        // Q / O use nhead_q; K / V use nhead_k. When nhead_q == nhead_k
        // this is the MHA special case.
        auto Q = Graph::tensor(
            TensorAttributes()
                .set_dim({s.batch, s.nhead_q, s.seqlen_q, s.hdim_q})
                .set_stride(
                    {s.nhead_q * s.seqlen_q * s.hdim_q, s.seqlen_q * s.hdim_q, s.hdim_q, 1})
                .set_uid(1));
        auto K = Graph::tensor(
            TensorAttributes()
                .set_dim({s.batch, s.nhead_k, s.seqlen_k, s.hdim_q})
                .set_stride(
                    {s.nhead_k * s.seqlen_k * s.hdim_q, s.seqlen_k * s.hdim_q, s.hdim_q, 1})
                .set_uid(2));
        auto V = Graph::tensor(
            TensorAttributes()
                .set_dim({s.batch, s.nhead_k, s.seqlen_k, s.hdim_v})
                .set_stride(
                    {s.nhead_k * s.seqlen_k * s.hdim_v, s.seqlen_k * s.hdim_v, s.hdim_v, 1})
                .set_uid(3));

        SdpaAttributes attrs;
        attrs.attn_scale_value = 1.0f / std::sqrt(static_cast<float>(s.hdim_q));

        auto [O, stats] = graph.sdpa(Q, K, V, std::move(attrs));
        O->set_output(true).set_uid(4);

        auto t0 = std::chrono::steady_clock::now();
        {
            StderrSilencer silencer;  // suppress hipRTC compile warnings
            HIPDNN_FE_CHECK(graph.build(handle));
        }
        auto t1 = std::chrono::steady_clock::now();
        r.build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        int64_t ws_bytes = 0;
        HIPDNN_FE_CHECK(graph.get_workspace_size(ws_bytes));

        const size_t q_bytes = s.q_count() * sizeof(__half);
        const size_t k_bytes = s.k_count() * sizeof(__half);
        const size_t v_bytes = s.v_count() * sizeof(__half);
        const size_t o_bytes = s.o_count() * sizeof(__half);

        void* d_q  = nullptr;
        void* d_k  = nullptr;
        void* d_v  = nullptr;
        void* d_o  = nullptr;
        void* d_ws = nullptr;
        HIP_CHECK(hipMalloc(&d_q, q_bytes));
        HIP_CHECK(hipMalloc(&d_k, k_bytes));
        HIP_CHECK(hipMalloc(&d_v, v_bytes));
        HIP_CHECK(hipMalloc(&d_o, o_bytes));
        if (ws_bytes > 0) HIP_CHECK(hipMalloc(&d_ws, ws_bytes));

        HIP_CHECK(hipMemcpy(d_q, q_host.data(), q_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_k, k_host.data(), k_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_v, v_host.data(), v_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_o, 0, o_bytes));

        std::unordered_map<std::shared_ptr<TensorAttributes>, void*> tensor_map = {
            {Q, d_q}, {K, d_k}, {V, d_v}, {O, d_o}};

        for (int i = 0; i < warmup; ++i) {
            HIPDNN_FE_CHECK(graph.execute(handle, tensor_map, d_ws));
        }
        HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start_ev, stop_ev;
        HIP_CHECK(hipEventCreate(&start_ev));
        HIP_CHECK(hipEventCreate(&stop_ev));
        HIP_CHECK(hipEventRecord(start_ev));
        for (int i = 0; i < repeat; ++i) {
            HIPDNN_FE_CHECK(graph.execute(handle, tensor_map, d_ws));
        }
        HIP_CHECK(hipEventRecord(stop_ev));
        HIP_CHECK(hipEventSynchronize(stop_ev));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start_ev, stop_ev));
        r.exec_ms = ms / repeat;

        r.o_host.resize(s.o_count());
        HIP_CHECK(hipMemcpy(r.o_host.data(), d_o, o_bytes, hipMemcpyDeviceToHost));

        hipEventDestroy(start_ev);
        hipEventDestroy(stop_ev);
        HIP_CHECK(hipFree(d_q));
        HIP_CHECK(hipFree(d_k));
        HIP_CHECK(hipFree(d_v));
        HIP_CHECK(hipFree(d_o));
        if (d_ws) HIP_CHECK(hipFree(d_ws));

        r.success = true;
    } catch (const std::exception& e) {
        r.error = e.what();
    }
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    int warmup = 2;
    int repeat = 5;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--repeat" && i + 1 < argc) repeat = std::atoi(argv[++i]);
    }

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    // Force the RTC backend (Phase 9+) for this demo; no Python/hipcc
    // subprocess involvement, no .so on disk. The plugin compiles each
    // shape in-process via hipRTC and registers the resulting kernel
    // directly into the dispatcher's FmhaRegistry.
    setenv("CK_FMHA_ENABLE_JIT", "1", /*overwrite=*/1);
    setenv("CK_FMHA_JIT_BACKEND", "rtc", /*overwrite=*/1);

    std::cout << "================================================================\n"
              << "  CK FMHA hipDNN plugin -- hipRTC end-to-end demo\n"
              << "  Device: " << prop.name << " (" << prop.gcnArchName << ")\n"
              << "  Warmup: " << warmup << "  Repeat: " << repeat << "\n"
              << "  JIT backend: hipRTC (CK_FMHA_JIT_BACKEND=rtc)\n"
              << "================================================================\n\n";

    const char* plugin_dir = std::getenv("HIPDNN_PLUGIN_PATH");
    if (plugin_dir != nullptr) {
        const char* paths[] = {plugin_dir};
        hipdnnSetEnginePluginPaths_ext(1, paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
        std::cout << "  Plugin path: " << plugin_dir << "\n\n";
    } else {
        std::cout << "  WARNING: HIPDNN_PLUGIN_PATH is not set.\n\n";
    }

    hipdnnHandle_t handle;
    hipdnnCreate(&handle);

    // Shapes exercise: MHA baseline, multi-hdim, and GQA (4:1 and
    // 8:1 ratios). Fields: {batch, nhead_q, nhead_k, Sq, Sk, Dq, Dv}.
    std::vector<Shape> shapes = {
        // MHA (nhead_q == nhead_k)
        {2, 4, 4, 128, 128, 128, 128},
        {2, 8, 8, 256, 256, 128, 128},
        {1, 8, 8, 512, 512, 128, 128},
        // GQA
        {2, 8, 2, 256, 256, 128, 128},  // 4:1 ratio, Llama-3-8B-ish
        {1, 16, 2, 512, 512, 128, 128}, // 8:1 ratio, 70B-class
    };

    auto table_header = [] {
        std::cout << std::left << std::setw(42) << "Shape" << std::setw(14) << "build(ms)"
                  << std::setw(12) << "exec(ms)" << std::setw(10) << "TFLOPS" << std::setw(10)
                  << "match?" << std::setw(12) << "max_abs" << std::setw(12) << "mean_abs"
                  << std::setw(12) << "ref_rms" << "\n";
        std::cout << std::string(124, '-') << "\n";
    };

    std::cout << std::fixed << std::setprecision(3);

    auto run_pass = [&](const std::string& title, int& passed, int& total) {
        std::cout << "=== " << title << " ===\n\n";
        table_header();
        for (const auto& s : shapes) {
            ++total;
            auto q_host = random_fp16(s.q_count(), 101);
            auto k_host = random_fp16(s.k_count(), 202);
            auto v_host = random_fp16(s.v_count(), 303);

            std::cout << std::setw(42) << s.label() << std::flush;

            auto r = run_fwd(handle, s, q_host, k_host, v_host, warmup, repeat);
            if (!r.success) {
                std::cout << "  SKIP: " << r.error << "\n";
                continue;
            }

            auto ref = cpu_attention_ref(q_host, k_host, v_host, s);
            auto m   = compare(r.o_host, ref);
            double tflops = r.exec_ms > 0 ? s.fwd_flops() / (r.exec_ms * 1e9) : 0.0;

            std::cout << std::setw(14) << r.build_ms << std::setw(12) << r.exec_ms << std::setw(10)
                      << tflops << std::setw(10) << (m.ok ? "OK" : "FAIL") << std::setw(12)
                      << m.max_abs << std::setw(12) << m.mean_abs << std::setw(12) << m.ref_rms
                      << "\n";
            if (m.ok) ++passed;
        }
        std::cout << "\n";
    };

    int total = 0, passed = 0;
    run_pass("Pass 1 (cold cache)", passed, total);
    run_pass("Pass 2 (hot cache, same shapes)", passed, total);

    std::cout << "================================================================\n"
              << "  Parity vs CPU reference: " << passed << "/" << total << " shapes match.\n"
              << "  Cold vs hot `build(ms)` demonstrates the RTC code-object cache:\n"
              << "  pass 1 recompiles, pass 2 deserializes the cached hipModule_t.\n"
              << "================================================================\n";

    hipdnnDestroy(handle);
    return passed == total ? 0 : 1;
}
