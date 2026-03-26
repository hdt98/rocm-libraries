// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// End-to-end demo: hipDNN Graph API → CK FMHA plugin → TFLOPS
//
// This exercises the FULL production path:
//   1. Construct hipDNN Graph with SdpaAttributes
//   2. graph.build(handle) → plugin discovered, applicability checked, plan built
//   3. graph.execute(handle, variantPack, workspace) → CK kernel launched
//   4. Measure wall-clock time and report TFLOPS
//
// Build:
//   cmake --build . --target ck_fmha_e2e_demo
//
// Run:
//   ./ck_fmha_e2e_demo [--warmup 5] [--repeat 20] [--bwd]

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <hipdnn_frontend.hpp>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
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

#define HIPDNN_FE_CHECK(err)                                                                  \
    do {                                                                                      \
        auto _e = (err);                                                                      \
        if (_e.get_code() != hipdnn_frontend::ErrorCode::OK) {                                \
            std::cerr << "hipDNN FE error: " << _e.get_message() << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                               \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace {

struct Shape {
    int batch, nhead_q, nhead_k, seqlen_q, seqlen_k, hdim_q, hdim_v;

    uint64_t fwd_flops() const {
        return 2ULL * batch * nhead_q * seqlen_q * seqlen_k * (hdim_q + hdim_v);
    }
    uint64_t bwd_flops() const {
        return 4ULL * fwd_flops();
    }

    std::string label() const {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "B=%d Hq=%d Hk=%d Sq=%d Sk=%d Dq=%d Dv=%d", batch, nhead_q,
                      nhead_k, seqlen_q, seqlen_k, hdim_q, hdim_v);
        return buf;
    }
};

std::vector<__half> random_fp16(size_t n, float scale = 0.02f) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<__half> v(n);
    for (auto& x : v) x = __float2half(dist(rng));
    return v;
}

void* hip_alloc(size_t bytes) {
    void* p = nullptr;
    HIP_CHECK(hipMalloc(&p, bytes));
    return p;
}

void hip_upload(void* dst, const void* src, size_t bytes) {
    HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
}

// --------------------------------------------------------------------------
// Forward SDPA through hipDNN Graph API
// --------------------------------------------------------------------------

double run_fwd_sdpa(hipdnnHandle_t handle, const Shape& s, bool with_stats, int warmup,
                    int repeat) {
    Graph graph;
    graph.set_io_data_type(DataType::HALF)
        .set_compute_data_type(DataType::FLOAT)
        .set_name("sdpa_fwd_bench");

    auto Q = Graph::tensor(
        TensorAttributes()
            .set_dim({s.batch, s.nhead_q, s.seqlen_q, s.hdim_q})
            .set_stride({s.nhead_q * s.seqlen_q * s.hdim_q, s.seqlen_q * s.hdim_q, s.hdim_q, 1})
            .set_uid(1));

    auto K = Graph::tensor(
        TensorAttributes()
            .set_dim({s.batch, s.nhead_k, s.seqlen_k, s.hdim_q})
            .set_stride({s.nhead_k * s.seqlen_k * s.hdim_q, s.seqlen_k * s.hdim_q, s.hdim_q, 1})
            .set_uid(2));

    auto V = Graph::tensor(
        TensorAttributes()
            .set_dim({s.batch, s.nhead_k, s.seqlen_k, s.hdim_v})
            .set_stride({s.nhead_k * s.seqlen_k * s.hdim_v, s.seqlen_k * s.hdim_v, s.hdim_v, 1})
            .set_uid(3));

    SdpaAttributes sdpa_attrs;
    sdpa_attrs.attn_scale_value = 1.0f / std::sqrt(static_cast<float>(s.hdim_q));
    if (with_stats) sdpa_attrs.generate_stats = true;

    auto [O, stats] = graph.sdpa(Q, K, V, std::move(sdpa_attrs));
    O->set_output(true).set_uid(4);
    if (stats) stats->set_output(true).set_uid(5);

    HIPDNN_FE_CHECK(graph.build(handle));

    int64_t ws_size = 0;
    HIPDNN_FE_CHECK(graph.get_workspace_size(ws_size));

    // Allocate GPU tensors
    size_t q_bytes = s.batch * s.nhead_q * s.seqlen_q * s.hdim_q * sizeof(__half);
    size_t k_bytes = s.batch * s.nhead_k * s.seqlen_k * s.hdim_q * sizeof(__half);
    size_t v_bytes = s.batch * s.nhead_k * s.seqlen_k * s.hdim_v * sizeof(__half);
    size_t o_bytes = s.batch * s.nhead_q * s.seqlen_q * s.hdim_v * sizeof(__half);

    auto q_host = random_fp16(q_bytes / sizeof(__half));
    auto k_host = random_fp16(k_bytes / sizeof(__half));
    auto v_host = random_fp16(v_bytes / sizeof(__half));

    void* d_q = hip_alloc(q_bytes);
    void* d_k = hip_alloc(k_bytes);
    void* d_v = hip_alloc(v_bytes);
    void* d_o = hip_alloc(o_bytes);
    void* d_ws = ws_size > 0 ? hip_alloc(ws_size) : nullptr;
    void* d_stats = nullptr;
    if (with_stats) d_stats = hip_alloc(s.batch * s.nhead_q * s.seqlen_q * sizeof(float));

    hip_upload(d_q, q_host.data(), q_bytes);
    hip_upload(d_k, k_host.data(), k_bytes);
    hip_upload(d_v, v_host.data(), v_bytes);

    std::unordered_map<std::shared_ptr<TensorAttributes>, void*> tensor_map = {
        {Q, d_q}, {K, d_k}, {V, d_v}, {O, d_o}};
    if (stats) tensor_map[stats] = d_stats;

    // Warmup
    for (int i = 0; i < warmup; ++i) HIPDNN_FE_CHECK(graph.execute(handle, tensor_map, d_ws));
    HIP_CHECK(hipDeviceSynchronize());

    // Timed runs
    hipEvent_t start_ev, stop_ev;
    HIP_CHECK(hipEventCreate(&start_ev));
    HIP_CHECK(hipEventCreate(&stop_ev));
    HIP_CHECK(hipEventRecord(start_ev));

    for (int i = 0; i < repeat; ++i) HIPDNN_FE_CHECK(graph.execute(handle, tensor_map, d_ws));

    HIP_CHECK(hipEventRecord(stop_ev));
    HIP_CHECK(hipEventSynchronize(stop_ev));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start_ev, stop_ev));
    double avg_ms = ms / repeat;

    // Cleanup
    hipFree(d_q);
    hipFree(d_k);
    hipFree(d_v);
    hipFree(d_o);
    if (d_ws) hipFree(d_ws);
    if (d_stats) hipFree(d_stats);
    hipEventDestroy(start_ev);
    hipEventDestroy(stop_ev);

    return avg_ms;
}

// --------------------------------------------------------------------------
// Backward SDPA through hipDNN Graph API
// --------------------------------------------------------------------------

double run_bwd_sdpa(hipdnnHandle_t handle, const Shape& s, int warmup, int repeat) {
    // Forward pass with stats (required for backward)
    Graph fwd_graph;
    fwd_graph.set_io_data_type(DataType::HALF)
        .set_compute_data_type(DataType::FLOAT)
        .set_name("sdpa_fwd_for_bwd");

    auto fQ = Graph::tensor(
        TensorAttributes()
            .set_dim({s.batch, s.nhead_q, s.seqlen_q, s.hdim_q})
            .set_stride({s.nhead_q * s.seqlen_q * s.hdim_q, s.seqlen_q * s.hdim_q, s.hdim_q, 1})
            .set_uid(1));
    auto fK = Graph::tensor(
        TensorAttributes()
            .set_dim({s.batch, s.nhead_k, s.seqlen_k, s.hdim_q})
            .set_stride({s.nhead_k * s.seqlen_k * s.hdim_q, s.seqlen_k * s.hdim_q, s.hdim_q, 1})
            .set_uid(2));
    auto fV = Graph::tensor(
        TensorAttributes()
            .set_dim({s.batch, s.nhead_k, s.seqlen_k, s.hdim_v})
            .set_stride({s.nhead_k * s.seqlen_k * s.hdim_v, s.seqlen_k * s.hdim_v, s.hdim_v, 1})
            .set_uid(3));

    SdpaAttributes fwd_attrs;
    fwd_attrs.attn_scale_value = 1.0f / std::sqrt(static_cast<float>(s.hdim_q));
    fwd_attrs.generate_stats = true;

    auto [fO, fStats] = fwd_graph.sdpa(fQ, fK, fV, std::move(fwd_attrs));
    fO->set_output(true).set_uid(4);
    fStats->set_output(true).set_uid(5);

    HIPDNN_FE_CHECK(fwd_graph.build(handle));

    // Backward graph
    Graph bwd_graph;
    bwd_graph.set_io_data_type(DataType::HALF)
        .set_compute_data_type(DataType::FLOAT)
        .set_name("sdpa_bwd_bench");

    auto bQ = Graph::tensor(
        TensorAttributes()
            .set_dim({s.batch, s.nhead_q, s.seqlen_q, s.hdim_q})
            .set_stride({s.nhead_q * s.seqlen_q * s.hdim_q, s.seqlen_q * s.hdim_q, s.hdim_q, 1})
            .set_uid(1));
    auto bK = Graph::tensor(
        TensorAttributes()
            .set_dim({s.batch, s.nhead_k, s.seqlen_k, s.hdim_q})
            .set_stride({s.nhead_k * s.seqlen_k * s.hdim_q, s.seqlen_k * s.hdim_q, s.hdim_q, 1})
            .set_uid(2));
    auto bV = Graph::tensor(
        TensorAttributes()
            .set_dim({s.batch, s.nhead_k, s.seqlen_k, s.hdim_v})
            .set_stride({s.nhead_k * s.seqlen_k * s.hdim_v, s.seqlen_k * s.hdim_v, s.hdim_v, 1})
            .set_uid(3));
    auto bO = Graph::tensor(
        TensorAttributes()
            .set_dim({s.batch, s.nhead_q, s.seqlen_q, s.hdim_v})
            .set_stride({s.nhead_q * s.seqlen_q * s.hdim_v, s.seqlen_q * s.hdim_v, s.hdim_v, 1})
            .set_uid(4));
    auto bdO = Graph::tensor(
        TensorAttributes()
            .set_dim({s.batch, s.nhead_q, s.seqlen_q, s.hdim_v})
            .set_stride({s.nhead_q * s.seqlen_q * s.hdim_v, s.seqlen_q * s.hdim_v, s.hdim_v, 1})
            .set_uid(6));
    auto bStats = Graph::tensor(TensorAttributes()
                                    .set_dim({s.batch, s.nhead_q, s.seqlen_q, 1})
                                    .set_stride({s.nhead_q * s.seqlen_q, s.seqlen_q, 1, 1})
                                    .set_data_type(DataType::FLOAT)
                                    .set_uid(5));

    SdpaBackwardAttributes bwd_attrs;
    bwd_attrs.attn_scale_value = 1.0f / std::sqrt(static_cast<float>(s.hdim_q));

    auto [dQ, dK, dV] = bwd_graph.sdpa_backward(bQ, bK, bV, bO, bdO, bStats, std::move(bwd_attrs));
    dQ->set_output(true).set_uid(7);
    dK->set_output(true).set_uid(8);
    dV->set_output(true).set_uid(9);

    HIPDNN_FE_CHECK(bwd_graph.build(handle));

    int64_t fwd_ws_size = 0, bwd_ws_size = 0;
    HIPDNN_FE_CHECK(fwd_graph.get_workspace_size(fwd_ws_size));
    HIPDNN_FE_CHECK(bwd_graph.get_workspace_size(bwd_ws_size));

    // Allocate all GPU tensors
    size_t q_elems = s.batch * s.nhead_q * s.seqlen_q * s.hdim_q;
    size_t k_elems = s.batch * s.nhead_k * s.seqlen_k * s.hdim_q;
    size_t v_elems = s.batch * s.nhead_k * s.seqlen_k * s.hdim_v;
    size_t o_elems = s.batch * s.nhead_q * s.seqlen_q * s.hdim_v;
    size_t stats_elems = s.batch * s.nhead_q * s.seqlen_q;

    auto q_host = random_fp16(q_elems);
    auto k_host = random_fp16(k_elems);
    auto v_host = random_fp16(v_elems);
    auto do_host = random_fp16(o_elems);

    void* d_q = hip_alloc(q_elems * 2);
    void* d_k = hip_alloc(k_elems * 2);
    void* d_v = hip_alloc(v_elems * 2);
    void* d_o = hip_alloc(o_elems * 2);
    void* d_do = hip_alloc(o_elems * 2);
    void* d_stats = hip_alloc(stats_elems * 4);
    void* d_dq = hip_alloc(q_elems * 2);
    void* d_dk = hip_alloc(k_elems * 2);
    void* d_dv = hip_alloc(v_elems * 2);
    void* fwd_ws = fwd_ws_size > 0 ? hip_alloc(fwd_ws_size) : nullptr;
    void* bwd_ws = bwd_ws_size > 0 ? hip_alloc(bwd_ws_size) : nullptr;

    hip_upload(d_q, q_host.data(), q_elems * 2);
    hip_upload(d_k, k_host.data(), k_elems * 2);
    hip_upload(d_v, v_host.data(), v_elems * 2);
    hip_upload(d_do, do_host.data(), o_elems * 2);

    // Run forward to populate O and stats
    std::unordered_map<std::shared_ptr<TensorAttributes>, void*> fwd_map = {
        {fQ, d_q}, {fK, d_k}, {fV, d_v}, {fO, d_o}, {fStats, d_stats}};
    HIPDNN_FE_CHECK(fwd_graph.execute(handle, fwd_map, fwd_ws));
    HIP_CHECK(hipDeviceSynchronize());

    // Backward timing
    std::unordered_map<std::shared_ptr<TensorAttributes>, void*> bwd_map = {
        {bQ, d_q},   {bK, d_k},  {bV, d_v},  {bO, d_o}, {bStats, d_stats},
        {bdO, d_do}, {dQ, d_dq}, {dK, d_dk}, {dV, d_dv}};

    for (int i = 0; i < warmup; ++i) HIPDNN_FE_CHECK(bwd_graph.execute(handle, bwd_map, bwd_ws));
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start_ev, stop_ev;
    HIP_CHECK(hipEventCreate(&start_ev));
    HIP_CHECK(hipEventCreate(&stop_ev));
    HIP_CHECK(hipEventRecord(start_ev));

    for (int i = 0; i < repeat; ++i) HIPDNN_FE_CHECK(bwd_graph.execute(handle, bwd_map, bwd_ws));

    HIP_CHECK(hipEventRecord(stop_ev));
    HIP_CHECK(hipEventSynchronize(stop_ev));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start_ev, stop_ev));
    double avg_ms = ms / repeat;

    hipFree(d_q);
    hipFree(d_k);
    hipFree(d_v);
    hipFree(d_o);
    hipFree(d_do);
    hipFree(d_stats);
    hipFree(d_dq);
    hipFree(d_dk);
    hipFree(d_dv);
    if (fwd_ws) hipFree(fwd_ws);
    if (bwd_ws) hipFree(bwd_ws);
    hipEventDestroy(start_ev);
    hipEventDestroy(stop_ev);

    return avg_ms;
}

}  // namespace

int main(int argc, char** argv) {
    int warmup = 5;
    int repeat = 20;
    bool run_bwd = false;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--warmup" && i + 1 < argc)
            warmup = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--repeat" && i + 1 < argc)
            repeat = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--bwd")
            run_bwd = true;
    }

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    std::cout << "============================================================\n"
              << "  CK FMHA hipDNN Plugin End-to-End Benchmark\n"
              << "  Device: " << prop.name << " (" << prop.gcnArchName << ")\n"
              << "  Warmup: " << warmup << "  Repeat: " << repeat << "\n"
              << "============================================================\n\n";

    const char* plugin_dir = std::getenv("HIPDNN_PLUGIN_PATH");
    if (plugin_dir != nullptr) {
        const char* paths[] = {plugin_dir};
        hipdnnSetEnginePluginPaths_ext(1, paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
        std::cout << "  Plugin path: " << plugin_dir << "\n";
    } else {
        std::cout << "  WARNING: Set HIPDNN_PLUGIN_PATH to the directory containing\n"
                  << "  ck_fmha_provider_plugin.so\n";
    }

    hipdnnHandle_t handle;
    hipdnnCreate(&handle);
    std::cout << std::endl;

    // Shapes from ck_fmha_testing_matrix.yaml (smoke tier)
    std::vector<Shape> shapes = {
        // Basic MHA matching minimal kernel config
        {2, 4, 4, 128, 128, 128, 128},
        // GQA_4to1_Prefill_Basic: Llama-3-8B baseline
        {1, 32, 8, 2048, 2048, 128, 128},
        {4, 32, 8, 2048, 2048, 128, 128},
        // GQA_16to1_Large: 70B-class models
        {1, 64, 4, 2048, 2048, 128, 128},  // ~811 TFLOPS
        {4, 64, 4, 2048, 2048, 128, 128},  // ~730 TFLOPS
        // Small_GQA_7to1_SubWarp
        {1, 14, 2, 1024, 1024, 64, 64},  // ~156 TFLOPS
        // CK_All_Hdim_Sweep
        {2, 8, 4, 1024, 1024, 32, 32},    // ~116 TFLOPS
        {2, 8, 4, 1024, 1024, 64, 64},    // ~179 TFLOPS
        {2, 8, 4, 1024, 1024, 128, 128},  // ~278 TFLOPS
        {2, 8, 4, 1024, 1024, 256, 256},  // ~308 TFLOPS
        // MLA_Sparse_Decode: R1-class asymmetric hdim
        {1, 128, 128, 1, 1024, 192, 128},
        {4, 128, 128, 1, 4096, 192, 128},
        // MQA_128to8_Decode: 405B-class decode
        {8, 128, 8, 1, 1024, 128, 128},   // ~8 TFLOPS
        {64, 128, 8, 1, 1024, 128, 128},  // ~14 TFLOPS
        // Extreme_GQA_Ratios
        {2, 48, 8, 1024, 1024, 128, 128},
        {2, 24, 4, 1024, 1024, 128, 128},
        // Prefill_Odd_Lengths
        {2, 32, 8, 113, 203, 128, 128},
        {2, 32, 8, 799, 799, 128, 128},
        {2, 32, 8, 3131, 3131, 128, 128},
        // CK_Tiny_Sequences: edge cases
        {1, 2, 1, 1, 10, 128, 128},
        {2, 2, 1, 33, 99, 128, 128},
    };

    // Forward benchmark
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== Forward SDPA (fp16, BHSD, no mask) ===\n\n";
    std::cout << std::left << std::setw(50) << "Shape" << std::setw(12) << "Time(ms)"
              << std::setw(12) << "TFLOPS" << "\n"
              << std::string(74, '-') << "\n";

    for (const auto& s : shapes) {
        try {
            double avg_ms = run_fwd_sdpa(handle, s, false, warmup, repeat);
            double tflops = s.fwd_flops() / (avg_ms * 1e9);
            std::cout << std::setw(50) << s.label() << std::setw(12) << avg_ms << std::setw(12)
                      << tflops << "\n";
        } catch (const std::exception& e) {
            std::cout << std::setw(50) << s.label() << "  SKIP: " << e.what() << "\n";
        }
    }

    if (run_bwd) {
        std::cout << "\n=== Backward SDPA (fp16, BHSD, no mask) ===\n\n";
        std::cout << std::left << std::setw(50) << "Shape" << std::setw(12) << "Time(ms)"
                  << std::setw(12) << "TFLOPS" << "\n"
                  << std::string(74, '-') << "\n";

        for (const auto& s : shapes) {
            if (s.seqlen_q == 1) continue;  // decode shapes don't do backward
            try {
                double avg_ms = run_bwd_sdpa(handle, s, warmup, repeat);
                double tflops = s.bwd_flops() / (avg_ms * 1e9);
                std::cout << std::setw(50) << s.label() << std::setw(12) << avg_ms << std::setw(12)
                          << tflops << "\n";
            } catch (const std::exception& e) {
                std::cout << std::setw(50) << s.label() << "  SKIP: " << e.what() << "\n";
            }
        }
    }

    hipdnnDestroy(handle);

    std::cout << "\n============================================================\n"
              << "  Done.\n"
              << "============================================================\n";
    return 0;
}
