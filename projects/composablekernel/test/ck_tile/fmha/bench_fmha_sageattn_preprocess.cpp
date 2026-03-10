// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Benchmark for sageattn_preprocess_run() — the four-kernel SA3 preprocessing
// pipeline (k_mean, preprocess Q/K, V tile-transpose, delta_s GEMM).
//
// Usage:
//   ./bin/bench_ck_tile_fmha_sageattn_preprocess [options]
//
//   -b <int>    batch size                   (default 4)
//   -h <int>    number of heads              (default 16)
//   -q <int>    seqlen_q                     (default 1024)
//   -k <int>    seqlen_k                     (default 4096)
//   -d <int>    hdim (128 or 256)            (default 128)
//   -t <str>    input type: fp16 | fp32      (default fp16)
//   -w <int>    warmup iterations            (default 5)
//   -r <int>    measurement iterations       (default 50)
//   --csv       print header+row in CSV format
//
// HBM traffic counted (one-way, reads+writes):
//   Reads:   Q [B,H,Sq,D], K [B,H,Sk,D], V [B,H,Sk,D]
//   Writes:  Q_hat [B,H,Sq,D/2], Q_scale [B,H,Sq,D/G], q_mean [B,H,T_q,D]
//            K_hat [B,H,Sk,D/2], K_scale [B,H,Sk,D/G], K' [B,H,Sk,D]
//            V_hat [B,H,D,Sk/2], V_scale [B,H,D,Sk/G]
//            delta_s [B,H,T_q,Sk] (float32)
//   where G=32 (MXFP4 scale granularity), T_q = ceil(Sq/64).

#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CK_USE_NATIVE_MX_SUPPORT

#include "ck_tile/host.hpp"
#include "ck_tile/ops/sageattn_preprocess/sageattn_preprocess.hpp"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void hip_check(hipError_t err, const char* file, int line)
{
    if(err != hipSuccess)
    {
        std::ostringstream ss;
        ss << file << ":" << line << " HIP error: " << hipGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
}
#define HIP_CHECK(x) hip_check((x), __FILE__, __LINE__)

struct BenchArgs
{
    int         batch   = 4;
    int         nhead   = 16;
    int         seqlen_q = 1024;
    int         seqlen_k = 4096;
    int         hdim    = 128;
    std::string dtype   = "fp16";
    int         warmup  = 5;
    int         repeat  = 50;
    bool        csv     = false;
};

static BenchArgs parse_args(int argc, char** argv)
{
    BenchArgs a;
    for(int i = 1; i < argc; ++i)
    {
        std::string s = argv[i];
        if((s == "-b") && i + 1 < argc)      a.batch    = std::stoi(argv[++i]);
        else if((s == "-h") && i + 1 < argc) a.nhead    = std::stoi(argv[++i]);
        else if((s == "-q") && i + 1 < argc) a.seqlen_q = std::stoi(argv[++i]);
        else if((s == "-k") && i + 1 < argc) a.seqlen_k = std::stoi(argv[++i]);
        else if((s == "-d") && i + 1 < argc) a.hdim     = std::stoi(argv[++i]);
        else if((s == "-t") && i + 1 < argc) a.dtype    = argv[++i];
        else if((s == "-w") && i + 1 < argc) a.warmup   = std::stoi(argv[++i]);
        else if((s == "-r") && i + 1 < argc) a.repeat   = std::stoi(argv[++i]);
        else if(s == "--csv")                 a.csv      = true;
        else if(s == "--help" || s == "-?")
        {
            std::cout <<
                "Usage: bench_ck_tile_fmha_sageattn_preprocess [options]\n"
                "  -b <batch>    default 4\n"
                "  -h <nhead>    default 16\n"
                "  -q <seqlen_q> default 1024\n"
                "  -k <seqlen_k> default 4096\n"
                "  -d <hdim>     default 128 (128 or 256)\n"
                "  -t <dtype>    fp16 | fp32  (default fp16)\n"
                "  -w <warmup>   default 5\n"
                "  -r <repeat>   default 50\n"
                "  --csv         CSV output\n";
            std::exit(0);
        }
    }
    return a;
}

// ---------------------------------------------------------------------------
// Core benchmark: templated on InputT and kCols (hdim)
// ---------------------------------------------------------------------------

template <typename InputT, int kRows, int kCols>
float run_benchmark(const BenchArgs& a)
{
    const int b  = a.batch;
    const int h  = a.nhead;
    const int sq = a.seqlen_q;
    const int sk = a.seqlen_k;
    const int hd = a.hdim;

    constexpr int kG = 32; // MXFP4 scale granularity

    const int num_q_tiles = (sq + kRows - 1) / kRows;
    const int num_k_tiles = (sk + kRows - 1) / kRows;

    // ---- Allocate GPU buffers ----
    const std::size_t elem = sizeof(InputT);

    ck_tile::DeviceMem q_dev   (std::size_t(b * h * sq * hd) * elem);
    ck_tile::DeviceMem k_dev   (std::size_t(b * h * sk * hd) * elem);
    ck_tile::DeviceMem v_dev   (std::size_t(b * h * sk * hd) * elem);

    ck_tile::DeviceMem q_hat_dev   (std::size_t(b * h * sq * (hd / 2)));
    ck_tile::DeviceMem q_scale_dev (std::size_t(b * h * sq * (hd / kG)));
    ck_tile::DeviceMem q_mean_dev  (std::size_t(b * h * num_q_tiles * hd) * elem);

    ck_tile::DeviceMem k_hat_dev   (std::size_t(b * h * sk * (hd / 2)));
    ck_tile::DeviceMem k_scale_dev (std::size_t(b * h * sk * (hd / kG)));

    ck_tile::DeviceMem v_hat_dev   (std::size_t(b * h * hd * (sk / 2)));
    ck_tile::DeviceMem v_scale_dev (std::size_t(b * h * hd * (sk / kG)));

    ck_tile::DeviceMem delta_s_dev (std::size_t(b * h * num_q_tiles * sk) * sizeof(float));

    ck_tile::DeviceMem k_mean_buf        (std::size_t(b * h * hd) * elem);
    ck_tile::DeviceMem k_prime_buf       (std::size_t(b * h * sk * hd) * elem);
    ck_tile::DeviceMem k_mean_partial_buf(std::size_t(b * h * hd) * sizeof(float));
    ck_tile::DeviceMem counter_buf       (std::size_t(b * h) * sizeof(int32_t));

    // Initialise inputs with zeros (benchmark only; correctness tested elsewhere).
    q_dev.SetZero();
    k_dev.SetZero();
    v_dev.SetZero();

    // ---- Build host args ----
    ck_tile::SageAttnPreprocessHostArgs hargs{};

    hargs.q_ptr                = q_dev.GetDeviceBuffer();
    hargs.seqlen_q             = sq;
    hargs.hdim                 = hd;
    hargs.stride_q             = hd;
    hargs.nhead_stride_q       = sq * hd;
    hargs.batch_stride_q       = h * sq * hd;
    hargs.q_hat_ptr            = static_cast<uint8_t*>(q_hat_dev.GetDeviceBuffer());
    hargs.stride_q_hat         = hd / 2;
    hargs.nhead_stride_q_hat   = sq * (hd / 2);
    hargs.batch_stride_q_hat   = h * sq * (hd / 2);
    hargs.q_scale_ptr          = static_cast<uint8_t*>(q_scale_dev.GetDeviceBuffer());
    hargs.stride_q_scale       = hd / kG;
    hargs.nhead_stride_q_scale = sq * (hd / kG);
    hargs.batch_stride_q_scale = h * sq * (hd / kG);
    hargs.q_mean_ptr           = q_mean_dev.GetDeviceBuffer();
    hargs.q_tile_size          = kRows;
    hargs.stride_q_mean        = hd;
    hargs.nhead_stride_q_mean  = num_q_tiles * hd;
    hargs.batch_stride_q_mean  = h * num_q_tiles * hd;

    hargs.k_ptr                = k_dev.GetDeviceBuffer();
    hargs.seqlen_k             = sk;
    hargs.stride_k             = hd;
    hargs.nhead_stride_k       = sk * hd;
    hargs.batch_stride_k       = h * sk * hd;
    hargs.k_hat_ptr            = static_cast<uint8_t*>(k_hat_dev.GetDeviceBuffer());
    hargs.stride_k_hat         = hd / 2;
    hargs.nhead_stride_k_hat   = sk * (hd / 2);
    hargs.batch_stride_k_hat   = h * sk * (hd / 2);
    hargs.k_scale_ptr          = static_cast<uint8_t*>(k_scale_dev.GetDeviceBuffer());
    hargs.stride_k_scale       = hd / kG;
    hargs.nhead_stride_k_scale = sk * (hd / kG);
    hargs.batch_stride_k_scale = h * sk * (hd / kG);

    hargs.v_ptr                = v_dev.GetDeviceBuffer();
    hargs.nhead_stride_v       = sk * hd;
    hargs.batch_stride_v       = h * sk * hd;
    hargs.v_hat_ptr            = static_cast<uint8_t*>(v_hat_dev.GetDeviceBuffer());
    hargs.stride_v_hat         = sk / 2;
    hargs.nhead_stride_v_hat   = hd * (sk / 2);
    hargs.batch_stride_v_hat   = h * hd * (sk / 2);
    hargs.v_scale_ptr          = static_cast<uint8_t*>(v_scale_dev.GetDeviceBuffer());
    hargs.stride_v_scale       = sk / kG;
    hargs.nhead_stride_v_scale = hd * (sk / kG);
    hargs.batch_stride_v_scale = h * hd * (sk / kG);

    hargs.batch       = b;
    hargs.nhead       = h;
    hargs.num_q_tiles = num_q_tiles;
    hargs.num_k_tiles = num_k_tiles;

    auto* k_mean_ptr   = static_cast<InputT*>(k_mean_buf.GetDeviceBuffer());
    auto* k_prime_ptr  = static_cast<InputT*>(k_prime_buf.GetDeviceBuffer());
    auto* k_partial_ptr = static_cast<float*>(k_mean_partial_buf.GetDeviceBuffer());
    auto* counter_ptr  = static_cast<int32_t*>(counter_buf.GetDeviceBuffer());
    auto* delta_s_ptr  = static_cast<float*>(delta_s_dev.GetDeviceBuffer());

    // ---- Timing setup ----
    hipStream_t stream = nullptr;
    hipEvent_t  ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    auto run_once = [&]() {
        ck_tile::sageattn_preprocess_run<InputT, kRows, kCols>(
            hargs, delta_s_ptr, k_mean_ptr, k_prime_ptr, k_partial_ptr, counter_ptr, stream);
    };

    // Warmup
    for(int i = 0; i < a.warmup; ++i)
        run_once();
    HIP_CHECK(hipStreamSynchronize(stream));

    // Timed loop
    HIP_CHECK(hipEventRecord(ev_start, stream));
    for(int i = 0; i < a.repeat; ++i)
        run_once();
    HIP_CHECK(hipEventRecord(ev_stop, stream));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, ev_start, ev_stop));

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));

    return total_ms / static_cast<float>(a.repeat);
}

// ---------------------------------------------------------------------------
// Print one benchmark result row
// ---------------------------------------------------------------------------

static void print_result(const BenchArgs& a, float ave_ms, bool csv_mode)
{
    const int b  = a.batch;
    const int h  = a.nhead;
    const int sq = a.seqlen_q;
    const int sk = a.seqlen_k;
    const int hd = a.hdim;
    constexpr int kG   = 32;
    constexpr int kM0  = 64;
    const int num_q_tiles = (sq + kM0 - 1) / kM0;

    const std::size_t elem_bytes =
        (a.dtype == "fp16") ? sizeof(ck_tile::fp16_t) : sizeof(float);

    // ---- HBM bytes ----
    // Reads
    const std::size_t read_Q        = std::size_t(b) * h * sq * hd * elem_bytes;
    const std::size_t read_K        = std::size_t(b) * h * sk * hd * elem_bytes;
    const std::size_t read_V        = std::size_t(b) * h * sk * hd * elem_bytes;
    // Writes
    const std::size_t write_Q_hat   = std::size_t(b) * h * sq * (hd / 2);           // uint8
    const std::size_t write_Q_scale = std::size_t(b) * h * sq * (hd / kG);          // uint8
    const std::size_t write_Q_mean  = std::size_t(b) * h * num_q_tiles * hd * elem_bytes;
    const std::size_t write_K_hat   = std::size_t(b) * h * sk * (hd / 2);
    const std::size_t write_K_scale = std::size_t(b) * h * sk * (hd / kG);
    const std::size_t write_K_prime = std::size_t(b) * h * sk * hd * elem_bytes;    // K'
    const std::size_t write_V_hat   = std::size_t(b) * h * hd * (sk / 2);
    const std::size_t write_V_scale = std::size_t(b) * h * hd * (sk / kG);
    const std::size_t write_delta_s = std::size_t(b) * h * num_q_tiles * sk * sizeof(float);

    const std::size_t total_bytes =
        read_Q + read_K + read_V +
        write_Q_hat + write_Q_scale + write_Q_mean +
        write_K_hat + write_K_scale + write_K_prime +
        write_V_hat + write_V_scale +
        write_delta_s;

    const double gb_per_s = static_cast<double>(total_bytes) / 1.0e6 / ave_ms;

    if(csv_mode)
    {
        std::cout
            << a.dtype << ","
            << b << "," << h << "," << sq << "," << sk << "," << hd << ","
            << std::fixed << std::setprecision(3) << ave_ms << ","
            << std::setprecision(1) << gb_per_s << "\n";
    }
    else
    {
        std::cout
            << "dtype=" << a.dtype
            << "  B=" << b << " H=" << h
            << " Sq=" << sq << " Sk=" << sk << " D=" << hd
            << "  |  " << std::fixed << std::setprecision(3) << ave_ms << " ms"
            << "  " << std::setprecision(1) << gb_per_s << " GB/s"
            << "  (total HBM " << std::setprecision(0) << total_bytes / 1.0e6 << " MB)\n";
    }
}

// ---------------------------------------------------------------------------
// Dispatch: (dtype, hdim) → template instantiation
// ---------------------------------------------------------------------------

static float dispatch(const BenchArgs& a)
{
    if(a.dtype == "fp16")
    {
        if(a.hdim == 128)
            return run_benchmark<ck_tile::fp16_t, 64, 128>(a);
        if(a.hdim == 256)
            return run_benchmark<ck_tile::fp16_t, 64, 256>(a);
    }
    else if(a.dtype == "fp32")
    {
        if(a.hdim == 128)
            return run_benchmark<float, 64, 128>(a);
        if(a.hdim == 256)
            return run_benchmark<float, 64, 256>(a);
    }
    throw std::runtime_error("Unsupported dtype/hdim: " + a.dtype +
                             " d=" + std::to_string(a.hdim));
}

// ---------------------------------------------------------------------------
// Benchmark suite: a representative set of large-batch shapes
// ---------------------------------------------------------------------------

static const std::vector<std::tuple<int,int,int,int,int>> kShapes = {
    // {batch, nhead, seqlen_q, seqlen_k, hdim}
    // Realistic LLM decoding: small seqlen_q, large seqlen_k
    { 1,  32, 1,    4096,  128},
    { 1,  32, 1,    4096,  256},
    { 1,  32, 1,    8192,  128},
    { 1,  32, 1,    8192,  256},
    // Prefill: large seqlen_q and seqlen_k
    { 1,  32, 1024, 1024,  128},
    { 1,  32, 1024, 4096,  128},
    { 1,  32, 4096, 4096,  128},
    { 1,  32, 4096, 4096,  256},
    // Multi-batch
    { 4,  16, 1024, 4096,  128},
    { 4,  16, 4096, 4096,  128},
    { 8,   8, 2048, 2048,  128},
    {16,  32, 512,  4096,  128},
    {16,  32, 1024, 4096,  128},
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    BenchArgs a = parse_args(argc, argv);

    // Detect whether this is a suite run (no explicit shape args) or single run.
    // Heuristic: if all shape defaults are unchanged AND no -b/-h/-q/-k/-d flags,
    // we run the built-in suite; otherwise run exactly what was specified.
    bool explicit_shape = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string s = argv[i];
        if(s == "-b" || s == "-h" || s == "-q" || s == "-k" || s == "-d" || s == "-t")
            explicit_shape = true;
    }

    if(a.csv)
        std::cout << "dtype,batch,nhead,seqlen_q,seqlen_k,hdim,ms,GB_per_s\n";

    if(explicit_shape)
    {
        float ms = dispatch(a);
        print_result(a, ms, a.csv);
    }
    else
    {
        // Run suite for both dtypes and both hdims.
        for(const std::string& dtype : {"fp16", "fp32"})
        {
            for(auto [b, h, sq, sk, hd] : kShapes)
            {
                BenchArgs cur = a;
                cur.batch    = b;
                cur.nhead    = h;
                cur.seqlen_q = sq;
                cur.seqlen_k = sk;
                cur.hdim     = hd;
                cur.dtype    = dtype;
                float ms = dispatch(cur);
                print_result(cur, ms, a.csv);
            }
        }
    }

    return 0;
}

#else // CK_USE_NATIVE_MX_SUPPORT not defined

#include <cstdio>
int main()
{
    puts("sageattn_preprocess benchmark requires CK_USE_NATIVE_MX_SUPPORT (gfx950).");
    return 0;
}

#endif
