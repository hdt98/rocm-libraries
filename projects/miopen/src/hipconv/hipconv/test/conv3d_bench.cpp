// Benchmark for hipconv 3D direct convolution kernels.
//
// Reports kernel time (ms), bandwidth (GB/s), and throughput (TFLOP/s)
// for the target inference shapes.
//
// Usage: ./conv3d_bench [warmup_iters] [timed_iters]
//   Defaults: warmup=5, timed=100

#include "conv3d/conv3d.hpp"
#include "hipconv/conv3d_params.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

#define HIP_CHECK(expr)                                                          \
    do {                                                                         \
        hipError_t _e = (expr);                                                  \
        if(_e != hipSuccess)                                                     \
        {                                                                        \
            fprintf(stderr, "HIP error %s at %s:%d\n",                          \
                    hipGetErrorString(_e), __FILE__, __LINE__);                  \
            std::exit(1);                                                        \
        }                                                                        \
    } while(0)

// Bytes read + written for one forward conv (approximate roofline lower bound).
// Reads: input + weights; Writes: output.  All fp16.
static double bandwidth_bytes(const hipconv::Conv3dParams& p)
{
    double in_bytes  = (double)p.n * p.id * p.ih * p.iw * p.c * 2;
    double wei_bytes = (double)p.k * p.kd * p.kh * p.kw * p.c * 2;
    double out_bytes = (double)p.n * p.od * p.oh * p.ow * p.k * 2;
    return in_bytes + wei_bytes + out_bytes;
}

// Total multiply-add FLOPs (2 per MAC).
static double flops(const hipconv::Conv3dParams& p)
{
    return 2.0 * p.n * p.od * p.oh * p.ow
               * p.kd * p.kh * p.kw
               * p.c  * p.k;
}

// ---------------------------------------------------------------------------
// Benchmark one shape
// ---------------------------------------------------------------------------
static void run_bench(const char* label,
                      hipconv::Conv3dParams p,
                      int warmup,
                      int iters)
{
    p.compute_output_size();

    // Find kernel config.
    auto cfgs = conv3d::get_valid_configs(p);
    if(cfgs.empty())
    {
        printf("%-36s  SKIP (no valid config)\n", label);
        return;
    }
    auto cfg = cfgs[0]; // best config (first in priority order)

    // Allocate device buffers with dummy data (zeros — correct for timing).
    size_t in_elems  = (size_t)p.n * p.id * p.ih * p.iw * p.c;
    size_t wei_elems = (size_t)p.k * p.kd * p.kh * p.kw * p.c;
    size_t out_elems = (size_t)p.n * p.od * p.oh * p.ow * p.k;

    void *in_d, *wei_d, *out_d;
    HIP_CHECK(hipMalloc(&in_d,  in_elems  * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&wei_d, wei_elems * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&out_d, out_elems * sizeof(_Float16)));
    HIP_CHECK(hipMemset(in_d,   0, in_elems  * sizeof(_Float16)));
    HIP_CHECK(hipMemset(wei_d,  0, wei_elems * sizeof(_Float16)));
    HIP_CHECK(hipMemset(out_d,  0, out_elems * sizeof(_Float16)));

    // Warmup.
    for(int i = 0; i < warmup; ++i)
        conv3d::launch(cfg, p, in_d, wei_d, out_d, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    // Timed iterations with HIP events.
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    HIP_CHECK(hipEventRecord(ev_start));
    for(int i = 0; i < iters; ++i)
        conv3d::launch(cfg, p, in_d, wei_d, out_d, nullptr);
    HIP_CHECK(hipEventRecord(ev_stop));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float elapsed_ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
    HIP_CHECK(hipFree(in_d));
    HIP_CHECK(hipFree(wei_d));
    HIP_CHECK(hipFree(out_d));

    double ms_per_iter  = elapsed_ms / iters;
    double gbps         = bandwidth_bytes(p) / (ms_per_iter * 1e-3) / 1e9;
    double tflops       = flops(p)           / (ms_per_iter * 1e-3) / 1e12;

    printf("%-36s  %6.3f ms   %7.2f GB/s   %6.3f TFLOP/s"
           "   [N=%d Di/Hi/Wi=%d/%d/%d C=%d K=%d"
           " Od/Oh/Ow=%d/%d/%d pad=%d/%d/%d]\n",
           label, ms_per_iter, gbps, tflops,
           p.n, p.id, p.ih, p.iw, p.c, p.k,
           p.od, p.oh, p.ow, p.pad_d, p.pad_h, p.pad_w);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int warmup = 3;
    int iters  = 20;
    if(argc > 1) warmup = atoi(argv[1]);
    if(argc > 2) iters  = atoi(argv[2]);

    // Print device name.
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Warmup iterations: %d  Timed iterations: %d\n\n", warmup, iters);

    printf("%-36s  %9s   %13s   %15s\n",
           "Shape", "Time", "Bandwidth", "Throughput");
    printf("%s\n", std::string(90, '-').c_str());

    // -----------------------------------------------------------------------
    // Target shapes
    // -----------------------------------------------------------------------

    // C=3, K=96: Di=6, Hi=1106, Wi=834, no padding
    run_bench("C=3/K=96  6x1106x834 no_pad",
              {.n=1, .id=6, .ih=1106, .iw=834, .c=3, .k=96,
               .pad_d=0, .pad_h=0, .pad_w=0},
              warmup, iters);

    // C=96, K=3: Di=6, Hi=97, Wi=832, pad_d=0 pad_h=pad_w=1
    run_bench("C=96/K=3  6x97x832  pad_011",
              {.n=1, .id=6, .ih=97, .iw=832, .c=96, .k=3,
               .pad_d=0, .pad_h=1, .pad_w=1},
              warmup, iters);

    // -----------------------------------------------------------------------
    // Reference shape: C=K=96 at the target sizes for roofline context
    // -----------------------------------------------------------------------
    printf("\n--- C=K=96 reference (same spatial sizes) ---\n");

    run_bench("C=K=96  6x1106x834 no_pad",
              {.n=1, .id=6, .ih=1106, .iw=834, .c=96, .k=96,
               .pad_d=0, .pad_h=0, .pad_w=0},
              warmup, iters);

    run_bench("C=K=96  6x97x832  pad_011",
              {.n=1, .id=6, .ih=97, .iw=832, .c=96, .k=96,
               .pad_d=0, .pad_h=1, .pad_w=1},
              warmup, iters);

    // -----------------------------------------------------------------------
    // Small shapes from unit tests (sanity / overhead baseline)
    // -----------------------------------------------------------------------
    printf("\n--- Small shapes (unit-test sizes, overhead baseline) ---\n");

    run_bench("C=3/K=96  1x4x8x16 pad_011",
              {.n=1, .id=4, .ih=8, .iw=16, .c=3, .k=96},
              warmup, iters);

    run_bench("C=K=96  1x4x8x16 pad_011",
              {.n=1, .id=4, .ih=8, .iw=16, .c=96, .k=96},
              warmup, iters);

    return 0;
}
