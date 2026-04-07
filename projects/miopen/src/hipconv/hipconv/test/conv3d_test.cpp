// Standalone correctness test for hipconv 3D kernels.
//
// Uses the internal conv3d:: namespace directly to avoid depending on the
// grouped 2D registry (which has a pre-existing compiler-crash issue on gfx950).
//
// Compares kernel output against a CPU fp32 reference.

#include "conv3d/conv3d.hpp"
#include "hipconv/conv3d_params.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

#define HIP_CHECK(expr)                                                              \
    do {                                                                             \
        hipError_t _e = (expr);                                                      \
        if(_e != hipSuccess)                                                         \
        {                                                                            \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),       \
                    __FILE__, __LINE__);                                             \
            std::exit(1);                                                            \
        }                                                                            \
    } while(0)

static float fp16_to_float(uint16_t h)
{
    _Float16 v;
    memcpy(&v, &h, 2);
    return static_cast<float>(v);
}

static uint16_t float_to_fp16(float f)
{
    _Float16 v = static_cast<_Float16>(f);
    uint16_t h;
    memcpy(&h, &v, 2);
    return h;
}

// Fill a host fp16 buffer with small random values in [-1, 1].
static void fill_random(std::vector<uint16_t>& buf, unsigned seed = 42)
{
    srand(seed);
    for(auto& x : buf)
        x = float_to_fp16((float)(rand() % 2001 - 1000) / 1000.f);
}

// ---------------------------------------------------------------------------
// CPU reference: NDHWC input, KTRSC weights, NOPQK output
// Supports arbitrary C, K, pad_d/h/w, and 3x3x3 filter.
// ---------------------------------------------------------------------------
static void conv3d_reference(const std::vector<uint16_t>& in_h,
                              const std::vector<uint16_t>& wei_h,
                              std::vector<float>&          out_ref,
                              const hipconv::Conv3dParams& p)
{
    const int KD = p.kd, KH = p.kh, KW = p.kw;

    for(int n = 0; n < p.n; ++n)
    for(int od = 0; od < p.od; ++od)
    for(int oh = 0; oh < p.oh; ++oh)
    for(int ow = 0; ow < p.ow; ++ow)
    for(int k  = 0; k  < p.k;  ++k)
    {
        double acc = 0.0;
        for(int t = 0; t < KD; ++t)
        for(int r = 0; r < KH; ++r)
        for(int s = 0; s < KW; ++s)
        for(int c = 0; c < p.c; ++c)
        {
            int id_t = od * p.stride_d + t * p.dilation_d - p.pad_d;
            int ih_r = oh * p.stride_h + r * p.dilation_h - p.pad_h;
            int iw_s = ow * p.stride_w + s * p.dilation_w - p.pad_w;

            float in_val = 0.f;
            if(id_t >= 0 && id_t < p.id &&
               ih_r >= 0 && ih_r < p.ih &&
               iw_s >= 0 && iw_s < p.iw)
            {
                // NDHWC: [n][id][ih][iw][c]
                size_t in_idx = ((size_t)n  * p.id * p.ih * p.iw * p.c)
                              + ((size_t)id_t * p.ih * p.iw * p.c)
                              + ((size_t)ih_r * p.iw * p.c)
                              + ((size_t)iw_s * p.c)
                              + c;
                in_val = fp16_to_float(in_h[in_idx]);
            }

            // KTRSC / KZYXC: [k][kd][kh][kw][c]
            size_t wei_idx = ((size_t)k * KD * KH * KW * p.c)
                           + ((size_t)t * KH * KW * p.c)
                           + ((size_t)r * KW * p.c)
                           + ((size_t)s * p.c)
                           + c;
            float w_val = fp16_to_float(wei_h[wei_idx]);
            acc += (double)in_val * w_val;
        }

        // NOPQK / NDHWK: [n][od][oh][ow][k]
        size_t out_idx = ((size_t)n  * p.od * p.oh * p.ow * p.k)
                       + ((size_t)od * p.oh * p.ow * p.k)
                       + ((size_t)oh * p.ow * p.k)
                       + ((size_t)ow * p.k)
                       + k;
        out_ref[out_idx] = static_cast<float>(acc);
    }
}

// ---------------------------------------------------------------------------
// Test runner for one Conv3dParams configuration.
// Returns true if all outputs are within tolerance.
// ---------------------------------------------------------------------------
static bool run_test(const char* name, hipconv::Conv3dParams p)
{
    p.compute_output_size();

    printf("%-30s  N=%d  Di/Hi/Wi=%d/%d/%d  C=%d K=%d  "
           "pad_d=%d pad_h=%d pad_w=%d  -> Od/Oh/Ow=%d/%d/%d\n",
           name, p.n, p.id, p.ih, p.iw, p.c, p.k,
           p.pad_d, p.pad_h, p.pad_w, p.od, p.oh, p.ow);

    // Find a valid kernel config.
    auto cfgs = conv3d::get_valid_configs(p);
    if(cfgs.empty())
    {
        printf("  SKIP: no valid config\n");
        return true; // not an error, just unsupported
    }
    auto cfg = cfgs[0];

    // Allocate host buffers.
    size_t in_elems  = (size_t)p.n * p.id * p.ih * p.iw * p.c;
    size_t wei_elems = (size_t)p.k * p.kd * p.kh * p.kw * p.c;
    size_t out_elems = (size_t)p.n * p.od * p.oh * p.ow * p.k;

    std::vector<uint16_t> in_h(in_elems), wei_h(wei_elems);
    std::vector<uint16_t> out_h(out_elems);
    std::vector<float>    out_ref(out_elems, 0.f);

    fill_random(in_h,  42);
    fill_random(wei_h, 7);

    // CPU reference.
    conv3d_reference(in_h, wei_h, out_ref, p);

    // Allocate device buffers.
    void *in_d, *wei_d, *out_d;
    HIP_CHECK(hipMalloc(&in_d,  in_elems  * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&wei_d, wei_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&out_d, out_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMemset(out_d, 0, out_elems * sizeof(uint16_t)));

    HIP_CHECK(hipMemcpy(in_d,  in_h.data(),  in_elems  * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(wei_d, wei_h.data(), wei_elems * sizeof(uint16_t), hipMemcpyHostToDevice));

    // Launch kernel.
    conv3d::launch(cfg, p, in_d, wei_d, out_d, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    // Copy back.
    HIP_CHECK(hipMemcpy(out_h.data(), out_d, out_elems * sizeof(uint16_t), hipMemcpyDeviceToHost));

    // Compare.
    float atol, rtol;
    conv3d::get_tolerance(cfg, p, atol, rtol);

    size_t n_err = 0;
    float  max_err = 0.f;
    for(size_t i = 0; i < out_elems; ++i)
    {
        float got = fp16_to_float(out_h[i]);
        float ref = out_ref[i];
        float err = std::abs(got - ref);
        float tol = atol + rtol * std::abs(ref);
        if(err > tol)
        {
            if(n_err < 5)
                printf("  MISMATCH [%zu]: got=%g ref=%g err=%g tol=%g\n",
                       i, got, ref, err, tol);
            ++n_err;
        }
        if(err > max_err) max_err = err;
    }

    HIP_CHECK(hipFree(in_d));
    HIP_CHECK(hipFree(wei_d));
    HIP_CHECK(hipFree(out_d));

    if(n_err == 0)
        printf("  PASS  max_err=%.3e\n", max_err);
    else
        printf("  FAIL  %zu/%zu mismatches  max_err=%.3e\n", n_err, out_elems, max_err);

    return n_err == 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main()
{
    bool all_pass = true;

    // --- C=K=96 ---
    printf("=== C=K=96 kernel ===\n");
    all_pass &= run_test("96c_pad_h1_n1",   {.n=1, .id=4, .ih=8,  .iw=16, .c=96, .k=96});
    all_pass &= run_test("96c_pad_h1_n2",   {.n=2, .id=4, .ih=16, .iw=16, .c=96, .k=96});
    all_pass &= run_test("96c_pad_h1_ow17", {.n=1, .id=4, .ih=8,  .iw=17, .c=96, .k=96});
    all_pass &= run_test("96c_no_pad",      {.n=1, .id=5, .ih=8,  .iw=16, .c=96, .k=96,
                                             .pad_d=0, .pad_h=0, .pad_w=0});
    all_pass &= run_test("96c_no_pad_n2",   {.n=2, .id=5, .ih=10, .iw=20, .c=96, .k=96,
                                             .pad_d=0, .pad_h=0, .pad_w=0});
    // Small OW to exercise waves_q=1 config.
    all_pass &= run_test("96c_small_ow",    {.n=1, .id=4, .ih=8,  .iw=4,  .c=96, .k=96});

    printf("\n=== C=3, K=96 kernel ===\n");
    all_pass &= run_test("3c96k_pad_h1",   {.n=1, .id=4, .ih=8,  .iw=16, .c=3, .k=96});
    all_pass &= run_test("3c96k_no_pad",   {.n=1, .id=5, .ih=8,  .iw=16, .c=3, .k=96,
                                            .pad_d=0, .pad_h=0, .pad_w=0});
    all_pass &= run_test("3c96k_n2",       {.n=2, .id=4, .ih=16, .iw=32, .c=3, .k=96});
    all_pass &= run_test("3c96k_small_ow", {.n=1, .id=4, .ih=8,  .iw=4,  .c=3, .k=96});
    // Larger Oh to exercise multi-tile (Oh > TILE_OH=16).
    all_pass &= run_test("3c96k_oh32",    {.n=1, .id=4, .ih=32, .iw=32, .c=3, .k=96});
    all_pass &= run_test("3c96k_oh33",    {.n=1, .id=4, .ih=33, .iw=32, .c=3, .k=96}); // partial last tile
    // Target-shape-sized Oh (exercises waves_q=4 config).
    all_pass &= run_test("3c96k_target",  {.n=1, .id=6, .ih=40, .iw=64, .c=3, .k=96,
                                           .pad_d=0, .pad_h=0, .pad_w=0});

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
