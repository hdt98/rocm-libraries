#pragma once

// 3D direct forward convolution kernel for:
//   C = K = 96 channels, groups = 1
//   3x3x3 filter, unit stride, unit dilation, pad_d=0, pad_h=pad_w=1
//   fp16 input/output, fp32 accumulation
//   Target: CDNA 4 (gfx950)
//
// MFMA instruction: mfma_f32_16x16x16f16
//   M = 16  (K_TILE: output channels per wave, MFMA outer/B dimension)
//   N = 16  (OW_TILE: output columns per wave, MFMA outer/A dimension)
//   K_red = 16 (C_TILE: input channels per MFMA call)
//   C_STEPS = 6 (= C / C_TILE = 96 / 16)
//
// Work decomposition:
//   Each workgroup computes one output row (n, od_out, oh_out, ow_tile, k_tile).
//   No circular accumulator buffer is needed: exactly one output OH per block.
//
//   Grid:
//     x = ceil(OW / block_q) * n_fold
//     y = K / K_TILE  (= 6 for K=96)
//     z = ceil(N / n_fold) * od_size * oh_size
//
// Weight strategy:
//   One filter-depth T-slice fits in weight_lds:
//     K_TILE * kh * kw * C fp16 = 16 * 9 * 96 * 2 = 27,648 bytes (~27 KB).
//   Three T-passes (T=0,1,2) load successively and accumulate into acc.
//
// Input loading:
//   For each T-pass, kh=3 rows are loaded: input at (id_base+T, oh+R-1) for R=0,1,2.
//   Loaded cooperatively via buffer_load_lds into a 3-row input_lds buffer.
//
// LDS budget (waves_q=4, block_q=64):
//   weight_lds : K_TILE * kh * kw * C8 uint4 = 1728 uint4 = 27,648 bytes
//   input_lds  : kh * BLOCK_W * C8 uint4 = 3*66*12 = 2376 uint4 = 38,016 bytes  ← OVERFLOW
//
// LDS budget (waves_q=1, block_q=16):
//   weight_lds : 1728 uint4 = 27,648 bytes
//   input_lds  : 3 * 18 * 12 = 648 uint4 = 10,368 bytes
//   Total      : 2376 * 16 = 38,016 bytes → fits for waves_q=1
//   For waves_q=2 (block_q=32): 3*34*12=1224 uint4=19,584 bytes → total ~47 KB ✓
//   For waves_q=4 (block_q=64): 3*66*12=2376 uint4=38,016 bytes → total ~66 KB ✗
//
// Therefore configs are limited to waves_q <= 2 given the LDS constraint.
//
// MFMA operand layout (mfma_f32_16x16x16f16):
//   A (input):  row-major 16(OW) x 16(C_TILE)
//               lane % 16 = OW position, lane / 16 = which 4-fp16 C group
//   B (weight): col-major 16(K_TILE) x 16(C_TILE)  [stored as K_TILE-major in LDS]
//               lane % 16 = output channel (k_local), lane / 16 = C group
//   C/D result: 16(OW) x 16(K_TILE) fp32
//               lane % 16 = OW position, acc[0..3] = 4 consecutive K channels
//
// Weight LDS layout: [k_local=0..K_TILE-1][R=0..KH-1][S=0..KW-1][c8=0..C8-1] in uint4
//   Linear index: k_local*(KH*KW*C8) + R*(KW*C8) + S*C8 + c8
//   For MFMA B operand with lane_kb = lane%K_TILE, C-group Ck:
//     b_idx = lane_kb*(KH*KW*C8) + R*(KW*C8) + S*C8 + Ck*2
//     (two consecutive uint4 = 16 fp16 = one C_TILE worth)
//
// Input LDS layout: [R=0..KH-1][W_local=0..BLOCK_W-1][c8=0..C8-1] in uint4
//   Linear index: R*BLOCK_W*C8 + W_local*C8 + c8
//   For MFMA A operand with lane_ow = lane%OW_TILE, C-group Ck:
//     a_idx = R*BLOCK_W*C8 + (wave_ow_base + lane_ow + S)*C8 + Ck*2

#include "../detail.h"
#include "../launch_params.h"
#include "../mathutil.h"
#include "../types.h"
#include "hipconv/conv3d_params.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace conv3d_96c
{

constexpr int WAVE_SIZE = 64;

// Compile-time MFMA tile dimensions for this variant.
constexpr int K_TILE  = 16;      // output channels per wave (MFMA M/B outer)
constexpr int OW_TILE = 16;      // output columns per wave  (MFMA N/A outer)
constexpr int C_TILE  = 16;      // input channels per MFMA call (MFMA K_red)
constexpr int C       = 96;      // total channels (must equal K, compile-time)
constexpr int C_STEPS = C / C_TILE; // = 6
constexpr int KD      = 3;
constexpr int KH      = 3;
constexpr int KW      = 3;

// uint4 groups per channel slice.
constexpr int C8 = C / 8; // = 12

struct Config
{
    int waves_q; // waves along OW: block_q = waves_q * OW_TILE
    int n_fold = 4;

    constexpr int block_q() const { return waves_q * OW_TILE; }
    constexpr int block_size() const { return waves_q * WAVE_SIZE; }

    // Input LDS in uint4 units for this config.
    constexpr int input_lds_c8() const { return KH * (block_q() + KW - 1) * C8; }

    // Weight LDS in uint4 units (constant across configs).
    static constexpr int weight_lds_c8() { return K_TILE * KH * KW * C8; }

    constexpr int total_lds_c8() const { return weight_lds_c8() + input_lds_c8(); }
};

constexpr Config configs[] = {
    {.waves_q = 2},
    {.waves_q = 1},
};
constexpr int NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);

inline bool is_valid_config(const hipconv::Conv3dParams& par, const Config& cfg)
{
    // Check LDS budget: must fit in 64 KB (4096 uint4).
    if(cfg.total_lds_c8() > 4096)
        return false;
    if(par.ow < cfg.block_q() && cfg.waves_q > 1)
        return false;
    return true;
}

inline LaunchParams get_launch_params(int config_idx, const hipconv::Conv3dParams& par)
{
    const auto& cfg  = configs[config_idx];
    auto blocks_ow   = divup(par.ow, cfg.block_q());
    auto blocks_ow_n = blocks_ow * cfg.n_fold;
    auto blocks_k    = divup(par.k, K_TILE);              // = 6 for K=96
    auto blocks_n    = divup(par.n, cfg.n_fold);
    auto blocks_z    = blocks_n * par.od * par.oh;

    LaunchParams lp;
    lp.grid       = dim3(blocks_ow_n, blocks_k, blocks_z);
    lp.block_size = dim3(cfg.block_size(), 1, 1);
    return lp;
}

} // namespace conv3d_96c


// -----------------------------------------------------------------------
// Device kernel
// -----------------------------------------------------------------------

template <conv3d_96c::Config cfg>
__device__ void conv3d_96c_fp16_cdna4_ndhwc(const _Float16* __restrict__ in,
                                             const _Float16* __restrict__ wei,
                                             _Float16* __restrict__ out,
                                             int N,
                                             int id,
                                             int ih,
                                             int iw,
                                             int od_size,
                                             int oh_size,
                                             int ow_size)
{
    using namespace conv3d_96c;

    // ------------------------------------------------------------------
    // Compile-time LDS sizes.
    // ------------------------------------------------------------------
    constexpr int BLOCK_W       = cfg.block_q() + (KW - 1); // input W tile width
    constexpr int INPUT_ROW_C8  = BLOCK_W * C8;             // uint4 per input row
    constexpr int WEIGHT_LDS_C8 = K_TILE * KH * KW * C8;   // = 16*9*12 = 1728
    constexpr int INPUT_LDS_C8  = KH * INPUT_ROW_C8;        // kh rows
    constexpr int TOTAL_LDS_C8  = WEIGHT_LDS_C8 + INPUT_LDS_C8;

    __shared__ uint4 lds[TOTAL_LDS_C8];
    uint4* const weight_lds = lds;
    uint4* const input_lds  = lds + WEIGHT_LDS_C8;

    // ------------------------------------------------------------------
    // Thread indices.
    // ------------------------------------------------------------------
    const int tid          = threadIdx.x;
    const int wave         = tid / WAVE_SIZE;
    const int lane         = tid % WAVE_SIZE;
    const int lane_ow      = lane % OW_TILE; // OW position within wave tile
    const int lane_kg      = lane / OW_TILE; // C-reduction group (0..3) / K output group
    const int wave_ow_base = wave * OW_TILE;

    // ------------------------------------------------------------------
    // Grid → problem coordinates.
    //   grid.x = ceil(OW/block_q) * n_fold
    //   grid.y = K / K_TILE
    //   grid.z = ceil(N/n_fold) * od_size * oh_size
    // ------------------------------------------------------------------
    const int block_n_mod  = blockIdx.x % cfg.n_fold;
    const int block_ow_idx = blockIdx.x / cfg.n_fold;
    const int block_k_idx  = blockIdx.y;

    const int blocks_per_n = od_size * oh_size;
    const int block_n_div  = blockIdx.z / blocks_per_n;
    const int rem_z        = blockIdx.z % blocks_per_n;
    const int block_od_idx = rem_z / oh_size;
    const int block_oh_idx = rem_z % oh_size;

    const int block_n = block_n_div * cfg.n_fold + block_n_mod;
    if(block_n >= N)
        return;

    const int block_ow_start = block_ow_idx * cfg.block_q();
    const int block_k_start  = block_k_idx  * K_TILE;
    const int od_out         = block_od_idx;
    const int oh_out         = block_oh_idx;
    const int id_base        = od_out; // pad_d=0, stride_d=1

    // ------------------------------------------------------------------
    // Global memory strides (in fp16 elements).
    // ------------------------------------------------------------------
    const size_t iw_C       = (size_t)iw  * C;
    const size_t ih_iw_C    = (size_t)ih  * iw_C;
    const size_t id_ih_iw_C = (size_t)id  * ih_iw_C; // per batch image
    const size_t ow_C       = (size_t)ow_size * C;
    const size_t oh_ow_C    = (size_t)oh_size * ow_C;
    const size_t od_oh_ow_C = (size_t)od_size * oh_ow_C; // per batch image

    // n-image byte offset in the input tensor.
    const size_t n_in_offset = (size_t)block_n * id_ih_iw_C;

    // ------------------------------------------------------------------
    // Buffer resources.
    // ------------------------------------------------------------------
    const size_t in_bytes          = (size_t)N * id_ih_iw_C * sizeof(_Float16);
    constexpr int rsrc_data_format = 1 << 15;
    auto input_rsrc                = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<_Float16*>(in), 0, in_bytes, rsrc_data_format);

    // Weight layout: [K][kd][kh][kw][C] fp16 (KTRSC).
    const size_t wei_bytes = (size_t)C * KD * KH * KW * C * sizeof(_Float16);
    auto weight_rsrc       = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<_Float16*>(wei), 0, wei_bytes, rsrc_data_format);

    // ------------------------------------------------------------------
    // Output addressing.
    // Each lane produces 4 output channels: block_k_start + lane_kg*4 + 0..3.
    // OW position: block_ow_start + wave_ow_base + lane_ow.
    // ------------------------------------------------------------------
    const int  global_ow  = block_ow_start + wave_ow_base + lane_ow;
    const int  out_k_base = block_k_start + lane_kg * 4;
    const bool out_valid  = (global_ow < ow_size) && (out_k_base + 3 < C);

    _Float16* out_lane = nullptr;
    if(out_valid)
    {
        out_lane = out
                   + (size_t)block_n * od_oh_ow_C
                   + (size_t)od_out  * oh_ow_C
                   + (size_t)oh_out  * ow_C
                   + (size_t)global_ow * C
                   + out_k_base;
    }

    // ------------------------------------------------------------------
    // Single accumulator per lane: 4 fp32 output channels for this
    // (od_out, oh_out, global_ow, out_k_base..+3) output voxel.
    // ------------------------------------------------------------------
    fp32x4_t acc = {0.f, 0.f, 0.f, 0.f};

    // ------------------------------------------------------------------
    // Main loop: kd=3 filter-depth passes.
    // For each T ∈ [0, KD): input depth = id_base + T.
    // ------------------------------------------------------------------
    static_for<KD>(
        [&]<int T>()
        {
            const int id_t = id_base + T; // always in [0, id-1] since pad_d=0

            // ---- Load weight T-slice into weight_lds ----
            // Byte offset: (block_k_start * KD*KH*KW*C + T * KH*KW*C) * sizeof(fp16).
            {
                const uint32_t wei_base =
                    (uint32_t)((size_t)(block_k_start * KD * KH * KW * C
                                        + T * KH * KW * C) * sizeof(_Float16));
                for(int j = tid; j < WEIGHT_LDS_C8; j += cfg.block_size())
                {
                    __builtin_amdgcn_raw_ptr_buffer_load_lds(
                        weight_rsrc,
                        &weight_lds[j],
                        16,
                        wei_base + (uint32_t)(j * sizeof(uint4)),
                        0, 0, 0);
                }
                asm volatile("s_waitcnt vmcnt(0)\n");
                __syncthreads();
            }

            // ---- Load kh input rows into input_lds ----
            // Row R covers input at (id_t, oh_out + R - pad_h) = (id_t, oh_out + R - 1).
            // input_lds layout: [R][W_local][c8] (row-major uint4).
            //
            // Cooperative load: all threads stride over each row's INPUT_ROW_C8 vectors.
            // For vector j in row R:
            //   W_local  = j / C8  → global W = block_ow_start - pad_w + W_local
            //   c8_local = j % C8  → byte offset within channel slice
            // OOB positions (W or H) → voffset = in_bytes → buffer returns 0 (zero-pad).
            static_for<KH>(
                [&]<int R>()
                {
                    const int ih_r = oh_out + R - 1; // input H row (pad_h = 1)
                    for(int j = tid; j < INPUT_ROW_C8; j += cfg.block_size())
                    {
                        const int    w_local  = j / C8;
                        const int    c8_local = j % C8;
                        const int    giw      = (block_ow_start - 1) + w_local; // pad_w=1
                        uint32_t     voffset;
                        if(giw >= 0 && giw < iw && ih_r >= 0 && ih_r < ih)
                        {
                            voffset = (uint32_t)(
                                (n_in_offset
                                 + (size_t)id_t  * ih_iw_C
                                 + (size_t)ih_r  * iw_C
                                 + (size_t)giw   * C
                                 + (size_t)c8_local * 8) * sizeof(_Float16));
                        }
                        else
                        {
                            voffset = (uint32_t)in_bytes; // OOB → zero
                        }
                        __builtin_amdgcn_raw_ptr_buffer_load_lds(
                            input_rsrc,
                            &input_lds[R * INPUT_ROW_C8 + j],
                            16,
                            voffset,
                            0, 0, 0);
                    }
                });

            asm volatile("s_waitcnt vmcnt(0)\n");
            __syncthreads();

            // ---- Accumulate: all (R, S, Ck) MFMA calls ----
            //
            // B operand (weight): weight_lds[k_local*(KH*KW*C8) + R*(KW*C8) + S*C8 + Ck*2]
            //   lane_kb = lane % K_TILE → k_local index
            //   Two consecutive uint4 at Ck*2 give 16 fp16 = one C_TILE slice.
            //
            // A operand (input): input_lds[R*INPUT_ROW_C8 + w_local*C8 + Ck*2]
            //   w_local = wave_ow_base + lane_ow + S
            //   lane_ow = lane % OW_TILE → OW position within wave tile
            //   Two consecutive uint4 at Ck*2 give 16 fp16 = one C_TILE slice.
            static_for<KH>(
                [&]<int R>()
                {
                    static_for<KW>(
                        [&]<int S>()
                        {
                            static_for<C_STEPS>(
                                [&]<int Ck>()
                                {
                                    const int lane_kb = lane % K_TILE;
                                    const int b_idx   = lane_kb * (KH * KW * C8)
                                                        + R * (KW * C8)
                                                        + S * C8
                                                        + Ck * 2;
                                    const fp16x4_t b_reg = *reinterpret_cast<const fp16x4_t*>(
                                        reinterpret_cast<const uint2*>(&weight_lds[b_idx]));

                                    const int w_local = wave_ow_base + lane_ow + S;
                                    const int a_idx   = R * INPUT_ROW_C8
                                                        + w_local * C8
                                                        + Ck * 2;
                                    const fp16x4_t a_reg = *reinterpret_cast<const fp16x4_t*>(
                                        reinterpret_cast<const uint2*>(&input_lds[a_idx]));

                                    acc = __builtin_amdgcn_mfma_f32_16x16x16f16(
                                        b_reg, a_reg, acc, 0, 0, 0);
                                });
                        });
                });

            // Sync before the next T-pass overwrites weight_lds and input_lds.
            __syncthreads();
        });

    // ------------------------------------------------------------------
    // Write output: 4 fp32 values → 4 fp16 stored at out_lane.
    // ------------------------------------------------------------------
    if(out_lane)
    {
        __half2 lo = __float22half2_rn({acc[0], acc[1]});
        __half2 hi = __float22half2_rn({acc[2], acc[3]});
        reinterpret_cast<__half2*>(out_lane)[0] = lo;
        reinterpret_cast<__half2*>(out_lane)[1] = hi;
    }
}


template <conv3d_96c::Config cfg>
__global__ void conv3d_96c_fp16_ndhwc_cdna4_kernel(const _Float16* __restrict__ in,
                                                    const _Float16* __restrict__ wei,
                                                    _Float16* __restrict__ out,
                                                    int N,
                                                    int id,
                                                    int ih,
                                                    int iw,
                                                    int od_size,
                                                    int oh_size,
                                                    int ow_size)
{
    conv3d_96c_fp16_cdna4_ndhwc<cfg>(in, wei, out, N, id, ih, iw, od_size, oh_size, ow_size);
}


namespace conv3d_96c
{

template <size_t... Is>
void launch_dispatch(int config_idx,
                     std::index_sequence<Is...>,
                     const LaunchParams& lp,
                     const hipconv::Conv3dParams& par,
                     const void* in,
                     const void* wei,
                     void* out,
                     hipStream_t stream)
{
    auto do_launch = [&]<size_t I>()
    {
        conv3d_96c_fp16_ndhwc_cdna4_kernel<configs[I]>
            <<<lp.grid, lp.block_size, lp.dynamic_shared_bytes, stream>>>(
                static_cast<const _Float16*>(in),
                static_cast<const _Float16*>(wei),
                static_cast<_Float16*>(out),
                par.n,
                par.id,
                par.ih,
                par.iw,
                par.od,
                par.oh,
                par.ow);
    };
    (void)((config_idx == static_cast<int>(Is)
                ? (do_launch.template operator()<Is>(), true)
                : false) ||
           ...);
}

inline void launch(int config_idx,
                   const LaunchParams& lp,
                   const hipconv::Conv3dParams& par,
                   const void* in,
                   const void* wei,
                   void* out,
                   hipStream_t stream)
{
    launch_dispatch(
        config_idx, std::make_index_sequence<NUM_CONFIGS>{}, lp, par, in, wei, out, stream);
}

} // namespace conv3d_96c
