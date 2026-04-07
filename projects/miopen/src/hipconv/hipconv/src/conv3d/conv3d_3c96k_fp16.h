#pragma once

// 3D direct forward convolution kernel for:
//   C = 3 input channels, K = 96 output channels, groups = 1
//   3x3x3 filter, unit stride, unit dilation
//   pad_d = 0; pad_h and pad_w are 0 or 1 (runtime parameters)
//   fp16 input/output, fp32 accumulation
//   Target: CDNA 4 (gfx950)
//
// MFMA instruction: mfma_f32_16x16x16f16
//   C_PAD = 16 (C=3 zero-padded to next multiple of 16 for MFMA K-reduction)
//   C_STEPS = 1 (single MFMA per (R,S) filter position)
//   K_TILE = 16   (output channels per wave, MFMA M/B outer)
//   OW_TILE = 16  (output columns per wave,  MFMA N/A outer)
//
// MFMA lane layout (64-lane wave):
//   lane_kb = lane % K_TILE  → output channel index within tile
//   lane_kg = lane / K_TILE  → C-reduction group (0..3), each covering 4 C-channels
//   B (weight): b_base = lane_kb*(KH*KW*C_PAD8) + R*(KW*C_PAD8) + S*C_PAD8 + lane_kg/2
//               b_reg  = uint2 half lane_kg%2 at that base
//   A (input):  a_base = (oh_local+R)*INPUT_ROW_C8 + w_local*C_PAD8 + lane_kg/2
//               a_reg  = uint2 half lane_kg%2 at that base
//   D (result): acc[oh_local][0..3] = 4 fp32 at K positions lane_kg*4..+3 for OW=lane_ow
//
// Oh-tiling:
//   Each block computes TILE_OH consecutive output rows instead of one.
//   This amortises the weight load (288 uint4 per T-pass) over TILE_OH rows,
//   reducing the grid by TILE_OH× and filling each block with TILE_OH×27 MFMAs.
//
//   Input LDS holds (TILE_OH + KH - 1) rows per T-pass:
//     rows ih = oh_out-pad_h .. oh_out+TILE_OH+KH-2-pad_h
//   Weight LDS is unchanged (K_TILE×KH×KW×C_PAD8 = 288 uint4 = 4,608 bytes).
//
// LDS layout:
//   weight_lds: [k_local][R][S][c_pad8]  (288 uint4, always first)
//   input_lds:  [row][W_local][c_pad8]   row = 0..TILE_OH+KH-2
//
// Grid:
//   x = ceil(OW / block_q) * n_fold
//   y = K / K_TILE  (= 6 for K=96)
//   z = ceil(N / n_fold) * Od * ceil(Oh / TILE_OH)
//
// Config table:
//   { waves_q=4, tile_oh=16 }  — 86K blocks for target shape; waves_q=4 fits LDS
//   { waves_q=2, tile_oh=16 }  — 172K blocks; fallback for narrow OW
//   { waves_q=2, tile_oh=8  }  — 344K blocks; fallback when TILE_OH=16 leaves partial tiles
//   { waves_q=1, tile_oh=16 }  — very narrow OW
//   { waves_q=1, tile_oh=8  }  — smallest config

#include "../detail.h"
#include "../launch_params.h"
#include "../mathutil.h"
#include "../types.h"
#include "hipconv/conv3d_params.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace conv3d_3c96k
{

constexpr int WAVE_SIZE = 64;

constexpr int C      = 3;    // real input channels
constexpr int K      = 96;   // output channels
constexpr int C_PAD  = 16;   // padded channel dim for MFMA K-reduction
constexpr int K_TILE = 16;   // output channels per wave (MFMA M)
constexpr int OW_TILE = 16;  // output columns per wave  (MFMA N)
constexpr int C_PAD8 = C_PAD / 8; // uint4 per spatial position (= 2)
constexpr int KD = 3, KH = 3, KW = 3;

struct Config
{
    int waves_q; // waves along OW: block_q = waves_q * OW_TILE
    int tile_oh; // output rows per block
    int n_fold = 4;

    constexpr int block_q()    const { return waves_q * OW_TILE; }
    constexpr int block_size() const { return waves_q * WAVE_SIZE; }

    // uint4 counts for LDS sizing.
    static constexpr int weight_lds_c8() { return K_TILE * KH * KW * C_PAD8; } // = 288
    constexpr int input_lds_c8()  const { return (tile_oh + KH - 1) * (block_q() + KW - 1) * C_PAD8; }
    constexpr int total_lds_c8()  const { return weight_lds_c8() + input_lds_c8(); }
};

constexpr Config configs[] = {
    {.waves_q = 4, .tile_oh = 16},
    {.waves_q = 2, .tile_oh = 16},
    {.waves_q = 2, .tile_oh =  8},
    {.waves_q = 1, .tile_oh = 16},
    {.waves_q = 1, .tile_oh =  8},
};
constexpr int NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);

inline bool is_valid_config(const hipconv::Conv3dParams& par, const Config& cfg)
{
    if(cfg.total_lds_c8() > 4096) // 64 KB limit
        return false;
    if(par.ow < cfg.block_q() && cfg.waves_q > 1)
        return false;
    return true;
}

inline LaunchParams get_launch_params(int config_idx, const hipconv::Conv3dParams& par)
{
    const auto& cfg   = configs[config_idx];
    auto blocks_ow    = divup(par.ow, cfg.block_q());
    auto blocks_ow_n  = blocks_ow * cfg.n_fold;
    auto blocks_k     = divup(par.k, K_TILE);
    auto blocks_n     = divup(par.n, cfg.n_fold);
    auto blocks_oh    = divup(par.oh, cfg.tile_oh);
    auto blocks_z     = blocks_n * par.od * blocks_oh;

    LaunchParams lp;
    lp.grid       = dim3(blocks_ow_n, blocks_k, blocks_z);
    lp.block_size = dim3(cfg.block_size(), 1, 1);
    return lp;
}

} // namespace conv3d_3c96k


// -----------------------------------------------------------------------
// Device kernel
// -----------------------------------------------------------------------

template <conv3d_3c96k::Config cfg>
__device__ void conv3d_3c96k_fp16_cdna4_ndhwc(const _Float16* __restrict__ in,
                                               const _Float16* __restrict__ wei,
                                               _Float16* __restrict__ out,
                                               int N,
                                               int id,
                                               int ih,
                                               int iw,
                                               int od_size,
                                               int oh_size,
                                               int ow_size,
                                               int pad_h,
                                               int pad_w)
{
#ifdef __HIP_DEVICE_COMPILE__
    using namespace conv3d_3c96k;

    // ------------------------------------------------------------------
    // Compile-time sizes.
    // ------------------------------------------------------------------
    constexpr int TILE_OH       = cfg.tile_oh;
    constexpr int INPUT_OH_ROWS = TILE_OH + KH - 1; // input rows needed per T-pass
    constexpr int BLOCK_W       = cfg.block_q() + (KW - 1);
    constexpr int INPUT_ROW_C8  = BLOCK_W * C_PAD8;
    constexpr int WEIGHT_LDS_C8 = K_TILE * KH * KW * C_PAD8;  // = 288
    constexpr int INPUT_LDS_C8  = INPUT_OH_ROWS * INPUT_ROW_C8;
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
    const int lane_ow      = lane % OW_TILE;
    const int lane_kg      = lane / OW_TILE; // C-reduction group (0..3)
    const int wave_ow_base = wave * OW_TILE;

    // ------------------------------------------------------------------
    // Grid decode.
    //   grid.x = ceil(OW/block_q) * n_fold
    //   grid.y = K / K_TILE
    //   grid.z = ceil(N/n_fold) * od_size * ceil(oh_size/TILE_OH)
    // ------------------------------------------------------------------
    const int block_n_mod  = blockIdx.x % cfg.n_fold;
    const int block_ow_idx = blockIdx.x / cfg.n_fold;
    const int block_k_idx  = blockIdx.y;

    const int blocks_oh    = divup(oh_size, TILE_OH);
    const int blocks_per_n = od_size * blocks_oh;
    const int block_n_div  = blockIdx.z / blocks_per_n;
    const int rem_z        = blockIdx.z % blocks_per_n;
    const int block_od_idx = rem_z / blocks_oh;
    const int block_oh_idx = rem_z % blocks_oh;

    const int block_n = block_n_div * cfg.n_fold + block_n_mod;
    if(block_n >= N)
        return;

    const int block_ow_start = block_ow_idx * cfg.block_q();
    const int block_k_start  = block_k_idx  * K_TILE;
    const int od_out         = block_od_idx;
    const int oh_out         = block_oh_idx * TILE_OH; // first output row of this block
    const int id_base        = od_out; // pad_d = 0, stride_d = 1

    // ------------------------------------------------------------------
    // Global memory strides (fp16 elements).
    // ------------------------------------------------------------------
    const size_t iw_C       = (size_t)iw * C;
    const size_t ih_iw_C    = (size_t)ih * iw_C;
    const size_t id_ih_iw_C = (size_t)id * ih_iw_C;
    const size_t ow_K       = (size_t)ow_size * K;
    const size_t oh_ow_K    = (size_t)oh_size * ow_K;
    const size_t od_oh_ow_K = (size_t)od_size * oh_ow_K;

    const size_t n_in_offset = (size_t)block_n * id_ih_iw_C;
    const size_t in_bytes    = (size_t)N * id_ih_iw_C * sizeof(_Float16);
    const size_t wei_bytes   = (size_t)K * KD * KH * KW * C * sizeof(_Float16);

    constexpr int rsrc_data_format = 1 << 15;
    auto weight_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<_Float16*>(wei), 0, wei_bytes, rsrc_data_format);

    // ------------------------------------------------------------------
    // Output validity and base pointer.
    // Each lane writes TILE_OH rows; only rows where oh_out+oh_local < oh_size are valid.
    // ------------------------------------------------------------------
    const int  global_ow  = block_ow_start + wave_ow_base + lane_ow;
    const int  out_k_base = block_k_start + lane_kg * 4;
    const bool out_valid  = (global_ow < ow_size) && (out_k_base + 3 < K);

    // Base output pointer for oh_local=0; advance by oh_ow_K per row.
    _Float16* out_base = nullptr;
    if(out_valid)
    {
        out_base = out
                   + (size_t)block_n * od_oh_ow_K
                   + (size_t)od_out  * oh_ow_K
                   + (size_t)oh_out  * ow_K
                   + (size_t)global_ow * K
                   + out_k_base;
    }

    // ------------------------------------------------------------------
    // Accumulators: one fp32x4_t per output row.
    //
    // AGPR placement on CDNA: the compiler places fp32x4_t values in AGPRs
    // only when it can prove the address is never taken.  An array captured
    // by [&] in a lambda has its address taken and spills to scratch.
    //
    // Workaround: within each T-pass, copy each acc[oh_local] into a local
    // by-value variable `r`, run all (R,S) MFMAs with `r` as the accumulator,
    // then write `r` back.  The compiler can place `r` in AGPRs since it
    // is a pure local never captured by reference.  The acc[] array still
    // exists as the persistent store between T-passes (held in VGPRs), but
    // the hot MFMA path operates on `r` in AGPRs.
    // ------------------------------------------------------------------
    fp32x4_t acc[TILE_OH];
    static_for<TILE_OH>([&]<int oh_local>() { acc[oh_local] = {0.f, 0.f, 0.f, 0.f}; });

    // ------------------------------------------------------------------
    // Main loop: KD=3 filter-depth passes.
    // Each pass loads weight T-slice + INPUT_OH_ROWS input rows, then
    // accumulates across all oh_local rows and (R,S) filter positions.
    // ------------------------------------------------------------------
    static_for<KD>(
        [&]<int T>()
        {
            const int id_t = id_base + T;

            // ---- Load weight T-slice into weight_lds ----
            // Global layout: [K][KD][KH][KW][C=3] fp16 (KTRSC).
            // LDS layout:    [k_local][R][S][c_pad8]  in uint4.
            //
            // Each (k_local, R, S) position has C=3 fp16 = 6 bytes in global memory.
            // We load as one uint (4 bytes = c=0,1) + one uint16 (2 bytes = c=2).
            // The LDS slot holds C_PAD=16 fp16 in 2 uint4; only the first 3 channels
            // are real, channels 3..15 are zero.
            //
            // Global stride between consecutive K values: KD*KH*KW*C = 81 fp16.
            // We iterate over (k_local, filter_pos) and compute voffsets explicitly.
            // The weight tensor is tiny (15 KB) — it fits in L2 after the first block.
            {
                // Zero the weight LDS (clears padding channels).
                for(int j = tid; j < WEIGHT_LDS_C8; j += cfg.block_size())
                    weight_lds[j] = {0, 0, 0, 0};
                __syncthreads();

                const int num_wei_pos = K_TILE * KH * KW; // = 144
                for(int j = tid; j < num_wei_pos; j += cfg.block_size())
                {
                    const int k_local    = j / (KH * KW);
                    const int filter_pos = j % (KH * KW);
                    const int R          = filter_pos / KW;
                    const int S          = filter_pos % KW;
                    const int k_global   = block_k_start + k_local;

                    // Global byte index of wei[k_global][T][R][S][0].
                    const size_t gbase = ((size_t)k_global * KD * KH * KW * C
                                          + (size_t)T      *      KH * KW * C
                                          + (size_t)R      *           KW * C
                                          + (size_t)S      *                C);

                    // Load 3 fp16 as uint32 (c=0,1) + uint16 (c=2).
                    // For OOB k_global use zero.
                    uint32_t w01 = 0; // c=0 in low 16 bits, c=1 in high 16 bits
                    uint16_t w2  = 0; // c=2
                    if(k_global < K)
                    {
                        const _Float16* wp = wei + gbase;
                        memcpy(&w01, wp,     4); // 2 fp16 = 4 bytes
                        memcpy(&w2,  wp + 2, 2); // 1 fp16 = 2 bytes
                    }

                    // Pack into LDS slot: [c=0,1 in .x][c=2 in .y low half][zeros].
                    const int lds_idx = k_local * (KH * KW * C_PAD8)
                                        + R * (KW * C_PAD8)
                                        + S * C_PAD8;
                    // slot0 covers C_PAD channels 0..7 (8 fp16 = 1 uint4):
                    //   .x = fp16[0], fp16[1]   (= w01)
                    //   .y = fp16[2], fp16[3]=0  (= w2 | 0<<16)
                    //   .z = .w = 0              (fp16[4..7])
                    weight_lds[lds_idx]     = {w01, w2, 0, 0};
                    weight_lds[lds_idx + 1] = {0, 0, 0, 0};  // C_PAD channels 8..15
                }

                __syncthreads();
            }

            // ---- Load INPUT_OH_ROWS input rows into input_lds ----
            // Row `row` in LDS corresponds to input H index oh_out + row - pad_h.
            // Each position: C=3 fp16 = 6 bytes loaded as uint32 (c=0,1) + uint16 (c=2).
            // Stored in LDS as 2 uint4 per position (C_PAD8=2): channels 3..15 are zero.
            {
                // Zero the input LDS (clears padding channels and OOB positions).
                for(int j = tid; j < INPUT_LDS_C8; j += cfg.block_size())
                    input_lds[j] = {0, 0, 0, 0};
                __syncthreads();

                // Flatten (row, position) into a single strided loop so all threads
                // participate in both row and width dimensions.
                const int total_slots = INPUT_OH_ROWS * BLOCK_W; // positions to load
                for(int j = tid; j < total_slots; j += cfg.block_size())
                {
                    const int row = j / BLOCK_W;
                    const int pos = j % BLOCK_W;
                    const int ih_r = oh_out + row - pad_h;
                    const int giw  = block_ow_start - pad_w + pos;

                    uint32_t i01 = 0;
                    uint16_t i2  = 0;
                    if(giw >= 0 && giw < iw && ih_r >= 0 && ih_r < ih && id_t < id)
                    {
                        const _Float16* ip = in + (n_in_offset
                                                    + (size_t)id_t * ih_iw_C
                                                    + (size_t)ih_r * iw_C
                                                    + (size_t)giw  * C);
                        memcpy(&i01, ip,     4); // c=0,1
                        memcpy(&i2,  ip + 2, 2); // c=2
                    }

                    const int lds_idx = row * INPUT_ROW_C8 + pos * C_PAD8;
                    input_lds[lds_idx]     = {i01, i2, 0, 0}; // c=0..7, only 0,1,2 real
                    input_lds[lds_idx + 1] = {0, 0, 0, 0};    // c=8..15 = 0
                }

                __syncthreads();
            }

            // ---- Accumulate: TILE_OH × KH × KW MFMAs ----
            //
            // acc[] is updated here via compile-time-indexed static_for.
            // The lambda captures acc by reference; on CDNA this unfortunately
            // prevents AGPR placement (accum_vgpr=0 observed in profiling).
            // The workaround is to unroll via a fold expression using a local
            // auto that receives the accumulator by value, computes all (R,S)
            // MFMAs for one row, and assigns the result back.  Since the
            // assignment target `acc[oh_local]` is at a compile-time index,
            // the compiler can treat each element as an independent register.
            [&]<int... oh_locals>(std::integer_sequence<int, oh_locals...>)
            {
                ([&]()
                {
                    constexpr int oh_local = oh_locals;
                    // Take a by-value copy so the compiler sees an
                    // unaddressed fp32x4_t that it can place in AGPRs.
                    fp32x4_t r = acc[oh_local];
                    static_for<KH>(
                        [&]<int R>()
                        {
                            static_for<KW>(
                                [&]<int S>()
                                {
                                    const int lane_kb = lane % K_TILE;

                                    // B (weight)
                                    const int b_base = lane_kb * (KH * KW * C_PAD8)
                                                       + R * (KW * C_PAD8)
                                                       + S * C_PAD8
                                                       + lane_kg / 2;
                                    const auto* b_u2 = reinterpret_cast<const uint2*>(
                                        &weight_lds[b_base]);
                                    const fp16x4_t b_reg = *reinterpret_cast<const fp16x4_t*>(
                                        &b_u2[lane_kg % 2]);

                                    // A (input): row index = oh_local + R (compile-time)
                                    const int w_local = wave_ow_base + lane_ow + S;
                                    const int a_base  = (oh_local + R) * INPUT_ROW_C8
                                                        + w_local * C_PAD8
                                                        + lane_kg / 2;
                                    const auto* a_u2 = reinterpret_cast<const uint2*>(
                                        &input_lds[a_base]);
                                    const fp16x4_t a_reg = *reinterpret_cast<const fp16x4_t*>(
                                        &a_u2[lane_kg % 2]);

                                    r = __builtin_amdgcn_mfma_f32_16x16x16f16(
                                        b_reg, a_reg, r, 0, 0, 0);
                                });
                        });
                    acc[oh_local] = r;
                }(), ...);
            }(std::make_integer_sequence<int, TILE_OH>{});

            // Sync before next T-pass overwrites LDS.
            __syncthreads();
        });

    // ------------------------------------------------------------------
    // Write output: TILE_OH rows, skipping any OOB rows.
    // out_base points to oh_out row; ow_K is the stride to the next row.
    // ------------------------------------------------------------------
    [&]<int... oh_locals>(std::integer_sequence<int, oh_locals...>)
    {
        ([&]()
        {
            constexpr int oh_local = oh_locals;
            if(out_base && (oh_out + oh_local) < oh_size)
            {
                _Float16* out_lane = out_base + (size_t)oh_local * ow_K;
                __half2 lo = __float22half2_rn({acc[oh_local][0], acc[oh_local][1]});
                __half2 hi = __float22half2_rn({acc[oh_local][2], acc[oh_local][3]});
                reinterpret_cast<__half2*>(out_lane)[0] = lo;
                reinterpret_cast<__half2*>(out_lane)[1] = hi;
            }
        }(), ...);
    }(std::make_integer_sequence<int, TILE_OH>{});

#endif // __HIP_DEVICE_COMPILE__
}


template <conv3d_3c96k::Config cfg>
__global__ void conv3d_3c96k_fp16_ndhwc_cdna4_kernel(const _Float16* __restrict__ in,
                                                      const _Float16* __restrict__ wei,
                                                      _Float16* __restrict__ out,
                                                      int N,
                                                      int id,
                                                      int ih,
                                                      int iw,
                                                      int od_size,
                                                      int oh_size,
                                                      int ow_size,
                                                      int pad_h,
                                                      int pad_w)
{
    conv3d_3c96k_fp16_cdna4_ndhwc<cfg>(
        in, wei, out, N, id, ih, iw, od_size, oh_size, ow_size, pad_h, pad_w);
}


namespace conv3d_3c96k
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
        conv3d_3c96k_fp16_ndhwc_cdna4_kernel<configs[I]>
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
                par.ow,
                par.pad_h,
                par.pad_w);
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

} // namespace conv3d_3c96k
