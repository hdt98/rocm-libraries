#pragma once

#include "matrix_layout.h"
#include "detail.h"
#include "types.h"
#include "mathutil.h"
#include "launch_params.h"
#include "kernel_variant.h"
#include "transpose_lds_layout.h"
#include "hip_util.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
namespace grouped_16c_wgrad
{
constexpr int WAVE_SIZE = 64;
constexpr int BLOCK_Q   = 16;

struct Config
{
    int waves_per_wg;
    int wave_q16 = 1;

    int kh = 3;
    int kw = 3;

    int group_size = 16;

    int block_p = 0; // 0 = all rows (no H tiling)

    hipconv::Direction direction = hipconv::Direction::Wgrad;

    constexpr int block_c() const { return waves_per_wg * group_size; }
    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }
};

constexpr Config configs[] = {
    {.waves_per_wg = 4, .wave_q16 = 4},
    {.waves_per_wg = 4},
    {.waves_per_wg = 3},
    {.waves_per_wg = 2},
    {.waves_per_wg = 1},
};

constexpr int NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);

inline bool is_valid_config(const hipconv::Conv2dParams& par, const Config& cfg)
{
    if(par.direction != cfg.direction)
        return false;
    if((par.groups % cfg.waves_per_wg) != 0)
        return false;
    // Wider waves mean fewer reductions in global memory.
    // TODO: Choose the narrowest (wave_q16) config that is at least as wide as the input tensor.
    return true;
}

inline LaunchParams get_launch_params(int config_idx, const hipconv::Conv2dParams& par)
{
    const auto& cfg = configs[config_idx];

    auto blocks_q = divup(par.q, cfg.wave_q16 * BLOCK_Q);
    auto blocks_p = (cfg.block_p > 0) ? divup(par.p, cfg.block_p) : 1;
    auto blocks_c = divup(par.groups, cfg.waves_per_wg);

    LaunchParams launch;
    launch.grid       = dim3(blocks_c, blocks_q * blocks_p, par.n);
    launch.block_size = dim3(cfg.block_size(), 1, 1);
    return launch;
}

inline size_t get_workspace_size(int /*config_idx*/, const hipconv::Conv2dParams& /*par*/)
{ return 0; }

} // namespace grouped_16c_wgrad

template <grouped_16c_wgrad::Config cfg>
__device__ void conv2d_grouped_16c_wgrad_fp16_cdna4_nhwc_impl(const _Float16* __restrict__ input,
                                                              const _Float16* __restrict__ delta,
                                                              float* __restrict__ wgrad,
                                                              int N,
                                                              int groups,
                                                              int c_per_group,
                                                              int k_per_group,
                                                              int hi,
                                                              int wi,
                                                              int ho,
                                                              int wo,
                                                              int fy,
                                                              int fx,
                                                              int sy,
                                                              int sx,
                                                              int dy,
                                                              int dx,
                                                              int py,
                                                              int px)
{
    using namespace grouped_16c_wgrad;
    using ResultLayout    = MatrixLayout<16, 16, 1, float>;
    using TransposeLayout = TransposeLDSLayout<16, 16>;
    using int16x4_t       = __attribute__((ext_vector_type(4))) short;

    constexpr int GROUP_SIZE = cfg.group_size;
    constexpr int BLOCK_C    = cfg.block_c();
    constexpr int BLOCK_C8   = BLOCK_C / 8;

    constexpr int BLOCK_Q_TOTAL = cfg.wave_q16 * BLOCK_Q;
    constexpr int BLOCK_W_TOTAL = BLOCK_Q_TOTAL + (cfg.kw - 1);

    constexpr int NUM_INPUT_LDS_BUFFERS    = 2;
    constexpr int INPUT_LDS_BUFFER_SIZE_C8 = BLOCK_C8 * BLOCK_W_TOTAL;
    constexpr int DELTA_LDS_BUFFER_SIZE_C8 = BLOCK_C8 * BLOCK_Q_TOTAL;

    constexpr int COMPUTE_LDS_SIZE =
        NUM_INPUT_LDS_BUFFERS * INPUT_LDS_BUFFER_SIZE_C8 + DELTA_LDS_BUFFER_SIZE_C8;
    constexpr int COMPUTE_LDS_BYTES = (int)sizeof(uint4) * COMPUTE_LDS_SIZE;

    constexpr int WEIGHTS_PER_WG    = cfg.waves_per_wg * GROUP_SIZE * cfg.kh * cfg.kw * GROUP_SIZE;
    constexpr int STAGING_LDS_BYTES = WEIGHTS_PER_WG * (int)sizeof(float);
    constexpr int TOTAL_LDS_BYTES =
        COMPUTE_LDS_BYTES > STAGING_LDS_BYTES ? COMPUTE_LDS_BYTES : STAGING_LDS_BYTES;

    __shared__ char lds_raw[TOTAL_LDS_BYTES];
    uint4* input_lds = reinterpret_cast<uint4*>(lds_raw);
    uint4* delta_lds = input_lds + NUM_INPUT_LDS_BUFFERS * INPUT_LDS_BUFFER_SIZE_C8;

    const int tid  = threadIdx.x;
    const int wave = tid / WAVE_SIZE;
    const int lane = tid % WAVE_SIZE;

    // Wave mapping: each wave computes one group, all kw positions.
    const int wave_group = wave;

    // Distribute groups across different XCD so that reductions hit L2 cache.
    const int block_group_idx = blockIdx.x;
    const int block_x_idx     = blockIdx.y;
    const int blocks_q        = divup(wo, BLOCK_Q_TOTAL);
    const int block_q_idx     = block_x_idx % blocks_q;
    const int block_p_idx     = block_x_idx / blocks_q;
    const int block_q         = block_q_idx * BLOCK_Q_TOTAL;
    const int block_group     = block_group_idx * cfg.waves_per_wg;

    // Output row range for this block.
    const int p_first = (cfg.block_p > 0) ? block_p_idx * cfg.block_p : 0;
    const int p_last  = (cfg.block_p > 0) ? min(p_first + cfg.block_p, ho) : ho;

    // Input row range: output row p uses input rows p-py .. p-py+kh-1.
    const int y_first  = max(0, p_first - py);
    const int y_last   = min(hi, p_last - py + cfg.kh - 1);
    const int block_c8 = block_group * (GROUP_SIZE / 8);

    const int C  = groups * GROUP_SIZE;
    const int C8 = C / 8;

    // Accumulators: one per (r, s) filter position.
    constexpr auto Zero = fp32x4_t{0.f, 0.f, 0.f, 0.f};
    fp32x4_t acc[cfg.kh * cfg.kw];
    for(int i = 0; i < cfg.kh * cfg.kw; i++)
        acc[i] = Zero;

    // Delta register circular buffer: kh rows × wave_q16 Q-tiles.
    fp16x4_t delta_regs[cfg.kh][cfg.wave_q16];

    constexpr int rsrc_data_format = 1 << 15;

    auto input_bytes = static_cast<size_t>(N) * hi * wi * C * sizeof(half);
    auto input_rsrc  = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<_Float16*>(input), 0, input_bytes, rsrc_data_format);

    auto delta_bytes = static_cast<size_t>(N) * ho * wo * C * sizeof(half);
    auto delta_rsrc  = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<_Float16*>(delta), 0, delta_bytes, rsrc_data_format);

    // Multi-pass loading setup.
    constexpr int INPUT_LOAD_PASSES = divup(BLOCK_W_TOTAL * BLOCK_C8, cfg.block_size());
    constexpr int DELTA_LOAD_PASSES = divup(BLOCK_Q_TOTAL * BLOCK_C8, cfg.block_size());

    const int tr_row = TransposeLayout::row(lane);
    const int tr_col = TransposeLayout::col(lane);

    // ds_read_tr helper: read from delta_lds at Q-tile qt.
    auto read_delta_tr = [&](int qt) -> fp16x4_t
    {
        auto* base =
            reinterpret_cast<__half*>(delta_lds) + qt * BLOCK_Q * BLOCK_C + wave_group * GROUP_SIZE;
        auto* addr = reinterpret_cast<int16x4_t*>(base + tr_row * BLOCK_C + tr_col);
        return __builtin_bit_cast(fp16x4_t, __builtin_amdgcn_ds_read_tr16_b64_v4i16(addr));
    };

    // ds_read_tr helper: read input at column s from input_lds buffer buf.
    auto read_input_tr = [&](int buf, int s) -> fp16x4_t
    {
        auto* base = reinterpret_cast<__half*>(input_lds) + buf * (INPUT_LDS_BUFFER_SIZE_C8 * 8) +
                     s * BLOCK_C + wave_group * GROUP_SIZE;
        auto* addr = reinterpret_cast<int16x4_t*>(base + tr_row * BLOCK_C + tr_col);
        return __builtin_bit_cast(fp16x4_t, __builtin_amdgcn_ds_read_tr16_b64_v4i16(addr));
    };

    const int block_n = blockIdx.z;
    {
        const size_t input_n_offset = (size_t)block_n * hi * wi * C8;
        const size_t delta_n_offset = (size_t)block_n * ho * wo * C8;

        // Precompute per-pass load info (invariant across rows).
        uint32_t input_voffsets[INPUT_LOAD_PASSES];
        bool input_active[INPUT_LOAD_PASSES];
        uint4* input_lds_addrs[INPUT_LOAD_PASSES];
        for(int pass = 0; pass < INPUT_LOAD_PASSES; pass++)
        {
            int lds_idx           = tid + pass * cfg.block_size();
            input_active[pass]    = (lds_idx < BLOCK_W_TOTAL * BLOCK_C8);
            input_lds_addrs[pass] = &input_lds[lds_idx];
            if(input_active[pass])
            {
                int col        = lds_idx / BLOCK_C8;
                int c8_idx     = lds_idx % BLOCK_C8;
                int global_col = (block_q - px) + col;
                if(global_col >= 0 && global_col < wi)
                    input_voffsets[pass] =
                        sizeof(uint4) *
                        (input_n_offset + (size_t)global_col * C8 + block_c8 + c8_idx);
                else
                    input_voffsets[pass] = input_bytes;
            }
        }

        uint32_t delta_voffsets[DELTA_LOAD_PASSES];
        bool delta_active[DELTA_LOAD_PASSES];
        uint4* delta_lds_addrs[DELTA_LOAD_PASSES];
        for(int pass = 0; pass < DELTA_LOAD_PASSES; pass++)
        {
            int lds_idx           = tid + pass * cfg.block_size();
            delta_active[pass]    = (lds_idx < BLOCK_Q_TOTAL * BLOCK_C8);
            delta_lds_addrs[pass] = &delta_lds[lds_idx];
            if(delta_active[pass])
            {
                int col      = lds_idx / BLOCK_C8;
                int c8_idx   = lds_idx % BLOCK_C8;
                int global_q = block_q + col;
                if(global_q >= 0 && global_q < wo)
                    delta_voffsets[pass] = sizeof(uint4) * (delta_n_offset + (size_t)global_q * C8 +
                                                            block_c8 + c8_idx);
                else
                    delta_voffsets[pass] = delta_bytes;
            }
        }

        const size_t input_row_stride = (size_t)wi * C8 * sizeof(uint4);
        const size_t delta_row_stride = (size_t)wo * C8 * sizeof(uint4);

        // Helper lambdas for multi-pass loads.
        auto load_input = [&](int buf, int y)
        {
            for(int pass = 0; pass < INPUT_LOAD_PASSES; pass++)
            {
                if(input_active[pass])
                    __builtin_amdgcn_raw_ptr_buffer_load_lds(
                        input_rsrc,
                        input_lds_addrs[pass] + buf * INPUT_LDS_BUFFER_SIZE_C8,
                        16,
                        input_voffsets[pass] + y * input_row_stride,
                        0,
                        0,
                        0);
            }
        };

        auto load_delta = [&](int p)
        {
            for(int pass = 0; pass < DELTA_LOAD_PASSES; pass++)
            {
                if(delta_active[pass])
                {
                    uint32_t voff = (p >= 0 && p < ho) ? delta_voffsets[pass] + p * delta_row_stride
                                                       : static_cast<uint32_t>(delta_bytes);
                    __builtin_amdgcn_raw_ptr_buffer_load_lds(
                        delta_rsrc, delta_lds_addrs[pass], 16, voff, 0, 0, 0);
                }
            }
        };

        auto read_all_delta_tr = [&](int slot)
        { static_for<cfg.wave_q16>([&]<int QT>() { delta_regs[slot][QT] = read_delta_tr(QT); }); };

        // Prologue: pre-fill delta circular buffer with rows before main loop.
        for(int r = 0; r < cfg.kh - 1; r++)
        {
            int p = y_first + py - (cfg.kh - 1) + r;
            load_delta(p);
            wait_vmcnt<0>();
            __syncthreads();
            read_all_delta_tr(r);
            __syncthreads();
        }

        // Issue first input row load and first main-loop delta load (both async).
        if(y_first < y_last)
            load_input(0, y_first);

        {
            int p_new = y_first + py;
            load_delta(p_new);
        }

        int tic = 1;
        int toc = 0;

        // Main loop: iterate over input rows, unrolled by kh.
        for(int y_base = y_first; y_base + cfg.kh <= y_last; y_base += cfg.kh)
        {
            static_for<cfg.kh>(
                [&]<int Y_LOCAL>()
                {
                    int y = y_base + Y_LOCAL;

                    wait_vmcnt<0>();
                    __syncthreads();

                    constexpr int DELTA_SLOT = (cfg.kh - 1 + Y_LOCAL) % cfg.kh;
                    read_all_delta_tr(DELTA_SLOT);

                    __syncthreads();

                    // Issue async loads for NEXT iteration.
                    if((y + 1) < y_last)
                    {
                        load_input(tic, y + 1);
                        int p_next = (y + 1) + py;
                        load_delta(p_next);
                    }

                    static_for<cfg.kw>(
                        [&]<int S>()
                        {
                            static_for<cfg.kh>(
                                [&]<int KR>()
                                {
                                    int p = y - KR + py;
                                    if(p >= p_first && p < p_last)
                                    {
                                        constexpr int SLOT =
                                            (cfg.kh - 1 + Y_LOCAL - KR + cfg.kh) % cfg.kh;
                                        static_for<cfg.wave_q16>(
                                            [&]<int QT>()
                                            {
                                                fp16x4_t input_reg =
                                                    read_input_tr(toc, QT * BLOCK_Q + S);
                                                acc[KR * cfg.kw + S] =
                                                    __builtin_amdgcn_mfma_f32_16x16x16f16(
                                                        input_reg,
                                                        delta_regs[SLOT][QT],
                                                        acc[KR * cfg.kw + S],
                                                        0,
                                                        0,
                                                        0);
                                            });
                                    }
                                });
                        });

                    tic ^= 1;
                    toc ^= 1;
                });
        }

        // Remainder: hi % kh leftover rows.
        {
            int num_rows   = y_last - y_first;
            int y_rem_base = y_first + (num_rows / cfg.kh) * cfg.kh;
            static_for<cfg.kh>(
                [&]<int Y_LOCAL>()
                {
                    if(Y_LOCAL >= num_rows % cfg.kh)
                        return;
                    int y = y_rem_base + Y_LOCAL;

                    wait_vmcnt<0>();
                    __syncthreads();

                    constexpr int DELTA_SLOT = (cfg.kh - 1 + Y_LOCAL) % cfg.kh;
                    read_all_delta_tr(DELTA_SLOT);

                    __syncthreads();

                    if((y + 1) < y_last)
                    {
                        load_input(tic, y + 1);
                        int p_next = (y + 1) + py;
                        load_delta(p_next);
                    }

                    static_for<cfg.kw>(
                        [&]<int S>()
                        {
                            static_for<cfg.kh>(
                                [&]<int KR>()
                                {
                                    int p = y - KR + py;
                                    if(p >= p_first && p < p_last)
                                    {
                                        constexpr int SLOT =
                                            (cfg.kh - 1 + Y_LOCAL - KR + cfg.kh) % cfg.kh;
                                        static_for<cfg.wave_q16>(
                                            [&]<int QT>()
                                            {
                                                fp16x4_t input_reg =
                                                    read_input_tr(toc, QT * BLOCK_Q + S);
                                                acc[KR * cfg.kw + S] =
                                                    __builtin_amdgcn_mfma_f32_16x16x16f16(
                                                        input_reg,
                                                        delta_regs[SLOT][QT],
                                                        acc[KR * cfg.kw + S],
                                                        0,
                                                        0,
                                                        0);
                                            });
                                    }
                                });
                        });

                    tic ^= 1;
                    toc ^= 1;
                });
        }
    }

    // Epilogue: LDS-staged coalesced atomicAdd.
    // Write accumulators → LDS in weight-tile layout, then coalesced atomicAdd from LDS.
    float* staging_lds = reinterpret_cast<float*>(lds_raw);

    const int lane_k    = ResultLayout::outer(lane);
    const int k_in_tile = wave_group * GROUP_SIZE + lane_k;

    wait_vmcnt<0>();
    __syncthreads();

    for(int r = 0; r < cfg.kh; r++)
        for(int s = 0; s < cfg.kw; s++)
            for(int idx = 0; idx < 4; idx++)
            {
                int c_local             = ResultLayout::inner(lane, idx);
                int lds_offset          = k_in_tile * cfg.kh * cfg.kw * GROUP_SIZE +
                                          r * cfg.kw * GROUP_SIZE + s * GROUP_SIZE + c_local;
                staging_lds[lds_offset] = acc[r * cfg.kw + s][idx];
            }
    __syncthreads();

    int global_k_base = block_group * GROUP_SIZE;
    float* wgrad_tile = wgrad + (size_t)global_k_base * cfg.kh * cfg.kw * GROUP_SIZE;

    for(int i = tid; i < WEIGHTS_PER_WG; i += cfg.block_size())
        atomicAdd(&wgrad_tile[i], staging_lds[i]);
}

template <grouped_16c_wgrad::Config cfg>
__global__ void conv2d_grouped_16c_wgrad_fp16_nhwc_cdna4(const _Float16* __restrict__ input,
                                                         const _Float16* __restrict__ delta,
                                                         float* __restrict__ wgrad,
                                                         int N,
                                                         int groups,
                                                         int c_per_group,
                                                         int k_per_group,
                                                         int hi,
                                                         int wi,
                                                         int ho,
                                                         int wo,
                                                         int fy,
                                                         int fx,
                                                         int sy,
                                                         int sx,
                                                         int dy,
                                                         int dx,
                                                         int py,
                                                         int px)
{
    conv2d_grouped_16c_wgrad_fp16_cdna4_nhwc_impl<cfg>(input,
                                                       delta,
                                                       wgrad,
                                                       N,
                                                       groups,
                                                       c_per_group,
                                                       k_per_group,
                                                       hi,
                                                       wi,
                                                       ho,
                                                       wo,
                                                       fy,
                                                       fx,
                                                       sy,
                                                       sx,
                                                       dy,
                                                       dx,
                                                       py,
                                                       px);
}


namespace grouped_16c_wgrad
{

template <size_t... Is>
void launch_dispatch(int config_idx,
                     std::index_sequence<Is...>,
                     const LaunchParams& lp,
                     const hipconv::Conv2dParams& par,
                     const void* in,
                     const void* wei,
                     void* out,
                     void* /*workspace*/,
                     hipStream_t stream)
{
    auto kernel_launch = [&]<size_t I>()
    {
        auto wgts_bytes = sizeof(float) * par.k * par.kh * par.kw * par.channels_per_group();
        HIP_CHECK(hipMemsetAsync(out, 0, wgts_bytes, stream));
        conv2d_grouped_16c_wgrad_fp16_nhwc_cdna4<configs[I]>
            <<<lp.grid, lp.block_size, 0, stream>>>(static_cast<const _Float16*>(in),
                                                    static_cast<const _Float16*>(wei),
                                                    static_cast<float*>(out),
                                                    par.n,
                                                    par.groups,
                                                    par.channels_per_group(),
                                                    par.filters_per_group(),
                                                    par.h,
                                                    par.w,
                                                    par.p,
                                                    par.q,
                                                    par.kh,
                                                    par.kw,
                                                    par.stride_h,
                                                    par.stride_w,
                                                    par.dilation_h,
                                                    par.dilation_w,
                                                    par.pad_h,
                                                    par.pad_w);
    };
    (void)((config_idx == static_cast<int>(Is) ? (kernel_launch.template operator()<Is>(), true)
                                               : false) ||
           ...);
}

inline void launch(int config_idx,
                   const LaunchParams& lp,
                   const hipconv::Conv2dParams& par,
                   const void* in,
                   const void* wei,
                   void* out,
                   void* workspace,
                   hipStream_t stream)
{
    launch_dispatch(config_idx,
                    std::make_index_sequence<NUM_CONFIGS>{},
                    lp,
                    par,
                    in,
                    wei,
                    out,
                    workspace,
                    stream);
}

constexpr KernelVariant make_variant()
{
    return {
        .is_applicable =
            [](const hipconv::Conv2dParams& par)
        {
            if(par.in_type != hipconv::DataType::fp16)
                return false;
            if(par.wei_type != hipconv::DataType::fp16)
                return false;
            if(par.direction != hipconv::Direction::Wgrad)
                return false;
            if(par.out_type != hipconv::DataType::fp32)
                return false;
            if(par.kh != 3 || par.kw != 3)
                return false;
            if(par.k != par.c)
                return false;
            if(par.channels_per_group() != 16)
                return false;
            if(par.c % 16 != 0)
                return false;
            if(par.stride_h != 1 || par.stride_w != 1)
                return false;
            if(par.dilation_h != 1 || par.dilation_w != 1)
                return false;
            if(par.pad_h > par.kh - 1 || par.pad_w > par.kw - 1)
                return false;
            return true;
        },
        .config_is_compatible = [](const hipconv::Conv2dParams& par, int idx)
        { return is_valid_config(par, configs[idx]); },
        .get_launch_params  = &get_launch_params,
        .launch             = &launch,
        .get_workspace_size = &get_workspace_size,
        .num_configs        = NUM_CONFIGS,
    };
}

} // namespace grouped_16c_wgrad
