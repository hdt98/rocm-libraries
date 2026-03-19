#pragma once

#include "matrix_layout.h"
#include "swizzle.h"
#include "types.h"
#include "mathutil.h"
#include "launch_params.h"
#include "kernel_variant.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace grouped_4c
{
// 64 threads per wave.
constexpr int WAVE_SIZE = 64;

// Block output is 16 columns wide.
constexpr int WARP_Q = 4;

// 4 channels per group.
constexpr int GROUP_SIZE = 4;

// Kernel configuration parameters.

struct Config
{
    int waves_c64;

    int waves_q4;

    int kh = 3;
    int kw = 3;

    int n_fold = 8;

    constexpr int num_waves() const { return waves_c64 * waves_q4; }

    constexpr int block_c() const { return waves_c64 * 64; }

    constexpr int block_q() const { return waves_q4 * 4; }

    constexpr int block_groups() const { return waves_c64 * 16; }

    constexpr int block_size() const { return num_waves() * WAVE_SIZE; }
};

// All instantiated configurations.
constexpr Config configs[] = {
    {.waves_c64 = 2, .waves_q4 = 8},
    {.waves_c64 = 2, .waves_q4 = 2},
    {.waves_c64 = 2, .waves_q4 = 1},
    {.waves_c64 = 1, .waves_q4 = 1},
};

constexpr int NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);

inline bool is_valid_config(const hipconv::Conv2dParams& par, const Config& cfg)
{
    if((par.groups % cfg.block_groups()) != 0)
    {
        return false;
    }
    return true;
}

inline LaunchParams get_launch_params(int config_idx, const hipconv::Conv2dParams& par)
{
    const auto& cfg = configs[config_idx];

    // Compute the grid size
    auto blocks_w      = divup(par.q, cfg.block_q());
    auto blocks_w_n    = blocks_w * cfg.n_fold;
    auto blocks_c      = divup(par.c, cfg.block_c());
    auto blocks_n_fold = divup(par.n, cfg.n_fold);

    LaunchParams launch;
    launch.grid       = dim3(blocks_w_n, blocks_c, blocks_n_fold);
    launch.block_size = dim3(cfg.block_size(), 1, 1);
    return launch;
}

} // namespace grouped_4c

template <grouped_4c::Config cfg>
__device__ void conv2d_grouped_4c_fp16_cdna4_nhwc_impl(const _Float16* __restrict__ in,
                                                       const _Float16* __restrict__ wei,
                                                       double alpha,
                                                       double beta,
                                                       _Float16* __restrict__ out,
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
    using namespace grouped_4c;

    // We will use the 4x4x4 matrix, batch size 16 matrix multiply instruction.
    constexpr int MFMA_M     = 4;
    constexpr int MFMA_K     = 4;
    constexpr int MFMA_N     = 4;
    constexpr int MFMA_BATCH = 16;
    using OperandLayout      = MatrixLayout<MFMA_M, MFMA_K, MFMA_BATCH, __half>;
    using ResultLayout       = MatrixLayout<MFMA_N, MFMA_K, MFMA_BATCH, float>;


    constexpr int GROUP_SIZE_4 = GROUP_SIZE / 4; // 1

    // Number of input columns loaded by each workgroup (output columns plus halo)
    constexpr int BLOCK_W = cfg.block_q() + (cfg.kw - 1);

    // uint4 vectors per channels fiber
    constexpr int BLOCK_C8 = cfg.block_c() / 8;

    // Vectors to store.
    constexpr int STORE_VECS = cfg.block_q() * BLOCK_C8;

    constexpr int NUM_INPUT_LDS_BUFFERS = 2;

    // Size of LDS buffer for inputs.
    constexpr int INPUT_LDS_BUFFER_SIZE_C8 = BLOCK_C8 * BLOCK_W;

    constexpr int INPUT_LDS_BUFFER_SIZE_C4 = INPUT_LDS_BUFFER_SIZE_C8 * 2;

    // Size of LDS buffer for outputs.
    constexpr int OUTPUT_LDS_BUFFER_SIZE = BLOCK_C8 * cfg.block_q();

    // LDS buffer for staging loads from global memory.
    __shared__ uint4 input_lds[NUM_INPUT_LDS_BUFFERS * INPUT_LDS_BUFFER_SIZE_C8];

    // LDS buffer for staging stores to global memory.
    __shared__ uint4 output_lds[OUTPUT_LDS_BUFFER_SIZE];

    const int tid  = threadIdx.x;
    const int wave = tid / WAVE_SIZE;
    const int lane = tid % WAVE_SIZE;

    // Workgroup coordinates.
    const int block_q_n_idx   = blockIdx.x;
    const int block_n_mod_idx = block_q_n_idx % cfg.n_fold;
    const int block_q_idx     = block_q_n_idx / cfg.n_fold;
    const int block_group_idx = blockIdx.y;
    const int block_n_div_idx = blockIdx.z;
    const int block_n_idx     = block_n_div_idx * cfg.n_fold + block_n_mod_idx;
    if(block_n_idx >= N)
        return;

    const int block_n     = block_n_idx;
    const int block_q     = block_q_idx * cfg.block_q();
    const int block_group = block_group_idx * cfg.block_groups();
    const int block_k     = block_group * GROUP_SIZE;
    const int block_c8    = block_k / 8; // in uint4 units

    // Base pointer for this batch image in NHWC layout (all in uint4 units)
    const int C  = groups * GROUP_SIZE;
    const int C8 = C / 8;

    const size_t offset_block = (size_t)block_n * hi * wi * C8;
    const size_t wi_stride    = (size_t)wi * C8;
    const size_t wo_stride    = (size_t)wo * C8;

    // Map threads to LDS input buffer addresses linearly.
    // This supports __builtin_amdgcn_raw_ptr_buffer_load_lds
    auto store_input_lds = &input_lds[tid];

    // Create the input buffer resource.
    auto input_bytes               = static_cast<size_t>(N) * hi * wi * C * sizeof(half);
    constexpr int rsrc_data_format = 1 << 15;
    auto input_rsrc                = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<_Float16*>(in), 0, input_bytes, rsrc_data_format);
    uint32_t input_voffset;

    // Map threads to global memory input buffer using a swizzle.
    using Sw             = SwizzleT<cfg.block_c()>;
    const int col        = Sw::x(tid);
    const int c8_thread  = Sw::c8(tid);
    const int global_col = (block_q - px) + col;
    // Trailing threads in partial waves are masked by EXEC (buffer_load_lds
    // honors the active mask; see ISA §9.1.9 "Memory Buffer Load to LDS").
    const bool load_active = (tid < (BLOCK_W)*BLOCK_C8);
    if(0 <= global_col && global_col < wi)
    {
        input_voffset =
            sizeof(uint4) * (offset_block + (size_t)global_col * C8 + block_c8 + c8_thread);
    }
    else
    {
        input_voffset = input_bytes;
    }

    // Load first chunk of inputs asynchronously.
    if(load_active)
    {
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            input_rsrc, store_input_lds, 16, input_voffset, 0, 0, 0);
    }

    // Each wave computes 16 groups of channels and 4 columns of the output tensor.
    auto wave_c64 = wave % cfg.waves_c64;
    auto wave_q4  = wave / cfg.waves_c64;

    // Map lanes to MFMA matrix coordinates, incorporating the wave's Q offset.
    const int thread_q   = wave_q4 * WARP_Q + ResultLayout::outer(lane);
    const int lane_c4    = ResultLayout::inner(lane) / 4;
    const int lane_batch = ResultLayout::batch(lane);

    // Map threads to output addresses for transferring through LDS to global memory.
    const bool store_active      = (tid < STORE_VECS);
    const uint4* load_output_lds = nullptr;
    uint4* store_output_global   = nullptr;
    if(store_active)
    {
        const int c8    = tid % BLOCK_C8;
        const int q     = block_q + col;
        load_output_lds = &output_lds[Sw::offset_uint4(col, c8)];
        if(q < wo)
        {
            const int K8        = C / 8;
            store_output_global = reinterpret_cast<uint4*>(out) + (size_t)block_n * ho * wo * K8 +
                                  (size_t)q * K8 + block_c8 + c8;
        }
    }

    // Load weights from global memory to mfma matrices in registers.
    fp16x4_t weights_reg[cfg.kh * cfg.kw];
    {
        auto lane_k     = OperandLayout::outer(lane);
        auto lane_c4    = OperandLayout::inner(lane) / 4;
        auto lane_batch = OperandLayout::batch(lane);
        auto k          = block_k + wave_c64 * 64 + lane_batch * GROUP_SIZE + lane_k;
        auto weights_global =
            reinterpret_cast<const uint2*>(wei) + k * (cfg.kh * cfg.kw * GROUP_SIZE_4) + lane_c4;
        for(int khw = 0; khw < cfg.kh * cfg.kw; khw++)
        {
            weights_reg[khw] = *(const fp16x4_t*)(weights_global + khw * GROUP_SIZE_4);
        }
    }

    // Pre-compute the kw input LDS offsets (uint2 units) for this thread.
    // lane_q, wave_group, and lane_c4 are constant per thread; hoisting avoids
    // recomputing the swizzle (which contains %) inside the hot loop.
    int input_lds_offsets[cfg.kw];
    static_for<cfg.kw>(
        [&]<int S>()
        {
            input_lds_offsets[S] =
                Sw::offset_uint2(thread_q + S, wave_c64 * 16 + lane_batch * GROUP_SIZE_4 + lane_c4);
        });

    // Pre-compute the output LDS swizzle offset (thread-constant).
    const int output_lds_offset =
        Sw::offset_uint2(thread_q, wave_c64 * 16 + lane_batch * GROUP_SIZE_4 + lane_c4);


    // Circular buffer of accumulators.
    constexpr auto Zero = fp32x4_t{0.f, 0.f, 0.f, 0.f};
    fp32x4_t acc[cfg.kh];
    for(int i = 0; i < cfg.kh; i++)
        acc[i] = Zero;

    int tic = 1;
    int toc = 0;

    // Main loop iterates over input rows.
    for(int y_base = 0; y_base + cfg.kh <= hi; y_base += cfg.kh)
    {
        // Inner loop unrolls the main loop by a factor of kh, the number of filter rows.
        // static_for makes the loop indices compile-time constants.

        static_for<cfg.kh>(
            [&]<int Y_LOCAL>()
            {
                asm volatile("s_waitcnt vmcnt(0)\n");
                __syncthreads();

                int y = y_base + Y_LOCAL;

                if(load_active && (y + 1) < hi)
                {
                    __builtin_amdgcn_raw_ptr_buffer_load_lds(
                        input_rsrc,
                        store_input_lds + tic * INPUT_LDS_BUFFER_SIZE_C8,
                        16,
                        input_voffset + (y + 1) * wi_stride * sizeof(uint4),
                        0,
                        0,
                        0);
                }

                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        const uint2* lds_ptr = reinterpret_cast<const uint2*>(input_lds) +
                                               toc * INPUT_LDS_BUFFER_SIZE_C4 +
                                               input_lds_offsets[S];
                        auto input_reg       = *(const fp16x4_t*)lds_ptr;

                        static_for<cfg.kh>(
                            [&]<int R>()
                            {
                                // Compute the position of the output row in the circular buffer of
                                // accumulators. This is a constexpr because of the Y_LOCAL and R
                                // are compile-time constants. Therefore acc[p_idx] compiles to a
                                // register address.
                                constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                                acc[p_idx]          = __builtin_amdgcn_mfma_f32_4x4x4f16(
                                    weights_reg[R * cfg.kw + S], input_reg, acc[p_idx], 0, 0, 0);
                            });
                    });

                tic ^= 1;
                toc ^= 1;

                constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                int p_out             = y + py - (cfg.kh - 1);
                if(p_out >= 0 && p_out < ho)
                {
                    __half2 halves[2];
                    halves[0]    = __float22half2_rn({acc[P_FLUSH][0], acc[P_FLUSH][1]});
                    halves[1]    = __float22half2_rn({acc[P_FLUSH][2], acc[P_FLUSH][3]});
                    auto out_reg = *reinterpret_cast<const fp16x4_t*>(halves);

                    auto* store_output_lds              = reinterpret_cast<uint2*>(output_lds);
                    store_output_lds[output_lds_offset] = *reinterpret_cast<const uint2*>(&out_reg);

                    __syncthreads(); // load_output_lds fully written

                    if(store_output_global)
                        store_output_global[p_out * wo_stride] = *load_output_lds;
                }
                acc[P_FLUSH] = Zero;
            });
    }

    // Remainder: the hi % kh leftover rows share the same structure as the main
    // loop — y_base is a multiple of kh so Y_LOCAL = y % kh, giving constexpr P.
    {
        int y_rem_base = (hi / cfg.kh) * cfg.kh;
        static_for<cfg.kh>(
            [&]<int Y_LOCAL>()
            {
                if(Y_LOCAL >= hi % cfg.kh)
                    return;
                int y = y_rem_base + Y_LOCAL;

                asm volatile("s_waitcnt vmcnt(0)\n");
                __syncthreads();

                if(load_active && (y + 1) < hi)
                {
                    __builtin_amdgcn_raw_ptr_buffer_load_lds(
                        input_rsrc,
                        store_input_lds + tic * INPUT_LDS_BUFFER_SIZE_C8,
                        16,
                        input_voffset + (y + 1) * wi_stride * sizeof(uint4),
                        0,
                        0,
                        0);
                }

                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        const uint2* lds_ptr = reinterpret_cast<const uint2*>(input_lds) +
                                               toc * INPUT_LDS_BUFFER_SIZE_C4 +
                                               input_lds_offsets[S];
                        fp16x4_t input_reg   = *(const fp16x4_t*)lds_ptr;

                        static_for<cfg.kh>(
                            [&]<int R>()
                            {
                                constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                                acc[p_idx]          = __builtin_amdgcn_mfma_f32_4x4x4f16(
                                    weights_reg[R * cfg.kw + S], input_reg, acc[p_idx], 0, 0, 0);
                            });
                    });

                tic ^= 1;
                toc ^= 1;

                constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                int p_out             = y + py - (cfg.kh - 1);
                if(p_out >= 0 && p_out < ho)
                {
                    __half2 halves[2];
                    halves[0]    = __float22half2_rn({acc[P_FLUSH][0], acc[P_FLUSH][1]});
                    halves[1]    = __float22half2_rn({acc[P_FLUSH][2], acc[P_FLUSH][3]});
                    auto out_reg = *reinterpret_cast<const fp16x4_t*>(halves);

                    auto* store_output_lds              = reinterpret_cast<uint2*>(output_lds);
                    store_output_lds[output_lds_offset] = *reinterpret_cast<const uint2*>(&out_reg);

                    __syncthreads(); // output_lds fully written

                    if(store_output_global)
                        store_output_global[p_out * wo_stride] = *load_output_lds;
                }
                acc[P_FLUSH] = Zero;
            });
    }

    // Flush output rows whose last input contribution (at filter row kh-1) would land at
    // y >= hi (out of bounds). These rows were never flushed by the main/remainder loops.
    // An output row p_out is last touched at input row y = p_out + kh-1 - py; it needs
    // flushing when that y >= hi, i.e. p_out >= hi - kh + 1 + py.
    //
    // Synchronize output LDS buffer access.
    __syncthreads();
    for(int p_out = hi - cfg.kh + 1 + py; p_out < ho; p_out++)
    {
        int p_idx = (p_out - py + cfg.kh) % cfg.kh;
        fp32x4_t slot;
        dispatch<cfg.kh>(p_idx,
                         [&]<int P>()
                         {
                             slot   = acc[P];
                             acc[P] = Zero;
                         });

        __half2 halves[2];
        halves[0]    = __float22half2_rn({slot[0], slot[1]});
        halves[1]    = __float22half2_rn({slot[2], slot[3]});
        auto out_reg = *reinterpret_cast<const fp16x4_t*>(halves);

        auto* store_output_lds              = reinterpret_cast<uint2*>(output_lds);
        store_output_lds[output_lds_offset] = *reinterpret_cast<const uint2*>(&out_reg);

        __syncthreads(); // output_lds fully written

        if(store_output_global)
            store_output_global[p_out * wo_stride] = *load_output_lds;
    }
}

template <grouped_4c::Config cfg>
__global__ void conv2d_grouped_4c_fp16_nhwc_cdna4(const _Float16* __restrict__ in,
                                                  const _Float16* __restrict__ wei,
                                                  double alpha,
                                                  double beta,
                                                  _Float16* __restrict__ out,
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
    conv2d_grouped_4c_fp16_cdna4_nhwc_impl<cfg>(in,
                                                wei,
                                                alpha,
                                                beta,
                                                out,
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

namespace grouped_4c
{
template <size_t... Is>
void launch_dispatch(int config_idx,
                     std::index_sequence<Is...>,
                     const LaunchParams& lp,
                     const hipconv::Conv2dParams& par,
                     const void* in,
                     const void* wei,
                     void* out,
                     hipStream_t stream)
{
    auto kernel_launch = [&]<size_t I>()
    {
        conv2d_grouped_4c_fp16_nhwc_cdna4<configs[I]>
            <<<lp.grid, lp.block_size, lp.dynamic_shared_bytes, stream>>>(
                static_cast<const _Float16*>(in),
                static_cast<const _Float16*>(wei),
                1.0,
                0.0,
                static_cast<_Float16*>(out),
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
                   hipStream_t stream)
{
    launch_dispatch(
        config_idx, std::make_index_sequence<NUM_CONFIGS>{}, lp, par, in, wei, out, stream);
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
            if(par.out_type != hipconv::DataType::fp16)
                return false;
            if(par.order != hipconv::TensorOrder::NHWC)
                return false;
            if(par.direction != hipconv::Direction::Fprop)
                return false;
            if(par.kh != 3 || par.kw != 3)
                return false;
            if(par.k != par.c)
                return false;
            if(par.channels_per_group() != 4)
                return false;
            if(par.c % 4 != 0)
                return false;
            if(par.stride_h != 1 || par.stride_w != 1)
                return false;
            return true;
        },
        .config_is_compatible = [](const hipconv::Conv2dParams& par, int idx)
        { return is_valid_config(par, configs[idx]); },
        .get_launch_params = &get_launch_params,
        .launch            = &launch,
        .num_configs       = NUM_CONFIGS,
    };
}

} // namespace grouped_4c
