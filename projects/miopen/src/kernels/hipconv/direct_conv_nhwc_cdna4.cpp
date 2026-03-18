// Direct convolution kernel for CDNA4 (gfx950/MI400) GPUs.
// NHWC layout, FP16/BF16, 3x3 filter, stride 1x1, no groups.
// Uses MFMA 16x16x32 instructions and double-buffered LDS.

#include "bunnies.hpp"
#include "bunnies_cdna4.hpp"
#include "types.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <array>

namespace hipconv_direct
{

// Kernel configuration parameters.
struct Config
{
    using arch = bunnies::arch_cdna4;

    KernelDataType type;

    static constexpr int wmma_size_w = 16;
    static constexpr int wmma_size_k = 16;
    static constexpr int wmma_size_c = 32;
    static constexpr int tile_size_h = 16;
    static constexpr int tile_size_w = 16;
    static constexpr int tile_size_k = 256;
    static constexpr int tile_size_c = 64;
    static constexpr int tiles_h     = 4;
    static constexpr int tiles_w     = 1;
    static constexpr int tiles_k     = 2;
    static constexpr int pad         = 1;

    constexpr int tiles() const { return tiles_h * tiles_w * tiles_k; }
};

template <Config cfg>
__device__ void conv2d_direct_cdna4_nhwc_fwd_impl(const ToType<cfg.type>* __restrict__ in,
                                                  const ToType<cfg.type>* __restrict__ wei,
                                                  double alpha,
                                                  double beta,
                                                  ToType<cfg.type>* __restrict__ out,
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
    using arch                    = Config::arch;
    using conv_dtype              = ToType<cfg.type>;
    constexpr int tiles           = cfg.tiles();
    constexpr int reg_stride_h    = cfg.tiles_h;
    constexpr int reg_stride_w    = cfg.wmma_size_w * cfg.tiles_w;
    constexpr int reg_stride_k    = cfg.wmma_size_k * cfg.tiles_k;
    constexpr int reg_tiles_h     = cfg.tile_size_h / reg_stride_h;
    constexpr int reg_tiles_w     = cfg.tile_size_w / reg_stride_w;
    constexpr int reg_tiles_k     = cfg.tile_size_k / reg_stride_k;
    constexpr int reg_tiles_c     = cfg.tile_size_c / cfg.wmma_size_c;
    constexpr int tile_size_h_pad = cfg.tile_size_h + 2 * cfg.pad;
    constexpr int tile_size_w_pad = cfg.tile_size_w + 2 * cfg.pad;
    using rt_a =
        bunnies::reg_tile<conv_dtype, 16, 32, reg_tiles_h, reg_tiles_c, bunnies::use::A, arch>;
    using rt_b =
        bunnies::reg_tile<conv_dtype, 32, 16, reg_tiles_c, reg_tiles_k / 2, bunnies::use::B, arch>;
    using rt_c =
        bunnies::reg_tile<float, 16, 16, reg_tiles_h, reg_tiles_k / 2, bunnies::use::Acc, arch>;

    const int wave_id     = bunnies::wave_id<arch>();
    const int wave_id_k   = wave_id % cfg.tiles_k;
    const int wave_id_w   = wave_id / cfg.tiles_k % cfg.tiles_w;
    const int wave_id_h   = wave_id / (cfg.tiles_k * cfg.tiles_w);
    const int wave_group  = wave_id / 4;
    const int lane        = bunnies::lane_id<arch>();
    const int num_h_tiles = 1 + (ho - 1) / cfg.tile_size_h;
    const int tile_n      = blockIdx.z / num_h_tiles;
    const int tile_h      = (blockIdx.z % num_h_tiles) * cfg.tile_size_h;
    const int tile_w      = blockIdx.y * cfg.tile_size_w;
    const int tile_k      = blockIdx.x * cfg.tile_size_k;

    const int K                 = k_per_group * groups;
    const int C                 = c_per_group * groups;
    const auto in_stride_w      = C;
    const auto in_stride_h      = wi * in_stride_w;
    const auto in_stride_n      = hi * in_stride_h;
    const auto weights_stride_s = C;
    const auto weights_stride_r = fx * weights_stride_s;
    const auto weights_stride_k = fy * weights_stride_r;
    const auto out_stride_w     = K;
    const auto out_stride_h     = wo * out_stride_w;
    const auto out_stride_n     = hi * out_stride_h;

    rt_c c_acc[2] = {};

    __shared__ conv_dtype in_lds0[tile_size_h_pad * tile_size_w_pad * cfg.tile_size_c];
    __shared__ conv_dtype in_lds1[tile_size_h_pad * tile_size_w_pad * cfg.tile_size_c];
    auto in_lds = std::array<conv_dtype*, 2>{&in_lds0[0], &in_lds1[0]};
    auto in_lds_view =
        bunnies::make_view_row_major<2>({tile_size_h_pad, tile_size_w_pad * cfg.tile_size_c});

    __shared__ conv_dtype weights_lds0[cfg.tile_size_k * cfg.tile_size_c];
    __shared__ conv_dtype weights_lds1[cfg.tile_size_k * cfg.tile_size_c];
    auto weights_lds      = std::array<conv_dtype*, 2>{&weights_lds0[0], &weights_lds1[0]};
    auto weights_lds_view = bunnies::make_view_row_major<3>(
        {2, cfg.tile_size_k / 2 / cfg.wmma_size_k, cfg.wmma_size_k * cfg.tile_size_c});

    auto const in_swizzle     = [](int w, int c) { return w * 64 ^ w % 8 * 8 ^ c; };
    auto const in_swizzle_inv = [](int offset) -> std::array<int, 2>
    { return {offset / 64, offset % 64 ^ offset / 64 % 8 * 8}; };
    auto in_ptr  = in + tile_n * in_stride_n;
    auto in_buf  = arch::make_buffer(in_ptr, in + N * in_stride_n - in_ptr);
    auto in_view = bunnies::make_memref<3>(in, {hi, wi, C}, {in_stride_h, in_stride_w, 1});

    auto const load_in_block = [&](int bufno, int c, int h_part, int w_part)
    {
        int h = h_part * tiles + wave_id;
        if(h < tile_size_h_pad)
        {
            int h0      = tile_h + h - cfg.pad;
            const int w = w_part * 8;
            if(w < cfg.tile_size_w)
            {
                auto* lds_ptr = in_lds[bufno] + in_lds_view(h, in_swizzle(w, 0));
                int in_offset_x4_w, in_offset_x4_c;
                asm volatile("v_lshrrev_b32 %0 3 %1\n" : "=v"(in_offset_x4_w) : "v"(lane));
                asm volatile("v_xor_b32 %0 %1 %2\n"
                             "v_and_b32 %0 %0 7\n"
                             "v_lshlrev_b32 %0 3 %0\n"
                             : "=v"(in_offset_x4_c)
                             : "v"(lane), "v"(in_offset_x4_w));
                int w0         = tile_w + w - cfg.pad + in_offset_x4_w;
                auto in_offset = h0 >= 0 && h0 < hi && w0 >= 0 && w0 < wi
                                     ? (h0 * in_stride_h + w0 * in_stride_w + in_offset_x4_c + c) *
                                           sizeof(conv_dtype)
                                     : -1;
                __builtin_amdgcn_raw_ptr_buffer_load_lds(in_buf, lds_ptr, 16, in_offset, 0, 0, 0);
            }
            if(w >= cfg.tile_size_w)
            {
                auto* lds_ptr = in_lds[bufno] + in_lds_view(h, in_swizzle(w, 0));
                int in_offset_x1_w, in_offset_x1_c;
                asm volatile("v_lshrrev_b32 %0 5 %1\n" : "=v"(in_offset_x1_w) : "v"(lane));
                asm volatile("v_lshlrev_b32 %0 2 %2\n"
                             "v_xor_b32 %0 %1 %0\n"
                             "v_and_b32 %0 %0 31\n"
                             "v_lshlrev_b32 %0 1 %0\n"
                             : "=v"(in_offset_x1_c)
                             : "v"(lane), "v"(in_offset_x1_w));
                int w0         = tile_w + w - cfg.pad + in_offset_x1_w;
                auto in_offset = h0 >= 0 && h0 < hi && w0 >= 0 && w0 < wi
                                     ? (h0 * in_stride_h + w0 * in_stride_w + in_offset_x1_c + c) *
                                           sizeof(conv_dtype)
                                     : -1;
                __builtin_amdgcn_raw_ptr_buffer_load_lds(in_buf, lds_ptr, 4, in_offset, 0, 0, 0);
            }
        }
    };

    auto const weights_swizzle     = [](int k, int c) { return k * 64 ^ k % 8 * 8 ^ c; };
    auto const weights_swizzle_inv = [](int offset) -> std::array<int, 2>
    { return {offset / 64, offset % 64 ^ offset / 64 % 8 * 8}; };
    auto weights_ptr  = wei + tile_k * weights_stride_k;
    auto weights_size = wei + K * weights_stride_k - weights_ptr;
    auto weights_buf  = arch::make_buffer(weights_ptr, weights_size);
    auto weights_view = bunnies::make_view<4>(
        0, {K - tile_k, 3, 3, C}, {weights_stride_k, weights_stride_r, weights_stride_s, 1});
    auto const load_weights_block = [&](int bufno, int r, int s, int c, int k_part)
    {
        auto view =
            weights_view.subview(bunnies::slice{0, cfg.tile_size_k}, r, s, bunnies::slice{0, C});
        using weights_cfg = bunnies::
            buffer_load_to_lds_config<arch, cfg.tile_size_k / 2, cfg.tile_size_c, 16, tiles>;
        bunnies::buffer_load_to_lds<weights_cfg>(wave_id,
                                                 weights_buf,
                                                 view,
                                                 {cfg.tile_size_k / 2 * k_part, c},
                                                 weights_swizzle_inv,
                                                 weights_lds[bufno],
                                                 weights_lds_view(k_part, 0, 0));
    };

    auto const in_lds_layout = [&](int r, int s)
    {
        return [&, r, s](int hb, int cb, int w, int c)
        {
            c += cb * cfg.wmma_size_c;
            return in_lds_view(cfg.tiles_h * hb + wave_id_h + r, in_swizzle(w + s, c));
        };
    };

    auto const weights_lds_layout = [&](int part_k)
    {
        return [&, part_k](int cb, int kb, int c, int k)
        {
            c += cb * cfg.wmma_size_c;
            return weights_lds_view(part_k, kb * cfg.tiles_k + wave_id_k, weights_swizzle(k, c));
        };
    };

#pragma unroll
    for(int h_part = 0; h_part < 1 + (tile_size_h_pad - 1) / tiles; ++h_part)
    {
#pragma unroll
        for(int w_part = 0; w_part < 1 + (tile_size_w_pad - 1) / 8; ++w_part)
        {
            load_in_block(0, 0, h_part, w_part);
            load_in_block(0, 0, h_part, w_part);
            load_in_block(0, 0, h_part, w_part);
        }
    }
    load_weights_block(0, 0, 0, 0, 0);
    load_weights_block(0, 0, 0, 0, 1);

    rt_a a;
    rt_b b[2];

    asm volatile("s_waitcnt vmcnt(0)\n");
    __builtin_amdgcn_s_barrier();

    if(wave_group == 1)
    {
        __builtin_amdgcn_s_barrier();
    }
#pragma unroll 1
    for(int c2 = 0; c2 < C; c2 += 2 * cfg.tile_size_c)
    {
        int buf_in      = 0;
        int buf_weights = 0;
#pragma unroll
        for(int c = c2; c < c2 + 2 * cfg.tile_size_c; c += cfg.tile_size_c)
        {
#pragma unroll
            for(int r = 0; r < 3; ++r)
            {
#pragma unroll
                for(int s = 0; s < 3; ++s)
                {
                    lds_load<arch::ds_load_b128>(a, in_lds[buf_in], in_lds_layout(r, s));
                    lds_load<arch::ds_load_b128>(
                        b[0], weights_lds[buf_weights], weights_lds_layout(0));
                    const int snext = (s + 1);
                    const int rnext = r + snext / 3;
                    const int cnext = c + (rnext / 3) * cfg.tile_size_c;
                    load_weights_block(buf_weights ^ 1, rnext % 3, snext % 3, cnext, 0);
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);

                    __builtin_amdgcn_s_setprio(1);
                    mma(c_acc[0], a, b[0], c_acc[0]);
                    __builtin_amdgcn_s_setprio(0);
                    asm volatile("s_waitcnt vmcnt(2)\n");
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);

                    lds_load<arch::ds_load_b128>(
                        b[1], weights_lds[buf_weights], weights_lds_layout(1));
                    load_weights_block(buf_weights ^ 1, rnext % 3, snext % 3, cnext, 1);
                    if(r < 3 && s < 3)
                    {
                        load_in_block(buf_in ^ 1, c + cfg.tile_size_c, r, s);
                    }
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);

                    __builtin_amdgcn_s_setprio(1);
                    mma(c_acc[1], a, b[1], c_acc[1]);
                    __builtin_amdgcn_s_setprio(0);
                    asm volatile("s_waitcnt vmcnt(3)\n");
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);

                    buf_weights ^= 1;
                }
            }
            buf_in ^= 1;
        }
    }

    if(wave_group == 0)
    {
        __builtin_amdgcn_s_barrier();
    }

    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();

    auto out_sub =
        bunnies::make_memref<4>(out, {N, ho, wo, K}, {out_stride_n, out_stride_h, out_stride_w, 1})
            .subview(tile_n,
                     bunnies::slice{0, ho},
                     bunnies::slice{0, wo},
                     bunnies::slice{tile_k + wave_id_k * cfg.wmma_size_k, cfg.tile_size_k});

#pragma unroll
    for(int rh = 0; rh < reg_tiles_h; ++rh)
    {
#pragma unroll
        for(int rk = 0; rk < reg_tiles_k; ++rk)
        {
            const auto h = tile_h + wave_id_h + rh * reg_stride_h;
            if(h < ho)
            {
                for(int item = 0; item < 4; ++item)
                {
                    auto [w, k] = rt_c::layout::map({lane, item});
                    w += tile_w;
                    k += reg_stride_k * rk;
                    if(w < wo)
                    {
                        *out_sub(h, w, k) = static_cast<conv_dtype>(
                            c_acc[rk / (reg_tiles_k / 2)].data[rh][rk % (reg_tiles_k / 2)][item]);
                    }
                }
            }
        }
    }
}

template <Config cfg>
__launch_bounds__(cfg.tiles() * Config::arch::wave_size, 2) __global__
    void conv2d_direct_nhwc_cdna4_fwd(const ToType<cfg.type>* __restrict__ in,
                                      const ToType<cfg.type>* __restrict__ wei,
                                      double alpha,
                                      double beta,
                                      ToType<cfg.type>* __restrict__ out,
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
    conv2d_direct_cdna4_nhwc_fwd_impl<cfg>(in,
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

} // namespace hipconv_direct

// extern "C" wrapper entry points for MIOpen HIP RTC (non-template, named kernels)
extern "C" __launch_bounds__(hipconv_direct::Config{.type = KernelDataType::fp16}.tiles() *
                                 hipconv_direct::Config::arch::wave_size,
                             2) __global__
    void conv2d_direct_nhwc_cdna4_fwd_fp16(const fp16_t* __restrict__ in,
                                           const fp16_t* __restrict__ wei,
                                           double alpha,
                                           double beta,
                                           fp16_t* __restrict__ out,
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
    hipconv_direct::conv2d_direct_nhwc_cdna4_fwd<
        hipconv_direct::Config{.type = KernelDataType::fp16}>(in,
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

extern "C" __launch_bounds__(hipconv_direct::Config{.type = KernelDataType::bf16}.tiles() *
                                 hipconv_direct::Config::arch::wave_size,
                             2) __global__
    void conv2d_direct_nhwc_cdna4_fwd_bf16(const bf16_t* __restrict__ in,
                                           const bf16_t* __restrict__ wei,
                                           double alpha,
                                           double beta,
                                           bf16_t* __restrict__ out,
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
    hipconv_direct::conv2d_direct_nhwc_cdna4_fwd<
        hipconv_direct::Config{.type = KernelDataType::bf16}>(in,
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
