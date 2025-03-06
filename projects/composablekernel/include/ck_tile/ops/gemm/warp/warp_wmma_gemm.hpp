#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma.hpp"

namespace ck_tile {

template <bool kTransLdA = false, bool kTransLdB = false, bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_f16_f16 =
    WarpGemmImpl<WarpGemmAtrributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_f16_f16,
                                       kTransLdA,
                                       kTransLdB,
                                       kTransC>>;

template <bool kTransLdA = false, bool kTransLdB = false, bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_bf16_bf16 =
    WarpGemmImpl<WarpGemmAtrributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf16_bf16,
                                       kTransLdA,
                                       kTransLdB,
                                       kTransC>>;

template <bool kTransLdA = false, bool kTransLdB = false, bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_fp8_fp8 =
    WarpGemmImpl<WarpGemmAtrributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_fp8_fp8,
                                       kTransLdA,
                                       kTransLdB,
                                       kTransC>>;

template <bool kTransLdA = false, bool kTransLdB = false, bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_bf8_bf8 =
    WarpGemmImpl<WarpGemmAtrributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_bf8,
                                       kTransLdA,
                                       kTransLdB,
                                       kTransC>>;

using WarpGemmWmma_f32_16x16x16_f16_f16_gfx12 =
    WarpGemmImpl<WarpGemmAtrributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_f16_f16_gfx12>>;
} // namespace ck_tile
