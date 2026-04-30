// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

// Packed vector instructions for SageAttention V2 optimization (Phase 1a)
// These functions wrap AMD GPU packed FP32 instructions (v_pk_add_f32, v_pk_fma_f32)
// Hardware support: GCN 3.0+, RDNA, CDNA (gfx90a, gfx940+)

namespace ck_tile {
namespace sageattn_pk {

// Packed FP32 addition: performs 2x fp32 additions in a single instruction
// Hardware: v_pk_add_f32 (GCN 3.0+, RDNA, CDNA)
// result[31:0]  = lhs[31:0]  + rhs[31:0]
// result[63:32] = lhs[63:32] + rhs[63:32]
CK_TILE_DEVICE fp32x2_t pk_add_f32(fp32x2_t lhs, fp32x2_t rhs)
{
    fp32x2_t result;
    asm volatile("v_pk_add_f32 %[result], %[lhs], %[rhs]"
                 : [result] "=v"(result)
                 : [lhs] "v"(lhs), [rhs] "v"(rhs));
    return result;
}

// Packed FP32 FMA: performs 2x fp32 fused multiply-add in a single instruction
// Hardware: v_pk_fma_f32 (GCN 3.0+, RDNA, CDNA)
// result[31:0]  = a[31:0]  * b[31:0]  + c[31:0]
// result[63:32] = a[63:32] * b[63:32] + c[63:32]
CK_TILE_DEVICE fp32x2_t pk_fma_f32(fp32x2_t a, fp32x2_t b, fp32x2_t c)
{
    fp32x2_t result;
    asm volatile("v_pk_fma_f32 %[result], %[a], %[b], %[c]"
                 : [result] "=v"(result)
                 : [a] "v"(a), [b] "v"(b), [c] "v"(c));
    return result;
}

#ifdef CK_TILE_SAGEATTN_DEBUG_SCALAR
// Debug version: scalar fallback for verification
CK_TILE_DEVICE fp32x2_t pk_add_f32_debug(fp32x2_t lhs, fp32x2_t rhs)
{
    return fp32x2_t{lhs[0] + rhs[0], lhs[1] + rhs[1]};
}

CK_TILE_DEVICE fp32x2_t pk_fma_f32_debug(fp32x2_t a, fp32x2_t b, fp32x2_t c)
{
    return fp32x2_t{a[0] * b[0] + c[0], a[1] * b[1] + c[1]};
}
#endif

} // namespace sageattn_pk
} // namespace ck_tile
