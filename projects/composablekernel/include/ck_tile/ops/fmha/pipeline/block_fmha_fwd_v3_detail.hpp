// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline_default_policy.hpp"

#if !defined(CK_TILE_FMHA_V3_ENABLE_ASM_MARKER)
#define CK_TILE_FMHA_V3_ENABLE_ASM_MARKER 1
#endif

#if !defined(CK_TILE_FMHA_V3_ASM_MARKER)
#if CK_TILE_FMHA_V3_ENABLE_ASM_MARKER
#define CK_TILE_FMHA_V3_ASM_MARKER(marker) \
    __builtin_amdgcn_sched_barrier(0);     \
    asm volatile("; [POYENC] " #marker);   \
    __builtin_amdgcn_sched_barrier(0);
#else
#define CK_TILE_FMHA_V3_ASM_MARKER(marker)
#endif
#endif

#if !defined(CK_TILE_FMHA_V3_ADD_SBARRIER_FOR_PHASE0)
#define CK_TILE_FMHA_V3_ADD_SBARRIER_FOR_PHASE0 1
#endif

namespace ck_tile {

// ---------------------------------------------------------------------------
// block_gemm_mfma_count_v: number of hardware MFMA instructions issued per
// warp in one full BlockGemm call.
//
//   warp gemm calls = MIterPerWarp * NIterPerWarp * KIterPerWarp
//   MFMAs per call  = WarpGemm::kK / WarpGemm::WarpGemmAttribute::Impl::kK  (kKIter)
//
// For bf16/fp16 kKIter=1; for fp8 kKIter=2 (K=32 warp gemm wraps 2x K=16 MFMA).
// ---------------------------------------------------------------------------
template <typename BlockGemm>
static constexpr ck_tile::index_t block_gemm_mfma_count_v =
    BlockGemm::MIterPerWarp * BlockGemm::NIterPerWarp * BlockGemm::KIterPerWarp *
    (BlockGemm::WarpGemm::kK / BlockGemm::WarpGemm::WarpGemmAttribute::Impl::kK);

// ---------------------------------------------------------------------------
// CoreLoopSchedulingParams: auto-derived instruction counts from tile/gemm config
// ---------------------------------------------------------------------------
template <typename PipelineProblem, typename Policy = BlockFmhaV3PipelineDefaultPolicy>
struct CoreLoopSchedulingParams
{
    using QKBlockGemm =
        ck_tile::remove_cvref_t<decltype(Policy::template GetQKBlockGemm<PipelineProblem>())>;
    using PVBlockGemm =
        ck_tile::remove_cvref_t<decltype(Policy::template GetPVBlockGemm<PipelineProblem>())>;

    static constexpr ck_tile::index_t kMfmaPerWarpGemm0 = block_gemm_mfma_count_v<QKBlockGemm>;
    static constexpr ck_tile::index_t kMfmaPerWarpGemm1 = block_gemm_mfma_count_v<PVBlockGemm>;

    static constexpr bool kIsMasking = PipelineProblem::FmhaMask::IsMasking;
};

// ---------------------------------------------------------------------------
// VALU intrinsic wrappers: inline asm anchors for instruction scheduling.
//
// These ensure the compiler does not sink/hoist specific VALU instructions
// across sched_barrier boundaries. Used by both fmha_fwd V3 and batch_prefill
// V3 pipelines.
// ---------------------------------------------------------------------------
namespace detail {

CK_TILE_DEVICE float fma_impl_vsv(float a, float b, float c) { return a * b + c; }

CK_TILE_DEVICE float add_impl_vv(float lhs, float rhs)
{
    float result;
    asm volatile("v_add_f32_e32 %[result], %[lhs], %[rhs]"
                 : [result] "=v"(result)
                 : [lhs] "v"(lhs), [rhs] "v"(rhs));
    return result;
}

CK_TILE_DEVICE float mul_impl_vv(float lhs, float rhs)
{
    float result;
    asm volatile("v_mul_f32_e32 %[result], %[lhs], %[rhs]"
                 : [result] "=v"(result)
                 : [lhs] "v"(lhs), [rhs] "v"(rhs));
    return result;
}

CK_TILE_DEVICE fp16x2_t cvt_pk_fp16_f32(float a, float b)
{
    fp16x2_t result;
    asm volatile("v_cvt_pk_f16_f32 %[result], %[a], %[b]"
                 : [result] "=v"(result)
                 : [a] "v"(a), [b] "v"(b));
    return result;
}

CK_TILE_DEVICE bf16x2_t cvt_pk_bf16_f32(float a, float b)
{
    bf16x2_t result;
    asm volatile("v_cvt_pk_bf16_f32 %[result], %[a], %[b]"
                 : [result] "=v"(result)
                 : [a] "v"(a), [b] "v"(b));
    return result;
}

CK_TILE_DEVICE fp32x2_t pk_mul_f32(fp32x2_t lhs, fp32x2_t rhs)
{
    fp32x2_t result;
    asm volatile("v_pk_mul_f32 %[result], %[lhs], %[rhs]"
                 : [result] "=v"(result)
                 : [lhs] "v"(lhs), [rhs] "v"(rhs));
    return result;
}

/// FP8 packed conversion with asm volatile to prevent code sinking.
/// This anchors the conversion instruction in Phase 0, and all predecessor
/// instructions (scale, saturate, NaN check) will automatically stay in Phase 0.
/// v_cvt_pk_fp8_f32 packs two FP8 values into lower 16 bits of a 32-bit VGPR.
CK_TILE_DEVICE uint32_t cvt_pk_fp8_f32(float a, float b)
{
    uint32_t result;
    asm volatile("v_cvt_pk_fp8_f32 %[result], %[a], %[b]"
                 : [result] "=v"(result)
                 : [a] "v"(a), [b] "v"(b));
    return result;
}

} // namespace detail

} // namespace ck_tile
