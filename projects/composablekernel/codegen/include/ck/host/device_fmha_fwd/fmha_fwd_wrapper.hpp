// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// This header is designed to be embedded and used at RTC compilation time.
//
// Phase 2 of the hipRTC migration: expanded from the original MGX 28-arg
// template to a superset covering the full fmha_fwd_traits surface. New
// parameters are appended after the MGX prefix with defaults that match
// MGX's hardcoded behaviour, so existing operation.cpp template strings
// compile unchanged.
//
// Coverage axes now represented:
//   - dtype (fp16, bf16, fp32; fp8 variants come in Phase 3)
//   - tile shape (6), block-warps x2 (6), warp-tile x2 (6) [MGX prefix]
//   - mask variants (no / top-left / bottom-right / window-generic)
//   - bias variants (no / elementwise / alibi)
//   - LSE output
//   - dropout (seed/offset variant)
//   - logits soft cap
//   - sink tokens + skip_min_seqlen_q
//   - group mode (variable-length) with seqstart/seqlen pointers
//   - GQA (nhead_q != nhead_k) via runtime descriptor field
//   - row-major vs col-major V layout
//   - three pipelines (QR, QR_ASYNC, QR_ASYNC_TRLOAD)
// Axes still deferred:
//   - fp8 descale/scale (Phase 3)
//   - V3 pipeline family (Phase 7, gfx950)
//   - trload backward (Phase 6)

#include <cmath>
#include <cstdint>
#include <cassert>
#include <type_traits>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"

namespace ck_tile {

enum class FmhaPipelineTag
{
    QR,                 // BlockFmhaPipelineQRKSVS
    QR_ASYNC,           // BlockFmhaPipelineQRKSVSAsync
    QR_ASYNC_TRLOAD,    // BlockFmhaPipelineQRKSVSAsyncTrload (gfx950+)
    // Phase 7: gfx950 V3 pipelines. BlockFmhaFwdV3Pipeline drives a
    // different kernel template (FmhaFwdV3Kernel) than the default
    // FmhaFwdKernel. The wrapper picks the matching Kernel via
    // std::conditional_t on these tags.
    V3,                 // BlockFmhaFwdV3Pipeline -> FmhaFwdV3Kernel
    QR_ASYNC_TRLOAD_V3  // Trload + V3 combination
};

// Wrapper-local mask enum. Mirrors the dispatcher's mask_enum but is
// separated from ck_tile's internal mask types because we want this to
// participate as a non-type template parameter on the wrapper.
enum class FmhaMaskType
{
    NoMask        = 0,
    TopLeft       = 1, // causal, top-left anchored
    BottomRight   = 2, // causal, bottom-right anchored
    WindowGeneric = 3  // arbitrary left/right bounds
};

// Wrapper-local bias enum. Maps to BlockAttentionBiasEnum.
enum class FmhaBiasType
{
    NoBias          = 0,
    ElementwiseBias = 1,
    ALiBi           = 2
};

// Wrapper-local quant-scale enum. Mirrors BlockAttentionQuantScaleEnum.
// Controls whether the kernel reads descale/scale tensors.
enum class FmhaQScaleType
{
    NoScale     = 0,
    PerTensor   = 1,
    BlockScale  = 2,
    KvBlockScale = 3
};

template <typename DataType_,
          // Block tile (MGX prefix)
          index_t kBM0,
          index_t kBN0,
          index_t kBK0,
          index_t kBN1,
          index_t kBK1,
          index_t kBK0Max,
          // Gemm0 block warps (MGX prefix)
          index_t kRM0,
          index_t kRN0,
          index_t kRK0,
          // Gemm1 block warps (MGX prefix)
          index_t kRM1,
          index_t kRN1,
          index_t kRK1,
          // Gemm0 warp tile (MGX prefix)
          index_t kWM0,
          index_t kWN0,
          index_t kWK0,
          // Gemm1 warp tile (MGX prefix)
          index_t kWM1,
          index_t kWN1,
          index_t kWK1,
          //
          bool kIsCausal,   // MGX: legacy bool; now means "use bottom-right causal"
          bool kIsVRowMajor,
          bool kHasBias,    // MGX: legacy bool; true means ElementwiseBias
          //
          bool kPadM,
          bool kPadN,
          bool kPadK,
          bool kPadO,
          //
          FmhaPipelineTag kPipelineTag,
          // --- Phase 2 additions (all default to MGX semantics) ---
          bool kHasLSE             = false,
          bool kHasDropout         = false,
          bool kHasLogitsSoftCap   = false,
          bool kHasSink            = false,
          bool kSkipMinSeqlenQ     = false,
          bool kIsGroupMode        = false,
          // Explicit mask/bias enums override kIsCausal/kHasBias when set to non-default.
          FmhaMaskType kMaskOverride = FmhaMaskType::NoMask,
          FmhaBiasType kBiasOverride = FmhaBiasType::NoBias,
          // Block-per-CU override (-1 = let ck_tile pick)
          index_t kBlockPerCu        = -1,
          // --- Phase 3: FP8 / mixed-precision / quant scale ---
          // Output data type. Defaults to DataType_ so homogeneous
          // dtype kernels (fp16, bf16, fp32, fp8, bf8) keep the MGX
          // behaviour. For mixed-precision FP8 outputs (e.g. fp8->fp16,
          // fp8->fp32, fp8->bf16), override with the desired O dtype.
          typename OutputDataType_ = DataType_,
          // Bias element type. Defaults to DataType_, except FP8/BF8
          // kernels want a float bias per FmhaFwdTypeConfig<FmhaFwdFp8*>.
          typename BiasDataType_   = DataType_,
          // Quant-scale configuration for FP8 static-quant support.
          FmhaQScaleType kQScaleType = FmhaQScaleType::NoScale>
struct FmhaFwdWrapper
{
    using BlockTile = sequence<kBM0, kBN0, kBK0, kBN1, kBK1, kBK0Max>;

    using Gemm0BlockWarps = sequence<kRM0, kRN0, kRK0>;
    using Gemm0WarpTile   = sequence<kWM0, kWN0, kWK0>;
    using Gemm1BlockWarps = sequence<kRM1, kRN1, kRK1>;
    using Gemm1WarpTile   = sequence<kWM1, kWN1, kWK1>;

    using FmhaShape = TileFmhaShape<BlockTile,
                                    Gemm0BlockWarps,
                                    Gemm0WarpTile,
                                    Gemm1BlockWarps,
                                    Gemm1WarpTile,
                                    kIsVRowMajor>;

    // Resolved mask type. If the enum override is non-default, honour it;
    // otherwise fall back to the legacy kIsCausal bool (which maps to
    // BottomRight to match MGX behaviour).
    static constexpr FmhaMaskType kResolvedMaskType =
        (kMaskOverride != FmhaMaskType::NoMask)
            ? kMaskOverride
            : (kIsCausal ? FmhaMaskType::BottomRight : FmhaMaskType::NoMask);

    static constexpr bool kHasMask = (kResolvedMaskType != FmhaMaskType::NoMask);

    // Resolved bias enum. If override is non-default, use it; otherwise
    // map legacy kHasBias to ElementwiseBias.
    static constexpr FmhaBiasType kResolvedBiasType =
        (kBiasOverride != FmhaBiasType::NoBias)
            ? kBiasOverride
            : (kHasBias ? FmhaBiasType::ElementwiseBias : FmhaBiasType::NoBias);

    static constexpr auto BiasEnum =
        (kResolvedBiasType == FmhaBiasType::ALiBi)
            ? BlockAttentionBiasEnum::ALIBI
            : ((kResolvedBiasType == FmhaBiasType::ElementwiseBias)
                   ? BlockAttentionBiasEnum::ELEMENTWISE_BIAS
                   : BlockAttentionBiasEnum::NO_BIAS);

    // Resolved QuantScaleEnum value. Defaults to NO_SCALE (MGX parity);
    // FP8 kernels pass PERTENSOR / BLOCKSCALE / KV_BLOCKSCALE.
    static constexpr auto QScaleEnum =
        (kQScaleType == FmhaQScaleType::PerTensor)
            ? BlockAttentionQuantScaleEnum::PERTENSOR
            : ((kQScaleType == FmhaQScaleType::BlockScale)
                   ? BlockAttentionQuantScaleEnum::BLOCKSCALE
                   : ((kQScaleType == FmhaQScaleType::KvBlockScale)
                          ? BlockAttentionQuantScaleEnum::KV_BLOCKSCALE
                          : BlockAttentionQuantScaleEnum::NO_SCALE));

    using FmhaTraits = TileFmhaTraits<kPadM,
                                      kPadN,
                                      kPadK,
                                      kPadO,
                                      kHasLogitsSoftCap,
                                      BiasEnum,
                                      /*kHasBiasGrad=*/false,
                                      kHasLSE,
                                      kHasDropout,
                                      QScaleEnum,
                                      kBlockPerCu,
                                      kSkipMinSeqlenQ,
                                      kHasSink>;

    // SimplifiedGenericAttentionMask<true> covers both top-left and
    // bottom-right causal; selection between them is runtime via
    // kargs.mask_type. Window generic masks also use the same
    // specialisation.
    using FmhaMask = std::conditional_t<kHasMask,
                                        SimplifiedGenericAttentionMask<true>,
                                        SimplifiedGenericAttentionMask<false>>;

    static constexpr bool kUseTrLoad =
        (kPipelineTag == FmhaPipelineTag::QR_ASYNC_TRLOAD) ||
        (kPipelineTag == FmhaPipelineTag::QR_ASYNC_TRLOAD_V3);

    static constexpr bool kIsV3Pipeline =
        (kPipelineTag == FmhaPipelineTag::V3) ||
        (kPipelineTag == FmhaPipelineTag::QR_ASYNC_TRLOAD_V3);

    using PipelineProblem =
        BlockFmhaPipelineProblem<DataType_,        // Q type
                                 DataType_,        // K type
                                 DataType_,        // V type
                                 float,            // Sacc type
                                 float,            // SMPLCompute type
                                 BiasDataType_,    // Bias type (float for FP8, DataType_ otherwise)
                                 unsigned short,   // RandVal type
                                 float,            // LSE type
                                 DataType_,        // P type
                                 float,            // Oacc type
                                 OutputDataType_,  // O type (DataType_ default, or fp16/bf16/fp32 for FP8 mixed-precision)
                                 FmhaShape,
                                 kIsGroupMode,
                                 ComposedAttention<
                                     kHasLogitsSoftCap ? LOGITS_SOFT_CAP : 0,
                                     CK_TILE_FMHA_FWD_FAST_EXP2>,
                                 FmhaMask,
                                 kUseTrLoad,
                                 FmhaTraits>;

    // Phase 7: V3 pipeline maps to BlockFmhaFwdV3Pipeline (+ the
    // qr_async_trload_v3 variant uses the trload intrinsics path).
    using Pipeline = std::conditional_t<
        kIsV3Pipeline,
        BlockFmhaFwdV3Pipeline<PipelineProblem>,
        std::conditional_t<kPipelineTag == FmhaPipelineTag::QR_ASYNC_TRLOAD,
                           BlockFmhaPipelineQRKSVSAsyncTrload<PipelineProblem>,
                           std::conditional_t<kPipelineTag == FmhaPipelineTag::QR_ASYNC,
                                              BlockFmhaPipelineQRKSVSAsync<PipelineProblem>,
                                              BlockFmhaPipelineQRKSVS<PipelineProblem>>>>;

    using Epilogue = Default2DEpilogue<Default2DEpilogueProblem<float, OutputDataType_, kPadM, kPadO>>;

    // Phase 7: V3 pipelines use the dedicated FmhaFwdV3Kernel template
    // with different tile-partitioner semantics.
    using Kernel = std::conditional_t<kIsV3Pipeline,
                                      FmhaFwdV3Kernel<Pipeline, Epilogue>,
                                      FmhaFwdKernel<Pipeline, Epilogue>>;

    // Innermost dimension is always contiguous (stride=1):
    //
    // Q: [batch, nhead_q, M, K]
    // K: [batch, nhead_k, N, K]
    // V (rowmajor): [batch, nhead_k, N, O]
    // V (colmajor): [batch, nhead_k, O, N]
    // O: [batch, nhead_q, M, O]
    // Bias: [batch, nhead_q, M, N]
    struct Descriptor
    {
        index_t batch, nhead, M, K;
        index_t q_stride_batch, q_stride_nhead, q_stride_m;

        index_t N;
        index_t k_stride_batch, k_stride_nhead, k_stride_n;

        index_t O;
        index_t v_stride_batch, v_stride_nhead, v_stride_n;

        index_t o_stride_batch, o_stride_nhead, o_stride_m;

        index_t bias_stride_batch, bias_stride_nhead, bias_stride_m;

        // GQA: number of K/V heads (<= nhead). nhead_q == nhead.
        index_t nhead_k = 0; // 0 means "same as nhead" for MGX backward compat

        // LSE [batch, nhead_q, M]
        index_t lse_stride_batch = 0;
        index_t lse_stride_nhead = 0;

        // Window mask bounds (ignored unless kResolvedMaskType==WindowGeneric).
        // For TopLeft/BottomRight causal, the kernel computes these
        // from seqlen_q/seqlen_k at runtime.
        index_t window_size_left  = -1;
        index_t window_size_right = 0;

        // Sink tokens
        index_t sink_size = 0;

        // Dropout probability (converted internally to rp_drop).
        float p_drop = 0.0f;
        // Plain-integer seed/offset pair (pointer variant not supported in
        // RTC path yet; that lands in Phase 5).
        uint64_t drop_seed   = 0;
        uint64_t drop_offset = 0;

        // Logits soft cap
        float logits_soft_cap = 0.0f;

        // Group mode variable-length data.
        // Pointers here are device pointers; the descriptor is captured by
        // the generated kernel as a const reference so passing null for
        // unused ones is safe.
        const int32_t* seqstart_q_ptr = nullptr;
        const int32_t* seqstart_k_ptr = nullptr;
        const int32_t* seqlen_k_ptr   = nullptr;
        index_t max_seqlen_q          = 0; // required when kIsGroupMode

        // ALiBi slopes (if any). [nhead_q] float array.
        const float* alibi_slope_ptr = nullptr;
        index_t stride_alibi_slope   = 0;

        // Sink token ptr (1 element per head, float).
        const float* sink_ptr = nullptr;

        // --- Phase 3: FP8 descale/scale ---
        // Only used when kQScaleType != NoScale.
        const void* q_descale_ptr = nullptr;
        const void* k_descale_ptr = nullptr;
        const void* v_descale_ptr = nullptr;
        index_t nhead_stride_q_descale = 0;
        index_t nhead_stride_k_descale = 0;
        index_t nhead_stride_v_descale = 0;
        index_t batch_stride_q_descale = 0;
        index_t batch_stride_k_descale = 0;
        index_t batch_stride_v_descale = 0;
        // For BLOCKSCALE:
        index_t block_scale_size_q  = 0;
        index_t block_scale_size_kv = 0;
        // For group-mode BLOCKSCALE:
        const int32_t* block_scale_seqstart_q_ptr = nullptr;
        const int32_t* block_scale_seqstart_k_ptr = nullptr;

        // Only reflects compile time arch availability, 
        // does not perform runtime descriptor validation.
        CK_TILE_HOST_DEVICE constexpr bool IsValid() const { return Kernel::kIsAvailable; }
    };

    // Each tensor is specified as (batch, nhead, dim0, dim1) and (stride0, stride1, stride2)
    // Innermost stride is always 1 and not passed.
    //
    // Backward-compatible constructor kept as the primary make_descriptor.
    // For features added in Phase 2 (LSE, GQA, group, dropout, window,
    // ALiBi, sink, soft-cap), construct the Descriptor directly and fill
    // the extra fields. The MGX test only uses this overload, so its
    // source compiles unchanged.
    template <typename QDims,
              typename QStrides,
              typename KDims,
              typename KStrides,
              typename VDims,
              typename VStrides,
              typename ODims,
              typename OStrides,
              typename BiasDims,
              typename BiasStrides>
    CK_TILE_HOST_DEVICE static constexpr auto make_descriptor(QDims q_dims,
                                                              QStrides q_strides,
                                                              KDims k_dims,
                                                              KStrides k_strides,
                                                              VDims v_dims,
                                                              VStrides v_strides,
                                                              ODims o_dims,
                                                              OStrides o_strides,
                                                              BiasDims bias_dims,
                                                              BiasStrides bias_strides)
    {
        (void)v_dims; (void)o_dims; (void)bias_dims;
        // GQA: if the K-tuple's nhead (`k_dims[1]`) matches the
        // Q-tuple's nhead, the descriptor stays in the MHA regime
        // and `nhead_k` is left at its sentinel value of 0 (the
        // wrapper treats 0 as "same as nhead_q"). When the two
        // differ we propagate the K-side nhead so the Run() path
        // computes `nhead_ratio_qk = nhead_q / nhead_k` correctly.
        Descriptor d{q_dims[number<0>{}],
                     q_dims[number<1>{}],
                     q_dims[number<2>{}],
                     q_dims[number<3>{}],
                     q_strides[number<0>{}],
                     q_strides[number<1>{}],
                     q_strides[number<2>{}],
                     //
                     k_dims[number<2>{}],
                     k_strides[number<0>{}],
                     k_strides[number<1>{}],
                     k_strides[number<2>{}],
                     //
                     v_dims[number<3>{}],
                     v_strides[number<0>{}],
                     v_strides[number<1>{}],
                     v_strides[number<2>{}],
                     //
                     o_strides[number<0>{}],
                     o_strides[number<1>{}],
                     o_strides[number<2>{}],
                     //
                     bias_strides[number<0>{}],
                     bias_strides[number<1>{}],
                     bias_strides[number<2>{}]};
        d.nhead_k = (k_dims[number<1>{}] == q_dims[number<1>{}]) ? 0 : k_dims[number<1>{}];
        return d;
    }

    CK_TILE_DEVICE static void Run(const Descriptor& desc,
                                   float scale_s,
                                   const DataType_* q_ptr,
                                   const DataType_* k_ptr,
                                   const DataType_* v_ptr,
                                   const DataType_* bias_ptr,
                                   OutputDataType_* o_ptr,
                                   // Phase 2 extras -- defaulted for MGX callers
                                   float* lse_ptr = nullptr,
                                   unsigned char* rand_val_ptr = nullptr)
    {
        using Kargs = typename Kernel::Kargs;
        Kargs kargs{};

        // --- Common fields (always present) ---
        kargs.q_ptr    = q_ptr;
        kargs.k_ptr    = k_ptr;
        kargs.v_ptr    = v_ptr;
        kargs.o_ptr    = o_ptr;

        // sink_ptr is part of FmhaFwdCommonKargs unconditionally.
        kargs.sink_ptr = kHasSink ? static_cast<const void*>(desc.sink_ptr) : nullptr;

        kargs.seqlen_q = desc.M;
        kargs.seqlen_k = desc.N;
        kargs.hdim_q   = desc.K;
        kargs.hdim_v   = desc.O;

        // GQA: nhead_k == 0 means "same as nhead" (MGX default).
        kargs.num_head_q     = desc.nhead;
        kargs.nhead_ratio_qk = (desc.nhead_k > 0) ? (desc.nhead / desc.nhead_k) : 1;

        kargs.scale_s = scale_s;

        kargs.stride_q = desc.q_stride_m;
        kargs.stride_k = desc.k_stride_n;
        kargs.stride_v = desc.v_stride_n;
        kargs.stride_o = desc.o_stride_m;

        kargs.nhead_stride_q = desc.q_stride_nhead;
        kargs.nhead_stride_k = desc.k_stride_nhead;
        kargs.nhead_stride_v = desc.v_stride_nhead;
        kargs.nhead_stride_o = desc.o_stride_nhead;

        // --- Conditional mix-in fields ---
        // Field availability matches the Kargs inheritance composition in
        // ck_tile::FmhaFwdKernel::FmhaFwd{Batch,Group}ModeKargs.

        // Bias plumbing. Kargs exposes `stride_bias` / `nhead_stride_bias`
        // / (batch only) `batch_stride_bias` when BiasEnum == ELEMENTWISE,
        // and `alibi_slope_ptr` / `alibi_slope_stride` when ALIBI.
        if constexpr(kResolvedBiasType == FmhaBiasType::ElementwiseBias)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = desc.bias_stride_m;
            kargs.nhead_stride_bias = desc.bias_stride_nhead;
            if constexpr(!kIsGroupMode)
                kargs.batch_stride_bias = desc.bias_stride_batch;
        }
        else if constexpr(kResolvedBiasType == FmhaBiasType::ALiBi)
        {
            kargs.alibi_slope_ptr    = desc.alibi_slope_ptr;
            kargs.alibi_slope_stride = desc.stride_alibi_slope;
        }

        // Mask plumbing.
        if constexpr(kResolvedMaskType == FmhaMaskType::TopLeft)
        {
            kargs.window_size_left  = -1;
            kargs.window_size_right = 0;
            kargs.sink_size         = desc.sink_size;
            kargs.mask_type         = GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT;
        }
        else if constexpr(kResolvedMaskType == FmhaMaskType::BottomRight)
        {
            kargs.window_size_left  = -1;
            kargs.window_size_right = 0;
            kargs.sink_size         = desc.sink_size;
            kargs.mask_type         = GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT;
        }
        else if constexpr(kResolvedMaskType == FmhaMaskType::WindowGeneric)
        {
            kargs.window_size_left  = desc.window_size_left;
            kargs.window_size_right = desc.window_size_right;
            kargs.sink_size         = desc.sink_size;
            kargs.mask_type         = GenericAttentionMaskEnum::MASK_GENERIC;
        }

        // Soft cap: FmhaFwdLogitsSoftCapKargs requires init to also set the reciprocal.
        if constexpr(kHasLogitsSoftCap)
        {
            kargs.init_logits_soft_cap(desc.logits_soft_cap);
        }

        // --- Phase 3: FP8 quant-scale plumbing ---
        // Kargs composition inherits FmhaFwdCommonQScaleKargs when PERTENSOR,
        // FmhaFwdBatch/GroupBlockScaleKargs when BLOCKSCALE. The set of
        // accessible fields depends on kIsGroupMode + QScaleEnum.
        if constexpr(kQScaleType == FmhaQScaleType::PerTensor ||
                     kQScaleType == FmhaQScaleType::BlockScale ||
                     kQScaleType == FmhaQScaleType::KvBlockScale)
        {
            kargs.q_descale_ptr = desc.q_descale_ptr;
            kargs.k_descale_ptr = desc.k_descale_ptr;
            kargs.v_descale_ptr = desc.v_descale_ptr;
        }
        if constexpr(kQScaleType == FmhaQScaleType::BlockScale ||
                     kQScaleType == FmhaQScaleType::KvBlockScale)
        {
            kargs.nhead_stride_q_descale = desc.nhead_stride_q_descale;
            kargs.nhead_stride_k_descale = desc.nhead_stride_k_descale;
            kargs.nhead_stride_v_descale = desc.nhead_stride_v_descale;
            kargs.block_scale_size_q     = desc.block_scale_size_q;
            kargs.block_scale_size_kv    = desc.block_scale_size_kv;
            if constexpr(kIsGroupMode)
            {
                kargs.block_scale_seqstart_q_ptr = desc.block_scale_seqstart_q_ptr;
                kargs.block_scale_seqstart_k_ptr = desc.block_scale_seqstart_k_ptr;
            }
            else
            {
                kargs.batch_stride_q_descale = desc.batch_stride_q_descale;
                kargs.batch_stride_k_descale = desc.batch_stride_k_descale;
                kargs.batch_stride_v_descale = desc.batch_stride_v_descale;
            }
        }

        // LSE output.
        if constexpr(kHasLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = desc.lse_stride_nhead;
            if constexpr(!kIsGroupMode)
                kargs.batch_stride_lse = desc.lse_stride_batch;
        }

        // Dropout. Uses init_dropout() to set rp_undrop/p_undrop_in_uint8_t.
        if constexpr(kHasDropout)
        {
            kargs.rand_val_ptr     = rand_val_ptr;
            kargs.is_store_randval = (rand_val_ptr != nullptr);
            kargs.init_dropout(desc.p_drop,
                               static_cast<uint64_t>(desc.drop_seed),
                               static_cast<uint64_t>(desc.drop_offset));
        }

        // Batch-mode-only fields (batch strides + optional cu_seqlens).
        // Group-mode fields (seqstart_*) are on FmhaFwdGroupModeKargs.
        if constexpr(kIsGroupMode)
        {
            kargs.seqstart_q_ptr = desc.seqstart_q_ptr;
            kargs.seqstart_k_ptr = desc.seqstart_k_ptr;
            kargs.seqlen_q_ptr   = nullptr;
            kargs.seqlen_k_ptr   = desc.seqlen_k_ptr;
            kargs.cu_seqlen_q_ptr = nullptr;
            kargs.cu_seqlen_k_ptr = nullptr;
        }
        else
        {
            kargs.batch_stride_q = desc.q_stride_batch;
            kargs.batch_stride_k = desc.k_stride_batch;
            kargs.batch_stride_v = desc.v_stride_batch;
            kargs.batch_stride_o = desc.o_stride_batch;
            kargs.cu_seqlen_q_ptr = nullptr;
            kargs.cu_seqlen_k_ptr = nullptr;
        }

        Kernel{}(kargs);
    }
};

} // namespace ck_tile
