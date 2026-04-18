// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Phase 5 pagedkv wrapper. Embedded into the ck_codegen_headers set
// at build time; instantiated at hipRTC compile time per Problem+Solution.
//
// TODO(phase5-followup): this wrapper currently presents the minimal
// public surface (batch, nhead, shapes, page_block_size) needed by
// Phase 9 plugin integration. Full Kargs composition for block_table /
// cache_batch_idx / seqlen_k_ptr etc. lands when a correctness test
// uses the wrapper end-to-end.

#include <cmath>
#include <cstdint>
#include <cassert>
#include <type_traits>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
// Reuse the FmhaPipelineTag / FmhaMaskType / FmhaBiasType enums from
// the Phase 2 forward wrapper.
#include "ck/host/device_fmha_fwd/fmha_fwd_wrapper.hpp"

namespace ck_tile {

template <typename DataType_,
          index_t kBM0,
          index_t kBN0,
          index_t kBK0,
          index_t kBN1,
          index_t kBK1,
          index_t kBK0Max,
          index_t kRM0,
          index_t kRN0,
          index_t kRK0,
          index_t kRM1,
          index_t kRN1,
          index_t kRK1,
          index_t kWM0,
          index_t kWN0,
          index_t kWK0,
          index_t kWM1,
          index_t kWN1,
          index_t kWK1,
          bool kIsCausal,
          bool kIsVRowMajor,
          bool kHasBias,
          bool kPadM,
          bool kPadN,
          bool kPadK,
          bool kPadO,
          FmhaPipelineTag kPipelineTag,
          index_t kPageBlockSize = 64>
struct FmhaFwdPagedKVWrapper
{
    // Pipeline problem and shape definitions mirror the forward wrapper
    // but bind to FmhaFwdPagedKVKernel.
    using BlockTile       = sequence<kBM0, kBN0, kBK0, kBN1, kBK1, kBK0Max>;
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

    static constexpr auto BiasEnum =
        kHasBias ? BlockAttentionBiasEnum::ELEMENTWISE_BIAS : BlockAttentionBiasEnum::NO_BIAS;

    using FmhaTraits = TileFmhaTraits<kPadM,
                                      kPadN,
                                      kPadK,
                                      kPadO,
                                      /*kHasLogitsSoftCap=*/false,
                                      BiasEnum,
                                      /*kHasBiasGrad=*/false,
                                      /*kStoreLSE=*/false,
                                      /*kHasDropout=*/false,
                                      BlockAttentionQuantScaleEnum::NO_SCALE,
                                      -1,
                                      /*kSkipMinSeqlenQ=*/false,
                                      /*kHasSink=*/false>;

    using FmhaMask = std::conditional_t<kIsCausal,
                                        SimplifiedGenericAttentionMask<true>,
                                        SimplifiedGenericAttentionMask<false>>;

    static constexpr bool kUseTrLoad = (kPipelineTag == FmhaPipelineTag::QR_ASYNC_TRLOAD);

    using PipelineProblem =
        BlockFmhaPipelineProblem<DataType_,
                                 DataType_,
                                 DataType_,
                                 float,
                                 float,
                                 DataType_,
                                 unsigned short,
                                 float,
                                 DataType_,
                                 float,
                                 DataType_,
                                 FmhaShape,
                                 /*kIsGroupMode=*/false,
                                 ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>,
                                 FmhaMask,
                                 kUseTrLoad,
                                 FmhaTraits>;

    using Pipeline = BlockFmhaPipelineQRKSVS<PipelineProblem>;
    using Epilogue = Default2DEpilogue<Default2DEpilogueProblem<float, DataType_, kPadM, kPadO>>;
    using Kernel   = FmhaFwdPagedKVKernel<Pipeline, Epilogue>;

    struct Descriptor
    {
        index_t batch, nhead_q, nhead_k, M, K, N, O;
        index_t q_stride_batch, q_stride_nhead, q_stride_m;
        index_t k_stride_batch, k_stride_nhead, k_stride_n;
        index_t v_stride_batch, v_stride_nhead, v_stride_n;
        index_t o_stride_batch, o_stride_nhead, o_stride_m;
        index_t bias_stride_batch = 0, bias_stride_nhead = 0, bias_stride_m = 0;

        // PagedKV-specific
        void* block_table_ptr                 = nullptr;
        index_t batch_stride_block_table      = 0;
        const void* cache_batch_idx           = nullptr;
        bool is_gappy                         = false;
        const void* seqstart_q_ptr            = nullptr;
        const void* seqlen_k_ptr              = nullptr;
        const void* sink_ptr                  = nullptr;

        CK_TILE_HOST_DEVICE constexpr bool IsValid() const { return Kernel::kIsAvailable; }
    };

    // Launch entry point. TODO(phase5-followup): fill full Kargs
    // composition matching FmhaFwdPagedKVKernel's specific mix-ins.
    CK_TILE_DEVICE static void Run(const Descriptor& desc,
                                   float scale_s,
                                   const DataType_* q_ptr,
                                   const DataType_* k_ptr,
                                   const DataType_* v_ptr,
                                   const DataType_* bias_ptr,
                                   DataType_* o_ptr)
    {
        (void)desc; (void)scale_s; (void)q_ptr; (void)k_ptr; (void)v_ptr; (void)bias_ptr; (void)o_ptr;
        // Scaffolding only; Phase 9 follow-up binds the full Kargs.
    }
};

} // namespace ck_tile
