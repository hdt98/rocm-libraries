// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Phase 5 SplitKV "split" wrapper. Pairs with
// device_fmha_fwd_splitkv_combine's combine kernel for the 2-stage
// split-KV forward plan.
//
// TODO(phase5-followup): wire full Kargs for split accumulators
// (lse_acc_ptr, o_acc_ptr, split_stride_*_acc). The scaffolding here
// is the template shape needed for Phase 9's plugin integration.

#include <cmath>
#include <cstdint>
#include <cassert>
#include <type_traits>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
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
          bool kHasLSE,
          bool kPadM,
          bool kPadN,
          bool kPadK,
          bool kPadO,
          FmhaPipelineTag kPipelineTag,
          index_t kMaxSplitsLog2 = 3>
struct FmhaFwdSplitKVWrapper
{
    using BlockTile       = sequence<kBM0, kBN0, kBK0, kBN1, kBK1, kBK0Max>;
    using Gemm0BlockWarps = sequence<kRM0, kRN0, kRK0>;
    using Gemm0WarpTile   = sequence<kWM0, kWN0, kWK0>;
    using Gemm1BlockWarps = sequence<kRM1, kRN1, kRK1>;
    using Gemm1WarpTile   = sequence<kWM1, kWN1, kWK1>;
    using FmhaShape       = TileFmhaShape<BlockTile,
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
                                      kHasLSE,
                                      /*kHasDropout=*/false,
                                      BlockAttentionQuantScaleEnum::NO_SCALE,
                                      -1,
                                      /*kSkipMinSeqlenQ=*/false,
                                      /*kHasSink=*/false>;

    using FmhaMask = std::conditional_t<kIsCausal,
                                        SimplifiedGenericAttentionMask<true>,
                                        SimplifiedGenericAttentionMask<false>>;

    using PipelineProblem =
        BlockFmhaPipelineProblem<DataType_, DataType_, DataType_, float, float,
                                 DataType_, unsigned short, float, DataType_, float,
                                 DataType_, FmhaShape,
                                 /*kIsGroupMode=*/false,
                                 ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>,
                                 FmhaMask,
                                 /*kUseTrLoad=*/false,
                                 FmhaTraits>;

    using Pipeline = BlockFmhaPipelineQRKSVS<PipelineProblem>;
    using Epilogue = Default2DEpilogue<Default2DEpilogueProblem<float, DataType_, kPadM, kPadO>>;
    using Kernel   = FmhaFwdSplitKVKernel<Pipeline, Epilogue>;

    struct Descriptor
    {
        index_t batch, nhead_q, nhead_k, M, K, N, O;
        index_t q_stride_batch, q_stride_nhead, q_stride_m;
        index_t k_stride_batch, k_stride_nhead, k_stride_n;
        index_t v_stride_batch, v_stride_nhead, v_stride_n;
        index_t o_stride_batch, o_stride_nhead, o_stride_m;
        index_t bias_stride_batch = 0, bias_stride_nhead = 0, bias_stride_m = 0;

        index_t num_splits = 1;
        index_t page_block_size = 0;

        CK_TILE_HOST_DEVICE constexpr bool IsValid() const { return Kernel::kIsAvailable; }
    };

    CK_TILE_DEVICE static void Run(const Descriptor& desc,
                                   float scale_s,
                                   const DataType_* q_ptr,
                                   const DataType_* k_ptr,
                                   const DataType_* v_ptr,
                                   const DataType_* bias_ptr,
                                   DataType_* o_ptr,
                                   float* lse_acc_ptr = nullptr,
                                   float* o_acc_ptr   = nullptr)
    {
        (void)desc; (void)scale_s; (void)q_ptr; (void)k_ptr; (void)v_ptr;
        (void)bias_ptr; (void)o_ptr; (void)lse_acc_ptr; (void)o_acc_ptr;
        // TODO(phase5-followup): populate FmhaFwdSplitKVKernel::Kargs.
    }
};

} // namespace ck_tile
