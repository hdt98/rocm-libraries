// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_v3_pipeline_default_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_detail.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// ---------------------------------------------------------------------------
// BatchPrefillCoreLoopScheduler: FP8-tuned scheduler for batch prefill V3.
//
// Forked from the feature branch CoreLoopScheduler with higher VALU budgets
// that account for v_pk_mul_f32 asm volatile being invisible to the compiler
// scheduler.  bf16/fp16 inherit the default base unchanged.
//
// Design: the scheduler is intentionally NOT specialized for QScaleEnum
// (pertensor vs KV_BLOCKSCALE) or kHasLogitsSoftCap. Rationale:
//
//   - KV_BLOCKSCALE: GEMM0 K iter 1 has ~28 VALU (3.5/MFMA) vs PERTENSOR's
//     ~60 VALU (7.5/MFMA). The VALU:6 budget over-requests for KV_BLOCKSCALE,
//     leaving MFMAs 10-11 empty. But benchmarking showed this is neutral
//     (0.99-1.00x): the compiler handles over-budget gracefully by skipping
//     empty slots without stalling. GEMM1 KV_BLOCKSCALE has +17 VALU for
//     v_descale ratio, also handled well by compiler overflow into last MFMA.
//
//   - LogitsSoftCap: fmha_logits_trans adds ~160 ops (32x softsign/tanh) to
//     GEMM0 K iter 1, all piling into MFMA 16 (62T + 176V in one slot).
//     Tried TRANS:8+VALU:24 per MFMA to spread them: 10-12% REGRESSION.
//     v_rcp_f32 has ~30-cycle latency with data dependency chains across
//     elements; forced interleaving breaks the compiler's batched v_rcp
//     scheduling which amortizes latency across all 32 elements.
//
// The compiler's default handling (batch at end, overflow into last MFMA)
// outperforms forced spreading for both cases.
// ---------------------------------------------------------------------------
template <typename PipelineProblem>
struct BatchPrefillCoreLoopSchedulerBase
{
    using Params = CoreLoopSchedulingParams<PipelineProblem>;

    CK_TILE_DEVICE static constexpr void schedule_gemm0_compute()
    {
        static_for<0, Params::kMfmaPerWarpGemm0, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::TRANS, 2, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 2, 0);
        });
    }

    CK_TILE_DEVICE static constexpr void schedule_gemm1_compute()
    {
        static_for<0, Params::kMfmaPerWarpGemm1, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 5, 0);
        });
    }

    CK_TILE_DEVICE static constexpr void schedule_load_phase()
    {
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 2, 0);
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::SALU, 1, 0);
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VMEM_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::SALU, 1, 0);
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VMEM_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::SALU, 2, 0);
    }

    template <ck_tile::index_t WaveGroup, ck_tile::index_t Phase>
    CK_TILE_DEVICE static constexpr void schedule(ck_tile::number<WaveGroup>,
                                                  ck_tile::number<Phase>)
    {
        constexpr ck_tile::index_t effective = (WaveGroup == 0) ? Phase : (Phase + 3) % 4;
        if constexpr(effective == 0)
            schedule_gemm0_compute();
        else if constexpr(effective == 2)
            schedule_gemm1_compute();
        else
            schedule_load_phase();
    }
};

template <typename PipelineProblem, typename QDataType, typename KDataType, typename VDataType>
struct BatchPrefillCoreLoopSchedulerImpl;

template <typename PipelineProblem>
struct BatchPrefillCoreLoopSchedulerImpl<PipelineProblem,
                                         ck_tile::bf16_t,
                                         ck_tile::bf16_t,
                                         ck_tile::bf16_t>
    : BatchPrefillCoreLoopSchedulerBase<PipelineProblem>
{
};

template <typename PipelineProblem>
struct BatchPrefillCoreLoopSchedulerImpl<PipelineProblem,
                                         ck_tile::half_t,
                                         ck_tile::half_t,
                                         ck_tile::half_t>
    : BatchPrefillCoreLoopSchedulerBase<PipelineProblem>
{
};

template <typename PipelineProblem>
struct BatchPrefillCoreLoopSchedulerImpl<PipelineProblem,
                                         ck_tile::fp8_t,
                                         ck_tile::fp8_t,
                                         ck_tile::fp8_t>
    : BatchPrefillCoreLoopSchedulerBase<PipelineProblem>
{
    using Base   = BatchPrefillCoreLoopSchedulerBase<PipelineProblem>;
    using Params = typename Base::Params;
    CK_TILE_DEVICE static constexpr void schedule_gemm0_compute()
    {
        // K iter 0: 32 TRANS (v_exp_f32) + 29 VALU (v_add reduction + v_sub + permlane)
        static_for<0, Params::kMfmaPerWarpGemm0 / 2, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::TRANS, 4, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 4, 0);
        });
        // K iter 1: ~89 VALU (v_mul scale + v_cvt_pk_fp8 + o_acc rescale)
        static_for<0, Params::kMfmaPerWarpGemm0 / 2, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 6, 0);
        });
    }

    CK_TILE_DEVICE static constexpr void schedule_gemm1_compute()
    {
        // V3 batch prefill uses kernel_attr_for<gfx950_t, kernel_attr<true>> to disable
        // packed FP32 ops via target attribute, so pk_mul_f32 is always present (asm volatile).
        // This VALU:4 preamble accounts for the pk_mul instructions invisible to the scheduler.
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 4, 0);
        // First half: v_perm + v_max3 + permlane chain + v_fma (~57 VALU)
        static_for<0, Params::kMfmaPerWarpGemm1 / 2, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 6, 0);
        });
        // Second half: v_fma chain + v_mul O rescale (~33 VALU)
        // pk_mul (16 ops in asm volatile) invisible to scheduler
        static_for<0, Params::kMfmaPerWarpGemm1 / 2, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 6, 0);
        });
    }

    template <ck_tile::index_t WaveGroup, ck_tile::index_t Phase>
    CK_TILE_DEVICE static constexpr void schedule(ck_tile::number<WaveGroup>,
                                                  ck_tile::number<Phase>)
    {
        constexpr ck_tile::index_t effective = (WaveGroup == 0) ? Phase : (Phase + 3) % 4;

        if constexpr(effective == 0)
            schedule_gemm0_compute();
        else if constexpr(effective == 2)
            schedule_gemm1_compute();
        else
            Base::schedule_load_phase();
    }
};

template <typename PipelineProblem>
struct BatchPrefillCoreLoopScheduler
    : BatchPrefillCoreLoopSchedulerImpl<PipelineProblem,
                                        typename PipelineProblem::QDataType,
                                        typename PipelineProblem::KDataType,
                                        typename PipelineProblem::VDataType>
{
};

/// V3 pipeline adapted for batch prefill with scatter-gather KV loads (paged KV cache).
///
/// This pipeline inherits the V3 4-phase double warp group architecture
/// (CoreLoopScheduler, double-buffered LDS, phase barriers) and replaces
/// contiguous K/V DRAM loads with scatter-gather loads using page table lookups.
///
/// Key differences from BlockFmhaFwdV3Pipeline:
///   - K/V DRAM windows are tile_scatter_gather instead of tile_window
///   - Per-iteration page offset recomputation (load_physical_pages + kv_offset_array_transform)
///   - Additional operator() parameters: page_idx, stride_k/v, page_stride_k/v
///   - Problem type requires kPageBlockSize, kVectorSize, kKVMemoryLayout
/// V3 batch prefill pipeline for gfx950 (MI350). Uses permlane32_swap, 8-warp
/// tile (256x64), paged KV cache with scatter-gather, and double-buffered LDS.
/// On non-gfx950 targets, operator() is a no-op returning -1.
template <typename Problem_, typename Policy_ = BlockFmhaBatchPrefillV3PipelineDefaultPolicy>
struct BlockFmhaBatchPrefillV3Pipeline
{
    using Problem             = ck_tile::remove_cvref_t<Problem_>;
    using Policy              = ck_tile::remove_cvref_t<Policy_>;
    using QDataType           = ck_tile::remove_cvref_t<typename Problem::QDataType>;
    using KDataType           = ck_tile::remove_cvref_t<typename Problem::KDataType>;
    using VDataType           = ck_tile::remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType        = ck_tile::remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType = ck_tile::remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using LSEDataType         = ck_tile::remove_cvref_t<typename Problem::LSEDataType>;
    using PDataType           = ck_tile::remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType        = ck_tile::remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType           = ck_tile::remove_cvref_t<typename Problem::ODataType>;
    using AttentionVariant    = ck_tile::remove_cvref_t<typename Problem::AttentionVariant>;
    using FmhaMask            = ck_tile::remove_cvref_t<typename Problem::FmhaMask>;
    static_assert(is_generic_attention_mask_v<FmhaMask>);

    static_assert(std::is_same_v<SaccDataType, SMPLComputeDataType>,
                  "we will the same dist tensor 'sp_compute' for both gemm0 & softmax");

    using BlockFmhaShape = ck_tile::remove_cvref_t<typename Problem::BlockFmhaShape>;

    using VLayout = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static_assert(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>);

    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;

    static constexpr ck_tile::index_t kM0           = BlockFmhaShape::kM0;
    static constexpr ck_tile::index_t kN0           = BlockFmhaShape::kN0;
    static constexpr ck_tile::index_t kK0           = BlockFmhaShape::kK0;
    static constexpr ck_tile::index_t kN1           = BlockFmhaShape::kN1;
    static constexpr ck_tile::index_t kK1           = BlockFmhaShape::kK1;
    static constexpr ck_tile::index_t kQKHeaddim    = BlockFmhaShape::kQKHeaddim;
    static constexpr ck_tile::index_t kSubQKHeaddim = BlockFmhaShape::kSubQKHeaddim;

    static_assert(kQKHeaddim == 128 && kSubQKHeaddim == 128, "only supports hdim=hdim_v=128");

    // Paged KV cache parameters
    static constexpr ck_tile::index_t kPageBlockSize = Problem::kPageBlockSize;
    static constexpr ck_tile::index_t kVectorSize    = Problem::kVectorSize;
    static constexpr auto kKVMemoryLayout            = Problem::kKVMemoryLayout;
    static_assert(kKVMemoryLayout == BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT,
                  "V3 batch prefill only supports LINEAR_LAYOUT (VECTORIZED requires sub-dword "
                  "async loads which violate buffer addressing constraints)");

    static constexpr bool kIsGroupMode      = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ       = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = Problem::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap = Problem::kHasLogitsSoftCap;
    static constexpr auto BiasEnum          = Problem::BiasEnum;
    static constexpr bool kStoreLSE         = Problem::kStoreLSE;
    static constexpr bool kHasDropout       = Problem::kHasDropout;
    static constexpr auto QScaleEnum        = Problem::QScaleEnum;
    static constexpr bool kSkipMinSeqlenQ   = Problem::kSkipMinSeqlenQ;
    static_assert((BiasEnum == BlockAttentionBiasEnum::NO_BIAS && !kHasDropout && !kSkipMinSeqlenQ),
                  "enable unsupported features");
    static_assert(QScaleEnum != BlockAttentionQuantScaleEnum::KV_BLOCKSCALE ||
                      kPageBlockSize >= kN0,
                  "KV_BLOCKSCALE requires kPageBlockSize >= kN0");
    static_assert(!kPadHeadDimQ && !kPadHeadDimV,
                  "V3 batch prefill requires hdim=128 which is always aligned, no padding needed");

    // For KV_BLOCKSCALE: shift value for exp2(x + shift) to scale P to [0, 2^shift]
    static constexpr float OCP_FP8_SHIFT  = 8.0f;
    static constexpr float FNUZ_FP8_SHIFT = 7.0f;

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    //
    // Unlike the contiguous V3 fwd pipeline which uses alignment=1 for padded dims,
    // scatter-gather relies on the tensor descriptor's GuaranteedLastDimensionVectorLength
    // to determine ScalarPerVector for buffer loads. Setting alignment=1 would result in
    // per-element (2-byte bf16) loads, violating the 4-byte dword minimum for async buffer loads.
    // Since hdim is always 128 (enforced by static_assert above), the full alignment is safe.
    static constexpr ck_tile::index_t kAlignmentQ = Policy::template GetAlignmentQ<Problem>();
    static constexpr ck_tile::index_t kAlignmentK = Policy::template GetAlignmentK<Problem>();
    static constexpr ck_tile::index_t kAlignmentV = Policy::template GetAlignmentV<Problem>();

    static constexpr ck_tile::index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();

    static constexpr ck_tile::index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            return 2;
        }
    }();

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        // V3 kernel allocates smem_k[2] and smem_v[2] separately (double buffered),
        // plus epilogue smem. Return max(K_single, V_single) * 2 for the pipeline portion.
        return 2 * Policy::template GetSmemSizeK<Problem>() +
               2 * Policy::template GetSmemSizeV<Problem>();
    }

    template <typename DataType, typename Descriptor>
    CK_TILE_DEVICE static constexpr auto make_lds_tile_window(DataType* __restrict__ base,
                                                              const Descriptor& desc)
    {
        using namespace ck_tile;

        auto tensor_view = make_tensor_view<address_space_enum::lds>(base, desc);
        return make_tile_window(tensor_view, desc.get_lengths(), {0, 0});
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename LSEElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_DEVICE auto
    operator()(multi_index<2> partition_index,
               const QDramBlockWindowTmp& __restrict__ q_dram_block_window_tmp, // M0*K0 tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& __restrict__ k_dram_block_window_tmp, // N0*K0 tile
               [[maybe_unused]] const KElementFunction& k_element_func,
               const VDramBlockWindowTmp& __restrict__ v_dram_block_window_tmp, // N1*K1 tile
               [[maybe_unused]] const VElementFunction& v_element_func,
               LSEDramBlockWindowTmp& __restrict__ lse_dram_window_tmp, // M0*1 tile
               const LSEElementFunction& lse_element_func,
               [[maybe_unused]] const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               FmhaMask mask,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               KDataType* __restrict__ smem_k0,
               KDataType* __restrict__ smem_k1,
               VDataType* __restrict__ smem_v0,
               VDataType* __restrict__ smem_v1,
               // Paged KV cache parameters
               const index_t* page_idx,
               index_t stride_k,
               index_t stride_v,
               index_t page_stride_k,
               index_t page_stride_v,
               index_t max_page_table_idx = 0x7FFFFFFF,
               // KV_BLOCKSCALE parameters
               const float* k_descale_ptr             = nullptr,
               const float* v_descale_ptr             = nullptr,
               index_t nblock_stride_kv_block_descale = 0,
               index_t nhead_stride_kv_block_descale  = 0) const
    {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx950__)
        // V3 pipeline is gfx950-only; return empty output on other targets.
        ignore                = partition_index;
        ignore                = q_dram_block_window_tmp;
        ignore                = q_element_func;
        ignore                = k_dram_block_window_tmp;
        ignore                = v_dram_block_window_tmp;
        ignore                = lse_dram_window_tmp;
        ignore                = lse_element_func;
        ignore                = p_compute_element_func;
        ignore                = o_acc_element_func;
        ignore                = mask;
        ignore                = scale_s;
        ignore                = variant;
        ignore                = variant_params;
        ignore                = block_indices;
        ignore                = smem_k0;
        ignore                = smem_k1;
        ignore                = smem_v0;
        ignore                = smem_v1;
        ignore                = page_idx;
        ignore                = stride_k;
        ignore                = stride_v;
        ignore                = page_stride_k;
        ignore                = page_stride_v;
        ignore                = max_page_table_idx;
        ignore                = k_descale_ptr;
        ignore                = v_descale_ptr;
        ignore                = nblock_stride_kv_block_descale;
        ignore                = nhead_stride_kv_block_descale;
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();
        decltype(gemm_1.MakeCBlockTile()) o_acc;
        auto lse_acc = make_static_distributed_tensor<LSEDataType>(
            Policy::template MakeLSEDDramTileDistribution<Problem>());
        return ck_tile::make_tuple(o_acc, lse_acc);
#else
        using namespace ck_tile;

        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        const index_t warp_id       = partition_index[0];
        const index_t warp_group_id = warp_id / 4;
        const index_t lane_id       = partition_index[1];

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();

        auto q_dram_window = make_tile_window(q_dram_block_window_tmp,
                                              Policy::template MakeQRegTileDistribution<Problem>(),
                                              partition_index);

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        auto k_lds_window_store = generate_tuple(
            [&](auto write_idx) {
                auto k_buf = (write_idx == 0 ? smem_k0 : smem_k1);
                return make_lds_tile_window(
                    k_buf, Policy::template MakeKLdsStoreBlockDescriptor<Problem>());
            },
            number<2>{});

        auto v_lds_window_store = generate_tuple(
            [&](auto write_idx) {
                auto v_buf = (write_idx == 0 ? smem_v0 : smem_v1);
                return make_lds_tile_window(
                    v_buf, Policy::template MakeVLdsStoreBlockDescriptor<Problem>());
            },
            number<2>{});

        constexpr auto all_zeros_partition_index = make_multi_index(0, 0);
        statically_indexed_array<decltype(make_tile_window(
                                     make_lds_tile_window<KDataType>(
                                         nullptr,
                                         Policy::template MakeKLdsLoadBlockDescriptor<Problem>()),
                                     Policy::template MakeKRegTileDistribution<Problem>(),
                                     all_zeros_partition_index)),
                                 2>
            k_lds_window_load;

        statically_indexed_array<decltype(make_tile_window(
                                     make_lds_tile_window<VDataType>(
                                         nullptr,
                                         Policy::template MakeVLdsLoadBlockDescriptor<Problem>()),
                                     Policy::template MakeVRegTileDistribution<Problem>(),
                                     all_zeros_partition_index)),
                                 2>
            v_lds_window_load;

        decltype(make_static_distributed_tensor<QDataType>(
            Policy::template MakeQRegTileDistribution<Problem>())) q_tile;

        union kv_tile_type
        {
            CK_TILE_DEVICE kv_tile_type() {}

            decltype(load_tile(k_lds_window_load(number<0>{}))) k_tile;

            decltype(load_tile_transpose(v_lds_window_load(number<0>{}))) v_tile;
        } kv_tile;

        union sp_compute_type
        {
            CK_TILE_DEVICE sp_compute_type() {}

            decltype(gemm_0.MakeCBlockTile()) sp_compute;
            decltype(make_static_distributed_tensor<PDataType>(
                Policy::template MakePRegTileDistribution<Problem>())) p;
        };
        statically_indexed_array<sp_compute_type, 2> sp;

        decltype(gemm_1.MakeCBlockTile()) o_acc;
        constexpr index_t fmha_alu_D_reg_cnt =
            6; // Threshold for determining how many fmha_alu_D_upd() unpacked
               // instructions to relocate to fmha_alu1().
        static_assert(fmha_alu_D_reg_cnt % 2 == 0 &&
                      fmha_alu_D_reg_cnt <= o_acc.thread_buf_.size());

        decltype(block_tile_reduce<SMPLComputeDataType>(
            sp(number<0>{}).sp_compute, sequence<1>{}, f_max, SMPLComputeDataType{0})) m;
        decltype(m) l;

        // initialize k_lds_window and v_lds_window with all_zeros_partition_index
        // The actual per-thread offset is computed below and passed to load_tile_with_offset

        static_for<0, 2, 1>{}([&](auto idx) {
            k_lds_window_load(idx) =
                make_tile_window(make_lds_tile_window(
                                     [&] {
                                         if constexpr(idx == 0)
                                             return smem_k0;
                                         else
                                             return smem_k1;
                                     }(),
                                     Policy::template MakeKLdsLoadBlockDescriptor<Problem>()),
                                 Policy::template MakeKRegTileDistribution<Problem>(),
                                 all_zeros_partition_index);
        });

        static_for<0, 2, 1>{}([&](auto idx) {
            v_lds_window_load(idx) =
                make_tile_window(make_lds_tile_window(
                                     [&] {
                                         if constexpr(idx == 0)
                                             return smem_v0;
                                         else
                                             return smem_v1;
                                     }(),
                                     Policy::template MakeVLdsLoadBlockDescriptor<Problem>()),
                                 Policy::template MakeVRegTileDistribution<Problem>(),
                                 all_zeros_partition_index);
        });

        // Compute per-thread LDS load offset using hardcoded formulas (empirically derived)
        const index_t k_lds_load_offset = [&] {
            if constexpr(std::is_same_v<KDataType, fp8_t>)
            {
                constexpr auto k_tile_dstr = Policy::template MakeKRegTileDistribution<Problem>();
                constexpr auto k_lds_desc = Policy::template MakeKLdsLoadBlockDescriptor<Problem>();
                constexpr index_t NDimY   = decltype(k_tile_dstr)::NDimY;
                auto top_index            = container_concat(partition_index, multi_index<NDimY>{});
                const auto adaptor_coord  = make_tensor_adaptor_coordinate(
                    k_tile_dstr.get_ps_ys_to_xs_adaptor(), top_index);
                const auto bottom_idx = adaptor_coord.get_bottom_index();
                const auto lds_coord  = make_tensor_coordinate(k_lds_desc, bottom_idx);
                return lds_coord.get_offset();
            }
            else
            {
                index_t start_row   = lane_id % 32;
                index_t start_col   = lane_id / 32 * 8;
                index_t warp_offset = (start_row / 8) * (4 * 4) / 2;
                return (start_row * 64) + start_col + warp_offset;
            }
        }();

        const index_t v_lds_load_offset = [&] {
            if constexpr(std::is_same_v<VDataType, fp8_t>)
            {
                constexpr auto v_tile_dstr = Policy::template MakeVRegTileDistribution<Problem>();
                constexpr auto v_lds_desc = Policy::template MakeVLdsLoadBlockDescriptor<Problem>();
                constexpr index_t NDimY   = decltype(v_tile_dstr)::NDimY;
                auto top_index            = container_concat(partition_index, multi_index<NDimY>{});
                const auto adaptor_coord  = make_tensor_adaptor_coordinate(
                    v_tile_dstr.get_ps_ys_to_xs_adaptor(), top_index);
                const auto bottom_idx = adaptor_coord.get_bottom_index();
                const auto lds_coord  = make_tensor_coordinate(v_lds_desc, bottom_idx);
                return lds_coord.get_offset();
            }
            else
            {
                index_t group_idx     = lane_id / 16;
                index_t local_lane_id = lane_id % 16;
                index_t start_row     = (group_idx / 2) * 4 + local_lane_id / 4;
                index_t start_col     = (group_idx % 2) * 16 + (local_lane_id % 4) * 4;
                return (start_row * 64) + start_col;
            }
        }();

        {
            auto origin_q      = load_tile(q_dram_window);
            auto transformed_q = tile_elementwise_in(q_element_func, origin_q);

            q_tile = transformed_q;
        }

        clear_tile(o_acc);
        set_tile(m, bit_cast<float>(0xff7fffff)); // a bit larger than -infinity
        clear_tile(l);

        const auto q_origin = q_dram_window.get_window_origin();
        const auto [seqlen_k_start, seqlen_k_end] =
            mask.GetTileRangeAlongX(q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});

        const auto num_total_loop = integer_divide_ceil(seqlen_k_end - seqlen_k_start, kN0);
        index_t kv_token_start    = seqlen_k_start;

        // =====================================================================
        // Scatter-gather K DRAM window setup (replaces contiguous make_tile_window)
        // =====================================================================
        auto k_dist               = Policy::template MakeKDramTileDistribution<Problem>();
        auto k_coord              = k_dist.calculate_index();
        using KDstrEncode         = typename decltype(k_dist)::DstrEncode;
        constexpr index_t NRepeat = KDstrEncode::hs_lengthss_[number<0>{}][number<0>{}];

        index_t current_seq_k = seqlen_k_start;
        statically_indexed_array<index_t, NRepeat> k_physical_pages{};
        statically_indexed_array<index_t, NRepeat> k_offsets;

        load_physical_pages<statically_indexed_array<index_t, NRepeat>,
                            decltype(k_coord),
                            0,
                            kPageBlockSize,
                            0,
                            NRepeat,
                            kN0 / NRepeat,
                            kKVMemoryLayout,
                            true,
                            kN0>(
            page_idx, k_coord, current_seq_k, k_physical_pages, max_page_table_idx);

        kv_offset_array_transform<statically_indexed_array<index_t, NRepeat>,
                                  decltype(k_coord),
                                  0,
                                  kPageBlockSize,
                                  0,
                                  NRepeat,
                                  kN0 / NRepeat,
                                  kKVMemoryLayout,
                                  true,
                                  kN0,
                                  kVectorSize>(
            k_physical_pages, stride_k, page_stride_k, k_coord, k_offsets, current_seq_k);

        auto k_dram_window =
            make_tile_scatter_gather(k_dram_block_window_tmp.get_bottom_tensor_view(),
                                     k_dram_block_window_tmp.get_window_lengths(),
                                     {seqlen_k_start, 0},
                                     k_dist,
                                     k_offsets);

        // SRD rebasing: move the buffer descriptor base pointer to each page's start
        // address using 48-bit pointer arithmetic, so voffset only needs the small
        // within-page offset. Only applies when kPageBlockSize >= kN0.
        auto rebase_k_window = [&](auto& window, index_t physical_page) {
            if constexpr(kPageBlockSize >= kN0)
            {
                physical_page = __builtin_amdgcn_readfirstlane(physical_page);
                const auto* base_ptr =
                    k_dram_block_window_tmp.get_bottom_tensor_view().buf_.p_data_;
                const auto* page_ptr =
                    base_ptr + static_cast<long_index_t>(physical_page) * page_stride_k;
                window.set_bottom_tensor_view_data_ptr(page_ptr);
            }
        };

        rebase_k_window(k_dram_window, k_physical_pages[number<0>{}]);

        // =====================================================================
        // Scatter-gather V DRAM window setup
        // =====================================================================
        auto v_dist       = Policy::template MakeVDramTileDistribution<Problem>();
        auto v_coord      = v_dist.calculate_index();
        using VDstrEncode = typename decltype(v_dist)::DstrEncode;

        // V tensor K-dimension decomposition for page index computation.
        // In V3's distribution, the sequence (K) dimension is Hs index 0 (hs_lengthss_[0]),
        // and head_dim (N) is Hs index 1. This differs from the batch prefill pipeline where
        // sequence is Hs index 1. The Ps+Hs → rhs mapping: rhs[0]=Ps, rhs[1]=Hs[0], rhs[2]=Hs[1].
        //
        // V3 distribution for dim 0 (seq): {KPerThread, NumWarps, KThreadPerWarp} (bf16)
        //                              or: {NumIssues, LaneGroups, NumWarps}       (fp8)
        // The FIRST element is the per-thread Y iteration count; the rest are Ps (partitions).
        constexpr index_t V_KIterInner = VDstrEncode::hs_lengthss_[number<0>{}][number<0>{}];

        constexpr index_t V_KIterOuter = 1;

        constexpr index_t V_PageIdxRepeat = V_KIterInner * V_KIterOuter;

        constexpr auto VPageIndexYDims = []() {
            // rhs[1] = first Hs dim (dim 0) = sequence dimension in V3's distribution.
            // Minor index 0 is the per-thread iteration (Y dim); the rest are Ps partitions.
            constexpr index_t Y_K1 = VDstrEncode::detail::rhs_major_minor_to_ys_[1][number<0>{}];
            return sequence<Y_K1>{};
        }();

        static_assert(decltype(VPageIndexYDims)::at(0) < VDstrEncode::NDimY,
                      "V page-index Y dim must be valid");

        statically_indexed_array<index_t, V_PageIdxRepeat> v_offsets;
        statically_indexed_array<index_t, V_PageIdxRepeat> v_physical_pages{};

        // Prefetch V physical pages helper
        // kCoordAxis=0 because V3 distribution has sequence as first Hs dim (axis 0)
        auto prefetch_v_physical_pages = [&](auto k_loop_start) {
            constexpr index_t kLoopStart = decltype(k_loop_start)::value;
            load_physical_pages<statically_indexed_array<index_t, V_KIterInner>,
                                decltype(v_coord),
                                0,
                                kPageBlockSize,
                                kLoopStart,
                                V_KIterInner,
                                1,
                                kKVMemoryLayout,
                                false,
                                kN0>(
                page_idx, v_coord, current_seq_k, v_physical_pages, max_page_table_idx);
        };

        // Update V offsets using pre-loaded physical pages
        auto update_v_offsets = [&](auto k_loop_start) {
            constexpr index_t kLoopStart = decltype(k_loop_start)::value;
            kv_offset_array_transform<statically_indexed_array<index_t, V_KIterInner>,
                                      decltype(v_coord),
                                      0,
                                      kPageBlockSize,
                                      kLoopStart,
                                      V_KIterInner,
                                      1,
                                      kKVMemoryLayout,
                                      false,
                                      kN0,
                                      kVectorSize>(
                v_physical_pages, stride_v, page_stride_v, v_coord, v_offsets, current_seq_k);
        };

        // Initial V offset computation
        prefetch_v_physical_pages(number<0>{});
        update_v_offsets(number<0>{});

        auto v_dram_window =
            make_tile_scatter_gather(v_dram_block_window_tmp.get_bottom_tensor_view(),
                                     v_dram_block_window_tmp.get_window_lengths(),
                                     {seqlen_k_start, 0},
                                     v_dist,
                                     v_offsets,
                                     number<0>{}, // HsGatherDim: sequence is first Hs dim in V3
                                     number<1>{}, // NumCoord
                                     VPageIndexYDims);

        auto rebase_v_window = [&](auto& window, index_t physical_page) {
            if constexpr(kPageBlockSize >= kN0)
            {
                physical_page = __builtin_amdgcn_readfirstlane(physical_page);
                const auto* base_ptr =
                    v_dram_block_window_tmp.get_bottom_tensor_view().buf_.p_data_;
                const auto* page_ptr =
                    base_ptr + static_cast<long_index_t>(physical_page) * page_stride_v;
                window.set_bottom_tensor_view_data_ptr(page_ptr);
            }
        };

        rebase_v_window(v_dram_window, v_physical_pages[number<0>{}]);

        // prefetch K tile
        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;
        static_assert(1 == k0_loops);
        static_assert(1 == k1_loops);
        static_assert(kN0 == kK1);

        constexpr index_t NumWarpGroups = Problem::kBlockSize / Policy::NumThreadPerWarpGroup;
        static_assert(NumWarpGroups == 2);

        constexpr int K_mem_su_ld_insts = k_dram_window.get_num_of_access();
        constexpr int V_mem_su_ld_insts = v_dram_window.get_num_of_access();

        index_t current_k_seq = seqlen_k_start;
        index_t current_v_seq = seqlen_k_start;

        // =====================================================================
        // K/V mem load lambdas (load-only, no page offset update)
        // =====================================================================
        auto K_mem_load = [&](auto k_lds_write_idx) {
            async_load_tile(k_lds_window_store(k_lds_write_idx),
                            k_dram_window,
                            number<-1>{},
                            bool_constant<false>{});
        };

        auto K_lds_load = [&](auto k_lds_read_idx) {
            kv_tile.k_tile =
                load_tile_with_offset(k_lds_window_load(k_lds_read_idx), k_lds_load_offset);
        };

        auto V_mem_load = [&](auto v_lds_write_idx) {
            async_load_tile(v_lds_window_store(v_lds_write_idx),
                            v_dram_window,
                            number<-1>{},
                            bool_constant<false>{});
        };

        // Page offset advance lambdas — split into issue/consume pairs so
        // the global_load_dword (page table lookup) can be issued BEFORE the
        // buffer_loads from cl_load, placing it oldest in the vmcnt FIFO.
        // At consume time, s_waitcnt(N) drains only the oldest global_load
        // while keeping the N buffer_loads in flight.
        //
        // Page table index clamping happens inside load_physical_pages() via
        // max_page_table_idx, so the counters can advance freely past seqlen_k_end.
        // Past-end lookups return a valid (but stale) page; the loaded data is
        // discarded by the loop exit.

        // Issue: fire global_load_dword for next iteration's page table (oldest in vmcnt FIFO)
        auto K_page_issue = [&]() {
            current_k_seq += kN0;
            load_physical_pages<statically_indexed_array<index_t, NRepeat>,
                                decltype(k_coord),
                                0,
                                kPageBlockSize,
                                0,
                                NRepeat,
                                kN0 / NRepeat,
                                kKVMemoryLayout,
                                true,
                                kN0>(
                page_idx, k_coord, current_k_seq, k_physical_pages, max_page_table_idx);
        };

        // Consume: use result to compute offsets (needs vmcnt to drain global_load_dword)
        auto K_page_consume = [&]() {
            // Wait for global_load_dword (oldest) to complete.
            // K_mem_su_ld_insts buffer_loads from cl_load(memK) remain in flight.
            s_waitcnt<K_mem_su_ld_insts>();
            kv_offset_array_transform<statically_indexed_array<index_t, NRepeat>,
                                      decltype(k_coord),
                                      0,
                                      kPageBlockSize,
                                      0,
                                      NRepeat,
                                      kN0 / NRepeat,
                                      kKVMemoryLayout,
                                      true,
                                      kN0,
                                      kVectorSize>(
                k_physical_pages, stride_k, page_stride_k, k_coord, k_offsets, current_k_seq);
            k_dram_window.update_page_idx(k_offsets);
            rebase_k_window(k_dram_window, k_physical_pages[number<0>{}]);
        };

        auto V_page_issue = [&]() {
            current_v_seq += kN0;
            current_seq_k = current_v_seq; // sync for prefetch_v_physical_pages
            prefetch_v_physical_pages(number<0>{});
        };

        auto V_page_consume = [&]() {
            // Wait for global_load_dword (oldest) to complete.
            // V_mem_su_ld_insts buffer_loads from cl_load(memV) remain in flight.
            s_waitcnt<V_mem_su_ld_insts>();
            update_v_offsets(number<0>{});
            v_dram_window.update_page_idx(v_offsets);
            rebase_v_window(v_dram_window, v_physical_pages[number<0>{}]);
        };

        auto V_lds_load = [&](auto v_lds_read_idx) {
            load_tile_transpose_with_offset(
                kv_tile.v_tile, v_lds_window_load(v_lds_read_idx), v_lds_load_offset);
        };

        decltype(m) m_old;
        SMPLComputeDataType o_acc_scale; // rescale o_acc in fmha_alu1() & fmha_alu_D_upd()
        statically_indexed_array<decltype(sp(number<0>{}).sp_compute), 2> sp_delta;

        // KV_BLOCKSCALE: per-page descale factors, double-buffered by LDS buffer index.
        // saved_k_descale[buf] / saved_v_descale[buf] hold descales for the K tile
        // currently in LDS buffer `buf`. Updated when a new K tile's pages are available,
        // before K_page_issue overwrites k_physical_pages.
        [[maybe_unused]] float saved_k_descale[2] = {1.0f, 1.0f};
        [[maybe_unused]] float saved_v_descale[2] = {1.0f, 1.0f};
        // v_descale of the most recent GEMM1 (1.0 = identity for first iteration).
        // o_acc is maintained in v_descale_prev-scaled space; the ratio
        // v_descale_prev / saved_v_descale[si] is folded into o_acc_scale.
        [[maybe_unused]] float v_descale_prev = 1.0f;
        // k_descale-adjusted scale_s for sp_delta FMA: scale_s_k = scale_s * k_descale[i].
        // The full-tile k_descale multiply on sp_compute is folded into:
        //   (1) scalar row_max: m_raw * k_descale, and
        //   (2) sp_delta FMA b-term: fma(s_raw, scale_s_k, -scale_s * m)
        [[maybe_unused]] float scale_s_k = scale_s;

        // Load descale factors from current k_physical_pages[0] and store into slot `buf_idx`.
        // Must be called BEFORE K_page_issue() overwrites k_physical_pages.
        auto save_descales = [&](index_t buf_idx) {
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::KV_BLOCKSCALE)
            {
                const index_t scale_offset =
                    k_physical_pages[number<0>{}] * nblock_stride_kv_block_descale +
                    block_indices.kv_head_idx * nhead_stride_kv_block_descale;
                saved_k_descale[buf_idx] = k_descale_ptr[scale_offset];
                saved_v_descale[buf_idx] = v_descale_ptr[scale_offset];
            }
        };

        auto fmha_logits_trans = [&](auto sp_reg_idx) {
            if constexpr(kHasLogitsSoftCap)
            {
                auto apply_logits_transform = [&variant, &variant_params, &block_indices](
                                                  auto& logits) {
                    logits = variant.LogitsTransform(variant_params,
                                                     variant.QueryTransform(variant_params, logits),
                                                     block_indices.batch_idx,
                                                     block_indices.qo_head_idx,
                                                     block_indices.kv_head_idx);
                };

                tile_elementwise_inout(apply_logits_transform, sp(sp_reg_idx).sp_compute);
            }
        };

        // Whether k_descale can be folded into scalar row_max + sp_delta FMA.
        // Not possible with LogitsSoftCap: the logits transform (tanh) needs
        // descaled values, so the full-tile k_descale pass must remain.
        static constexpr bool kFoldKDescale =
            (QScaleEnum == BlockAttentionQuantScaleEnum::KV_BLOCKSCALE) && !kHasLogitsSoftCap;

        auto fmha_alu0 = [&](auto sp_reg_idx) {
            m_old = m; // m{j-1}
            // kFoldKDescale: reduce on raw (undescaled) values with -MAX_FLOAT init,
            // then fold k_descale into the scalar row_max. This eliminates the
            // full-tile pk_mul_f32 pass after GEMM0 (sp_compute *= k_descale).
            if constexpr(kFoldKDescale)
            {
                // Reset m to -MAX so the in-place reduce starts fresh
                static_for<0, m.thread_buf_.size(), 1>{}(
                    [&](auto i) { m.thread_buf_[i] = -numeric<SMPLComputeDataType>::max(); });
            }
            block_tile_reduce(m, sp(sp_reg_idx).sp_compute, sequence<1>{}, f_max);
            block_tile_reduce_sync(m, f_max, bool_constant<false>{}, bool_constant<false>{});

            if constexpr(kFoldKDescale)
            {
                constexpr index_t si = decltype(sp_reg_idx)::value;
                // Fold k_descale into scalar row_max: max(s_raw) * d_i == max(s_raw * d_i)
                m.thread_buf_[0] *= saved_k_descale[si];
                // Running max in descaled domain (same leaky-max as before)
                m.thread_buf_[0] = f_max(m.thread_buf_[0], m_old.thread_buf_[0]);
                // FP8 shift: exp2(s*ss_k - ss*m) implicitly scales P by 2^shift
#if CK_TILE_USE_OCP_FP8
                m.thread_buf_[0] -= OCP_FP8_SHIFT;
#else
                m.thread_buf_[0] -= FNUZ_FP8_SHIFT;
#endif
                // Precompute k_descale-adjusted scale_s for sp_delta FMA
                scale_s_k = scale_s * saved_k_descale[si];
            }
            else if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::KV_BLOCKSCALE)
            {
                // LogitsSoftCap + KV_BLOCKSCALE: sp_compute already descaled by
                // full-tile pass in gemm/cl_calc. Reduction init was m_old, so
                // m already incorporates the running max. Apply FP8 shift.
#if CK_TILE_USE_OCP_FP8
                m.thread_buf_[0] -= OCP_FP8_SHIFT;
#else
                m.thread_buf_[0] -= FNUZ_FP8_SHIFT;
#endif
            }

            constexpr auto p_spans =
                std::decay_t<decltype(sp(sp_reg_idx).sp_compute)>::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    if constexpr(kHasLogitsSoftCap)
                    {
                        sp_delta(sp_reg_idx)(i_j_idx) =
                            sp(sp_reg_idx).sp_compute(i_j_idx) - m(i_j_idx);
                    }
                    else if constexpr(kFoldKDescale)
                    {
                        // fma(s_raw, scale_s * k_descale, -scale_s * m)
                        // = scale_s * (s_raw * k_descale - m_latest + S)
                        sp_delta(sp_reg_idx)(i_j_idx) = detail::fma_impl_vsv(
                            sp(sp_reg_idx).sp_compute(i_j_idx), scale_s_k, -scale_s * m(i_j_idx));
                    }
                    else
                    {
                        sp_delta(sp_reg_idx)(i_j_idx) = detail::fma_impl_vsv(
                            sp(sp_reg_idx).sp_compute(i_j_idx), scale_s, -scale_s * m(i_j_idx));
                    }
                });
            });
        };

        auto fmha_alu1 = [&](auto sp_reg_idx) {
            constexpr auto p_spans =
                std::decay_t<decltype(sp(sp_reg_idx).sp_compute)>::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    sp(sp_reg_idx).sp_compute(i_j_idx) =
                        ck_tile::exp2(sp_delta(sp_reg_idx)(i_j_idx));
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                sp(sp_reg_idx).sp_compute,
                sequence<1>{},
                f_sum,
                SMPLComputeDataType{0}); // rowsum(Pcompute{j})
            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{}, bool_constant<false>{});

            // l{j}
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const auto tmp       = [&] {
                    if constexpr(kHasLogitsSoftCap)
                    {
                        return ck_tile::exp2(m_old[i_idx] - m[i_idx]);
                    }
                    else
                    {
                        return ck_tile::exp2(scale_s * (m_old[i_idx] - m[i_idx]));
                    }
                }();
                l(i_idx) = detail::add_impl_vv(tmp * l[i_idx], rowsum_p[i_idx]);
            });

            // update partial o_acc [0, fmha_alu_D_reg_cnt)
            static_for<0, fmha_alu_D_reg_cnt, 1>{}([&](auto idx) {
                o_acc.thread_buf_[idx] = detail::mul_impl_vv(o_acc.thread_buf_[idx], o_acc_scale);
            });

            // P conversion with inline asm anchoring
            static_assert(sp(sp_reg_idx).p.thread_buf_.size() % 2 == 0);
            static_for<0, sp(sp_reg_idx).p.thread_buf_.size(), 2>{}([&](auto idx) {
                float x = p_compute_element_func(sp(sp_reg_idx).sp_compute.thread_buf_[idx]);
                float y = p_compute_element_func(sp(sp_reg_idx).sp_compute.thread_buf_[idx + 1]);
                if constexpr(std::is_same_v<PDataType, fp16_t>)
                {
                    auto casted                           = detail::cvt_pk_fp16_f32(x, y);
                    sp(sp_reg_idx).p.thread_buf_[idx]     = casted.x;
                    sp(sp_reg_idx).p.thread_buf_[idx + 1] = casted.y;
                }
                else if constexpr(std::is_same_v<PDataType, bf16_t>)
                {
                    auto casted                           = detail::cvt_pk_bf16_f32(x, y);
                    sp(sp_reg_idx).p.thread_buf_[idx]     = casted.x;
                    sp(sp_reg_idx).p.thread_buf_[idx + 1] = casted.y;
                }
                else if constexpr(std::is_same_v<PDataType, fp8_t>)
                {
                    uint32_t packed = detail::cvt_pk_fp8_f32(x, y);
                    sp(sp_reg_idx).p.thread_buf_[idx] =
                        bit_cast<fp8_t>(static_cast<uint8_t>(packed & 0xFF));
                    sp(sp_reg_idx).p.thread_buf_[idx + 1] =
                        bit_cast<fp8_t>(static_cast<uint8_t>((packed >> 8) & 0xFF));
                }
            });
        };

        // KV_BLOCKSCALE k_descale full-tile pass: only needed when k_descale
        // cannot be folded into fmha_alu0 (i.e. LogitsSoftCap requires descaled input).
        auto apply_k_descale = [&](auto sp_reg_idx) {
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::KV_BLOCKSCALE &&
                         !kFoldKDescale)
            {
                constexpr index_t si = decltype(sp_reg_idx)::value;
                fp32x2_t pk_k_descale;
                pk_k_descale.x = saved_k_descale[si];
                pk_k_descale.y = saved_k_descale[si];
                static_for<0, sp(sp_reg_idx).sp_compute.thread_buf_.size(), 2>{}([&](auto idx) {
                    fp32x2_t input;
                    input.x     = sp(sp_reg_idx).sp_compute.thread_buf_[idx];
                    input.y     = sp(sp_reg_idx).sp_compute.thread_buf_[idx + 1];
                    auto output = detail::pk_mul_f32(input, pk_k_descale);
                    sp(sp_reg_idx).sp_compute.thread_buf_[idx]     = output.x;
                    sp(sp_reg_idx).sp_compute.thread_buf_[idx + 1] = output.y;
                });
            }
        };

        auto gemm = [&](auto sp_reg_idx, auto gemm_idx) {
            if constexpr(gemm_idx == 0)
            {
                clear_tile(sp(sp_reg_idx).sp_compute); // initialize C
                gemm_0(sp(sp_reg_idx).sp_compute,
                       get_slice_tile(q_tile,
                                      sequence<0, (k0_loops - 1) * kK0>{},
                                      sequence<kM0, k0_loops * kK0>{}),
                       get_slice_tile(kv_tile.k_tile,
                                      sequence<0, (k0_loops - 1) * kK0>{},
                                      sequence<kN0, k0_loops * kK0>{}));
                apply_k_descale(sp_reg_idx);
            }
            else
            {
                gemm_1(o_acc,
                       get_slice_tile(sp(sp_reg_idx).p,
                                      sequence<0, (k1_loops - 1) * kK1>{},
                                      sequence<kM0, k1_loops * kK1>{}),
                       get_slice_tile(kv_tile.v_tile,
                                      sequence<0, (k1_loops - 1) * kK1>{},
                                      sequence<kN1, k1_loops * kK1>{}));
            }
        };

        auto cl_calc = [&](auto sp_reg_idx, auto gemm_idx) {
            if constexpr(gemm_idx == 0)
            {
                clear_tile(sp(sp_reg_idx).sp_compute); // initialize C
                gemm_0(sp(sp_reg_idx).sp_compute,
                       get_slice_tile(q_tile,
                                      sequence<0, (k0_loops - 1) * kK0>{},
                                      sequence<kM0, k0_loops * kK0>{}),
                       get_slice_tile(kv_tile.k_tile,
                                      sequence<0, (k0_loops - 1) * kK0>{},
                                      sequence<kN0, k0_loops * kK0>{}));
                apply_k_descale(sp_reg_idx);
            }
            else
            {
                gemm_1(o_acc,
                       get_slice_tile(sp(sp_reg_idx).p,
                                      sequence<0, (k1_loops - 1) * kK1>{},
                                      sequence<kM0, k1_loops * kK1>{}),
                       get_slice_tile(kv_tile.v_tile,
                                      sequence<0, (k1_loops - 1) * kK1>{},
                                      sequence<kN1, k1_loops * kK1>{}));
                fmha_alu0(number<1>{} - sp_reg_idx);
            }
        };

        constexpr index_t num_unpack_insts =
            (kHasLogitsSoftCap ? 48 : (std::is_same_v<KDataType, fp8_t> ? 36 : 26));
        fp32x2_t pk_o_acc_scale;
        auto fmha_alu_D_upd_unpack = [&](auto sp_reg_idx) {
            o_acc_scale = [&] {
                if constexpr(kHasLogitsSoftCap)
                {
                    return ck_tile::exp2(m_old.thread_buf_[0] - m.thread_buf_[0]);
                }
                else
                {
                    return ck_tile::exp2(scale_s * (m_old.thread_buf_[0] - m.thread_buf_[0]));
                }
            }();

            // Fold v_descale ratio into o_acc_scale: transition o_acc from
            // v_descale_prev-scaled space to saved_v_descale[si]-scaled space.
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::KV_BLOCKSCALE)
            {
                constexpr index_t si = decltype(sp_reg_idx)::value;
                o_acc_scale *= v_descale_prev / saved_v_descale[si];
                v_descale_prev = saved_v_descale[si];
            }

            static_assert(num_unpack_insts % 2 == 0 &&
                          (fmha_alu_D_reg_cnt + num_unpack_insts) <= o_acc.thread_buf_.size());
            static_for<fmha_alu_D_reg_cnt, fmha_alu_D_reg_cnt + num_unpack_insts, 1>{}(
                [&](auto idx) { o_acc.thread_buf_[idx] *= o_acc_scale; });
            pk_o_acc_scale.x = o_acc_scale;
            pk_o_acc_scale.y = o_acc_scale;
        };

        auto fmha_alu_D_upd_pack = [&] {
            constexpr index_t issued_unpack_insts = fmha_alu_D_reg_cnt + num_unpack_insts;
            static_for<issued_unpack_insts, o_acc.thread_buf_.size(), 2>{}([&](auto idx) {
                fp32x2_t input;
                input.x = o_acc.thread_buf_[idx];
                input.y = o_acc.thread_buf_[idx + 1];

                auto output = detail::pk_mul_f32(input, pk_o_acc_scale);

                o_acc.thread_buf_[idx]     = output.x;
                o_acc.thread_buf_[idx + 1] = output.y;
            });
        };

        auto fmha_alu_D_upd = [&](auto sp_reg_idx) {
            fmha_alu_D_upd_unpack(sp_reg_idx);
            fmha_alu_D_upd_pack();
        };

        auto fmha_mask = [&](auto sp_reg_idx) {
            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                bool need_perpixel_check = mask.IsEdgeTile(
                    q_origin.at(number<0>{}), kv_token_start, number<kM0>{}, number<kN0>{});
                if(need_perpixel_check)
                {
                    set_tile_if(
                        sp(sp_reg_idx).sp_compute,
                        -numeric<SMPLComputeDataType>::infinity(),
                        [&](auto tile_idx) {
                            const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                            const auto col = kv_token_start + tile_idx.at(number<1>{});
                            return !variant.LogitsMask(variant_params,
                                                       block_indices.batch_idx,
                                                       row,
                                                       col,
                                                       block_indices.qo_head_idx,
                                                       block_indices.kv_head_idx);
                        },
                        partition_index);
                }
            }
        };

        auto cl_load = [&](auto load_type, auto mem_wr_idx, auto lds_rd_idx) {
            if constexpr(load_type == 0)
            {
                V_mem_load(mem_wr_idx);
                K_lds_load(lds_rd_idx);
            }
            else
            {
                K_mem_load(mem_wr_idx);
                V_lds_load(lds_rd_idx);
            }
        };

        auto core_loop = [&](auto cl_p) {
            auto gemm0 = number<0>{};
            auto gemm1 = number<1>{};

            auto memV = number<0>{};
            auto memK = number<1>{};

            using Scheduler = BatchPrefillCoreLoopScheduler<Problem>;

            auto iteration = [&](auto pi) {
                auto xdl_SP_p01_reg_idx = number<1>{} - pi;
                auto xdl_SP_p23_reg_idx = pi;

                auto K_w0_lds_wr_idx = number<1>{} - pi;
                auto V_w0_lds_wr_idx = pi;
                auto K_w0_lds_rd_idx = pi;
                auto V_w0_lds_rd_idx = pi;

                auto K_w4_lds_wr_idx = number<1>{} - pi;
                auto V_w4_lds_wr_idx = number<1>{} - pi;
                auto K_w4_lds_rd_idx = number<1>{} - pi;
                auto V_w4_lds_rd_idx = pi;

                bool result = true;

                if constexpr(cl_p == 0)
                {
                    __builtin_amdgcn_sched_barrier(0);
                    // phase0
                    if constexpr(pi == 0)
                    {
                        CK_TILE_FMHA_V3_ASM_MARKER("phase0 Wave0-3 (pi=0)");
                    }
                    else
                    {
                        CK_TILE_FMHA_V3_ASM_MARKER("phase0 Wave0-3 (pi=1)");
                    }
                    s_waitcnt<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>();
                    __builtin_amdgcn_sched_barrier(0);
#if CK_TILE_FMHA_V3_ADD_SBARRIER_FOR_PHASE0
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
#endif
                    if constexpr(pi == 1)
                    {
                        if constexpr(!std::is_same_v<KDataType, fp8_t>)
                        {
                            asm volatile("s_nop 1");
                            __builtin_amdgcn_sched_barrier(0);
                        }
                        else
                        {
                            asm volatile("s_nop 3");
                            __builtin_amdgcn_sched_barrier(0);
                        }
                    }
                    else
                    {
                        if constexpr(std::is_same_v<KDataType, fp8_t>)
                        {
                            asm volatile("s_nop 3");
                            __builtin_amdgcn_sched_barrier(0);
                        }
                    }
                    cl_calc(xdl_SP_p01_reg_idx, gemm0);
                    fmha_alu1(xdl_SP_p23_reg_idx);
                    fmha_logits_trans(xdl_SP_p01_reg_idx);

                    Scheduler::schedule(cl_p, number<0>{});
                    __builtin_amdgcn_sched_barrier(0);
                    // phase1
                    CK_TILE_FMHA_V3_ASM_MARKER("phase1 Wave0-3");
                    s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    save_descales(K_w0_lds_wr_idx);
                    K_page_issue();                                  // global_load_dword FIRST
                    __builtin_amdgcn_sched_barrier(0);               // prevent reorder
                    cl_load(memK, K_w0_lds_wr_idx, V_w0_lds_rd_idx); // buffer_loads SECOND
                    Scheduler::schedule(cl_p, number<1>{});
                    fmha_mask(xdl_SP_p01_reg_idx);
                    __builtin_amdgcn_sched_barrier(0); // prevent reorder
                    K_page_consume();                  // vmcnt(K_mem_su_ld_insts)

                    __builtin_amdgcn_sched_barrier(0);
                    // phase2
                    CK_TILE_FMHA_V3_ASM_MARKER("phase2 Wave0-3");
                    s_waitcnt<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>();
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    if constexpr(std::is_same_v<KDataType, fp8_t>)
                    {
                        asm volatile("s_nop 3");
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    cl_calc(xdl_SP_p23_reg_idx, gemm1);
                    fmha_alu_D_upd_unpack(xdl_SP_p23_reg_idx);
                    Scheduler::schedule(cl_p, number<2>{});
                    __builtin_amdgcn_sched_barrier(0);
                    fmha_alu_D_upd_pack();

                    __builtin_amdgcn_sched_barrier(0);
                    // phase3
                    CK_TILE_FMHA_V3_ASM_MARKER("phase3 Wave0-3");
                    s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    V_page_issue();                                  // global_load_dword FIRST
                    __builtin_amdgcn_sched_barrier(0);               // prevent reorder
                    cl_load(memV, V_w0_lds_wr_idx, K_w0_lds_rd_idx); // buffer_loads SECOND
                    Scheduler::schedule(cl_p, number<3>{});
                    // Page offset update at loop increment
                    kv_token_start += kN0;
                    if(num_total_loop <= ++i_total_loops)
                    {
                        result = false;
                    }
                    __builtin_amdgcn_sched_barrier(0); // prevent reorder
                    V_page_consume();                  // vmcnt(V_mem_su_ld_insts)
                }
                else
                {
#if CK_TILE_FMHA_V3_ADD_SBARRIER_FOR_PHASE0
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
#endif
                    __builtin_amdgcn_sched_barrier(0);
                    // phase0
                    if constexpr(pi == 0)
                    {
                        CK_TILE_FMHA_V3_ASM_MARKER("phase0 Wave4-7 (pi=0)");
                    }
                    else
                    {
                        CK_TILE_FMHA_V3_ASM_MARKER("phase0 Wave4-7 (pi=1)");
                    }
                    V_page_issue();                                  // global_load_dword FIRST
                    __builtin_amdgcn_sched_barrier(0);               // prevent reorder
                    cl_load(memV, V_w4_lds_wr_idx, K_w4_lds_rd_idx); // buffer_loads SECOND
                    Scheduler::schedule(cl_p, number<0>{});
                    __builtin_amdgcn_sched_barrier(0); // prevent reorder
                    V_page_consume();                  // vmcnt(V_mem_su_ld_insts)
                    __builtin_amdgcn_sched_barrier(0);
                    // phase1
                    CK_TILE_FMHA_V3_ASM_MARKER("phase1 Wave4-7");
                    s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts, waitcnt_arg::kMaxExpCnt, 0>();
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);

                    if constexpr(!std::is_same_v<KDataType, fp8_t>)
                    {
                        asm volatile("s_nop 1");
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    else
                    {
                        asm volatile("s_nop 3");
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    cl_calc(xdl_SP_p01_reg_idx, gemm0);
                    fmha_alu1(xdl_SP_p23_reg_idx);
                    fmha_logits_trans(xdl_SP_p01_reg_idx);

                    Scheduler::schedule(cl_p, number<1>{});
                    __builtin_amdgcn_sched_barrier(0);
                    // phase2
                    CK_TILE_FMHA_V3_ASM_MARKER("phase2 Wave4-7");
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    save_descales(K_w4_lds_wr_idx);
                    K_page_issue();                                  // global_load_dword FIRST
                    __builtin_amdgcn_sched_barrier(0);               // prevent reorder
                    cl_load(memK, K_w4_lds_wr_idx, V_w4_lds_rd_idx); // buffer_loads SECOND
                    Scheduler::schedule(cl_p, number<2>{});
                    fmha_mask(xdl_SP_p01_reg_idx);
                    __builtin_amdgcn_sched_barrier(0); // prevent reorder
                    K_page_consume();                  // vmcnt(K_mem_su_ld_insts)

                    // Page offset update at loop increment
                    kv_token_start += kN0;
                    if(num_total_loop <= ++i_total_loops)
                    {
                        result = false;
                    }

                    __builtin_amdgcn_sched_barrier(0);
                    // phase3
                    CK_TILE_FMHA_V3_ASM_MARKER("phase3 Wave4-7");
                    s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts, waitcnt_arg::kMaxExpCnt, 0>();
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    if constexpr(!std::is_same_v<KDataType, fp8_t>)
                    {
                        asm volatile("s_nop 1");
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    else
                    {
                        asm volatile("s_nop 3");
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    cl_calc(xdl_SP_p23_reg_idx, gemm1);
                    fmha_alu_D_upd_unpack(xdl_SP_p23_reg_idx);
                    Scheduler::schedule(cl_p, number<3>{});
                    __builtin_amdgcn_sched_barrier(0);
                    fmha_alu_D_upd_pack();
                }
                return result;
            };
            return iteration(number<0>{}) && iteration(number<1>{});
        };

        auto fmha_post_process = [&](auto d) {
            auto ps_pi        = number<1>{} - d;
            auto V_lds_rd_idx = ps_pi;

            if(1 < num_total_loop)
            {
                s_waitcnt<K_mem_su_ld_insts>();
            }
            else
            {
                s_waitcnt<0>();
            }
            __builtin_amdgcn_s_barrier();

            V_lds_load(V_lds_rd_idx);
            fmha_alu1(ps_pi);

            s_waitcnt<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>();

            auto xdl_SP_p23_reg_idx = ps_pi;
            gemm(xdl_SP_p23_reg_idx, /*gemm_idx=*/number<1>{});
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::KV_BLOCKSCALE)
            {
                constexpr index_t si = decltype(ps_pi)::value;
                v_descale_prev       = saved_v_descale[si];
            }
        };

        if(num_total_loop > 0)
        {
            // pre-stage
            {
                CK_TILE_FMHA_V3_ASM_MARKER("before pre-stage");
                // Save descales for K0 (buf 0) before K_page_issue overwrites k_physical_pages
                save_descales(0);
                // (1) load K0 to LDS & VGPR
                K_page_issue();          // global_load for K1 offset (FIRST)
                K_mem_load(number<0>{}); // buffer_load K0 (SECOND)
                K_page_consume();        // s_waitcnt vmcnt(K_mem_su_ld_insts), transform

                s_waitcnt<0>();
                __builtin_amdgcn_s_barrier();

                K_lds_load(number<0>{}); // lds_K0

                s_waitcnt<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>();
                __builtin_amdgcn_s_barrier();

                // (2) prefetch K1 and V0 to LDS in parallel with GEMM0
                V_page_issue();          // global_load for V1 offset
                V_mem_load(number<0>{}); // buffer_load V0
                V_page_consume();        // s_waitcnt vmcnt(V_mem_su_ld_insts), transform
                if(1 < num_total_loop)
                {
                    // Save descales for K1 (buf 1) — k_physical_pages still has K1 pages
                    save_descales(1);
                    K_page_issue();          // global_load for K2 offset
                    K_mem_load(number<1>{}); // buffer_load K1
                    K_page_consume();        // s_waitcnt vmcnt(K_mem_su_ld_insts), transform
                }

                // (3) mfma (Q*K0) + softmax
                gemm(number<0>{}, /*gemm_idx=*/number<0>{});
                fmha_logits_trans(number<0>{});
                fmha_mask(number<0>{});
                fmha_alu0(number<0>{});
                fmha_alu_D_upd(number<0>{});

                kv_token_start += kN0;
                ++i_total_loops;
                if(num_total_loop <= i_total_loops)
                {
                    goto label_main_loops_exit;
                }

                if(2 < num_total_loop)
                {
                    // Save descales for K2 (buf 0) — k_physical_pages has K2 pages
                    save_descales(0);
                    // K2 at seq=start+2*kN0 (k offsets already point here)
                    K_page_issue();          // global_load for K3 offset
                    K_mem_load(number<0>{}); // buffer_load K2
                    K_page_consume();        // s_waitcnt vmcnt(K_mem_su_ld_insts), transform
                }

                // drain K1 + V0 async loads before core_loop reads K1 from LDS
                s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                __builtin_amdgcn_s_barrier();

                CK_TILE_FMHA_V3_ASM_MARKER("end pre-stage");
            }

            if(1 < num_total_loop)
            {
                // V offsets point to start+kN0 (advanced after V0 load)
                if(warp_group_id == 0)
                {
                    V_page_issue();          // global_load for V2 offset
                    V_mem_load(number<1>{}); // buffer_load V1
                    V_page_consume();        // s_waitcnt vmcnt(V_mem_su_ld_insts), transform
                    K_lds_load(number<1>{}); // K1

                    __builtin_amdgcn_s_setprio(0);
                    __builtin_amdgcn_s_barrier();
                    while(core_loop(number<0>{}))
                        ;
                }
                if(warp_group_id != 0)
                {
                    __builtin_amdgcn_s_setprio(1);
                    __builtin_amdgcn_s_barrier();
                    while(core_loop(number<1>{}))
                        ;
                }
            }
        label_main_loops_exit:
            if(num_total_loop % 2)
            {
                fmha_post_process(number<1>{});
            }
            if(!(num_total_loop % 2))
            {
                fmha_post_process(number<0>{});
            }
        }

        // store lse
        if constexpr(kStoreLSE)
        {
            auto lse = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

            constexpr auto lse_spans = decltype(lse)::get_distributed_spans();
            sweep_tile_span(lse_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                lse(i_idx)           = m[i_idx] / C_LOG2E + log(l[i_idx]);
            });

            store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
        }

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::KV_BLOCKSCALE)
                {
                    if constexpr(FmhaMask::IsMasking)
                        return l[i_idx] == 0.f ? 0.f : v_descale_prev / l[i_idx];
                    else
                        return v_descale_prev / l[i_idx];
                }
                else
                {
                    if constexpr(FmhaMask::IsMasking)
                        return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                    else
                        return 1 / l[i_idx];
                }
            }();
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        o_acc = tile_elementwise_in(o_acc_element_func, o_acc);

        return o_acc;
#endif // !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx950__)
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_DEVICE auto
    operator()(multi_index<2> partition_index,
               const QDramBlockWindowTmp& __restrict__ q_dram_block_window_tmp, // M0*K0 tile
               const KDramBlockWindowTmp& __restrict__ k_dram_block_window_tmp, // N0*K0 tile
               const VDramBlockWindowTmp& __restrict__ v_dram_block_window_tmp, // N1*K1 tile
               LSEDramBlockWindowTmp& __restrict__ lse_dram_block_window_tmp,   // M0*1 tile
               FmhaMask mask,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               KDataType* __restrict__ smem_k0,
               KDataType* __restrict__ smem_k1,
               VDataType* __restrict__ smem_v0,
               VDataType* __restrict__ smem_v1,
               // Paged KV cache parameters
               const index_t* page_idx,
               index_t stride_k,
               index_t stride_v,
               index_t page_stride_k,
               index_t page_stride_v,
               index_t max_page_table_idx = 0x7FFFFFFF,
               // KV_BLOCKSCALE parameters
               const float* k_descale_ptr             = nullptr,
               const float* v_descale_ptr             = nullptr,
               index_t nblock_stride_kv_block_descale = 0,
               index_t nhead_stride_kv_block_descale  = 0) const
    {
        using namespace ck_tile;

        return operator()(partition_index,
                          q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          identity{},
                          v_dram_block_window_tmp,
                          identity{},
                          lse_dram_block_window_tmp,
                          identity{},
                          identity{},
                          identity{},
                          identity{},
                          mask,
                          scale_s,
                          variant,
                          variant_params,
                          block_indices,
                          smem_k0,
                          smem_k1,
                          smem_v0,
                          smem_v1,
                          page_idx,
                          stride_k,
                          stride_v,
                          page_stride_k,
                          page_stride_v,
                          max_page_table_idx,
                          k_descale_ptr,
                          v_descale_ptr,
                          nblock_stride_kv_block_descale,
                          nhead_stride_kv_block_descale);
    }
};

} // namespace ck_tile
