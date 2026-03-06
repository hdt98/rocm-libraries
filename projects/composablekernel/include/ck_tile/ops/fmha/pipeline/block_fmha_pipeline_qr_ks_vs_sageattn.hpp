// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/block/cast_tile_mx.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_sageattn_default_policy.hpp"
#include "ck_tile/ops/gemm/warp/warp_wmma_gemm_gfx11_utils.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// SageAttention V3 FMHA pipeline (qr_ks_vs variant).
// Differences from BlockFmhaPipelineQRKSVS (MX mode):
//   1. Requires QScaleEnum == BlockAttentionQuantScaleEnum::SAGEATTN_V3
//   2. operator() accepts a delta_s DRAM block window and p_scale_factor
//   3. After GEMM0, adds delta_s correction to s_acc (broadcast along N)
//   4. Level-1 P scaling: multiplies p_compute by p_scale_factor before cast_tile_mx
//   5. After GEMM1, divides the result by p_scale_factor before accumulating into o_acc
template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSSageAttnDefaultPolicy>
struct BlockFmhaPipelineQRKSVSSageAttn
{
    using Problem               = remove_cvref_t<Problem_>;
    using Policy                = remove_cvref_t<Policy_>;
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType          = remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType   = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using PDataType             = remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType          = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using QScaleDataType        = remove_cvref_t<typename Problem::QScaleDataType>;
    using KScaleDataType        = remove_cvref_t<typename Problem::KScaleDataType>;
    using VScaleDataType        = remove_cvref_t<typename Problem::VScaleDataType>;
    using PScaleDataType        = remove_cvref_t<typename Problem::PScaleDataType>;
    using AttentionVariant      = remove_cvref_t<typename Problem::AttentionVariant>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;

    using BlockFmhaShape             = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout                    = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static constexpr bool kQLoadOnce = true;
    static_assert(kQLoadOnce == Policy::QLoadOnce);

    // SA3 requires SAGEATTN_V3 quant scale mode
    static_assert(Problem::QScaleEnum == BlockAttentionQuantScaleEnum::SAGEATTN_V3,
                  "BlockFmhaPipelineQRKSVSSageAttn requires QScaleEnum == SAGEATTN_V3");
    // SA3 requires column-major V layout (transposed V stored as [hdim_v, seqlen_k])
    static_assert(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::ColumnMajor>,
                  "BlockFmhaPipelineQRKSVSSageAttn requires VLayout == ColumnMajor");

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0           = BlockFmhaShape::kM0;
    static constexpr index_t kN0           = BlockFmhaShape::kN0;
    static constexpr index_t kK0           = BlockFmhaShape::kK0;
    static constexpr index_t kN1           = BlockFmhaShape::kN1;
    static constexpr index_t kK1           = BlockFmhaShape::kK1;
    static constexpr index_t kQKHeaddim    = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim = BlockFmhaShape::kSubQKHeaddim;

    static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

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
    static constexpr bool kHasSink          = Problem::kHasSink;

    static constexpr ck_tile::index_t kQKScaleGranularity = Problem::kQKScaleGranularity;
    static constexpr ck_tile::index_t kVScaleGranularity  = Problem::kVScaleGranularity;

    static_assert((CK_TILE_FMHA_FWD_FAST_EXP2 &&
                   (kHasLogitsSoftCap && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS ||
                    !kHasLogitsSoftCap)) ||
                  (!CK_TILE_FMHA_FWD_FAST_EXP2 && !kHasLogitsSoftCap));

    static constexpr index_t kAlignmentQ = kPadHeadDimQ ? numeric_traits<QDataType>::PackedSize
                                                        : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK = kPadHeadDimQ ? numeric_traits<KDataType>::PackedSize
                                                        : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV =
        kPadSeqLenK ? numeric_traits<VDataType>::PackedSize
                    : Policy::template GetAlignmentV<Problem>();

    static constexpr index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();
    static constexpr index_t kAlignmentRandVal =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentRandVal<Problem>();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            if constexpr(kQKHeaddim <= 32)
                return 2;
            else if constexpr(kQKHeaddim <= 64)
                return 3;
            else if constexpr(kQKHeaddim <= 128)
                return 2;
            else
                return 1;
        }
    }();

    static constexpr const char* name = "qr_sageattn";

    using DropoutType = std::conditional_t<kHasDropout, BlockDropout, NullBlockDropout>;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename BiasElementFunction,
              typename LSEElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices,
              typename QScaleDramBlockWindowTmp,
              typename KScaleDramBlockWindowTmp,
              typename VScaleDramBlockWindowTmp,
              typename DeltaSDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,       // M0*K0 tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*K0 tile
               const KElementFunction& k_element_func,
               const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
               const VElementFunction& v_element_func,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               const BiasElementFunction& bias_element_func,
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
               LSEDramBlockWindowTmp& lse_dram_window_tmp,               // M0*1 tile
               const LSEElementFunction& lse_element_func,
               const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               DropoutType& dropout,
               const QScaleDramBlockWindowTmp& q_scale_dram_block_window_tmp,
               const KScaleDramBlockWindowTmp& k_scale_dram_block_window_tmp,
               const VScaleDramBlockWindowTmp& v_scale_dram_block_window_tmp,
               const DeltaSDramBlockWindowTmp& delta_s_dram_block_window_tmp, // [1, seqlen_k]
               float p_scale_factor,
               const float sink_v) const
    {
        // p_compute_element_func is part of the standard pipeline API but not used in SA3:
        // p_norm is computed directly as p_compute * p_scale_factor, bypassing gfx11 permutation.
        (void)p_compute_element_func;

        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kSubQKHeaddim ==
                              QDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        static_assert(
            std::is_same_v<QScaleDataType,
                           remove_cvref_t<typename QScaleDramBlockWindowTmp::DataType>> &&
            std::is_same_v<KScaleDataType,
                           remove_cvref_t<typename KScaleDramBlockWindowTmp::DataType>> &&
            std::is_same_v<VScaleDataType,
                           remove_cvref_t<typename VScaleDramBlockWindowTmp::DataType>>);
        static_assert(kM0 == QScaleDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                      kSubQKHeaddim ==
                          QScaleDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] *
                              kQKScaleGranularity &&
                      kN0 == KScaleDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                      kK0 == KScaleDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] *
                                 kQKScaleGranularity &&
                      kN1 == VScaleDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                      kK1 == VScaleDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] *
                                 kVScaleGranularity);

        // K tile in LDS
        KDataType* k_lds_ptr = static_cast<KDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQ<Problem>()));
        auto k_lds           = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});

        // V tile in LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(smem_ptr),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        // Block GEMMs
        auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              q_dram_block_window_tmp.get_window_lengths(),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());

        auto q = load_tile(q_dram_window);

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(s_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        auto o_acc = OaccBlockTileType{};
        auto m     = MLBlockTileType{};
        auto l     = MLBlockTileType{};

        clear_tile(o_acc);
        if(__builtin_isinf_sign(sink_v) >= 0)
        {
#if CK_TILE_FMHA_FWD_FAST_EXP2
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI ||
                         BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                set_tile(m, sink_v * scale_s * C_LOG2E);
            else
                set_tile(m, sink_v * C_LOG2E);
#else
            set_tile(m, sink_v);
#endif
            set_tile(l, SMPLComputeDataType{1.0f});
        }
        else
        {
            set_tile(m, -numeric<SMPLComputeDataType>::infinity());
            clear_tile(l);
        }
        const auto q_origin = q_dram_window.get_window_origin();

        const auto tile_range_result = [&mask, &q_origin]() {
            if constexpr(kHasSink)
                return mask.GetSinkTileRangeAlongX(
                    q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});
            else
            {
                auto [start, end] =
                    mask.GetTileRangeAlongX(q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});
                return ck_tile::make_tuple(0, start, end);
            }
        }();
        const auto sink_seq_end   = tile_range_result.get(ck_tile::number<0>{});
        const auto seqlen_k_start = tile_range_result.get(ck_tile::number<1>{});
        const auto seqlen_k_end   = tile_range_result.get(ck_tile::number<2>{});

        const auto kv_load_start = (sink_seq_end == 0 && seqlen_k_start > 0) ? seqlen_k_start : 0;
        const auto num_sink_loop = integer_divide_ceil(sink_seq_end, kN0);
        const auto num_total_loop =
            integer_divide_ceil(seqlen_k_end - seqlen_k_start, kN0) + num_sink_loop;

        if constexpr(FmhaMask::IsMasking || kPadSeqLenK)
        {
            if(num_total_loop <= 0)
            {
                if constexpr(kStoreLSE)
                {
                    auto lse =
                        make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                    set_tile(lse, SMPLComputeDataType{sink_v * scale_s});
                    store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
                }
                return o_acc;
            }
        }

        auto k_dram_block_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             {kv_load_start, 0});

        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), kv_load_start},
                             Policy::template MakeBiasDramTileDistribution<decltype(gemm_0)>());

        auto randval_dram_window = dropout.template MakeRandvalDramWindow<decltype(gemm_0)>(
            randval_dram_block_window_tmp, kv_load_start);

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             {0, kv_load_start},
                             Policy::template MakeVDramTileDistribution<Problem>());

        auto q_tile = tile_elementwise_in(q_element_func, q);

        // Load Q scale (MXFP4, always present for SA3)
        auto q_scale_dram_window =
            make_tile_window(q_scale_dram_block_window_tmp.get_bottom_tensor_view(),
                             q_scale_dram_block_window_tmp.get_window_lengths(),
                             q_scale_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeQScaleRegTileDistribution<Problem>());
        auto q_scale = load_tile(q_scale_dram_window);

        auto k_scale_dram_block_window =
            make_tile_window(k_scale_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_scale_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_k_start, 0});

        auto v_scale_dram_window =
            make_tile_window(v_scale_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_scale_dram_block_window_tmp.get_window_lengths(),
                             {0, seqlen_k_start / kVScaleGranularity},
                             Policy::template MakeVScaleRegTileDistribution<Problem>());

        // delta_s window: shape [1, kN0] (single row = delta_s for this q_block, one K tile)
        // delta_s_dram_block_window_tmp covers [1, seqlen_k]; we slide along N via move_tile_window
        const auto delta_s_origin = delta_s_dram_block_window_tmp.get_window_origin();
        auto delta_s_dram_window =
            make_tile_window(delta_s_dram_block_window_tmp.get_bottom_tensor_view(),
                             delta_s_dram_block_window_tmp.get_window_lengths(),
                             {delta_s_origin.at(number<0>{}), kv_load_start},
                             Policy::template MakeBiasDramTileDistribution<decltype(gemm_0)>());

        // Prefetch first K tile
        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(2 <= k0_loops);
        static_assert(1 <= k1_loops);
        do
        {
            // STAGE 1: QK GEMM (MXFP4)
            auto k_dram_window = make_tile_window(
                k_dram_block_window.get_bottom_tensor_view(),
                k_dram_block_window.get_window_lengths(),
                k_dram_block_window.get_window_origin(),
                Policy::template MakeKDramTileDistribution<Problem>());
            auto k_scale_dram_window =
                make_tile_window(k_scale_dram_block_window.get_bottom_tensor_view(),
                                 k_scale_dram_block_window.get_window_lengths(),
                                 k_scale_dram_block_window.get_window_origin(),
                                 Policy::template MakeKScaleRegTileDistribution<Problem>());
            auto load_k_scale_block_tile = [&] {
                auto t = load_tile(k_scale_dram_window);
                move_tile_window(k_scale_dram_window, {0, kK0 / kQKScaleGranularity});
                return t;
            };

            auto k_block_tile = load_tile(k_dram_window);
            {
                move_tile_window(k_dram_window, {0, kK0});
                clear_tile(s_acc);
                store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
                k_block_tile = load_tile(k_dram_window);
            }
            auto k_scale_block_tile = load_k_scale_block_tile();

            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                __builtin_amdgcn_sched_barrier(0);
            const auto bias_tile = load_tile(bias_dram_window);
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                __builtin_amdgcn_sched_barrier(0);

            // Load delta_s tile [1, kN0] for this K tile
            const auto delta_s_tile = load_tile(delta_s_dram_window);
            move_tile_window(delta_s_dram_window, {0, kN0});

            auto run_gemm_0 = [&](auto i_k0) {
                auto q_slice = get_slice_tile(
                    q_tile, sequence<0, i_k0 * kK0>{}, sequence<kM0, (i_k0 + 1) * kK0>{});
                auto q_scale_slice =
                    get_slice_tile(q_scale,
                                   sequence<0, i_k0*(kK0 / kQKScaleGranularity)>{},
                                   sequence<kM0, (i_k0 + 1) * (kK0 / kQKScaleGranularity)>{});
                gemm_0(s_acc, q_slice, q_scale_slice, k_lds_window, k_scale_block_tile);
            };

            if constexpr(k0_loops > 2)
            {
                static_for<0, k0_loops - 2, 1>{}([&](auto i_k0) {
                    block_sync_lds();
                    run_gemm_0(number<i_k0>{});
                    block_sync_lds();
                    move_tile_window(k_dram_window, {0, kK0});
                    store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
                    k_block_tile        = load_tile(k_dram_window);
                    k_scale_block_tile  = load_k_scale_block_tile();
                });
            }

            const auto v_prefetch = load_tile(v_dram_window);
            {
                block_sync_lds();
                run_gemm_0(number<k0_loops - 2>{});
                block_sync_lds();
                store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
                k_scale_block_tile = load_k_scale_block_tile();
                block_sync_lds();
                run_gemm_0(number<k0_loops - 1>{});
            }

            // STAGE 2: scale_s, add delta_s, bias, mask, softmax
            s_acc = tile_elementwise_in(s_acc_element_func, s_acc);

            // Add delta_s correction: broadcast per-column offset to s_acc
            // delta_s_tile has shape [kM0, kN0]; M-stride=0 in global memory means all
            // rows physically read from row 0, so each thread uses its own (idx0, idx1).
            {
                constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx   = make_tuple(idx0, idx1);
                        constexpr auto delta_idx = make_tuple(idx0, idx1);
                        s_acc(i_j_idx) += type_convert<SaccDataType>(delta_s_tile[delta_idx]);
                    });
                });
            }

            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
                tile_elementwise_inout(
                    [&](auto& x, const auto& y) {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                        x += type_convert<SaccDataType>(bias_element_func(y));
#else
                        x += log2e_v<SaccDataType> *
                             type_convert<SaccDataType>(bias_element_func(y));
#endif
                    },
                    s_acc,
                    bias_tile);
            }
            else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                const auto k_origin    = k_dram_block_window.get_window_origin();
                constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            s_acc.get_tile_distribution(), make_tuple(idx0, idx1));
                        const auto row     = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                        const auto col     = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        s_acc(i_j_idx) *= scale_s;
                        position_encoding.update(s_acc(i_j_idx), row, col);
                    });
                });
            }
            else
            {
                if constexpr(kHasLogitsSoftCap)
                {
                    auto apply_logits_transform =
                        [&variant, &variant_params, &block_indices](auto& x) {
                            x = variant.LogitsTransform(variant_params,
                                                        variant.QueryTransform(variant_params, x),
                                                        block_indices.batch_idx,
                                                        block_indices.qo_head_idx,
                                                        block_indices.kv_head_idx);
                        };
                    tile_elementwise_inout(apply_logits_transform, s_acc);
                }
                else
                {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                    tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
#endif
                }
            }
            if constexpr(kHasSink)
            {
                if(i_total_loops == 0)
                    move_tile_window(bias_dram_window, {0, seqlen_k_start - sink_seq_end});
            }
            move_tile_window(bias_dram_window, {0, kN0});
            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                const auto k_origin      = k_dram_block_window.get_window_origin();
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.at(number<0>{}),
                                                           k_origin.at(number<0>{}),
                                                           number<kM0>{},
                                                           number<kN0>{});
                if(need_perpixel_check)
                {
                    auto apply_mask = [&](auto&& mask_func) {
                        set_tile_if(
                            s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                                const auto row =
                                    q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                                const auto col =
                                    k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                                return !mask_func(variant_params,
                                                  block_indices.batch_idx,
                                                  row,
                                                  col,
                                                  block_indices.qo_head_idx,
                                                  block_indices.kv_head_idx);
                            });
                    };

                    if constexpr(kHasSink)
                    {
                        apply_mask([&](auto&&... args) {
                            return variant.LogitsSinkMask(std::forward<decltype(args)>(args)...);
                        });
                    }
                    else
                    {
                        apply_mask([&](auto&&... args) {
                            return variant.LogitsMask(std::forward<decltype(args)>(args)...);
                        });
                    }
                }
            }

            // Online softmax
            const auto s = cast_tile<SMPLComputeDataType>(s_acc);
            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s, sequence<1>{}, f_max, -numeric<SMPLComputeDataType>::infinity());
            block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

            const auto m_old = m;
            tile_elementwise_inout(
                [](auto& e0, auto e1) { e0 = max(e0, e1); }, m, m_local);

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s.get_tile_distribution());

            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                const auto tmp = exp2(m_old[i_idx] - m[i_idx]);
#else
                const auto tmp = exp(m_old[i_idx] - m[i_idx]);
#endif
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                    p_compute(i_j_idx) = exp2(s[i_j_idx] - m[i_idx]);
#else
                    p_compute(i_j_idx) = exp(s[i_j_idx] - m[i_idx]);
#endif
                });
                // Rescale o_acc for the new maximum
                sweep_tile_span(
                    decltype(o_acc)::get_distributed_spans()[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        o_acc(i_j_idx) *= tmp;
                    });
                l(i_idx) = tmp * l[i_idx];
            });

            // Compute row-sum of p_compute and accumulate into l
            auto rowsum = block_tile_reduce<SMPLComputeDataType>(
                p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0});
            block_tile_reduce_sync(rowsum, f_sum, bool_constant<false>{});
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                l(i_idx) = l[i_idx] + rowsum[i_idx];
            });

            // Dropout (if enabled)
            if constexpr(kHasDropout)
            {
                block_sync_lds();
                auto randval_ptr = reinterpret_cast<char*>(smem_ptr);

                index_t seq_offset = [&]() {
                    if constexpr(!kHasSink)
                        return seqlen_k_start + i_total_loops * kN0;

                    const bool in_sink_phase = (num_sink_loop > i_total_loops);
                    if(i_total_loops == num_sink_loop)
                        move_tile_window(randval_dram_window,
                                         {0, seqlen_k_start - sink_seq_end});

                    return in_sink_phase ? (kv_load_start + i_total_loops * kN0)
                                         : (seqlen_k_start + (i_total_loops - num_sink_loop) * kN0);
                }();

                dropout.template Run<decltype(gemm_0), SMPLComputeDataType, RandValOutputDataType>(
                    randval_ptr, seq_offset, p_compute, randval_dram_window);
            }

            // Store prefetched V
            block_sync_lds();
            store_tile(v_lds_window, tile_elementwise_in(v_element_func, v_prefetch));
            move_tile_window(v_dram_window, {0, kK1});

            auto load_v_scale_block_tile = [&] {
                auto t = load_tile(v_scale_dram_window);
                move_tile_window(v_scale_dram_window, {0, kK1 / kVScaleGranularity});
                return t;
            };
            auto v_scale_block_tile = load_v_scale_block_tile();

            // Level-1 P scaling: multiply p_compute by p_scale_factor before MXFP4 quantization.
            // This maps P̃ ∈ (0,1] to p_norm ∈ (0, p_scale_factor], filling the FP4 dynamic range
            // and minimizing e8m0 rounding loss in cast_tile_mx (default p_scale_factor = 6.0f).
            auto p_norm = make_static_distributed_tensor<SMPLComputeDataType>(
                p_compute.get_tile_distribution());
            constexpr auto pn_spans = decltype(p_norm)::get_distributed_spans();
            sweep_tile_span(pn_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(pn_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    p_norm(i_j_idx)        = p_compute(i_j_idx) * p_scale_factor;
                });
            });

            // Level-2: MXFP4 quantization of p_norm via cast_tile_mx
            auto p_result = make_static_distributed_tensor<PDataType>(
                p_norm.get_tile_distribution());
            auto p_scale_result = make_static_distributed_tensor<PScaleDataType>(
                Policy::template MakePScaleRegTileDistribution<Problem>());

            constexpr auto config =
                decltype(gemm_1)::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WG = remove_cvref_t<decltype(config.template at<0>())>;

            cast_tile_mx<kVScaleGranularity, WG::WarpGemmAttribute::Impl::kAMLane>(
                p_result, p_scale_result, p_norm);

            const auto& p       = p_result;
            const auto& p_scale = p_scale_result;

            // STAGE 3: PV GEMM (MXFP4)
            auto o_acc0 = OaccBlockTileType{};
            clear_tile(o_acc0);

            const float inv_p_scale_factor = 1.0f / p_scale_factor;

            auto run_gemm_1 = [&](auto i_k1) {
                auto p_slice =
                    get_slice_tile(p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{});
                auto p_scale_slice =
                    get_slice_tile(p_scale,
                                   sequence<0, i_k1*(kK1 / kVScaleGranularity)>{},
                                   sequence<kM0, (i_k1 + 1) * (kK1 / kVScaleGranularity)>{});
                gemm_1(o_acc0, p_slice, p_scale_slice, v_lds_window, v_scale_block_tile);
            };

            if constexpr(k1_loops > 1)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    const auto v = load_tile(v_dram_window);
                    block_sync_lds();
                    run_gemm_1(number<i_k1>{});
                    block_sync_lds();
                    store_tile(v_lds_window, tile_elementwise_in(v_element_func, v));
                    move_tile_window(v_dram_window, {0, kK1});
                    v_scale_block_tile = load_v_scale_block_tile();
                });
            }
            if constexpr(kHasSink)
            {
                if(i_total_loops == 0)
                {
                    move_tile_window(k_dram_block_window, {seqlen_k_start - sink_seq_end, 0});
                    move_tile_window(v_dram_window, {0, seqlen_k_start - sink_seq_end});
                }
            }
            move_tile_window(k_dram_block_window, {kN0, 0});
            move_tile_window(k_scale_dram_block_window, {kN0, 0});
            {
                block_sync_lds();
                run_gemm_1(number<k1_loops - 1>{});
                block_sync_lds();
            }

            // Level-1 correction: divide gemm1 output by p_scale_factor to recover P̃ × V
            constexpr auto o0_spans = decltype(o_acc0)::get_distributed_spans();
            sweep_tile_span(o0_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(o0_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    o_acc(i_j_idx) += o_acc0(i_j_idx) * inv_p_scale_factor;
                });
            });

        } while(++i_total_loops < num_total_loop);

        // Store LSE
        if constexpr(kStoreLSE)
        {
            auto lse = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

            constexpr auto lse_spans = decltype(lse)::get_distributed_spans();
            sweep_tile_span(lse_spans[number<0>{}], [&, m_ = m, l_ = l](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                if(l_[i_idx] == 0.0f)
                {
                    lse(i_idx) = -numeric<LSEDataType>::infinity();
                }
                else
                {
#if CK_TILE_FMHA_FWD_FAST_EXP2
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        lse(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                    }
                    else
                    {
                        if constexpr(kHasLogitsSoftCap)
                        {
                            lse(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                        }
                        else
                        {
                            lse(i_idx) = m_[i_idx] * scale_s / C_LOG2E + log(l_[i_idx]);
                        }
                    }
#else
                    lse(i_idx) = m_[i_idx] + log(l_[i_idx]);
#endif
                }
            });

            store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
        }

        // Normalize O by the softmax denominator l
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                if constexpr(FmhaMask::IsMasking ||
                             BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                {
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                }
                else
                    return 1 / l[i_idx];
            }();
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        o_acc = tile_elementwise_in(o_acc_element_func, o_acc);

        return o_acc;
    }
};

} // namespace ck_tile
