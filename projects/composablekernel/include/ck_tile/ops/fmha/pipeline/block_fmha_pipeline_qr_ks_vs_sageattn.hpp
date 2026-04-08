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
// Optimized: no sink, no LSE store, no dropout.
// Applies deferred rescale (O1) and XOR cross-warp reduce (O2).
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

    static_assert(Problem::QScaleEnum == BlockAttentionQuantScaleEnum::SAGEATTN_V3,
                  "BlockFmhaPipelineQRKSVSSageAttn requires QScaleEnum == SAGEATTN_V3");
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
               const DeltaSDramBlockWindowTmp& delta_s_dram_block_window_tmp,
               float p_scale_factor,
               const float sink_v) const
    {
        (void)p_compute_element_func;
        (void)randval_dram_block_window_tmp;
        (void)lse_dram_window_tmp;
        (void)lse_element_func;
        (void)dropout;
        (void)sink_v;

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
        auto cum_log_scale = MLBlockTileType{};

        clear_tile(o_acc);
        clear_tile(cum_log_scale);
        set_tile(m, -numeric<SMPLComputeDataType>::infinity());
        clear_tile(l);

        const auto q_origin = q_dram_window.get_window_origin();

        auto [seqlen_k_start, seqlen_k_end] =
            mask.GetTileRangeAlongX(q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});

        const auto num_total_loop = integer_divide_ceil(seqlen_k_end - seqlen_k_start, kN0);

        if constexpr(FmhaMask::IsMasking || kPadSeqLenK)
        {
            if(num_total_loop <= 0)
            {
                return o_acc;
            }
        }

        auto k_dram_block_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_k_start, 0});

        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), seqlen_k_start},
                             Policy::template MakeBiasDramTileDistribution<decltype(gemm_0)>());

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             {0, seqlen_k_start},
                             Policy::template MakeVDramTileDistribution<Problem>());

        auto q_tile = tile_elementwise_in(q_element_func, q);

        // Load Q scale
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

        const auto delta_s_origin = delta_s_dram_block_window_tmp.get_window_origin();
        auto delta_s_dram_window =
            make_tile_window(delta_s_dram_block_window_tmp.get_bottom_tensor_view(),
                             delta_s_dram_block_window_tmp.get_window_lengths(),
                             {delta_s_origin.at(number<0>{}), seqlen_k_start},
                             Policy::template MakeBiasDramTileDistribution<decltype(gemm_0)>());

        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(2 <= k0_loops);
        static_assert(1 <= k1_loops);
        do
        {
            // STAGE 1: QK GEMM
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

            // Add delta_s correction
            {
                constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        s_acc(i_j_idx) += type_convert<SaccDataType>(delta_s_tile[i_j_idx]);
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
                    set_tile_if(
                        s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                            const auto row =
                                q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                            const auto col =
                                k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                            return !variant.LogitsMask(variant_params,
                                                      block_indices.batch_idx,
                                                      row,
                                                      col,
                                                      block_indices.qo_head_idx,
                                                      block_indices.kv_head_idx);
                        });
                }
            }

            // Online softmax
            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                cast_tile<SMPLComputeDataType>(s_acc),
                sequence<1>{},
                f_max,
                -numeric<SMPLComputeDataType>::infinity());
            block_tile_reduce_xor_sync(m_local, f_max);

            const auto s = cast_tile<SMPLComputeDataType>(s_acc);
            const auto m_old = m;
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local);

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s.get_tile_distribution());

            static const auto get_validated_m = [](SMPLComputeDataType raw_m) {
                if constexpr(FmhaMask::IsMasking)
                {
                    return raw_m == -numeric<SMPLComputeDataType>::infinity()
                               ? type_convert<SMPLComputeDataType>(0.f)
                               : raw_m;
                }
                else
                {
                    return raw_m;
                }
            };

            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
#if CK_TILE_FMHA_FWD_FAST_EXP2
            // Deferred rescale: P = exp2(scale_s * s - scale_s * m - cum_log_scale)
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                auto row_base =
                    scale_s * get_validated_m(m[i_idx]) + cum_log_scale[i_idx];
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    p_compute(i_j_idx) = exp2(scale_s * s[i_j_idx] - row_base);
                });
            });
#else
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const auto tmp = exp(m_old[i_idx] - m[i_idx]);
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    p_compute(i_j_idx) = exp(s[i_j_idx] - m[i_idx]);
                });
                sweep_tile_span(
                    decltype(o_acc)::get_distributed_spans()[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        o_acc(i_j_idx) *= tmp;
                    });
                l(i_idx) = tmp * l[i_idx];
            });
#endif

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0});
            block_tile_reduce_xor_sync(rowsum_p, f_sum);

            // l, cum_log_scale update
            {
                constexpr auto ml_spans = decltype(m)::get_distributed_spans();
#if CK_TILE_FMHA_FWD_FAST_EXP2
                sweep_tile_span(ml_spans[number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    const auto m_old_val = m_old[i_idx];
                    const auto m_val     = get_validated_m(m[i_idx]);
                    const auto m_old_safe =
                        (m_old_val == -numeric<SMPLComputeDataType>::infinity())
                            ? m_val
                            : m_old_val;
                    cum_log_scale(i_idx) += scale_s * m_old_safe - scale_s * m_val;
                    l(i_idx) = l[i_idx] + rowsum_p[i_idx];
                });
#else
                sweep_tile_span(ml_spans[number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    l(i_idx) = l[i_idx] + rowsum_p[i_idx];
                });
#endif
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

            // Level-1 P scaling
            auto p_norm = make_static_distributed_tensor<SMPLComputeDataType>(
                p_compute.get_tile_distribution());
            constexpr auto pn_spans = decltype(p_norm)::get_distributed_spans();
            sweep_tile_span(pn_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(pn_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    p_norm(i_j_idx)        = p_compute(i_j_idx) * p_scale_factor;
                });
            });

            // Level-2: MXFP4 quantization
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

            // STAGE 3: PV GEMM
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
            move_tile_window(k_dram_block_window, {kN0, 0});
            move_tile_window(k_scale_dram_block_window, {kN0, 0});
            {
                block_sync_lds();
                run_gemm_1(number<k1_loops - 1>{});
                block_sync_lds();
            }

            // Level-1 correction
            constexpr auto o0_spans = decltype(o_acc0)::get_distributed_spans();
            sweep_tile_span(o0_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(o0_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    o_acc(i_j_idx) += o_acc0(i_j_idx) * inv_p_scale_factor;
                });
            });

        } while(++i_total_loops < num_total_loop);

        // Normalize O
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
