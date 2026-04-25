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

namespace sageattn_pk {
CK_TILE_DEVICE fp32x2_t pk_add_f32(fp32x2_t lhs, fp32x2_t rhs)
{
    fp32x2_t result;
    asm volatile("v_pk_add_f32 %[result], %[lhs], %[rhs]"
                 : [result] "=v"(result)
                 : [lhs] "v"(lhs), [rhs] "v"(rhs));
    return result;
}

CK_TILE_DEVICE fp32x2_t pk_fma_f32(fp32x2_t a, fp32x2_t b, fp32x2_t c)
{
    fp32x2_t result;
    asm volatile("v_pk_fma_f32 %[result], %[a], %[b], %[c]"
                 : [result] "=v"(result)
                 : [a] "v"(a), [b] "v"(b), [c] "v"(c));
    return result;
}
} // namespace sageattn_pk

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

    // Async K load (HBM → LDS direct via buffer_load_dwordx4 with LDS bit).
    // For pk_fp4_t: K must be pre-shuffled so contiguous thread access matches
    // the warp-interleaved LDS layout expected by the GEMM.
    // Disabled for padded configs where alignment < 4 bytes (can't do dword loads).
    static constexpr bool kUseAsyncKLoad = !Problem::kPadHeadDimQ;

    // When async K load is active, use the async copy policy variant for K descriptors.
    // Batch async K: use k0_loops LDS buffers so all K sub-tiles load at once.
    static constexpr index_t kAsyncKBuffers =
        BlockFmhaShape::kQKHeaddim / BlockFmhaShape::kK0;
    using AsyncKPolicy = BlockFmhaPipelineQXKSVSCustomPolicy<
        true, true, kAsyncKBuffers, 1>;

    static constexpr bool kUseAsyncVLoad =
        kUseAsyncKLoad && (kAsyncKBuffers > 1);
    static constexpr index_t k1_loops_static = BlockFmhaShape::kN0 / BlockFmhaShape::kK1;
    using AsyncVPolicy = BlockFmhaPipelineQXKSVSCustomPolicy<
        true, true, 1, k1_loops_static>;

    static_assert(Problem::QScaleEnum == BlockAttentionQuantScaleEnum::SAGEATTN_V3,
                  "BlockFmhaPipelineQRKSVSSageAttn requires QScaleEnum == SAGEATTN_V3");
    static_assert(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::ColumnMajor>,
                  "BlockFmhaPipelineQRKSVSSageAttn requires VLayout == ColumnMajor");
    static_assert(!Problem::kHasLogitsSoftCap,
                  "BlockFmhaPipelineQRKSVSSageAttn does not support logits soft cap");
    static_assert(!Problem::kHasDropout,
                  "BlockFmhaPipelineQRKSVSSageAttn does not support dropout");
    static_assert(Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS,
                  "BlockFmhaPipelineQRKSVSSageAttn only supports NO_BIAS");

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
    static constexpr auto BiasEnum          = Problem::BiasEnum;
    static constexpr bool kHasLogitsSoftCap = Problem::kHasLogitsSoftCap;
    static constexpr bool kStoreLSE         = Problem::kStoreLSE;
    static constexpr bool kHasDropout       = Problem::kHasDropout;
    static constexpr auto QScaleEnum        = Problem::QScaleEnum;
    static constexpr bool kHasSink          = Problem::kHasSink;

    static constexpr ck_tile::index_t kQKScaleGranularity = Problem::kQKScaleGranularity;
    static constexpr ck_tile::index_t kVScaleGranularity  = Problem::kVScaleGranularity;

    static_assert(CK_TILE_FMHA_FWD_FAST_EXP2,
                  "BlockFmhaPipelineQRKSVSSageAttn requires CK_TILE_FMHA_FWD_FAST_EXP2");

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

    // Contiguous DRAM distribution for async K load (pre-shuffled K).
    // Warp is the SLOWEST N sub-dim → each warp loads a contiguous N block.
    // N = issue * (NumWarps*LaneGroups) + warp * LaneGroups + lane_group
    // K = (lane % LanesPerK) * KVector + item
    template <typename Problem__>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistributionContiguous()
    {
        constexpr index_t kNPerBlock = Problem__::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem__::BlockFmhaShape::kK1;
        constexpr index_t NumWarps   = Problem__::BlockFmhaShape::NumWarps;
        constexpr index_t WarpSize   = ck_tile::get_warp_size();
        constexpr index_t KVector    = AsyncKPolicy::template GetAlignmentK<Problem__>();

        static_assert(WarpSize * KVector >= kKPerBlock && WarpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK  = kKPerBlock / KVector;
        constexpr index_t LaneGroups = WarpSize / LanesPerK;
        constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);

        // N0=NumIssues(replicate), N1=NumWarps(partition,slow), N2=LaneGroups(partition,fast)
        // K0=LanesPerK(partition), K1=KVector(replicate)
        // P[0]=warp_id → N1(minor=1), P[1]=lane_id → N2(minor=2) + K0(minor=0)
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<NumIssues, NumWarps, LaneGroups>,
                                             sequence<LanesPerK, KVector>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    using DropoutType = NullBlockDropout;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        if constexpr(kUseAsyncKLoad)
        {
            if constexpr(kAsyncKBuffers > 1)
            {
                if constexpr(kUseAsyncVLoad)
                {
                    constexpr index_t k_async_bytes =
                        AsyncKPolicy::template GetSingleSmemElementSpaceSize<Problem>() *
                        sizeof(KDataType) * AsyncKPolicy::NumKVLdsBuffers;
                    constexpr index_t v_smem = AsyncVPolicy::template GetSmemSizeV<Problem>();
                    return k_async_bytes + v_smem;
                }
                else
                    return AsyncKPolicy::template GetSmemSizeKV<Problem>() +
                           Policy::template GetSmemSizeKV<Problem>();
            }
            else
                return AsyncKPolicy::template GetSmemSize<Problem>();
        }
        else
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
               const float sink_v,
               const int32_t* k_scale_packed_ptr = nullptr,
               const int32_t* v_scale_packed_ptr = nullptr,
               const float* delta_s_raw_ptr = nullptr) const
    {
        (void)p_compute_element_func;
        (void)randval_dram_block_window_tmp;
        (void)lse_dram_window_tmp;
        (void)lse_element_func;
        (void)bias_element_func;
        (void)position_encoding;
        (void)dropout;
        (void)sink_v;
        (void)k_scale_dram_block_window_tmp;
        (void)v_scale_dram_block_window_tmp;

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

        // Baseline K LDS (used when async K load is disabled)
        auto k_lds           = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});

        // Async K LDS store/load descriptors (used when async K load is enabled)
        constexpr auto LdsSeq = AsyncKPolicy::template GetLdsBufferSequence<Problem>();
        auto k_lds_store = generate_tuple(
            [&](auto i_buf) {
                return make_tile_window(
                    make_tensor_view<address_space_enum::lds>(
                        k_lds_ptr,
                        AsyncKPolicy::template MakeKLdsStoreBlockDescriptor<Problem>(i_buf)),
                    AsyncKPolicy::template MakeKLdsStoreBlockDescriptor<Problem>(
                        i_buf).get_lengths(),
                    {0, 0, 0});
            },
            number<AsyncKPolicy::NumKVLdsBuffers>{});
        auto k_lds_load_view = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, AsyncKPolicy::template MakeKLdsLoadBlockDescriptor<Problem>());
        auto k_lds_load = make_tile_window(
            k_lds_load_view,
            AsyncKPolicy::template MakeKLdsLoadBlockDescriptor<Problem>().get_lengths(),
            {0, 0});

        // V tile in LDS — offset past K LDS when K/V are separated
        auto* v_lds_ptr = [&]() {
            if constexpr(kUseAsyncKLoad && kAsyncKBuffers > 1)
            {
                constexpr index_t k_async_bytes =
                    AsyncKPolicy::template GetSingleSmemElementSpaceSize<Problem>() *
                    sizeof(KDataType) * AsyncKPolicy::NumKVLdsBuffers;
                return reinterpret_cast<VDataType*>(
                    static_cast<char*>(smem_ptr) + k_async_bytes);
            }
            else
                return reinterpret_cast<VDataType*>(smem_ptr);
        }();

        // V async LDS store: BaseElementOffset encodes V's absolute LDS position.
        // async_load_tile_raw computes M0 from the descriptor's built-in offset,
        // so BaseElementOffset must skip past the entire K async LDS region.
        constexpr index_t kVLdsBaseElementOffset =
            AsyncKPolicy::template GetSingleSmemElementSpaceSize<Problem>() *
            AsyncKPolicy::NumKVLdsBuffers;
        auto make_v_store_window = [&](auto ibuf) {
            constexpr auto v_store_desc =
                AsyncVPolicy::template MakeVLdsStoreBlockDescriptor<Problem>(
                    ibuf, number<kVLdsBaseElementOffset>{});
            return make_tile_window(
                make_tensor_view<address_space_enum::lds>(
                    reinterpret_cast<VDataType*>(smem_ptr), v_store_desc),
                v_store_desc.get_lengths(),
                {0, 0, 0});
        };

        constexpr index_t v_single_buf_bytes =
            AsyncVPolicy::template GetSmemSizeV<Problem>() /
            AsyncVPolicy::NumPrefetchV;


        // V register-staged LDS (fallback when async V is disabled)
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            v_lds_ptr,
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

        (void)bias_dram_block_window_tmp;

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             {0, seqlen_k_start},
                             [&]() {
                                 if constexpr(kUseAsyncVLoad)
                                     return AsyncVPolicy::template
                                         MakeVDramTileDistributionContiguous<Problem>();
                                 else
                                     return Policy::template
                                         MakeVDramTileDistribution<Problem>();
                             }());

        auto q_tile = tile_elementwise_in(q_element_func, q);

        // Load Q scale
        auto q_scale_dram_window =
            make_tile_window(q_scale_dram_block_window_tmp.get_bottom_tensor_view(),
                             q_scale_dram_block_window_tmp.get_window_lengths(),
                             q_scale_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeQScaleRegTileDistribution<Problem>());
        auto q_scale = load_tile(q_scale_dram_window);

        (void)delta_s_dram_block_window_tmp;

        // delta_s: manual buffer load (4 unique values per thread vs 62 from load_tile)
        using BlockGemm0 = remove_cvref_t<decltype(gemm_0)>;
        constexpr index_t kCNLane0       = BlockGemm0::WarpGemm::WarpGemmAttribute::Impl::kCNLane;
        constexpr index_t kNIterPerWarp0 = BlockGemm0::NIterPerWarp;
        constexpr index_t kCMPerLane0    = BlockGemm0::CMPerLane;
        constexpr index_t kMIterPerWarp0 = BlockGemm0::MIterPerWarp;

        auto delta_s_res = make_wave_buffer_resource(
            delta_s_raw_ptr != nullptr ? delta_s_raw_ptr
                                       : reinterpret_cast<const float*>(smem_ptr));

        // Packed K/V scale: per-block dwordx4 loading
        // Each dwordx4 holds scale for 1 block × 1 k_group:
        //   [ki0_sg0, ..., ki0_sgN, ki1_sg0, ..., ki1_sgN]
        // where sg indexes NumScaleGroupsB within each k0_iter.
        using BlockGemm0Type = remove_cvref_t<decltype(gemm_0)>;
        using BlockGemm1Type = remove_cvref_t<decltype(gemm_1)>;
        constexpr index_t kNumSGB0 = BlockGemm0Type::NumScaleGroupsB;
        constexpr index_t kNumSGB1 = BlockGemm1Type::NumScaleGroupsB;
        constexpr index_t kABScaleKLane0 = kK0 / kQKScaleGranularity;
        constexpr index_t kScalePerBlock_K =
            (kQKHeaddim / kK0) * kNumSGB0;
        constexpr index_t kScalePerBlock_V =
            (kN0 / kK1) * kNumSGB1;
        const index_t k_group = get_lane_id() / (64 / kABScaleKLane0);
        const index_t lane_n  = get_lane_id() % kCNLane0;

        auto k_scale_res = make_wave_buffer_resource(
            k_scale_packed_ptr != nullptr ? k_scale_packed_ptr
                                          : reinterpret_cast<const int32_t*>(smem_ptr));
        auto v_scale_res = make_wave_buffer_resource(
            v_scale_packed_ptr != nullptr ? v_scale_packed_ptr
                                          : reinterpret_cast<const int32_t*>(smem_ptr));

        auto load_k_scale_dwordx4 = [&](index_t block_idx) {
            return llvm_amdgcn_raw_buffer_load_i32x4(
                k_scale_res,
                static_cast<index_t>(
                    (block_idx * kABScaleKLane0 * kScalePerBlock_K +
                     k_group * kScalePerBlock_K) * sizeof(int32_t)),
                0, 0);
        };
        auto load_v_scale_dwordx4 = [&](index_t block_idx) {
            return llvm_amdgcn_raw_buffer_load_i32x4(
                v_scale_res,
                static_cast<index_t>(
                    (block_idx * kABScaleKLane0 * kScalePerBlock_V +
                     k_group * kScalePerBlock_V) * sizeof(int32_t)),
                0, 0);
        };

        const index_t scale_block_start = seqlen_k_start / kN0;
        int32x4_t k_scale_4 = load_k_scale_dwordx4(scale_block_start);
        int32x4_t v_scale_4 = load_v_scale_dwordx4(scale_block_start);

        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(2 <= k0_loops);
        static_assert(1 <= k1_loops);

        // K DRAM window setup (contiguous distribution for pre-shuffled K)
        auto k_dram_window = make_tile_window(
            k_dram_block_window.get_bottom_tensor_view(),
            k_dram_block_window.get_window_lengths(),
            k_dram_block_window.get_window_origin(),
            [&]() {
                if constexpr(kUseAsyncKLoad)
                    return MakeKDramTileDistributionContiguous<Problem>();
                else
                    return Policy::template MakeKDramTileDistribution<Problem>();
            }());

        // Prologue: prefetch K sub-tiles
        auto k_block_tile = decltype(load_tile(k_dram_window)){};
        if constexpr(kUseAsyncKLoad)
        {
            if constexpr(kAsyncKBuffers > 1)
            {
                // Batch: load all k0 sub-tiles into separate LDS buffers
                static_for<0, k0_loops, 1>{}([&](auto i_k0) {
                    async_load_tile_raw(
                        k_lds_store(number<LdsSeq.at(number<i_k0>{})>{}),
                        k_dram_window);
                    if constexpr(i_k0 < k0_loops - 1)
                        move_tile_window(k_dram_window, {0, kK0});
                });
                // V async prologue: load k1 sub-tiles into pair A (buf 0,1)
                if constexpr(kUseAsyncVLoad)
                {
                    // Zero-fill V LDS: async store leaves gaps between warps
                    // (warp stride 2080 > 64 lanes × 16 bytes = 1024 per warp).
                    // V load descriptor reads through these gaps, so they must
                    // contain valid data (zeros).
                    {
                        constexpr index_t v_lds_total =
                            AsyncVPolicy::template GetSmemSizeV<Problem>();
                        auto* vp = reinterpret_cast<uint32_t*>(v_lds_ptr);
                        const index_t tid = get_thread_id();
                        for(index_t i = tid; i < v_lds_total / 4; i += kBlockSize)
                            vp[i] = 0;
                        __builtin_amdgcn_s_barrier();
                    }
                    static_for<0, k1_loops, 1>{}([&](auto i_k1) {
                        async_load_tile_raw(
                            make_v_store_window(i_k1),
                            v_dram_window);
                        if constexpr(i_k1 < k1_loops - 1)
                            move_tile_window(v_dram_window, {0, kK1});
                    });
                }
            }
            else
            {
                // Single buffer: load only k0=0
                async_load_tile_raw(
                    k_lds_store(LdsSeq.at(number<0>{})), k_dram_window);
                move_tile_window(k_dram_window, {0, kK0});
            }
        }
        else
        {
            k_block_tile = load_tile(k_dram_window);
        }

        do
        {
            // Save current K origin before K_next prefetch may advance it.
            // Mask check needs the *current* block's N coordinate.
            const auto k_origin_cur = k_dram_block_window.get_window_origin();

            // STAGE 1: QK GEMM
            float delta_s_unique[kNIterPerWarp0];
            clear_tile(s_acc);

            if constexpr(!kUseAsyncKLoad)
            {
                // Baseline: k_block_tile holds sub-tile[0] from prologue/prev iter
                move_tile_window(k_dram_window, {0, kK0});
                store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
                k_block_tile = load_tile(k_dram_window);
            }

            // K scale from dwordx4: kNumSGB0 int32 per k0_iter
            // dwordx4 layout = [ki0_sg0..ki0_sgN, ki1_sg0..ki1_sgN]
            // k_scale_arr_k0_X points into k_scale_4 at offset X*kNumSGB0
            const int32_t* k_scale_arr_k0_0 =
                reinterpret_cast<const int32_t*>(&k_scale_4) + 0 * kNumSGB0;
            const int32_t* k_scale_arr_k0_1 =
                reinterpret_cast<const int32_t*>(&k_scale_4) + 1 * kNumSGB0;

            // GEMM0 helper: produces Q slice + Q scale slice for sub-tile i_k0
            auto run_gemm_0 = [&](auto i_k0, const int32_t* k_scale_arr, auto& k_win) {
                auto q_slice = get_slice_tile(
                    q_tile, sequence<0, i_k0 * kK0>{},
                    sequence<kM0, (i_k0 + 1) * kK0>{});
                auto q_scale_slice = get_slice_tile(
                    q_scale, sequence<0, i_k0*(kK0 / kQKScaleGranularity)>{},
                    sequence<kM0, (i_k0 + 1) * (kK0 / kQKScaleGranularity)>{});
                gemm_0(s_acc, q_slice, q_scale_slice, k_win, k_scale_arr);
            };

            // V prefetch — declared here so it's visible after both branches
            auto v_prefetch = decltype(load_tile(v_dram_window)){};

            if constexpr(kUseAsyncKLoad)
            {
                if constexpr(kAsyncKBuffers > 1)
                {
                    // Batch: all K sub-tiles loaded by prologue/prev-iter
                    async_load_fence();
                    __builtin_amdgcn_s_barrier();

                    // Issue delta_s loads early: GEMM0 + K_next + softmax
                    // will hide HBM latency before delta_s is consumed
                    {
                        const index_t delta_s_byte_base = static_cast<index_t>(
                            (seqlen_k_start + i_total_loops * kN0) * sizeof(float));
                        static_for<0, kNIterPerWarp0, 1>{}([&](auto ni) {
                            const index_t byte_offset = delta_s_byte_base +
                                static_cast<index_t>(
                                    (ni * kCNLane0 + lane_n) * sizeof(float));
                            delta_s_unique[ni] = bit_cast<float>(
                                llvm_amdgcn_raw_buffer_load_i32(
                                    delta_s_res, byte_offset, 0, 0));
                        });
                    }
                    __builtin_amdgcn_sched_barrier(0);

                    // Dense GEMM0 — no intermediate stalls
                    static_for<0, k0_loops, 1>{}([&](auto i_k0) {
                        const int32_t* k_arr =
                            reinterpret_cast<const int32_t*>(&k_scale_4) +
                            i_k0 * kNumSGB0;
                        auto k_slice = get_slice_tile(
                            k_lds_load,
                            sequence<LdsSeq.at(number<i_k0>{}) * kN0, 0>{},
                            sequence<(LdsSeq.at(number<i_k0>{}) + 1) * kN0, kK0>{});
                        run_gemm_0(number<i_k0>{}, k_arr, k_slice);
                    });

                    if constexpr(k0_loops <= 2)
                        __builtin_amdgcn_sched_barrier(0);

                    // Prefetch K_next: issue async loads right after GEMM0
                    // ~600 VALU (softmax+GEMM1) will hide HBM latency
                    move_tile_window(k_dram_block_window, {kN0, 0});
                    if(i_total_loops + 1 < num_total_loop)
                    {
                        k_dram_window.set_window_origin(
                            k_dram_block_window.get_window_origin());
                        static_for<0, k0_loops, 1>{}([&](auto i_k0) {
                            async_load_tile_raw(
                                k_lds_store(number<LdsSeq.at(number<i_k0>{})>{}),
                                k_dram_window);
                            if constexpr(i_k0 < k0_loops - 1)
                                move_tile_window(k_dram_window, {0, kK0});
                        });
                        __builtin_amdgcn_sched_group_barrier(0x020, 2, 0);
                    }

                    if constexpr(!kUseAsyncVLoad)
                        v_prefetch = load_tile(v_dram_window);
                }
                else
                {
                    // Serial: single-buffer, original async path
                    async_load_fence();
                    __builtin_amdgcn_s_barrier();

                    {
                        auto k_slice = get_slice_tile(
                            k_lds_load,
                            sequence<LdsSeq.at(number<0>{}) * kN0, 0>{},
                            sequence<(LdsSeq.at(number<0>{}) + 1) * kN0, kK0>{});
                        run_gemm_0(number<0>{}, k_scale_arr_k0_0, k_slice);
                    }

                    static_for<1, k0_loops, 1>{}([&](auto i_k0) {
                        async_load_tile_raw(
                            k_lds_store(number<LdsSeq.at(number<i_k0>{})>{}),
                            k_dram_window);
                        if constexpr(i_k0 < k0_loops - 1)
                            move_tile_window(k_dram_window, {0, kK0});

                        async_load_fence();
                        __builtin_amdgcn_s_barrier();

                        const int32_t* k_arr =
                            reinterpret_cast<const int32_t*>(&k_scale_4) +
                            i_k0 * kNumSGB0;
                        auto k_slice = get_slice_tile(
                            k_lds_load,
                            sequence<LdsSeq.at(number<i_k0>{}) * kN0, 0>{},
                            sequence<(LdsSeq.at(number<i_k0>{}) + 1) * kN0, kK0>{});
                        run_gemm_0(number<i_k0>{}, k_arr, k_slice);
                    });

                    if constexpr(k0_loops <= 2)
                        __builtin_amdgcn_sched_barrier(0);
                    v_prefetch = load_tile(v_dram_window);
                }
            }
            else
            {
                // Baseline path: load_tile → store_tile → GEMM0
                if constexpr(k0_loops > 2)
                {
                    static_for<0, k0_loops - 2, 1>{}([&](auto i_k0) {
                        block_sync_lds();
                        const int32_t* k_arr =
                            reinterpret_cast<const int32_t*>(&k_scale_4) +
                            i_k0 * kNumSGB0;
                        run_gemm_0(number<i_k0>{}, k_arr, k_lds_window);
                        block_sync_lds();
                        move_tile_window(k_dram_window, {0, kK0});
                        store_tile(k_lds_window,
                                   tile_elementwise_in(k_element_func, k_block_tile));
                        k_block_tile = load_tile(k_dram_window);
                    });
                }

                v_prefetch = load_tile(v_dram_window);
                {
                    block_sync_lds();
                    run_gemm_0(number<k0_loops - 2>{}, k_scale_arr_k0_0, k_lds_window);
                    block_sync_lds();
                    store_tile(k_lds_window,
                               tile_elementwise_in(k_element_func, k_block_tile));
                    block_sync_lds();
                    run_gemm_0(number<k0_loops - 1>{}, k_scale_arr_k0_1, k_lds_window);
                }
            }

            // For non-batch paths, load delta_s here (batch path loads after fence)
            if constexpr(!kUseAsyncKLoad || kAsyncKBuffers <= 1)
            {
                const index_t delta_s_byte_base = static_cast<index_t>(
                    (seqlen_k_start + i_total_loops * kN0) * sizeof(float));
                static_for<0, kNIterPerWarp0, 1>{}([&](auto ni) {
                    const index_t byte_offset = delta_s_byte_base +
                        static_cast<index_t>((ni * kCNLane0 + lane_n) * sizeof(float));
                    delta_s_unique[ni] = bit_cast<float>(
                        llvm_amdgcn_raw_buffer_load_i32(
                            delta_s_res, byte_offset, 0, 0));
                });
            }

            // STAGE 2: scale_s, add delta_s, bias, mask, softmax
            s_acc = tile_elementwise_in(s_acc_element_func, s_acc);

            // Add delta_s correction via packed fp32x2 (v_pk_add_f32)
            // s_acc buffer layout: [MIterPerWarp][NIterPerWarp][CMPerLane]
            // Consecutive CMPerLane elements share the same delta_s → natural pk pair
            {
                auto& s_acc_buf = s_acc.get_thread_buffer();
                static_assert(kCMPerLane0 % 2 == 0);
                static_for<0, kMIterPerWarp0, 1>{}([&](auto mIter) {
                    static_for<0, kNIterPerWarp0, 1>{}([&](auto nIter) {
                        const auto dv =
                            type_convert<SaccDataType>(delta_s_unique[nIter]);
                        const fp32x2_t delta_pk = {dv, dv};
                        static_for<0, kCMPerLane0 / 2, 1>{}([&](auto mi_half) {
                            constexpr index_t buf_idx =
                                mIter * kNIterPerWarp0 * kCMPerLane0 +
                                nIter * kCMPerLane0 + mi_half * 2;
                            fp32x2_t s_pk = {s_acc_buf[number<buf_idx>{}],
                                             s_acc_buf[number<buf_idx + 1>{}]};
                            s_pk = sageattn_pk::pk_add_f32(s_pk, delta_pk);
                            s_acc_buf(number<buf_idx>{})     = s_pk[0];
                            s_acc_buf(number<buf_idx + 1>{}) = s_pk[1];
                        });
                    });
                });
            }

            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                bool need_perpixel_check = mask.IsEdgeTile(
                    q_origin.at(number<0>{}),
                    k_origin_cur.at(number<0>{}),
                    number<kM0>{},
                    number<kN0>{});
                if(need_perpixel_check)
                {
                    set_tile_if(
                        s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                            const auto row =
                                q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                            const auto col =
                                k_origin_cur.at(number<0>{}) + tile_idx.at(number<1>{});
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

            // exp2 with packed FMA: p(i,j) = exp2(scale_s * s(i,j) - row_base(i))
            // Pair consecutive CMPerLane elements (different M rows, same nIter)
            // so pk_fma computes two exponents in one instruction.
            const auto log2_p_scale =
                type_convert<SMPLComputeDataType>(__builtin_amdgcn_logf(p_scale_factor));
            {
                auto& s_buf = s.get_thread_buffer();
                auto& p_buf = p_compute.get_thread_buffer();
                const auto& m_buf = m.get_thread_buffer();
                const auto& cls_buf = cum_log_scale.get_thread_buffer();
                const fp32x2_t scale_pk = {scale_s, scale_s};

                static_for<0, kMIterPerWarp0, 1>{}([&](auto mIter) {
                    static_for<0, kCMPerLane0 / 2, 1>{}([&](auto mi_half) {
                        constexpr index_t m0 = mIter * kCMPerLane0 + mi_half * 2;
                        constexpr index_t m1 = m0 + 1;
                        const fp32x2_t neg_rb = {
                            -(scale_s * get_validated_m(m_buf[number<m0>{}]) +
                              cls_buf[number<m0>{}] - log2_p_scale),
                            -(scale_s * get_validated_m(m_buf[number<m1>{}]) +
                              cls_buf[number<m1>{}] - log2_p_scale)};

                        static_for<0, kNIterPerWarp0, 1>{}([&](auto nIter) {
                            constexpr index_t si =
                                mIter * kNIterPerWarp0 * kCMPerLane0 +
                                nIter * kCMPerLane0 + mi_half * 2;
                            const fp32x2_t s_pk = {s_buf[number<si>{}],
                                                   s_buf[number<si + 1>{}]};
                            auto exponent =
                                sageattn_pk::pk_fma_f32(scale_pk, s_pk, neg_rb);
                            p_buf(number<si>{})     = exp2(exponent[0]);
                            p_buf(number<si + 1>{}) = exp2(exponent[1]);
                        });
                    });
                });
            }

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0});
            block_tile_reduce_xor_sync(rowsum_p, f_sum);

            // l, cum_log_scale update
            {
                constexpr auto ml_spans = decltype(m)::get_distributed_spans();
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
            }

            // V scale from dwordx4: kScalePerBlock_V int32 for this block
            const int32_t* v_scale_arr =
                reinterpret_cast<const int32_t*>(&v_scale_4);

            // MXFP4 quantization
            auto p_result = make_static_distributed_tensor<PDataType>(
                p_compute.get_tile_distribution());
            auto p_scale_result = make_static_distributed_tensor<PScaleDataType>(
                Policy::template MakePScaleRegTileDistribution<Problem>());

            constexpr auto config =
                decltype(gemm_1)::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WG = remove_cvref_t<decltype(config.template at<0>())>;

            cast_tile_mx<kVScaleGranularity, WG::WarpGemmAttribute::Impl::kAMLane>(
                p_result, p_scale_result, p_compute);

            const auto& p       = p_result;
            const auto& p_scale = p_scale_result;

            if constexpr(kUseAsyncVLoad)
            {
                // V async 2-buffer in-place with interleaved V_next:
                // k1=0: GEMM1 reads buf 0 → V_next overwrites buf 0
                // k1=1: GEMM1 reads buf 1 (buf 0 V_next in flight, no conflict)
                //        → V_next overwrites buf 1
                // MFMA compute hides V_next HBM latency.
                static_for<0, k1_loops, 1>{}([&](auto i_k1) {
                    if constexpr(i_k1 > 0)
                        block_sync_lds();

                    auto p_slice = get_slice_tile(
                        p, sequence<0, i_k1 * kK1>{},
                        sequence<kM0, (i_k1 + 1) * kK1>{});
                    auto p_scale_slice = get_slice_tile(
                        p_scale,
                        sequence<0, i_k1 * (kK1 / kVScaleGranularity)>{},
                        sequence<kM0, (i_k1 + 1) * (kK1 / kVScaleGranularity)>{});

                    const index_t v_read_buf = static_cast<index_t>(i_k1);
                    auto* v_read_ptr = reinterpret_cast<VDataType*>(
                        reinterpret_cast<char*>(v_lds_ptr) +
                        v_read_buf * v_single_buf_bytes);
                    auto v_lds_read_view = make_tensor_view<address_space_enum::lds>(
                        v_read_ptr,
                        AsyncVPolicy::template MakeVLdsLoadBlockDescriptor<Problem>());
                    auto v_win = make_tile_window(
                        v_lds_read_view,
                        make_tuple(number<kN1>{}, number<kK1>{}),
                        {0, 0});
                    gemm_1(o_acc, p_slice, p_scale_slice,
                           v_win, v_scale_arr);

                    // V_next: buf i_k1 just read, safe to overwrite.
                    if(i_total_loops + 1 < num_total_loop)
                    {
                        move_tile_window(v_dram_window, {0, kK1});
                        async_load_tile_raw(
                            make_v_store_window(i_k1), v_dram_window);
                    }
                });

                if constexpr(!kUseAsyncKLoad || kAsyncKBuffers <= 1)
                    move_tile_window(k_dram_block_window, {kN0, 0});
            }
            else
            {
                // V register-staged path (original)
                // Store prefetched V
                if constexpr(!kUseAsyncKLoad || kAsyncKBuffers <= 1 || k1_loops > 1)
                    block_sync_lds();
                store_tile(v_lds_window, tile_elementwise_in(v_element_func, v_prefetch));
                move_tile_window(v_dram_window, {0, kK1});

                auto run_gemm_1_packed = [&](auto i_k1) {
                    auto p_slice = get_slice_tile(
                        p, sequence<0, i_k1 * kK1>{},
                        sequence<kM0, (i_k1 + 1) * kK1>{});
                    auto p_scale_slice = get_slice_tile(
                        p_scale,
                        sequence<0, i_k1 * (kK1 / kVScaleGranularity)>{},
                        sequence<kM0, (i_k1 + 1) * (kK1 / kVScaleGranularity)>{});
                    gemm_1(o_acc, p_slice, p_scale_slice, v_lds_window, v_scale_arr);
                };

                if constexpr(k1_loops > 1)
                {
                    static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                        const auto v = load_tile(v_dram_window);
                        block_sync_lds();
                        run_gemm_1_packed(number<i_k1>{});
                        block_sync_lds();
                        store_tile(v_lds_window, tile_elementwise_in(v_element_func, v));
                        move_tile_window(v_dram_window, {0, kK1});
                    });
                }
                if constexpr(!kUseAsyncKLoad || kAsyncKBuffers <= 1)
                    move_tile_window(k_dram_block_window, {kN0, 0});
                {
                    block_sync_lds();
                    run_gemm_1_packed(number<k1_loops - 1>{});
                    block_sync_lds();
                }
            }

            // Reload dwordx4 scale for next block
            if(i_total_loops + 1 < num_total_loop)
            {
                k_scale_4 = load_k_scale_dwordx4(
                    scale_block_start + i_total_loops + 1);
                v_scale_4 = load_v_scale_dwordx4(
                    scale_block_start + i_total_loops + 1);
            }

            // Prefetch K for next iteration (non-batch paths only;
            // batch async K is prefetched after GEMM0 above)
            if constexpr(!kUseAsyncKLoad || kAsyncKBuffers <= 1)
            {
                if(i_total_loops + 1 < num_total_loop)
                {
                    if constexpr(kUseAsyncKLoad)
                    {
                        k_dram_window.set_window_origin(
                            k_dram_block_window.get_window_origin());
                        async_load_tile_raw(
                            k_lds_store(LdsSeq.at(number<0>{})), k_dram_window);
                        move_tile_window(k_dram_window, {0, kK0});
                    }
                    else
                    {
                        k_dram_window = make_tile_window(
                            k_dram_block_window.get_bottom_tensor_view(),
                            k_dram_block_window.get_window_lengths(),
                            k_dram_block_window.get_window_origin(),
                            Policy::template MakeKDramTileDistribution<Problem>());
                        k_block_tile = load_tile(k_dram_window);
                    }
                    __builtin_amdgcn_sched_group_barrier(0x020, 2, 0);
                }
            }

        } while(++i_total_loops < num_total_loop);

        // Normalize O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                if constexpr(FmhaMask::IsMasking)
                    return l[i_idx] == 0.f ? 0.f : 1.0f / l[i_idx];
                else
                    return 1.0f / l[i_idx];
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
